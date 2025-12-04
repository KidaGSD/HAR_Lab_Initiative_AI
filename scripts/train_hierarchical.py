import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import copy
import math
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import WeightedRandomSampler
import hashlib
import pickle

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. Run: pip install wandb")

# --- Configuration ---
CONFIG = {
    'lle': {
        'in_channels': 8,  # 6 raw + accel/gyro norms
        'cnn_filters': [32, 64, 128],
        'gru_hidden': 256,
        'gru_layers': 2,
        'embedding_dim': 128,
        'window_size': 50, # 1 second at 50Hz
        'se_reduction': 8   # SE channel attention reduction
    },
    'hla': {
        'hidden_dim': 128,
        'num_layers': 2,
        'num_classes': 8, # 8 Scenarios
        'seq_len': 20,    # 30 seconds context (can test 20 for latency tradeoff)
        'nhead': 4,
        'dropout': 0.1,
        'type': 'transformer' # transformer encoder for long-range modeling
    },
    'training': {
        'batch_size': 256,     # safer default to mitigate OOM
        'lr': 0.0001,            # Base LR (will warmup then cosine)
        'epochs': 50,
        'patience': 20,
        'alpha': 1.0,
        'beta': 0,           # Default: focus on scenario; action loss used in probe or if explicitly enabled
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'warmup_epochs': 5,     # Warmup then cosine anneal
        'gradient_accumulation_steps': 1,  # 1 = no accumulation, 2+ = accumulate
        'use_focal_loss': True,  # enable focal loss
        'focal_gamma': 2.0,      # focusing parameter
        'focal_alpha': 1.0,      # class weighting (1.0 = no weighting, or use class_weights)
        'label_smoothing': 0.1,  # label smoothing amount

    },
    'data': {
        'action_label_pad': 0.5,   # seconds to expand action labels on each side
        'per_video_center': True,  # subtract per-video mean after global z-score
        'add_norm_features': True,  # add accel/gyro norms as extra channels
        'stride': 5,
        'augment': True
    }
}

# --- Utils ---
class EarlyStopping:
    def __init__(self, patience=50, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

# --- Dataset Caching ---
def get_dataset_cache_key(processed_dir, uids, seq_len, stride, 
                         per_video_center, add_norm_features, action_pad):
    """Generate unique cache key for dataset configuration"""
    config_str = f"{seq_len}_{stride}_{per_video_center}_{add_norm_features}_{action_pad}"
    uids_str = "_".join(sorted(uids))[:100]  # Truncate for hash
    cache_key = hashlib.md5(f"{config_str}_{uids_str}".encode()).hexdigest()[:16]
    return cache_key

def save_cached_dataset(dataset, cache_path):
    """Save dataset to disk (exclude large objects that can be reloaded)"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'samples': dataset.samples,
            'global_mean': dataset.global_mean,
            'global_std': dataset.global_std,
            'scenario_map': dataset.scenario_map,
            'action_map': dataset.action_map,
            'num_scenarios': dataset.num_scenarios,
            'idx_to_scenario': dataset.idx_to_scenario,
            'idx_to_action': dataset.idx_to_action,
        }, f)
    print(f"✅ Cached dataset ({len(dataset.samples):,} samples) to {cache_path.name}")

def load_cached_dataset(cache_path, scenario_labels_path, action_labels_path):
    """Load dataset from cache"""
    with open(cache_path, 'rb') as f:
        cached = pickle.load(f)
    
    # Create minimal dataset
    dataset = HierarchicalDataset.__new__(HierarchicalDataset)
    dataset.samples = cached['samples']
    dataset.global_mean = cached['global_mean']
    dataset.global_std = cached['global_std']
    dataset.scenario_map = cached['scenario_map']
    dataset.action_map = cached['action_map']
    dataset.num_scenarios = cached['num_scenarios']
    dataset.idx_to_scenario = cached['idx_to_scenario']
    dataset.idx_to_action = cached['idx_to_action']
    dataset.training = False
    
    # Reload labels (lightweight)
    dataset.scenario_df = pd.read_csv(scenario_labels_path).set_index('video_uid')
    dataset.action_df = pd.read_csv(action_labels_path)
    dataset.action_df = dataset.action_df[dataset.action_df['action'].isin(dataset.action_map.keys())]
    
    # Set config flags
    dataset.action_pad = CONFIG['data'].get('action_label_pad', 0.5)
    dataset.per_video_center = CONFIG['data'].get('per_video_center', True)
    dataset.add_norm_features = CONFIG['data'].get('add_norm_features', True)
    
    return dataset

# --- Dataset ---
class HierarchicalDataset(torch.utils.data.Dataset):
    def __init__(self, take_uids, processed_dir, scenario_labels_path, action_labels_path,
             cache_dir=None, use_cache=True):
        self.samples = []
        self.training = False
        
        # Check cache first
        if cache_dir and use_cache:
            seq_len = CONFIG['hla']['seq_len']
            stride = CONFIG['data'].get('stride', 5)
            per_video_center = CONFIG['data'].get('per_video_center', True)
            add_norm_features = CONFIG['data'].get('add_norm_features', True)
            action_pad = CONFIG['data'].get('action_label_pad', 0.5)
            
            cache_key = get_dataset_cache_key(
                processed_dir, take_uids, seq_len, stride,
                per_video_center, add_norm_features, action_pad
            )
            cache_path = Path(cache_dir) / f"dataset_{cache_key}.pkl"
            
            if cache_path.exists():
                print(f"📦 Loading cached dataset: {cache_path.name}")
                cached_dataset = load_cached_dataset(cache_path, scenario_labels_path, action_labels_path)
                self.__dict__.update(cached_dataset.__dict__)
                print(f"   Loaded {len(self.samples):,} samples in <1s")
                self.training = False 
                return  # Skip expensive loading!
            else:
                print(f"💾 Cache not found, creating new dataset (will cache to {cache_path.name})")
        
        # Load Labels
        print("Loading label files...")
        self.scenario_df = pd.read_csv(scenario_labels_path).set_index('video_uid')
        self.action_df = pd.read_csv(action_labels_path)
        
        # Map Scenario Names to Integers
        self.scenario_map = {name: i for i, name in enumerate(sorted(self.scenario_df['scenario'].unique()))}
        self.num_scenarios = len(self.scenario_map)
        self.idx_to_scenario = {v: k for k, v in self.scenario_map.items()}
        print(f"Scenarios: {self.scenario_map}")
        
        # Map Action Names to Integers (6 Classes)
        self.action_map = {
            'Stationary': 0, 
            'Locomotion': 1, 
            'Essential Operation': 2, 
            'Object Transfer': 3,
            'Search': 4,
            'Error / Correction': 5
        }
        self.idx_to_action = {v: k for k, v in self.action_map.items()}
        # Keep only clean action labels (drop Unknown/Uncertain)
        self.action_df = self.action_df[self.action_df['action'].isin(self.action_map.keys())]
        self.action_pad = CONFIG['data'].get('action_label_pad', 0.5)
        self.per_video_center = CONFIG['data'].get('per_video_center', True)
        self.add_norm_features = CONFIG['data'].get('add_norm_features', True)

        # === GLOBAL NORMALIZATION (with optional feature augmentation) ===
        print("\nComputing global normalization statistics...")
        all_data = []
        valid_uids = []
        skipped_nan = []

        for uid in tqdm(take_uids, desc='Collecting normalization data'):
            seq_path = Path(processed_dir) / uid / 'seq.npz'
            if not seq_path.exists():
                continue
            try:
                data = np.load(seq_path)
                traj = data['traj']  # (N, 50, 6)
                traj = self._augment_traj(traj)  # add norms if enabled
                
                # Check for NaN/Inf BEFORE adding to normalization stats
                if np.isnan(traj).any() or np.isinf(traj).any():
                    skipped_nan.append(uid)
                    continue  # Skip videos with NaN for normalization stats
                
                if len(traj) > 0 and uid in self.scenario_df.index:
                    all_data.append(traj)
                    valid_uids.append(uid)
            except Exception:
                continue

        if len(all_data) == 0:
            raise ValueError("No valid data found for normalization!")

        if skipped_nan:
            print(f"⚠️ Skipped {len(skipped_nan)} videos with NaN/Inf for normalization stats")
            if len(skipped_nan) <= 10:
                print(f"   Skipped UIDs: {skipped_nan}")
            
        all_data = np.concatenate(all_data, axis=0)  # (Total_Windows, 50, C)

        # Compute statistics with NaN-safe operations
        self.global_mean = np.nanmean(all_data, axis=(0, 1))  # (C,) - NaN-safe mean
        self.global_std = np.nanstd(all_data, axis=(0, 1)) + 1e-6  # (C,) - NaN-safe std

        # Double-check for NaN in statistics
        if np.isnan(self.global_mean).any() or np.isnan(self.global_std).any():
            print(f"⚠️ WARNING: NaN detected in normalization statistics!")
            print(f"   Mean: {self.global_mean}")
            print(f"   Std: {self.global_std}")
            # Replace NaN with 0 for mean, 1 for std
            self.global_mean = np.nan_to_num(self.global_mean, nan=0.0)
            self.global_std = np.nan_to_num(self.global_std, nan=1.0)

        print(f"Global statistics computed from {len(valid_uids)} videos:")
        print(f"  Mean: {self.global_mean}")
        print(f"  Std:  {self.global_std}")
        print(f"  Data range: [{np.nanmin(all_data):.2f}, {np.nanmax(all_data):.2f}]")
        # ==========================================
        
        # Iterate Videos
        for uid in tqdm(take_uids, desc='Loading Data'):
            seq_path = Path(processed_dir) / uid / 'seq.npz'
            if not seq_path.exists():
                continue
                
            try:
                data = np.load(seq_path)
                traj = data['traj'] # (N, 50, 6)
                traj = self._augment_traj(traj)
                timestamps = data['timestamp'] # (N, 50)
                
                # GLOBAL normalization + optional per-video centering
                if len(traj) > 0:
                    traj = (traj - self.global_mean) / self.global_std
                    if self.per_video_center:
                        traj = traj - traj.mean(axis=(0, 1), keepdims=True)
                    
                    # Check for NaN after normalization and skip this video if found
                    if np.isnan(traj).any() or np.isinf(traj).any():
                        print(f"⚠️ Skipping {uid}: NaN/Inf after normalization")
                        continue
                
                if len(traj) == 0:
                    continue
                
                # Get Scenario Label (Global for video)
                if uid not in self.scenario_df.index:
                    continue
                scenario_name = self.scenario_df.loc[uid, 'scenario']
                if scenario_name not in self.scenario_map:
                    continue 
                scenario_label = self.scenario_map[scenario_name]
                
                # Get Action Labels for this video
                video_actions = self.action_df[self.action_df['video_uid'] == uid]
                
                # Create Windows
                seq_len = CONFIG['hla']['seq_len']
                stride = CONFIG['data'].get('stride', 5)  # denser stride for more supervision
                
                num_seqs = (len(traj) - seq_len) // stride + 1
                if num_seqs <= 0:
                    continue
                
                for i in range(num_seqs):
                    start_idx = i * stride
                    end_idx = start_idx + seq_len
                    
                    # Window Sequence
                    window_seq = traj[start_idx:end_idx] # (Seq, 50, C)
                    ts_windows = timestamps[start_idx:end_idx] # (Seq, 50)
                    
                    action_labels_seq = []
                    
                    for w_idx in range(seq_len):
                        w_ts = ts_windows[w_idx]
                        w_start = w_ts[0]
                        w_end = w_ts[-1]
                        pad = self.action_pad
                        
                        # Find actions within padded window
                        match = video_actions[
                            (video_actions['timestamp_sec'] >= w_start - pad) & 
                            (video_actions['timestamp_sec'] <= w_end + pad)
                        ]
                        
                        if not match.empty:
                            # Majority vote within the window
                            counts = match['action'].value_counts()
                            act_name = counts.idxmax()
                            if act_name in self.action_map:
                                action_labels_seq.append(self.action_map[act_name])
                            else:
                                action_labels_seq.append(-1) # Unknown class
                        else:
                            action_labels_seq.append(-1) # No label in this window
                    
                    # Guard against variable-length windows
                    if window_seq.shape[0] != seq_len or len(action_labels_seq) != seq_len:
                        continue
                    
                    self.samples.append({
                        'video_uid': uid,
                        # Ensure contiguous, resizable tensors to avoid DataLoader storage resize errors
                        'inputs': torch.tensor(np.ascontiguousarray(window_seq), dtype=torch.float32), # (Seq, 50, C)
                        'scenario_label': torch.tensor(scenario_label, dtype=torch.long),
                        'action_labels': torch.tensor(action_labels_seq, dtype=torch.long) # (Seq,)
                    })
                
            except Exception as e:
                print(f"Error loading {uid}: {e}")
            
        # Save to cache after loading
        if cache_dir and use_cache:
            cache_path = Path(cache_dir) / f"dataset_{cache_key}.pkl"
            print(f"💾 Caching dataset...")
            save_cached_dataset(self, cache_path)

    def _augment_traj(self, traj):
        # traj: (N, 50, 6)
        if not self.add_norm_features:
            return traj
        accel = traj[..., :3]
        gyro = traj[..., 3:6]
        accel_norm = np.linalg.norm(accel, axis=2, keepdims=True)
        gyro_norm = np.linalg.norm(gyro, axis=2, keepdims=True)
        return np.concatenate([traj, accel_norm, gyro_norm], axis=2)
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        inputs = sample['inputs'].clone()  # (Seq, 50, C)
        
        # Add augmentation during training
        if getattr(self, 'training', False) and CONFIG['data'].get('augment', False):
            # Gaussian noise (small)
            if np.random.rand() < 0.5:
                noise = torch.randn_like(inputs) * 0.02
                inputs = inputs + noise
            
            # Random scaling (preserve relative magnitudes)
            if np.random.rand() < 0.5:
                scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.1  # ±5%
                inputs = inputs * scale
            
            # Time masking (mask random timesteps)
            if np.random.rand() < 0.3:
                mask_len = np.random.randint(1, 5)  # Mask 1-4 timesteps
                mask_start = np.random.randint(0, inputs.shape[1] - mask_len)
                inputs[:, mask_start:mask_start+mask_len, :] = 0
        
        return {
            'inputs': inputs,
            'scenario_label': sample['scenario_label'],
            'action_labels': sample['action_labels']
        }

# --- Models ---
class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: (B, C, T)
        b, c, t = x.shape
        se = x.mean(dim=2)  # (B, C)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).view(b, c, 1)
        return x * se

class LLE(nn.Module):
    """Low-Level Encoder with Variable Dilation CNNs + SE channel attention."""
    def __init__(self, config):
        super().__init__()
        filters = config['cnn_filters']
        in_ch = config['in_channels']
        self.se_reduction = config.get('se_reduction', 8)
        
        # Variable dilations to capture different periodicities [1, 2, 4]
        self.dilations = [1, 2, 4]
        
        # Create parallel multi-dilation conv blocks
        self.conv_blocks = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        
        for i in range(len(filters)):
            in_channels = in_ch if i == 0 else filters[i-1]
            
            # Each dilation gets equal share of output filters
            filters_per_dilation = filters[i] // len(self.dilations)
            remainder = filters[i] % len(self.dilations)
            
            # Create parallel convolutions with different dilations
            parallel_convs = nn.ModuleList()
            for j, dilation in enumerate(self.dilations):
                # Give remainder filters to first convolution
                out_ch = filters_per_dilation + (remainder if j == 0 else 0)
                parallel_convs.append(
                    nn.Conv1d(in_channels, out_ch, kernel_size=3, 
                             padding=dilation, dilation=dilation)
                )
            
            self.conv_blocks.append(parallel_convs)
            self.bns.append(nn.BatchNorm1d(filters[i]))
            self.se_blocks.append(SqueezeExcite(filters[i], reduction=self.se_reduction))
        
        self.gru = nn.GRU(filters[-1], config['gru_hidden'], config['gru_layers'], batch_first=True)
        self.fc = nn.Linear(config['gru_hidden'], config['embedding_dim'])
        
    def forward(self, x):
        # x: (B*Seq, 50, 6) -> (B*Seq, 6, 50)
        x = x.transpose(1, 2)
        
        # Apply multi-dilation conv blocks + SE channel attention
        for conv_block, bn, se in zip(self.conv_blocks, self.bns, self.se_blocks):
            conv_outputs = [conv(x) for conv in conv_block]
            x = torch.cat(conv_outputs, dim=1)  # Concat along channel dimension
            x = F.relu(bn(x))
            x = se(x)
        
        x = x.transpose(1, 2)
        _, h = self.gru(x)
        return self.fc(h[-1])

class HLA(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        if config.get('type', 'transformer') == 'transformer':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=config['nhead'],
                dim_feedforward=config['hidden_dim'] * 4,
                dropout=config.get('dropout', 0.1),
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
            self.pos_embed = nn.Parameter(torch.zeros(config['seq_len'] + 1, input_dim))
            self.head = nn.Linear(input_dim, config['num_classes'])
        else:
            self.gru = nn.GRU(input_dim, config['hidden_dim'], config['num_layers'], batch_first=True)
            self.head = nn.Linear(config['hidden_dim'], config['num_classes'])
        
    def forward(self, x):
        # x: (B, Seq, Emb)
        if hasattr(self, 'encoder'):
            seq_len = x.size(1)
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B,1,E)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, Seq+1, E)
            pos = self.pos_embed[:seq_len + 1, :].unsqueeze(0).to(x.device)
            x = x + pos
            enc = self.encoder(x)  # (B, Seq+1, Emb)
            cls_out = enc[:, 0, :]
            return self.head(cls_out)
        else:
            _, h = self.gru(x)
            return self.head(h[-1])

class HierarchicalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lle = LLE(config['lle'])
        self.hla = HLA(config['hla'], config['lle']['embedding_dim'])
        
        # Probing Head for LLE (Action Classification)
        # 6 Classes now
        self.action_head = nn.Linear(config['lle']['embedding_dim'], 6) 
        
    def forward(self, x):
        # x: (B, Seq, 50, 6)
        b, s, w, c = x.shape
        
        # Flatten for LLE
        x_flat = x.view(b*s, w, c)
        
        # LLE Forward
        embeddings = self.lle(x_flat) # (B*S, Emb)
        
        # Action Logits (for Probing/Auxiliary Loss)
        action_logits = self.action_head(embeddings) # (B*S, 6)
        action_logits = action_logits.view(b, s, 6)
        
        # Reshape for HLA
        embeddings_seq = embeddings.view(b, s, -1)
        
        # HLA Forward
        scenario_logits = self.hla(embeddings_seq) # (B, NumScenarios)
        
        return scenario_logits, action_logits


class FocalLoss(nn.Module):
    """
    Focal Loss with support for label smoothing and class weights.
    
    Focal Loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    - gamma: focusing parameter (higher = more focus on hard examples)
    - alpha: class weighting (can be per-class or scalar)
    - label_smoothing: smooths target distribution
    """
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight  # Class weights
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, num_classes) - logits
            targets: (B,) - class indices
        """
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Handle label smoothing
        if self.label_smoothing > 0:
            # Create smoothed target distribution
            confidence = 1.0 - self.label_smoothing
            smooth_value = self.label_smoothing / (num_classes - 1)
            
            # One-hot encoding
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(smooth_value)
            true_dist.scatter_(1, targets.unsqueeze(1), confidence)
            
            # For focal loss with label smoothing, we compute focal term on true class
            # but use smoothed distribution for cross-entropy
            ce_loss = -(true_dist * log_probs).sum(dim=1)
            
            # Get predicted probability for true class (for focal term)
            probs = F.softmax(inputs, dim=1)
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            # Standard focal loss without smoothing
            ce_loss = F.cross_entropy(inputs, targets, weight=None, reduction='none')
            probs = F.softmax(inputs, dim=1)
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply class weights if provided
        if self.weight is not None:
            if self.weight.device != inputs.device:
                self.weight = self.weight.to(inputs.device)
            weight_t = self.weight.gather(0, targets)
            focal_loss = self.alpha * weight_t * focal_weight * ce_loss
        else:
            focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Training ---
def train(args):
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("DATA SPLIT CONFIGURATION")
    print("="*80)
    
    # Load scenario labels with splits
    scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
    
    # Show overall split distribution
    print("\nOverall split distribution:")
    print(scenario_df['split'].value_counts().sort_index())
    
    # Use ONLY train and val splits
    # Exclude: 'test' (for final evaluation), 'multi' (ambiguous/multi-scenario videos)
    train_uids = scenario_df[scenario_df['split'] == 'train']['video_uid'].tolist()
    val_uids = scenario_df[scenario_df['split'] == 'val']['video_uid'].tolist()
    test_uids = scenario_df[scenario_df['split'] == 'test']['video_uid'].tolist()
    
    print(f"\nUsing splits:")
    print(f"  Train: {len(train_uids)} videos (67.3%)")
    print(f"  Val:   {len(val_uids)} videos (11.1%)")
    print(f"  Test:  {len(test_uids)} videos (10.9%) - Reserved for final evaluation")
    print(f"  Multi: Excluded (ambiguous scenarios)")
    
    # Check scenario distribution in train/val
    print("\nScenario distribution in train split:")
    train_scenarios = scenario_df[scenario_df['split'] == 'train']['scenario'].value_counts()
    for scenario, count in train_scenarios.items():
        print(f"  {scenario}: {count}")
    
    print("\nScenario distribution in val split:")
    val_scenarios = scenario_df[scenario_df['split'] == 'val']['scenario'].value_counts()
    for scenario, count in val_scenarios.items():
        print(f"  {scenario}: {count}")
    
    # Check for processed data availability
    from pathlib import Path
    processed_dir = Path(args.processed_dir)
    available_uids = set([p.parent.name for p in processed_dir.glob('*/seq.npz')])
    
    print(f"\nProcessed data availability:")
    print(f"  Total processed files: {len(available_uids)}")
    
    # Intersect with available processed data
    train_uids_available = [u for u in train_uids if u in available_uids]
    val_uids_available = [u for u in val_uids if u in available_uids]
    test_uids_available = [u for u in test_uids if u in available_uids]
    
    def format_availability(label, available, total):
        if total == 0:
            return f"  {label}: 0/0 (no {label.lower()} split in this run)"
        pct = available / total * 100
        return f"  {label}: {available}/{total} ({pct:.1f}% available)"

    print(f"\nFinal data splits (with processed data):")
    print(format_availability("Train", len(train_uids_available), len(train_uids)))
    print(format_availability("Val", len(val_uids_available), len(val_uids)))
    print(format_availability("Test", len(test_uids_available), len(test_uids)))
    print("="*80 + "\n")
    
    # Use the available UIDs
    train_uids = train_uids_available
    val_uids = val_uids_available
    
    # Use validated labels
    action_labels_path = "data/labels/action_labels_llm_validated.csv"
    
    # Create cache directory
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_ds = HierarchicalDataset(
        train_uids, 
        args.processed_dir, 
        "data/labels/scenario_labels.csv", 
        action_labels_path,
        cache_dir=cache_dir,
        use_cache=True  # Set to False to force reload
    )
    train_ds.training = True

    val_ds = HierarchicalDataset(
        val_uids, 
        args.processed_dir, 
        "data/labels/scenario_labels.csv", 
        action_labels_path,
        cache_dir=cache_dir,
        use_cache=True
    )
    val_ds.training = False



    # ===== DATASET STATISTICS =====
    print("\n" + "="*80)
    print("DATASET STATISTICS (After NaN Filtering)")
    print("="*80)

    train_samples = len(train_ds)
    val_samples = len(val_ds)
    total_samples = train_samples + val_samples

    print(f"\n📊 Dataset Sizes:")
    print(f"  Train samples: {train_samples:,}")
    print(f"  Val samples:   {val_samples:,}")
    print(f"  Total samples: {total_samples:,}")

    # Videos used
    train_videos_used = len(set([s['video_uid'] for s in train_ds.samples]))
    val_videos_used = len(set([s['video_uid'] for s in val_ds.samples]))

    print(f"\n📹 Videos Used:")
    print(f"  Train videos: {train_videos_used}")
    print(f"  Val videos:   {val_videos_used}")
    print(f"  Total videos: {train_videos_used + val_videos_used}")

    # Samples per video
    if train_videos_used > 0:
        avg_train = train_samples / train_videos_used
        print(f"\n📊 Samples per Video:")
        print(f"  Train: {avg_train:.1f} samples/video")
    if val_videos_used > 0:
        avg_val = val_samples / val_videos_used
        print(f"  Val:   {avg_val:.1f} samples/video")

    # Videos skipped
    train_videos_requested = len(train_uids)
    val_videos_requested = len(val_uids)
    train_skipped = train_videos_requested - train_videos_used
    val_skipped = val_videos_requested - val_videos_used

    if train_skipped > 0 or val_skipped > 0:
        print(f"\n⚠️ Videos Skipped (NaN/filtering):")
        print(f"  Train: {train_skipped}/{train_videos_requested} skipped ({train_skipped/train_videos_requested*100:.1f}%)")
        print(f"  Val:   {val_skipped}/{val_videos_requested} skipped ({val_skipped/val_videos_requested*100:.1f}%)")

    # Batch statistics
    bs = int(os.environ.get("BATCH_SIZE", CONFIG['training']['batch_size']))
    train_batches = (train_samples + bs - 1) // bs
    val_batches = (val_samples + bs - 1) // bs

    print(f"\n🔄 Training Configuration:")
    print(f"  Batch size: {bs}")
    print(f"  Train batches/epoch: {train_batches}")
    print(f"  Val batches/epoch: {val_batches}")
    print(f"  Steps per epoch: {train_batches}")

    print("="*80 + "\n")

    # Continue with existing code...
    # Scenario imbalance handling: use class weights...
    
    # Scenario imbalance handling: use class weights (no sampler to match val distribution)
    scenario_labels_train = torch.tensor([s['scenario_label'].item() for s in train_ds.samples])
    class_counts = torch.bincount(scenario_labels_train)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    # Allow overriding batch size via env
    bs = int(os.environ.get("BATCH_SIZE", CONFIG['training']['batch_size']))

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Initialize Model
    if args.baseline:
        from baseline_models import create_baseline_model
        model = create_baseline_model(args.baseline, CONFIG).to(device)
        print(f"Using baseline model: {args.baseline}")
    else:
        model = HierarchicalModel(CONFIG).to(device)
    
    # --- PROBING MODE ---
    if args.probe:
        print("\n" + "="*80)
        print("PROBING MODE ACTIVATED")
        print("="*80)
        if not args.checkpoint:
            raise ValueError("Must provide --checkpoint for probing mode")
            
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Handle both formats: direct state_dict or wrapped in dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Assume the dict itself is the state_dict
                state_dict = checkpoint
        else:
            # Direct state_dict (OrderedDict)
            state_dict = checkpoint
        
        # Strip 'module.' prefix if present (from DataParallel)
        if any(k.startswith('module.') for k in state_dict.keys()):
            print("⚠️  Detected 'module.' prefix in checkpoint (from DataParallel). Stripping...")
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # Remove 'module.' prefix (7 chars)
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        
        # Load with strict=False to allow missing keys (in case architecture changed)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing keys (will use random initialization): {len(missing_keys)}")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"   - {key}")
            else:
                print(f"   (showing first 10 of {len(missing_keys)})")
                for key in missing_keys[:10]:
                    print(f"   - {key}")
        
        if unexpected_keys:
            print(f"⚠️  Unexpected keys (ignored): {len(unexpected_keys)}")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"   - {key}")
        
        # Freeze LLE and HLA
        for param in model.lle.parameters():
            param.requires_grad = False
        for param in model.hla.parameters():
            param.requires_grad = False
            
        # Reset Action Head (Linear Probe)
        model.action_head = nn.Linear(CONFIG['lle']['embedding_dim'], 6).to(device)
        
        print("LLE and HLA frozen. Training only Action Head.")
        
        # Use Action Loss ONLY
        CONFIG['training']['alpha'] = 0.0 # Disable Scenario Loss
        CONFIG['training']['beta'] = 1.0  # Enable Action Loss
    # --------------------
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        # Specify device_ids to match the actual device
        model = nn.DataParallel(model, device_ids=[device.index] if device.index is not None else None)
    else:
        # Single GPU - ensure model is on the correct device
        model = model.to(device)
        
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=CONFIG['training']['lr'],
        weight_decay=CONFIG['training']['weight_decay']  # NEW: L2 regularization
    )
    
    # Learning Rate Scheduler: Warmup then Cosine Annealing
    warmup_epochs = CONFIG['training']['warmup_epochs']
    total_epochs = CONFIG['training']['epochs']
    min_factor = 0.2  # do not decay below 20% of base LR
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    print(f"Learning rate scheduler: Warmup {warmup_epochs} epochs -> Cosine annealing")
    
    # Loss Functions
    # Paper (Section 3.2): "We train... using... a weighted cross-entropy loss to handle class imbalance"
    
    # Calculate class weights for scenarios from training data
    print(f"Scenario class distribution: {class_counts.tolist()}")
    print(f"Scenario class weights: {class_weights.tolist()}")
    
    label_smoothing = CONFIG['training'].get('label_smoothing', 0.0)
    use_focal = CONFIG['training'].get('use_focal_loss', False)

    if use_focal:
        focal_gamma = CONFIG['training'].get('focal_gamma', 2.0)
        focal_alpha = CONFIG['training'].get('focal_alpha', 1.0)
        
        # Use class weights as alpha if provided, otherwise use scalar
        alpha = class_weights.to(device) if isinstance(focal_alpha, (list, torch.Tensor)) else focal_alpha
        
        criterion_scenario = FocalLoss(
            alpha=alpha,
            gamma=focal_gamma,
            weight=None,  # We use alpha for class weighting instead
            label_smoothing=label_smoothing,
            reduction='mean'
        )
        print(f"Using Focal Loss: gamma={focal_gamma}, alpha={focal_alpha}, label_smoothing={label_smoothing}")
    else:
        criterion_scenario = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=label_smoothing
        )
        print(f"Using CrossEntropyLoss: label_smoothing={label_smoothing}")

    criterion_action = nn.CrossEntropyLoss(ignore_index=-1)  # Kept for future probing
    
    # Early Stopping
    early_stopper = EarlyStopping(patience=CONFIG['training']['patience'])
    
    # Initialize W&B
    wandb_run = None
    if WANDB_AVAILABLE and not args.no_wandb:
        try:
            wandb_run = wandb.init(
                project="har-imu-training",
                name=f"hierarchical-{args.run_name}" if args.run_name else None,
                config={
                    **CONFIG,
                    "train_videos": len(train_uids),
                    "val_videos": len(val_uids),
                    "train_samples": len(train_ds),
                    "val_samples": len(val_ds),
                }
            )
            wandb.watch(model, log='all', log_freq=100)
        except Exception as e:
            print(f"W&B init failed ({e}); continuing without W&B logging.")
            wandb_run = None
    
    print("Starting training...")
    current_lr = optimizer.param_groups[0]['lr']
    
    for epoch in range(CONFIG['training']['epochs']):
        model.train()
        total_loss = 0

        # Get accumulation steps
        accumulation_steps = CONFIG['training'].get('gradient_accumulation_steps', 1)

        # Zero gradients at start of epoch
        optimizer.zero_grad()

        # Track batches processed for accumulation
        batches_processed = 0       

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            inputs = batch['inputs'].to(device)
            scenario_labels = batch['scenario_label'].to(device)
            action_labels = batch['action_labels'].to(device) # (B, S)
            
            s_logits, a_logits = model(inputs)
            
            # Scenario Loss (Primary)
            loss_s = criterion_scenario(s_logits, scenario_labels)
            
            # Action Loss (only if beta > 0)
            if CONFIG['training']['beta'] > 0:
                loss_a = criterion_action(a_logits.view(-1, 6), action_labels.view(-1))
                loss = CONFIG['training']['alpha'] * loss_s + CONFIG['training']['beta'] * loss_a
            else:
                # Semi-supervised: scenario loss only (EgoCharm methodology)
                loss = loss_s
            
            # Scale loss by accumulation steps (important!)
            loss = loss / accumulation_steps
            
            # Backward pass (accumulates gradients)
            loss.backward()
            
            batches_processed += 1
            total_loss += loss.item() * accumulation_steps  # Unscale for logging
            
            # Update weights only after accumulating enough gradients
            if batches_processed % accumulation_steps == 0:
                # Gradient clipping before update
                if CONFIG['training']['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['training']['grad_clip'])
                
                # Update weights
                optimizer.step()
                optimizer.zero_grad()  # Clear gradients for next accumulation
        
        # Handle remaining gradients if batch count isn't divisible by accumulation_steps
        if batches_processed % accumulation_steps != 0:
            if CONFIG['training']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['training']['grad_clip'])
            optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_train_loss:.4f}")
        
        # --- Validation ---
        model.eval()
        
        # Scenario Metrics
        all_s_preds = []
        all_s_labels = []
        
        # Action Metrics
        all_a_preds = []
        all_a_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['inputs'].to(device)
                s_labels = batch['scenario_label'].to(device)
                a_labels = batch['action_labels'].to(device)
                
                s_logits, a_logits = model(inputs)
                
                # Scenario Preds
                s_preds = torch.argmax(s_logits, dim=1)
                all_s_preds.extend(s_preds.cpu().numpy())
                all_s_labels.extend(s_labels.cpu().numpy())
                
                # Action Preds (Flatten and filter ignore_index)
                a_preds = torch.argmax(a_logits, dim=2).view(-1)
                a_labels_flat = a_labels.view(-1)
                
                mask = a_labels_flat != -1
                all_a_preds.extend(a_preds[mask].cpu().numpy())
                all_a_labels.extend(a_labels_flat[mask].cpu().numpy())
        
        # Calculate Metrics
        val_s_f1 = f1_score(all_s_labels, all_s_preds, average='macro')
        val_s_acc = (np.array(all_s_preds) == np.array(all_s_labels)).mean()
        
        val_a_f1 = 0
        val_a_acc = 0
        if len(all_a_labels) > 0:
            val_a_f1 = f1_score(all_a_labels, all_a_preds, average='macro')
            val_a_acc = (np.array(all_a_preds) == np.array(all_a_labels)).mean()
            
        print(f"Val Scenario F1: {val_s_f1:.4f} | Acc: {val_s_acc:.4f}")
        print(f"Val Action F1: {val_a_f1:.4f} | Acc: {val_a_acc:.4f}")
        

        # Log to W&B
        if wandb_run is not None:
            try:
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_scenario_f1": val_s_f1,
                    "val_scenario_acc": val_s_acc,
                    "val_action_f1": val_a_f1,
                    "val_action_acc": val_a_acc,
                    "learning_rate": current_lr
                }
                
                wandb.log(log_dict)
            except Exception as e:
                print(f"⚠️ WandB logging failed (continuing training): {e}")
            
            # Confusion Matrix (Every 5 epochs)
            if (epoch + 1) % 5 == 0:
                try:
                    wandb.log({
                        "conf_mat_scenario": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=all_s_labels,
                            preds=all_s_preds,
                            class_names=list(train_ds.scenario_map.keys())
                        )
                    })
                except Exception as e:
                    print(f"⚠️ WandB confusion matrix logging failed (continuing training): {e}")
        
        # Step the learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr:.2e}")
        
        # Early Stopping check
        early_stopper(val_s_f1, model)
        
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break
            
    # Save Best Model
    os.makedirs(args.output_dir, exist_ok=True)
    if early_stopper.best_model_state:
        torch.save(early_stopper.best_model_state, Path(args.output_dir) / "best_model.pth")
        print(f"Best model saved (Scenario F1: {early_stopper.best_score:.4f})")
    
    # Save Last Model
    torch.save(model.state_dict(), Path(args.output_dir) / "last_model.pth")
    print("Last model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-uids-file", type=str, default="target_uids.csv")
    parser.add_argument("--processed-dir", type=str, default="data/processed_ego4d")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument('--run-name', type=str, default='hierarchical_har')
    parser.add_argument('--probe', action='store_true', help='Train action probe on frozen model')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint for probing')
    parser.add_argument('--cv', action='store_true', help='Use K-fold cross validation')
    parser.add_argument('--n-folds', type=int, default=4, help='Number of CV folds (default: 4)')
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument('--baseline', type=str, choices=['cnn_mlp', 'egocharm', 'cnn_lstm_gru'], 
                   help='Use baseline model instead of default')
    args = parser.parse_args()
    
    # Cross Validation Mode
    if args.cv:
        from sklearn.model_selection import StratifiedKFold
        import json
        
        print("=" * 80)
        print(f"{args.n_folds}-FOLD CROSS VALIDATION MODE")
        print("=" * 80)
        
        # Load scenario labels
        scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
        usable_df = scenario_df[scenario_df['split'].isin(['train', 'val'])].copy()
        
        print(f"\nUsing {len(usable_df)} videos for {args.n_folds}-fold CV")
        print(f"Excluded: test={len(scenario_df[scenario_df['split']=='test'])}, multi={len(scenario_df[scenario_df['split']=='multi'])}")
        
        # Create stratified folds
        video_scenarios = usable_df[['video_uid', 'scenario']].copy()
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(video_scenarios['video_uid'], video_scenarios['scenario'])):
            print(f"\n{'='*80}")
            print(f"FOLD {fold_idx + 1}/{args.n_folds}")
            print(f"{'='*80}")
            
            # Get fold-specific UIDs
            fold_train_uids = video_scenarios.iloc[train_idx]['video_uid'].tolist()
            fold_val_uids = video_scenarios.iloc[val_idx]['video_uid'].tolist()
            
            # Create fold-specific args
            fold_args = argparse.Namespace(**vars(args))
            fold_args.cv = False  # Disable CV for individual fold
            fold_args.run_name = f"{args.run_name}_fold{fold_idx+1}" if not args.no_wandb else None
            fold_args.output_dir = str(Path(args.output_dir) / f"fold{fold_idx+1}")
            
            # Temporarily override train function to use fold UIDs
            # We do this by modifying the scenario_df before passing to train
            original_scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
            
            # Mark fold UIDs appropriately
            temp_scenario_df = original_scenario_df.copy()
            # Preserve original test/multi labels; exclude other non-fold rows
            temp_scenario_df.loc[~temp_scenario_df['split'].isin(['test', 'multi']), 'split'] = 'excluded'
            temp_scenario_df.loc[temp_scenario_df['video_uid'].isin(fold_train_uids), 'split'] = 'train'
            temp_scenario_df.loc[temp_scenario_df['video_uid'].isin(fold_val_uids), 'split'] = 'val'
            
            # Save temporarily
            temp_scenario_df.to_csv("data/labels/scenario_labels_temp.csv", index=False)
            
            # Modify args to use temp file
            original_labels_path = "data/labels/scenario_labels.csv"
            os.rename("data/labels/scenario_labels.csv", "data/labels/scenario_labels_backup.csv")
            os.rename("data/labels/scenario_labels_temp.csv", "data/labels/scenario_labels.csv")
            
            try:
                # Train this fold
                train(fold_args)
                
                # Record results (would need to modify train() to return metrics)
                # For now, we'll just print completion
                print(f"Fold {fold_idx+1} training complete")
                
            finally:
                # Restore original file
                os.rename("data/labels/scenario_labels.csv", "data/labels/scenario_labels_temp.csv")
                os.rename("data/labels/scenario_labels_backup.csv", "data/labels/scenario_labels.csv")
                os.remove("data/labels/scenario_labels_temp.csv")
        
        print("\n" + "=" * 80)
        print("CROSS VALIDATION COMPLETE")
        print("=" * 80)
        print(f"\nAll {args.n_folds} folds completed. Check WandB for detailed results.")
        print("You can compare fold performances in the WandB dashboard.")
        
    else:
        # Standard single train/val split
        train(args)
