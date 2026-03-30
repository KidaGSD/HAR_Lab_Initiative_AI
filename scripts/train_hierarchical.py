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
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import WeightedRandomSampler

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
        'seq_len': 30,    # 30 seconds context (can test 20 for latency tradeoff)
        'nhead': 4,
        'dropout': 0.1,
        'type': 'transformer' # transformer encoder for long-range modeling
    },
    'training': {
        'batch_size': 256,     # default; override with BATCH_SIZE env on larger GPUs
        'lr': 1e-4,            # Base LR (will warmup then cosine)
        'epochs': 50,
        'patience': 15,
        'alpha': 1.0,
        'beta': 0.5,           # Default: focus on scenario; action loss used in probe or if explicitly enabled
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'warmup_epochs': 5     # Warmup then cosine anneal
    },
    'data': {
        'action_label_pad': 0.5,   # seconds to expand action labels on each side
        'per_video_center': True,  # subtract per-video mean after global z-score
        'add_norm_features': True, # add accel/gyro norms as extra channels
        'excluded_actions': []
    }
}

DEFAULT_ACTION_CLASSES = [
    'Stationary',
    'Locomotion',
    'Essential Operation',
    'Object Transfer',
    'Search',
    'Error / Correction',
]

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


def load_checkpoint_flexible(model, checkpoint_path, device):
    """Load checkpoint with/without DataParallel prefixes."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    def _count_overlap_keys(sd, ref_keys):
        return sum(1 for k in sd.keys() if k in ref_keys)

    model_keys = set(model.state_dict().keys())
    overlap_raw = _count_overlap_keys(state_dict, model_keys)
    if overlap_raw == 0:
        # Translate keys only when there is no direct overlap.
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
        if has_module_prefix:
            translated = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        else:
            translated = {f"module.{k}": v for k, v in state_dict.items()}
        overlap_translated = _count_overlap_keys(translated, model_keys)
        if overlap_translated > 0:
            state_dict = translated
            print(f"Checkpoint key translation applied ({overlap_translated} overlapping tensors).")
        else:
            raise RuntimeError(
                "Could not match checkpoint keys to model keys (0 overlapping tensors), "
                "even after DataParallel key translation."
            )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    loaded_count = len(model_keys) - len(missing)
    print(
        f"Checkpoint loaded: {loaded_count}/{len(model_keys)} tensors matched; "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )


def _sanitize_metric_name(name):
    safe = str(name).strip().lower()
    for ch in [" ", "/", "-", "(", ")", ","]:
        safe = safe.replace(ch, "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def summarize_class_metrics(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    rows = []
    payload = {}
    for idx, class_name in enumerate(class_names):
        metric_prefix = _sanitize_metric_name(class_name)
        rows.append({
            "name": class_name,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        })
        payload[f"{metric_prefix}_precision"] = float(precision[idx])
        payload[f"{metric_prefix}_recall"] = float(recall[idx])
        payload[f"{metric_prefix}_f1"] = float(f1[idx])
        payload[f"{metric_prefix}_support"] = int(support[idx])
    return rows, payload


def get_active_action_names():
    excluded = set(CONFIG.get('data', {}).get('excluded_actions', []))
    return [name for name in DEFAULT_ACTION_CLASSES if name not in excluded]

# --- Dataset --- #Custom Pytorch dataset #Loads IMUs, labels, normalize, and create training samp 
class HierarchicalDataset(torch.utils.data.Dataset):
    def __init__(self, take_uids, processed_dir, scenario_labels_source, action_labels_source):
        self.samples = []
        
        # Load Labels
        print("Loading label files...")
        if isinstance(scenario_labels_source, pd.DataFrame):
            self.scenario_df = scenario_labels_source.copy().set_index('video_uid')
        else:
            self.scenario_df = pd.read_csv(scenario_labels_source).set_index('video_uid')
        if isinstance(action_labels_source, pd.DataFrame):
            self.action_df = action_labels_source.copy()
        else:
            self.action_df = pd.read_csv(action_labels_source)
        
        # Optional scenario exclusion (e.g., Gardening)
        excluded = set(CONFIG.get('data', {}).get('excluded_scenarios', []))
        if excluded:
            print(f"Excluding scenarios: {excluded}")
            self.scenario_df = self.scenario_df[~self.scenario_df['scenario'].isin(excluded)]
        
        # Map Scenario Names to Integers
        self.scenario_map = {name: i for i, name in enumerate(sorted(self.scenario_df['scenario'].unique()))}
        self.num_scenarios = len(self.scenario_map)
        self.idx_to_scenario = {v: k for k, v in self.scenario_map.items()}
        print(f"Scenarios: {self.scenario_map}")
        
        # Map active action names to integers after optional exclusions.
        active_action_names = get_active_action_names()
        self.action_map = {name: i for i, name in enumerate(active_action_names)}
        print(f"Actions: {self.action_map}")
        # Normalize equivalent action names used in refined gold exports.
        self.action_df["action"] = self.action_df["action"].replace(
            {"Task Operation": "Essential Operation"}
        )
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
        dropped_windows_for_stats = 0
        
        for uid in tqdm(take_uids, desc='Collecting normalization data'):
            seq_path = Path(processed_dir) / uid / 'seq.npz'
            if not seq_path.exists():
                continue
            try:
                data = np.load(seq_path)
                traj = data['traj']  # (N, 50, 6)
                traj = self._augment_traj(traj)  # add norms if enabled
                if len(traj) > 0:
                    finite_mask = np.isfinite(traj).all(axis=(1, 2))
                    dropped_windows_for_stats += int((~finite_mask).sum())
                    traj = traj[finite_mask]
                if len(traj) > 0 and uid in self.scenario_df.index:
                    all_data.append(traj)
                    valid_uids.append(uid)
            except Exception:
                continue
        
        if len(all_data) == 0:
            raise ValueError("No valid data found for normalization!")
            
        all_data = np.concatenate(all_data, axis=0)  # (Total_Windows, 50, C)
        self.global_mean = all_data.mean(axis=(0, 1))  # (C,) - mean per channel
        self.global_std = all_data.std(axis=(0, 1)) + 1e-6  # (C,) - std per channel
        if not np.isfinite(self.global_mean).all() or not np.isfinite(self.global_std).all():
            raise ValueError(
                "Global mean/std contains NaN/Inf after sanitization. "
                "Please inspect seq.npz files for severe corruption."
            )
        
        print(f"Global statistics computed from {len(valid_uids)} videos:")
        print(f"  Mean: {self.global_mean}")
        print(f"  Std:  {self.global_std}")
        print(f"  Data range: [{all_data.min():.2f}, {all_data.max():.2f}]")
        if dropped_windows_for_stats > 0:
            print(f"  Dropped invalid windows for stats: {dropped_windows_for_stats}")
        # ==========================================
        
        # Iterate Videos
        dropped_windows_for_training = 0
        for uid in tqdm(take_uids, desc='Loading Data'):
            seq_path = Path(processed_dir) / uid / 'seq.npz'
            if not seq_path.exists():
                continue
                
            try:
                data = np.load(seq_path)
                traj = data['traj'] # (N, 50, 6)
                traj = self._augment_traj(traj)
                timestamps = data['timestamp'] # (N, 50)
                
                # Drop invalid windows before normalization/sampling
                if len(traj) > 0:
                    finite_mask = np.isfinite(traj).all(axis=(1, 2))
                    dropped_windows_for_training += int((~finite_mask).sum())
                    traj = traj[finite_mask]
                    timestamps = timestamps[finite_mask]
                
                # GLOBAL normalization + optional per-video centering
                if len(traj) > 0:
                    traj = (traj - self.global_mean) / self.global_std
                    if self.per_video_center:
                        traj = traj - traj.mean(axis=(0, 1), keepdims=True)
                
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
                stride = 5  # denser stride for more supervision
                
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
        
        if dropped_windows_for_training > 0:
            print(f"Dropped invalid windows during loading: {dropped_windows_for_training}")

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
        return self.samples[idx]

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

class LLE(nn.Module): #CNN and SE and GRU
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

# --- Training ---
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("DATA SPLIT CONFIGURATION")
    print("="*80)
    
    # Single labels CSV contains action + scenario annotations.
    labels_df = pd.read_csv(args.labels_csv)
    split_source_df = pd.read_csv(args.scenario_labels_csv)

    required_label_cols = {"video_uid", "scenario", "action", "timestamp_sec"}
    missing_required = sorted(required_label_cols - set(labels_df.columns))
    if missing_required:
        raise ValueError(
            f"--labels-csv is missing required columns: {missing_required}"
        )

    # Build per-video scenario/split table.
    if {"video_uid", "scenario", "split"}.issubset(labels_df.columns):
        scenario_df = (
            labels_df[["video_uid", "scenario", "split"]]
            .dropna(subset=["video_uid", "scenario", "split"])
            .drop_duplicates(subset=["video_uid"], keep="first")
        )
    else:
        scenario_df = (
            labels_df[["video_uid", "scenario"]]
            .dropna(subset=["video_uid", "scenario"])
            .drop_duplicates(subset=["video_uid"], keep="first")
        )
        scenario_df = scenario_df.merge(
            split_source_df[["video_uid", "split"]],
            on="video_uid",
            how="left",
        )
        missing_split = int(scenario_df["split"].isna().sum())
        if missing_split > 0:
            print(f"Warning: {missing_split} videos missing split mapping; assigning to train.")
            scenario_df["split"] = scenario_df["split"].fillna("train")

    # Row-level split counts from labels CSV (requested visibility).
    # If labels already contain split, avoid merge suffixes (split_x/split_y).
    if "split" in labels_df.columns:
        labels_rows_df = labels_df.copy()
    else:
        labels_rows_df = labels_df.merge(
            scenario_df[["video_uid", "split"]].drop_duplicates(subset=["video_uid"]),
            on="video_uid",
            how="left",
        )
    labels_rows_df["split"] = labels_rows_df["split"].fillna("unmapped")
    
    # Show overall split distribution
    print("\nOverall split distribution:")
    print(scenario_df['split'].value_counts().sort_index())
    print("\nLabel rows per split (from --labels-csv):")
    print(labels_rows_df["split"].value_counts().sort_index())
    
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
    # Effective row counts after processed-data availability filtering.
    labels_rows_effective = labels_rows_df[labels_rows_df["video_uid"].isin(available_uids)].copy()
    excluded = set(CONFIG.get("data", {}).get("excluded_scenarios", []))
    if excluded and "scenario" in labels_rows_effective.columns:
        labels_rows_effective = labels_rows_effective[
            ~labels_rows_effective["scenario"].isin(excluded)
        ]
    print("\nEffective label rows per split (after processed-data + exclusions):")
    print(labels_rows_effective["split"].value_counts().sort_index())
    print("="*80 + "\n")
    
    # Use the available UIDs
    train_uids = train_uids_available
    val_uids = val_uids_available
    test_uids = test_uids_available
    
    train_ds = HierarchicalDataset(
        train_uids, 
        args.processed_dir, 
        scenario_df,
        labels_df,
    )
    
    val_ds = HierarchicalDataset(
        val_uids, 
        args.processed_dir, 
        scenario_df,
        labels_df,
    )
    
    test_ds = HierarchicalDataset(
        test_uids,
        args.processed_dir,
        scenario_df,
        labels_df,
    )
    
    # Keep model output classes aligned with active scenario set (e.g., excluding Gardening).
    num_scenarios = len(train_ds.scenario_map)
    CONFIG['hla']['num_classes'] = num_scenarios

    # Scenario imbalance handling: use class weights (no sampler to match val distribution)
    # minlength ensures weight tensor size always matches model output dimension.
    scenario_labels_train = torch.tensor([s['scenario_label'].item() for s in train_ds.samples])
    class_counts = torch.bincount(scenario_labels_train, minlength=num_scenarios).float()
    class_weights = torch.zeros_like(class_counts)
    nonzero_mask = class_counts > 0
    class_weights[nonzero_mask] = 1.0 / class_counts[nonzero_mask]
    class_weights = class_weights * (nonzero_mask.sum() / class_weights[nonzero_mask].sum())
    
    # Allow overriding batch size via env
    bs = int(os.environ.get("BATCH_SIZE", CONFIG['training']['batch_size']))

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory_device="cuda"
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory_device="cuda"
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory_device="cuda"
    )
    
    # Initialize Model
    model = HierarchicalModel(CONFIG).to(device)
    if args.init_checkpoint:
        print(f"Loading initialization checkpoint: {args.init_checkpoint}")
        load_checkpoint_flexible(model, args.init_checkpoint, device)
        print("Checkpoint loaded for fine-tuning.")
    
    # --- PROBING MODE ---
    if args.probe:
        print("\n" + "="*80)
        print("PROBING MODE ACTIVATED")
        print("="*80)
        if not args.checkpoint:
            raise ValueError("Must provide --checkpoint for probing mode")
            
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
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
        model = nn.DataParallel(model)
        
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
    
    criterion_scenario = nn.CrossEntropyLoss(weight=class_weights.to(device))
    criterion_action = nn.CrossEntropyLoss(ignore_index=-1)  # Kept for future probing
    
    # Early Stopping
    early_stopper = EarlyStopping(patience=CONFIG['training']['patience'])
    
    # Initialize W&B
    wandb_run = None
    if WANDB_AVAILABLE and not args.no_wandb:
        try:
            wandb_run = wandb.init(
                project="har-imu-training",
                name=f"{args.run_name}" if args.run_name else None,
                config={
                    **CONFIG,
                    "train_videos": len(train_uids),
                    "val_videos": len(val_uids),
                    "test_videos": len(test_uids),
                    "train_samples": len(train_ds),
                    "val_samples": len(val_ds),
                    "test_samples": len(test_ds),
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
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs = batch['inputs'].to(device)
            scenario_labels = batch['scenario_label'].to(device)
            action_labels = batch['action_labels'].to(device) # (B, S)
            
            optimizer.zero_grad()
            
            s_logits, a_logits = model(inputs)
            
            # Scenario Loss (Primary)
            loss_s = criterion_scenario(s_logits, scenario_labels)
            
            # Action Loss (only if beta > 0)
            # Paper uses semi-supervised: train ONLY on scenario labels
            if CONFIG['training']['beta'] > 0:
                loss_a = criterion_action(a_logits.view(-1, 6), action_labels.view(-1))
                loss = CONFIG['training']['alpha'] * loss_s + CONFIG['training']['beta'] * loss_a
            else:
                # Semi-supervised: scenario loss only (EgoCharm methodology)
                loss = loss_s
            
            loss.backward()
            
            # NEW: Gradient Clipping (Paper: Supplemental S2)
            if CONFIG['training']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['training']['grad_clip'])
            
            optimizer.step()
            
            total_loss += loss.item()
            
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
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_scenario_f1": val_s_f1,
                "val_scenario_acc": val_s_acc,
                "val_action_f1": val_a_f1,
                "val_action_acc": val_a_acc,
                "learning_rate": current_lr  # Log LR for monitoring
            })
            
            # Confusion Matrix (Every 5 epochs)
            if (epoch + 1) % 5 == 0:
                wandb.log({
                    "conf_mat_scenario": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=all_s_labels,
                        preds=all_s_preds,
                        class_names=list(train_ds.scenario_map.keys())
                    )
                })
        
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

    # --- Final Test Evaluation (on best model state) ---
    if len(test_ds) == 0:
        print("No test samples available; skipping final test evaluation.")
        return

    # Evaluate using the best checkpoint selected by validation.
    if early_stopper.best_model_state:
        model.load_state_dict(early_stopper.best_model_state)
    model.eval()

    test_s_preds = []
    test_s_labels = []
    test_a_preds = []
    test_a_labels = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['inputs'].to(device)
            s_labels = batch['scenario_label'].to(device)
            a_labels = batch['action_labels'].to(device)

            s_logits, a_logits = model(inputs)
            s_preds = torch.argmax(s_logits, dim=1)
            test_s_preds.extend(s_preds.cpu().numpy())
            test_s_labels.extend(s_labels.cpu().numpy())

            a_preds = torch.argmax(a_logits, dim=2).view(-1)
            a_labels_flat = a_labels.view(-1)
            mask = a_labels_flat != -1
            test_a_preds.extend(a_preds[mask].cpu().numpy())
            test_a_labels.extend(a_labels_flat[mask].cpu().numpy())

    test_s_f1 = f1_score(test_s_labels, test_s_preds, average='macro')
    test_s_acc = (np.array(test_s_preds) == np.array(test_s_labels)).mean()
    test_a_f1 = 0.0
    test_a_acc = 0.0
    if len(test_a_labels) > 0:
        test_a_f1 = f1_score(test_a_labels, test_a_preds, average='macro')
        test_a_acc = (np.array(test_a_preds) == np.array(test_a_labels)).mean()

    print("\nFinal Test Results (best model):")
    print(f"  Test Scenario F1: {test_s_f1:.4f} | Acc: {test_s_acc:.4f}")
    print(f"  Test Action   F1: {test_a_f1:.4f} | Acc: {test_a_acc:.4f}")

    scenario_rows, scenario_payload = summarize_class_metrics(
        test_s_labels,
        test_s_preds,
        list(train_ds.scenario_map.keys()),
    )
    print("  Scenario per-class metrics:")
    for row in scenario_rows:
        print(
            f"    {row['name']}: "
            f"P={row['precision']:.4f} R={row['recall']:.4f} "
            f"F1={row['f1']:.4f} support={row['support']}"
        )

    action_rows = []
    action_payload = {}
    if len(test_a_labels) > 0:
        action_class_names = [train_ds.idx_to_action[i] for i in range(len(train_ds.action_map))]
        action_rows, action_payload = summarize_class_metrics(
            test_a_labels,
            test_a_preds,
            action_class_names,
        )
        print("  Action per-class metrics:")
        for row in action_rows:
            print(
                f"    {row['name']}: "
                f"P={row['precision']:.4f} R={row['recall']:.4f} "
                f"F1={row['f1']:.4f} support={row['support']}"
            )

    if wandb_run is not None:
        payload = {
            "test_scenario_f1": test_s_f1,
            "test_scenario_acc": test_s_acc,
            "test_action_f1": test_a_f1,
            "test_action_acc": test_a_acc,
        }
        payload.update({f"test_scenario_{k}": v for k, v in scenario_payload.items()})
        payload.update({f"test_action_{k}": v for k, v in action_payload.items()})
        wandb.log(payload)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-uids-file", type=str, default="target_uids.csv")
    parser.add_argument("--processed-dir", type=str, default="data/processed_ego4d")
    parser.add_argument("--labels-csv", type=str, default="data/labels/action_labels_llm_clean_refined.csv")
    parser.add_argument("--scenario-labels-csv", type=str, default="data/annotation_rounds/final_gold_dataset/HAR_dataset_155.csv")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument('--run-name', type=str, default='hierarchical_har')
    parser.add_argument('--probe', action='store_true', help='Train action probe on frozen model')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint for probing')
    parser.add_argument('--init-checkpoint', type=str, default=None, help='Initialize model from checkpoint for fine-tuning')
    parser.add_argument('--cv', action='store_true', help='Use K-fold cross validation')
    parser.add_argument('--n-folds', type=int, default=4, help='Number of CV folds (default: 4)')
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--exclude-scenarios",
        type=str,
        default="",
        help='Comma-separated scenarios to exclude, e.g. "Gardening,Desk Work"',
    )
    parser.add_argument(
        "--exclude-actions",
        type=str,
        default="",
        help='Comma-separated actions to exclude, e.g. "Search,Error / Correction"',
    )
    args = parser.parse_args()
    
    # Apply runtime exclusions to global CONFIG
    if args.exclude_scenarios.strip():
        excluded = [s.strip() for s in args.exclude_scenarios.split(",") if s.strip()]
        CONFIG.setdefault("data", {})["excluded_scenarios"] = excluded
        print(f"Runtime scenario exclusion enabled: {excluded}")
    if args.exclude_actions.strip():
        excluded_actions = [s.strip() for s in args.exclude_actions.split(",") if s.strip()]
        CONFIG.setdefault("data", {})["excluded_actions"] = excluded_actions
        print(f"Runtime action exclusion enabled: {excluded_actions}")
    
    # Cross Validation Mode
    if args.cv:
        from sklearn.model_selection import StratifiedKFold
        import json
        
        print("=" * 80)
        print(f"{args.n_folds}-FOLD CROSS VALIDATION MODE")
        print("=" * 80)
        
        # Load scenario labels
        scenario_df = pd.read_csv(args.scenario_labels_csv)
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
            original_scenario_df = pd.read_csv(args.scenario_labels_csv)
            
            # Mark fold UIDs appropriately
            temp_scenario_df = original_scenario_df.copy()
            # Preserve original test/multi labels; exclude other non-fold rows
            temp_scenario_df.loc[~temp_scenario_df['split'].isin(['test', 'multi']), 'split'] = 'excluded'
            temp_scenario_df.loc[temp_scenario_df['video_uid'].isin(fold_train_uids), 'split'] = 'train'
            temp_scenario_df.loc[temp_scenario_df['video_uid'].isin(fold_val_uids), 'split'] = 'val'
            
            # Save temporarily
            scenario_path = Path(args.scenario_labels_csv)
            temp_path = str(scenario_path.with_name("scenario_labels_temp.csv"))
            backup_path = str(scenario_path.with_name("scenario_labels_backup.csv"))
            temp_scenario_df.to_csv(temp_path, index=False)
            
            # Modify args to use temp file
            os.rename(args.scenario_labels_csv, backup_path)
            os.rename(temp_path, args.scenario_labels_csv)
            
            try:
                # Train this fold
                train(fold_args)
                
                # Record results (would need to modify train() to return metrics)
                # For now, we'll just print completion
                print(f"Fold {fold_idx+1} training complete")
                
            finally:
                # Restore original file
                os.rename(args.scenario_labels_csv, temp_path)
                os.rename(backup_path, args.scenario_labels_csv)
                os.remove(temp_path)
        
        print("\n" + "=" * 80)
        print("CROSS VALIDATION COMPLETE")
        print("=" * 80)
        print(f"\nAll {args.n_folds} folds completed. Check WandB for detailed results.")
        print("You can compare fold performances in the WandB dashboard.")
        
    else:
        # Standard single train/val split
        train(args)
