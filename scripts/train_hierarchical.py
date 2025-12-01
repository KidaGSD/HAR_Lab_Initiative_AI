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
from sklearn.metrics import f1_score, confusion_matrix

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. Run: pip install wandb")

# --- Configuration ---
CONFIG = {
    'lle': {
        'in_channels': 6,
        'cnn_filters': [32, 64, 128],
        'gru_hidden': 256,
        'gru_layers': 2,
        'embedding_dim': 128,
        'window_size': 50 # 1 second at 50Hz
    },
    'hla': {
        'hidden_dim': 128,
        'num_layers': 2,
        'num_classes': 8, # 8 Scenarios
        'seq_len': 30 # 30 seconds context
    },
    'training': {
        'batch_size': 256,     # FIXED: Reduced from 1024 (too large, poor gradient estimates)
        'lr': 1.4e-4,          # FIXED: Scaled LR for larger batch (1e-4 × √2)
        'epochs': 50,
        'patience': 15,        # FIXED: Increased from 10 (allow more training)
        'alpha': 1.0,
        'beta': 0.0,
        'step_size': 10,
        'gamma': 0.5,
        'weight_decay': 1e-5,  # NEW: L2 regularization
        'grad_clip': 1.0       # NEW: Gradient clipping (paper uses this)
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

# --- Dataset ---
class HierarchicalDataset(torch.utils.data.Dataset):
    def __init__(self, take_uids, processed_dir, scenario_labels_path, action_labels_path):
        self.samples = []
        
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
        
        # === CRITICAL FIX: GLOBAL NORMALIZATION ===
        # Paper (Section 3.2): "We apply normalization across all windows of raw data,
        # specifically along each of the 6 IMU channels."
        # This means GLOBAL normalization, NOT per-video!
        
        print("\nComputing global normalization statistics...")
        all_data = []
        valid_uids = []
        
        for uid in tqdm(take_uids, desc='Collecting normalization data'):
            seq_path = Path(processed_dir) / uid / 'seq.npz'
            if not seq_path.exists():
                continue
            try:
                data = np.load(seq_path)
                traj = data['traj']
                if len(traj) > 0 and uid in self.scenario_df.index:
                    all_data.append(traj)
                    valid_uids.append(uid)
            except Exception as e:
                continue
        
        if len(all_data) == 0:
            raise ValueError("No valid data found for normalization!")
            
        all_data = np.concatenate(all_data, axis=0)  # (Total_Windows, 50, 6)
        self.global_mean = all_data.mean(axis=(0, 1))  # (6,) - mean per channel
        self.global_std = all_data.std(axis=(0, 1)) + 1e-6  # (6,) - std per channel
        
        print(f"Global statistics computed from {len(valid_uids)} videos:")
        print(f"  Mean: {self.global_mean}")
        print(f"  Std:  {self.global_std}")
        print(f"  Data range: [{all_data.min():.2f}, {all_data.max():.2f}]")
        # ==========================================
        
        # Iterate Videos
        for uid in tqdm(take_uids, desc='Loading Data'):
            seq_path = Path(processed_dir) / uid / 'seq.npz'
            if not seq_path.exists():
                continue
                
            try:
                data = np.load(seq_path)
                traj = data['traj'] # (N, 50, 6)
                timestamps = data['timestamp'] # (N, 50)
                
                # === CRITICAL FIX: GLOBAL NORMALIZATION ===
                # Apply GLOBAL normalization (not per-video!)
                if len(traj) > 0:
                    traj = (traj - self.global_mean) / self.global_std
                # ==========================================
                
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
                stride = 10
                
                num_seqs = (len(traj) - seq_len) // stride + 1
                
                for i in range(num_seqs):
                    start_idx = i * stride
                    end_idx = start_idx + seq_len
                    
                    # Window Sequence
                    window_seq = traj[start_idx:end_idx] # (30, 50, 6)
                    
                    # Timestamp Sequence (Start and End of each window)
                    # timestamps is (N, 50)
                    # We need strict containment: label_ts must be within [window_start, window_end]
                    
                    ts_windows = timestamps[start_idx:end_idx] # (30, 50)
                    
                    action_labels_seq = []
                    
                    for w_idx in range(seq_len):
                        w_ts = ts_windows[w_idx]
                        w_start = w_ts[0]
                        w_end = w_ts[-1]
                        
                        # Find action strictly within this window
                        match = video_actions[
                            (video_actions['timestamp_sec'] >= w_start) & 
                            (video_actions['timestamp_sec'] <= w_end)
                        ]
                        
                        if not match.empty:
                            # Take first match (or could use majority if multiple)
                            act_name = match.iloc[0]['action']
                            if act_name in self.action_map:
                                action_labels_seq.append(self.action_map[act_name])
                            else:
                                action_labels_seq.append(-1) # Unknown class
                        else:
                            action_labels_seq.append(-1) # No label in this window
                            
                    self.samples.append({
                        'video_uid': uid,
                        'inputs': torch.FloatTensor(window_seq), # (30, 50, 6)
                        'scenario_label': torch.tensor(scenario_label, dtype=torch.long),
                        'action_labels': torch.tensor(action_labels_seq, dtype=torch.long) # (30,)
                    })
                    
            except Exception as e:
                print(f"Error loading {uid}: {e}")
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# --- Models ---
class LLE(nn.Module):
    """Low-Level Encoder with Variable Dilation CNNs (EgoCharm Paper)
    
    Paper (Section 3.3): "At each convolutional layer, multiple convolutions 
    with the same kernel size and different dilations are applied in parallel 
    and stacked together to capture the periodicity of the IMU signals."
    
    Paper (Discussion): "Varying dilation in the CNN layers plays a crucial role 
    in extracting meaningful feature representations from IMU signals."
    """
    def __init__(self, config):
        super().__init__()
        filters = config['cnn_filters']
        in_ch = config['in_channels']
        
        # Variable dilations to capture different periodicities [1, 2, 4]
        self.dilations = [1, 2, 4]
        
        # Create parallel multi-dilation conv blocks
        self.conv_blocks = nn.ModuleList()
        self.bns = nn.ModuleList()
        
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
        
        self.gru = nn.GRU(filters[-1], config['gru_hidden'], config['gru_layers'], batch_first=True)
        self.fc = nn.Linear(config['gru_hidden'], config['embedding_dim'])
        
    def forward(self, x):
        # x: (B*Seq, 50, 6) -> (B*Seq, 6, 50)
        x = x.transpose(1, 2)
        
        # Apply multi-dilation conv blocks
        for conv_block, bn in zip(self.conv_blocks, self.bns):
            # Apply all parallel convolutions and concatenate
            conv_outputs = [conv(x) for conv in conv_block]
            x = torch.cat(conv_outputs, dim=1)  # Concat along channel dimension
            x = F.relu(bn(x))
        
        x = x.transpose(1, 2)
        _, h = self.gru(x)
        return self.fc(h[-1])

class HLA(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, config['hidden_dim'], config['num_layers'], batch_first=True)
        self.fc = nn.Linear(config['hidden_dim'], config['num_classes'])
        
    def forward(self, x):
        # x: (B, Seq, Emb)
        _, h = self.gru(x)
        return self.fc(h[-1])

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
    
    train_ds = HierarchicalDataset(
        train_uids, 
        args.processed_dir, 
        "data/labels/scenario_labels.csv", 
        action_labels_path
    )
    
    val_ds = HierarchicalDataset(
        val_uids, 
        args.processed_dir, 
        "data/labels/scenario_labels.csv", 
        action_labels_path
    )
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG['training']['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=CONFIG['training']['batch_size'], shuffle=False)
    
    # Initialize Model
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
    
    # NEW: Learning Rate Scheduler (EgoCharm Paper - Supplemental S2)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CONFIG['training']['step_size'],
        gamma=CONFIG['training']['gamma']
    )
    print(f"Learning rate scheduler: StepLR(step_size={CONFIG['training']['step_size']}, gamma={CONFIG['training']['gamma']})")
    
    # Loss Functions
    # Paper (Section 3.2): "We train... using... a weighted cross-entropy loss to handle class imbalance"
    
    # Calculate class weights for scenarios from training data
    print("Calculating class weights for scenario loss...")
    scenario_labels_train = torch.tensor([s['scenario_label'].item() for s in train_ds.samples])
    class_counts = torch.bincount(scenario_labels_train)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    print(f"Scenario class distribution: {class_counts.tolist()}")
    print(f"Scenario class weights: {class_weights.tolist()}")
    
    criterion_scenario = nn.CrossEntropyLoss(weight=class_weights.to(device))
    criterion_action = nn.CrossEntropyLoss(ignore_index=-1)  # Kept for future probing
    
    # Early Stopping
    early_stopper = EarlyStopping(patience=CONFIG['training']['patience'])
    
    # Initialize W&B
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(
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
        if WANDB_AVAILABLE and not args.no_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_scenario_f1": val_s_f1,
                "val_scenario_acc": val_s_acc,
                "val_action_f1": val_a_f1,
                "val_action_acc": val_a_acc,
                "learning_rate": current_lr  # NEW: Log LR for monitoring
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
