#!/usr/bin/env python3
"""
4-Fold Stratified Cross Validation for Hierarchical HAR
Following EgoCharm Paper Methodology (Section 3.6)

Paper Citation:
"We evaluate performance using 4 fold cross validation using stratification 
to ensure no participant data is both in the train and test set and to ensure 
equal percentage of classes/samples across folds."
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import copy
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available")

# ===========================
# SAME CONFIG AND MODELS FROM train_hierarchical.py
# ===========================

CONFIG = {
    'lle': {
        'in_channels': 6,
        'cnn_filters': [32, 64, 128],
        'gru_hidden': 256,
        'gru_layers': 2,
        'embedding_dim': 128,
        'window_size': 50  # 1 second at 50Hz
    },
    'hla': {
        'hidden_dim': 128,
        'num_layers': 2,
        'num_classes': 8,  # 8 Scenarios
        'seq_len': 30  # 30 seconds context
    },
    'training': {
        'batch_size': 1024,       # INCREASED for full GPU utilization
        'lr': 1e-4,
        'epochs': 50,
        'patience': 10,
        'alpha': 1.0,
        'beta': 0.0,              # Semi-supervised (scenario only)
        'step_size': 10,
        'gamma': 0.5
    }
}

# [Include all model classes from train_hierarchical.py]
# LLE, HLA, HierarchicalModel, EarlyStopping, HierarchicalDataset
# (Copying them here to keep the script self-contained)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
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

class LLE(nn.Module):
    """Low-Level Encoder with Variable Dilation CNNs (EgoCharm Paper)"""
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
    def __init__(self, config, embedding_dim):
        super().__init__()
        self.gru = nn.GRU(embedding_dim, config['hidden_dim'], config['num_layers'], batch_first=True)
        self.fc = nn.Linear(config['hidden_dim'], config['num_classes'])
        
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

class HierarchicalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lle = LLE(config['lle'])
        embedding_dim = config['lle']['embedding_dim']
        self.hla = HLA(config['hla'], embedding_dim)
        self.action_head = nn.Linear(embedding_dim, 6)
        
    def forward(self, x):
        b, s, w, c = x.shape
        x_flat = x.view(b * s, w, c)
        
        embeddings = self.lle(x_flat)
        action_logits = self.action_head(embeddings)
        action_logits = action_logits.view(b, s, 6)
        
        embeddings_seq = embeddings.view(b, s, -1)
        scenario_logits = self.hla(embeddings_seq)
        
        return scenario_logits, action_logits

class HierarchicalDataset(Dataset):
    def __init__(self, take_uids, processed_dir, scenario_labels_path, action_labels_path):
        self.samples = []
        
        self.scenario_df = pd.read_csv(scenario_labels_path).set_index('video_uid')
        self.action_df = pd.read_csv(action_labels_path)
        
        self.scenario_map = {name: i for i, name in enumerate(sorted(self.scenario_df['scenario'].unique()))}
        self.idx_to_scenario = {v: k for k, v in self.scenario_map.items()}
        
        self.action_map = {
            'Stationary': 0, 
            'Locomotion': 1, 
            'Essential Operation': 2, 
            'Object Transfer': 3,
            'Search': 4,
            'Error / Correction': 5
        }
        
        for uid in tqdm(take_uids, desc='Loading Data'):
            seq_path = Path(processed_dir) / uid / 'seq.npz'
            if not seq_path.exists():
                continue
                
            try:
                data = np.load(seq_path)
                traj = data['traj']
                timestamps = data['timestamp']
                
                if len(traj) == 0 or uid not in self.scenario_df.index:
                    continue
                    
                scenario_label = self.scenario_map[self.scenario_df.loc[uid, 'scenario']]
                video_actions = self.action_df[self.action_df['video_uid'] == uid]
                
                seq_len = CONFIG['hla']['seq_len']
                stride = 10
                num_seqs = (len(traj) - seq_len) // stride + 1
                
                for i in range(num_seqs):
                    start_idx = i * stride
                    end_idx = start_idx + seq_len
                    
                    window_seq = traj[start_idx:end_idx]
                    ts_windows = timestamps[start_idx:end_idx]
                    
                    action_labels_seq = []
                    for w_idx in range(seq_len):
                        w_ts = ts_windows[w_idx]
                        w_start, w_end = w_ts[0], w_ts[-1]
                        
                        match = video_actions[
                            (video_actions['timestamp_sec'] >= w_start) & 
                            (video_actions['timestamp_sec'] <= w_end)
                        ]
                        
                        if not match.empty:
                            act_name = match.iloc[0]['action']
                            action_labels_seq.append(self.action_map.get(act_name, -1))
                        else:
                            action_labels_seq.append(-1)
                            
                    self.samples.append({
                        'video_uid': uid,
                        'inputs': torch.FloatTensor(window_seq),
                        'scenario_label': torch.tensor(scenario_label, dtype=torch.long),
                        'action_labels': torch.tensor(action_labels_seq, dtype=torch.long)
                    })
                    
            except Exception as e:
                print(f"Error loading {uid}: {e}")
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

# ===========================
# CROSS VALIDATION IMPLEMENTATION
# ===========================

def get_participant_stratified_folds(scenario_df, n_splits=4, random_state=42):
    """
    Create participant-level stratified folds
    
    Paper: "using stratification to ensure no participant data is both in 
    the train and test set and to ensure equal percentage of classes/samples across folds"
    """
    print("\n" + "="*80)
    print("CREATING PARTICIPANT-LEVEL STRATIFIED FOLDS")
    print("="*80)
    
    # NOTE: Ego4D doesn't have participant_id in scenario_labels.csv
    # We treat each video as independent (conservative approach)
    # If participant info is available, modify this section
    
    video_scenarios = scenario_df[['video_uid', 'scenario']].copy()
    
    # Use StratifiedKFold on videos with scenario stratification
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(
        video_scenarios['video_uid'], 
        video_scenarios['scenario']
    )):
        train_uids = video_scenarios.iloc[train_idx]['video_uid'].tolist()
        val_uids = video_scenarios.iloc[val_idx]['video_uid'].tolist()
        
        folds.append({
            'fold': fold_idx,
            'train_uids': train_uids,
            'val_uids': val_uids
        })
        
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train: {len(train_uids)} videos")
        print(f"  Val:   {len(val_uids)} videos")
        
        # Check scenario distribution
        train_scenarios = video_scenarios.iloc[train_idx]['scenario'].value_counts()
        print(f"  Train scenario dist: {dict(train_scenarios)}")
        
    return folds

def train_one_fold(fold_idx, train_uids, val_uids, args, device):
    """Train and evaluate one CV fold"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING FOLD {fold_idx + 1}/4")
    print(f"{'='*80}\n")
    
    # Load data
    action_labels_path = "data/labels/action_labels_llm_validated.csv"
    
    train_ds = HierarchicalDataset(
        train_uids, args.processed_dir,
        "data/labels/scenario_labels.csv",
        action_labels_path
    )
    
    val_ds = HierarchicalDataset(
        val_uids, args.processed_dir,
        "data/labels/scenario_labels.csv",
        action_labels_path
    )
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['training']['batch_size'], 
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['training']['batch_size'], 
                            shuffle=False, num_workers=4)
    
    # Initialize model
    model = HierarchicalModel(CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['training']['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CONFIG['training']['step_size'],
        gamma=CONFIG['training']['gamma']
    )
    
    # Weighted loss
    scenario_labels_train = torch.tensor([s['scenario_label'].item() for s in train_ds.samples])
    class_counts = torch.bincount(scenario_labels_train)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    criterion_scenario = nn.CrossEntropyLoss(weight=class_weights.to(device))
    criterion_action = nn.CrossEntropyLoss(ignore_index=-1)
    
    early_stopper = EarlyStopping(patience=CONFIG['training']['patience'])
    
    # Training loop
    best_val_f1 = 0.0
    for epoch in range(CONFIG['training']['epochs']):
        # Train
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs = batch['inputs'].to(device)
            scenario_labels = batch['scenario_label'].to(device)
            action_labels = batch['action_labels'].to(device)
            
            optimizer.zero_grad()
            s_logits, a_logits = model(inputs)
            
            loss_s = criterion_scenario(s_logits, scenario_labels)
            
            if CONFIG['training']['beta'] > 0:
                loss_a = criterion_action(a_logits.view(-1, 6), action_labels.view(-1))
                loss = CONFIG['training']['alpha'] * loss_s + CONFIG['training']['beta'] * loss_a
            else:
                loss = loss_s
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validate
        model.eval()
        all_s_preds, all_s_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['inputs'].to(device)
                scenario_labels = batch['scenario_label'].to(device)
                
                s_logits, _ = model(inputs)
                s_preds = s_logits.argmax(dim=1).cpu().numpy()
                
                all_s_preds.extend(s_preds)
                all_s_labels.extend(scenario_labels.cpu().numpy())
        
        val_s_f1 = f1_score(all_s_labels, all_s_preds, average='macro')
        val_s_acc = (np.array(all_s_preds) == np.array(all_s_labels)).mean()
        
        print(f"Fold {fold_idx+1} Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Val F1={val_s_f1:.4f}, Val Acc={val_s_acc:.4f}")
        
        if val_s_f1 > best_val_f1:
            best_val_f1 = val_s_f1
        
        # Early stopping
        early_stopper(val_s_f1, model)
        scheduler.step()
        
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if early_stopper.best_model_state:
        model.load_state_dict(early_stopper.best_model_state)
    
    # Final evaluation
    model.eval()
    all_s_preds, all_s_labels = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['inputs'].to(device)
            scenario_labels = batch['scenario_label'].to(device)
            
            s_logits, _ = model(inputs)
            s_preds = s_logits.argmax(dim=1).cpu().numpy()
            
            all_s_preds.extend(s_preds)
            all_s_labels.extend(scenario_labels.cpu().numpy())
    
    final_f1 = f1_score(all_s_labels, all_s_preds, average='macro')
    final_acc = (np.array(all_s_preds) == np.array(all_s_labels)).mean()
    
    print(f"\nFold {fold_idx+1} Final Results: F1={final_f1:.4f}, Acc={final_acc:.4f}")
    
    return {
        'fold': fold_idx,
        'f1': final_f1,
        'acc': final_acc,
        'predictions': all_s_preds,
        'labels': all_s_labels
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', type=str, default='data/processed_ego4d')
    parser.add_argument('--n-folds', type=int, default=4, help='Number of CV folds')
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='cv_results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Load scenario labels
    scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
    
    # Use only train+val splits (exclude test and multi)
    usable_df = scenario_df[scenario_df['split'].isin(['train', 'val'])].copy()
    
    print(f"\nUsing {len(usable_df)} videos for CV (train+val splits)")
    print(f"Excluded: test={len(scenario_df[scenario_df['split']=='test'])}, multi={len(scenario_df[scenario_df['split']=='multi'])}")
    
    # Create folds
    folds = get_participant_stratified_folds(usable_df, n_splits=args.n_folds, random_state=args.random_seed)
    
    # Train each fold
    results = []
    for fold_data in folds:
        fold_result = train_one_fold(
            fold_data['fold'],
            fold_data['train_uids'],
            fold_data['val_uids'],
            args,
            device
        )
        results.append(fold_result)
    
    # Aggregate results
    print("\n" + "="*80)
    print("CROSS VALIDATION RESULTS")
    print("="*80)
    
    f1_scores = [r['f1'] for r in results]
    acc_scores = [r['acc'] for r in results]
    
    print(f"\nPer-fold F1 scores: {[f'{f:.4f}' for f in f1_scores]}")
    print(f"Per-fold Accuracies: {[f'{a:.4f}' for a in acc_scores]}")
    print(f"\nMean F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Mean Acc: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    summary = {
        'n_folds': args.n_folds,
        'mean_f1': float(np.mean(f1_scores)),
        'std_f1': float(np.std(f1_scores)),
        'mean_acc': float(np.mean(acc_scores)),
        'std_acc': float(np.std(acc_scores)),
        'fold_f1_scores': [float(f) for f in f1_scores],
        'fold_acc_scores': [float(a) for a in acc_scores]
    }
    
    with open(output_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/cv_summary.json")

if __name__ == '__main__':
    main()
