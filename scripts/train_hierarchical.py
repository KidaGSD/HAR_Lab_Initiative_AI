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
        'batch_size': 128,
        'lr': 1e-3,
        'epochs': 50, # Increased for early stopping
        'patience': 10, # Early stopping patience
        'alpha': 1.0, # Weight for Scenario Loss
        'beta': 1.0   # Weight for Action Loss
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
        
        # Iterate Videos
        for uid in tqdm(take_uids, desc='Loading Data'):
            seq_path = Path(processed_dir) / uid / 'seq.npz'
            if not seq_path.exists():
                continue
                
            try:
                data = np.load(seq_path)
                traj = data['traj'] # (N, 50, 6)
                timestamps = data['timestamp'] # (N, 50)
                
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
    def __init__(self, config):
        super().__init__()
        filters = config['cnn_filters']
        in_ch = config['in_channels']
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch if i==0 else filters[i-1], filters[i], 3, padding='same')
            for i in range(len(filters))
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(f) for f in filters])
        self.gru = nn.GRU(filters[-1], config['gru_hidden'], config['gru_layers'], batch_first=True)
        self.fc = nn.Linear(config['gru_hidden'], config['embedding_dim'])
        
    def forward(self, x):
        # x: (B*Seq, 50, 6) -> (B*Seq, 6, 50)
        x = x.transpose(1, 2)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
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
    
    # Load UIDs
    df = pd.read_csv(args.target_uids_file)
    uids = df['video_uid'].tolist()
    
    scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
    
    # Filter out 'test'
    train_uids = scenario_df[scenario_df['split'] == 'train']['video_uid'].tolist()
    val_uids = scenario_df[scenario_df['split'] == 'val']['video_uid'].tolist()
    
    # Intersect with available UIDs
    available_uids = set(uids)
    train_uids = [u for u in train_uids if u in available_uids]
    val_uids = [u for u in val_uids if u in available_uids]
    
    print(f"Split: Train={len(train_uids)}, Val={len(val_uids)}")
    
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
    
    # Model
    model = HierarchicalModel(CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['training']['lr'])
    
    # Loss
    criterion_scenario = nn.CrossEntropyLoss()
    criterion_action = nn.CrossEntropyLoss(ignore_index=-1) # Masked Loss
    
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
    
    for epoch in range(CONFIG['training']['epochs']):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs = batch['inputs'].to(device)
            scenario_labels = batch['scenario_label'].to(device)
            action_labels = batch['action_labels'].to(device) # (B, S)
            
            optimizer.zero_grad()
            
            s_logits, a_logits = model(inputs)
            
            # Scenario Loss
            loss_s = criterion_scenario(s_logits, scenario_labels)
            
            # Action Loss (Flatten)
            loss_a = criterion_action(a_logits.view(-1, 6), action_labels.view(-1))
            
            # Combined Loss
            loss = CONFIG['training']['alpha'] * loss_s + CONFIG['training']['beta'] * loss_a
            
            loss.backward()
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
    parser.add_argument("--run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()
    train(args)
