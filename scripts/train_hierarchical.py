import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import os

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
        'num_classes': 8, # 8 Scenarios (Fitness removed)
        'seq_len': 30 # 30 seconds context
    },
    'training': {
        'batch_size': 128,
        'lr': 1e-3,
        'epochs': 20,
        'alpha': 1.0, # Weight for Scenario Loss
        'beta': 1.0   # Weight for Action Loss
    }
}

# --- Dataset ---
class HierarchicalDataset(torch.utils.data.Dataset):
    def __init__(self, take_uids, processed_dir, scenario_labels_path, action_labels_path):
        self.samples = []
        
        # Load Labels
        print("Loading label files...")
        self.scenario_df = pd.read_csv(scenario_labels_path).set_index('video_uid')
        self.action_df = pd.read_csv(action_labels_path)
        
        # Map Scenario Names to Integers
        self.scenario_map = {name: i for i, name in enumerate(self.scenario_df['scenario'].unique())}
        self.num_scenarios = len(self.scenario_map)
        print(f"Scenarios: {self.scenario_map}")
        
        # Map Action Names to Integers
        self.action_map = {'Stationary': 0, 'Locomotion': 1, 'Manual Work': 2, 'Scanning': 3}
        
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
                    continue # Skip if scenario was filtered out (e.g. Fitness)
                scenario_label = self.scenario_map[scenario_name]
                
                # Get Action Labels for this video
                video_actions = self.action_df[self.action_df['video_uid'] == uid]
                
                # Create Windows
                # We need sequences of 30 windows for HLA
                # Stride of 10 windows (overlap)
                seq_len = CONFIG['hla']['seq_len']
                stride = 10
                
                num_seqs = (len(traj) - seq_len) // stride + 1
                
                for i in range(num_seqs):
                    start_idx = i * stride
                    end_idx = start_idx + seq_len
                    
                    # Window Sequence
                    window_seq = traj[start_idx:end_idx] # (30, 50, 6)
                    
                    # Timestamp Sequence (Center of each window)
                    # timestamps is (N, 50), we take mean of each window
                    ts_seq = timestamps[start_idx:end_idx].mean(axis=1) # (30,)
                    
                    # Align Action Labels
                    # For each window in sequence, find if there is a matching action label
                    action_labels_seq = []
                    
                    for ts in ts_seq:
                        # Find action with closest timestamp (within 0.5s tolerance)
                        # This is slow, can be optimized, but fine for now
                        match = video_actions[
                            (video_actions['timestamp_sec'] >= ts - 0.5) & 
                            (video_actions['timestamp_sec'] <= ts + 0.5)
                        ]
                        
                        if not match.empty:
                            # Take first match
                            act_name = match.iloc[0]['action']
                            if act_name in self.action_map:
                                action_labels_seq.append(self.action_map[act_name])
                            else:
                                action_labels_seq.append(-1) # Unknown action class
                        else:
                            action_labels_seq.append(-1) # No label
                            
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
        self.action_head = nn.Linear(config['lle']['embedding_dim'], 4) # 4 Actions
        
    def forward(self, x):
        # x: (B, Seq, 50, 6)
        b, s, w, c = x.shape
        
        # Flatten for LLE
        x_flat = x.view(b*s, w, c)
        
        # LLE Forward
        embeddings = self.lle(x_flat) # (B*S, Emb)
        
        # Action Logits (for Probing/Auxiliary Loss)
        action_logits = self.action_head(embeddings) # (B*S, 4)
        action_logits = action_logits.view(b, s, 4)
        
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
    
    # Dataset
    dataset = HierarchicalDataset(
        uids, 
        args.processed_dir, 
        "data/labels/scenario_labels.csv", 
        "data/labels/master_annotations.csv"
    )
    
    if len(dataset) == 0:
        print("No data loaded. Check processed_dir and labels.")
        return
        
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG['training']['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=CONFIG['training']['batch_size'], shuffle=False)
    
    # Model
    model = HierarchicalModel(CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['training']['lr'])
    
    # Loss
    criterion_scenario = nn.CrossEntropyLoss()
    criterion_action = nn.CrossEntropyLoss(ignore_index=-1) # Masked Loss!
    
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
            # a_logits: (B, S, 4) -> (B*S, 4)
            # action_labels: (B, S) -> (B*S)
            loss_a = criterion_action(a_logits.view(-1, 4), action_labels.view(-1))
            
            # Combined Loss
            loss = CONFIG['training']['alpha'] * loss_s + CONFIG['training']['beta'] * loss_a
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")
        
        # Validation (Simple Accuracy)
        model.eval()
        correct_s = 0
        total_s = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['inputs'].to(device)
                labels = batch['scenario_label'].to(device)
                s_logits, _ = model(inputs)
                preds = torch.argmax(s_logits, dim=1)
                correct_s += (preds == labels).sum().item()
                total_s += labels.size(0)
        
        print(f"Val Scenario Acc: {correct_s / total_s:.4f}")
        
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), Path(args.output_dir) / "hierarchical_model.pth")
    print("Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-uids-file", type=str, default="target_uids.csv")
    parser.add_argument("--processed-dir", type=str, default="data/processed_ego4d")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()
    train(args)
