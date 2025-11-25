import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import json

# --- Configuration ---
CONFIG = {
    'ssl': {
        'embedding_dim': 128,
        'cnn_filters': [32, 64, 128],
        'kernel_size': 3,
        'gru_hidden': 256,
        'gru_layers': 2,
        'batch_size': 32,
        'lr': 1e-3,
        'epochs': 20,
        'temperature': 0.1
    }
}

# --- Dataset ---
class Ego4DDataset(torch.utils.data.Dataset):
    def __init__(self, take_uids, processed_dir):
        self.samples = []
        for uid in tqdm(take_uids, desc='Loading'):
            seq_path = Path(processed_dir) / uid / 'seq.npz'
            if not seq_path.exists():
                continue
            try:
                data = np.load(seq_path)
                # data['traj'] shape: (N, 150, 6)
                
                if 'traj' not in data or len(data['traj']) == 0:
                    continue
                    
                for i in range(len(data['traj'])):
                    self.samples.append({
                        'take_uid': uid,
                        'window_idx': i,
                        'traj': torch.FloatTensor(data['traj'][i]),
                    })
            except Exception as e:
                print(f"Error loading {uid}: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# --- Model ---
class Encoder(nn.Module):
    def __init__(self, in_ch, config):
        super().__init__()
        filters = config['ssl']['cnn_filters']
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch if i==0 else filters[i-1], filters[i], 3, padding='same')
            for i in range(len(filters))
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(f) for f in filters])
        self.gru = nn.GRU(filters[-1], config['ssl']['gru_hidden'], config['ssl']['gru_layers'], batch_first=True)
        self.fc = nn.Linear(config['ssl']['gru_hidden'], config['ssl']['embedding_dim'])
    
    def forward(self, x):
        # x: (B, L, C) -> (B, C, L) for Conv1d
        x = x.transpose(1, 2)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
        x = x.transpose(1, 2)
        _, h = self.gru(x)
        return self.fc(h[-1])

class SSLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # IMU has 6 channels (accel+gyro)
        self.imu_enc = Encoder(6, config) 
        
        # Projection head for SimCLR
        self.projector = nn.Sequential(
            nn.Linear(config['ssl']['embedding_dim'], config['ssl']['embedding_dim']),
            nn.ReLU(),
            nn.Linear(config['ssl']['embedding_dim'], config['ssl']['embedding_dim'])
        )
    
    def encode(self, batch):
        return self.imu_enc(batch['traj'])
    
    def forward(self, batch):
        emb = self.encode(batch)
        proj = self.projector(emb)
        return emb, proj

# --- Loss ---
def nt_xent_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    
    # Mask out self-similarity
    mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)
    
    # Positive pairs: (i, i+N) and (i+N, i)
    sim_i_j = torch.diag(sim, N)
    sim_j_i = torch.diag(sim, -N)
    
    positives = torch.cat([sim_i_j, sim_j_i], dim=0)
    numerator = torch.exp(positives)
    denominator = torch.sum(torch.exp(sim), dim=1)
    
    loss = -torch.log(numerator / denominator)
    return torch.mean(loss)

# --- Training Loop ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Visualization ---
def plot_history(history, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SSL Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(output_dir) / 'loss_curve.png')
    plt.close()

def plot_svdd_scores(scores, output_dir):
    plt.figure(figsize=(10, 5))
    sns.histplot(scores, bins=50, kde=True)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title('SVDD Score Distribution (Validation)')
    plt.axvline(x=1.0, color='r', linestyle='--', label='Threshold (R)')
    plt.legend()
    plt.savefig(Path(output_dir) / 'svdd_scores.png')
    plt.close()

# --- Training Loop ---
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load UIDs
    df = pd.read_csv(args.target_uids_file)
    uids = df['video_uid'].tolist()
    
    # Split
    np.random.shuffle(uids)
    n = len(uids)
    train_uids = uids[:int(0.8*n)]
    val_uids = uids[int(0.8*n):]
    
    print(f"Train: {len(train_uids)}, Val: {len(val_uids)}")
    
    train_ds = Ego4DDataset(train_uids, args.processed_dir)
    val_ds = Ego4DDataset(val_uids, args.processed_dir)
    
    if len(train_ds) == 0:
        print("No training data found. Run pipeline first.")
        return
        
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG['ssl']['batch_size'], shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=CONFIG['ssl']['batch_size'], shuffle=False, drop_last=True)
    
    model = SSLModel(CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['ssl']['lr'])
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print("\n" + "="*40)
    print(f"{'Epoch':^5} | {'Train Loss':^12} | {'Val Loss':^12} | {'Best':^5}")
    print("="*40)
    
    for epoch in range(CONFIG['ssl']['epochs']):
        model.train()
        train_loss = 0
        
        # Use tqdm for progress within epoch
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # SimCLR augmentation: 
            # In standard SimCLR, we apply augmentations to create two views.
            # Here, for time-series, we can use:
            # 1. Jittering / Scaling
            # 2. Time warping (harder)
            # 3. Or just use the same window as two views with dropout? (Weak)
            # Better: Crop two sub-windows from the same larger window?
            # But our windows are already small (3s).
            
            # For now, let's implement a simple augmentation: Add noise + Scale
            
            traj = batch['traj'].to(device) # (B, 150, 6)
            
            # Augmentation 1
            noise1 = torch.randn_like(traj) * 0.05
            traj1 = traj + noise1
            
            # Augmentation 2
            scale = 1.0 + (torch.rand(traj.shape[0], 1, 1, device=device) - 0.5) * 0.2 # +/- 10% scaling
            traj2 = traj * scale
            
            batch1 = {'traj': traj1}
            batch2 = {'traj': traj2}
            
            _, proj1 = model(batch1)
            _, proj2 = model(batch2)
            
            loss = nt_xent_loss(proj1, proj2, temperature=CONFIG['ssl']['temperature'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                traj = batch['traj'].to(device)
                
                # For validation, we can compare original vs slightly noisy
                batch1 = {'traj': traj}
                batch2 = {'traj': traj + torch.randn_like(traj)*0.01}
                
                _, proj1 = model(batch1)
                _, proj2 = model(batch2)
                
                loss = nt_xent_loss(proj1, proj2, temperature=CONFIG['ssl']['temperature'])
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Print epoch results (like notebook)
        print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Path(args.output_dir) / "best_model.pth")
            print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
            
        # Plot every epoch
        plot_history(history, args.output_dir)

    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f}")

    # --- SVDD Training ---
    print("\nStarting SVDD training...")
    # Load best model
    model.load_state_dict(torch.load(Path(args.output_dir) / "best_model.pth"))
    model.eval()
    
    # Extract embeddings for all train data
    train_embs = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Extracting Train Embs"):
            traj = batch['traj'].to(device)
            batch_input = {'traj': traj}
            emb, _ = model(batch_input)
            train_embs.append(emb.cpu().numpy())
    train_embs = np.concatenate(train_embs)
    
    # Fit SVDD
    svdd = DeepSVDD(nu=0.1) # Default nu
    svdd.fit(train_embs)
    
    # Evaluate on Val
    val_embs = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Extracting Val Embs"):
            traj = batch['traj'].to(device)
            batch_input = {'traj': traj}
            emb, _ = model(batch_input)
            val_embs.append(emb.cpu().numpy())
    val_embs = np.concatenate(val_embs)
    
    val_scores = svdd.score(val_embs)
    print(f"Val Anomaly Scores: Mean={val_scores.mean():.4f}, Max={val_scores.max():.4f}")
    
    # Plot SVDD scores
    plot_svdd_scores(val_scores, args.output_dir)
    print(f"Saved SVDD score plot to {Path(args.output_dir) / 'svdd_scores.png'}")
    
    # Save SVDD model (center and R)
    svdd_state = {'center': svdd.center, 'R': svdd.R}
    torch.save(svdd_state, Path(args.output_dir) / "svdd_model.pth")
    print("Saved SVDD model.")

class DeepSVDD:
    def __init__(self, nu=0.1):
        self.nu = nu
        self.center = None
        self.R = None
    
    def fit(self, embs):
        # embs: (N, dim) numpy array
        self.center = torch.FloatTensor(embs.mean(0))
        dists = torch.sum((torch.FloatTensor(embs) - self.center)**2, 1)
        self.R = torch.quantile(dists, 1-self.nu).item()
    
    def score(self, embs):
        dists = torch.sum((torch.FloatTensor(embs) - self.center)**2, 1).numpy()
        # Score > 1 means anomaly (distance > R)
        # We can return raw distance or normalized
        # Original notebook clipped at 1, but that loses info for strong anomalies.
        # Let's return ratio dist/R
        return dists / self.R
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-uids-file", type=str, default="target_uids.csv")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="models/ego4d_ssl")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)

if __name__ == "__main__":
    main()
