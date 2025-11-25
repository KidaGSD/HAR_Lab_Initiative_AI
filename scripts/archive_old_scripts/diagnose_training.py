#!/usr/bin/env python3
"""Diagnostic script to check training setup"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('scripts')

# Simplified dataset class for testing
class Ego4DDataset(torch.utils.data.Dataset):
    def __init__(self, take_uids, processed_dir):
        self.samples = []
        for uid in tqdm(take_uids, desc='Loading'):
            seq_path = Path(processed_dir) / uid / 'seq.npz'
            if not seq_path.exists():
                continue
            try:
                data = np.load(seq_path)
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

# Load UIDs
df = pd.read_csv('target_uids.csv')
uids = df['video_uid'].tolist()

# Split
np.random.seed(42)  # Fixed seed for reproducibility
np.random.shuffle(uids)
n = len(uids)
train_uids = uids[:int(0.8*n)]
val_uids = uids[int(0.8*n):]

print(f"Total UIDs: {len(uids)}")
print(f"Train UIDs: {len(train_uids)}")
print(f"Val UIDs: {len(val_uids)}")
print(f"\nTrain UIDs sample: {train_uids[:3]}")
print(f"Val UIDs sample: {val_uids[:3]}")

# Check if there's overlap
overlap = set(train_uids) & set(val_uids)
print(f"\nOverlap between train and val: {len(overlap)} UIDs")
if overlap:
    print(f"WARNING: Train and Val sets overlap! {overlap}")

# Load datasets
print("\nLoading datasets...")
train_ds = Ego4DDataset(train_uids, 'data/processed_ego4d')
val_ds = Ego4DDataset(val_uids, 'data/processed_ego4d')

print(f"\nTrain dataset: {len(train_ds)} windows")
print(f"Val dataset: {len(val_ds)} windows")

# Check if samples overlap
print("\nChecking sample UIDs...")
train_sample_uids = set([s['take_uid'] for s in train_ds.samples[:100]])
val_sample_uids = set([s['take_uid'] for s in val_ds.samples[:100]])
sample_overlap = train_sample_uids & val_sample_uids
print(f"Sample overlap: {len(sample_overlap)} UIDs")
if sample_overlap:
    print(f"WARNING: Sample UIDs overlap! {sample_overlap}")

# Create dataloaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

print(f"\nTrain batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Check data shapes
print("\nChecking data shapes...")
train_batch = next(iter(train_loader))
val_batch = next(iter(val_loader))

print(f"Train batch traj shape: {train_batch['traj'].shape}")
print(f"Val batch traj shape: {val_batch['traj'].shape}")

# Check if data is identical
print("\nChecking if data is different...")
train_mean = train_batch['traj'].mean().item()
val_mean = val_batch['traj'].mean().item()
print(f"Train batch mean: {train_mean:.6f}")
print(f"Val batch mean: {val_mean:.6f}")

# Check statistics across full dataset
print("\nCalculating dataset statistics...")
train_means = []
for batch in tqdm(train_loader, desc="Train stats"):
    train_means.append(batch['traj'].mean().item())

val_means = []
for batch in tqdm(val_loader, desc="Val stats"):
    val_means.append(batch['traj'].mean().item())

print(f"\nTrain data mean: {np.mean(train_means):.6f} ± {np.std(train_means):.6f}")
print(f"Val data mean: {np.mean(val_means):.6f} ± {np.std(val_means):.6f}")

print("\n✓ Diagnostic complete!")
