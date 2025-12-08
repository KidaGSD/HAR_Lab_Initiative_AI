#!/usr/bin/env python3
"""
Feature Space Visualization Script (PCA/t-SNE)
Extracts and visualizes high-level (Scenario) and low-level (Action) embeddings.
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.models.hierarchical import HierarchicalModel
from src.data.hierarchical_dataset import HierarchicalDataset

def extract_embeddings(model, loader, device):
    """Extract embeddings using forward hooks."""
    print("Extracting features...")
    
    scenario_feats = []
    action_feats = []
    scenario_labels = []
    action_labels = []
    
    # Register hooks to capture embeddings before classification heads
    def hla_hook(module, input, output):
        # Capture HLA output (high-level embeddings)
        scenario_feats.append(output.detach().cpu().numpy())
        
    def lle_hook(module, input, output):
        # Capture LLE output (low-level embeddings) 
        # output shape: (B, Seq, EmbeddingDim)
        batch_size, seq_len, emb_dim = output.shape
        # Flatten to (B*Seq, EmbeddingDim)
        action_feats.append(output.reshape(-1, emb_dim).detach().cpu().numpy())

    # Register hooks at the embedding layers (before heads)
    handle_hla = model.hla.register_forward_hook(hla_hook)
    handle_lle = model.lle.register_forward_hook(lle_hook)
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            sequences = batch['sequence'].to(device)  # Fixed: was 'inputs'
            s_label = batch['scenario_label']  # (B,)
            a_label = batch['action_label']    # Fixed: was 'action_labels'
            
            # Forward pass triggers hooks
            model(sequences)
            
            # Store labels
            scenario_labels.append(s_label.numpy())
            # Action labels need to be expanded to match flattened embeddings
            # a_label is (B,) but we have (B*Seq,) embeddings
            # For now, repeat each action label seq_len times
            batch_size = len(s_label)
            seq_len = sequences.shape[1]
            action_labels.append(np.repeat(a_label.numpy(), seq_len))
            
    # Cleanup hooks
    handle_hla.remove()
    handle_lle.remove()
    
    # Concatenate all
    S_feats = np.concatenate(scenario_feats, axis=0)
    A_feats = np.concatenate(action_feats, axis=0)
    S_labels = np.concatenate(scenario_labels, axis=0)
    A_labels = np.concatenate(action_labels, axis=0)
    
    return S_feats, S_labels, A_feats, A_labels

def plot_embedding(features, labels, label_map, title, save_path_base, method='pca'):
    """Generate publication-quality 2D scatter plot with centroids."""
    print(f"Generating {method.upper()} plot for {title}...")
    
    # Style settings for publication
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Subsample if too large (for t-SNE performance/visual clarity)
    if len(features) > 5000:
        print(f"Subsampling {len(features)} -> 5000 for visualization")
        idx = np.random.choice(len(features), 5000, replace=False)
        features = features[idx]
        labels = labels[idx]
    
    # Dimensionality Reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(features)
        explained_var = reducer.explained_variance_ratio_
        x_label = f"PC1 ({explained_var[0]:.1%} var)"
        y_label = f"PC2 ({explained_var[1]:.1%} var)"
    else:
        reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        embedding = reducer.fit_transform(features)
        x_label = "t-SNE Dim 1"
        y_label = "t-SNE Dim 2"
        
    # Create DataFrame
    df = pd.DataFrame(embedding, columns=['x', 'y'])
    idx_to_name = {v: k for k, v in label_map.items()}
    df['Label'] = [idx_to_name.get(l, 'Unknown') for l in labels]
    
    # Filter Unknowns
    df = df[df['Label'] != 'Unknown'].copy()
    
    # Sort labels alphabetically for consistent legend
    df.sort_values('Label', inplace=True)
    
    # Plotting
    plt.figure(figsize=(14, 10))
    
    # Main scatter plot
    n_classes = df['Label'].nunique()
    palette = sns.color_palette("husl", n_classes) if n_classes > 10 else "bright"
    
    ax = sns.scatterplot(
        data=df, x='x', y='y', hue='Label', style='Label',
        palette=palette, s=100, alpha=0.7, edgecolor='w', linewidth=0.5
    )
    
    # Calculate and plot centroids with text labels
    # This helps identify clusters even if colors are similar
    centroids = df.groupby('Label')[['x', 'y']].median()
    
    for label, coords in centroids.iterrows():
        plt.text(
            coords['x'], coords['y'], str(label),
            horizontalalignment='center',
            verticalalignment='center',
            size='medium',
            weight='bold',
            color='black',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3)
        )

    # Aesthetics
    plt.title(f"{title} Feature Space ({method.upper()})", fontsize=20, pad=20)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    
    # Legend outside
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Classes")
    
    plt.tight_layout()
    
    # Save both PNG and PDF (vector graphics for papers)
    plt.savefig(f"{save_path_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path_base}.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_base}.png & .pdf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to model config (e.g., configs/beta_0.5.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--output-dir", type=str, default="outputs/plots")
    parser.add_argument("--split", type=str, default="val", choices=['train', 'val', 'test'],
                       help="Which split to visualize")
    parser.add_argument("--use-tsne", action="store_true", help="Use t-SNE instead of PCA")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (e.g., 0, 1, 2). If not specified, uses CUDA_VISIBLE_DEVICES or auto-detects")
    args = parser.parse_args()
    
    # Manual GPU assignment if specified
    if args.gpu is not None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"🎯 Using manually specified GPU {args.gpu}")
    
    # Config
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📍 Device: {device}")
    
    # Load Data using current HierarchicalDataset structure
    print(f"Loading {args.split} data...")
    
    # Load split file to get UIDs
    split_file = f"data/splits/{args.split}_uids.txt"
    if not os.path.exists(split_file):
        print(f"⚠️  Split file not found: {split_file}")
        print("Looking for UIDs in scenario_labels.csv...")
        scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
        take_uids = scenario_df['video_uid'].tolist()
    else:
        with open(split_file, 'r') as f:
            take_uids = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(take_uids)} videos in {args.split} split")
    
    dataset = HierarchicalDataset(
        take_uids=take_uids,
        processed_dir="data/processed",
        scenario_labels_path="data/labels/scenario_labels.csv",
        action_labels_path="data/labels/action_labels_4class.csv",
        config=config,
        training=False  # No augmentation for visualization
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=4
    )
    
    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = HierarchicalModel(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Extract
    S_feats, S_labels, A_feats, A_labels = extract_embeddings(model, loader, device)
    
    # Create Output Dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot Method
    method = 'tsne' if args.use_tsne else 'pca'
    model_name = Path(args.checkpoint).parent.name
    
    # Plot Scenario Features (High Level)
    plot_embedding(
        S_feats, S_labels, dataset.scenario_map, 
        f"Scenario (High-Level) - {args.split.upper()}", 
        f"{args.output_dir}/{model_name}_scenario_{args.split}_{method}",
        method
    )
    
    # Plot Action Features (Low Level)
    plot_embedding(
        A_feats, A_labels, dataset.action_map, 
        f"Action (Low-Level) - {args.split.upper()}", 
        f"{args.output_dir}/{model_name}_action_{args.split}_{method}",
        method
    )
    
    print(f"\n✓ Visualization complete! Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
