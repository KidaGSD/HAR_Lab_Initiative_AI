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
    action_labels = [] # Flattened
    
    # Register hooks to capture inputs to classification heads
    # Input to hla.head is the High-Level Embedding (Video/Scenario level)
    # Input to action_head is the Low-Level Embedding (Window/Action level)
    
    def hla_hook(module, input, output):
        # input is a tuple (tensor,), shape (B, HiddenDim)
        scenario_feats.append(input[0].detach().cpu().numpy())
        
    def action_hook(module, input, output):
        # input is a tuple (tensor,), shape (B*Seq, EmbeddingDim)
        action_feats.append(input[0].detach().cpu().numpy())

    handle_hla = model.hla.head.register_forward_hook(hla_hook)
    handle_action = model.action_head.register_forward_hook(action_hook)
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = batch['inputs'].to(device)
            s_label = batch['scenario_label']  # (B,)
            a_label = batch['action_labels']   # (B, Seq)
            
            # Forward pass triggers hooks
            model(inputs)
            
            # Store labels
            scenario_labels.append(s_label.numpy())
            action_labels.append(a_label.view(-1).numpy())
            
    # Cleanup hooks
    handle_hla.remove()
    handle_action.remove()
    
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
    parser.add_argument("--config", type=str, default="configs/beta_0.3.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--output-dir", type=str, default="outputs/plots")
    parser.add_argument("--use-tsne", action="store_true", help="Use t-SNE instead of PCA")
    args = parser.parse_args()
    
    # Config
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data (Use ALL data as requested)
    print("Loading ALL data...")
    scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
    all_uids = scenario_df['video_uid'].tolist()
    
    dataset = HierarchicalDataset(
        all_uids,
        "data/processed_ego4d",
        "data/labels/scenario_labels.csv",
        "data/labels/action_labels_4class.csv",
        config,
        training=False  # No augmentation for visualization
    )
    
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'], 
                       shuffle=False, num_workers=4)
    
    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = HierarchicalModel(config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    
    # Extract
    S_feats, S_labels, A_feats, A_labels = extract_embeddings(model, loader, device)
    
    # Create Output Dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot Method
    method = 'tsne' if args.use_tsne else 'pca'
    
    # Plot Scenario Features (High Level)
    plot_embedding(
        S_feats, S_labels, dataset.scenario_map, 
        "Scenario (High-Level)", 
        f"{args.output_dir}/scenario_features_{method}",
        method
    )
    
    # Plot Action Features (Low Level)
    plot_embedding(
        A_feats, A_labels, dataset.action_map, 
        "Action (Low-Level)", 
        f"{args.output_dir}/action_features_{method}",
        method
    )

if __name__ == "__main__":
    main()
