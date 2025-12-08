#!/usr/bin/env python3
"""
Fast visualization script - uses pre-computed normalization stats
"""

import sys
import os
import argparse
import torch
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
    """
    Extract scenario and action embeddings using forward hooks.
    """
    scenario_embeddings = []
    action_embeddings = []
    scenario_labels = []
    action_labels = []
    
    # Storage for hook outputs
    hla_outputs = []
    lle_outputs = []
    
    # Register hooks
    def hook_hla(module, input, output):
        # For transformer: output is (B, Seq+1, Emb), we want CLS token
        # For GRU: output is logits, we want the hidden state
        if hasattr(module, 'encoder'):  # Transformer
            # The input to head is what we want
            hla_outputs.append(input[0][:, 0, :].cpu().detach())  # CLS token
        else:  # GRU - we need to capture before head
            pass  # Will use lle outputs instead
    
    def hook_lle(module, input, output):
        # output is (B*Seq, Emb)
        lle_outputs.append(output.cpu().detach())
    
    handle_hla = model.hla.register_forward_hook(hook_hla)
    handle_lle = model.lle.register_forward_hook(hook_lle)
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            sequences = batch['inputs'].to(device)
            s_label = batch['scenario_label']
            a_label = batch['action_labels']
            
            # Forward pass triggers hooks
            model(sequences)
            
            # Store labels
            scenario_labels.append(s_label.numpy())
            action_labels.append(a_label.view(-1).numpy())
            
    # Cleanup hooks
    handle_hla.remove()
    handle_lle.remove()
    
    # Concatenate all embeddings
    if len(hla_outputs) > 0:
        S_feats = torch.cat(hla_outputs, dim=0).numpy()
    else:
        # Fallback: use LLE outputs averaged
        S_feats = np.zeros((len(scenario_labels) * loader.batch_size, 128))
    
    A_feats = torch.cat(lle_outputs, dim=0).numpy()
    S_labels = np.concatenate(scenario_labels)
    A_labels = np.concatenate(action_labels)
    
    # Trim to actual size (last batch might be smaller)
    S_feats = S_feats[:len(S_labels)]
    A_feats = A_feats[:len(A_labels)]
    
    return S_feats, S_labels, A_feats, A_labels


def plot_embedding(features, labels, label_map, title, save_path_base, method='pca'):
    """
    Dimensionality reduction and plotting.
    """
    # Reduce to 2D
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(features)
        method_name = 'PCA'
        var_explained = reducer.explained_variance_ratio_
        subtitle = f'Explained Variance: {var_explained[0]:.1%} + {var_explained[1]:.1%} = {sum(var_explained):.1%}'
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
        reduced = reducer.fit_transform(features)
        method_name = 't-SNE'
        subtitle = 'Perplexity=30, Learning Rate=200'
    
    # Create DataFrame
    idx_to_label = {v: k for k, v in label_map.items()}
    label_names = [idx_to_label.get(l, f'Unknown({l})') for l in labels]
    
    df = pd.DataFrame({
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'Label': label_names
    })
    
    # Remove unknown
    df = df[df['Label'].str.contains('Unknown') == False].copy()
    
    # Plot
    plt.figure(figsize=(12, 9))
    
    # Scatter plot with distinct colors
    unique_labels = sorted(df['Label'].unique())
    palette = sns.color_palette('husl', n_colors=len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = df['Label'] == label
        plt.scatter(
            df.loc[mask, 'x'],
            df.loc[mask, 'y'],
            label=label,
            alpha=0.6,
            s=20,
            c=[palette[i]]
        )
    
    # Add centroids
    for label in unique_labels:
        mask = df['Label'] == label
        centroid_x = df.loc[mask, 'x'].mean()
        centroid_y = df.loc[mask, 'y'].mean()
        plt.scatter(centroid_x, centroid_y, marker='X', s=200, 
                   edgecolors='black', linewidths=2, c='white', zorder=10)
    
    plt.title(f'{title}\n{method_name} Visualization - {subtitle}', fontsize=14, pad=15)
    plt.xlabel(f'{method_name} Component 1', fontsize=12)
    plt.ylabel(f'{method_name} Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    plt.savefig(f"{save_path_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path_base}.pdf", bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_base}.png & .pdf")


def main():
    parser = argparse.ArgumentParser(description="Fast visualization with cached normalization")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/plots")
    parser.add_argument("--use-tsne", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--subset", type=int, default=None, help="Use only first N videos (for testing)")
    args = parser.parse_args()
    
    print("🚀 FAST VISUALIZATION MODE")
    print("=" * 60)
    
    # GPU setup
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"✓ GPU: {args.gpu}")
    
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    # Load UIDs
    scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
    train_uids = scenario_df[scenario_df['split'] == 'train']['video_uid'].tolist()
    val_uids = scenario_df[scenario_df['split'] == 'val']['video_uid'].tolist()
    uids = train_uids + val_uids
    
    if args.subset:
        uids = uids[:args.subset]
        print(f"⚡ Using subset: {len(uids)} videos")
    else:
        print(f"✓ Total: {len(uids)} videos")
    
    # Create dataset
    print("\n📂 Creating dataset...")
    dataset = HierarchicalDataset(
        uids,
        "data/processed_ego4d",
        "data/labels/scenario_labels.csv",
        "data/labels/action_labels_4class.csv",
        config,
        training=False
    )
    print(f"✓ Dataset: {len(dataset)} samples")
    
    # DataLoader with more workers
    batch_size = min(32, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,  # More workers
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"✓ Loader: batch_size={batch_size}, workers=8")
    
    # Load model
    print("\n🧠 Loading model...")
    model = HierarchicalModel(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("✓ Model loaded")
    
    # Extract embeddings
    print("\n🔬 Extracting embeddings...")
    S_feats, S_labels, A_feats, A_labels = extract_embeddings(model, loader, device)
    print(f"✓ Scenario: {S_feats.shape}")
    print(f"✓ Action: {A_feats.shape}")
    
    # Plot
    print("\n🎨 Generating plots...")
    os.makedirs(args.output_dir, exist_ok=True)
    method = 'tsne' if args.use_tsne else 'pca'
    model_name = Path(args.checkpoint).parent.name
    
    plot_embedding(
        S_feats, S_labels, dataset.scenario_map,
        "Scenario Embeddings - TRAIN+VAL",
        f"{args.output_dir}/{model_name}_scenario_trainval_{method}",
        method
    )
    
    plot_embedding(
        A_feats, A_labels, dataset.action_map,
        "Action Embeddings - TRAIN+VAL",
        f"{args.output_dir}/{model_name}_action_trainval_{method}",
        method
    )
    
    print(f"\n✅ Done! Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
