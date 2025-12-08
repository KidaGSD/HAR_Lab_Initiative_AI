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
    Publication-quality dimensionality reduction and plotting.
    """
    print(f"    Reducing {features.shape[0]} samples to 2D using {method.upper()}...")
    
    # Reduce to 2D
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(features)
        method_name = 'PCA'
        var_explained = reducer.explained_variance_ratio_
        subtitle = f'Variance: {var_explained[0]:.1%} + {var_explained[1]:.1%} = {sum(var_explained):.1%}'
    else:  # t-SNE
        perplexity = min(30, max(5, len(features)//100))  # Adaptive perplexity
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                       learning_rate=200, n_iter=1000, verbose=0)
        reduced = reducer.fit_transform(features)
        method_name = 't-SNE'
        subtitle = f'Perplexity={perplexity}, Iter=1000'
    
    # Create DataFrame
    idx_to_label = {v: k for k, v in label_map.items()}
    label_names = [idx_to_label.get(l, f'Unknown') for l in labels]
    
    df = pd.DataFrame({
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'Label': label_names
    })
    
    # Remove unknown
    df = df[~df['Label'].str.contains('Unknown')].copy()
    
    print(f"    Plotting {len(df)} points across {df['Label'].nunique()} classes...")
    
    # Create figure with high DPI
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    
    # Get unique labels and assign distinct visual properties
    unique_labels = sorted(df['Label'].unique())
    n_classes = len(unique_labels)
    
    # Use distinguishable colors
    if n_classes <= 10:
        colors = sns.color_palette('tab10', n_colors=n_classes)
    else:
        colors = sns.color_palette('husl', n_colors=n_classes)
    
    # Different markers for each class (up to 7 scenarios/4 actions)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    centroid_markers = ['X', 'P', '*', 'd', 'H', '8', 'p', 's', '^', 'v']
    
    # Plot each class
    for i, label in enumerate(unique_labels):
        mask = df['Label'] == label
        class_data = df[mask]
        
        # Use smaller, semi-transparent points for large datasets
        alpha = min(0.7, max(0.3, 500 / len(class_data)))  # Adaptive transparency
        point_size = min(25, max(5, 2000 / len(class_data)))  # Adaptive size
        
        # Scatter plot
        ax.scatter(
            class_data['x'],
            class_data['y'],
            label=label,
            alpha=alpha,
            s=point_size,
            c=[colors[i]],
            marker=markers[i % len(markers)],
            edgecolors='white',
            linewidths=0.3
        )
        
        # Add centroid with distinct marker
        centroid_x = class_data['x'].mean()
        centroid_y = class_data['y'].mean()
        ax.scatter(
            centroid_x, centroid_y,
            marker=centroid_markers[i % len(centroid_markers)],
            s=300,
            c=[colors[i]],
            edgecolors='black',
            linewidths=2.5,
            zorder=100,
            alpha=1.0
        )
        
        # Add text label near centroid
        ax.annotate(
            label,
            (centroid_x, centroid_y),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7, edgecolor='black'),
            zorder=101
        )
    
    # Styling
    ax.set_title(f'{title}\n{method_name} Visualization - {subtitle}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(f'{method_name} Component 1', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'{method_name} Component 2', fontsize=13, fontweight='bold')
    
    # Legend with custom styling
    legend = ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        title='Classes',
        title_fontsize=11
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Background
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # Tight layout
    plt.tight_layout()
    
    # Save both formats
    png_path = f"{save_path_base}.png"
    pdf_path = f"{save_path_base}.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    ✓ Saved: {Path(png_path).name} & {Path(pdf_path).name}")



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

python scripts/visualize_embeddings_fast.py \
  --config configs/beta_1.0.yaml \
  --checkpoint checkpoints/experiments_20251206_224347/beta_1.0/best_model.pth \
  --gpu 7 \
  --use-tsne \
  --subset 100 \
  --output-dir outputs/embeddings_viz