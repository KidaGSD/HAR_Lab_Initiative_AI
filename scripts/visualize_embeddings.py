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
            sequences = batch['inputs'].to(device)  # HierarchicalDataset returns 'inputs'
            s_label = batch['scenario_label']  # (B,)
            a_label = batch['action_labels']   # (B, Seq)
            
            # Forward pass triggers hooks
            model(sequences)
            
            # Store labels
            scenario_labels.append(s_label.numpy())
            # Action labels need to be expanded to match flattened embeddings
            # a_label is (B, Seq) - flatten it
            batch_size = len(s_label)
            seq_len = sequences.shape[1]
            action_labels.append(a_label.view(-1).numpy())
            
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
    parser = argparse.ArgumentParser(description="Visualize hierarchical model embeddings (train+val data)")
    parser.add_argument("--config", type=str, required=True, help="Path to model config (e.g., configs/beta_0.5.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--output-dir", type=str, default="outputs/plots", help="Output directory for plots")
    parser.add_argument("--use-tsne", action="store_true", help="Use t-SNE instead of PCA")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (e.g., 0, 1, 2). If not specified, uses CUDA_VISIBLE_DEVICES or auto-detects")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎨 HIERARCHICAL HAR EMBEDDING VISUALIZATION")
    print("=" * 60)
    
    # ============ VALIDATION PHASE ============
    print("\n📋 Validating environment and paths...")
    
    # 1. Check config file
    if not os.path.exists(args.config):
        print(f"❌ ERROR: Config file not found: {args.config}")
        print(f"   Available configs in configs/:")
        for f in Path("configs").glob("*.yaml"):
            print(f"     - {f}")
        sys.exit(1)
    print(f"✓ Config file found: {args.config}")
    
    # 2. Check checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"❌ ERROR: Checkpoint not found: {args.checkpoint}")
        print(f"   Try: find checkpoints/ -name 'best_model.pth'")
        sys.exit(1)
    print(f"✓ Checkpoint found: {args.checkpoint}")
    
    # 3. Check data directories
    required_paths = {
        "Labels (scenario)": "data/labels/scenario_labels.csv",
        "Labels (action)": "data/labels/action_labels_4class.csv",
        "Processed data": "data/processed_ego4d"
    }
    
    for name, path in required_paths.items():
        if not os.path.exists(path):
            print(f"❌ ERROR: {name} not found: {path}")
            if "processed" in path:
                print(f"   Make sure you're running from the project root directory")
                print(f"   Current directory: {os.getcwd()}")
            sys.exit(1)
        print(f"✓ {name} found")
    
    # 4. Check if processed_ego4d has data
    processed_dir = Path("data/processed_ego4d")
    sample_files = list(processed_dir.glob("*/seq.npz"))
    if len(sample_files) == 0:
        print(f"❌ ERROR: No .npz files found in {processed_dir}/")
        print(f"   Directory exists but appears empty")
        sys.exit(1)
    print(f"✓ Found {len(sample_files)} processed video files")
    
    # 5. Manual GPU assignment
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"✓ GPU manually set to: {args.gpu}")
    
    # ============ INITIALIZATION PHASE ============
    print("\n🔧 Initializing...")
    
    try:
        config = load_config(args.config)
        print(f"✓ Config loaded successfully")
    except Exception as e:
        print(f"❌ ERROR loading config: {e}")
        sys.exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"✓ Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"⚠️  Using CPU (will be slower)")
    
    # ============ DATA LOADING PHASE ============
    print(f"\n📂 Loading data for visualization...")
    
    # Load UIDs from scenario_labels.csv 'split' column (same as training)
    try:
        scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
        
        # Get train and val UIDs from split column
        train_uids = scenario_df[scenario_df['split'] == 'train']['video_uid'].tolist()
        val_uids = scenario_df[scenario_df['split'] == 'val']['video_uid'].tolist()
        
        print(f"✓ Loaded {len(train_uids)} training UIDs from scenario_labels.csv")
        print(f"✓ Loaded {len(val_uids)} validation UIDs from scenario_labels.csv")
        
        # Combine train and val
        uids = train_uids + val_uids
        print(f"\n✓ Total: {len(uids)} videos (train + val only)")
        
        if len(uids) == 0:
            print(f"❌ ERROR: No train or val UIDs found in scenario_labels.csv!")
            print(f"   Check that 'split' column contains 'train' and 'val' values")
            sys.exit(1)
            
    except FileNotFoundError:
        print(f"❌ ERROR: scenario_labels.csv not found at data/labels/")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ ERROR: Missing column in scenario_labels.csv: {e}")
        print(f"   Required columns: 'video_uid', 'split'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR reading scenario labels: {e}")
        sys.exit(1)
    
    # Create dataset
    print(f"\n🔨 Creating dataset...")
    try:
        dataset = HierarchicalDataset(
            uids,
            "data/processed_ego4d",
            "data/labels/scenario_labels.csv",
            "data/labels/action_labels_4class.csv",
            config,
            training=False
        )
        print(f"✓ Dataset created: {len(dataset)} samples")
        print(f"  Scenarios: {dataset.num_scenarios} classes")
        print(f"  Actions: {dataset.num_action_classes} classes")
    except ValueError as e:
        print(f"❌ ERROR creating dataset: {e}")
        print(f"\nPossible causes:")
        print(f"  1. No valid .npz files for the given UIDs")
        print(f"  2. Data normalization failed (no valid data)")
        print(f"  3. Mismatch between UIDs and available data")
        sys.exit(1)
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create dataloader
    try:
        batch_size = min(config['training']['batch_size'], len(dataset))
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=min(4, os.cpu_count() or 4)
        )
        print(f"✓ DataLoader created (batch_size={batch_size})")
    except Exception as e:
        print(f"❌ ERROR creating DataLoader: {e}")
        sys.exit(1)
    
    # ============ MODEL LOADING PHASE ============
    print(f"\n🧠 Loading model...")
    try:
        model = HierarchicalModel(config).to(device)
        print(f"✓ Model architecture created")
        
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded checkpoint (with state_dict wrapper)")
        else:
            model.load_state_dict(checkpoint)
            print(f"✓ Loaded checkpoint (direct state_dict)")
        
        model.eval()
        print(f"✓ Model set to eval mode")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        print(f"\nPossible causes:")
        print(f"  1. Checkpoint incompatible with current model architecture")
        print(f"  2. Checkpoint corrupted")
        print(f"  3. Config mismatch with saved model")
        sys.exit(1)
    
    # ============ EMBEDDING EXTRACTION PHASE ============
    print(f"\n🔬 Extracting embeddings...")
    try:
        S_feats, S_labels, A_feats, A_labels = extract_embeddings(model, loader, device)
        print(f"✓ Extracted embeddings:")
        print(f"  Scenario: {S_feats.shape[0]} samples × {S_feats.shape[1]} dims")
        print(f"  Action:   {A_feats.shape[0]} samples × {A_feats.shape[1]} dims")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"❌ GPU OUT OF MEMORY!")
            print(f"   Try: --gpu <different_gpu> or reduce batch size in config")
        else:
            print(f"❌ RUNTIME ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ============ VISUALIZATION PHASE ============
    print(f"\n🎨 Generating visualizations...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    method = 'tsne' if args.use_tsne else 'pca'
    model_name = Path(args.checkpoint).parent.name
    
    try:
        # Plot Scenario Features
        print(f"  📊 Plotting scenario embeddings ({method.upper()})...")
        plot_embedding(
            S_feats, S_labels, dataset.scenario_map, 
            f"Scenario (High-Level) - TRAIN+VAL", 
            f"{args.output_dir}/{model_name}_scenario_trainval_{method}",
            method
        )
        
        # Plot Action Features
        print(f"  📊 Plotting action embeddings ({method.upper()})...")
        plot_embedding(
            A_feats, A_labels, dataset.action_map, 
            f"Action (Low-Level) - TRAIN+VAL", 
            f"{args.output_dir}/{model_name}_action_trainval_{method}",
            method
        )
        
        print(f"\n✅ VISUALIZATION COMPLETE!")
        print(f"\n📁 Output files:")
        for f in sorted(Path(args.output_dir).glob(f"{model_name}*{method}*")):
            size = f.stat().st_size / 1024
            print(f"   {f.name} ({size:.1f} KB)")
        
        print(f"\n💡 Next steps:")
        print(f"   1. View plots: open {args.output_dir}/")
        print(f"   2. Download: scp -r server:{os.getcwd()}/{args.output_dir} ./")
        print(f"   3. Use .pdf files for LaTeX reports")
        
    except Exception as e:
        print(f"❌ ERROR during plotting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
