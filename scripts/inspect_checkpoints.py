#!/usr/bin/env python3
"""
Inspect checkpoints and provide project overview.

This script shows:
1. What's in each checkpoint file
2. Model architecture and parameters
3. How to access WandB results
4. How to evaluate models

Usage:
    python scripts/inspect_checkpoints.py
    python scripts/inspect_checkpoints.py --checkpoint checkpoints/fold1/best_model.pth
    python scripts/inspect_checkpoints.py --wandb --run-name "hierarchical_har_fold1"
"""

from pathlib import Path
import argparse
import sys

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Some features will be limited.")

try:
    from wandb import Api
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  WandB not available. Install with: pip install wandb")


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inspect_checkpoint(checkpoint_path):
    """Inspect a checkpoint file and show its contents"""
    print(f"\n{'='*80}")
    print(f"Inspecting: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch not available. Cannot load checkpoint.")
        print(f"   File size: {Path(checkpoint_path).stat().st_size / (1024*1024):.2f} MB")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check format
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("📦 Format: Dictionary with 'model_state_dict'")
                print(f"   Additional keys: {list(checkpoint.keys())}")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("📦 Format: Dictionary with 'state_dict'")
                print(f"   Additional keys: {list(checkpoint.keys())}")
            else:
                state_dict = checkpoint
                print("📦 Format: Dictionary (state_dict only)")
        else:
            state_dict = checkpoint
            print("📦 Format: Direct state_dict (OrderedDict)")
        
        # Analyze model structure
        print(f"\n📊 Model Structure:")
        print(f"   Total layers: {len(state_dict)}")
        
        # Group by component
        lle_keys = [k for k in state_dict.keys() if k.startswith('lle.')]
        hla_keys = [k for k in state_dict.keys() if k.startswith('hla.')]
        action_keys = [k for k in state_dict.keys() if k.startswith('action_head.')]
        
        print(f"\n   LLE (Low-Level Encoder): {len(lle_keys)} layers")
        print(f"   HLA (High-Level Architecture): {len(hla_keys)} layers")
        print(f"   Action Head: {len(action_keys)} layers")
        
        # Show key architecture indicators
        print(f"\n🔍 Architecture Indicators:")
        
        if any('conv_blocks' in k for k in lle_keys):
            print("   ✓ Multi-dilation CNN detected (main model)")
        elif any('convs' in k for k in lle_keys):
            print("   ✓ Simple CNN detected (baseline)")
        elif any('mlp' in k for k in lle_keys):
            print("   ✓ MLP detected (baseline)")
        
        if any('encoder' in k for k in hla_keys):
            print("   ✓ Transformer Encoder detected (HLA)")
        elif any('gru' in k for k in hla_keys):
            print("   ✓ GRU detected (HLA)")
        elif any('mlp' in k for k in hla_keys):
            print("   ✓ MLP detected (HLA baseline)")
        
        # Count parameters (approximate)
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"\n📈 Parameter Count (approximate):")
        print(f"   Total: {total_params:,}")
        
        # Show sample keys
        print(f"\n🔑 Sample Layer Keys (first 10):")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"   {i+1}. {key}: {shape}")
        
        if len(state_dict) > 10:
            print(f"   ... and {len(state_dict) - 10} more layers")
        
        return checkpoint
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None


def inspect_all_folds(checkpoints_dir="checkpoints"):
    """Inspect all fold checkpoints"""
    checkpoints_dir = Path(checkpoints_dir)
    
    print(f"\n{'='*80}")
    print("PROJECT OVERVIEW: Checkpoint Inspection")
    print(f"{'='*80}\n")
    
    print("📁 Available Checkpoints:\n")
    
    # Check fold directories
    fold_dirs = sorted([d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith('fold')])
    
    if fold_dirs:
        print(f"✅ Found {len(fold_dirs)} fold directories:")
        for fold_dir in fold_dirs:
            best_model = fold_dir / "best_model.pth"
            last_model = fold_dir / "last_model.pth"
            
            print(f"\n   {fold_dir.name}/")
            if best_model.exists():
                size_mb = best_model.stat().st_size / (1024 * 1024)
                print(f"      ✓ best_model.pth ({size_mb:.2f} MB)")
            else:
                print(f"      ✗ best_model.pth (missing)")
            
            if last_model.exists():
                size_mb = last_model.stat().st_size / (1024 * 1024)
                print(f"      ✓ last_model.pth ({size_mb:.2f} MB)")
            else:
                print(f"      ✗ last_model.pth (missing)")
    
    # Check root checkpoints
    root_best = checkpoints_dir / "best_model.pth"
    root_last = checkpoints_dir / "last_model.pth"
    
    print(f"\n   Root checkpoints/")
    if root_best.exists():
        size_mb = root_best.stat().st_size / (1024 * 1024)
        print(f"      ✓ best_model.pth ({size_mb:.2f} MB)")
    if root_last.exists():
        size_mb = root_last.stat().st_size / (1024 * 1024)
        print(f"      ✓ last_model.pth ({size_mb:.2f} MB)")
    
    # Inspect first fold if available
    if fold_dirs:
        print(f"\n{'='*80}")
        print("DETAILED INSPECTION: First Fold")
        print(f"{'='*80}")
        first_fold_best = fold_dirs[0] / "best_model.pth"
        if first_fold_best.exists():
            inspect_checkpoint(first_fold_best)


def get_wandb_results(run_name=None, project="har-imu-training"):
    """Get results from WandB"""
    if not WANDB_AVAILABLE:
        print("❌ WandB not available. Install with: pip install wandb")
        return None
    
    print(f"\n{'='*80}")
    print("WandB Results")
    print(f"{'='*80}\n")
    
    api = Api()
    
    try:
        runs = api.runs(project, filters={"display_name": {"$regex": run_name}} if run_name else {})
        
        if len(runs) == 0:
            print(f"❌ No runs found matching '{run_name}'")
            return None
        
        print(f"📊 Found {len(runs)} run(s):\n")
        
        for i, run in enumerate(runs[:10], 1):  # Show first 10
            print(f"{i}. Run: {run.name}")
            print(f"   ID: {run.id}")
            print(f"   URL: {run.url}")
            print(f"   Status: {run.state}")
            print(f"   Created: {run.created_at}")
            
            # Get config
            if run.config:
                print(f"\n   📋 Hyperparameters:")
                for key, value in list(run.config.items())[:10]:
                    print(f"      {key}: {value}")
                if len(run.config) > 10:
                    print(f"      ... and {len(run.config) - 10} more")
            
            # Get summary metrics
            if run.summary:
                print(f"\n   📈 Final Metrics:")
                for key, value in list(run.summary.items())[:10]:
                    if isinstance(value, (int, float)):
                        print(f"      {key}: {value:.4f}" if isinstance(value, float) else f"      {key}: {value}")
                if len(run.summary) > 10:
                    print(f"      ... and {len(run.summary) - 10} more")
            
            print()
        
        return runs
        
    except Exception as e:
        print(f"❌ Error accessing WandB: {e}")
        print("   Make sure you're logged in: wandb login")
        return None


def show_project_overview():
    """Show comprehensive project overview"""
    print(f"\n{'='*80}")
    print("PROJECT OVERVIEW: Hierarchical Activity Recognition")
    print(f"{'='*80}\n")
    
    print("📋 Current Architecture:")
    print(f"   LLE: Multi-dilation CNN (dilations [1,2,4]) → GRU → 128-dim embedding")
    print(f"   HLA: Transformer Encoder (2 layers, 4 heads) → 7 scenario classes")
    print(f"   Input: IMU data only (8 channels: 6 raw + 2 norm features)")
    print(f"   Gaze: Available in data but NOT currently used")
    
    print(f"\n📊 Data:")
    print(f"   Scenarios: 7 classes (Gardening excluded)")
    print(f"   Actions: 6 classes")
    print(f"   Videos: ~1,652 with IMU data")
    print(f"   Windows: ~355,580 labeled windows")
    
    print(f"\n💾 Checkpoint Structure:")
    print(f"   checkpoints/fold1/best_model.pth - Best model from fold 1")
    print(f"   checkpoints/fold1/last_model.pth  - Last epoch model from fold 1")
    print(f"   ... (same for fold2, fold3, fold4)")
    print(f"   checkpoints/best_model.pth        - Best model from single training")
    
    print(f"\n📦 What's Saved in Checkpoints:")
    print(f"   ✓ Model weights (state_dict)")
    print(f"   ✗ Hyperparameters (stored in WandB)")
    print(f"   ✗ Training metrics (stored in WandB)")
    print(f"   ✗ Training history (stored in WandB)")
    
    print(f"\n🔍 How to Access Results:")
    print(f"   1. WandB Dashboard: https://wandb.ai/wandbleo/har-imu-training")
    print(f"   2. Use this script: python scripts/inspect_checkpoints.py --wandb")
    print(f"   3. Use test script: python scripts/test_model.py --checkpoint <path>")
    
    print(f"\n📈 Evaluation:")
    print(f"   python scripts/test_model.py --checkpoint checkpoints/fold1/best_model.pth")
    print(f"   This will show: F1 scores, accuracy, confusion matrices")


def main():
    parser = argparse.ArgumentParser(description="Inspect checkpoints and project overview")
    parser.add_argument('--checkpoint', type=str, default=None, help='Specific checkpoint to inspect')
    parser.add_argument('--wandb', action='store_true', help='Show WandB results')
    parser.add_argument('--run-name', type=str, default=None, help='WandB run name to search')
    parser.add_argument('--all-folds', action='store_true', help='Inspect all fold checkpoints')
    args = parser.parse_args()
    
    # Show project overview
    show_project_overview()
    
    # Inspect specific checkpoint
    if args.checkpoint:
        inspect_checkpoint(args.checkpoint)
    
    # Inspect all folds
    elif args.all_folds:
        inspect_all_folds()
    
    # Show WandB results
    if args.wandb:
        get_wandb_results(args.run_name)
    
    # Default: inspect all folds
    if not args.checkpoint and not args.wandb:
        inspect_all_folds()


if __name__ == "__main__":
    main()

