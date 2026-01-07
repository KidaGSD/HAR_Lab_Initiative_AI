#!/usr/bin/env python3
"""
Test evaluation script for hierarchical activity recognition models.

Usage:
    python scripts/test_model.py --checkpoint checkpoints/best_model.pth
    python scripts/test_model.py --checkpoint checkpoints/best_model.pth --output-dir results/test_eval
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import json

# Import wandb with fallback
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. Metrics won't be logged to WandB.")

# Import model and dataset classes from training script
import sys
sys.path.append(str(Path(__file__).parent))
from train_hierarchical import (
    HierarchicalModel, HierarchicalDataset, CONFIG
)

# Import baseline models
sys.path.append(str(Path(__file__).parent.parent))
try:
    from src.models.baseline_models import create_baseline_model
    BASELINE_AVAILABLE = True
except ImportError:
    BASELINE_AVAILABLE = False
    print("WARNING: Baseline models not available. Cannot load baseline checkpoints.")


def detect_model_type(checkpoint_path, state_dict):
    """Detect if checkpoint is from baseline or hierarchical model"""
    checkpoint_path_str = str(checkpoint_path).lower()
    
    # Check path for baseline indicators
    if 'baseline' in checkpoint_path_str or any(name in checkpoint_path_str for name in ['mlp_mlp', 'cnn_mlp', 'imu2clip', 'cnn_lstm_gru']):
        # Extract model name from path
        for model_name in ['mlp_mlp', 'cnn_mlp', 'imu2clip', 'cnn_lstm_gru']:
            if model_name in checkpoint_path_str:
                return 'baseline', model_name
    
    # Check state_dict structure
    # Baseline models have 'lle.convs' or 'lle.mlp', hierarchical has 'lle.conv_blocks'
    if 'lle.convs' in state_dict or 'lle.mlp' in state_dict:
        # Try to infer model name from structure
        if 'lle.mlp' in state_dict:
            return 'baseline', 'mlp_mlp'
        elif 'lle.convs' in state_dict:
            # Check if it's CNN-MLP or IMU2CLIP by filter size
            # This is a heuristic - you might need to adjust
            return 'baseline', 'cnn_mlp'  # Default, could be improved
    
    return 'hierarchical', None


def load_model(checkpoint_path, device, config=None):
    """Load model from checkpoint (handles both baseline and hierarchical models)"""
    if config is None:
        config = CONFIG.copy()
    
    # Load checkpoint first to inspect
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume the dict itself is the state_dict
            state_dict = checkpoint
    else:
        # Direct state_dict (OrderedDict)
        state_dict = checkpoint
    
    # Detect model type
    model_type, model_name = detect_model_type(checkpoint_path, state_dict)
    
    # Detect number of action classes from checkpoint
    action_head_key = 'action_head.weight'
    if action_head_key not in state_dict:
        # Try with module prefix
        action_head_key = 'module.action_head.weight'
    
    num_action_classes = None
    if action_head_key in state_dict:
        num_action_classes = state_dict[action_head_key].shape[0]
        print(f"Detected {num_action_classes} action classes from checkpoint")
    
    # Detect number of scenario classes from checkpoint
    # Try different possible keys for scenario output layer
    scenario_head_keys = [
        'hla.mlp.6.weight',  # MLP_HLA last layer
        'hla.head.weight',   # GRU_HLA or Transformer head
        'hla.mlp.4.weight',  # Alternative MLP structure
    ]
    
    num_scenario_classes = None
    for key in scenario_head_keys:
        if key in state_dict:
            num_scenario_classes = state_dict[key].shape[0]
            print(f"Detected {num_scenario_classes} scenario classes from checkpoint (key: {key})")
            break
    
    # If not found, try with module prefix
    if num_scenario_classes is None:
        for key in scenario_head_keys:
            prefixed_key = f'module.{key}'
            if prefixed_key in state_dict:
                num_scenario_classes = state_dict[prefixed_key].shape[0]
                print(f"Detected {num_scenario_classes} scenario classes from checkpoint (key: {prefixed_key})")
                break
    
    # Strip 'module.' prefix if present (from DataParallel)
    if any(k.startswith('module.') for k in state_dict.keys()):
        print("⚠️  Detected 'module.' prefix in checkpoint (from DataParallel). Stripping...")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix (7 chars)
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    # Update config with detected number of classes BEFORE creating model
    if num_scenario_classes:
        config['hla']['num_classes'] = num_scenario_classes
        print(f"Updated config: num_scenario_classes = {num_scenario_classes}")
    
    # Create appropriate model
    if model_type == 'baseline':
        if not BASELINE_AVAILABLE:
            raise ValueError("Baseline models not available. Cannot load baseline checkpoint.")
        
        if model_name is None:
            raise ValueError("Could not determine baseline model name from checkpoint path.")
        
        print(f"Loading baseline model: {model_name}")
        
        # Update config with correct number of action classes
        if num_action_classes:
            if 'lla' not in config:
                config['lla'] = {}
            config['lla']['num_classes'] = num_action_classes
        
        # Create baseline model
        model = create_baseline_model(model_name, config).to(device)
        
    else:
        print("Loading hierarchical model")
        # Update config with correct number of action classes
        if num_action_classes:
            if 'action' not in config:
                config['action'] = {}
            config['action']['num_classes'] = num_action_classes
        
        model = HierarchicalModel(config).to(device)  # Will now use config value ✅
        
        # Replace action_head if checkpoint has different number of classes
        if num_action_classes and num_action_classes != 6:
            print(f"⚠️  Replacing action_head: 6 -> {num_action_classes} classes")
            embedding_dim = config['lle']['embedding_dim']
            model.action_head = nn.Linear(embedding_dim, num_action_classes).to(device)
            
            # Also need to patch the forward method to use dynamic num_classes
            # Store it as an attribute
            model.num_action_classes = num_action_classes
            
            # Create a new forward method that uses num_action_classes
            original_forward = model.forward
            def patched_forward(x):
                b, s, w, c = x.shape
                x_flat = x.view(b*s, w, c)
                embeddings = model.lle(x_flat)
                action_logits = model.action_head(embeddings)
                action_logits = action_logits.view(b, s, model.num_action_classes)
                embeddings_seq = embeddings.view(b, s, -1)
                scenario_logits = model.hla(embeddings_seq)
                return scenario_logits, action_logits
            
            model.forward = patched_forward
    
    # Load with strict=False to allow missing keys
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️  Missing keys: {len(missing_keys)}")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"   - {key}")
    
    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"   - {key}")
    
    model.eval()
    print("✓ Model loaded successfully")
    return model


def evaluate_model(model, test_loader, device, dataset):
    """Evaluate model on test set"""
    model.eval()
    
    all_scenario_preds = []
    all_scenario_labels = []
    all_action_preds = []
    all_action_labels = []
    
    scenario_correct = 0
    action_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = batch['inputs'].to(device)  # (B, Seq, 50, C)
            scenario_labels = batch['scenario_label'].to(device)  # (B,)
            action_labels = batch['action_labels'].to(device)  # (B, Seq)
            
            # Forward pass
            scenario_logits, action_logits = model(inputs)
            
            # Scenario predictions
            scenario_preds = scenario_logits.argmax(dim=1)
            scenario_correct += (scenario_preds == scenario_labels).sum().item()
            
            # Action predictions (per-window)
            action_preds = action_logits.argmax(dim=2)  # (B, Seq)
            
            # Collect predictions and labels
            all_scenario_preds.extend(scenario_preds.cpu().numpy())
            all_scenario_labels.extend(scenario_labels.cpu().numpy())
            
            # Flatten action predictions and labels
            valid_mask = action_labels != -1  # Ignore padding/unknown
            if valid_mask.any():
                all_action_preds.extend(action_preds[valid_mask].cpu().numpy())
                all_action_labels.extend(action_labels[valid_mask].cpu().numpy())
                action_correct += (action_preds[valid_mask] == action_labels[valid_mask]).sum().item()
            
            total_samples += scenario_labels.size(0)
    
    # Calculate metrics
    scenario_acc = scenario_correct / total_samples
    scenario_f1 = f1_score(all_scenario_labels, all_scenario_preds, average='weighted')
    scenario_f1_macro = f1_score(all_scenario_labels, all_scenario_preds, average='macro')
    
    # Action metrics (only on valid labels)
    if len(all_action_labels) > 0:
        action_acc = action_correct / len(all_action_labels)
        action_f1 = f1_score(all_action_labels, all_action_preds, average='weighted')
        action_f1_macro = f1_score(all_action_labels, all_action_preds, average='macro')
    else:
        action_acc = 0.0
        action_f1 = 0.0
        action_f1_macro = 0.0
    
    return {
        'scenario_acc': scenario_acc,
        'scenario_f1_weighted': scenario_f1,
        'scenario_f1_macro': scenario_f1_macro,
        'action_acc': action_acc,
        'action_f1_weighted': action_f1,
        'action_f1_macro': action_f1_macro,
        'scenario_preds': all_scenario_preds,
        'scenario_labels': all_scenario_labels,
        'action_preds': all_action_preds,
        'action_labels': all_action_labels,
        'total_samples': total_samples
    }


def print_results(results, dataset, output_dir=None):
    """Print and save evaluation results"""
    print("\n" + "="*80)
    print("TEST SET EVALUATION RESULTS")
    print("="*80)
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Scenario Accuracy:  {results['scenario_acc']:.4f}")
    print(f"  Scenario F1 (weighted): {results['scenario_f1_weighted']:.4f}")
    print(f"  Scenario F1 (macro):  {results['scenario_f1_macro']:.4f}")
    print(f"  Action Accuracy:     {results['action_acc']:.4f}")
    print(f"  Action F1 (weighted): {results['action_f1_weighted']:.4f}")
    print(f"  Action F1 (macro):    {results['action_f1_macro']:.4f}")
    print(f"  Total samples:        {results['total_samples']:,}")
    
    # Scenario confusion matrix
    print("\n" + "="*80)
    print("SCENARIO CONFUSION MATRIX")
    print("="*80)
    scenario_cm = confusion_matrix(results['scenario_labels'], results['scenario_preds'])
    scenario_classes = [dataset.idx_to_scenario[i] for i in sorted(dataset.scenario_map.values())]
    
    print("\nPredicted ->")
    print("Actual ↓", end="")
    for cls in scenario_classes:
        print(f"{cls:>15}", end="")
    print()
    
    for i, cls in enumerate(scenario_classes):
        print(f"{cls:>15}", end="")
        for j in range(len(scenario_classes)):
            print(f"{scenario_cm[i, j]:>15}", end="")
        print()
    
    # Scenario classification report
    print("\n" + "="*80)
    print("SCENARIO CLASSIFICATION REPORT")
    print("="*80)
    scenario_report = classification_report(
        results['scenario_labels'],
        results['scenario_preds'],
        target_names=scenario_classes,
        digits=4
    )
    print(scenario_report)
    
    # Action confusion matrix
    if len(results['action_labels']) > 0:
        print("\n" + "="*80)
        print("ACTION CONFUSION MATRIX")
        print("="*80)
        
        # Get unique classes from both labels and predictions
        unique_labels = sorted(set(results['action_labels']))
        unique_preds = sorted(set(results['action_preds']))
        all_unique_classes = sorted(set(unique_labels + unique_preds))
        
        # Create confusion matrix
        action_cm = confusion_matrix(
            results['action_labels'], 
            results['action_preds'],
            labels=all_unique_classes
        )
        
        # Get class names for the classes actually present
        action_classes = [dataset.idx_to_action.get(i, f"Class_{i}") for i in all_unique_classes]
        
        print(f"\nNote: Model predicts {len(unique_preds)} classes, dataset has {len(unique_labels)} classes")
        print(f"Confusion matrix shape: {action_cm.shape}")
        
        print("\nPredicted ->")
        print("Actual ↓", end="")
        for cls in action_classes:
            print(f"{cls:>20}", end="")
        print()
        
        # Print confusion matrix - use actual dimensions
        for i, cls in enumerate(action_classes):
            print(f"{cls:>20}", end="")
            for j in range(action_cm.shape[1]):  # Use actual number of columns
                if j < len(action_classes):
                    print(f"{action_cm[i, j]:>20}", end="")
                else:
                    print(f"{0:>20}", end="")  # Pad with 0 if needed
            print()
        
        # Action classification report
        print("\n" + "="*80)
        print("ACTION CLASSIFICATION REPORT")
        print("="*80)
        action_report = classification_report(
            results['action_labels'],
            results['action_preds'],
            labels=all_unique_classes,
            target_names=action_classes,
            digits=4,
            zero_division=0
        )
        print(action_report)
    
    # Save results to file
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics = {
            'scenario_acc': float(results['scenario_acc']),
            'scenario_f1_weighted': float(results['scenario_f1_weighted']),
            'scenario_f1_macro': float(results['scenario_f1_macro']),
            'action_acc': float(results['action_acc']),
            'action_f1_weighted': float(results['action_f1_weighted']),
            'action_f1_macro': float(results['action_f1_macro']),
            'total_samples': int(results['total_samples'])
        }
        
        with open(output_dir / 'test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save confusion matrices as CSV
        scenario_cm_df = pd.DataFrame(
            scenario_cm,
            index=scenario_classes,
            columns=scenario_classes
        )
        scenario_cm_df.to_csv(output_dir / 'scenario_confusion_matrix.csv')
        
        if len(results['action_labels']) > 0:
            action_cm_df = pd.DataFrame(
                action_cm,
                index=action_classes,
                columns=action_classes
            )
            action_cm_df.to_csv(output_dir / 'action_confusion_matrix.csv')
        
        print(f"\n💾 Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--processed-dir', type=str, default='data/processed_ego4d',
                        help='Directory containing processed IMU data')
    parser.add_argument('--scenario-labels', type=str, default='data/labels/scenario_labels.csv',
                        help='Path to scenario labels CSV')
    parser.add_argument('--action-labels', type=str, default='data/labels/action_labels_llm_validated.csv',
                        help='Path to action labels CSV')
    parser.add_argument('--baseline-action-labels', type=str, default='data/labels/action_labels_4class.csv',
                        help='Path to action labels CSV for baseline models (4-class)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save evaluation results (optional)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for evaluation')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='WandB project name to log results (optional)')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='WandB run name (optional)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., cuda:0, cuda:2, cpu). Default: auto-detect')
    
    args = parser.parse_args()
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    print("\n" + "="*80)
    print("LOADING TEST DATA")
    print("="*80)
    
    scenario_df = pd.read_csv(args.scenario_labels)
    test_uids = scenario_df[scenario_df['split'] == 'test']['video_uid'].tolist()
    
    # Check for processed data availability
    processed_dir = Path(args.processed_dir)
    available_uids = set([p.parent.name for p in processed_dir.glob('*/seq.npz')])
    test_uids_available = [u for u in test_uids if u in available_uids]
    
    print(f"Test videos in labels: {len(test_uids)}")
    print(f"Test videos with processed data: {len(test_uids_available)}")
    
    if len(test_uids_available) == 0:
        print("❌ ERROR: No test videos with processed data found!")
        return
    
    # Detect if baseline model from checkpoint path
    checkpoint_path_str = str(args.checkpoint).lower()
    is_baseline = 'baseline' in checkpoint_path_str or any(name in checkpoint_path_str for name in ['mlp_mlp', 'cnn_mlp', 'imu2clip', 'cnn_lstm_gru'])
    
    # Use appropriate action labels
    if is_baseline:
        action_labels_path = args.baseline_action_labels
        print(f"Using 4-class action labels for baseline model: {action_labels_path}")
    else:
        action_labels_path = args.action_labels
    
    # Create test dataset
    test_ds = HierarchicalDataset(
        test_uids_available,
        args.processed_dir,
        args.scenario_labels,
        action_labels_path  # Use detected path
    )
    test_ds.training = False
    
    print(f"Test dataset created: {len(test_ds)} samples")
    print(f"Scenarios: {test_ds.scenario_map}")
    
    # Update CONFIG with actual number of scenarios from dataset
    CONFIG['hla']['num_classes'] = test_ds.num_scenarios
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Load model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model = load_model(args.checkpoint, device, CONFIG)
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    results = evaluate_model(model, test_loader, device, test_ds)
    
    # Print results
    print_results(results, test_ds, args.output_dir)
    
    # Log to WandB if requested
    if args.wandb_project and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"test_eval_{Path(args.checkpoint).stem}",
            config={
                'checkpoint': str(args.checkpoint),
                'test_samples': results['total_samples']
            }
        )
        
        wandb.log({
            'test_scenario_acc': results['scenario_acc'],
            'test_scenario_f1_weighted': results['scenario_f1_weighted'],
            'test_scenario_f1_macro': results['scenario_f1_macro'],
            'test_action_acc': results['action_acc'],
            'test_action_f1_weighted': results['action_f1_weighted'],
            'test_action_f1_macro': results['action_f1_macro'],
        })
        
        # Log confusion matrices
        scenario_cm = confusion_matrix(results['scenario_labels'], results['scenario_preds'])
        scenario_classes = [test_ds.idx_to_scenario[i] for i in sorted(test_ds.scenario_map.values())]
        
        wandb.log({
            "test_scenario_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=results['scenario_labels'],
                preds=results['scenario_preds'],
                class_names=scenario_classes
            )
        })
        
        if len(results['action_labels']) > 0:
            action_cm = confusion_matrix(results['action_labels'], results['action_preds'])
            action_classes = [test_ds.idx_to_action[i] for i in sorted(test_ds.action_map.values())]
            
            wandb.log({
                "test_action_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=results['action_labels'],
                    preds=results['action_preds'],
                    class_names=action_classes
                )
            })
        
        wandb.finish()
        print("\n✓ Results logged to WandB")


if __name__ == "__main__":
    main()