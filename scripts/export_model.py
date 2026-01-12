#!/usr/bin/env python3
"""
Export trained hierarchical model to ONNX/TorchScript format for deployment.
Also saves normalization statistics needed for inference.

Usage:
    python scripts/export_model.py --checkpoint checkpoints/fold1/best_model.pth --output-dir deployment
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from pathlib import Path
import sys

# Import model classes
sys.path.append(str(Path(__file__).parent))
from train_hierarchical import (
    HierarchicalModel, HierarchicalDataset, CONFIG
)


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Strip 'module.' prefix if present (from DataParallel)
    if any(k.startswith('module.') for k in state_dict.keys()):
        print("⚠️  Detected 'module.' prefix in checkpoint. Stripping...")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    # Create model
    model = HierarchicalModel(CONFIG).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✓ Model loaded successfully")
    return model


def export_to_onnx(model, output_path, device, seq_len=30, window_size=50, in_channels=10):
    """Export model to ONNX format"""
    print(f"\nExporting to ONNX: {output_path}")
    
    # Create dummy input: (batch_size=1, seq_len=30, window_size=50, channels=10)
    dummy_input = torch.randn(1, seq_len, window_size, in_channels).to(device)
    
    # Set model to eval mode and disable gradients
    model.eval()
    with torch.no_grad():
        # Test forward pass first
        try:
            _ = model(dummy_input)
        except Exception as e:
            print(f"⚠️  Model forward pass failed: {e}")
            raise
    
    # Try direct ONNX export first (simpler, works if model is ONNX-compatible)
    print("   Attempting direct ONNX export...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=['input'],
            output_names=['scenario_logits', 'action_logits'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'scenario_logits': {0: 'batch_size'},
                'action_logits': {0: 'batch_size'}
            },
            opset_version=11,
            do_constant_folding=True,
            verbose=False,
            export_params=True
        )
        print(f"✓ ONNX model exported directly: {output_path}")
    except Exception as e:
        print(f"⚠️  Direct ONNX export failed: {e}")
        print("   TransformerEncoderLayer may not be ONNX-compatible")
        print("   Recommendation: Use TorchScript format instead (--export-format torchscript)")
        print("   TorchScript is more reliable for complex models with transformers")
        raise RuntimeError(f"ONNX export failed. TransformerEncoderLayer is not ONNX-compatible.\n"
                         f"Use TorchScript instead: --export-format torchscript\n"
                         f"Error: {e}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validation passed")
    except ImportError:
        print("⚠️  onnx package not installed, skipping validation")
    except Exception as e:
        print(f"⚠️  ONNX validation warning: {e}")


def export_to_torchscript(model, output_path, device, seq_len=30, window_size=50, in_channels=10):
    """Export model to TorchScript format"""
    print(f"\nExporting to TorchScript: {output_path}")
    
    # Create dummy input
    dummy_input = torch.randn(1, seq_len, window_size, in_channels).to(device)
    
    model.eval()
    
    # Try scripting first (better for TransformerEncoderLayer with optimized kernels)
    print("   Attempting TorchScript scripting (better for TransformerEncoderLayer)...")
    try:
        scripted_model = torch.jit.script(model)
        scripted_model.eval()
        
        # Verify the scripted model works
        with torch.no_grad():
            _ = scripted_model(dummy_input)
        
        scripted_model.save(str(output_path))
        print(f"✓ TorchScript model exported via scripting: {output_path}")
    except Exception as e:
        print(f"⚠️  TorchScript scripting failed: {e}")
        print("   Trying tracing instead (may have issues with TransformerEncoderLayer)...")
        
        # Fallback to tracing
        try:
            with torch.no_grad():
                traced_model = torch.jit.trace(model, dummy_input, strict=False)
            traced_model.eval()
            
            # Verify the traced model works
            with torch.no_grad():
                _ = traced_model(dummy_input)
            
            traced_model.save(str(output_path))
            print(f"✓ TorchScript model exported via tracing: {output_path}")
        except Exception as e2:
            print(f"❌ TorchScript tracing also failed: {e2}")
            print("\n   Troubleshooting:")
            print("   - TransformerEncoderLayer uses optimized kernels that may not trace/script well")
            print("   - Try exporting with --export-format torchscript only (skip ONNX)")
            print("   - Or consider using a GRU-based HLA instead of Transformer")
            raise RuntimeError(f"TorchScript export failed.\nScripting error: {e}\nTracing error: {e2}")
    
    # Verify TorchScript model
    try:
        loaded_model = torch.jit.load(str(output_path))
        with torch.no_grad():
            output = loaded_model(dummy_input)
        print("✓ TorchScript model validation passed")
    except Exception as e:
        print(f"⚠️  TorchScript validation warning: {e}")


def save_normalization_stats(dataset, output_path, config):
    """Extract and save normalization statistics from dataset"""
    print(f"\nSaving normalization statistics: {output_path}")
    
    stats = {
        'imu_mean': dataset.imu_mean.tolist(),
        'imu_std': dataset.imu_std.tolist(),
        'gaze_mean': dataset.gaze_mean.tolist(),
        'gaze_std': dataset.gaze_std.tolist(),
        'config': {
            'use_gaze': config['data'].get('use_gaze', True),
            'add_norm_features': config['data'].get('add_norm_features', True),
            'per_video_center': config['data'].get('per_video_center', True),
            'seq_len': config['hla']['seq_len'],
            'window_size': config['lle']['window_size'],
            'stride': config['data'].get('stride', 5),
            'in_channels': config['lle']['in_channels']
        },
        'scenario_map': dataset.scenario_map,
        'action_map': dataset.action_map,
        'idx_to_scenario': dataset.idx_to_scenario,
        'idx_to_action': dataset.idx_to_action
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Normalization stats saved: {output_path}")
    print(f"   IMU mean: {np.array(stats['imu_mean'])}")
    print(f"   IMU std:  {np.array(stats['imu_std'])}")
    if stats['config']['use_gaze']:
        print(f"   Gaze mean: {np.array(stats['gaze_mean'])}")
        print(f"   Gaze std:  {np.array(stats['gaze_std'])}")


def validate_exported_model(original_model, onnx_path, device, seq_len=30, window_size=50, in_channels=10):
    """Validate that exported ONNX model produces same outputs as PyTorch model"""
    print(f"\nValidating exported model...")
    
    # Create test input
    test_input = torch.randn(1, seq_len, window_size, in_channels).to(device)
    
    # Get PyTorch output
    original_model.eval()
    with torch.no_grad():
        pytorch_scenario, pytorch_action = original_model(test_input)
    
    # Get ONNX output
    try:
        import onnxruntime as ort
        ort_session = ort.InferenceSession(onnx_path)
        
        # Convert input to numpy
        input_numpy = test_input.cpu().numpy()
        
        # Run inference
        outputs = ort_session.run(None, {'input': input_numpy})
        onnx_scenario = torch.tensor(outputs[0])
        onnx_action = torch.tensor(outputs[1])
        
        # Compare outputs
        scenario_diff = torch.abs(pytorch_scenario - onnx_scenario).max().item()
        action_diff = torch.abs(pytorch_action - onnx_action).max().item()
        
        print(f"   Scenario logits max diff: {scenario_diff:.6f}")
        print(f"   Action logits max diff: {action_diff:.6f}")
        
        if scenario_diff < 1e-3 and action_diff < 1e-3:
            print("✓ Model validation passed (differences < 1e-3)")
        else:
            print("⚠️  Model validation warning: differences may be significant")
            
    except ImportError:
        print("⚠️  onnxruntime not installed, skipping validation")
    except Exception as e:
        print(f"⚠️  Validation error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Export model for deployment')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--output-dir', type=str, default='deployment',
                        help='Output directory for exported models and stats')
    parser.add_argument('--processed-dir', type=str, default='data/processed_ego4d',
                        help='Directory containing processed IMU data')
    parser.add_argument('--scenario-labels', type=str, default='data/labels/scenario_labels.csv',
                        help='Path to scenario labels CSV')
    parser.add_argument('--action-labels', type=str, default='data/labels/action_labels_llm_validated.csv',
                        help='Path to action labels CSV')
    parser.add_argument('--export-format', type=str, default='torchscript', choices=['onnx', 'torchscript', 'both'],
                        help='Export format: onnx, torchscript, or both (default: torchscript, recommended for TransformerEncoderLayer)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., cuda:0, cpu). Default: auto-detect')
    
    args = parser.parse_args()
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Get model configuration
    seq_len = CONFIG['hla']['seq_len']
    window_size = CONFIG['lle']['window_size']
    in_channels = CONFIG['lle']['in_channels']
    
    print(f"\nModel configuration:")
    print(f"   Sequence length: {seq_len}")
    print(f"   Window size: {window_size}")
    print(f"   Input channels: {in_channels}")
    
    # Export models
    if args.export_format in ['onnx', 'both']:
        onnx_path = output_dir / 'model.onnx'
        export_to_onnx(model, onnx_path, device, seq_len, window_size, in_channels)
    
    if args.export_format in ['torchscript', 'both']:
        torchscript_path = output_dir / 'model.pt'
        export_to_torchscript(model, torchscript_path, device, seq_len, window_size, in_channels)
    
    # Load dataset to extract normalization stats
    print(f"\nLoading dataset to extract normalization statistics...")
    import pandas as pd
    scenario_df = pd.read_csv(args.scenario_labels)
    train_uids = scenario_df[scenario_df['split'] == 'train']['video_uid'].tolist()[:100]  # Use subset for speed
    
    # Check for processed data availability
    processed_dir = Path(args.processed_dir)
    available_uids = set([p.parent.name for p in processed_dir.glob('*/seq.npz')])
    train_uids_available = [u for u in train_uids if u in available_uids]
    
    if len(train_uids_available) == 0:
        print("⚠️  No training videos with processed data found. Using test set...")
        test_uids = scenario_df[scenario_df['split'] == 'test']['video_uid'].tolist()[:50]
        train_uids_available = [u for u in test_uids if u in available_uids]
    
    if len(train_uids_available) == 0:
        print("❌ ERROR: No videos with processed data found!")
        return
    
    print(f"Using {len(train_uids_available)} videos for normalization stats")
    
    # Create dataset (this will compute normalization stats)
    dataset = HierarchicalDataset(
        train_uids_available,
        args.processed_dir,
        args.scenario_labels,
        args.action_labels,
        use_cache=False  # Force recomputation to get stats
    )
    
    # Save normalization stats
    stats_path = output_dir / 'normalization_stats.json'
    save_normalization_stats(dataset, stats_path, CONFIG)
    
    # Validate exported model if ONNX was exported
    if args.export_format in ['onnx', 'both']:
        validate_exported_model(model, output_dir / 'model.onnx', device, seq_len, window_size, in_channels)
    
    print(f"\n✅ Export complete! Files saved to: {output_dir}")
    print(f"   - Model: {output_dir / 'model.onnx' if args.export_format in ['onnx', 'both'] else output_dir / 'model.pt'}")
    print(f"   - Normalization stats: {stats_path}")


if __name__ == "__main__":
    main()

