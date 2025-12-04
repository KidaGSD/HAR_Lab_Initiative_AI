# scripts/check_data_quality.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Import the dataset class
sys.path.append(str(Path(__file__).parent))
from train_hierarchical import HierarchicalDataset, CONFIG

def check_data_quality(processed_dir, scenario_labels_path, action_labels_path, sample_uids=None):
    """Comprehensive data quality check"""
    
    print("="*80)
    print("DATA QUALITY DIAGNOSTICS")
    print("="*80)
    
    # Load labels
    scenario_df = pd.read_csv(scenario_labels_path).set_index('video_uid')
    action_df = pd.read_csv(action_labels_path)
    
    # Get UIDs
    if sample_uids is None:
        # Check first 100 UIDs or all if less
        all_uids = [p.name for p in Path(processed_dir).iterdir() if p.is_dir()]
        sample_uids = all_uids[:100]
    
    print(f"\nChecking {len(sample_uids)} videos...")
    
    # === 1. Check raw .npz files ===
    print("\n" + "="*80)
    print("1. CHECKING RAW .NPZ FILES")
    print("="*80)
    
    issues = {
        'missing_files': [],
        'nan_in_data': [],
        'inf_in_data': [],
        'zero_std_channels': [],
        'constant_channels': [],
        'extreme_values': [],
        'shape_mismatches': []
    }
    
    all_trajs = []
    valid_uids = []
    
    for uid in tqdm(sample_uids, desc="Scanning files"):
        seq_path = Path(processed_dir) / uid / 'seq.npz'
        
        if not seq_path.exists():
            issues['missing_files'].append(uid)
            continue
        
        try:
            data = np.load(seq_path)
            traj = data['traj']  # (N, 50, 6)
            timestamps = data.get('timestamp', None)
            
            # Check for NaN/Inf
            if np.isnan(traj).any():
                issues['nan_in_data'].append((uid, np.isnan(traj).sum()))
            
            if np.isinf(traj).any():
                issues['inf_in_data'].append((uid, np.isinf(traj).sum()))
            
            # Check for extreme values
            if traj.max() > 1e6 or traj.min() < -1e6:
                issues['extreme_values'].append((uid, traj.min(), traj.max()))
            
            # Check shape
            if len(traj.shape) != 3 or traj.shape[1] != 50 or traj.shape[2] != 6:
                issues['shape_mismatches'].append((uid, traj.shape))
            
            # Check for constant channels (would cause zero std)
            for c in range(traj.shape[2]):
                channel_data = traj[:, :, c]
                if channel_data.std() == 0:
                    issues['constant_channels'].append((uid, c))
            
            if len(traj) > 0:
                all_trajs.append(traj)
                valid_uids.append(uid)
                
        except Exception as e:
            print(f"\n⚠️ Error loading {uid}: {e}")
            continue
    
    # Print issues
    print(f"\n📊 SUMMARY:")
    print(f"  Valid files: {len(valid_uids)}/{len(sample_uids)}")
    print(f"  Missing files: {len(issues['missing_files'])}")
    print(f"  Files with NaN: {len(issues['nan_in_data'])}")
    print(f"  Files with Inf: {len(issues['inf_in_data'])}")
    print(f"  Files with extreme values: {len(issues['extreme_values'])}")
    print(f"  Files with constant channels: {len(issues['constant_channels'])}")
    
    if issues['nan_in_data']:
        print(f"\n⚠️ NaN found in:")
        for uid, count in issues['nan_in_data'][:5]:
            print(f"    {uid}: {count} NaN values")
    
    if issues['inf_in_data']:
        print(f"\n⚠️ Inf found in:")
        for uid, count in issues['inf_in_data'][:5]:
            print(f"    {uid}: {count} Inf values")
    
    if issues['constant_channels']:
        print(f"\n⚠️ Constant channels (zero std):")
        for uid, channel in issues['constant_channels'][:10]:
            print(f"    {uid}: channel {channel}")
    
    # === 2. Check normalization statistics ===
    print("\n" + "="*80)
    print("2. CHECKING NORMALIZATION STATISTICS")
    print("="*80)
    
    if len(all_trajs) == 0:
        print("❌ No valid trajectories found!")
        return
    
    # Simulate the augmentation
    all_data_augmented = []
    for traj in all_trajs:
        accel = traj[..., :3]
        gyro = traj[..., 3:6]
        accel_norm = np.linalg.norm(accel, axis=2, keepdims=True)
        gyro_norm = np.linalg.norm(gyro, axis=2, keepdims=True)
        traj_aug = np.concatenate([traj, accel_norm, gyro_norm], axis=2)
        all_data_augmented.append(traj_aug)
    
    all_data = np.concatenate(all_data_augmented, axis=0)  # (Total_Windows, 50, C)
    
    global_mean = all_data.mean(axis=(0, 1))  # (C,)
    global_std = all_data.std(axis=(0, 1))  # (C,)
    
    print(f"\nGlobal statistics (after augmentation, {all_data.shape[0]} windows):")
    print(f"  Shape: {all_data.shape}")
    print(f"  Mean per channel: {global_mean}")
    print(f"  Std per channel:  {global_std}")
    print(f"  Min: {all_data.min():.6f}, Max: {all_data.max():.6f}")
    
    # Check for zero std
    zero_std_mask = global_std < 1e-6
    if zero_std_mask.any():
        print(f"\n⚠️ WARNING: Channels with near-zero std:")
        for c in np.where(zero_std_mask)[0]:
            print(f"    Channel {c}: std={global_std[c]:.2e}, mean={global_mean[c]:.6f}")
    
    # Check for NaN/Inf in stats
    if np.isnan(global_mean).any() or np.isnan(global_std).any():
        print(f"\n❌ ERROR: NaN in normalization statistics!")
        print(f"  Mean has NaN: {np.isnan(global_mean).any()}")
        print(f"  Std has NaN: {np.isnan(global_std).any()}")
    
    if np.isinf(global_mean).any() or np.isinf(global_std).any():
        print(f"\n❌ ERROR: Inf in normalization statistics!")
        print(f"  Mean has Inf: {np.isinf(global_mean).any()}")
        print(f"  Std has Inf: {np.isinf(global_std).any()}")
    
    # === 3. Check normalized data ===
    print("\n" + "="*80)
    print("3. CHECKING NORMALIZED DATA (Sample)")
    print("="*80)
    
    # Test normalization on a few videos
    test_uids = valid_uids[:5]
    normalized_samples = []
    
    for uid in test_uids:
        seq_path = Path(processed_dir) / uid / 'seq.npz'
        data = np.load(seq_path)
        traj = data['traj']
        
        # Augment
        accel = traj[..., :3]
        gyro = traj[..., 3:6]
        accel_norm = np.linalg.norm(accel, axis=2, keepdims=True)
        gyro_norm = np.linalg.norm(gyro, axis=2, keepdims=True)
        traj_aug = np.concatenate([traj, accel_norm, gyro_norm], axis=2)
        
        # Normalize
        traj_norm = (traj_aug - global_mean) / (global_std + 1e-6)
        
        # Per-video centering
        if CONFIG['data'].get('per_video_center', True):
            traj_norm = traj_norm - traj_norm.mean(axis=(0, 1), keepdims=True)
        
        # Check for issues
        if np.isnan(traj_norm).any():
            print(f"⚠️ {uid}: NaN after normalization!")
        if np.isinf(traj_norm).any():
            print(f"⚠️ {uid}: Inf after normalization!")
        if traj_norm.max() > 100 or traj_norm.min() < -100:
            print(f"⚠️ {uid}: Extreme normalized values: [{traj_norm.min():.2f}, {traj_norm.max():.2f}]")
        
        normalized_samples.append(traj_norm)
    
    if normalized_samples:
        all_norm = np.concatenate(normalized_samples, axis=0)
        print(f"\nNormalized data stats (sample):")
        print(f"  Shape: {all_norm.shape}")
        print(f"  Mean: {all_norm.mean():.6f}, Std: {all_norm.std():.6f}")
        print(f"  Min: {all_norm.min():.6f}, Max: {all_norm.max():.6f}")
        print(f"  Has NaN: {np.isnan(all_norm).any()}")
        print(f"  Has Inf: {np.isinf(all_norm).any()}")
    
    # === 4. Check labels ===
    print("\n" + "="*80)
    print("4. CHECKING LABELS")
    print("="*80)
    
    scenario_map = {name: i for i, name in enumerate(sorted(scenario_df['scenario'].unique()))}
    print(f"Scenarios: {scenario_map}")
    
    # Check scenario labels
    scenario_counts = scenario_df['scenario'].value_counts()
    print(f"\nScenario distribution:")
    for scenario, count in scenario_counts.items():
        print(f"  {scenario}: {count}")
    
    # Check for videos with no scenario label
    missing_scenario = [uid for uid in valid_uids if uid not in scenario_df.index]
    if missing_scenario:
        print(f"\n⚠️ {len(missing_scenario)} videos missing scenario labels")
    
    # Check action labels
    action_map = {
        'Stationary': 0, 'Locomotion': 1, 'Essential Operation': 2,
        'Object Transfer': 3, 'Search': 4, 'Error / Correction': 5
    }
    action_counts = action_df['action'].value_counts()
    print(f"\nAction distribution:")
    for action, count in action_counts.items():
        print(f"  {action}: {count}")
    
    # === 5. Test dataset loading ===
    print("\n" + "="*80)
    print("5. TESTING DATASET LOADING")
    print("="*80)
    
    try:
        test_uids = valid_uids[:10]
        test_ds = HierarchicalDataset(
            test_uids,
            processed_dir,
            scenario_labels_path,
            action_labels_path
        )
        
        print(f"\n✅ Dataset created successfully!")
        print(f"  Samples: {len(test_ds)}")
        
        # Check a few samples
        for i in range(min(5, len(test_ds))):
            sample = test_ds[i]
            inputs = sample['inputs']
            scenario_label = sample['scenario_label']
            action_labels = sample['action_labels']
            
            # Check for NaN/Inf
            if torch.isnan(inputs).any():
                print(f"⚠️ Sample {i}: NaN in inputs!")
            if torch.isinf(inputs).any():
                print(f"⚠️ Sample {i}: Inf in inputs!")
            
            # Check label validity
            if scenario_label < 0 or scenario_label >= len(scenario_map):
                print(f"⚠️ Sample {i}: Invalid scenario label: {scenario_label}")
            
            if (action_labels < -1).any() or (action_labels >= 6).any():
                invalid_mask = (action_labels < -1) | (action_labels >= 6)
                print(f"⚠️ Sample {i}: Invalid action labels: {action_labels[invalid_mask]}")
            
            print(f"  Sample {i}: inputs shape={inputs.shape}, scenario={scenario_label}, "
                  f"inputs range=[{inputs.min():.4f}, {inputs.max():.4f}], "
                  f"has NaN={torch.isnan(inputs).any()}")
        
    except Exception as e:
        print(f"\n❌ ERROR creating dataset: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)

# In check_data_quality.py, replace lines 290-309 with:

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=str, default="data/processed_ego4d")  # Fixed default
    parser.add_argument("--scenario-labels", type=str, default="data/labels/scenario_labels.csv")
    parser.add_argument("--action-labels", type=str, default="data/labels/action_labels_llm_validated.csv")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of videos to check")
    
    args = parser.parse_args()
    
    # Check if directory exists
    processed_path = Path(args.processed_dir)
    if not processed_path.exists():
        print(f"❌ ERROR: Directory does not exist: {args.processed_dir}")
        print(f"\nTrying to find processed data...")
        # Try alternative paths
        alternatives = [
            "data/processed",
            "data/processed_ego4d",
            "../data/processed_ego4d"
        ]
        for alt in alternatives:
            alt_path = Path(alt)
            if alt_path.exists():
                print(f"✅ Found: {alt_path}")
                args.processed_dir = str(alt_path)
                processed_path = alt_path
                break
        else:
            print("❌ Could not find processed data directory.")
            print("Please specify the correct path with --processed-dir")
            sys.exit(1)
    
    # Get sample UIDs - handle both directory structure and glob pattern
    if processed_path.is_dir():
        all_uids = [p.name for p in processed_path.iterdir() if p.is_dir()]
    else:
        # Try glob pattern if it's a file pattern
        all_uids = [p.parent.name for p in processed_path.parent.glob('*/seq.npz')]
    
    if len(all_uids) == 0:
        print(f"❌ ERROR: No UIDs found in {args.processed_dir}")
        print("Please check that the directory contains subdirectories with seq.npz files")
        sys.exit(1)
    
    print(f"Found {len(all_uids)} UIDs in {args.processed_dir}")
    sample_uids = all_uids[:args.sample_size]
    
    check_data_quality(
        args.processed_dir,
        args.scenario_labels,
        args.action_labels,
        sample_uids
    )