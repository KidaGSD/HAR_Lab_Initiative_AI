# analyze_cv_folds.py
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

def analyze_cv_folds(n_folds=4, processed_dir="data/processed_ego4d"):
    """Analyze CV fold composition to understand performance differences"""
    
    # Load scenario labels
    scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
    usable_df = scenario_df[scenario_df['split'].isin(['train', 'val'])].copy()
    
    # Check for duplicate video_uids
    print("=" * 80)
    print("CHECKING FOR DUPLICATE VIDEO UIDs")
    print("=" * 80)
    duplicates = usable_df[usable_df.duplicated(subset=['video_uid'], keep=False)]
    if len(duplicates) > 0:
        print(f"⚠️  WARNING: Found {len(duplicates)} rows with duplicate video_uids!")
        print(f"   Unique duplicate video_uids: {duplicates['video_uid'].nunique()}")
        print("\n   First 10 duplicate entries:")
        print(duplicates[['video_uid', 'scenario', 'split']].head(10).to_string(index=False))
        print("\n   This could cause data leakage in cross-validation!")
        print("   Dropping duplicates (keeping first occurrence)...")
        usable_df = usable_df.drop_duplicates(subset=['video_uid'], keep='first')
        print(f"   After deduplication: {len(usable_df)} videos (removed {len(scenario_df[scenario_df['split'].isin(['train', 'val'])]) - len(usable_df)} duplicates)")
    else:
        print("✓  No duplicate video_uids found")
    
    print("\n" + "=" * 80)
    print(f"ANALYZING {n_folds}-FOLD CROSS VALIDATION COMPOSITION")
    print("=" * 80)
    
    # Create stratified folds (same as training)
    video_scenarios = usable_df[['video_uid', 'scenario']].copy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_stats = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(video_scenarios['video_uid'], video_scenarios['scenario'])):
        fold_train_uids = video_scenarios.iloc[train_idx]['video_uid'].tolist()
        fold_val_uids = video_scenarios.iloc[val_idx]['video_uid'].tolist()
        
        # Get scenario distributions
        train_scenarios = scenario_df[scenario_df['video_uid'].isin(fold_train_uids)]['scenario'].value_counts()
        val_scenarios = scenario_df[scenario_df['video_uid'].isin(fold_val_uids)]['scenario'].value_counts()
        
        # Count samples (windows) per fold
        train_samples = 0
        val_samples = 0
        
        processed_path = Path(processed_dir)
        for uid in fold_train_uids:
            seq_path = processed_path / uid / 'seq.npz'
            if seq_path.exists():
                try:
                    data = np.load(seq_path)
                    train_samples += len(data['traj'])
                except:
                    pass
        
        for uid in fold_val_uids:
            seq_path = processed_path / uid / 'seq.npz'
            if seq_path.exists():
                try:
                    data = np.load(seq_path)
                    val_samples += len(data['traj'])
                except:
                    pass
        
        fold_stats.append({
            'fold': fold_idx + 1,
            'train_videos': len(fold_train_uids),
            'val_videos': len(fold_val_uids),
            'train_samples': train_samples,
            'val_samples': val_samples,
            'train_scenarios': train_scenarios.to_dict(),
            'val_scenarios': val_scenarios.to_dict(),
            'train_uids': fold_train_uids,
            'val_uids': fold_val_uids
        })
    
    # Print summary
    print("\n" + "=" * 80)
    print("FOLD SUMMARY")
    print("=" * 80)
    print(f"{'Fold':<6} {'Train Videos':<15} {'Val Videos':<15} {'Train Samples':<15} {'Val Samples':<15}")
    print("-" * 80)
    for stat in fold_stats:
        print(f"{stat['fold']:<6} {stat['train_videos']:<15} {stat['val_videos']:<15} {stat['train_samples']:<15} {stat['val_samples']:<15}")
    
    # Scenario distribution comparison
    print("\n" + "=" * 80)
    print("SCENARIO DISTRIBUTION PER FOLD (Validation Set)")
    print("=" * 80)
    
    all_scenarios = sorted(scenario_df['scenario'].unique())
    print(f"\n{'Scenario':<25}", end="")
    for i in range(n_folds):
        print(f"{'Fold' + str(i+1):<12}", end="")
    print()
    print("-" * (25 + 12 * n_folds))
    
    for scenario in all_scenarios:
        print(f"{scenario:<25}", end="")
        for stat in fold_stats:
            count = stat['val_scenarios'].get(scenario, 0)
            print(f"{count:<12}", end="")
        print()
    
    # Find the "best" fold (highest val samples - might correlate with performance)
    print("\n" + "=" * 80)
    print("FOLD ANALYSIS")
    print("=" * 80)
    
    # Check for imbalances
    val_sample_counts = [s['val_samples'] for s in fold_stats]
    train_sample_counts = [s['train_samples'] for s in fold_stats]
    
    print(f"\nValidation sample counts: {val_sample_counts}")
    print(f"Mean: {np.mean(val_sample_counts):.0f}, Std: {np.std(val_sample_counts):.0f}")
    print(f"Range: {min(val_sample_counts)} - {max(val_sample_counts)}")
    
    print(f"\nTraining sample counts: {train_sample_counts}")
    print(f"Mean: {np.mean(train_sample_counts):.0f}, Std: {np.std(train_sample_counts):.0f}")
    
    # Check scenario balance
    print("\n" + "=" * 80)
    print("SCENARIO BALANCE CHECK (Validation)")
    print("=" * 80)
    
    for scenario in all_scenarios:
        counts = [s['val_scenarios'].get(scenario, 0) for s in fold_stats]
        if max(counts) > 0:
            imbalance_ratio = max(counts) / (min([c for c in counts if c > 0]) or 1)
            if imbalance_ratio > 2.0:
                print(f"⚠️  {scenario}: {counts} (imbalance ratio: {imbalance_ratio:.2f}x)")
            else:
                print(f"✓   {scenario}: {counts}")
    
    # Detailed fold comparison
    print("\n" + "=" * 80)
    print("DETAILED FOLD COMPARISON")
    print("=" * 80)
    
    for i, stat in enumerate(fold_stats):
        print(f"\nFold {stat['fold']}:")
        print(f"  Train: {stat['train_videos']} videos, {stat['train_samples']} samples")
        print(f"  Val:   {stat['val_videos']} videos, {stat['val_samples']} samples")
        print(f"  Val scenario breakdown:")
        for scenario, count in sorted(stat['val_scenarios'].items(), key=lambda x: -x[1]):
            print(f"    {scenario}: {count} videos")
    
    return fold_stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=4)
    parser.add_argument('--processed-dir', type=str, default='data/processed_ego4d')
    args = parser.parse_args()
    
    stats = analyze_cv_folds(args.n_folds, args.processed_dir)
    
    # Save to CSV for further analysis
    summary_df = pd.DataFrame([
        {
            'fold': s['fold'],
            'train_videos': s['train_videos'],
            'val_videos': s['val_videos'],
            'train_samples': s['train_samples'],
            'val_samples': s['val_samples']
        }
        for s in stats
    ])
    summary_df.to_csv('cv_fold_analysis.csv', index=False)
    print(f"\n✅ Saved summary to cv_fold_analysis.csv")