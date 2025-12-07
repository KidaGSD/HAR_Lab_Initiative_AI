#!/usr/bin/env python3
"""
Diagnostic script to investigate why Fold 4 performs exceptionally well.
Checks for data leakage, scenario imbalances, and distribution issues.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from collections import Counter

def diagnose_fold4(n_folds=4, processed_dir="data/processed_ego4d"):
    """Comprehensive diagnosis of Fold 4 performance anomaly"""
    
    print("=" * 80)
    print("FOLD 4 PERFORMANCE ANOMALY DIAGNOSIS")
    print("=" * 80)
    
    # Load scenario labels
    scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
    usable_df = scenario_df[scenario_df['split'].isin(['train', 'val'])].copy()
    
    # ===== CHECK 1: Duplicate UIDs =====
    print("\n" + "=" * 80)
    print("CHECK 1: DUPLICATE VIDEO UIDs")
    print("=" * 80)
    duplicates = usable_df[usable_df.duplicated(subset=['video_uid'], keep=False)]
    if len(duplicates) > 0:
        print(f"⚠️  CRITICAL: Found {len(duplicates)} rows with duplicate video_uids!")
        print(f"   Unique duplicate video_uids: {duplicates['video_uid'].nunique()}")
        print("\n   Duplicate entries:")
        print(duplicates[['video_uid', 'scenario', 'split']].to_string(index=False))
        print("\n   This WILL cause data leakage!")
        usable_df = usable_df.drop_duplicates(subset=['video_uid'], keep='first')
        print(f"   After deduplication: {len(usable_df)} videos")
    else:
        print("✓  No duplicate video_uids found")
    
    # ===== CHECK 2: Create folds and check for leakage =====
    print("\n" + "=" * 80)
    print("CHECK 2: FOLD CREATION AND DATA LEAKAGE")
    print("=" * 80)
    
    video_scenarios = usable_df[['video_uid', 'scenario']].copy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_stats = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(video_scenarios['video_uid'], video_scenarios['scenario'])):
        fold_train_uids = set(video_scenarios.iloc[train_idx]['video_uid'].tolist())
        fold_val_uids = set(video_scenarios.iloc[val_idx]['video_uid'].tolist())
        
        # Check for overlap (data leakage)
        overlap = fold_train_uids & fold_val_uids
        if overlap:
            print(f"⚠️  CRITICAL: Fold {fold_idx+1} has {len(overlap)} overlapping UIDs between train and val!")
            print(f"   Overlapping UIDs: {list(overlap)[:10]}")
            print(f"   This is DATA LEAKAGE - model saw validation data during training!")
        else:
            print(f"✓  Fold {fold_idx+1}: No overlap between train/val UIDs")
        
        # ADD THIS: Check if Fold 4 specifically has any overlap
        if fold_idx == 3:  # Fold 4 (0-indexed)
            print(f"\n🔍 DETAILED FOLD 4 CHECK:")
            print(f"   Train UIDs: {len(fold_train_uids)}")
            print(f"   Val UIDs: {len(fold_val_uids)}")
            print(f"   Overlap: {len(overlap)}")
            if overlap:
                print(f"   ⚠️  FOLD 4 HAS {len(overlap)} OVERLAPPING VIDEOS!")
                print(f"   This explains the 92% accuracy - model trained on validation data!")
        
        # Get scenario distributions
        train_scenarios = scenario_df[scenario_df['video_uid'].isin(fold_train_uids)]['scenario'].value_counts()
        val_scenarios = scenario_df[scenario_df['video_uid'].isin(fold_val_uids)]['scenario'].value_counts()
        
        fold_stats.append({
            'fold': fold_idx + 1,
            'train_uids': fold_train_uids,
            'val_uids': fold_val_uids,
            'train_scenarios': train_scenarios.to_dict(),
            'val_scenarios': val_scenarios.to_dict(),
            'overlap': overlap
        })
    
    # ===== CHECK 3: Scenario distribution in Fold 4 vs others =====
    print("\n" + "=" * 80)
    print("CHECK 3: SCENARIO DISTRIBUTION ANALYSIS (Focus on Fold 4)")
    print("=" * 80)
    
    all_scenarios = sorted(scenario_df['scenario'].unique())
    
    print("\nValidation Set Scenario Distribution:")
    print(f"{'Scenario':<25}", end="")
    for i in range(n_folds):
        print(f"{'Fold' + str(i+1):<15}", end="")
    print()
    print("-" * (25 + 15 * n_folds))
    
    for scenario in all_scenarios:
        print(f"{scenario:<25}", end="")
        counts = []
        for stat in fold_stats:
            count = stat['val_scenarios'].get(scenario, 0)
            counts.append(count)
            marker = " ⭐" if stat['fold'] == 4 and count > 0 else ""
            print(f"{count:<15}", end="")
        print()
    
    # ===== CHECK 4: Fold 4 specific analysis =====
    print("\n" + "=" * 80)
    print("CHECK 4: FOLD 4 SPECIFIC ANALYSIS")
    print("=" * 80)
    
    fold4_stat = fold_stats[3]  # Fold 4 (0-indexed)
    other_folds = fold_stats[:3]  # Folds 1-3
    
    print(f"\nFold 4 Validation Set:")
    print(f"  Total videos: {len(fold4_stat['val_uids'])}")
    print(f"  Scenarios: {dict(sorted(fold4_stat['val_scenarios'].items(), key=lambda x: -x[1]))}")
    
    # Calculate scenario diversity (entropy)
    def calculate_entropy(scenario_dict):
        total = sum(scenario_dict.values())
        if total == 0:
            return 0
        probs = [v / total for v in scenario_dict.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)
    
    fold4_entropy = calculate_entropy(fold4_stat['val_scenarios'])
    other_entropies = [calculate_entropy(s['val_scenarios']) for s in other_folds]
    
    print(f"\nScenario Diversity (Entropy):")
    print(f"  Fold 4: {fold4_entropy:.3f}")
    print(f"  Folds 1-3: {np.mean(other_entropies):.3f} ± {np.std(other_entropies):.3f}")
    
    if fold4_entropy > np.mean(other_entropies) + 0.5:
        print("  ⚠️  Fold 4 has significantly MORE diverse scenarios (might be easier to learn)")
    elif fold4_entropy < np.mean(other_entropies) - 0.5:
        print("  ⚠️  Fold 4 has significantly LESS diverse scenarios (might be dominated by easy classes)")
    
    # Check if Fold 4 has more balanced distribution
    fold4_val_counts = list(fold4_stat['val_scenarios'].values())
    if len(fold4_val_counts) > 0:
        fold4_balance = np.std(fold4_val_counts) / np.mean(fold4_val_counts) if np.mean(fold4_val_counts) > 0 else 0
        other_balances = []
        for s in other_folds:
            counts = list(s['val_scenarios'].values())
            if len(counts) > 0:
                balance = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
                other_balances.append(balance)
        
        print(f"\nScenario Balance (CV of counts, lower = more balanced):")
        print(f"  Fold 4: {fold4_balance:.3f}")
        print(f"  Folds 1-3: {np.mean(other_balances):.3f} ± {np.std(other_balances):.3f}")
        
        if fold4_balance < np.mean(other_balances) - 0.2:
            print("  ⚠️  Fold 4 has MORE BALANCED scenario distribution (easier to predict)")
    
    # ===== CHECK 5: Check if Fold 4 validation set overlaps with original train split =====
    print("\n" + "=" * 80)
    print("CHECK 5: ORIGINAL SPLIT OVERLAP")
    print("=" * 80)
    
    original_train_uids = set(scenario_df[scenario_df['split'] == 'train']['video_uid'].tolist())
    original_val_uids = set(scenario_df[scenario_df['split'] == 'val']['video_uid'].tolist())
    
    for i, stat in enumerate(fold_stats):
        fold_num = i + 1
        fold_val_overlap_train = len(stat['val_uids'] & original_train_uids)
        fold_val_overlap_val = len(stat['val_uids'] & original_val_uids)
        
        print(f"\nFold {fold_num} validation set:")
        print(f"  Overlaps with original 'train' split: {fold_val_overlap_train}/{len(stat['val_uids'])} ({fold_val_overlap_train/len(stat['val_uids'])*100:.1f}%)")
        print(f"  Overlaps with original 'val' split: {fold_val_overlap_val}/{len(stat['val_uids'])} ({fold_val_overlap_val/len(stat['val_uids'])*100:.1f}%)")
        
        if fold_num == 4:
            if fold_val_overlap_train / len(stat['val_uids']) > 0.8:
                print("  ⚠️  Fold 4 validation is mostly from original TRAIN split (might be easier)")
            elif fold_val_overlap_val / len(stat['val_uids']) > 0.8:
                print("  ⚠️  Fold 4 validation is mostly from original VAL split")
    
    # ===== CHECK 6: Sample counts per fold =====
    print("\n" + "=" * 80)
    print("CHECK 6: SAMPLE COUNTS PER FOLD")
    print("=" * 80)
    
    processed_path = Path(processed_dir)
    for stat in fold_stats:
        val_samples = 0
        for uid in stat['val_uids']:
            seq_path = processed_path / uid / 'seq.npz'
            if seq_path.exists():
                try:
                    data = np.load(seq_path)
                    val_samples += len(data['traj'])
                except:
                    pass
        
        marker = " ⭐" if stat['fold'] == 4 else ""
        print(f"Fold {stat['fold']}: {val_samples:,} validation samples{marker}")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)
    
    issues_found = []
    
    if len(duplicates) > 0:
        issues_found.append("❌ Duplicate video UIDs found (data leakage risk)")
    
    for stat in fold_stats:
        if stat['overlap']:
            issues_found.append(f"❌ Fold {stat['fold']} has train/val UID overlap")
    
    if fold4_entropy < np.mean(other_entropies) - 0.5:
        issues_found.append("⚠️  Fold 4 has less diverse scenarios (might be dominated by easy classes)")
    elif fold4_balance < np.mean(other_balances) - 0.2:
        issues_found.append("⚠️  Fold 4 has more balanced scenario distribution (easier to predict)")
    
    if issues_found:
        print("\nPotential issues identified:")
        for issue in issues_found:
            print(f"  {issue}")
    else:
        print("\n✓ No obvious data leakage issues found.")
        print("  The performance difference might be due to:")
        print("  - Natural variation in CV folds")
        print("  - Easier scenario distribution in Fold 4")
        print("  - Random initialization effects")
    
    return fold_stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=4)
    parser.add_argument('--processed-dir', type=str, default='data/processed_ego4d')
    args = parser.parse_args()
    
    stats = diagnose_fold4(args.n_folds, args.processed_dir)
