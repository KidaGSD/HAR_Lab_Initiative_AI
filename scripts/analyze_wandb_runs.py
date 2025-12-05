#!/usr/bin/env python3
"""
WandB Run Analysis Script
Analyzes training runs for the Hierarchical IMU Activity Recognition project.

Usage:
    python scripts/analyze_wandb_runs.py --run1 0cj4mzg2 --run2 m3z6vejh
    python scripts/analyze_wandb_runs.py --run1 0cj4mzg2  # Single run analysis
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import StratifiedKFold


def get_run_metrics(api, run_id, project="wandbleo/har-imu-training"):
    """Fetch run history and config from WandB."""
    run = api.run(f"{project}/{run_id}")
    history = run.history()

    # Filter only training metrics rows (exclude confusion matrix log rows)
    metrics = history[history['epoch'].notna()][
        ['epoch', 'train_loss', 'val_scenario_f1', 'val_scenario_acc', 'learning_rate']
    ].reset_index(drop=True)

    return run, metrics


def get_confusion_matrix(api, run_id, project="wandbleo/har-imu-training"):
    """Download and parse the latest confusion matrix from a run."""
    run = api.run(f"{project}/{run_id}")

    # Find the latest confusion matrix file
    files = list(run.files())
    conf_files = [f for f in files if 'conf_mat' in f.name and f.name.endswith('.json')]

    if not conf_files:
        print(f"No confusion matrix files found for run {run_id}")
        return None, None

    # Sort by step number in filename and get latest
    def extract_step(fname):
        parts = fname.name.split('_')
        for p in parts:
            if p.isdigit():
                return int(p)
        return 0

    conf_files.sort(key=extract_step, reverse=True)
    latest_file = conf_files[0]

    # Download
    latest_file.download(replace=True)

    with open(latest_file.name, 'r') as f:
        data = json.load(f)

    rows = data['data']
    classes = sorted(set([r[0] for r in rows]))
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Build confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for row in rows:
        true_label, pred_label, count = row[0], row[1], row[2]
        i, j = class_to_idx[true_label], class_to_idx[pred_label]
        cm[i, j] = count

    return cm, classes


def compute_class_metrics(cm, classes):
    """Compute per-class precision, recall, F1, and support."""
    n_classes = len(classes)
    metrics = []

    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = cm[i, :].sum()

        metrics.append({
            'class': c,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        })

    return pd.DataFrame(metrics)


def print_confusion_matrix(cm, classes):
    """Pretty print confusion matrix."""
    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")

    # Header
    short_names = [c[:10] for c in classes]
    header = '            ' + ' '.join([f'{n:>10}' for n in short_names])
    print(header)

    for i, c in enumerate(classes):
        row_str = f'{short_names[i]:>10}: ' + ' '.join([f'{cm[i,j]:>10}' for j in range(len(classes))])
        print(row_str)


def analyze_confusions(cm, classes, top_k=5):
    """Find top confusion pairs."""
    n_classes = len(classes)
    confusions = []

    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                total = cm[i, :].sum()
                pct = cm[i, j] / total * 100
                confusions.append({
                    'true': classes[i],
                    'pred': classes[j],
                    'count': cm[i, j],
                    'pct_of_true': pct
                })

    confusions.sort(key=lambda x: -x['count'])
    return confusions[:top_k]


def analyze_single_run(api, run_id):
    """Full analysis of a single run."""
    print("=" * 80)
    print(f"ANALYZING RUN: {run_id}")
    print("=" * 80)

    run, metrics = get_run_metrics(api, run_id)

    print(f"\nRun Name: {run.name}")
    print(f"State: {run.state}")

    print("\n--- Training Config ---")
    print(f"  train_samples: {run.config.get('train_samples')}")
    print(f"  val_samples: {run.config.get('val_samples')}")
    print(f"  batch_size: {run.config['training']['batch_size']}")
    print(f"  lr: {run.config['training']['lr']}")
    print(f"  beta (action weight): {run.config['training']['beta']}")

    print("\n--- Training Progress ---")
    print(f"Total epochs: {len(metrics)}")
    print(f"Initial train_loss: {metrics['train_loss'].iloc[0]:.4f}")
    print(f"Final train_loss: {metrics['train_loss'].iloc[-1]:.4f}")
    print(f"Loss reduction: {(1 - metrics['train_loss'].iloc[-1]/metrics['train_loss'].iloc[0])*100:.1f}%")

    best_idx = metrics['val_scenario_f1'].idxmax()
    print(f"\nBest val_scenario_f1: {metrics['val_scenario_f1'].max():.4f} at epoch {metrics['epoch'].iloc[best_idx]:.0f}")
    print(f"Final val_scenario_f1: {metrics['val_scenario_f1'].iloc[-1]:.4f}")

    # Confusion matrix analysis
    cm, classes = get_confusion_matrix(api, run_id)
    if cm is not None:
        print_confusion_matrix(cm, classes)

        class_metrics = compute_class_metrics(cm, classes)
        print("\n--- Per-Class Metrics ---")
        print(class_metrics.to_string(index=False))

        print(f"\nMacro F1: {class_metrics['f1'].mean():.4f}")

        print("\n--- Best Performing Classes ---")
        best = class_metrics.nlargest(3, 'f1')
        for _, row in best.iterrows():
            print(f"  {row['class']}: F1={row['f1']:.3f}")

        print("\n--- Worst Performing Classes ---")
        worst = class_metrics.nsmallest(3, 'f1')
        for _, row in worst.iterrows():
            print(f"  {row['class']}: F1={row['f1']:.3f}")

        print("\n--- Top Confusions ---")
        confusions = analyze_confusions(cm, classes)
        for conf in confusions:
            print(f"  {conf['true']} -> {conf['pred']}: {conf['count']} ({conf['pct_of_true']:.1f}%)")

    return run, metrics, cm, classes


def compare_runs(api, run_id1, run_id2):
    """Compare two runs side by side."""
    print("\n" + "=" * 80)
    print("COMPARISON ANALYSIS")
    print("=" * 80)

    run1, metrics1 = get_run_metrics(api, run_id1)
    run2, metrics2 = get_run_metrics(api, run_id2)

    print(f"\n{'Metric':<30} {'Run 1':>15} {'Run 2':>15} {'Diff':>15}")
    print("-" * 75)

    best_f1_1 = metrics1['val_scenario_f1'].max()
    best_f1_2 = metrics2['val_scenario_f1'].max()
    print(f"{'Best Val F1':<30} {best_f1_1:>15.4f} {best_f1_2:>15.4f} {best_f1_2 - best_f1_1:>+15.4f}")

    final_loss_1 = metrics1['train_loss'].iloc[-1]
    final_loss_2 = metrics2['train_loss'].iloc[-1]
    print(f"{'Final Train Loss':<30} {final_loss_1:>15.4f} {final_loss_2:>15.4f} {final_loss_2 - final_loss_1:>+15.4f}")

    epochs_1 = len(metrics1)
    epochs_2 = len(metrics2)
    print(f"{'Total Epochs':<30} {epochs_1:>15} {epochs_2:>15} {epochs_2 - epochs_1:>+15}")

    # Per-class comparison
    cm1, classes = get_confusion_matrix(api, run_id1)
    cm2, _ = get_confusion_matrix(api, run_id2)

    if cm1 is not None and cm2 is not None:
        class_metrics1 = compute_class_metrics(cm1, classes)
        class_metrics2 = compute_class_metrics(cm2, classes)

        print("\n--- Per-Class F1 Comparison ---")
        print(f"{'Class':<25} {'Run 1':>10} {'Run 2':>10} {'Diff':>10}")
        print("-" * 55)

        for i, c in enumerate(classes):
            f1_1 = class_metrics1[class_metrics1['class'] == c]['f1'].values[0]
            f1_2 = class_metrics2[class_metrics2['class'] == c]['f1'].values[0]
            diff = f1_2 - f1_1
            print(f"{c:<25} {f1_1:>10.3f} {f1_2:>10.3f} {diff:>+10.3f}")


def analyze_dataset_distribution():
    """Analyze the dataset label distribution."""
    print("\n" + "=" * 80)
    print("DATASET DISTRIBUTION ANALYSIS")
    print("=" * 80)

    scenario_df = pd.read_csv('data/labels/scenario_labels.csv')

    print(f"\nTotal videos: {len(scenario_df)}")
    print("\nSplit distribution:")
    print(scenario_df['split'].value_counts().to_string())

    print("\nScenario distribution (training set):")
    train_df = scenario_df[scenario_df['split'] == 'train']
    scenario_counts = train_df['scenario'].value_counts()

    max_count = scenario_counts.max()
    min_count = scenario_counts.min()
    print(f"Imbalance ratio: {max_count / min_count:.1f}x")

    for scenario, count in scenario_counts.items():
        pct = count / len(train_df) * 100
        bar = '█' * int(count / max_count * 30)
        print(f"  {scenario:<25}: {count:>4} ({pct:>5.1f}%) {bar}")

    return scenario_df


def analyze_fold_differences():
    """Analyze differences between k-fold splits."""
    print("\n" + "=" * 80)
    print("K-FOLD SPLIT ANALYSIS")
    print("=" * 80)

    scenario_df = pd.read_csv('data/labels/scenario_labels.csv')
    usable_df = scenario_df[scenario_df['split'].isin(['train', 'val'])].copy()
    video_scenarios = usable_df[['video_uid', 'scenario']].copy()

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # Difficulty weights based on observed F1 scores
    difficulty_weights = {
        'Gardening': 10,
        'Cleaning': 5,
        'Cooking': 4,
        'Mechanical Repair': 4,
        'Walking Outdoors': 3,
        'Carpentry': 3,
        'Playing Instrument': 2,
        'Desk Work': 2
    }

    print("\nFold validation set composition:")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(video_scenarios['video_uid'], video_scenarios['scenario'])):
        fold_val = video_scenarios.iloc[val_idx]

        # Calculate difficulty score
        difficulty_score = 0
        for scenario, weight in difficulty_weights.items():
            count = len(fold_val[fold_val['scenario'] == scenario])
            difficulty_score += count * weight
        normalized_score = difficulty_score / len(fold_val)

        print(f"\nFold {fold_idx + 1}: {len(fold_val)} val videos, difficulty score: {normalized_score:.2f}")
        val_dist = fold_val['scenario'].value_counts()
        for scenario in sorted(val_dist.index):
            count = val_dist[scenario]
            print(f"    {scenario:<25}: {count:>3}")


def main():
    parser = argparse.ArgumentParser(description='Analyze WandB training runs')
    parser.add_argument('--run1', type=str, help='First run ID to analyze')
    parser.add_argument('--run2', type=str, help='Second run ID to compare (optional)')
    parser.add_argument('--project', type=str, default='wandbleo/har-imu-training',
                        help='WandB project path')
    parser.add_argument('--dataset', action='store_true', help='Analyze dataset distribution')
    parser.add_argument('--folds', action='store_true', help='Analyze k-fold differences')

    args = parser.parse_args()

    api = wandb.Api()

    if args.dataset:
        analyze_dataset_distribution()

    if args.folds:
        analyze_fold_differences()

    if args.run1:
        analyze_single_run(api, args.run1)

        if args.run2:
            analyze_single_run(api, args.run2)
            compare_runs(api, args.run1, args.run2)

    if not args.run1 and not args.dataset and not args.folds:
        parser.print_help()


if __name__ == '__main__':
    main()
