#!/usr/bin/env python3
"""
Experiment Analysis & Report Generator
Auto-collects W&B results, generates visualizations, and builds report.
Run after experiments complete or periodically during training.
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import numpy as np

# Configuration
WANDB_PROJECT = "wandbleo/har-imu-training"
OUTPUT_DIR = Path("reports")
FIGURES_DIR = OUTPUT_DIR / "figures"

def setup_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

def fetch_recent_runs(n_runs=20, filter_prefix="beta_"):
    """Fetch recent runs from W&B"""
    api = wandb.Api()
    runs = api.runs(WANDB_PROJECT, order="-created_at")
    
    results = []
    for run in runs[:n_runs]:
        if filter_prefix and filter_prefix not in run.name:
            continue
            
        config = run.config
        summary = run.summary
        
        results.append({
            "run_id": run.id,
            "name": run.name,
            "state": run.state,
            "created": run.created_at,
            "beta": config.get("training", {}).get("beta", "N/A"),
            "alpha": config.get("training", {}).get("alpha", "N/A"),
            "lr": config.get("training", {}).get("lr", "N/A"),
            "epochs_run": summary.get("epoch", "N/A"),
            "val_scenario_f1": summary.get("val_scenario_f1", None),
            "val_scenario_acc": summary.get("val_scenario_acc", None),
            "val_action_f1": summary.get("val_action_f1", None),
            "val_action_acc": summary.get("val_action_acc", None),
            "train_loss": summary.get("train_loss", None),
        })
    
    return pd.DataFrame(results)

def fetch_run_history(run_id):
    """Fetch detailed history for a single run"""
    api = wandb.Api()
    run = api.run(f"{WANDB_PROJECT}/{run_id}")
    return run.history(samples=500)

def plot_beta_comparison(df):
    """Bar chart comparing metrics across beta values"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Filter valid runs
    valid = df[df["val_scenario_f1"].notna()].copy()
    valid = valid.sort_values("beta")
    
    # Scenario F1
    ax1 = axes[0]
    bars1 = ax1.bar(valid["beta"].astype(str), valid["val_scenario_f1"], color="steelblue", edgecolor="black")
    ax1.set_xlabel("Beta Value", fontsize=12)
    ax1.set_ylabel("Scenario F1", fontsize=12)
    ax1.set_title("Scenario Classification (7 Classes)", fontsize=14)
    ax1.set_ylim(0.5, 0.7)
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        ax1.annotate(f'{bar.get_height():.3f}', 
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=10)
    
    # Action F1
    ax2 = axes[1]
    action_valid = valid[valid["val_action_f1"].notna()]
    if not action_valid.empty:
        bars2 = ax2.bar(action_valid["beta"].astype(str), action_valid["val_action_f1"], color="coral", edgecolor="black")
        ax2.set_xlabel("Beta Value", fontsize=12)
        ax2.set_ylabel("Action F1", fontsize=12)
        ax2.set_title("Action Classification (4 Classes)", fontsize=14)
        ax2.set_ylim(0.2, 0.5)
        ax2.grid(axis='y', alpha=0.3)
        for bar in bars2:
            ax2.annotate(f'{bar.get_height():.3f}', 
                         xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "beta_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'beta_comparison.png'}")

def plot_learning_curves(run_id, run_name):
    """Plot training curves for a single run"""
    history = fetch_run_history(run_id)
    if history.empty:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    if "train_loss" in history.columns:
        axes[0].plot(history["epoch"], history["train_loss"], 'b-', linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Train Loss")
        axes[0].set_title("Training Loss")
        axes[0].grid(alpha=0.3)
    
    # Scenario F1
    if "val_scenario_f1" in history.columns:
        axes[1].plot(history["epoch"], history["val_scenario_f1"], 'g-', linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Scenario F1")
        axes[1].set_title("Scenario F1")
        axes[1].grid(alpha=0.3)
    
    # Action F1
    if "val_action_f1" in history.columns:
        axes[2].plot(history["epoch"], history["val_action_f1"], 'r-', linewidth=2)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Action F1")
        axes[2].set_title("Action F1")
        axes[2].grid(alpha=0.3)
    
    plt.suptitle(f"Learning Curves: {run_name}", fontsize=14)
    plt.tight_layout()
    
    safe_name = run_name.replace("/", "_").replace(" ", "_")
    plt.savefig(FIGURES_DIR / f"curves_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / f'curves_{safe_name}.png'}")

def generate_summary_table(df):
    """Generate markdown summary table"""
    valid = df[df["val_scenario_f1"].notna()].copy()
    valid = valid.sort_values("val_scenario_f1", ascending=False)
    
    table = "| Rank | Run Name | Beta | Scenario F1 | Action F1 | Epochs |\n"
    table += "|------|----------|------|-------------|-----------|--------|\n"
    
    for i, row in valid.iterrows():
        action_f1 = f"{row['val_action_f1']:.3f}" if pd.notna(row['val_action_f1']) else "N/A"
        table += f"| {valid.index.get_loc(i)+1} | {row['name'][:30]} | {row['beta']} | {row['val_scenario_f1']:.3f} | {action_f1} | {row['epochs_run']} |\n"
    
    return table

def generate_report(df):
    """Generate full markdown report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Find best runs
    best_scenario = df.loc[df["val_scenario_f1"].idxmax()] if df["val_scenario_f1"].notna().any() else None
    best_action = df.loc[df["val_action_f1"].idxmax()] if df["val_action_f1"].notna().any() else None
    
    report = f"""# Experiment Results Report
Generated: {timestamp}

## Executive Summary

| Metric | Best Value | Best Config |
|--------|------------|-------------|
| Scenario F1 | {best_scenario['val_scenario_f1']:.3f if best_scenario is not None else 'N/A'} | beta={best_scenario['beta'] if best_scenario is not None else 'N/A'} |
| Action F1 | {best_action['val_action_f1']:.3f if best_action is not None else 'N/A'} | beta={best_action['beta'] if best_action is not None else 'N/A'} |

## Beta Value Comparison

![Beta Comparison](figures/beta_comparison.png)

## All Experiments

{generate_summary_table(df)}

## Key Findings

1. **Best Beta for Scenario**: {best_scenario['beta'] if best_scenario is not None else 'TBD'}
2. **Best Beta for Action**: {best_action['beta'] if best_action is not None else 'TBD'}
3. **Trade-off**: Higher beta improves action F1 but may affect scenario F1

## Learning Curves

See `figures/curves_*.png` for individual run learning curves.

## Recommendations

- Run CV on best config for robust validation
- Consider Focal Loss if action F1 < 0.4
- Try higher dropout if overfitting observed

---
*Auto-generated by experiment_analysis.py*
"""
    
    report_path = OUTPUT_DIR / "experiment_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}")
    
    return report

def main():
    print("="*60)
    print("Experiment Analysis & Report Generator")
    print("="*60)
    
    setup_dirs()
    
    print("\n[1/4] Fetching runs from W&B...")
    df = fetch_recent_runs(n_runs=30, filter_prefix="beta_")
    print(f"Found {len(df)} matching runs")
    
    if df.empty:
        print("No runs found. Exiting.")
        return
    
    # Save raw data
    df.to_csv(OUTPUT_DIR / "experiments.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'experiments.csv'}")
    
    print("\n[2/4] Generating beta comparison plot...")
    plot_beta_comparison(df)
    
    print("\n[3/4] Generating learning curves for top runs...")
    valid = df[df["val_scenario_f1"].notna()].head(5)
    for _, row in valid.iterrows():
        try:
            plot_learning_curves(row["run_id"], row["name"])
        except Exception as e:
            print(f"  Error plotting {row['name']}: {e}")
    
    print("\n[4/4] Generating report...")
    generate_report(df)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Report: {OUTPUT_DIR / 'experiment_report.md'}")
    print(f"Figures: {FIGURES_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
