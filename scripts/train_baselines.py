#!/usr/bin/env python3
"""
Baseline Models Training Script
Trains baseline models and logs to W&B for comparison with the hierarchical model.

Examples:
    python scripts/train_baselines.py --model imu2clip
    python scripts/train_baselines.py --model imu2clip --labels-csv data/labels/action_labels_llm_clean_refined.csv
    python scripts/train_baselines.py --all
"""

import sys
import argparse
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hierarchical_dataset import HierarchicalDataset
from src.models.baseline_models import create_baseline_model, BASELINE_DESCRIPTIONS
from src import wandb_safe
from src.config import load_config


def build_scenario_split_table(labels_df, split_source_df):
    """Build a per-video scenario/split table from labels + split source."""
    required_label_cols = {"video_uid", "scenario", "action", "timestamp_sec"}
    missing_required = sorted(required_label_cols - set(labels_df.columns))
    if missing_required:
        raise ValueError(f"--labels-csv is missing required columns: {missing_required}")

    if {"video_uid", "scenario", "split"}.issubset(labels_df.columns):
        scenario_df = (
            labels_df[["video_uid", "scenario", "split"]]
            .dropna(subset=["video_uid", "scenario", "split"])
            .drop_duplicates(subset=["video_uid"], keep="first")
        )
    else:
        if not {"video_uid", "split"}.issubset(split_source_df.columns):
            raise ValueError(
                "--scenario-labels-csv must contain at least 'video_uid' and 'split' "
                "when --labels-csv does not include split information."
            )
        scenario_df = (
            labels_df[["video_uid", "scenario"]]
            .dropna(subset=["video_uid", "scenario"])
            .drop_duplicates(subset=["video_uid"], keep="first")
        )
        scenario_df = scenario_df.merge(
            split_source_df[["video_uid", "split"]],
            on="video_uid",
            how="left",
        )
        missing_split = int(scenario_df["split"].isna().sum())
        if missing_split > 0:
            print(f"Warning: {missing_split} videos missing split mapping; assigning to train.")
            scenario_df["split"] = scenario_df["split"].fillna("train")

    return scenario_df


def summarize_label_rows(label_df, scenario_df, title):
    # If labels already contain split, avoid split_x/split_y merge suffixes.
    if "split" in label_df.columns:
        rows_df = label_df.copy()
    else:
        rows_df = label_df.merge(
            scenario_df[["video_uid", "split"]].drop_duplicates(subset=["video_uid"]),
            on="video_uid",
            how="left",
        )
    rows_df["split"] = rows_df["split"].fillna("unmapped")
    print(f"\n{title}:")
    print(rows_df["split"].value_counts().sort_index())
    return rows_df


def select_available_uids(uids, available_uids):
    return [uid for uid in uids if uid in available_uids]


def format_availability(label, available, total):
    if total == 0:
        return f"  {label}: 0/0 (no {label.lower()} split in this run)"
    pct = available / total * 100
    return f"  {label}: {available}/{total} ({pct:.1f}% available)"


def make_run_name(args, model_name):
    if args.run_name:
        return args.run_name if not args.all else f"{args.run_name}_{model_name}"
    if args.run_suffix:
        return f"baseline_{model_name}_{args.run_suffix}"
    return f"baseline_{model_name}"


def evaluate_split(model, loader, device, beta, num_action_classes):
    model.eval()
    all_s_preds, all_s_labels = [], []
    all_a_preds, all_a_labels = [], []

    with torch.no_grad():
        for batch in loader:
            inputs = batch['inputs'].to(device)
            s_labels = batch['scenario_label'].to(device)
            a_labels = batch['action_labels'].to(device)

            s_logits, a_logits = model(inputs)
            s_preds = torch.argmax(s_logits, dim=1)
            all_s_preds.extend(s_preds.cpu().numpy())
            all_s_labels.extend(s_labels.cpu().numpy())

            if beta > 0:
                a_preds = torch.argmax(a_logits, dim=-1).view(-1)
                a_labels_flat = a_labels.view(-1)
                valid_mask = a_labels_flat != -1
                all_a_preds.extend(a_preds[valid_mask].cpu().numpy())
                all_a_labels.extend(a_labels_flat[valid_mask].cpu().numpy())

    if not all_s_labels:
        return {
            "scenario_f1": 0.0,
            "scenario_acc": 0.0,
            "action_f1": 0.0,
            "action_acc": 0.0,
            "num_action_classes": num_action_classes,
        }

    scenario_f1 = f1_score(all_s_labels, all_s_preds, average='macro')
    scenario_acc = (np.array(all_s_preds) == np.array(all_s_labels)).mean()

    action_f1, action_acc = 0.0, 0.0
    if beta > 0 and len(all_a_labels) > 0:
        action_f1 = f1_score(all_a_labels, all_a_preds, average='macro')
        action_acc = (np.array(all_a_preds) == np.array(all_a_labels)).mean()

    return {
        "scenario_f1": scenario_f1,
        "scenario_acc": scenario_acc,
        "action_f1": action_f1,
        "action_acc": action_acc,
        "num_action_classes": num_action_classes,
    }


def train_baseline(model_name, config, args):
    """Train a single baseline model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training Baseline: {model_name}")
    print(f"Description: {BASELINE_DESCRIPTIONS.get(model_name, 'N/A')}")
    print(f"{'='*60}")

    labels_df = pd.read_csv(args.labels_csv)
    split_source_df = pd.read_csv(args.scenario_labels_csv)
    test_labels_df = pd.read_csv(args.test_labels_csv) if args.test_labels_csv else labels_df.copy()
    scenario_df = build_scenario_split_table(labels_df, split_source_df)

    print("\nSplit distribution:")
    print(scenario_df['split'].value_counts().sort_index())
    summarize_label_rows(labels_df, scenario_df, "Label rows per split (training labels)")
    if args.test_labels_csv:
        summarize_label_rows(test_labels_df, scenario_df, "Label rows per split (test labels)")

    train_uids = scenario_df[scenario_df['split'] == 'train']['video_uid'].tolist()
    val_uids = scenario_df[scenario_df['split'] == 'val']['video_uid'].tolist()
    test_uids = scenario_df[scenario_df['split'] == 'test']['video_uid'].tolist()

    available_uids = {p.parent.name for p in Path(args.processed_dir).glob("*/seq.npz")}
    print(f"\nProcessed data availability:")
    print(f"  Total processed files: {len(available_uids)}")

    train_uids = select_available_uids(train_uids, available_uids)
    val_uids = select_available_uids(val_uids, available_uids)
    test_uids = select_available_uids(test_uids, available_uids)

    print("\nFinal data splits (with processed data):")
    print(format_availability("Train", len(train_uids), len(scenario_df[scenario_df['split'] == 'train'])))
    print(format_availability("Val", len(val_uids), len(scenario_df[scenario_df['split'] == 'val'])))
    print(format_availability("Test", len(test_uids), len(scenario_df[scenario_df['split'] == 'test'])))

    train_ds = HierarchicalDataset(
        train_uids,
        args.processed_dir,
        scenario_df,
        labels_df,
        config,
        training=True,
    )
    val_ds = HierarchicalDataset(
        val_uids,
        args.processed_dir,
        scenario_df,
        labels_df,
        config,
        training=False,
    )
    test_ds = HierarchicalDataset(
        test_uids,
        args.processed_dir,
        scenario_df,
        test_labels_df,
        config,
        training=False,
    )

    model_config = copy.deepcopy(config)
    model_config['hla']['num_classes'] = len(train_ds.scenario_map)
    model_config.setdefault('lla', {})['num_classes'] = train_ds.num_action_classes

    bs = model_config['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True) if len(test_ds) > 0 else None

    # Create model
    model = create_baseline_model(model_name, model_config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Get action classes
    num_action_classes = train_ds.num_action_classes

    # Optimizer & Loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_config['training']['lr'],
        weight_decay=model_config['training']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_config['training']['epochs'])

    criterion_scenario = nn.CrossEntropyLoss()
    criterion_action = nn.CrossEntropyLoss(ignore_index=-1)

    # W&B
    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb_safe.init(
            project="har-imu-training",
            name=make_run_name(args, model_name),
            config={
                **model_config,
                "model_type": "baseline",
                "baseline_name": model_name,
                "num_params": num_params,
                "labels_csv": args.labels_csv,
                "scenario_labels_csv": args.scenario_labels_csv,
                "test_labels_csv": args.test_labels_csv or args.labels_csv,
            }
        )

    # Output dir
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0
    beta = model_config['training'].get('beta', 0.3)
    final_val_metrics = None

    for epoch in range(model_config['training']['epochs']):
        # Training
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            inputs = batch['inputs'].to(device)
            scenario_labels = batch['scenario_label'].to(device)
            action_labels = batch['action_labels'].to(device)
            
            optimizer.zero_grad()
            s_logits, a_logits = model(inputs)
            
            loss_s = criterion_scenario(s_logits, scenario_labels)
            if beta > 0:
                loss_a = criterion_action(a_logits.view(-1, num_action_classes), action_labels.view(-1))
                loss = loss_s + beta * loss_a
            else:
                loss = loss_s
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        final_val_metrics = evaluate_split(model, val_loader, device, beta, num_action_classes)
        val_s_f1 = final_val_metrics["scenario_f1"]
        val_s_acc = final_val_metrics["scenario_acc"]
        val_a_f1 = final_val_metrics["action_f1"]
        val_a_acc = final_val_metrics["action_acc"]

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | Scenario F1={val_s_f1:.4f} | Action F1={val_a_f1:.4f}")

        # Log to W&B
        if wandb_run:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_scenario_f1": val_s_f1,
                "val_scenario_acc": val_s_acc,
            }
            if beta > 0:
                log_dict["val_action_f1"] = val_a_f1
                log_dict["val_action_acc"] = val_a_acc
            wandb_safe.log(wandb_run, log_dict)

        # Save best model
        if val_s_f1 > best_f1:
            best_f1 = val_s_f1
            torch.save(model.state_dict(), output_dir / "best_model.pth")

        scheduler.step()

    # Save final model
    torch.save(model.state_dict(), output_dir / "last_model.pth")

    test_metrics = None
    if test_loader is not None:
        best_model_path = output_dir / "best_model.pth"
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_metrics = evaluate_split(model, test_loader, device, beta, num_action_classes)
        print(
            "Final Test Results: "
            f"Scenario F1={test_metrics['scenario_f1']:.4f} | "
            f"Action F1={test_metrics['action_f1']:.4f}"
        )

    # Final results
    results = {
        "model": model_name,
        "best_scenario_f1": best_f1,
        "final_action_f1": final_val_metrics["action_f1"] if final_val_metrics else 0.0,
        "test_scenario_f1": test_metrics["scenario_f1"] if test_metrics else 0.0,
        "test_action_f1": test_metrics["action_f1"] if test_metrics else 0.0,
        "num_params": num_params,
    }

    if wandb_run:
        summary_payload = {"best_scenario_f1": best_f1}
        if test_metrics:
            summary_payload.update(
                {
                    "test_scenario_f1": test_metrics["scenario_f1"],
                    "test_scenario_acc": test_metrics["scenario_acc"],
                    "test_action_f1": test_metrics["action_f1"],
                    "test_action_acc": test_metrics["action_acc"],
                }
            )
        wandb_safe.log(wandb_run, summary_payload)
        wandb_run.finish()

    print(f"\nBest Scenario F1: {best_f1:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--model", type=str, choices=['mlp_mlp', 'cnn_mlp', 'imu2clip', 'cnn_lstm_gru'],
                        help="Baseline model to train")
    parser.add_argument("--all", action="store_true", help="Train all baselines")
    parser.add_argument("--config", type=str, default="configs/beta_0.3.yaml", help="Config file")
    parser.add_argument("--processed-dir", type=str, default="data/processed_ego4d")
    parser.add_argument("--labels-csv", type=str, default="data/labels/action_labels_4class.csv")
    parser.add_argument("--scenario-labels-csv", type=str, default="data/labels/scenario_labels.csv")
    parser.add_argument("--test-labels-csv", type=str, default="", help="Optional separate labels file for test evaluation")
    parser.add_argument("--output-dir", type=str, default="checkpoints/baselines")
    parser.add_argument("--run-name", type=str, default="", help="Explicit W&B run name")
    parser.add_argument("--run-suffix", type=str, default="", help="Suffix for W&B run name")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (for quick testing)")
    parser.add_argument(
        "--exclude-scenarios",
        type=str,
        default="",
        help='Comma-separated scenarios to exclude, e.g. "Gardening,Desk Work"',
    )
    parser.add_argument(
        "--exclude-actions",
        type=str,
        default="",
        help='Comma-separated actions to exclude, e.g. "Search,Error / Correction"',
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Override epochs if specified
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
        print(f"Epochs overridden to: {args.epochs}")

    if args.exclude_scenarios.strip():
        excluded = [s.strip() for s in args.exclude_scenarios.split(",") if s.strip()]
        config.setdefault("data", {})["excluded_scenarios"] = excluded
        print(f"Runtime scenario exclusion enabled: {excluded}")
    if args.exclude_actions.strip():
        excluded_actions = [s.strip() for s in args.exclude_actions.split(",") if s.strip()]
        config.setdefault("data", {})["excluded_actions"] = excluded_actions
        print(f"Runtime action exclusion enabled: {excluded_actions}")

    models_to_train = []
    if args.all:
        models_to_train = ['mlp_mlp', 'cnn_mlp', 'imu2clip', 'cnn_lstm_gru']
    elif args.model:
        models_to_train = [args.model]
    else:
        parser.error("Specify --model or --all")

    all_results = []
    for model_name in models_to_train:
        results = train_baseline(model_name, config, args)
        all_results.append(results)

    # Summary
    print("\n" + "="*60)
    print("BASELINE COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'Params':>10} {'Val S-F1':>10} {'Val A-F1':>10} {'Test S-F1':>10} {'Test A-F1':>10}")
    print("-"*80)
    for r in all_results:
        print(
            f"{r['model']:<15} {r['num_params']:>10,} "
            f"{r['best_scenario_f1']:>10.4f} {r['final_action_f1']:>10.4f} "
            f"{r['test_scenario_f1']:>10.4f} {r['test_action_f1']:>10.4f}"
        )

    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(Path(args.output_dir) / "baseline_results.csv", index=False)
    print(f"\nResults saved to {args.output_dir}/baseline_results.csv")


if __name__ == "__main__":
    main()
