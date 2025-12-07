#!/usr/bin/env python3
"""
Baseline Models Training Script
Trains all baseline models and logs to W&B for comparison with Hierarchical model.

Usage:
    python scripts/train_baselines.py --model mlp_mlp
    python scripts/train_baselines.py --all  # Train all baselines
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hierarchical_dataset import HierarchicalDataset
from src.models.baseline_models import create_baseline_model, BASELINE_DESCRIPTIONS
from src import wandb_safe
from src.config import load_config


def train_baseline(model_name, config, args):
    """Train a single baseline model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training Baseline: {model_name}")
    print(f"Description: {BASELINE_DESCRIPTIONS.get(model_name, 'N/A')}")
    print(f"{'='*60}")
    
    # Load data
    scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
    train_uids = scenario_df[scenario_df['split'] == 'train']['video_uid'].tolist()
    val_uids = scenario_df[scenario_df['split'] == 'val']['video_uid'].tolist()
    
    train_ds = HierarchicalDataset(
        train_uids,
        args.processed_dir,
        "data/labels/scenario_labels.csv",
        "data/labels/action_labels_4class.csv",
        config
    )
    val_ds = HierarchicalDataset(
        val_uids,
        args.processed_dir,
        "data/labels/scenario_labels.csv",
        "data/labels/action_labels_4class.csv",
        config
    )
    
    bs = config['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    
    # Create model
    model = create_baseline_model(model_name, config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Get action classes
    num_action_classes = train_ds.num_action_classes
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    
    criterion_scenario = nn.CrossEntropyLoss()
    criterion_action = nn.CrossEntropyLoss(ignore_index=-1)
    
    # W&B
    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb_safe.init(
            project="har-imu-training",
            name=f"baseline_{model_name}_{args.run_suffix}" if args.run_suffix else f"baseline_{model_name}",
            config={
                **config,
                "model_type": "baseline",
                "baseline_name": model_name,
                "num_params": num_params,
            }
        )
    
    # Output dir
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_f1 = 0
    beta = config['training'].get('beta', 0.3)
    
    for epoch in range(config['training']['epochs']):
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
        model.eval()
        all_s_preds, all_s_labels = [], []
        all_a_preds, all_a_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
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
        
        val_s_f1 = f1_score(all_s_labels, all_s_preds, average='macro')
        val_s_acc = (np.array(all_s_preds) == np.array(all_s_labels)).mean()
        
        val_a_f1, val_a_acc = 0.0, 0.0
        if beta > 0 and len(all_a_labels) > 0:
            val_a_f1 = f1_score(all_a_labels, all_a_preds, average='macro')
            val_a_acc = (np.array(all_a_preds) == np.array(all_a_labels)).mean()
        
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
    
    # Final results
    results = {
        "model": model_name,
        "best_scenario_f1": best_f1,
        "final_action_f1": val_a_f1,
        "num_params": num_params,
    }
    
    if wandb_run:
        wandb_safe.log(wandb_run, {"best_scenario_f1": best_f1})
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
    parser.add_argument("--output-dir", type=str, default="checkpoints/baselines")
    parser.add_argument("--run-suffix", type=str, default="", help="Suffix for W&B run name")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
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
    print(f"{'Model':<15} {'Params':>10} {'Scenario F1':>12} {'Action F1':>10}")
    print("-"*60)
    for r in all_results:
        print(f"{r['model']:<15} {r['num_params']:>10,} {r['best_scenario_f1']:>12.4f} {r['final_action_f1']:>10.4f}")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(Path(args.output_dir) / "baseline_results.csv", index=False)
    print(f"\nResults saved to {args.output_dir}/baseline_results.csv")


if __name__ == "__main__":
    main()
