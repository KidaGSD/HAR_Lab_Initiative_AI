import os
import argparse
import math
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from src.data.hierarchical_dataset import HierarchicalDataset
from src.models.hierarchical import HierarchicalModel
from src import wandb_safe


class EarlyStopping:
    def __init__(self, patience=50, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            self.counter = 0


def build_loaders(config, train_uids, val_uids, processed_dir):
    train_ds = HierarchicalDataset(
        train_uids,
        processed_dir,
        "data/labels/scenario_labels.csv",
        "data/labels/action_labels_llm_validated.csv",
        config
    )
    val_ds = HierarchicalDataset(
        val_uids,
        processed_dir,
        "data/labels/scenario_labels.csv",
        "data/labels/action_labels_llm_validated.csv",
        config
    )
    bs = int(os.environ.get("BATCH_SIZE", config['training']['batch_size']))
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    return train_loader, val_loader, train_ds, val_ds


def train_one_split(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    # Load scenario splits
    import pandas as pd
    scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
    train_uids = scenario_df[scenario_df['split'] == 'train']['video_uid'].tolist()
    val_uids = scenario_df[scenario_df['split'] == 'val']['video_uid'].tolist()

    # Build loaders
    train_loader, val_loader, train_ds, val_ds = build_loaders(config, train_uids, val_uids, args.processed_dir)

    # Model
    model = HierarchicalModel(config).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # Scheduler: warmup + cosine
    warmup_epochs = config['training']['warmup_epochs']
    total_epochs = config['training']['epochs']
    min_factor = 0.2
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Loss
    scenario_labels_train = torch.tensor([s['scenario_label'].item() for s in train_ds.samples])
    class_counts = torch.bincount(scenario_labels_train)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    criterion_scenario = nn.CrossEntropyLoss(weight=class_weights.to(device))
    criterion_action = nn.CrossEntropyLoss(ignore_index=-1)

    early_stopper = EarlyStopping(patience=config['training']['patience'])

    # WandB
    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb_safe.init(
            project="har-imu-training",
            name=f"hierarchical-{args.run_name}" if args.run_name else None,
            config={
                **config,
                "train_videos": len(train_uids),
                "val_videos": len(val_uids),
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
            }
        )

    print("Starting training...")
    current_lr = optimizer.param_groups[0]['lr']

    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs = batch['inputs'].to(device)
            scenario_labels = batch['scenario_label'].to(device)
            action_labels = batch['action_labels'].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                s_logits, a_logits = model(inputs)
                loss_s = criterion_scenario(s_logits, scenario_labels)
                if config['training']['beta'] > 0:
                    loss_a = criterion_action(a_logits.view(-1, 6), action_labels.view(-1))
                    loss = config['training']['alpha'] * loss_s + config['training']['beta'] * loss_a
                else:
                    loss = loss_s
            scaler.scale(loss).backward()
            if config['training']['grad_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        all_s_preds, all_s_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['inputs'].to(device)
                s_labels = batch['scenario_label'].to(device)
                s_logits, _ = model(inputs)
                s_preds = torch.argmax(s_logits, dim=1)
                all_s_preds.extend(s_preds.cpu().numpy())
                all_s_labels.extend(s_labels.cpu().numpy())
        val_s_f1 = f1_score(all_s_labels, all_s_preds, average='macro')
        val_s_acc = (np.array(all_s_preds) == np.array(all_s_labels)).mean()
        print(f"Val Scenario F1: {val_s_f1:.4f} | Acc: {val_s_acc:.4f}")

        if wandb_run is not None:
            wandb_safe.log(wandb_run, {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_scenario_f1": val_s_f1,
                "val_scenario_acc": val_s_acc,
                "learning_rate": current_lr
            })
            if (epoch + 1) % 5 == 0:
                wandb_safe.log_confmat(
                    wandb_run,
                    y_true=all_s_labels,
                    preds=all_s_preds,
                    class_names=list(train_ds.scenario_map.keys())
                )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr:.2e}")

        early_stopper(val_s_f1, model)
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break

    os.makedirs(args.output_dir, exist_ok=True)
    if early_stopper.best_model_state:
        torch.save(early_stopper.best_model_state, Path(args.output_dir) / "best_model.pth")
        print(f"Best model saved (Scenario F1: {early_stopper.best_score:.4f})")
    torch.save(model.state_dict(), Path(args.output_dir) / "last_model.pth")
    print("Last model saved.")


def run_cv(args, config):
    import pandas as pd
    scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
    usable_df = scenario_df[scenario_df['split'].isin(['train', 'val'])].copy()
    print(f"\nUsing {len(usable_df)} videos for {args.n_folds}-fold CV")
    print(f"Excluded: test={len(scenario_df[scenario_df['split']=='test'])}, multi={len(scenario_df[scenario_df['split']=='multi'])}")

    video_scenarios = usable_df[['video_uid', 'scenario']].copy()
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(video_scenarios['video_uid'], video_scenarios['scenario'])):
        print("\n" + "="*80)
        print(f"FOLD {fold_idx + 1}/{args.n_folds}")
        print("="*80)

        fold_train_uids = video_scenarios.iloc[train_idx]['video_uid'].tolist()
        fold_val_uids = video_scenarios.iloc[val_idx]['video_uid'].tolist()

        original_scenario_df = scenario_df.copy()
        temp_scenario_df = original_scenario_df.copy()
        temp_scenario_df.loc[temp_scenario_df['video_uid'].isin(fold_train_uids), 'split'] = 'train'
        temp_scenario_df.loc[temp_scenario_df['video_uid'].isin(fold_val_uids), 'split'] = 'val'
        temp_scenario_df.loc[~temp_scenario_df['video_uid'].isin(fold_train_uids + fold_val_uids), 'split'] = 'excluded'
        temp_scenario_df.to_csv("data/labels/scenario_labels_temp.csv", index=False)

        os.rename("data/labels/scenario_labels.csv", "data/labels/scenario_labels_backup.csv")
        os.rename("data/labels/scenario_labels_temp.csv", "data/labels/scenario_labels.csv")

        try:
            fold_args = argparse.Namespace(**vars(args))
            fold_args.cv = False
            fold_args.run_name = f"{args.run_name}_fold{fold_idx+1}" if not args.no_wandb else None
            fold_args.output_dir = str(Path(args.output_dir) / f"fold{fold_idx+1}")
            train_one_split(fold_args, config)
            print(f"Fold {fold_idx+1} training complete")
        finally:
            os.rename("data/labels/scenario_labels.csv", "data/labels/scenario_labels_temp.csv")
            os.rename("data/labels/scenario_labels_backup.csv", "data/labels/scenario_labels.csv")
            os.remove("data/labels/scenario_labels_temp.csv")

    print("\n" + "=" * 80)
    print("CROSS VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nAll {args.n_folds} folds completed.")
