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

    def reset(self):
        """Reset early stopping state for new fold."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None


def build_loaders(config, train_uids, val_uids, processed_dir):
    train_ds = HierarchicalDataset(
        train_uids,
        processed_dir,
        "data/labels/scenario_labels.csv",
        "data/labels/action_labels_4class.csv",
        config
    )
    val_ds = HierarchicalDataset(
        val_uids,
        processed_dir,
        "data/labels/scenario_labels.csv",
        "data/labels/action_labels_4class.csv",
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


def train_one_fold(args, config, train_uids, val_uids, fold_idx=None, wandb_run=None):
    """
    Train a single fold. If fold_idx is provided, metrics are prefixed with fold number.

    Args:
        args: Command line arguments
        config: Training configuration
        train_uids: List of training video UIDs
        val_uids: List of validation video UIDs
        fold_idx: Fold index (0-based) for CV, None for single split training
        wandb_run: Existing WandB run to use (for CV), or None to create new one

    Returns:
        best_f1: Best validation F1 score achieved
        best_epoch: Epoch where best F1 was achieved
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    # Metric prefix for this fold
    prefix = f"fold{fold_idx + 1}/" if fold_idx is not None else ""

    # Build loaders
    train_loader, val_loader, train_ds, val_ds = build_loaders(
        config, train_uids, val_uids, args.processed_dir
    )

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
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

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
    
    # Action loss with optional class weights
    num_action_classes = train_ds.num_action_classes
    if config['training'].get('use_action_class_weights', False):
        # Weights inversely proportional to class frequency (4-class: Manipulation 70%, Stationary 13%, Locomotion 10%, Search_Interrupt 7%)
        action_weights = torch.tensor([1.0, 1.3, 0.2, 1.8]).to(device)  # Stationary, Locomotion, Manipulation, Search_Interrupt
        criterion_action = nn.CrossEntropyLoss(weight=action_weights, ignore_index=-1)
    else:
        criterion_action = nn.CrossEntropyLoss(ignore_index=-1)

    early_stopper = EarlyStopping(patience=config['training']['patience'])

    # WandB - only init if not provided (single split mode)
    own_wandb_run = False
    if wandb_run is None and not args.no_wandb:
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
        own_wandb_run = True

    print("Starting training...")
    current_lr = optimizer.param_groups[0]['lr']
    best_epoch = 0

    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        fold_desc = f"Fold {fold_idx + 1} " if fold_idx is not None else ""
        for batch in tqdm(train_loader, desc=f"{fold_desc}Epoch {epoch+1}"):
            inputs = batch['inputs'].to(device)
            scenario_labels = batch['scenario_label'].to(device)
            action_labels = batch['action_labels'].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                s_logits, a_logits = model(inputs)
                loss_s = criterion_scenario(s_logits, scenario_labels)
                if config['training']['beta'] > 0:
                    loss_a = criterion_action(a_logits.view(-1, num_action_classes), action_labels.view(-1))
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
                
                # Action predictions (if beta > 0)
                if config['training']['beta'] > 0:
                    a_preds = torch.argmax(a_logits, dim=-1).view(-1)
                    a_labels_flat = a_labels.view(-1)
                    # Filter out ignore_index (-1)
                    valid_mask = a_labels_flat != -1
                    all_a_preds.extend(a_preds[valid_mask].cpu().numpy())
                    all_a_labels.extend(a_labels_flat[valid_mask].cpu().numpy())
                    
        val_s_f1 = f1_score(all_s_labels, all_s_preds, average='macro')
        val_s_acc = (np.array(all_s_preds) == np.array(all_s_labels)).mean()
        print(f"Val Scenario F1: {val_s_f1:.4f} | Acc: {val_s_acc:.4f}")
        
        # Action metrics
        val_a_f1, val_a_acc = 0.0, 0.0
        if config['training']['beta'] > 0 and len(all_a_labels) > 0:
            val_a_f1 = f1_score(all_a_labels, all_a_preds, average='macro')
            val_a_acc = (np.array(all_a_preds) == np.array(all_a_labels)).mean()
            print(f"Val Action F1: {val_a_f1:.4f} | Acc: {val_a_acc:.4f}")

        # Log with fold prefix
        if wandb_run is not None:
            log_dict = {
                f"{prefix}epoch": epoch + 1,
                f"{prefix}train_loss": avg_train_loss,
                f"{prefix}val_scenario_f1": val_s_f1,
                f"{prefix}val_scenario_acc": val_s_acc,
                f"{prefix}learning_rate": current_lr
            }
            if config['training']['beta'] > 0:
                log_dict[f"{prefix}val_action_f1"] = val_a_f1
                log_dict[f"{prefix}val_action_acc"] = val_a_acc
            
            # Per-class F1 for detailed analysis
            scenario_f1_per_class = f1_score(all_s_labels, all_s_preds, average=None)
            scenario_names = list(train_ds.scenario_map.keys())
            for i, name in enumerate(scenario_names):
                if i < len(scenario_f1_per_class):
                    log_dict[f"{prefix}scenario_f1_{name}"] = scenario_f1_per_class[i]
            
            if config['training']['beta'] > 0 and len(all_a_labels) > 0:
                action_f1_per_class = f1_score(all_a_labels, all_a_preds, average=None)
                action_names = list(train_ds.action_map.keys())
                for i, name in enumerate(action_names):
                    if i < len(action_f1_per_class):
                        log_dict[f"{prefix}action_f1_{name}"] = action_f1_per_class[i]
            
            wandb_safe.log(wandb_run, log_dict)
            
            # Log confusion matrices every 5 epochs
            if (epoch + 1) % 5 == 0:
                wandb_safe.log_confmat(
                    wandb_run,
                    y_true=all_s_labels,
                    preds=all_s_preds,
                    class_names=list(train_ds.scenario_map.keys()),
                    key=f"{prefix}conf_mat_scenario"
                )
                # Action confusion matrix
                if config['training']['beta'] > 0 and len(all_a_labels) > 0:
                    wandb_safe.log_confmat(
                        wandb_run,
                        y_true=all_a_labels,
                        preds=all_a_preds,
                        class_names=list(train_ds.action_map.keys()),
                        key=f"{prefix}conf_mat_action"
                    )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr:.2e}")

        early_stopper(val_s_f1, model)
        if early_stopper.best_score == val_s_f1:
            best_epoch = epoch + 1
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    if early_stopper.best_model_state:
        torch.save(early_stopper.best_model_state, Path(args.output_dir) / "best_model.pth")
        print(f"Best model saved (Scenario F1: {early_stopper.best_score:.4f})")
    torch.save(model.state_dict(), Path(args.output_dir) / "last_model.pth")
    print("Last model saved.")

    # Only finish wandb if we created it ourselves
    if own_wandb_run and wandb_run is not None:
        wandb_safe.safe_finish(wandb_run)

    return early_stopper.best_score, best_epoch


def train_one_split(args, config):
    """Train on predefined train/val split (legacy interface)."""
    import pandas as pd
    scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
    train_uids = scenario_df[scenario_df['split'] == 'train']['video_uid'].tolist()
    val_uids = scenario_df[scenario_df['split'] == 'val']['video_uid'].tolist()

    return train_one_fold(args, config, train_uids, val_uids, fold_idx=None, wandb_run=None)


def run_cv(args, config):
    """Run k-fold cross validation with all folds in a single WandB run."""
    import pandas as pd
    scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
    usable_df = scenario_df[scenario_df['split'].isin(['train', 'val'])].copy()
    print(f"\nUsing {len(usable_df)} videos for {args.n_folds}-fold CV")
    print(f"Excluded: test={len(scenario_df[scenario_df['split']=='test'])}, multi={len(scenario_df[scenario_df['split']=='multi'])}")

    video_scenarios = usable_df[['video_uid', 'scenario']].copy()
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    # Initialize single WandB run for all folds
    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb_safe.init(
            project="har-imu-training",
            name=f"cv-{args.n_folds}fold-{args.run_name}" if args.run_name else f"cv-{args.n_folds}fold",
            config={
                **config,
                "n_folds": args.n_folds,
                "cv_mode": True,
            }
        )

    # Track results for each fold
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(video_scenarios['video_uid'], video_scenarios['scenario'])):
        print("\n" + "="*80)
        print(f"FOLD {fold_idx + 1}/{args.n_folds}")
        print("="*80)

        fold_train_uids = video_scenarios.iloc[train_idx]['video_uid'].tolist()
        fold_val_uids = video_scenarios.iloc[val_idx]['video_uid'].tolist()

        print(f"Train videos: {len(fold_train_uids)}, Val videos: {len(fold_val_uids)}")

        # Update scenario_labels.csv temporarily for data loading
        original_scenario_df = scenario_df.copy()
        temp_scenario_df = original_scenario_df.copy()
        temp_scenario_df.loc[temp_scenario_df['video_uid'].isin(fold_train_uids), 'split'] = 'train'
        temp_scenario_df.loc[temp_scenario_df['video_uid'].isin(fold_val_uids), 'split'] = 'val'
        temp_scenario_df.loc[~temp_scenario_df['video_uid'].isin(fold_train_uids + fold_val_uids), 'split'] = 'excluded'
        temp_scenario_df.to_csv("data/labels/scenario_labels_temp.csv", index=False)

        os.rename("data/labels/scenario_labels.csv", "data/labels/scenario_labels_backup.csv")
        os.rename("data/labels/scenario_labels_temp.csv", "data/labels/scenario_labels.csv")

        try:
            # Create fold-specific output directory
            fold_output_dir = str(Path(args.output_dir) / f"fold{fold_idx + 1}")
            fold_args = argparse.Namespace(**vars(args))
            fold_args.output_dir = fold_output_dir

            # Train this fold
            best_f1, best_epoch = train_one_fold(
                fold_args, config,
                fold_train_uids, fold_val_uids,
                fold_idx=fold_idx,
                wandb_run=wandb_run
            )

            fold_results.append({
                'fold': fold_idx + 1,
                'best_f1': best_f1,
                'best_epoch': best_epoch,
                'train_videos': len(fold_train_uids),
                'val_videos': len(fold_val_uids),
            })

            # Log fold summary to wandb
            if wandb_run is not None:
                wandb_safe.log_summary(wandb_run, {
                    f"fold{fold_idx + 1}_best_f1": best_f1,
                    f"fold{fold_idx + 1}_best_epoch": best_epoch,
                })

            print(f"Fold {fold_idx + 1} complete: Best F1 = {best_f1:.4f} at epoch {best_epoch}")

        except Exception as e:
            print(f"Fold {fold_idx + 1} failed with error: {e}")
            import traceback
            traceback.print_exc()
            fold_results.append({
                'fold': fold_idx + 1,
                'best_f1': None,
                'best_epoch': None,
                'error': str(e),
            })

        finally:
            # Restore original scenario_labels.csv
            os.rename("data/labels/scenario_labels.csv", "data/labels/scenario_labels_temp.csv")
            os.rename("data/labels/scenario_labels_backup.csv", "data/labels/scenario_labels.csv")
            if os.path.exists("data/labels/scenario_labels_temp.csv"):
                os.remove("data/labels/scenario_labels_temp.csv")

    # Compute and log CV summary
    print("\n" + "=" * 80)
    print("CROSS VALIDATION COMPLETE")
    print("=" * 80)

    valid_results = [r for r in fold_results if r.get('best_f1') is not None]
    if valid_results:
        f1_scores = [r['best_f1'] for r in valid_results]
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)

        print(f"\nResults Summary:")
        print("-" * 40)
        for r in fold_results:
            if r.get('best_f1') is not None:
                print(f"  Fold {r['fold']}: F1 = {r['best_f1']:.4f} (epoch {r['best_epoch']})")
            else:
                print(f"  Fold {r['fold']}: FAILED - {r.get('error', 'Unknown error')}")
        print("-" * 40)
        print(f"  Mean F1: {mean_f1:.4f} +/- {std_f1:.4f}")
        print(f"  Completed: {len(valid_results)}/{args.n_folds} folds")

        # Log final CV summary
        if wandb_run is not None:
            wandb_safe.log_summary(wandb_run, {
                "cv_mean_f1": mean_f1,
                "cv_std_f1": std_f1,
                "cv_min_f1": min(f1_scores),
                "cv_max_f1": max(f1_scores),
                "cv_completed_folds": len(valid_results),
            })

    # Finish wandb run
    if wandb_run is not None:
        wandb_safe.safe_finish(wandb_run)

    return fold_results
