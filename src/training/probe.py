import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

from src.data.hierarchical_dataset import HierarchicalDataset
from src.models.hierarchical import HierarchicalModel
from src.training.losses import FocalLoss
from src import wandb_safe


def train_probe(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    # Load dataset (train/val splits as-is)
    import pandas as pd
    scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
    train_uids = scenario_df[scenario_df['split'] == 'train']['video_uid'].tolist()
    val_uids = scenario_df[scenario_df['split'] == 'val']['video_uid'].tolist()

    train_ds = HierarchicalDataset(
        train_uids,
        args.processed_dir,
        "data/labels/scenario_labels.csv",
        "data/labels/action_labels_4class.csv",
        config,
        training=False  # Disable augmentation for probe
    )
    val_ds = HierarchicalDataset(
        val_uids,
        args.processed_dir,
        "data/labels/scenario_labels.csv",
        "data/labels/action_labels_4class.csv",
        config,
        training=False
    )
    
    # Get num_action_classes from dataset
    num_action_classes = train_ds.num_action_classes

    bs = int(os.environ.get("BATCH_SIZE", config['training']['batch_size']))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    # Load model
    model = HierarchicalModel(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    
    # IMPROVED: Fine-tune last GRU layer + action head (not fully frozen)
    for p in model.lle.parameters():
        p.requires_grad = False
    for p in model.hla.parameters():
        p.requires_grad = False
    
    # Unfreeze GRU last layer for fine-tuning
    for p in model.lle.gru.parameters():
        p.requires_grad = True
    for p in model.lle.fc.parameters():
        p.requires_grad = True
    
    # IMPROVED: Replace linear action head with small MLP for better capacity
    embedding_dim = config['lle']['embedding_dim']
    model.action_head = nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(embedding_dim // 2, num_action_classes)
    ).to(device)
    
    # Optimizer with different LR for different parts
    optimizer = torch.optim.Adam([
        {'params': model.lle.gru.parameters(), 'lr': config['training']['lr'] * 0.1},  # Lower LR for fine-tune
        {'params': model.lle.fc.parameters(), 'lr': config['training']['lr'] * 0.1},
        {'params': model.action_head.parameters(), 'lr': config['training']['lr']}  # Higher LR for new layers
    ], weight_decay=config['training']['weight_decay'])
    
    # Focal Loss with aggressive class weights for 4-class action
    action_weights = torch.tensor([5.0, 7.0, 0.5, 10.0]).to(device)
    criterion_action = FocalLoss(gamma=2.0, alpha=action_weights, ignore_index=-1)
    print(f"Improved Probe: Fine-tuning GRU + MLP action head")
    print(f"Using Focal Loss (gamma=2.0) with weights: {action_weights}")

    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb_safe.init(
            project="har-imu-training",
            name=f"probe-improved-{args.run_name}" if args.run_name else "probe-improved",
            config=config,
        )

    best_f1 = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Probe Epoch {epoch+1}"):
            inputs = batch['inputs'].to(device)
            action_labels = batch['action_labels'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                _, a_logits = model(inputs)
                loss = criterion_action(a_logits.view(-1, num_action_classes), action_labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_a_preds, all_a_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['inputs'].to(device)
                a_labels = batch['action_labels'].to(device)
                _, a_logits = model(inputs)
                preds = torch.argmax(a_logits, dim=2).view(-1)
                labels = a_labels.view(-1)
                mask = labels != -1
                if mask.sum() == 0:
                    continue
                all_a_preds.extend(preds[mask].cpu().numpy())
                all_a_labels.extend(labels[mask].cpu().numpy())
        if len(all_a_labels) > 0:
            val_a_f1 = f1_score(all_a_labels, all_a_preds, average='macro')
            val_a_acc = (np.array(all_a_preds) == np.array(all_a_labels)).mean()
        else:
            val_a_f1, val_a_acc = 0, 0
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Action F1: {val_a_f1:.4f} | Acc: {val_a_acc:.4f}")

        if val_a_f1 > best_f1:
            best_f1 = val_a_f1
            torch.save(model.state_dict(), Path(args.output_dir) / "best_probe.pth")
            print(f"✓ Best probe model saved (F1: {best_f1:.4f})")

        if wandb_run is not None:
            wandb_safe.log(wandb_run, {
                "epoch": epoch + 1,
                "probe_train_loss": avg_loss,
                "probe_val_action_f1": val_a_f1,
                "probe_val_action_acc": val_a_acc,
            })

    torch.save(model.state_dict(), Path(args.output_dir) / "last_probe.pth")
    print(f"Probe training finished. Best F1: {best_f1:.4f}")
