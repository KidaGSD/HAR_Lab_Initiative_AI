import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

from src.data.hierarchical_dataset import HierarchicalDataset
from src.models.hierarchical import HierarchicalModel
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
        "data/labels/action_labels_llm_validated.csv",
        config
    )
    val_ds = HierarchicalDataset(
        val_uids,
        args.processed_dir,
        "data/labels/scenario_labels.csv",
        "data/labels/action_labels_llm_validated.csv",
        config
    )

    bs = int(os.environ.get("BATCH_SIZE", config['training']['batch_size']))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    # Load model and freeze encoder
    model = HierarchicalModel(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    for p in model.lle.parameters():
        p.requires_grad = False
    for p in model.hla.parameters():
        p.requires_grad = False

    # Only train action head
    optimizer = torch.optim.Adam(model.action_head.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    criterion_action = nn.CrossEntropyLoss(ignore_index=-1)

    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb_safe.init(
            project="har-imu-training",
            name=f"probe-{args.run_name}" if args.run_name else None,
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
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                _, a_logits = model(inputs)
                loss = criterion_action(a_logits.view(-1, 6), action_labels.view(-1))
            loss.backward()
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
        print(f"Probe Val Action F1: {val_a_f1:.4f} | Acc: {val_a_acc:.4f}")

        if val_a_f1 > best_f1:
            best_f1 = val_a_f1
            torch.save(model.state_dict(), Path(args.output_dir) / "best_probe.pth")

        if wandb_run is not None:
            wandb_safe.log(wandb_run, {
                "epoch": epoch + 1,
                "probe_train_loss": avg_loss,
                "probe_val_action_f1": val_a_f1,
                "probe_val_action_acc": val_a_acc,
            })

    torch.save(model.state_dict(), Path(args.output_dir) / "last_probe.pth")
    print("Probe training finished.")
