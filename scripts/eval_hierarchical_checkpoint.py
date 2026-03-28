#!/usr/bin/env python3
"""
Evaluate a trained hierarchical checkpoint on a split (default: test)
without re-running training.

Examples:
  python scripts/eval_hierarchical_checkpoint.py \
    --checkpoint checkpoints/checkpoints/new_runs/check_hierarchical_355_1.0/best_model.pth \
    --labels-csv data/annotation_rounds/final_gold_dataset/HAR_dataset.csv \
    --split test --exclude-scenarios "Gardening"

  # Backfill summary metrics directly onto an existing run without
  # touching its training history/charts.
  python scripts/eval_hierarchical_checkpoint.py \
    --checkpoint .../best_model.pth \
    --labels-csv data/annotation_rounds/final_gold_dataset/HAR_dataset.csv \
    --wandb-run-id yvc8dvoj
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from train_hierarchical import CONFIG, HierarchicalDataset, HierarchicalModel, load_checkpoint_flexible


def build_scenario_table(labels_df: pd.DataFrame, split_source_df: pd.DataFrame) -> pd.DataFrame:
    if {"video_uid", "scenario", "split"}.issubset(labels_df.columns):
        return (
            labels_df[["video_uid", "scenario", "split"]]
            .dropna(subset=["video_uid", "scenario", "split"])
            .drop_duplicates(subset=["video_uid"], keep="first")
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
    scenario_df["split"] = scenario_df["split"].fillna("train")
    return scenario_df


def summarize_class_metrics(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    rows = []
    payload = {}
    for idx, class_name in enumerate(class_names):
        safe_name = class_name.strip().lower()
        for ch in [" ", "/", "-", "(", ")", ","]:
            safe_name = safe_name.replace(ch, "_")
        while "__" in safe_name:
            safe_name = safe_name.replace("__", "_")
        safe_name = safe_name.strip("_")
        rows.append({
            "name": class_name,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        })
        payload[f"{safe_name}_precision"] = float(precision[idx])
        payload[f"{safe_name}_recall"] = float(recall[idx])
        payload[f"{safe_name}_f1"] = float(f1[idx])
        payload[f"{safe_name}_support"] = int(support[idx])
    return rows, payload


def evaluate_split(model, loader, device):
    model.eval()
    s_preds_all, s_labels_all = [], []
    a_preds_all, a_labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["inputs"].to(device)
            s_labels = batch["scenario_label"].to(device)
            a_labels = batch["action_labels"].to(device)

            s_logits, a_logits = model(x)
            s_preds = torch.argmax(s_logits, dim=1)
            s_preds_all.extend(s_preds.cpu().numpy())
            s_labels_all.extend(s_labels.cpu().numpy())

            a_preds = torch.argmax(a_logits, dim=2).view(-1)
            a_labels_flat = a_labels.view(-1)
            mask = a_labels_flat != -1
            a_preds_all.extend(a_preds[mask].cpu().numpy())
            a_labels_all.extend(a_labels_flat[mask].cpu().numpy())

    s_f1 = f1_score(s_labels_all, s_preds_all, average="macro")
    s_acc = (np.array(s_preds_all) == np.array(s_labels_all)).mean()
    a_f1 = 0.0
    a_acc = 0.0
    if len(a_labels_all) > 0:
        a_f1 = f1_score(a_labels_all, a_preds_all, average="macro")
        a_acc = (np.array(a_preds_all) == np.array(a_labels_all)).mean()
    scenario_rows, scenario_payload = summarize_class_metrics(
        s_labels_all,
        s_preds_all,
        [loader.dataset.idx_to_scenario[i] for i in range(len(loader.dataset.scenario_map))],
    )
    action_rows = []
    action_payload = {}
    if len(a_labels_all) > 0:
        action_rows, action_payload = summarize_class_metrics(
            a_labels_all,
            a_preds_all,
            [loader.dataset.idx_to_action[i] for i in range(len(loader.dataset.action_map))],
        )
    return (
        s_f1,
        s_acc,
        a_f1,
        a_acc,
        scenario_rows,
        scenario_payload,
        action_rows,
        action_payload,
        s_labels_all,
        s_preds_all,
        a_labels_all,
        a_preds_all,
    )


def _checkpoint_head_dims(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    scenario_dim = None
    action_dim = None
    for key in ("hla.head.weight", "module.hla.head.weight"):
        if key in sd:
            scenario_dim = int(sd[key].shape[0])
            break
    for key in ("action_head.weight", "module.action_head.weight"):
        if key in sd:
            action_dim = int(sd[key].shape[0])
            break
    return scenario_dim, action_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--labels-csv", required=True, help="Single labels CSV containing scenario/action columns")
    parser.add_argument("--scenario-labels-csv", default="data/labels/scenario_labels.csv", help="Fallback split mapping by video_uid")
    parser.add_argument("--processed-dir", default="data/processed_ego4d")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--exclude-scenarios", default="", help='Comma-separated scenarios, e.g. "Gardening"')
    parser.add_argument("--exclude-actions", default="", help='Comma-separated actions, e.g. "Search,Error / Correction"')
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="har-imu-training")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-id", default="", help="Optional existing run id to update in summary only, without touching its history")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    labels_df = pd.read_csv(args.labels_csv)
    split_source_df = pd.read_csv(args.scenario_labels_csv)

    required = {"video_uid", "scenario", "action", "timestamp_sec"}
    missing = sorted(required - set(labels_df.columns))
    if missing:
        raise ValueError(f"--labels-csv missing required columns: {missing}")

    if args.exclude_scenarios.strip():
        excluded = [s.strip() for s in args.exclude_scenarios.split(",") if s.strip()]
        CONFIG.setdefault("data", {})["excluded_scenarios"] = excluded
        print(f"Excluding scenarios: {excluded}")
    if args.exclude_actions.strip():
        excluded_actions = [s.strip() for s in args.exclude_actions.split(",") if s.strip()]
        CONFIG.setdefault("data", {})["excluded_actions"] = excluded_actions
        print(f"Excluding actions: {excluded_actions}")

    scenario_df = build_scenario_table(labels_df, split_source_df)

    target_uids = scenario_df[scenario_df["split"] == args.split]["video_uid"].tolist()
    processed_dir = Path(args.processed_dir)
    available_uids = {p.parent.name for p in processed_dir.glob("*/seq.npz")}
    target_uids = [u for u in target_uids if u in available_uids]
    print(f"{args.split} videos with processed data: {len(target_uids)}")

    ds = HierarchicalDataset(target_uids, args.processed_dir, scenario_df, labels_df)
    print(f"{args.split} samples: {len(ds)}")
    if len(ds) == 0:
        raise ValueError(f"No samples found for split={args.split}")

    ckpt_scenario_dim, ckpt_action_dim = _checkpoint_head_dims(args.checkpoint, device="cpu")
    if ckpt_scenario_dim is not None and ckpt_scenario_dim != len(ds.scenario_map):
        raise ValueError(
            "Scenario class count mismatch between checkpoint and evaluation dataset: "
            f"checkpoint={ckpt_scenario_dim}, eval_dataset={len(ds.scenario_map)}. "
            "This usually means different excluded scenarios (e.g., missing --exclude-scenarios)."
        )
    if ckpt_action_dim is not None and ckpt_action_dim != len(ds.action_map):
        raise ValueError(
            "Action class count mismatch between checkpoint and evaluation dataset: "
            f"checkpoint={ckpt_action_dim}, eval_dataset={len(ds.action_map)}."
        )

    CONFIG["hla"]["num_classes"] = len(ds.scenario_map)
    CONFIG["training"]["num_actions"] = len(ds.action_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalModel(CONFIG).to(device)
    load_checkpoint_flexible(model, args.checkpoint, device)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    (
        s_f1,
        s_acc,
        a_f1,
        a_acc,
        scenario_rows,
        scenario_payload,
        action_rows,
        action_payload,
        s_labels_all,
        s_preds_all,
        a_labels_all,
        a_preds_all,
    ) = evaluate_split(model, loader, device)
    print(f"{args.split} scenario_f1={s_f1:.4f} scenario_acc={s_acc:.4f}")
    print(f"{args.split} action_f1={a_f1:.4f} action_acc={a_acc:.4f}")
    print("Scenario per-class metrics:")
    for row in scenario_rows:
        print(
            f"  {row['name']}: "
            f"P={row['precision']:.4f} R={row['recall']:.4f} "
            f"F1={row['f1']:.4f} support={row['support']}"
        )
    if action_rows:
        print("Action per-class metrics:")
        for row in action_rows:
            print(
                f"  {row['name']}: "
                f"P={row['precision']:.4f} R={row['recall']:.4f} "
                f"F1={row['f1']:.4f} support={row['support']}"
            )

    if not args.no_wandb and WANDB_AVAILABLE:
        payload = {
            f"{args.split}_scenario_f1": s_f1,
            f"{args.split}_scenario_acc": s_acc,
            f"{args.split}_action_f1": a_f1,
            f"{args.split}_action_acc": a_acc,
        }
        payload.update({f"{args.split}_scenario_{k}": v for k, v in scenario_payload.items()})
        payload.update({f"{args.split}_action_{k}": v for k, v in action_payload.items()})
        if args.wandb_run_id:
            if not args.wandb_entity:
                raise ValueError("--wandb-entity is required when using --wandb-run-id")
            try:
                api = wandb.Api()
                source_run = api.run(f"{args.wandb_entity}/{args.wandb_project}/{args.wandb_run_id}")
                for k, v in payload.items():
                    source_run.summary[k] = v
                source_run.summary[f"{args.split}_eval_checkpoint"] = args.checkpoint
                source_run.summary.update()
                print(
                    f"Updated summary on existing run {args.wandb_run_id} "
                    "without modifying run history."
                )
            except Exception as e:
                print(f"Warning: could not update source run summary ({e}).")
        else:
            eval_run_name = args.run_name or f"eval_{Path(args.checkpoint).parent.name}_{args.split}"
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=eval_run_name,
                job_type="eval",
                config={
                    "mode": "eval_only",
                    "split": args.split,
                    "checkpoint": args.checkpoint,
                    "labels_csv": args.labels_csv,
                },
            )
            wandb.log(payload)
            scenario_class_names = [loader.dataset.idx_to_scenario[i] for i in range(len(loader.dataset.scenario_map))]
            wandb.log({
                f"{args.split}_conf_mat_scenario": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=s_labels_all,
                    preds=s_preds_all,
                    class_names=scenario_class_names,
                )
            })
            if len(a_labels_all) > 0:
                action_class_names = [loader.dataset.idx_to_action[i] for i in range(len(loader.dataset.action_map))]
                wandb.log({
                    f"{args.split}_conf_mat_action": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=a_labels_all,
                        preds=a_preds_all,
                        class_names=action_class_names,
                    )
                })
            for k, v in payload.items():
                run.summary[k] = v
            run.finish()


if __name__ == "__main__":
    main()

