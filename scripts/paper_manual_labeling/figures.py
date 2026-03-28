#!/usr/bin/env python3
"""Matplotlib figures for the manual-labeling paper section."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .common import OUTPUT_DIR, PROJECT_ROOT

# Precomputed narration redundancy (doc/NARRATION_DEDUPLICATION_ANALYSIS.md)
REDUNDANCY_STAGES: List[Tuple[str, int]] = [
    ("Total rows", 355_580),
    ("Unique raw narration_text", 143_860),
    ("After basic normalization", 136_270),
    ("Estimated unique after near-dup removal", 132_000),
]

# Dedup workload (doc/BATCH_ASSIGNMENT_STRATEGY.md §4.3)
DEDUP_STAGES: List[Tuple[str, int]] = [
    ("Assigned (pre-dedup)", 30_000),
    ("After validated removal", 30_000 - 864),
    ("After within-batch dedup", 30_000 - 864 - 8_466),
    ("After cross-batch dedup (final assigned)", 17_534),
]

EXISTING_FIGURES = [
    (
        "r001_validation_6000",
        PROJECT_ROOT / "data/analysis/r001_validation_6000/figures",
    ),
    (
        "r001_validation_10000",
        PROJECT_ROOT / "data/analysis/r001_validation_10000/figures",
    ),
]


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def figure_class_distribution_old_vs_new(
    paper_stats: Dict[str, Any], out_path: Path
) -> None:
    plt = _ensure_matplotlib()
    llm = paper_stats["full_llm_refined"]["action_counts"]
    gold = paper_stats["final_gold_splits"]["action_counts"]
    classes = sorted(set(llm) | set(gold))
    x = range(len(classes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        [i - w / 2 for i in x],
        [llm.get(c, 0) for c in classes],
        width=w,
        label="Full LLM corpus",
        color="#4C72B0",
    )
    ax.bar(
        [i + w / 2 for i in x],
        [gold.get(c, 0) for c in classes],
        width=w,
        label="Gold train+val+test",
        color="#DD8452",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(classes, rotation=25, ha="right")
    ax.set_ylabel("Row count")
    ax.legend()
    ax.set_title("Action class counts: full LLM vs gold splits")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def figure_narration_redundancy(out_path: Path) -> None:
    plt = _ensure_matplotlib()
    labels = [s[0] for s in REDUNDANCY_STAGES]
    vals = [s[1] for s in REDUNDANCY_STAGES]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(labels[::-1], vals[::-1], color="#55A868")
    ax.set_xlabel("Count")
    ax.set_title("Narration redundancy funnel (full LLM corpus)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def figure_dedup_workload(out_path: Path) -> None:
    plt = _ensure_matplotlib()
    labels = [s[0] for s in DEDUP_STAGES]
    vals = [s[1] for s in DEDUP_STAGES]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(labels[::-1], vals[::-1], color="#C44E52")
    ax.set_xlabel("Row count")
    ax.set_title("Round-1 validation pool: deduplication stages")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def mirror_existing_figures(dest_subdir: Path) -> List[str]:
    copied: List[str] = []
    for name, src_dir in EXISTING_FIGURES:
        if not src_dir.is_dir():
            continue
        sub = dest_subdir / name
        sub.mkdir(parents=True, exist_ok=True)
        for png in sorted(src_dir.glob("*.png")):
            shutil.copy2(png, sub / png.name)
            copied.append(str((sub / png.name).relative_to(PROJECT_ROOT)))
    return copied


def run_figures(paper_stats: Dict[str, Any]) -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    figure_class_distribution_old_vs_new(
        paper_stats, fig_dir / "fig_class_counts_llm_vs_gold.png"
    )
    figure_narration_redundancy(fig_dir / "fig_narration_redundancy_funnel.png")
    figure_dedup_workload(fig_dir / "fig_round1_dedup_stages.png")
    mirrored = mirror_existing_figures(fig_dir / "from_repo")
    summary = {
        "new_figures": [
            str((fig_dir / "fig_class_counts_llm_vs_gold.png").relative_to(PROJECT_ROOT)),
            str((fig_dir / "fig_narration_redundancy_funnel.png").relative_to(PROJECT_ROOT)),
            str((fig_dir / "fig_round1_dedup_stages.png").relative_to(PROJECT_ROOT)),
        ],
        "mirrored_r001_figures": mirrored,
    }
    (OUTPUT_DIR / "figures_index.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
