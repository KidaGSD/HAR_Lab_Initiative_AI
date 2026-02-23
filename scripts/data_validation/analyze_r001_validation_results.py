#!/usr/bin/env python3
"""
Generate numerical, visualization, and case analyses for Round-1 merged validation results.

Input:
  - data/annotation_rounds/r001_merged_validation/round1_validated_rows_with_decisions.csv

Outputs (under --output-dir):
  - figures/*.png
  - tables/*.csv
  - r001_validation_analysis_en.md
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
INPUT_DEFAULT = PROJECT_ROOT / "data/annotation_rounds/r001_merged_validation/round1_validated_rows_with_decisions.csv"
OUTPUT_DEFAULT = PROJECT_ROOT / "data/analysis/r001_validation_6000"

VERDICT_ORDER = ["Gold", "Bad", "Skip", "Delete Row"]
ACTION_ORDER = ["Object Transfer", "Stationary", "Essential Operation", "Locomotion", "Search"]

sns.set_theme(style="whitegrid")


def ensure_dirs(base: Path) -> Tuple[Path, Path]:
    fig_dir = base / "figures"
    table_dir = base / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, table_dir


def clean_str_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def first_verb_from_norm_text(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return ""
    toks = t.split()
    while toks and toks[0] in {
        "c",
        "o",
        "man",
        "woman",
        "person",
        "the",
        "a",
        "an",
        "lady",
        "boy",
        "girl",
        "child",
        "driver",
        "x",
        "y",
        "z",
        "kid",
    }:
        toks = toks[1:]
    return toks[0] if toks else ""


def first_match_or_empty(series: pd.Series, pattern: str) -> pd.Series:
    return series.str.extract(pattern, expand=False).fillna("")


def compute_entropy(values: Iterable[str]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    vc = pd.Series(vals).value_counts(normalize=True)
    return float(-(vc * np.log2(vc + 1e-12)).sum())


def make_verdict_distribution(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> pd.DataFrame:
    counts = (
        df["validated_final_verdict"]
        .value_counts()
        .reindex(VERDICT_ORDER, fill_value=0)
        .rename_axis("verdict")
        .reset_index(name="count")
    )
    counts["pct"] = counts["count"] / counts["count"].sum() * 100.0
    counts.to_csv(table_dir / "verdict_distribution.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    ax = sns.barplot(data=counts, x="verdict", y="count", order=VERDICT_ORDER, palette="Set2")
    for i, r in counts.iterrows():
        ax.text(i, r["count"] + max(5, counts["count"].max() * 0.01), f'{int(r["count"])}\n({r["pct"]:.1f}%)', ha="center", va="bottom", fontsize=9)
    ax.set_title("Validation Verdict Distribution (n=6,101)")
    ax.set_xlabel("Final Verdict")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig01_verdict_distribution.png", dpi=180)
    plt.close()
    return counts


def make_batch_stacked(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> pd.DataFrame:
    tmp = df.copy()
    tmp["batch"] = pd.to_numeric(tmp["batch"], errors="coerce")
    batch_tbl = (
        tmp.pivot_table(index="batch", columns="validated_final_verdict", values="video_uid", aggfunc="count", fill_value=0)
        .reindex(columns=VERDICT_ORDER, fill_value=0)
        .sort_index()
    )
    batch_tbl["Gold+Bad"] = batch_tbl["Gold"] + batch_tbl["Bad"]
    batch_tbl["gold_rate_gold_bad_pct"] = np.where(
        batch_tbl["Gold+Bad"] > 0,
        batch_tbl["Gold"] / batch_tbl["Gold+Bad"] * 100.0,
        np.nan,
    )
    batch_tbl.to_csv(table_dir / "batch_verdict_summary.csv")

    plot_tbl = batch_tbl[VERDICT_ORDER]
    plt.figure(figsize=(10, 5.2))
    plot_tbl.plot(kind="bar", stacked=True, colormap="Set3", edgecolor="black", linewidth=0.2)
    plt.title("Verdict Composition by Batch")
    plt.xlabel("Batch")
    plt.ylabel("Count")
    plt.legend(title="Verdict", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig02_batch_stacked_verdicts.png", dpi=180)
    plt.close()
    return batch_tbl.reset_index()


def make_class_bad_rate(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> pd.DataFrame:
    gb = df[df["validated_final_verdict"].isin(["Gold", "Bad"])].copy()
    t = gb.pivot_table(index="action", columns="validated_final_verdict", values="video_uid", aggfunc="count", fill_value=0)
    t = t.reindex(ACTION_ORDER, fill_value=0)
    t["total"] = t["Gold"] + t["Bad"]
    t["bad_rate_pct"] = np.where(t["total"] > 0, t["Bad"] / t["total"] * 100.0, 0.0)
    out = t.reset_index().sort_values("bad_rate_pct", ascending=True)
    out.to_csv(table_dir / "class_bad_rate.csv", index=False)

    plt.figure(figsize=(8.2, 4.8))
    ax = sns.barplot(data=out, x="bad_rate_pct", y="action", palette="mako")
    for _, r in out.iterrows():
        ax.text(r["bad_rate_pct"] + 0.4, out.index[out["action"] == r["action"]][0], f'{r["Bad"]}/{r["total"]}', va="center", fontsize=9)
    ax.set_title("Bad Rate by Original LLM Action Class (Gold/Bad only)")
    ax.set_xlabel("Bad Rate (%)")
    ax.set_ylabel("Original Action")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig03_class_bad_rate.png", dpi=180)
    plt.close()
    return out


def make_bad_transition_heatmap(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> pd.DataFrame:
    bad = df[df["validated_final_verdict"] == "Bad"].copy()
    cm = (
        bad.pivot_table(index="action", columns="validated_final_action", values="video_uid", aggfunc="count", fill_value=0)
        .reindex(index=ACTION_ORDER, columns=ACTION_ORDER, fill_value=0)
    )
    cm.to_csv(table_dir / "bad_transition_matrix.csv")

    plt.figure(figsize=(7.5, 6.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", cbar_kws={"label": "Count"})
    plt.title("Bad-Only Transition Matrix (Original Action -> Final Corrected Action)")
    plt.xlabel("Final Corrected Action")
    plt.ylabel("Original LLM Action")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig04_bad_transition_heatmap.png", dpi=180)
    plt.close()
    return cm.reset_index()


def make_scenario_bad_rate(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> pd.DataFrame:
    gb = df[df["validated_final_verdict"].isin(["Gold", "Bad"])].copy()
    t = gb.pivot_table(index="scenario", columns="validated_final_verdict", values="video_uid", aggfunc="count", fill_value=0)
    t["total"] = t["Gold"] + t["Bad"]
    t["bad_rate_pct"] = np.where(t["total"] > 0, t["Bad"] / t["total"] * 100.0, 0.0)
    out = t.reset_index().sort_values("bad_rate_pct", ascending=False)
    out.to_csv(table_dir / "scenario_bad_rate.csv", index=False)

    plt.figure(figsize=(9, 5.2))
    ax = sns.barplot(data=out, x="bad_rate_pct", y="scenario", palette="rocket")
    for _, r in out.iterrows():
        ax.text(r["bad_rate_pct"] + 0.25, out.index[out["scenario"] == r["scenario"]][0], f'n={int(r["total"])}', va="center", fontsize=9)
    ax.set_title("Bad Rate by Scenario (Gold/Bad only)")
    ax.set_xlabel("Bad Rate (%)")
    ax.set_ylabel("Scenario")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig05_scenario_bad_rate.png", dpi=180)
    plt.close()
    return out


def make_annotator_scatter(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> pd.DataFrame:
    gb = df[df["validated_final_verdict"].isin(["Gold", "Bad"])].copy()
    t = gb.pivot_table(index="validated_annotator", columns="validated_final_verdict", values="video_uid", aggfunc="count", fill_value=0)
    t["total"] = t["Gold"] + t["Bad"]
    t["bad_rate_pct"] = np.where(t["total"] > 0, t["Bad"] / t["total"] * 100.0, 0.0)
    out = t.reset_index().sort_values("total", ascending=False)
    out.to_csv(table_dir / "annotator_quality_summary.csv", index=False)

    plt.figure(figsize=(8.2, 5.2))
    ax = sns.scatterplot(data=out, x="total", y="bad_rate_pct", size="total", hue="bad_rate_pct", palette="viridis", sizes=(80, 650), legend=False)
    for _, r in out.iterrows():
        ax.text(r["total"] + 6, r["bad_rate_pct"], str(r["validated_annotator"]), fontsize=9, va="center")
    ax.set_title("Annotator Volume vs Bad Rate (Gold/Bad only)")
    ax.set_xlabel("Gold+Bad Labeled Rows")
    ax.set_ylabel("Bad Rate (%)")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig06_annotator_bad_rate_volume.png", dpi=180)
    plt.close()
    return out


def make_lead_time_plot(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> pd.DataFrame:
    tmp = df.copy()
    tmp["validated_lead_time"] = pd.to_numeric(tmp["validated_lead_time"], errors="coerce")
    q = tmp.groupby("validated_final_verdict")["validated_lead_time"].agg(
        n="count",
        mean="mean",
        median="median",
        p90=lambda x: np.nanpercentile(x, 90),
        max="max",
    ).reindex(VERDICT_ORDER)
    q.to_csv(table_dir / "lead_time_by_verdict.csv")

    plt.figure(figsize=(8.4, 5.4))
    plot_tmp = tmp[tmp["validated_final_verdict"].isin(VERDICT_ORDER)].copy()
    sns.boxplot(data=plot_tmp, x="validated_final_verdict", y="validated_lead_time", order=VERDICT_ORDER, showfliers=False, palette="Pastel1")
    plt.yscale("log")
    plt.title("Lead Time by Final Verdict (log scale)")
    plt.xlabel("Final Verdict")
    plt.ylabel("Lead Time (seconds, log scale)")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig07_lead_time_by_verdict_box.png", dpi=180)
    plt.close()
    return q.reset_index()


def make_ambiguity_analysis(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> pd.DataFrame:
    gb = df[df["validated_final_verdict"].isin(["Gold", "Bad"])].copy()
    grp = gb.groupby("normalized_narration_for_analysis")

    rows: List[Dict[str, object]] = []
    for narr, sub in grp:
        narr = (narr or "").strip()
        if not narr:
            continue
        labels = sub["validated_final_action"].astype(str).tolist()
        vc = pd.Series(labels).value_counts()
        support = int(len(labels))
        n_labels = int(vc.shape[0])
        dominant_label = vc.index[0]
        dominant_count = int(vc.iloc[0])
        dominant_ratio = float(dominant_count / support) if support else 0.0
        entropy = compute_entropy(labels)
        rows.append(
            {
                "normalized_narration_for_analysis": narr,
                "support": support,
                "n_labels": n_labels,
                "dominant_label": dominant_label,
                "dominant_count": dominant_count,
                "dominant_ratio": dominant_ratio,
                "label_entropy_bits": entropy,
                "label_histogram": "; ".join([f"{k}:{int(v)}" for k, v in vc.items()]),
            }
        )

    out = pd.DataFrame(rows).sort_values(["support", "dominant_ratio"], ascending=[False, True])
    out.to_csv(table_dir / "narration_ambiguity_summary.csv", index=False)

    scat = out[out["support"] >= 3].copy()
    plt.figure(figsize=(8.6, 5.4))
    sns.scatterplot(data=scat, x="support", y="dominant_ratio", hue="n_labels", palette="viridis", alpha=0.75)
    plt.axhline(0.67, color="red", linestyle="--", linewidth=1, label="0.67 majority threshold")
    plt.title("Narration-Level Label Stability (Gold/Bad final actions)")
    plt.xlabel("Support (rows per normalized narration)")
    plt.ylabel("Dominant Class Ratio")
    plt.legend(title="# Final Classes", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig08_ambiguity_support_vs_dominance.png", dpi=180)
    plt.close()

    amb_top = out[(out["n_labels"] >= 2) & (out["support"] >= 3)].head(20).copy()
    amb_top.to_csv(table_dir / "top_ambiguous_narrations.csv", index=False)

    plt.figure(figsize=(10, 6.5))
    plot_amb = amb_top.sort_values("support", ascending=True)
    sns.barplot(data=plot_amb, x="support", y="normalized_narration_for_analysis", hue="n_labels", dodge=False, palette="flare")
    plt.title("Top Ambiguous Narrations (support >= 3)")
    plt.xlabel("Support")
    plt.ylabel("Normalized Narration")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig09_top_ambiguous_narrations.png", dpi=180)
    plt.close()
    return out


def make_pattern_rule_eval(df: pd.DataFrame, fig_dir: Path, table_dir: Path) -> pd.DataFrame:
    gb = df[df["validated_final_verdict"].isin(["Gold", "Bad"])].copy()
    rules = [
        ("looks around -> Search", r"\blooks\s+around\b", "Search"),
        ("adjusts the camera -> Essential Operation", r"\badjusts\s+the\s+camera\b", "Essential Operation"),
        ("looks at person -> Stationary", r"\blooks\s+at\s+person\b", "Stationary"),
        ("throws .* ball -> Object Transfer", r"\bthrows?\b.*\bball\b", "Object Transfer"),
    ]
    rows = []
    narr_l = clean_str_series(gb["narration_text"]).str.lower()
    for name, pat, target in rules:
        mask = narr_l.str.contains(pat, regex=True)
        sub = gb[mask]
        support = int(sub.shape[0])
        if support == 0:
            rows.append(
                {
                    "rule_name": name,
                    "pattern": pat,
                    "target_label": target,
                    "support": 0,
                    "correct_target_count": 0,
                    "precision": np.nan,
                    "final_action_distribution": "",
                }
            )
            continue
        correct = int((sub["validated_final_action"] == target).sum())
        dist = sub["validated_final_action"].value_counts().to_dict()
        rows.append(
            {
                "rule_name": name,
                "pattern": pat,
                "target_label": target,
                "support": support,
                "correct_target_count": correct,
                "precision": correct / support,
                "final_action_distribution": "; ".join([f"{k}:{v}" for k, v in dist.items()]),
            }
        )

    out = pd.DataFrame(rows).sort_values("support", ascending=False)
    out.to_csv(table_dir / "candidate_rule_precision.csv", index=False)

    plot_tbl = out.copy()
    plot_tbl["precision_pct"] = plot_tbl["precision"] * 100.0

    plt.figure(figsize=(10.5, 5.2))
    ax = sns.barplot(data=plot_tbl, x="precision_pct", y="rule_name", palette="crest")
    for _, r in plot_tbl.iterrows():
        txt = f'support={int(r["support"])}'
        ax.text(min(99.2, (r["precision_pct"] if pd.notna(r["precision_pct"]) else 0) + 1.2), plot_tbl.index[plot_tbl["rule_name"] == r["rule_name"]][0], txt, va="center", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_title("Candidate Rule Precision on Gold/Bad Validation Rows")
    ax.set_xlabel("Precision (%)")
    ax.set_ylabel("Rule")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig10_candidate_rule_precision.png", dpi=180)
    plt.close()
    return out


def export_case_tables(df: pd.DataFrame, table_dir: Path) -> Dict[str, pd.DataFrame]:
    gb = df[df["validated_final_verdict"].isin(["Gold", "Bad"])].copy()
    bad = gb[gb["validated_final_verdict"] == "Bad"].copy()

    trans = (
        bad.groupby(["action", "validated_final_action"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    top_trans = trans.head(15).copy()
    top_trans.to_csv(table_dir / "top_bad_transitions.csv", index=False)

    # Representative cases per top transition
    case_rows = []
    for _, r in top_trans.head(8).iterrows():
        src, dst = r["action"], r["validated_final_action"]
        sub = bad[(bad["action"] == src) & (bad["validated_final_action"] == dst)].head(8)
        for _, x in sub.iterrows():
            case_rows.append(
                {
                    "transition": f"{src} -> {dst}",
                    "batch": x["batch"],
                    "scenario": x["scenario"],
                    "annotator": x["validated_annotator"],
                    "narration_text": x["narration_text"],
                    "original_action": x["action"],
                    "final_action": x["validated_final_action"],
                    "reasoning_excerpt": str(x["validated_reasoning"])[:220],
                }
            )
    case_df = pd.DataFrame(case_rows)
    case_df.to_csv(table_dir / "case_examples_top_transitions.csv", index=False)

    # Mixed-outcome narrations
    narr_grp = gb.groupby("normalized_narration_for_analysis")
    mixed_rows = []
    for narr, sub in narr_grp:
        narr = str(narr).strip()
        if not narr:
            continue
        actions = sub["validated_final_action"].value_counts()
        verdicts = sub["validated_final_verdict"].value_counts()
        if actions.shape[0] <= 1:
            continue
        mixed_rows.append(
            {
                "normalized_narration_for_analysis": narr,
                "support": int(sub.shape[0]),
                "final_action_histogram": "; ".join([f"{k}:{int(v)}" for k, v in actions.items()]),
                "verdict_histogram": "; ".join([f"{k}:{int(v)}" for k, v in verdicts.items()]),
            }
        )
    mixed_df = pd.DataFrame(mixed_rows).sort_values("support", ascending=False).head(50)
    mixed_df.to_csv(table_dir / "top_mixed_outcome_narrations.csv", index=False)

    return {
        "top_bad_transitions": top_trans,
        "case_examples_top_transitions": case_df,
        "top_mixed_outcome_narrations": mixed_df,
    }


def write_markdown_report(
    out_md: Path,
    verdict_counts: pd.DataFrame,
    class_bad: pd.DataFrame,
    scenario_bad: pd.DataFrame,
    annotator_q: pd.DataFrame,
    rule_eval: pd.DataFrame,
    ambiguity_df: pd.DataFrame,
    case_tables: Dict[str, pd.DataFrame],
) -> None:
    total = int(verdict_counts["count"].sum())
    gold = int(verdict_counts.loc[verdict_counts["verdict"] == "Gold", "count"].iloc[0])
    bad = int(verdict_counts.loc[verdict_counts["verdict"] == "Bad", "count"].iloc[0])
    skip = int(verdict_counts.loc[verdict_counts["verdict"] == "Skip", "count"].iloc[0])
    delete = int(verdict_counts.loc[verdict_counts["verdict"] == "Delete Row", "count"].iloc[0])
    gold_rate = gold / (gold + bad) * 100.0 if (gold + bad) else float("nan")

    class_top = class_bad.sort_values("bad_rate_pct", ascending=False).head(5)
    scen_top = scenario_bad.head(5)
    amb_groups = ambiguity_df[(ambiguity_df["n_labels"] >= 2)].shape[0]
    amb_support3 = ambiguity_df[(ambiguity_df["n_labels"] >= 2) & (ambiguity_df["support"] >= 3)]

    # Majority policy quick summary
    majority_cand = amb_support3[amb_support3["dominant_ratio"] >= 0.67]
    majority_rows_flip = int((majority_cand["support"] - majority_cand["dominant_count"]).sum()) if not majority_cand.empty else 0

    lines: List[str] = []
    lines.append("# Round-1 Validation Analysis (English)")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Input file: `data/annotation_rounds/r001_merged_validation/round1_validated_rows_with_decisions.csv`")
    lines.append(f"- Rows analyzed: `{total:,}`")
    lines.append("- This report includes visualization, numerical summaries, and concrete case-level error analysis.")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Final verdict distribution: Gold `{gold:,}`, Bad `{bad:,}`, Skip `{skip:,}`, Delete Row `{delete:,}`.")
    lines.append(f"- Gold rate among Gold+Bad: `{gold_rate:.2f}%`.")
    lines.append("- Errors are boundary-driven (especially around `Stationary`, `Essential Operation`, and `Object Transfer`), not random noise.")
    lines.append(f"- Ambiguous normalized narrations (>=2 final classes): `{amb_groups}` groups.")
    lines.append(
        f"- Conservative majority policy (`support>=3`, `dominant_ratio>=0.67`) would auto-resolve `{majority_cand.shape[0]}` narration groups with an estimated `{majority_rows_flip}` relabels."
    )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    fig_titles = [
        ("figures/fig01_verdict_distribution.png", "Verdict distribution"),
        ("figures/fig02_batch_stacked_verdicts.png", "Verdict composition by batch"),
        ("figures/fig03_class_bad_rate.png", "Bad rate by original action class"),
        ("figures/fig04_bad_transition_heatmap.png", "Bad-only transition matrix"),
        ("figures/fig05_scenario_bad_rate.png", "Bad rate by scenario"),
        ("figures/fig06_annotator_bad_rate_volume.png", "Annotator volume vs bad rate"),
        ("figures/fig07_lead_time_by_verdict_box.png", "Lead time by verdict"),
        ("figures/fig08_ambiguity_support_vs_dominance.png", "Narration-level ambiguity scatter"),
        ("figures/fig09_top_ambiguous_narrations.png", "Top ambiguous narrations"),
        ("figures/fig10_candidate_rule_precision.png", "Candidate rule precision"),
    ]
    for fig_path, title in fig_titles:
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}]({fig_path})")
        lines.append("")
    lines.append("")
    lines.append("## Numerical Highlights")
    lines.append("")
    lines.append("### Highest Bad-Rate Classes")
    lines.append("")
    lines.append("| Original Action | Gold | Bad | Bad Rate (%) |")
    lines.append("|---|---:|---:|---:|")
    for _, r in class_top.iterrows():
        lines.append(f'| {r["action"]} | {int(r["Gold"])} | {int(r["Bad"])} | {r["bad_rate_pct"]:.2f} |')
    lines.append("")
    lines.append("### Highest Bad-Rate Scenarios")
    lines.append("")
    lines.append("| Scenario | Gold | Bad | Bad Rate (%) |")
    lines.append("|---|---:|---:|---:|")
    for _, r in scen_top.iterrows():
        lines.append(f'| {r["scenario"]} | {int(r["Gold"])} | {int(r["Bad"])} | {r["bad_rate_pct"]:.2f} |')
    lines.append("")
    lines.append("### Candidate Rule Precision (Gold/Bad rows)")
    lines.append("")
    lines.append("| Rule | Support | Precision (%) |")
    lines.append("|---|---:|---:|")
    for _, r in rule_eval.sort_values("support", ascending=False).iterrows():
        prec = "NA" if pd.isna(r["precision"]) else f'{r["precision"]*100:.2f}'
        lines.append(f'| {r["rule_name"]} | {int(r["support"])} | {prec} |')
    lines.append("")
    lines.append("## Case Analysis")
    lines.append("")
    lines.append("### Top Error Transitions")
    lines.append("")
    lines.append("| Transition | Count |")
    lines.append("|---|---:|")
    for _, r in case_tables["top_bad_transitions"].head(10).iterrows():
        lines.append(f'| {r["action"]} -> {r["validated_final_action"]} | {int(r["count"])} |')
    lines.append("")
    lines.append("### Representative Cases")
    lines.append("")
    lines.append("See `tables/case_examples_top_transitions.csv` for concrete narration-level examples, scenarios, and reasoning excerpts.")
    lines.append("")
    lines.append("### Narration-Level Ambiguity")
    lines.append("")
    lines.append("- High-support ambiguous narrations are captured in `tables/top_ambiguous_narrations.csv`.")
    lines.append("- Mixed-outcome narrations (same normalized narration with multiple final classes) are in `tables/top_mixed_outcome_narrations.csv`.")
    lines.append("")
    lines.append("## Interpretation for Next Iteration")
    lines.append("")
    lines.append("1. Keep high-precision rules (`looks around -> Search`, `adjusts the camera -> Essential Operation`) before fine-tuning.")
    lines.append("2. Do not hard-code low-precision rules (`looks at person -> Stationary`) without additional context features.")
    lines.append("3. Use majority resolution only with support and dominance thresholds to avoid over-correction.")
    lines.append("4. Prioritize targeted guideline updates for class boundaries where transitions concentrate.")
    lines.append("")
    lines.append("## Generated Data Assets")
    lines.append("")
    lines.append("- `tables/verdict_distribution.csv`")
    lines.append("- `tables/batch_verdict_summary.csv`")
    lines.append("- `tables/class_bad_rate.csv`")
    lines.append("- `tables/bad_transition_matrix.csv`")
    lines.append("- `tables/scenario_bad_rate.csv`")
    lines.append("- `tables/annotator_quality_summary.csv`")
    lines.append("- `tables/lead_time_by_verdict.csv`")
    lines.append("- `tables/narration_ambiguity_summary.csv`")
    lines.append("- `tables/top_ambiguous_narrations.csv`")
    lines.append("- `tables/candidate_rule_precision.csv`")
    lines.append("- `tables/top_bad_transitions.csv`")
    lines.append("- `tables/case_examples_top_transitions.csv`")
    lines.append("- `tables/top_mixed_outcome_narrations.csv`")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze ~6,000 Round-1 validation rows with figures, numeric tables, and case analysis.")
    p.add_argument("--input-csv", default=str(INPUT_DEFAULT))
    p.add_argument("--output-dir", default=str(OUTPUT_DEFAULT))
    return p


def main() -> int:
    args = build_parser().parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    fig_dir, table_dir = ensure_dirs(output_dir)
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)

    # Basic cleanup
    for col in [
        "validated_final_verdict",
        "validated_final_action",
        "validated_corrected_action",
        "action",
        "scenario",
        "batch",
        "validated_annotator",
        "normalized_narration_for_analysis",
        "narration_text",
        "validated_lead_time",
    ]:
        if col in df.columns:
            df[col] = clean_str_series(df[col])
        else:
            df[col] = ""

    # Keep only rows with an explicit final verdict.
    df = df[df["validated_final_verdict"].isin(VERDICT_ORDER)].copy()

    verdict_counts = make_verdict_distribution(df, fig_dir, table_dir)
    _batch_tbl = make_batch_stacked(df, fig_dir, table_dir)
    class_bad = make_class_bad_rate(df, fig_dir, table_dir)
    _bad_cm = make_bad_transition_heatmap(df, fig_dir, table_dir)
    scenario_bad = make_scenario_bad_rate(df, fig_dir, table_dir)
    annotator_q = make_annotator_scatter(df, fig_dir, table_dir)
    _lead = make_lead_time_plot(df, fig_dir, table_dir)
    ambiguity_df = make_ambiguity_analysis(df, fig_dir, table_dir)
    rule_eval = make_pattern_rule_eval(df, fig_dir, table_dir)
    case_tables = export_case_tables(df, table_dir)

    report_path = output_dir / "r001_validation_analysis_en.md"
    write_markdown_report(
        out_md=report_path,
        verdict_counts=verdict_counts,
        class_bad=class_bad,
        scenario_bad=scenario_bad,
        annotator_q=annotator_q,
        rule_eval=rule_eval,
        ambiguity_df=ambiguity_df,
        case_tables=case_tables,
    )

    print("Analysis complete.")
    print(f"- Input: {input_csv}")
    print(f"- Output dir: {output_dir}")
    print(f"- Report: {report_path}")
    print(f"- Figures: {fig_dir}")
    print(f"- Tables: {table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
