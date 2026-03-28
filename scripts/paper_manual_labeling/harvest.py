#!/usr/bin/env python3
"""Aggregate dataset counts and distributions for the manual-labeling paper section."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .common import (
    FULL_LLM_REFINED,
    GOLD_DIR,
    OUTPUT_DIR,
    R001_MERGED,
    R001_REFINED,
    R002_REFINED,
    canonical_action,
)


def _count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as f:
        return sum(1 for _ in f) - 1


def _action_dist_from_llm() -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    n_videos = set()
    with FULL_LLM_REFINED.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            counts[canonical_action(row.get("action", ""))] += 1
            uid = (row.get("video_uid") or "").strip()
            if uid:
                n_videos.add(uid)
    total = sum(counts.values())
    return {
        "path": str(FULL_LLM_REFINED.relative_to(FULL_LLM_REFINED.parents[2])),
        "rows": total,
        "unique_videos": len(n_videos),
        "action_counts": dict(counts),
        "action_fractions": {a: counts[a] / total for a in sorted(counts)},
    }


def _gold_splits() -> Dict[str, Any]:
    splits = ["train.csv", "val.csv", "test.csv"]
    combined_counts: Counter[str] = Counter()
    videos: set[str] = set()
    split_info: Dict[str, Any] = {}
    total_rows = 0
    for name in splits:
        p = GOLD_DIR / name
        c: Counter[str] = Counter()
        sv: set[str] = set()
        with p.open(encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                combined_counts[canonical_action(row.get("action", ""))] += 1
                c[canonical_action(row.get("action", ""))] += 1
                uid = (row.get("video_uid") or "").strip()
                if uid:
                    sv.add(uid)
                    videos.add(uid)
        n = sum(c.values())
        total_rows += n
        split_info[name] = {
            "rows": n,
            "unique_videos": len(sv),
            "action_counts": dict(c),
        }
    return {
        "paths": [str((GOLD_DIR / s).relative_to(GOLD_DIR.parents[2])) for s in splits],
        "rows_total": total_rows,
        "unique_videos_union": len(videos),
        "action_counts": dict(combined_counts),
        "action_fractions": {
            a: combined_counts[a] / total_rows for a in sorted(combined_counts)
        },
        "splits": split_info,
    }


def _verdicts_r001_merged() -> Dict[str, Any] | None:
    if not R001_MERGED.is_file():
        return None
    verdicts: Counter[str] = Counter()
    with R001_MERGED.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            v = (row.get("validated_final_verdict") or "").strip()
            if v:
                verdicts[v] += 1
    total = sum(verdicts.values())
    gold_bad = verdicts.get("Gold", 0) + verdicts.get("Bad", 0)
    rate = verdicts.get("Gold", 0) / gold_bad if gold_bad else None
    return {
        "path": str(R001_MERGED.relative_to(R001_MERGED.parents[2])),
        "rows": total,
        "verdict_counts": dict(verdicts),
        "gold_rate_among_gold_plus_bad": rate,
    }


def _load_refined_verdicts(path: Path, verdict_field: str) -> Counter[str]:
    out: Counter[str] = Counter()
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            v = (row.get(verdict_field) or "").strip()
            if v:
                out[v] += 1
    return out


def _refined_rounds_summary() -> Dict[str, Any]:
    r001_rows = _count_csv_rows(R001_REFINED) if R001_REFINED.is_file() else 0
    r002_rows = _count_csv_rows(R002_REFINED) if R002_REFINED.is_file() else 0
    r001_v = _load_refined_verdicts(R001_REFINED, "status_main") if R001_REFINED.is_file() else Counter()
    r002_v = _load_refined_verdicts(R002_REFINED, "verdict") if R002_REFINED.is_file() else Counter()
    merged_v: Counter[str] = Counter()
    merged_v.update(r001_v)
    merged_v.update(r002_v)
    gb = merged_v.get("Gold", 0) + merged_v.get("Bad", 0)
    return {
        "r001_combined_rows": r001_rows,
        "r002_combined_rows": r002_rows,
        "combined_rows": r001_rows + r002_rows,
        "r001_verdict_counts": dict(r001_v),
        "r002_verdict_counts": dict(r002_v),
        "merged_verdict_counts": dict(merged_v),
        "merged_gold_rate_among_gold_plus_bad": (
            merged_v.get("Gold", 0) / gb if gb else None
        ),
    }


def _doc_new_taxonomy_reference() -> Dict[str, Any]:
    """Static snapshot cited in doc/new_taxonomy.md for reconciliation."""
    return {
        "source_doc": "doc/new_taxonomy.md",
        "full_llm_rows": 355_580,
        "gold_validated_rows_cited": 13_931,
        "bad_rows_cited": 1_984,
        "overall_llm_accuracy_cited": 0.875,
        "per_class_llm_accuracy_cited": {
            "Essential Operation": 0.949,
            "Object Transfer": 0.935,
            "Search": 0.803,
            "Locomotion": 0.782,
            "Stationary": 0.740,
        },
        "note": "Compare merged_refined and r001_merged counts to these; drift expected as exports grow.",
    }


def run_harvest() -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "full_llm_refined": _action_dist_from_llm(),
        "final_gold_splits": _gold_splits(),
        "r001_merged_validation": _verdicts_r001_merged(),
        "refined_exports_rounds": _refined_rounds_summary(),
        "doc_new_taxonomy_reference": _doc_new_taxonomy_reference(),
        "har155_csv_rows": _count_csv_rows(GOLD_DIR / "HAR_dataset_155.csv")
        if (GOLD_DIR / "HAR_dataset_155.csv").is_file()
        else None,
    }
    out_json = OUTPUT_DIR / "paper_stats.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _md_table(rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
    return "\n".join([lines[0], sep] + lines[1:])


def write_paper_stats_md(payload: Dict[str, Any]) -> Path:
    llm = payload["full_llm_refined"]
    gold = payload["final_gold_splits"]
    refined = payload["refined_exports_rounds"]
    lines = [
        "# Manual labeling — harvested statistics (auto-generated)",
        "",
        "Generated by `python -m scripts.paper_manual_labeling.harvest`.",
        "",
        "## Full LLM corpus (`action_labels_llm_clean_refined.csv`)",
        "",
        f"- Rows: **{llm['rows']:,}**",
        f"- Unique videos: **{llm['unique_videos']:,}**",
        "",
        "### Action distribution",
        "",
        _md_table(
            [["Class", "Count", "Fraction"]]
            + [
                [
                    a,
                    str(llm["action_counts"][a]),
                    f"{llm['action_fractions'][a]:.4f}",
                ]
                for a in sorted(llm["action_counts"])
            ]
        ),
        "",
        "## Final gold splits (`final_gold_dataset/{train,val,test}.csv`)",
        "",
        f"- Rows (train+val+test): **{gold['rows_total']:,}**",
        f"- Unique videos (union): **{gold['unique_videos_union']:,}**",
        "",
        "### Per split",
        "",
    ]
    for sn, info in gold["splits"].items():
        lines.append(f"- **{sn}**: {info['rows']:,} rows, {info['unique_videos']} videos")
    lines += [
        "",
        "### Action distribution (splits combined)",
        "",
        _md_table(
            [["Class", "Count", "Fraction"]]
            + [
                [
                    a,
                    str(gold["action_counts"][a]),
                    f"{gold['action_fractions'][a]:.4f}",
                ]
                for a in sorted(gold["action_counts"])
            ]
        ),
        "",
        "## Refined annotation exports (r001 + r002 combined CSVs)",
        "",
        f"- r001 combined rows: **{refined['r001_combined_rows']:,}**",
        f"- r002 combined rows: **{refined['r002_combined_rows']:,}**",
        f"- Total: **{refined['combined_rows']:,}**",
        "",
        "### Merged verdict counts",
        "",
        _md_table(
            [["Verdict", "Count"]]
            + [
                [k, str(v)]
                for k, v in sorted(
                    refined["merged_verdict_counts"].items(),
                    key=lambda x: -x[1],
                )
            ]
        ),
        "",
    ]
    gr = refined.get("merged_gold_rate_among_gold_plus_bad")
    if gr is not None:
        lines.append(
            f"- Gold / (Gold + Bad): **{gr:.4f}** (on merged refined exports)"
        )
    lines += ["", "## Reconciliation note", ""]
    ref = payload["doc_new_taxonomy_reference"]
    lines.append(
        f"See [{ref['source_doc']}]({ref['source_doc']}) for earlier cited totals "
        "(e.g. 13,931 gold rows); current exports may differ as rounds complete."
    )
    path = OUTPUT_DIR / "paper_stats.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
