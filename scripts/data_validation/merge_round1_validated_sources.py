#!/usr/bin/env python3
"""
Merge multiple validated Label Studio CSV exports into the Round-1 30,000 assigned CSV.

This script is non-destructive:
- It never modifies the source CSVs.
- It writes all outputs into a new output directory.

Inputs:
- Base Round-1 assigned CSV (30,000 rows), typically:
  data/annotation_rounds/r001_numerical/round1_assigned_only.csv
- One or more validated export CSV files from data/1000_validated/

Outputs:
1) Merged 30,000 CSV with validation columns appended.
2) Validated subset (all matched rows).
3) Validated subset with explicit decisions (Gold/Bad/Skip/Delete Row).
4) Deep-analysis subset: unique normalized narration + final action (Gold/Bad only).
5) Conflict log for rows with multiple validation candidates.
6) Unmatched validation rows log.
7) JSON summary with key counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

BASE_DEFAULT = str(PROJECT_ROOT / "data/annotation_rounds/r001_numerical/round1_assigned_only.csv")
VALIDATED_DEFAULTS = [
    str(PROJECT_ROOT / "data/1000_validated/2400_project-7-at-2026-02-23-17-18-d8daa73b.csv"),
    str(PROJECT_ROOT / "data/1000_validated/Deduped_2500_project-8-at-2026-02-23-17-26-c66f8a5c.csv"),
    str(PROJECT_ROOT / "data/1000_validated/Batch12_Finished_project-1-at-2026-02-23-02-46-283f88ad.csv"),
]
OUTPUT_DIR_DEFAULT = str(PROJECT_ROOT / "data/annotation_rounds/r001_merged_validation")

VALID_VERDICTS = {"Gold", "Bad", "Skip", "Delete Row"}


def clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_iso_dt(value: str) -> datetime:
    s = clean(value)
    if not s:
        return datetime.min
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def normalize_ts(value: str) -> str:
    s = clean(value)
    if not s:
        return ""
    try:
        d = Decimal(s)
        out = format(d.normalize(), "f")
        if out == "-0":
            return "0"
        return out
    except (InvalidOperation, ValueError):
        return s


def normalize_narration_for_analysis(text: str) -> str:
    t = clean(text).lower()
    t = re.sub(r"[.,;:!?]+$", "", t)
    t = re.sub(r"#\w+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def canonical_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (
        clean(row.get("video_uid")),
        normalize_ts(row.get("timestamp_sec")),
        clean(row.get("narration_text")),
        clean(row.get("action")),
    )


def row_is_blank(row: Dict[str, str]) -> bool:
    return all(not clean(v) for v in row.values())


def safe_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    i = 2
    while True:
        candidate = path.with_name(f"{stem}_v{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def source_priority(source_name: str) -> int:
    s = source_name.lower()
    if "deduped_2500" in s:
        return 3
    if "2400_project-7" in s:
        return 2
    if "batch12" in s or "project-1" in s:
        return 1
    return 0


def derive_verdict(status_main: str, status_secondary: str) -> str:
    main = clean(status_main)
    secondary = clean(status_secondary)
    if secondary in {"Skip", "Delete Row"}:
        return secondary
    if main in {"Gold", "Bad"}:
        return main
    return ""


@dataclass(frozen=True, order=True)
class CandidateScore:
    updated_at: datetime
    created_at: datetime
    source_pri: int
    source_rank: int
    row_seq: int


def candidate_score(row: Dict[str, str]) -> CandidateScore:
    return CandidateScore(
        updated_at=parse_iso_dt(row.get("updated_at", "")),
        created_at=parse_iso_dt(row.get("created_at", "")),
        source_pri=source_priority(row.get("_source_file", "")),
        source_rank=int(row.get("_source_rank", "0")),
        row_seq=int(row.get("_source_row_seq", "0")),
    )


def pick_best(rows: List[Dict[str, str]]) -> Dict[str, str]:
    return max(rows, key=candidate_score)


def load_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows: List[Dict[str, str]] = []
        for raw in reader:
            row: Dict[str, str] = {}
            for k, v in raw.items():
                if k is None:
                    continue
                row[k] = clean(v)
            if row_is_blank(row):
                continue
            rows.append(row)
    return rows, fieldnames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge multiple validated exports into Round-1 30k CSV and generate analysis subsets."
    )
    parser.add_argument("--base-csv", default=BASE_DEFAULT)
    parser.add_argument(
        "--validated-csv",
        action="append",
        default=[],
        help="Path to validated CSV. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    base_csv = Path(args.base_csv)
    validated_csvs = [Path(p) for p in (args.validated_csv or [])]
    if not validated_csvs:
        validated_csvs = [Path(p) for p in VALIDATED_DEFAULTS]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not base_csv.exists():
        raise FileNotFoundError(f"Base CSV not found: {base_csv}")
    for p in validated_csvs:
        if not p.exists():
            raise FileNotFoundError(f"Validated CSV not found: {p}")

    base_rows, base_fields = load_csv_rows(base_csv)
    if not base_rows:
        raise ValueError(f"Base CSV has no rows: {base_csv}")

    required_base_cols = {"video_uid", "timestamp_sec", "narration_text", "action"}
    missing_base = [c for c in required_base_cols if c not in base_fields]
    if missing_base:
        raise ValueError(f"Base CSV missing required columns: {missing_base}")

    base_key_to_index: Dict[Tuple[str, str, str, str], int] = {}
    for i, row in enumerate(base_rows):
        k = canonical_key(row)
        if k in base_key_to_index:
            raise ValueError(f"Base key is not unique for row {i}: {k}")
        base_key_to_index[k] = i

    validated_raw_rows: List[Dict[str, str]] = []
    validated_raw_count = 0
    for src_rank, csv_path in enumerate(validated_csvs):
        rows, _ = load_csv_rows(csv_path)
        validated_raw_count += len(rows)
        for seq, row in enumerate(rows):
            normalized = dict(row)
            normalized.setdefault("status", "")
            normalized.setdefault("status_main", "")
            normalized.setdefault("status_secondary", "")
            normalized.setdefault("corrected_action", "")
            normalized.setdefault("id", "")
            normalized.setdefault("annotation_id", "")
            normalized.setdefault("annotator", "")
            normalized.setdefault("created_at", "")
            normalized.setdefault("updated_at", "")
            normalized.setdefault("lead_time", "")
            normalized.setdefault("reasoning", "")
            normalized["_source_file"] = csv_path.name
            normalized["_source_rank"] = str(src_rank)
            normalized["_source_row_seq"] = str(seq)
            validated_raw_rows.append(normalized)

    # Step 1: de-duplicate by validation ID (if duplicate IDs exist across sources).
    by_id: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    synthetic_counter = 0
    for row in validated_raw_rows:
        rid = clean(row.get("id"))
        if not rid:
            synthetic_counter += 1
            rid = f"__missing_id__{synthetic_counter}"
        by_id[rid].append(row)

    validated_id_deduped: List[Dict[str, str]] = []
    duplicate_id_groups = 0
    for rid, rows in by_id.items():
        if len(rows) > 1:
            duplicate_id_groups += 1
        best = pick_best(rows)
        best = dict(best)
        best["_resolved_id"] = rid
        validated_id_deduped.append(best)

    # Step 2: map to base keys and resolve per-key conflicts.
    matched_by_key: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    unmatched_rows: List[Dict[str, str]] = []

    for row in validated_id_deduped:
        k = canonical_key(row)
        if k not in base_key_to_index:
            extra = dict(row)
            extra["unmatched_reason"] = "key_not_in_base"
            unmatched_rows.append(extra)
            continue
        matched_by_key[k].append(row)

    selected_by_key: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    conflict_log_rows: List[Dict[str, str]] = []
    conflict_key_count = 0
    for key, candidates in matched_by_key.items():
        chosen = pick_best(candidates)
        selected_by_key[key] = chosen
        if len(candidates) > 1:
            conflict_key_count += 1
        for cand in sorted(candidates, key=candidate_score, reverse=True):
            out = dict(cand)
            out["is_selected"] = "1" if cand is chosen else "0"
            out["conflict_group_size"] = str(len(candidates))
            out["base_video_uid"] = key[0]
            out["base_timestamp_sec"] = key[1]
            out["base_narration_text"] = key[2]
            out["base_action"] = key[3]
            conflict_log_rows.append(out)

    # Step 3: append validation columns to 30k base rows.
    appended_cols = [
        "validation_matched",
        "validation_candidate_count",
        "validation_conflict_flag",
        "validated_source_file",
        "validated_id",
        "validated_annotation_id",
        "validated_annotator",
        "validated_created_at",
        "validated_updated_at",
        "validated_lead_time",
        "validated_status_raw",
        "validated_status_main",
        "validated_status_secondary",
        "validated_final_verdict",
        "validated_corrected_action",
        "validated_final_action",
        "validated_reasoning",
        "validated_final_action_is_from_correction",
        "normalized_narration_for_analysis",
    ]

    merged_rows: List[Dict[str, str]] = []
    for row in base_rows:
        out = dict(row)
        key = canonical_key(row)
        candidates = matched_by_key.get(key, [])
        chosen = selected_by_key.get(key)
        out["validation_matched"] = "1" if chosen else "0"
        out["validation_candidate_count"] = str(len(candidates))
        out["validation_conflict_flag"] = "1" if len(candidates) > 1 else "0"

        if not chosen:
            for c in appended_cols[3:]:
                out[c] = ""
            merged_rows.append(out)
            continue

        verdict = derive_verdict(chosen.get("status_main", ""), chosen.get("status_secondary", ""))
        corrected_action = clean(chosen.get("corrected_action", ""))
        final_action = clean(chosen.get("action", ""))
        from_correction = "0"
        if verdict == "Bad" and corrected_action:
            final_action = corrected_action
            from_correction = "1"

        out["validated_source_file"] = clean(chosen.get("_source_file", ""))
        out["validated_id"] = clean(chosen.get("id", ""))
        out["validated_annotation_id"] = clean(chosen.get("annotation_id", ""))
        out["validated_annotator"] = clean(chosen.get("annotator", ""))
        out["validated_created_at"] = clean(chosen.get("created_at", ""))
        out["validated_updated_at"] = clean(chosen.get("updated_at", ""))
        out["validated_lead_time"] = clean(chosen.get("lead_time", ""))
        out["validated_status_raw"] = clean(chosen.get("status", ""))
        out["validated_status_main"] = clean(chosen.get("status_main", ""))
        out["validated_status_secondary"] = clean(chosen.get("status_secondary", ""))
        out["validated_final_verdict"] = verdict
        out["validated_corrected_action"] = corrected_action
        out["validated_final_action"] = final_action
        out["validated_reasoning"] = clean(chosen.get("reasoning", ""))
        out["validated_final_action_is_from_correction"] = from_correction
        out["normalized_narration_for_analysis"] = normalize_narration_for_analysis(
            out.get("narration_text", "")
        )

        merged_rows.append(out)

    # Subset exports
    all_matched_rows = [r for r in merged_rows if clean(r.get("validation_matched")) == "1"]
    decision_rows = [
        r for r in all_matched_rows if clean(r.get("validated_final_verdict")) in VALID_VERDICTS
    ]
    gold_bad_rows = [
        r for r in decision_rows if clean(r.get("validated_final_verdict")) in {"Gold", "Bad"}
    ]

    # Deep-analysis subset: unique normalized narration + final action, Gold/Bad only.
    deep_best: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in gold_bad_rows:
        pair = (
            clean(row.get("normalized_narration_for_analysis")),
            clean(row.get("validated_final_action")),
        )
        if not pair[0] or not pair[1]:
            continue
        prev = deep_best.get(pair)
        if prev is None:
            deep_best[pair] = row
            continue
        prev_score = CandidateScore(
            updated_at=parse_iso_dt(prev.get("validated_updated_at", "")),
            created_at=parse_iso_dt(prev.get("validated_created_at", "")),
            source_pri=source_priority(prev.get("validated_source_file", "")),
            source_rank=0,
            row_seq=0,
        )
        cur_score = CandidateScore(
            updated_at=parse_iso_dt(row.get("validated_updated_at", "")),
            created_at=parse_iso_dt(row.get("validated_created_at", "")),
            source_pri=source_priority(row.get("validated_source_file", "")),
            source_rank=0,
            row_seq=0,
        )
        if cur_score > prev_score:
            deep_best[pair] = row
    deep_rows = list(deep_best.values())

    # Output paths (safe/non-overwriting)
    merged_path = safe_output_path(output_dir / "round1_assigned_only_30000_with_validations.csv")
    matched_path = safe_output_path(output_dir / "round1_validated_rows_all_matched.csv")
    decision_path = safe_output_path(output_dir / "round1_validated_rows_with_decisions.csv")
    deep_path = safe_output_path(
        output_dir / "round1_deep_analysis_subset_unique_narration_final_action.csv"
    )
    conflict_path = safe_output_path(output_dir / "round1_validation_conflict_log.csv")
    unmatched_path = safe_output_path(output_dir / "round1_validation_unmatched_rows.csv")
    summary_path = safe_output_path(output_dir / "round1_merge_summary.json")

    merged_fieldnames = base_fields + [c for c in appended_cols if c not in base_fields]
    write_csv(merged_path, merged_fieldnames, merged_rows)
    write_csv(matched_path, merged_fieldnames, all_matched_rows)
    write_csv(decision_path, merged_fieldnames, decision_rows)
    write_csv(deep_path, merged_fieldnames, deep_rows)

    conflict_fieldnames = sorted(
        {k for row in conflict_log_rows for k in row.keys()} | {"is_selected", "conflict_group_size"}
    )
    write_csv(conflict_path, conflict_fieldnames, conflict_log_rows)

    unmatched_fieldnames = sorted({k for row in unmatched_rows for k in row.keys()} | {"unmatched_reason"})
    if unmatched_fieldnames:
        write_csv(unmatched_path, unmatched_fieldnames, unmatched_rows)
    else:
        write_csv(unmatched_path, ["unmatched_reason"], [])

    summary = {
        "base_csv": str(base_csv),
        "validated_csvs": [str(p) for p in validated_csvs],
        "outputs": {
            "merged_30000_csv": str(merged_path),
            "validated_all_matched_csv": str(matched_path),
            "validated_decision_csv": str(decision_path),
            "deep_analysis_subset_csv": str(deep_path),
            "conflict_log_csv": str(conflict_path),
            "unmatched_csv": str(unmatched_path),
        },
        "counts": {
            "base_rows": len(base_rows),
            "validated_raw_rows": validated_raw_count,
            "validated_after_id_dedup": len(validated_id_deduped),
            "matched_rows_in_base": len(all_matched_rows),
            "decision_rows": len(decision_rows),
            "gold_bad_rows": len(gold_bad_rows),
            "deep_analysis_rows_unique_norm_narration_final_action": len(deep_rows),
            "conflict_keys_in_base": conflict_key_count,
            "duplicate_id_groups_in_validated_sources": duplicate_id_groups,
            "unmatched_validated_rows": len(unmatched_rows),
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Merge complete.")
    print(f"- Base rows kept: {len(base_rows):,} (must remain 30,000)")
    print(f"- Matched validation rows: {len(all_matched_rows):,}")
    print(f"- Decision rows (Gold/Bad/Skip/Delete): {len(decision_rows):,}")
    print(f"- Deep analysis subset rows (~5k target): {len(deep_rows):,}")
    print(f"- Conflict keys resolved: {conflict_key_count:,}")
    print(f"- Outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
