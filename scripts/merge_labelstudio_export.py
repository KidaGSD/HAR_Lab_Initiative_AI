#!/usr/bin/env python3
"""
Merge Label Studio JSON export back into canonical local labels CSV.

Expected Label Studio export format: JSON list of tasks (Export -> JSON).
Each task should include `data.task_uid` (recommended), or `video_uid` + `timestamp_sec`.

This script adds unified columns for multi-source labels:
- llm_action
- vlm_verdict
- human_verdict
- human_action_6class
- human_action_4class
- human_note
- human_annotator
- human_updated_at
- status_before_human
- status_unified
- action_human_override
- label_source

Example:
  python scripts/merge_labelstudio_export.py \
      --base-csv data/labels/action_labels_llm_clean_refined.csv \
      --ls-export-json data/annotation_rounds/r001/tasks/export.json \
      --output-csv data/labels/action_labels_unified.csv \
      --apply-human-correction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd



def make_task_uid(video_uid: str, timestamp_sec: float) -> str:
    ts_ms = int(round(float(timestamp_sec) * 1000.0))
    return f"{video_uid}__{ts_ms:010d}"



def first_choice(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str):
        return value
    return ""



def parse_annotation_result(result: List[dict]) -> Dict[str, str]:
    out = {
        "human_verdict": "",
        "human_action_6class": "",
        "human_action_4class": "",
        "vlm_verdict": "",
        "human_note": "",
    }

    for item in result or []:
        from_name = str(item.get("from_name", "")).strip().lower()
        value = item.get("value", {}) or {}

        choices = first_choice(value.get("choices", []))
        text_value = value.get("text", [])
        text_first = ""
        if isinstance(text_value, list) and text_value:
            text_first = str(text_value[0])
        elif isinstance(text_value, str):
            text_first = text_value

        if from_name in {"human_verdict", "verdict", "quality_verdict",
                        "status_main", "status_secondary"}:
            # status_main has Gold/Bad, status_secondary has Skip/Delete Row
            # Only overwrite if we don't already have a Gold/Bad verdict
            if from_name in {"status_secondary"} and out["human_verdict"] in {"Gold", "Bad"}:
                pass  # Keep the Gold/Bad from status_main
            else:
                out["human_verdict"] = choices
        elif from_name in {"human_action_6class", "corrected_action_6class",
                           "action_fix_6class", "corrected_action"}:
            out["human_action_6class"] = choices
        elif from_name in {"human_action_4class", "corrected_action_4class", "action_fix_4class"}:
            out["human_action_4class"] = choices
        elif from_name in {"vlm_verdict", "vlm_check", "vlm_quality"}:
            out["vlm_verdict"] = choices
        elif from_name in {"human_note", "note", "comment"}:
            out["human_note"] = text_first

    # Normalize casing for action labels (interface may use lowercase variants)
    _ACTION_CASE_MAP = {
        "essential operation": "Essential Operation",
        "object transfer": "Object Transfer",
        "search": "Search",
        "stationary": "Stationary",
        "locomotion": "Locomotion",
        "error / correction": "Error / Correction",
    }
    if out["human_action_6class"]:
        out["human_action_6class"] = _ACTION_CASE_MAP.get(
            out["human_action_6class"].strip().lower(),
            out["human_action_6class"]
        )
    if out["human_action_4class"]:
        _4CLASS_MAP = {
            "stationary": "Stationary",
            "locomotion": "Locomotion",
            "manipulation": "Manipulation",
            "search_interrupt": "Search_Interrupt",
        }
        out["human_action_4class"] = _4CLASS_MAP.get(
            out["human_action_4class"].strip().lower(),
            out["human_action_4class"]
        )

    return out



def parse_annotator_name(completed_by: Any) -> str:
    if isinstance(completed_by, dict):
        for key in ("email", "username", "id"):
            if key in completed_by and completed_by[key] is not None:
                return str(completed_by[key])
        return ""
    if completed_by is None:
        return ""
    return str(completed_by)



def load_labelstudio_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "tasks" in obj and isinstance(obj["tasks"], list):
            return obj["tasks"]
        if "results" in obj and isinstance(obj["results"], list):
            return obj["results"]
    raise ValueError(f"Unsupported Label Studio export JSON structure: {path}")



def pick_latest_annotation(annotations: List[dict]) -> Optional[dict]:
    if not annotations:
        return None

    def sort_key(a: dict) -> str:
        return str(a.get("updated_at") or a.get("created_at") or "")

    anns = sorted(annotations, key=sort_key)
    return anns[-1]



def parse_tasks_to_flat_rows(tasks: List[dict]) -> pd.DataFrame:
    rows: List[dict] = []
    skipped_no_key = 0

    for task in tasks:
        data = task.get("data", {}) or {}
        annotations = task.get("annotations", []) or []
        ann = pick_latest_annotation(annotations)
        if ann is None:
            continue

        task_uid = str(data.get("task_uid", "")).strip()
        if not task_uid:
            video_uid = str(data.get("video_uid", "")).strip()
            ts = data.get("timestamp_sec")
            if video_uid and ts is not None and str(ts) != "":
                try:
                    task_uid = make_task_uid(video_uid, float(ts))
                except Exception:
                    task_uid = ""

        if not task_uid:
            skipped_no_key += 1
            continue

        parsed = parse_annotation_result(ann.get("result", []) or [])

        rows.append(
            {
                "task_uid": task_uid,
                **parsed,
                "human_annotator": parse_annotator_name(ann.get("completed_by")),
                "human_updated_at": str(ann.get("updated_at") or ann.get("created_at") or ""),
                "ls_annotation_id": str(ann.get("id", "")),
            }
        )

    if skipped_no_key > 0:
        print(f"Warning: skipped {skipped_no_key} annotated tasks without task key.")

    if not rows:
        return pd.DataFrame(columns=[
            "task_uid",
            "human_verdict",
            "human_action_6class",
            "human_action_4class",
            "vlm_verdict",
            "human_note",
            "human_annotator",
            "human_updated_at",
            "ls_annotation_id",
        ])

    flat = pd.DataFrame(rows)
    # keep latest record per task_uid
    flat = flat.sort_values("human_updated_at").drop_duplicates("task_uid", keep="last")
    return flat



def map_status_from_verdict(verdict: str, fallback: str) -> str:
    v = str(verdict or "").strip().lower()
    if v == "gold":
        return "gold"
    if v == "bad":
        return "bad"
    if v in {"delete row", "delete", "drop"}:
        return "deleted"
    if v in {"skip", "uncertain"}:
        return fallback
    return fallback



def map_label_source(verdict: str, action_override: str) -> str:
    v = str(verdict or "").strip().lower()
    o = str(action_override or "").strip()

    if not v:
        return "llm"
    if v == "gold":
        return "human_verified"
    if v == "bad" and o:
        return "human_corrected"
    if v == "bad" and not o:
        return "human_flagged_bad"
    if v in {"delete row", "delete", "drop"}:
        return "human_deleted"
    if v in {"skip", "uncertain"}:
        return "human_skipped"
    return "llm"



def run(args: argparse.Namespace) -> int:
    base_csv = Path(args.base_csv)
    ls_json = Path(args.ls_export_json)

    if not base_csv.exists():
        raise FileNotFoundError(f"base CSV not found: {base_csv}")
    if not ls_json.exists():
        raise FileNotFoundError(f"Label Studio export JSON not found: {ls_json}")

    print(f"Loading base CSV: {base_csv}")
    base = pd.read_csv(base_csv)

    for col in ["video_uid", "timestamp_sec", "action"]:
        if col not in base.columns:
            raise ValueError(f"Missing required column in base CSV: {col}")

    if "status" not in base.columns:
        base["status"] = "silver"

    if "task_uid" not in base.columns:
        base["task_uid"] = [make_task_uid(v, t) for v, t in zip(base["video_uid"], base["timestamp_sec"])]

    if "llm_action" not in base.columns:
        base["llm_action"] = base["action"]

    print(f"Loading Label Studio export: {ls_json}")
    tasks = load_labelstudio_json(ls_json)
    ann_flat = parse_tasks_to_flat_rows(tasks)
    print(f"Parsed annotated tasks: {len(ann_flat):,}")

    merged = base.merge(ann_flat, on="task_uid", how="left")

    merged["status_before_human"] = merged["status"]

    # Build override action (prefer 6-class, fallback 4-class)
    merged["action_human_override"] = merged["human_action_6class"].fillna("")
    use_4 = merged["action_human_override"].eq("") & merged["human_action_4class"].fillna("").ne("")
    merged.loc[use_4, "action_human_override"] = merged.loc[use_4, "human_action_4class"]

    merged["status_unified"] = [
        map_status_from_verdict(v, s)
        for v, s in zip(merged["human_verdict"].fillna(""), merged["status_before_human"].fillna("silver"))
    ]

    merged["label_source"] = [
        map_label_source(v, o)
        for v, o in zip(merged["human_verdict"].fillna(""), merged["action_human_override"].fillna(""))
    ]

    if args.apply_human_correction:
        has_override = merged["action_human_override"].fillna("").ne("")
        merged.loc[has_override, "action"] = merged.loc[has_override, "action_human_override"]

    # If user wants final status written to status column
    if args.write_back_status:
        merged["status"] = merged["status_unified"]

    if args.round_id:
        merged["human_round_id"] = args.round_id

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    flat_out = out_csv.with_name(out_csv.stem + "_human_flat.csv")
    ann_flat.to_csv(flat_out, index=False)

    matched = int(merged["human_verdict"].fillna("").ne("").sum())
    print("\nMerge completed:")
    print(f"  Base rows: {len(base):,}")
    print(f"  Annotated matches: {matched:,}")
    print(f"  Output CSV: {out_csv}")
    print(f"  Flat human annotations: {flat_out}")

    return 0



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge Label Studio JSON export into canonical labels CSV")
    p.add_argument("--base-csv", default="data/labels/action_labels_llm_clean_refined.csv")
    p.add_argument("--ls-export-json", required=True)
    p.add_argument("--output-csv", default="data/labels/action_labels_unified.csv")
    p.add_argument("--round-id", default="")
    p.add_argument("--apply-human-correction", action="store_true")
    p.add_argument("--write-back-status", action="store_true")
    return p



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
