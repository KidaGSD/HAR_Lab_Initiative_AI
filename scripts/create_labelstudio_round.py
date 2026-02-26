#!/usr/bin/env python3
"""
Create a unified Label Studio annotation round package.

This script does end-to-end preparation:
1) Select rows for annotation from a canonical labels CSV.
2) Produce selected video UID list.
3) Export short video clips around each timestamp.
4) Generate Label Studio import JSON tasks and manifests.

Output structure:
  <output_root>/<round_id>/
    tasks/
      selected_rows.csv
      selected_videos.csv
      labelstudio_tasks.json
      video_uids.txt
    media/
      clips/*.mp4
    manifests/
      round_manifest.json
      missing_videos.csv

Example:
  python scripts/create_labelstudio_round.py \
      --labels-csv data/labels/action_labels_llm_clean_refined.csv \
      --video-root ../data/ego4d_data/videos/v2/full_scale \
      --round-id r001_human_audit \
      --target-rows 2500 \
      --hard-ratio 0.6 \
      --allowed-splits train val
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


REASONING_HARD_PATTERNS = re.compile(
    r"fallback|ctxfixed|parse error|invalid|ambiguous|uncertain|cannot determine|unsure",
    flags=re.IGNORECASE,
)

ACTION_HARD_SET = {
    "unknown",
    "error",
    "error / correction",
    "invalid",
    "uncertain",
}


@dataclass
class ClipResult:
    task_uid: str
    found_video: bool
    clip_exported: bool
    video_path: str
    clip_path: str
    start_sec: float
    end_sec: float
    error: str



def make_task_uid(video_uid: str, timestamp_sec: float) -> str:
    ts_ms = int(round(float(timestamp_sec) * 1000.0))
    return f"{video_uid}__{ts_ms:010d}"



def parse_status_list(raw: str) -> List[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]



def ensure_required_columns(df: pd.DataFrame, required: Iterable[str], path: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")



def add_selection_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_action_lower"] = out["action"].astype(str).str.strip().str.lower()
    out["_reasoning_lower"] = out.get("reasoning", "").fillna("").astype(str).str.lower()
    out["is_hard"] = out["_action_lower"].isin(ACTION_HARD_SET) | out["_reasoning_lower"].str.contains(
        REASONING_HARD_PATTERNS
    )
    out["task_uid"] = [make_task_uid(v, t) for v, t in zip(out["video_uid"], out["timestamp_sec"])]
    return out



def stratified_sample_by_scenario(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    if n >= len(df):
        return df.copy()

    if "scenario" not in df.columns:
        return df.sample(n=n, random_state=seed).copy()

    counts = df["scenario"].value_counts()
    target = counts / counts.sum() * n

    quotas = target.apply(math.floor).astype(int)
    remainders = (target - quotas).sort_values(ascending=False)

    remaining = n - int(quotas.sum())
    if remaining > 0:
        for scenario in remainders.index:
            if remaining <= 0:
                break
            if quotas.loc[scenario] < counts.loc[scenario]:
                quotas.loc[scenario] += 1
                remaining -= 1

    sampled_parts: List[pd.DataFrame] = []
    for i, (scenario, q) in enumerate(quotas.items()):
        if q <= 0:
            continue
        g = df[df["scenario"] == scenario]
        sampled_parts.append(g.sample(n=min(q, len(g)), random_state=seed + i + 1))

    sampled = pd.concat(sampled_parts, ignore_index=False) if sampled_parts else df.iloc[0:0].copy()

    if len(sampled) < n:
        remain_df = df.drop(index=sampled.index, errors="ignore")
        if not remain_df.empty:
            extra_n = min(n - len(sampled), len(remain_df))
            extra = remain_df.sample(n=extra_n, random_state=seed + 999)
            sampled = pd.concat([sampled, extra], ignore_index=False)

    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed + 2024)

    return sampled.copy()



def enforce_max_rows_per_video(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or df.empty:
        return df.copy()

    shuffled = df.sample(frac=1.0, random_state=seed)
    return shuffled.groupby("video_uid", group_keys=False).head(max_rows).copy()



def join_split_info(
    selected: pd.DataFrame,
    scenario_labels_csv: Optional[Path],
    allowed_splits: List[str],
) -> pd.DataFrame:
    if scenario_labels_csv is None:
        return selected
    if not scenario_labels_csv.exists():
        print(f"Warning: scenario labels file not found: {scenario_labels_csv}. Skip split filtering.")
        return selected

    scen = pd.read_csv(scenario_labels_csv)
    if "video_uid" not in scen.columns:
        print(f"Warning: scenario labels has no video_uid column: {scenario_labels_csv}. Skip split filtering.")
        return selected

    if "split" not in scen.columns:
        print(f"Warning: scenario labels has no split column: {scenario_labels_csv}. Skip split filtering.")
        return selected

    keep_cols = ["video_uid", "split"]
    merged = selected.merge(scen[keep_cols].drop_duplicates("video_uid"), on="video_uid", how="left")

    if allowed_splits:
        allowed = {x.lower() for x in allowed_splits}
        before = len(merged)
        merged = merged[merged["split"].astype(str).str.lower().isin(allowed)].copy()
        print(f"Split filtering: kept {len(merged)}/{before} rows for splits={sorted(allowed)}")

    return merged



def build_video_index(video_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in video_root.rglob("*.mp4"):
        stem = p.stem
        if stem not in index:
            index[stem] = p
    return index



def get_video_duration_sec(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)



def export_clip(
    src: Path,
    dst: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        str(dst),
    ]
    subprocess.run(cmd, check=True)



def prepare_round(args: argparse.Namespace) -> int:
    labels_path = Path(args.labels_csv)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")

    round_dir = Path(args.output_root) / args.round_id
    tasks_dir = round_dir / "tasks"
    media_dir = round_dir / "media"
    clips_dir = media_dir / "clips"
    manifests_dir = round_dir / "manifests"

    tasks_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading labels: {labels_path}")
    labels = pd.read_csv(labels_path)
    ensure_required_columns(
        labels,
        ["video_uid", "timestamp_sec", "narration_text", "scenario", "action"],
        str(labels_path),
    )

    if "status" not in labels.columns:
        labels["status"] = "silver"

    labels = add_selection_signals(labels)

    status_allow = set(parse_status_list(args.statuses))
    eligible = labels[labels["status"].astype(str).str.lower().isin(status_allow)].copy()
    print(f"Eligible rows by status {sorted(status_allow)}: {len(eligible):,}")

    eligible = join_split_info(
        eligible,
        Path(args.scenario_labels_csv) if args.scenario_labels_csv else None,
        args.allowed_splits or [],
    )

    if args.scenarios:
        scenario_allow = set(args.scenarios)
        before = len(eligible)
        eligible = eligible[eligible["scenario"].isin(scenario_allow)].copy()
        print(f"Scenario filtering: kept {len(eligible):,}/{before:,} rows in {sorted(scenario_allow)}")

    if eligible.empty:
        print("No eligible rows after filtering.")
        return 1

    target_rows = min(int(args.target_rows), len(eligible))
    hard_target = min(int(round(target_rows * float(args.hard_ratio))), int(eligible["is_hard"].sum()))
    random_target = max(target_rows - hard_target, 0)

    hard_pool = eligible[eligible["is_hard"]].copy()
    easy_pool = eligible[~eligible["is_hard"]].copy()

    sample_hard = stratified_sample_by_scenario(hard_pool, hard_target, args.seed)
    sample_easy = stratified_sample_by_scenario(easy_pool, random_target, args.seed + 123)

    sample_hard["sample_bucket"] = "hard"
    sample_easy["sample_bucket"] = "random"

    selected = pd.concat([sample_hard, sample_easy], ignore_index=False)
    selected = selected.drop_duplicates(subset=["task_uid"]).copy()

    if args.max_rows_per_video > 0:
        before = len(selected)
        selected = enforce_max_rows_per_video(selected, args.max_rows_per_video, args.seed + 66)
        after = len(selected)
        if after < before:
            print(f"Applied --max-rows-per-video={args.max_rows_per_video}: {before} -> {after}")

    if len(selected) > target_rows:
        selected = selected.sample(n=target_rows, random_state=args.seed + 77)

    # Fill back if under target due per-video limits
    if len(selected) < target_rows:
        needed = target_rows - len(selected)
        rest = eligible[~eligible["task_uid"].isin(selected["task_uid"])].copy()
        rest = enforce_max_rows_per_video(rest, args.max_rows_per_video, args.seed + 88)
        if not rest.empty:
            extra = rest.sample(n=min(needed, len(rest)), random_state=args.seed + 89)
            extra["sample_bucket"] = "backfill"
            selected = pd.concat([selected, extra], ignore_index=False)

    selected = selected.reset_index(drop=True)

    selected_videos = (
        selected.groupby(["video_uid", "scenario"], as_index=False)
        .size()
        .rename(columns={"size": "selected_rows"})
        .sort_values(["scenario", "selected_rows"], ascending=[True, False])
    )

    # Optional clip export
    video_index: Dict[str, Path] = {}
    duration_cache: Dict[str, float] = {}
    clip_results: List[ClipResult] = []

    if not args.skip_clips:
        if not args.video_root:
            raise ValueError("--video-root is required unless --skip-clips is set")
        video_root = Path(args.video_root)
        if not video_root.exists():
            raise FileNotFoundError(f"video root does not exist: {video_root}")
        print(f"Building video index from: {video_root}")
        video_index = build_video_index(video_root)
        print(f"Indexed videos: {len(video_index):,}")

    tasks_payload: List[dict] = []
    output_rows: List[dict] = []

    for row in selected.to_dict(orient="records"):
        video_uid = str(row["video_uid"])
        ts = float(row["timestamp_sec"])
        task_uid = str(row["task_uid"])

        clip_relpath = f"clips/{task_uid}.mp4"
        clip_abs = clips_dir / f"{task_uid}.mp4"
        video_field_value = ""

        found_video = False
        clip_ok = False
        src_path = ""
        err = ""
        start_sec = max(0.0, ts - float(args.clip_pre_sec))
        end_sec = ts + float(args.clip_post_sec)

        if args.skip_clips:
            # Keep placeholder path. Caller can host media separately.
            video_field_value = args.video_field_template.format(clip_relpath=clip_relpath)
            clip_ok = True
        else:
            src = video_index.get(video_uid)
            if src is None:
                err = "video_not_found"
            else:
                found_video = True
                src_path = str(src)
                try:
                    if src_path not in duration_cache:
                        duration_cache[src_path] = get_video_duration_sec(src)
                    duration = duration_cache[src_path]
                    end_sec = min(end_sec, max(duration, start_sec + 0.2))
                    export_clip(src, clip_abs, start_sec, end_sec)
                    clip_ok = True
                    video_field_value = args.video_field_template.format(clip_relpath=clip_relpath)
                except Exception as e:  # pragma: no cover
                    err = str(e)

        clip_results.append(
            ClipResult(
                task_uid=task_uid,
                found_video=found_video,
                clip_exported=clip_ok,
                video_path=src_path,
                clip_path=str(clip_abs),
                start_sec=round(start_sec, 3),
                end_sec=round(end_sec, 3),
                error=err,
            )
        )

        if not clip_ok and not args.include_missing_video_tasks:
            continue

        task_data = {
            "task_uid": task_uid,
            "video": video_field_value,
            "video_uid": video_uid,
            "timestamp_sec": ts,
            "scenario": str(row.get("scenario", "")),
            "narration_text": str(row.get("narration_text", "")),
            "llm_action": str(row.get("action", "")),
            "llm_reasoning": str(row.get("reasoning", "")),
            "llm_status": str(row.get("status", "")),
            "sample_bucket": str(row.get("sample_bucket", "")),
            "is_hard": bool(row.get("is_hard", False)),
        }

        tasks_payload.append({"data": task_data})

        out_row = dict(row)
        out_row["clip_relpath"] = clip_relpath
        out_row["video"] = video_field_value
        out_row["clip_exported"] = clip_ok
        output_rows.append(out_row)

    out_rows_df = pd.DataFrame(output_rows)

    selected_rows_path = tasks_dir / "selected_rows.csv"
    selected_videos_path = tasks_dir / "selected_videos.csv"
    tasks_json_path = tasks_dir / "labelstudio_tasks.json"
    video_uids_path = tasks_dir / "video_uids.txt"
    missing_csv_path = manifests_dir / "missing_videos.csv"
    round_manifest_path = manifests_dir / "round_manifest.json"

    out_rows_df.to_csv(selected_rows_path, index=False)
    selected_videos.to_csv(selected_videos_path, index=False)
    with tasks_json_path.open("w", encoding="utf-8") as f:
        json.dump(tasks_payload, f, ensure_ascii=False, indent=2)
    with video_uids_path.open("w", encoding="utf-8") as f:
        for uid in sorted(selected_videos["video_uid"].astype(str).unique()):
            f.write(uid + "\n")

    clip_df = pd.DataFrame([c.__dict__ for c in clip_results])
    missing_df = clip_df[~clip_df["clip_exported"]].copy()
    missing_df.to_csv(missing_csv_path, index=False)

    clips_exported = int(clip_df["clip_exported"].sum()) if not clip_df.empty else 0
    clips_missing_or_failed = int((~clip_df["clip_exported"]).sum()) if not clip_df.empty else 0
    if args.skip_clips:
        clips_exported = 0
        clips_missing_or_failed = 0

    manifest = {
        "round_id": args.round_id,
        "labels_csv": str(labels_path),
        "video_root": str(args.video_root) if args.video_root else "",
        "target_rows": target_rows,
        "statuses": sorted(status_allow),
        "allowed_splits": args.allowed_splits or [],
        "hard_ratio": float(args.hard_ratio),
        "clip_window_sec": {"pre": float(args.clip_pre_sec), "post": float(args.clip_post_sec)},
        "counts": {
            "eligible_rows": int(len(eligible)),
            "selected_rows_requested": int(target_rows),
            "selected_rows_written": int(len(out_rows_df)),
            "selected_videos": int(out_rows_df["video_uid"].nunique()) if not out_rows_df.empty else 0,
            "tasks_json_entries": int(len(tasks_payload)),
            "clips_exported": clips_exported,
            "clips_missing_or_failed": clips_missing_or_failed,
        },
        "output_paths": {
            "selected_rows_csv": str(selected_rows_path),
            "selected_videos_csv": str(selected_videos_path),
            "tasks_json": str(tasks_json_path),
            "video_uids_txt": str(video_uids_path),
            "missing_videos_csv": str(missing_csv_path),
        },
    }
    with round_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nRound package created:")
    print(f"  Round dir: {round_dir}")
    print(f"  Selected rows: {manifest['counts']['selected_rows_written']:,}")
    print(f"  Selected videos: {manifest['counts']['selected_videos']:,}")
    print(f"  Exported clips: {manifest['counts']['clips_exported']:,}")
    print(f"  Missing/failed clips: {manifest['counts']['clips_missing_or_failed']:,}")
    print(f"  Label Studio tasks JSON: {tasks_json_path}")

    return 0



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a unified Label Studio annotation round package")
    parser.add_argument("--labels-csv", default="data/labels/action_labels_llm_clean_refined.csv")
    parser.add_argument("--scenario-labels-csv", default="data/labels/scenario_labels.csv")
    parser.add_argument("--output-root", default="data/annotation_rounds")
    parser.add_argument("--round-id", required=True)

    parser.add_argument("--target-rows", type=int, default=2000)
    parser.add_argument("--statuses", default="silver", help="Comma-separated status filter, e.g. silver,bad")
    parser.add_argument("--hard-ratio", type=float, default=0.6)
    parser.add_argument("--max-rows-per-video", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--allowed-splits", nargs="*", default=[], help="Filter by split values in scenario_labels.csv")
    parser.add_argument("--scenarios", nargs="*", default=[], help="Filter by scenario names")

    parser.add_argument("--skip-clips", action="store_true", help="Do not export clips; only make task package")
    parser.add_argument("--video-root", default="", help="Root folder containing <video_uid>.mp4 files")
    parser.add_argument("--clip-pre-sec", type=float, default=2.0)
    parser.add_argument("--clip-post-sec", type=float, default=6.0)
    parser.add_argument(
        "--video-field-template",
        default="/data/local-files/?d={clip_relpath}",
        help="Template for Label Studio video field. Use {clip_relpath}.",
    )
    parser.add_argument(
        "--include-missing-video-tasks",
        action="store_true",
        help="Include tasks even if clip export fails (video field may be empty)",
    )

    return parser



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.hard_ratio < 0 or args.hard_ratio > 1:
        raise ValueError("--hard-ratio must be in [0, 1]")
    if args.target_rows <= 0:
        raise ValueError("--target-rows must be > 0")

    return prepare_round(args)


if __name__ == "__main__":
    raise SystemExit(main())
