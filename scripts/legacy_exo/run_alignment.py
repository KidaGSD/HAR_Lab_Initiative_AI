#!/usr/bin/env python3
"""
Alignment & QA pipeline for Ego-Exo4D takes.

Usage examples:
  python scripts/run_alignment.py --uids-file ready_for_alignment.txt
  python scripts/run_alignment.py --take-uid d8013909-... --take-uid 6f942bbd-...
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_loader import get_from_config, load_config

MPL_DIR = Path(os.environ.get("MPLCONFIGDIR", Path(".cache") / "matplotlib")).expanduser()
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class AlignmentResult:
    take_uid: str
    take_name: str
    aligned: bool
    duration_gaze_s: Optional[float]
    duration_traj_s: Optional[float]
    sample_rates_hz: Dict[str, Optional[float]]
    missing_ratio: Dict[str, Optional[float]]
    note: str
    qa_json_path: Path
    plots: List[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--take-uid",
        action="append",
        dest="take_uids",
        help="Single take_uid to align (can repeat).",
    )
    parser.add_argument(
        "--uids-file",
        type=Path,
        help="Text/CSV file with take_uid column (space/tab/comma separated).",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Root directory containing takes.json and takes/ folders (default: config.cache.raw_root).",
    )
    parser.add_argument(
        "--qa-root",
        type=Path,
        default=None,
        help="Directory to store QA artifacts (default: config.alignment.qa_root or run-root/qa).",
    )
    parser.add_argument(
        "--ready-file",
        type=Path,
        default=None,
        help="File to append aligned take_uids + summary (default: config.alignment.ready_file).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate QA artifacts even if outputs already exist.",
    )
    parser.add_argument(
        "--max-takes",
        type=int,
        default=None,
        help="Limit number of takes processed (useful for smoke tests).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Config file with default paths.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Optional runs/<run_id> directory; QA defaults to <run_root>/qa.",
    )
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cache_default = get_from_config(cfg, ["cache", "raw_root"], "egoexo_cache")
    qa_default = get_from_config(cfg, ["alignment", "qa_root"], "processed_windows/qa")
    ready_default = get_from_config(cfg, ["alignment", "ready_file"], "state/active/ready_for_windowing.txt")

    if args.cache_root is None:
        args.cache_root = Path(cache_default)
    if args.run_root:
        run_root = Path(args.run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        if args.qa_root is None:
            args.qa_root = run_root / "qa"
        if args.ready_file is None:
            state_dir = run_root / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            args.ready_file = state_dir / "ready_for_windowing.txt"
    if args.qa_root is None:
        args.qa_root = Path(qa_default)
    if args.ready_file is None:
        args.ready_file = Path(ready_default)
    if not args.qa_root.exists():
        args.qa_root.mkdir(parents=True, exist_ok=True)
    ready_parent = args.ready_file.parent
    if not ready_parent.exists():
        ready_parent.mkdir(parents=True, exist_ok=True)


def load_take_map(cache_root: Path) -> Dict[str, str]:
    takes_json = cache_root / "takes.json"
    if not takes_json.exists():
        raise FileNotFoundError(f"{takes_json} not found. Run egoexo metadata download first.")
    data = json.loads(takes_json.read_text())
    if isinstance(data, dict):
        takes = data.get("takes", data.get("data", []))
    else:
        takes = data
    return {entry["take_uid"]: entry["take_name"] for entry in takes if entry.get("take_uid") and entry.get("take_name")}


def gather_take_uids(args: argparse.Namespace) -> List[str]:
    uids: List[str] = []
    if args.take_uids:
        uids.extend(args.take_uids)
    if args.uids_file and args.uids_file.exists():
        text = args.uids_file.read_text().strip().splitlines()
        for line in text:
            if not line.strip():
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            uids.append(parts[0])
    if not uids:
        raise ValueError("No take_uid provided. Use --take-uid or --uids-file.")
    # preserve order but drop duplicates
    seen = set()
    ordered = []
    for uid in uids:
        if uid not in seen:
            ordered.append(uid)
            seen.add(uid)
    if args.max_takes:
        ordered = ordered[: args.max_takes]
    return ordered


def estimate_rate_hz(timestamps_us: np.ndarray) -> Optional[float]:
    if len(timestamps_us) < 2:
        return None
    diffs = np.diff(np.sort(timestamps_us))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    median = np.median(diffs)
    if median <= 0:
        return None
    return 1_000_000.0 / median


def gap_ratio(timestamps_us: np.ndarray, multiplier: float = 3.0) -> Optional[float]:
    if len(timestamps_us) < 2:
        return None
    diffs = np.diff(np.sort(timestamps_us))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    median = np.median(diffs)
    gaps = (diffs > (multiplier * median)).sum()
    return float(gaps) / len(diffs)


def normalize_time(df: pd.DataFrame, column: str, t0: int) -> pd.Series:
    return (df[column].to_numpy(dtype=np.int64) - t0) / 1_000_000.0


def load_gaze(path: Path) -> pd.DataFrame:
    csv_path = path / "eye_gaze" / "personalized_eye_gaze.csv"
    if not csv_path.exists():
        csv_path = path / "eye_gaze" / "general_eye_gaze.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Gaze CSV not found under {path}")
    df = pd.read_csv(csv_path)
    if "tracking_timestamp_us" not in df.columns:
        raise ValueError(f"Gaze CSV missing tracking_timestamp_us: {csv_path}")
    # derive average yaw/pitch
    yaw_cols = [c for c in df.columns if "yaw" in c and c.endswith("cpf")]
    pitch_cols = [c for c in df.columns if "pitch" in c and c.endswith("cpf")]
    if yaw_cols:
        df["avg_yaw_rads"] = df[yaw_cols].mean(axis=1)
    if pitch_cols:
        df["avg_pitch_rads"] = df[pitch_cols].mean(axis=1)
    return df


def load_trajectory(path: Path) -> pd.DataFrame:
    csv_path = path / "trajectory" / "closed_loop_trajectory.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Trajectory CSV not found under {path}")
    df = pd.read_csv(csv_path)
    if "tracking_timestamp_us" not in df.columns:
        raise ValueError(f"Trajectory CSV missing tracking_timestamp_us: {csv_path}")
    lin_cols = [
        "device_linear_velocity_x_device",
        "device_linear_velocity_y_device",
        "device_linear_velocity_z_device",
    ]
    ang_cols = [
        "angular_velocity_x_device",
        "angular_velocity_y_device",
        "angular_velocity_z_device",
    ]
    if all(col in df.columns for col in lin_cols):
        df["linear_speed"] = np.linalg.norm(df[lin_cols].to_numpy(dtype=float), axis=1)
    if all(col in df.columns for col in ang_cols):
        df["angular_speed"] = np.linalg.norm(df[ang_cols].to_numpy(dtype=float), axis=1)
    return df


def plot_quicklook(take_uid: str, gaze_df: pd.DataFrame, traj_df: pd.DataFrame, qa_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    if "linear_speed" in traj_df:
        ax1.plot(traj_df["time_s"], traj_df["linear_speed"], label="linear_speed (m/s)")
    if "angular_speed" in traj_df:
        ax1.plot(traj_df["time_s"], traj_df["angular_speed"], label="angular_speed (rad/s)", alpha=0.7)
    ax1.set_ylabel("Head motion")
    ax1.legend(loc="upper right")
    if "avg_yaw_rads" in gaze_df:
        ax2.plot(gaze_df["time_s"], np.degrees(gaze_df["avg_yaw_rads"]), label="yaw (deg)")
    if "avg_pitch_rads" in gaze_df:
        ax2.plot(gaze_df["time_s"], np.degrees(gaze_df["avg_pitch_rads"]), label="pitch (deg)")
    ax2.set_ylabel("Gaze (deg)")
    ax2.set_xlabel("Time (s, normalized)")
    ax2.legend(loc="upper right")
    fig.suptitle(f"Take {take_uid}: head IMU vs gaze")
    qa_dir.mkdir(parents=True, exist_ok=True)
    plot_path = qa_dir / f"qa_pose_gaze_{take_uid}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def align_take(
    take_uid: str,
    take_name: str,
    cache_root: Path,
    qa_root: Path,
    overwrite: bool = False,
) -> AlignmentResult:
    take_dir = cache_root / "takes" / take_name
    if not take_dir.exists():
        raise FileNotFoundError(f"{take_dir} not found. Did you download the take?")

    qa_dir = qa_root
    qa_json = qa_dir / f"{take_uid}.json"
    existing = qa_json.exists()
    if existing and not overwrite:
        data = json.loads(qa_json.read_text())
        return AlignmentResult(
            take_uid=take_uid,
            take_name=take_name,
            aligned=data.get("aligned", False),
            duration_gaze_s=data.get("duration", {}).get("gaze_s"),
            duration_traj_s=data.get("duration", {}).get("trajectory_s"),
            sample_rates_hz=data.get("sample_rate_hz", {}),
            missing_ratio=data.get("missing_ratio", {}),
            note=data.get("notes", ""),
            qa_json_path=qa_json,
            plots=[qa_dir / f"qa_pose_gaze_{take_uid}.png"],
        )

    gaze_df = load_gaze(take_dir)
    traj_df = load_trajectory(take_dir)

    t0 = int(min(gaze_df["tracking_timestamp_us"].min(), traj_df["tracking_timestamp_us"].min()))
    gaze_df = gaze_df.copy()
    traj_df = traj_df.copy()
    gaze_df["time_s"] = normalize_time(gaze_df, "tracking_timestamp_us", t0)
    traj_df["time_s"] = normalize_time(traj_df, "tracking_timestamp_us", t0)

    gaze_duration = float(gaze_df["time_s"].max() - gaze_df["time_s"].min()) if len(gaze_df) else None
    traj_duration = float(traj_df["time_s"].max() - traj_df["time_s"].min()) if len(traj_df) else None
    gaze_rate = estimate_rate_hz(gaze_df["tracking_timestamp_us"].to_numpy())
    traj_rate = estimate_rate_hz(traj_df["tracking_timestamp_us"].to_numpy())
    gaze_gap = gap_ratio(gaze_df["tracking_timestamp_us"].to_numpy())
    traj_gap = gap_ratio(traj_df["tracking_timestamp_us"].to_numpy())

    aligned = (
        gaze_duration is not None
        and traj_duration is not None
        and abs(gaze_duration - traj_duration) < 0.5
        and gaze_gap is not None
        and traj_gap is not None
    )

    plot_path = plot_quicklook(take_uid, gaze_df, traj_df, qa_dir)

    qa_dir.mkdir(parents=True, exist_ok=True)
    qa_payload = {
        "take_uid": take_uid,
        "take_name": take_name,
        "aligned": bool(aligned),
        "duration": {
            "gaze_s": gaze_duration,
            "trajectory_s": traj_duration,
        },
        "sample_rate_hz": {
            "gaze": gaze_rate,
            "trajectory": traj_rate,
        },
        "missing_ratio": {
            "gaze": gaze_gap,
            "trajectory": traj_gap,
        },
        "notes": "",
    }
    qa_json.write_text(json.dumps(qa_payload, indent=2))

    note = ""
    if not aligned:
        note = "Duration mismatch or gaps; inspect CSVs."

    return AlignmentResult(
        take_uid=take_uid,
        take_name=take_name,
        aligned=aligned,
        duration_gaze_s=gaze_duration,
        duration_traj_s=traj_duration,
        sample_rates_hz={"gaze": gaze_rate, "trajectory": traj_rate},
        missing_ratio={"gaze": gaze_gap, "trajectory": traj_gap},
        note=note,
        qa_json_path=qa_json,
        plots=[plot_path],
    )


def append_ready(ready_file: Path, result: AlignmentResult) -> None:
    samples = {
        "gaze": float(result.sample_rates_hz.get("gaze") or 0.0),
        "trajectory": float(result.sample_rates_hz.get("trajectory") or 0.0),
    }
    line = f"{result.take_uid}\t{result.take_name}\tsamples={samples}\n"
    with ready_file.open("a") as fp:
        fp.write(line)


def main() -> None:
    args = parse_args()
    apply_config(args)
    take_map = load_take_map(args.cache_root)
    take_uids = gather_take_uids(args)

    summary: List[AlignmentResult] = []
    for idx, take_uid in enumerate(take_uids, 1):
        take_name = take_map.get(take_uid)
        if not take_name:
            print(f"[warn] take_uid {take_uid} not found in takes.json; skipping.", file=sys.stderr)
            continue
        try:
            result = align_take(
                take_uid=take_uid,
                take_name=take_name,
                cache_root=args.cache_root,
                qa_root=args.qa_root,
                overwrite=args.overwrite,
            )
            summary.append(result)
            status = "OK" if result.aligned else "WARN"
            print(f"[{idx}/{len(take_uids)}] {take_uid} ({take_name}) -> {status}")
            if result.aligned and args.ready_file:
                append_ready(args.ready_file, result)
        except Exception as exc:
            print(f"[error] {take_uid}: {exc}", file=sys.stderr)

    print(f"Processed {len(summary)} takes. QA artifacts: {args.qa_root}")


if __name__ == "__main__":
    main()
