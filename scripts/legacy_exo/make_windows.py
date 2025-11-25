#!/usr/bin/env python3
"""
Window extraction: generate statistical features + fixed-length sequences per take.

Usage example:
    python scripts/make_windows.py \
        --uids-file ready_for_windowing.txt \
        --window 3.0 \
        --hop 0.2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_loader import get_from_config, load_config

# Columns available in CSVs
GAZE_YAW_COLS = [
    "left_yaw_rads_cpf",
    "right_yaw_rads_cpf",
    "left_yaw_low_rads_cpf",
    "right_yaw_low_rads_cpf",
    "left_yaw_high_rads_cpf",
    "right_yaw_high_rads_cpf",
]
GAZE_PITCH_COLS = [
    "pitch_rads_cpf",
    "pitch_low_rads_cpf",
    "pitch_high_rads_cpf",
]
FEATURE_VERSION = "window_v3_realhz"
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uids-file",
        type=Path,
        default=Path("ready_for_windowing.txt"),
        help="UID list（每行: take_uid [take_name ...]）。",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Ego-Exo4D cache 根目录（默认 config.cache.raw_root）。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="窗口化输出目录（默认 config.windowing.output_root 或 run-root/windows）。",
    )
    parser.add_argument("--window", type=float, default=None, help="窗口长度（秒）。")
    parser.add_argument("--hop", type=float, default=None, help="滑窗步长（秒）。")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="序列长度（采样点数）用于序列化输出。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="配置文件路径。",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="可选 runs/<run_id> 目录；输出默认写入 <run_root>/windows。",
    )
    return parser.parse_args()


def read_uids(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    uids: List[Tuple[str, str]] = []
    seen = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        uid = parts[0]
        name = parts[1] if len(parts) > 1 else uid
        if uid not in seen:
            uids.append((uid, name))
            seen.add(uid)
    return uids


def load_gaze_df(take_dir: Path) -> pd.DataFrame:
    gaze_dir = take_dir / "eye_gaze"
    csv_candidates = [
        gaze_dir / "personalized_eye_gaze.csv",
        gaze_dir / "general_eye_gaze.csv",
    ]
    for csv_path in csv_candidates:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            break
    else:
        raise FileNotFoundError(f"No gaze CSV under {gaze_dir}")

    if "tracking_timestamp_us" not in df.columns:
        raise ValueError(f"tracking_timestamp_us missing in {csv_path}")

    yaw_cols = [c for c in GAZE_YAW_COLS if c in df.columns]
    pitch_cols = [c for c in GAZE_PITCH_COLS if c in df.columns]
    if yaw_cols:
        df["avg_yaw_rads"] = df[yaw_cols].mean(axis=1)
    if pitch_cols:
        df["avg_pitch_rads"] = df[pitch_cols].mean(axis=1)
    keep_cols = [
        "tracking_timestamp_us",
        "avg_yaw_rads",
        "avg_pitch_rads",
    ]
    if "depth_m" in df.columns:
        keep_cols.append("depth_m")
    df = df[keep_cols].dropna()
    return df


def load_traj_df(take_dir: Path) -> pd.DataFrame:
    traj_path = take_dir / "trajectory" / "closed_loop_trajectory.csv"
    if not traj_path.exists():
        raise FileNotFoundError(f"{traj_path} missing")
    df = pd.read_csv(traj_path)
    if "tracking_timestamp_us" not in df.columns:
        raise ValueError("trajectory CSV missing tracking_timestamp_us")

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
        df["linear_speed"] = np.linalg.norm(df[lin_cols].to_numpy(), axis=1)
    else:
        df["linear_speed"] = 0.0
    if all(col in df.columns for col in ang_cols):
        df["angular_speed"] = np.linalg.norm(df[ang_cols].to_numpy(), axis=1)
    else:
        df["angular_speed"] = 0.0
    keep_cols = [
        "tracking_timestamp_us",
        "linear_speed",
        "angular_speed",
    ]
    if "angular_velocity_z_device" in df.columns:
        keep_cols.append("angular_velocity_z_device")
    else:
        df["angular_velocity_z_device"] = 0.0
        keep_cols.append("angular_velocity_z_device")
    return df[keep_cols]


def normalize_time(
    gaze_df: pd.DataFrame, traj_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    t0 = min(
        gaze_df["tracking_timestamp_us"].min(),
        traj_df["tracking_timestamp_us"].min(),
    )
    gaze_df = gaze_df.copy()
    traj_df = traj_df.copy()
    gaze_df["time_s"] = (gaze_df["tracking_timestamp_us"].to_numpy() - t0) / 1_000_000.0
    traj_df["time_s"] = (traj_df["tracking_timestamp_us"].to_numpy() - t0) / 1_000_000.0
    gaze_rate = estimate_sample_rate(gaze_df["tracking_timestamp_us"].to_numpy())
    traj_rate = estimate_sample_rate(traj_df["tracking_timestamp_us"].to_numpy())
    return gaze_df, traj_df, gaze_rate, traj_rate


def estimate_sample_rate(timestamps_us: np.ndarray) -> float:
    if len(timestamps_us) < 2:
        return 0.0
    diffs = np.diff(np.sort(timestamps_us))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.0
    median_dt = np.median(diffs)
    if median_dt <= 0:
        return 0.0
    return 1_000_000.0 / median_dt


def resample_timeseries(
    df: pd.DataFrame,
    cols: List[str],
    target_hz: float,
    t_start: float,
    t_end: float,
) -> pd.DataFrame:
    if len(df) < 2 or t_end - t_start <= 0:
        return pd.DataFrame(columns=["time_s"] + cols)
    dt = 1.0 / target_hz
    times = df["time_s"].to_numpy()
    values = df[cols].to_numpy()
    target_times = np.arange(t_start, t_end, dt)
    if len(target_times) == 0:
        return pd.DataFrame(columns=["time_s"] + cols)
    resampled = {"time_s": target_times}
    for col_idx, col in enumerate(cols):
        resampled[col] = np.interp(
            target_times,
            times,
            values[:, col_idx],
            left=np.nan,
            right=np.nan,
        )
    out = pd.DataFrame(resampled).dropna()
    return out


def robust_normalize(series: pd.Series) -> pd.Series:
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    scale = mad if mad > EPS else float(np.std(series))
    if scale <= EPS:
        scale = 1.0
    return (series - median) / scale


def compute_scan_return_rate(yaw_values: np.ndarray, sample_rate: float, threshold: float = 0.02) -> float:
    if len(yaw_values) < 3:
        return 0.0
    diffs = np.diff(yaw_values)
    mask = np.abs(diffs) >= threshold
    if mask.sum() < 2:
        return 0.0
    signs = np.sign(diffs[mask])
    changes = np.sum(np.diff(signs) != 0)
    duration = len(yaw_values) / sample_rate
    if duration <= 0:
        return 0.0
    return float(changes) / duration


def compute_eye_head_metrics(
    gaze_yaw_vel: np.ndarray,
    head_yaw_vel: np.ndarray,
    sample_rate: float,
    max_lag_s: float = 0.5,
) -> Tuple[float, float]:
    if len(gaze_yaw_vel) < 5 or len(head_yaw_vel) < 5:
        return 0.0, 0.0
    max_lag = int(max_lag_s * sample_rate)
    best_corr = -2.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            gv = gaze_yaw_vel[-lag:]
            hv = head_yaw_vel[: len(gv)]
        elif lag > 0:
            gv = gaze_yaw_vel[: len(gaze_yaw_vel) - lag]
            hv = head_yaw_vel[lag:]
        else:
            gv = gaze_yaw_vel
            hv = head_yaw_vel
        length = min(len(gv), len(hv))
        if length < 10:
            continue
        gv = gv[:length]
        hv = hv[:length]
        corr = np.corrcoef(gv, hv)[0, 1]
        if np.isnan(corr):
            continue
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    lag_ms = best_lag / sample_rate * 1000.0
    if not np.isfinite(lag_ms):
        lag_ms = 0.0
    if not np.isfinite(best_corr):
        best_corr = 0.0
    return float(lag_ms), float(best_corr)


def interpolate_sequence(
    times: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
    seq_len: int,
) -> np.ndarray:
    if len(times) == 0:
        return np.zeros((seq_len, values.shape[1]))
    target = np.linspace(start, end, seq_len, endpoint=False)
    seq = np.zeros((seq_len, values.shape[1]))
    for dim in range(values.shape[1]):
        seq[:, dim] = np.interp(
            target,
            times,
            values[:, dim],
            left=values[0, dim],
            right=values[-1, dim],
        )
    return seq


def compute_window_features(
    gaze_seg: pd.DataFrame,
    traj_seg: pd.DataFrame,
    gaze_orig_count: int,
    traj_orig_count: int,
    window: float,
    gaze_hz: float,
    imu_hz: float,
    lag_ms: float,
    corr: float,
) -> Dict[str, float]:
    features: Dict[str, float] = {}
    features["gaze_yaw_mean"] = float(gaze_seg["avg_yaw_rads"].mean())
    features["gaze_yaw_std"] = float(gaze_seg["avg_yaw_rads"].std(ddof=0) or 0.0)
    features["gaze_pitch_mean"] = float(gaze_seg["avg_pitch_rads"].mean())
    features["gaze_pitch_std"] = float(gaze_seg["avg_pitch_rads"].std(ddof=0) or 0.0)
    features["gaze_dispersion"] = math.sqrt(
        features["gaze_yaw_std"] ** 2 + features["gaze_pitch_std"] ** 2
    )
    features["traj_linear_mean"] = float(traj_seg["linear_speed"].mean())
    features["traj_linear_std"] = float(traj_seg["linear_speed"].std(ddof=0) or 0.0)
    features["traj_angular_mean"] = float(traj_seg["angular_speed"].mean())
    features["traj_angular_std"] = float(traj_seg["angular_speed"].std(ddof=0) or 0.0)
    features["gaze_count"] = int(len(gaze_seg))
    features["traj_count"] = int(len(traj_seg))
    expected_gaze = max(window * gaze_hz, EPS)
    expected_traj = max(window * imu_hz, EPS)
    features["gaze_missing_ratio"] = float(
        max(0.0, 1.0 - min(gaze_orig_count / expected_gaze, 1.0))
    )
    features["traj_missing_ratio"] = float(
        max(0.0, 1.0 - min(traj_orig_count / expected_traj, 1.0))
    )
    features["scan_return_rate"] = float(
        compute_scan_return_rate(gaze_seg["avg_yaw_rads"].to_numpy(), gaze_hz)
    )
    if "linear_speed_norm" in traj_seg.columns:
        diffs = np.diff(traj_seg["linear_speed_norm"].to_numpy())
        features["phase_velocity"] = float(np.mean(np.abs(diffs)) * imu_hz) if len(diffs) > 0 else 0.0
    else:
        features["phase_velocity"] = 0.0
    features["eye_head_lag_ms"] = lag_ms
    features["eye_head_corr"] = corr
    return features


def process_take(
    take_uid: str,
    take_name: str,
    cache_root: Path,
    output_root: Path,
    window: float,
    hop: float,
    seq_len: int,
    gaze_hz: float,
    imu_hz: float,
) -> None:
    take_dir = cache_root / "takes" / take_name
    if not take_dir.exists():
        raise FileNotFoundError(f"{take_dir} missing")

    gaze_df = load_gaze_df(take_dir)
    traj_df = load_traj_df(take_dir)
    gaze_df, traj_df, gaze_rate, traj_rate = normalize_time(gaze_df, traj_df)
    effective_gaze_hz = gaze_rate or gaze_hz
    effective_imu_hz = traj_rate or imu_hz

    start_time = max(gaze_df["time_s"].min(), traj_df["time_s"].min())
    end_time = min(gaze_df["time_s"].max(), traj_df["time_s"].max())
    if end_time - start_time < window:
        print(f"[warn] {take_uid} insufficient overlap")
        return

    gaze_res = resample_timeseries(
        gaze_df,
        ["avg_yaw_rads", "avg_pitch_rads"],
        effective_gaze_hz,
        start_time,
        end_time,
    )
    traj_res = resample_timeseries(
        traj_df,
        ["linear_speed", "angular_speed", "angular_velocity_z_device"],
        effective_imu_hz,
        start_time,
        end_time,
    )
    if gaze_res.empty or traj_res.empty:
        print(f"[warn] {take_uid} resample produced empty data")
        return

    for col in ["avg_yaw_rads", "avg_pitch_rads"]:
        gaze_res[f"{col}_norm"] = robust_normalize(gaze_res[col])
    gaze_res["yaw_vel"] = np.gradient(gaze_res["avg_yaw_rads"], 1.0 / gaze_hz, edge_order=1)
    gaze_res["pitch_vel"] = np.gradient(gaze_res["avg_pitch_rads"], 1.0 / gaze_hz, edge_order=1)
    gaze_res["yaw_vel_norm"] = robust_normalize(gaze_res["yaw_vel"])
    gaze_res["pitch_vel_norm"] = robust_normalize(gaze_res["pitch_vel"])

    traj_res["head_yaw_vel"] = np.gradient(
        traj_res["angular_velocity_z_device"],
        1.0 / imu_hz,
        edge_order=1,
    )
    for col in ["linear_speed", "angular_speed", "angular_velocity_z_device", "head_yaw_vel"]:
        traj_res[f"{col}_norm"] = robust_normalize(traj_res[col])

    stats_rows: List[Dict[str, float]] = []
    gaze_seqs: List[np.ndarray] = []
    traj_seqs: List[np.ndarray] = []

    t = start_time
    window_idx = 0
    while t + window <= end_time:
        gaze_seg_res = gaze_res[(gaze_res["time_s"] >= t) & (gaze_res["time_s"] < t + window)]
        traj_seg_res = traj_res[(traj_res["time_s"] >= t) & (traj_res["time_s"] < t + window)]
        if len(gaze_seg_res) < 2 or len(traj_seg_res) < 2:
            t += hop
            continue
        gaze_orig = gaze_df[(gaze_df["time_s"] >= t) & (gaze_df["time_s"] < t + window)]
        traj_orig = traj_df[(traj_df["time_s"] >= t) & (traj_df["time_s"] < t + window)]

        head_vel_interp = np.interp(
            gaze_seg_res["time_s"].to_numpy(),
            traj_seg_res["time_s"].to_numpy(),
            traj_seg_res["head_yaw_vel"].to_numpy(),
            left=np.nan,
            right=np.nan,
        )
        valid = ~np.isnan(head_vel_interp)
        if valid.sum() >= 5:
            lag_ms, corr = compute_eye_head_metrics(
                gaze_seg_res["yaw_vel"].to_numpy()[valid],
                head_vel_interp[valid],
                effective_gaze_hz,
            )
        else:
            lag_ms, corr = 0.0, 0.0

        features = compute_window_features(
            gaze_seg_res,
            traj_seg_res,
            len(gaze_orig),
            len(traj_orig),
            window,
            effective_gaze_hz,
            effective_imu_hz,
            lag_ms,
            corr,
        )
        features.update(
            {
                "take_uid": take_uid,
                "take_name": take_name,
                "window_index": window_idx,
                "window_start_s": float(t),
                "window_end_s": float(t + window),
            }
        )
        stats_rows.append(features)

        gaze_values = gaze_seg_res[
            [
                "avg_yaw_rads_norm",
                "avg_pitch_rads_norm",
                "yaw_vel_norm",
                "pitch_vel_norm",
            ]
        ].to_numpy()
        traj_values = traj_seg_res[
            [
                "linear_speed_norm",
                "angular_speed_norm",
                "head_yaw_vel_norm",
            ]
        ].to_numpy()
        gaze_seqs.append(
            interpolate_sequence(
                gaze_seg_res["time_s"].to_numpy(),
                gaze_values,
                t,
                t + window,
                seq_len,
            )
        )
        traj_seqs.append(
            interpolate_sequence(
                traj_seg_res["time_s"].to_numpy(),
                traj_values,
                t,
                t + window,
                seq_len,
            )
        )
        t += hop
        window_idx += 1

    if not stats_rows:
        print(f"[warn] {take_uid} produced no windows")
        return

    take_out = output_root / take_uid
    take_out.mkdir(parents=True, exist_ok=True)
    stats_df = pd.DataFrame(stats_rows)
    stats_path = take_out / "stats.parquet"
    stats_df.to_parquet(stats_path, index=False)

    seq_path = take_out / "seq.npz"
    gaze_channels = ["yaw_norm", "pitch_norm", "yaw_vel_norm", "pitch_vel_norm"]
    traj_channels = ["linear_speed_norm", "angular_speed_norm", "head_yaw_vel_norm"]
    np.savez_compressed(
        seq_path,
        gaze=np.stack(gaze_seqs, axis=0),
        traj=np.stack(traj_seqs, axis=0),
        gaze_channels=np.array(gaze_channels),
        traj_channels=np.array(traj_channels),
    )

    meta = {
        "take_uid": take_uid,
        "take_name": take_name,
        "window_size_s": window,
        "hop_s": hop,
        "seq_len": seq_len,
        "num_windows": int(len(stats_rows)),
        "stats_path": str(stats_path),
        "seq_path": str(seq_path),
        "gaze_sample_hz": effective_gaze_hz,
        "imu_sample_hz": effective_imu_hz,
        "gaze_channels": gaze_channels,
        "traj_channels": traj_channels,
        "feature_version": FEATURE_VERSION,
    }
    (take_out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] {take_uid}: {len(stats_rows)} windows -> {take_out}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.cache_root is None:
        args.cache_root = Path(get_from_config(cfg, ["cache", "raw_root"], "egoexo_cache"))
    if args.run_root:
        run_root = Path(args.run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        if args.output_root is None:
            args.output_root = run_root / "windows"
        else:
            args.output_root = Path(args.output_root)
    if args.output_root is None:
        args.output_root = Path(get_from_config(cfg, ["windowing", "output_root"], "aligned_windows"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.window is None:
        args.window = float(get_from_config(cfg, ["windowing", "window_sec"], 3.0))
    if args.hop is None:
        args.hop = float(get_from_config(cfg, ["windowing", "hop_sec"], 0.2))
    if args.seq_len is None:
        args.seq_len = int(get_from_config(cfg, ["windowing", "seq_len"], 50))
    gaze_hz = float(get_from_config(cfg, ["windowing", "gaze_target_hz"], 30.0))
    imu_hz = float(get_from_config(cfg, ["windowing", "imu_target_hz"], 25.0))

    uids = read_uids(args.uids_file)
    if not uids:
        raise ValueError(f"No take_uids found in {args.uids_file}")

    for take_uid, take_name in uids:
        try:
            process_take(
                take_uid=take_uid,
                take_name=take_name,
                cache_root=args.cache_root,
                output_root=args.output_root,
                window=args.window,
                hop=args.hop,
                seq_len=args.seq_len,
                gaze_hz=gaze_hz,
                imu_hz=imu_hz,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {take_uid}: {exc}")


if __name__ == "__main__":
    main()
