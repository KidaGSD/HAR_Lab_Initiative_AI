#!/usr/bin/env python3
"""
Batch downloader for Ego-Exo4D modalities (mirrors egoexo_download.ipynb behavior).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_loader import get_from_config, load_config
DEFAULT_PARTS = (
    "take_trajectory",
    "take_eye_gaze",
    "take_vrs_noimagestream",
    "downscaled_takes/448",
)


def parse_access_key_file(path: Path) -> Tuple[str, str]:
    """
    Expected format (same as egoexo_download.ipynb):
        Access ID: <id>
        Access Key: <secret>
    """
    if not path.exists():
        raise FileNotFoundError(f"access key file not found: {path}")
    access_id = access_secret = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.lower().startswith("access id"):
            access_id = line.split(":", 1)[1].strip()
        elif line.lower().startswith("access key"):
            access_secret = line.split(":", 1)[1].strip()
    if not access_id or not access_secret:
        raise ValueError(f"Failed to parse Access ID/Key from {path}")
    return access_id, access_secret


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uids-file", type=Path, required=True, help="candidate_takes.csv or UID list")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination cache directory (default: config.cache.raw_root).",
    )
    parser.add_argument(
        "--parts",
        type=str,
        default=",".join(DEFAULT_PARTS),
        help=f"Comma-separated list of egoexo parts (default: {', '.join(DEFAULT_PARTS)}).",
    )
    parser.add_argument("--batch-size", type=int, default=20, help="Number of take_uids per command.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on total take_uids.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory for JSON logs (default: config.downloads.log_dir).")
    parser.add_argument("--yes", action="store_true", help="Pass --yes to egoexo.")
    parser.add_argument(
        "--access-key-file",
        type=Path,
        default=Path("access_key.txt"),
        help="Access key file (same format as notebook).",
    )
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region (default: us-east-1).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Config file providing default paths.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Optional runs/<run_id> directory; log-dir defaults to <run_root>/state/logs.",
    )
    parser.add_argument(
        "--include-scenarios",
        type=str,
        default=None,
        help="Comma-separated scenario names to include (case-insensitive).",
    )
    parser.add_argument(
        "--exclude-scenarios",
        type=str,
        default=None,
        help="Comma-separated scenario names to exclude.",
    )
    return parser.parse_args()


def collect_uids(path: Path) -> List[Tuple[str, Optional[str]]]:
    if path.suffix.lower() == ".csv":
        with path.open() as fp:
            reader = csv.DictReader(fp)
            if "take_uid" not in reader.fieldnames:
                raise ValueError("CSV must contain take_uid column")
            return [
                (row["take_uid"], row.get("scenario"))
                for row in reader
                if row.get("take_uid")
            ]
    with path.open() as fp:
        return [(line.strip().split()[0], None) for line in fp if line.strip()]


def chunked(seq: Iterable[str], size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in seq:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_parts(parts_arg: str) -> List[str]:
    return [p.strip() for p in parts_arg.split(",") if p.strip()] or list(DEFAULT_PARTS)


def normalize_list(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    return [item.strip().lower() for item in arg.split(",") if item.strip()]


def filter_entries(
    entries: List[Tuple[str, Optional[str]]],
    include: Optional[List[str]],
    exclude: Optional[List[str]],
) -> List[Tuple[str, Optional[str]]]:
    if not include and not exclude:
        return entries
    filtered: List[Tuple[str, Optional[str]]] = []
    for uid, scenario in entries:
        scenario_norm = scenario.lower() if isinstance(scenario, str) else None
        if include:
            if scenario_norm is None or scenario_norm not in include:
                continue
        if exclude and scenario_norm in exclude:
            continue
        filtered.append((uid, scenario))
    return filtered


def apply_config(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cache_root = get_from_config(cfg, ["cache", "raw_root"], "egoexo_cache")
    downloads_output = get_from_config(cfg, ["downloads", "output_root"], cache_root)
    downloads_log = get_from_config(cfg, ["downloads", "log_dir"], "logs")
    runs_root = get_from_config(cfg, ["outputs", "runs_root"], "runs")

    if args.output is None:
        args.output = Path(downloads_output)
    if args.log_dir is None:
        default_log_dir = Path(downloads_log)
    else:
        default_log_dir = args.log_dir

    if args.run_root:
        run_root = Path(args.run_root)
        args.run_root = run_root
        default_log_dir = run_root / "state" / "logs"
        run_root.mkdir(parents=True, exist_ok=True)
    else:
        Path(runs_root).mkdir(parents=True, exist_ok=True)

    args.log_dir = default_log_dir
    args.include_scenarios = normalize_list(args.include_scenarios)
    args.exclude_scenarios = normalize_list(args.exclude_scenarios)


def run_batch(batch: List[str], parts: List[str], args: argparse.Namespace, env: dict) -> tuple[str, str]:
    cmd = [
        "egoexo",
        "-o",
        str(args.output),
        "--parts",
    ] + parts + ["--uids"] + batch
    if args.yes:
        cmd.append("--yes")
    cmd_str = " ".join(cmd)
    if args.dry_run:
        print("[dry-run]", cmd_str)
        return "dry_run", cmd_str

    print("[exec]", cmd_str)
    try:
        subprocess.run(cmd, check=True, env=env)
        return "success", cmd_str
    except subprocess.CalledProcessError as exc:
        return f"failed:{exc.returncode}", cmd_str


def main() -> None:
    args = parse_args()
    apply_config(args)
    access_id, access_secret = parse_access_key_file(args.access_key_file)
    env = os.environ.copy()
    env.update(
        {
            "AWS_ACCESS_KEY_ID": access_id,
            "AWS_SECRET_ACCESS_KEY": access_secret,
            "AWS_DEFAULT_REGION": args.region,
            "AWS_REGION": args.region,
        }
    )

    entries = collect_uids(args.uids_file)
    entries = filter_entries(entries, args.include_scenarios, args.exclude_scenarios)
    uids = [uid for uid, _ in entries]
    if not uids:
        print("No take_uids found in", args.uids_file, file=sys.stderr)
        sys.exit(1)
    if args.limit:
        uids = uids[: args.limit]
    parts = parse_parts(args.parts)

    if not args.log_dir.exists():
        args.log_dir.mkdir(parents=True, exist_ok=True)
    log_entries = []
    log_path = args.log_dir / f"download_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

    for batch_idx, batch in enumerate(chunked(uids, args.batch_size), start=1):
        status, cmd_str = run_batch(batch, parts, args, env)
        log_entries.append(
            {
                "batch_index": batch_idx,
                "take_uids": batch,
                "status": status,
                "command": cmd_str,
                "timestamp_utc": datetime.utcnow().isoformat(),
                "parts": parts,
                "output": str(args.output),
                "dry_run": args.dry_run,
            }
        )
        if status.startswith("failed"):
            print(f"[warn] batch {batch_idx} failed; see log {log_path}", file=sys.stderr)

    with log_path.open("w") as fp:
        json.dump(log_entries, fp, indent=2)
    print(f"Wrote log to {log_path}")


if __name__ == "__main__":
    main()
