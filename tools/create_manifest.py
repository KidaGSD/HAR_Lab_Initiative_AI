#!/usr/bin/env python3
"""Generate a run-level manifest with basic metadata.

Example:
  python tools/create_manifest.py --run-id 20251111_refactor --uids-file state/active/ready_for_windowing.txt
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def read_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "NA"


def read_lines(path: Optional[Path]) -> List[str]:
    if not path or not path.exists():
        return []
    return [line.strip().split()[0] for line in path.read_text().splitlines() if line.strip()]


def collect_state_files(state_dir: Path) -> Dict[str, Dict[str, float]]:
    info: Dict[str, Dict[str, float]] = {}
    if not state_dir.exists():
        return info
    for fname in [
        "candidate_takes.csv",
        "candidate_takes.txt",
        "ready_for_alignment.txt",
        "ready_for_windowing.txt",
    ]:
        fpath = state_dir / fname
        if fpath.exists():
            stat = fpath.stat()
            info[fname] = {"size_bytes": stat.st_size, "mtime": stat.st_mtime}
    logs_dir = state_dir / "logs"
    if logs_dir.exists():
        info["logs_dir"] = {"entries": len(list(logs_dir.iterdir()))}
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Run identifier, e.g., 20251111_alignment")
    parser.add_argument("--output-root", type=Path, default=Path("runs"), help="Root directory to store run artifacts.")
    parser.add_argument("--uids-file", type=Path, default=None, help="Optional UID list included in manifest.")
    parser.add_argument("--state-dir", type=Path, default=Path("state/active"), help="Directory containing active state files.")
    parser.add_argument("--window", type=float, default=None)
    parser.add_argument("--hop", type=float, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--weak-label-rule", default=None)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    uids = read_lines(args.uids_file)
    state_info = collect_state_files(args.state_dir)

    manifest = {
        "run_id": args.run_id,
        "created_utc": dt.datetime.utcnow().isoformat() + "Z",
        "git_commit": read_git_commit(),
        "parameters": {
            "window_sec": args.window,
            "hop_sec": args.hop,
            "seq_len": args.seq_len,
            "weak_label_rule": args.weak_label_rule,
        },
        "inputs": {
            "uids_file": str(args.uids_file) if args.uids_file else None,
            "uids_count": len(uids),
            "state_dir": str(args.state_dir),
            "state_files": state_info,
        },
        "outputs_root": str(run_dir),
        "notes": args.notes,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
