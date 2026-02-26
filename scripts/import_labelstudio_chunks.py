#!/usr/bin/env python3
"""
Import tasks into a Label Studio project.
Requires: pip install label-studio

Supports three input modes:
1) Single CSV: --input-csv (no chunking; use for <100k rows)
2) CSV chunk mode: --chunks-dir + optional --chunks
3) Prebuilt JSON: --tasks-json (list of {"data": {...}})

Usage:
  python scripts/import_labelstudio_chunks.py --test-auth
  python scripts/import_labelstudio_chunks.py --input-csv data/annotation_rounds/r001_validate/round1_assigned_only.csv
  python scripts/import_labelstudio_chunks.py --chunks 1 2
  python scripts/import_labelstudio_chunks.py --tasks-json data/annotation_rounds/r001/tasks/labelstudio_tasks.json
"""

import argparse
import json
import os
from typing import Any, Dict, List

import pandas as pd



def _test_auth(url: str, api_key: str) -> int:
    """Test API key by listing projects."""
    try:
        from label_studio_sdk import Client
    except ImportError:
        print("Error: pip install label-studio")
        return 1

    print(f"Testing connection to {url} ...")
    ls = Client(url=url, api_key=api_key)

    try:
        if hasattr(ls, "check_connection"):
            ls.check_connection()
            print("check_connection() OK")
    except Exception as e:  # pragma: no cover
        print(f"check_connection failed: {e}")

    try:
        projects = ls.list_projects()
        print(f"\nAPI key valid. Found {len(projects)} project(s):")
        for p in projects:
            pid = p.get("id", getattr(p, "id", "?")) if isinstance(p, dict) else getattr(p, "id", "?")
            title = p.get("title", getattr(p, "title", "?")) if isinstance(p, dict) else getattr(p, "title", "?")
            print(f"  - [{pid}] {title}")
        print("\nAuth test passed. You can run the import.")
        return 0
    except Exception as e:
        err_msg = str(e).lower()
        print(f"\nAuth failed: {e}")
        if "401" in err_msg or "unauthorized" in err_msg or "invalid token" in err_msg:
            print("Use a LEGACY token (Account & Settings -> Legacy Tokens), not a Personal Access Token.")
        return 1



def _clean_value(v: Any) -> Any:
    if pd.isna(v):
        return ""
    if isinstance(v, (int, float, bool, str)):
        return v
    return str(v)



def row_to_task(row: pd.Series) -> dict:
    """Convert any CSV row to Label Studio task format with all columns preserved."""
    data: Dict[str, Any] = {}
    for k, v in row.to_dict().items():
        data[str(k)] = _clean_value(v)
    return {"data": data}


def load_tasks_from_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "tasks" in obj and isinstance(obj["tasks"], list):
            return obj["tasks"]
        if "results" in obj and isinstance(obj["results"], list):
            return obj["results"]
    raise ValueError(f"Unsupported JSON task format: {path}")



def find_project(ls, project_id: int, project_name: str):
    if project_id is not None:
        return ls.get_project(project_id)

    projects = ls.list_projects()
    match = next(
        (
            p
            for p in projects
            if (p.get("title") if isinstance(p, dict) else getattr(p, "title", None)) == project_name
        ),
        None,
    )
    if not match:
        return None

    pid = match["id"] if isinstance(match, dict) else match.id
    return ls.get_project(pid)


def main():
    parser = argparse.ArgumentParser(description="Import tasks into Label Studio project HAR_dataset.")
    parser.add_argument(
        "--chunks-dir",
        default="data/labels/action_labels_refined_splits",
        help="Directory containing chunk_001.csv, chunk_002.csv, ...",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=None,
        help="Chunk numbers to import (e.g. 1 2 3). Default: all chunks in --chunks-dir",
    )
    parser.add_argument(
        "--input-csv",
        default="",
        help="Path to single CSV file. If set, import directly (no chunking).",
    )
    parser.add_argument(
        "--tasks-json",
        default="",
        help="Path to JSON tasks file (list of {'data': {...}}). If set, chunk mode is skipped.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"),
        help="Label Studio URL (or set LABEL_STUDIO_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LABEL_STUDIO_API_KEY"),
        help="Label Studio API key (or set LABEL_STUDIO_API_KEY)",
    )
    parser.add_argument(
        "--project",
        default="HAR_dataset",
        help="Label Studio project name (or use --project-id)",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=None,
        help="Label Studio project ID (overrides --project)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Import batch size (default: 100)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print sample tasks without importing")
    parser.add_argument("--test-auth", action="store_true", help="Test API key and exit")
    args = parser.parse_args()

    if not args.api_key and not args.dry_run and not args.test_auth:
        print("Error: Set LABEL_STUDIO_API_KEY or pass --api-key (or use --dry-run)")
        return 1

    if args.test_auth:
        return _test_auth(args.url, args.api_key)

    all_tasks: List[dict] = []

    if args.tasks_json:
        all_tasks = load_tasks_from_json(args.tasks_json)
        print(f"Loaded {len(all_tasks)} tasks from JSON: {args.tasks_json}")
    elif args.input_csv:
        if not os.path.exists(args.input_csv):
            print(f"Error: CSV not found: {args.input_csv}")
            return 1
        df = pd.read_csv(args.input_csv)
        all_tasks = [row_to_task(row) for _, row in df.iterrows()]
        print(f"Loaded {len(all_tasks)} tasks from CSV: {args.input_csv}")
    else:
        chunk_files = sorted(
            f for f in os.listdir(args.chunks_dir) if f.startswith("chunk_") and f.endswith(".csv")
        )
        if not chunk_files:
            print(f"Error: No chunk_*.csv files in {args.chunks_dir}")
            return 1

        if args.chunks is not None:
            allow = {f"chunk_{i:03d}.csv" for i in args.chunks}
            chunk_files = [f for f in chunk_files if f in allow]
            if not chunk_files:
                print(f"Error: No matching chunks for {args.chunks}")
                return 1

        for cf in chunk_files:
            path = os.path.join(args.chunks_dir, cf)
            df = pd.read_csv(path)
            tasks = [row_to_task(row) for _, row in df.iterrows()]
            all_tasks.extend(tasks)
            print(f"  {cf}: {len(tasks)} tasks")

    if not all_tasks:
        print("No tasks to import.")
        return 1

    print(f"\nTotal tasks to import: {len(all_tasks)}")

    if args.dry_run:
        print("\nDry run sample task:")
        print(all_tasks[0])
        return 0

    try:
        from label_studio_sdk import Client
    except ImportError:
        print("Error: pip install label-studio")
        return 1

    ls = Client(url=args.url, api_key=args.api_key)

    try:
        project = find_project(ls, args.project_id, args.project)
        if project is None:
            print(f"Error: Project '{args.project}' not found. Create it or use --project-id.")
            return 1
    except Exception as e:
        err_msg = str(e).lower()
        if "401" in err_msg or "unauthorized" in err_msg or "invalid token" in err_msg:
            print("\nAUTH ERROR: Use a LEGACY token, not a Personal Access Token.")
            print("In Label Studio: Account & Settings -> Legacy Tokens")
        raise

    imported = 0
    bs = max(1, int(args.batch_size))
    for i in range(0, len(all_tasks), bs):
        batch = all_tasks[i : i + bs]
        project.import_tasks(batch)
        imported += len(batch)
        print(f"  Imported {imported}/{len(all_tasks)} tasks...")

    print(f"\nDone. Imported {imported} tasks into project '{args.project}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
