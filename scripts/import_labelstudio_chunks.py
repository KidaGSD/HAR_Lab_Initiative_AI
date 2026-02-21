#!/usr/bin/env python3
"""
Import label chunks into a Label Studio project.
Requires: pip install label-studio

Set LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY (or pass --url / --api-key).
Project name: HAR_dataset

Usage:
    python scripts/import_labelstudio_chunks.py --test-auth        # test API key, list projects
    python scripts/import_labelstudio_chunks.py                    # import all chunks
    python scripts/import_labelstudio_chunks.py --chunks 1 2 3    # import specific chunks only
"""

import argparse
import os

import pandas as pd


def _test_auth(url: str, api_key: str) -> int:
    """Test API key: connect, list projects, optionally show account info."""
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
            print("✓ check_connection() OK")
    except Exception as e:
        print(f"check_connection failed: {e}")

    try:
        projects = ls.list_projects()
        print(f"\n✓ API key valid. Found {len(projects)} project(s):")
        for p in projects:
            pid = p.get("id", getattr(p, "id", "?"))
            title = p.get("title", getattr(p, "title", "?"))
            print(f"  - [{pid}] {title}")
        print("\nAuth test passed. You can run the import.")
        return 0
    except Exception as e:
        err_msg = str(e).lower()
        print(f"\n✗ Auth failed: {e}")
        if "401" in err_msg or "unauthorized" in err_msg or "invalid token" in err_msg:
            print("\nUse a LEGACY token (Account & Settings → Legacy Tokens), not a Personal Access Token.")
        return 1


def row_to_task(row: pd.Series) -> dict:
    """Convert a CSV row to Label Studio task format."""
    return {
        "data": {
            "batch": int(row["batch"]),
            "video_uid": str(row["video_uid"]),
            "timestamp_sec": float(row["timestamp_sec"]),
            "narration_text": str(row["narration_text"]),
            "scenario": str(row["scenario"]),
            "action": str(row["action"]),
            "reasoning": str(row["reasoning"]),
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Import action label chunks into Label Studio project HAR_dataset."
    )
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
        help="Chunk numbers to import (e.g. 1 2 3). Default: all.",
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
        help="Label Studio project name (or use --project-id for numeric ID)",
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=None,
        help="Label Studio project ID (overrides --project if set)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tasks without importing",
    )
    parser.add_argument(
        "--test-auth",
        action="store_true",
        help="Test API key: list projects and exit (no import)",
    )
    args = parser.parse_args()

    if not args.api_key and not args.dry_run and not args.test_auth:
        print("Error: Set LABEL_STUDIO_API_KEY or pass --api-key (or use --dry-run to preview)")
        return 1

    if args.test_auth:
        return _test_auth(args.url, args.api_key)

    chunk_files = sorted(
        f for f in os.listdir(args.chunks_dir)
        if f.startswith("chunk_") and f.endswith(".csv")
    )
    if not chunk_files:
        print(f"Error: No chunk_*.csv files in {args.chunks_dir}")
        return 1

    if args.chunks is not None:
        chunk_files = [f for f in chunk_files if any(f"chunk_{i:03d}.csv" == f for i in args.chunks)]
        if not chunk_files:
            print(f"Error: No matching chunks for {args.chunks}")
            return 1

    all_tasks = []
    for cf in chunk_files:
        path = os.path.join(args.chunks_dir, cf)
        df = pd.read_csv(path)
        tasks = [row_to_task(row) for _, row in df.iterrows()]
        all_tasks.extend(tasks)
        print(f"  {cf}: {len(tasks)} tasks")

    print(f"\nTotal tasks to import: {len(all_tasks)}")

    if args.dry_run:
        print("\nDry run. Sample task:")
        print(all_tasks[0])
        return 0

    try:
        from label_studio_sdk import Client
    except ImportError:
        print("Error: pip install label-studio")
        return 1

    ls = Client(url=args.url, api_key=args.api_key)

    try:
        if args.project_id is not None:
            project = ls.get_project(args.project_id)
        else:
            # List projects and find by title
            projects = ls.list_projects()
            match = next(
            (p for p in projects
             if (p.get("title") if isinstance(p, dict) else getattr(p, "title", None)) == args.project),
            None
        )
        if not match:
            print(f"Error: Project '{args.project}' not found. Create it in Label Studio or use --project-id <id>.")
            return 1
        pid = match["id"] if isinstance(match, dict) else match.id
        project = ls.get_project(pid)
    except Exception as e:
        err_msg = str(e).lower()
        if "401" in err_msg or "unauthorized" in err_msg or "invalid token" in err_msg:
            print("\n" + "="*60)
            print("AUTH ERROR: Use a LEGACY token, not a Personal Access Token.")
            print("="*60)
            print("In Label Studio: Account & Settings → Legacy Tokens")
            print("Create a new Legacy token and use it with --api-key")
            print("="*60 + "\n")
        raise

    # Import in batches (Label Studio recommends ~100–500 per request for large imports)
    batch_size = 100
    imported = 0
    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i : i + batch_size]
        project.import_tasks(batch)
        imported += len(batch)
        print(f"  Imported {imported}/{len(all_tasks)} tasks...")

    print(f"\nDone. Imported {imported} tasks into project '{args.project}'.")
    return 0


if __name__ == "__main__":
    exit(main())
