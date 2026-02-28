#!/usr/bin/env python3
"""
Remove rows already used in Round 1 from the no-articles pool.

Default behavior:
- Input pool: data/labels/action_labels_llm_clean_refined_no_articles.csv
- Exclude set: data/annotation_rounds/r001_deduped/round1_deduped.csv
- Output: data/labels/action_labels_llm_clean_refined_no_articles_round2_pool.csv

Matching key is:
    video_uid + rounded(timestamp_sec * 1000)  # millisecond task key
"""

import argparse
from pathlib import Path

import pandas as pd


def make_task_key(df: pd.DataFrame) -> pd.Series:
    uid = df["video_uid"].astype(str).str.strip()
    ts_ms = pd.to_numeric(df["timestamp_sec"], errors="coerce").fillna(-1).mul(1000).round().astype("int64")
    return uid + "__" + ts_ms.astype(str)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove rows in round1_deduped from no-articles CSV."
    )
    parser.add_argument(
        "--pool",
        default="data/labels/action_labels_llm_clean_refined_no_articles.csv",
        help="CSV to filter (default: no-articles CSV)",
    )
    parser.add_argument(
        "--exclude",
        default="data/annotation_rounds/r001_deduped/round1_deduped.csv",
        help="CSV containing rows to remove (default: round1_deduped)",
    )
    parser.add_argument(
        "--output",
        default="data/labels/action_labels_llm_clean_refined_no_articles_round2_pool.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    pool_path = Path(args.pool)
    exclude_path = Path(args.exclude)
    output_path = Path(args.output)

    if not pool_path.exists():
        print(f"Error: pool file not found: {pool_path}")
        return 1
    if not exclude_path.exists():
        print(f"Error: exclude file not found: {exclude_path}")
        return 1

    pool_df = pd.read_csv(pool_path)
    ex_df = pd.read_csv(exclude_path)

    required_cols = {"video_uid", "timestamp_sec"}
    if not required_cols.issubset(pool_df.columns):
        print(f"Error: pool CSV must contain columns: {sorted(required_cols)}")
        return 1
    if not required_cols.issubset(ex_df.columns):
        print(f"Error: exclude CSV must contain columns: {sorted(required_cols)}")
        return 1

    pool_keys = make_task_key(pool_df)
    ex_keys = set(make_task_key(ex_df))

    keep_mask = ~pool_keys.isin(ex_keys)
    out_df = pool_df[keep_mask].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    removed = int((~keep_mask).sum())
    print(f"Pool rows:    {len(pool_df):,}")
    print(f"Exclude rows: {len(ex_df):,}")
    print(f"Removed:      {removed:,}")
    print(f"Remaining:    {len(out_df):,}")
    print(f"Saved:        {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
