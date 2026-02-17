#!/usr/bin/env python3
"""
Split action labels into chunks of max 10k rows for Label Studio import.
Videos are never split across chunks — all rows for a video_uid stay together.

Output: data/labels/action_labels_refined_splits/chunk_001.csv, chunk_002.csv, ...

Usage:
    python scripts/split_labels_for_labelstudio.py
    python scripts/split_labels_for_labelstudio.py --max-rows 5000
"""

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Split action labels into chunks (max rows per chunk, videos kept whole)."
    )
    parser.add_argument(
        "--input",
        default="data/labels/action_labels_llm_clean_refined.csv",
        help="Path to action labels CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="data/labels/action_labels_refined_splits",
        help="Output directory for chunk CSV files",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10_000,
        help="Maximum rows per chunk (default: 10000)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return 1

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    total_rows = len(df)
    print(f"Total rows: {total_rows}")

    # Group by video_uid, preserve order
    groups = list(df.groupby("video_uid", sort=False))

    os.makedirs(args.output_dir, exist_ok=True)

    chunks = []
    current_chunk = []
    current_count = 0

    for video_uid, group_df in groups:
        n = len(group_df)
        # If a single video exceeds max_rows, it goes in its own chunk
        if current_count + n > args.max_rows and current_count > 0:
            chunks.append(pd.concat(current_chunk, ignore_index=True))
            current_chunk = []
            current_count = 0
        current_chunk.append(group_df)
        current_count += n

    if current_chunk:
        chunks.append(pd.concat(current_chunk, ignore_index=True))

    for i, chunk_df in enumerate(chunks, start=1):
        path = os.path.join(args.output_dir, f"chunk_{i:03d}.csv")
        chunk_df.to_csv(path, index=False)
        print(f"  chunk_{i:03d}.csv: {len(chunk_df)} rows, {chunk_df['video_uid'].nunique()} videos")

    print(f"\nSaved {len(chunks)} chunks to {args.output_dir}/")
    print(f"Total rows across chunks: {sum(len(c) for c in chunks)}")
    return 0


if __name__ == "__main__":
    exit(main())
