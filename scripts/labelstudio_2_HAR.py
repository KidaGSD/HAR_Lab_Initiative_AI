#!/usr/bin/env python3
"""
Copy a CSV to an output folder, excluding specified columns.

Usage:
  python scripts/drop_csv_columns.py input.csv output_dir --drop col1 col2 col3
  python scripts/drop_csv_columns.py data/raw.csv data/clean --drop status notes
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Copy CSV to output folder with specified columns removed.",
    )
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("output_dir", help="Output directory for the new CSV")
    args = parser.parse_args()

    # Columns to drop (hardcoded)
    columns_to_drop = [
        "annotator",
        "annotation_id",
        "status",
        "created_at",
        "lead_time",
        "updated_at",
        "id"  # Often redundant with annotation_id
        "inner_id",
        "project",
        "task_id",


    ]

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    
    # Drop columns that exist (ignore non-existent)
    to_drop = [c for c in columns_to_drop if c in df.columns]
    missing = set(columns_to_drop) - set(df.columns)
    
    if missing:
        print(f"Note: Columns not found (skipped): {missing}")

    if not to_drop:
        print("Warning: None of the target columns exist in the CSV.")
    
    df_out = df.drop(columns=to_drop)
    output_path = output_dir / input_path.name
    df_out.to_csv(output_path, index=False)

    print(f"Dropped: {to_drop}")
    print(f"Output: {output_path} ({len(df_out)} rows, {len(df_out.columns)} columns)")
    return 0


if __name__ == "__main__":
    exit(main())
