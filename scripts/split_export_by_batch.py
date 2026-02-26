#!/usr/bin/env python3
"""
Split a Label Studio export CSV into multiple files based on the 'batch' column.
Outputs files named HAR_B{batch_number}.csv (e.g., HAR_B01.csv, HAR_B12.csv).

Usage:
  python scripts/split_export_by_batch.py input.csv output_dir
"""

import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Split export CSV by batch.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("output_dir", help="Directory to save the split files")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Error: Input file not found: {args.input_csv}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    if "batch" not in df.columns:
        print("Error: 'batch' column missing from CSV.")
        return 1

    # Get unique batches, ignoring NaNs
    batches = df["batch"].dropna().unique()
    print(f"Found {len(batches)} batches: {sorted(batches)}")

    for batch_val in batches:
        # Filter rows
        batch_df = df[df["batch"] == batch_val]
        
        # Determine filename
        try:
            # Try to format as 2-digit integer (e.g. 1 -> 01)
            batch_num = int(batch_val)
            filename = f"HAR_B{batch_num:02d}.csv"
        except ValueError:
            # Fallback for non-numeric batches
            filename = f"HAR_B{batch_val}.csv"

        output_path = os.path.join(args.output_dir, filename)
        batch_df.to_csv(output_path, index=False)
        print(f"  Saved {filename}: {len(batch_df)} rows")

    print("Done.")

if __name__ == "__main__":
    main()
