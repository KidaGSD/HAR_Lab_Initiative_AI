#!/usr/bin/env python3
"""
Combine all CSV files from a directory into a single CSV file, sorted by batch.
Assumes files are named like HAR_B01.csv, HAR_B02.csv, etc.

Usage:
  python scripts/combine_refined_batches.py input_dir output_file
"""

import argparse
import os
import pandas as pd
import re

def get_batch_number(filename):
    """Extract batch number from filename (e.g., 'HAR_B01.csv' -> 1)."""
    match = re.search(r'HAR_B(\d+)', filename)
    if match:
        return int(match.group(1))
    return float('inf')  # Put files without batch number at the end

def main():
    parser = argparse.ArgumentParser(description="Combine refined batch CSVs.")
    parser.add_argument("input_dir", help="Directory containing the CSV files")
    parser.add_argument("output_file", help="Path to the output combined CSV file")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1

    # List all CSV files
    files = [f for f in os.listdir(args.input_dir) if f.endswith('.csv') and f.startswith('HAR_B')]
    
    # Sort files by batch number
    files.sort(key=get_batch_number)
    
    if not files:
        print(f"No matching CSV files found in {args.input_dir}")
        return 1

    print(f"Found {len(files)} files to combine (in order):")
    for f in files:
        print(f"  - {f}")

    combined_df = pd.DataFrame()
    
    for filename in files:
        filepath = os.path.join(args.input_dir, filename)
        try:
            df = pd.read_csv(filepath)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return 1

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    combined_df.to_csv(args.output_file, index=False)
    print(f"\nSuccessfully combined {len(combined_df)} rows into {args.output_file}")

if __name__ == "__main__":
    main()
