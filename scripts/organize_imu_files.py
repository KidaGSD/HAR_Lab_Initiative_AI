#!/usr/bin/env python3
"""
Organize existing IMU CSV files into the expected directory structure.
Moves/copies files from source directory to data/ego4d_data/v2/imu/
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil

def organize_imu_files(target_uids_file, source_dir, output_dir, copy=False):
    """
    Organize IMU CSV files into the expected structure.
    
    Args:
        target_uids_file: CSV file with video_uid column
        source_dir: Directory where IMU CSV files currently are
        output_dir: Target directory (e.g., data/ego4d_data/v2/imu)
        copy: If True, copy files instead of moving
    """
    target_uids_file = Path(target_uids_file)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load target UIDs
    df = pd.read_csv(target_uids_file)
    target_uids = set(df['video_uid'].tolist())
    print(f"Targeting {len(target_uids)} videos.")
    
    # Find all CSV files in source directory (recursive)
    source_files = list(source_dir.rglob("*.csv"))
    print(f"Found {len(source_files)} CSV files in source directory.")
    
    # Map UIDs to source files
    uid_to_file = {}
    for csv_file in source_files:
        # Try different naming patterns
        # Pattern 1: {uid}.csv
        # Pattern 2: {uid}_imu.csv
        # Pattern 3: imu_{uid}.csv
        # Pattern 4: {uid} in filename
        
        filename = csv_file.stem  # filename without extension
        
        # Check if filename matches a UID exactly
        if filename in target_uids:
            uid_to_file[filename] = csv_file
            continue
        
        # Check if filename contains UID with suffix/prefix
        for uid in target_uids:
            if filename == f"{uid}_imu" or filename == f"imu_{uid}":
                uid_to_file[uid] = csv_file
                break
            # Check if UID is in filename (more flexible)
            elif uid in filename and 'imu' in filename.lower():
                uid_to_file[uid] = csv_file
                break
    
    print(f"Matched {len(uid_to_file)} files to UIDs.")
    
    # Move/copy files
    action = "Copying" if copy else "Moving"
    success_count = 0
    skipped_count = 0
    
    for uid in tqdm(target_uids, desc=action):
        if uid not in uid_to_file:
            skipped_count += 1
            continue
        
        source_file = uid_to_file[uid]
        target_file = output_dir / f"{uid}.csv"
        
        # Skip if target already exists
        if target_file.exists():
            skipped_count += 1
            continue
        
        try:
            if copy:
                shutil.copy2(source_file, target_file)
            else:
                shutil.move(str(source_file), str(target_file))
            success_count += 1
        except Exception as e:
            print(f"\nError {action.lower()} {uid}: {e}")
    
    print(f"\n✓ {action} complete!")
    print(f"  Successfully {action.lower()}: {success_count}/{len(target_uids)}")
    print(f"  Skipped (already exists or not found): {skipped_count}")
    print(f"  Output directory: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize IMU CSV files into expected directory structure"
    )
    parser.add_argument(
        "--target-uids-file", 
        type=str, 
        default="target_uids.csv",
        help="CSV file with video_uid column"
    )
    parser.add_argument(
        "--source-dir", 
        type=str, 
        required=True,
        help="Directory where IMU CSV files currently are (will search recursively)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/ego4d_data/v2/imu",
        help="Target directory (default: data/ego4d_data/v2/imu)"
    )
    parser.add_argument(
        "--copy", 
        action="store_true",
        help="Copy files instead of moving them"
    )
    
    args = parser.parse_args()
    
    organize_imu_files(
        args.target_uids_file,
        args.source_dir,
        args.output_dir,
        copy=args.copy
    )
