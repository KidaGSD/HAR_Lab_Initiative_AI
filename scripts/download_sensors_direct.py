#!/usr/bin/env python3
"""Direct S3 downloader for Ego4D IMU and Gaze data without needing manifests."""

import argparse
import pandas as pd
import boto3
import os
from pathlib import Path
from tqdm import tqdm
import sys

def download_sensors_direct(target_uids_file, output_dir, manifest_dir=None):
    """Download IMU and Gaze data directly from S3 for target UIDs using manifests."""
    target_uids_file = Path(target_uids_file)
    output_dir = Path(output_dir)
    if manifest_dir:
        manifest_dir = Path(manifest_dir)
    
    # Load target UIDs
    df = pd.read_csv(target_uids_file)
    target_uids = set(df['video_uid'].tolist())
    print(f"Targeting {len(target_uids)} videos.")
    
    # Setup S3 client
    # Use default credential chain (env vars, ~/.aws/credentials, IAM role)
    aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    if aws_key and aws_secret:
        session = boto3.Session(
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name="us-west-1"
        )
    else:
        # Let boto3 use default credential chain (~/.aws/credentials)
        session = boto3.Session(region_name="us-west-1")
    
    s3 = session.client('s3')
    
    datasets = ['imu', 'gaze']
    
    for ds_name in datasets:
        print(f"\n=== Downloading {ds_name.upper()} data ===")
        ds_out_dir = output_dir / ds_name
        ds_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Manifest
        manifest_path = None
        if manifest_dir:
            manifest_path = manifest_dir / ds_name / "manifest.csv"
        
        manifest_df = None
        if manifest_path and manifest_path.exists():
            print(f"Loading manifest from {manifest_path}")
            manifest_df = pd.read_csv(manifest_path)
            # Filter manifest for target UIDs
            manifest_df = manifest_df[manifest_df['video_uid'].isin(target_uids)]
        else:
            print(f"Warning: No manifest found for {ds_name} at {manifest_path}. Skipping.")
            continue
            
        success_count = 0
        for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc=f"Downloading {ds_name}"):
            uid = row['video_uid']
            s3_url = row['s3_path']
            
            # Parse S3 URL
            # s3://bucket/key
            if not s3_url.startswith("s3://"):
                print(f"Invalid S3 URL for {uid}: {s3_url}")
                continue
                
            parts = s3_url.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1]
            
            local_path = ds_out_dir / f"{uid}.csv"
            
            if local_path.exists():
                success_count += 1
                continue
            
            try:
                s3.download_file(bucket, key, str(local_path))
                success_count += 1
            except Exception as e:
                print(f"\nFailed to download {uid} ({ds_name}): {e}")
        
        print(f"Successfully downloaded/found {success_count}/{len(target_uids)} {ds_name} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Ego4D sensor data directly from S3")
    parser.add_argument("--target-uids-file", type=str, required=True, help="CSV file with video_uid column")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for sensor data")
    parser.add_argument("--manifest-dir", type=str, help="Directory containing manifests (e.g., data/ego4d_data/v2)")
    args = parser.parse_args()
    
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        print("ERROR: Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
        sys.exit(1)
    
    download_sensors_direct(args.target_uids_file, args.output_dir, args.manifest_dir)
    print("\n✓ Download complete!")
