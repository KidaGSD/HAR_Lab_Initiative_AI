#!/usr/bin/env python3
"""Direct S3 downloader for Ego4D IMU and Gaze data without needing manifests."""

import argparse
import pandas as pd
import boto3
import os
from pathlib import Path
from tqdm import tqdm
import sys

from src.aws_utils import get_aws_identity, make_boto3_session


def download_sensors_direct(
    target_uids_file,
    output_dir,
    manifest_dir=None,
    *,
    aws_region: str = "us-west-1",
    aws_profile: str | None = None,
    aws_creds_file: str | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
):
    """Download IMU and Gaze data directly from S3 for target UIDs using manifests."""
    target_uids_file = Path(target_uids_file)
    output_dir = Path(output_dir)
    if manifest_dir:
        manifest_dir = Path(manifest_dir)
    
    # Load target UIDs
    df = pd.read_csv(target_uids_file)
    target_uids = set(df['video_uid'].tolist())
    print(f"Targeting {len(target_uids)} videos.")
    
    # Setup S3 client (supports env vars, ~/.aws/credentials profiles, and optional creds file).
    session = make_boto3_session(
        region=aws_region,
        profile=aws_profile,
        creds_file=aws_creds_file,
        access_key_id=aws_access_key_id,
        secret_access_key=aws_secret_access_key,
        session_token=aws_session_token,
    )
    ok, ident = get_aws_identity(session)
    if ok:
        print(f"Using AWS Identity: {ident}")
    else:
        print(f"ERROR: Could not verify AWS credentials: {ident}")
        print(
            "Fix by either:\n"
            "  - Setting env vars: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY [/ AWS_SESSION_TOKEN]\n"
            "  - Using an AWS profile: --aws-profile <name> (from ~/.aws/credentials)\n"
            "  - Using a creds file: --aws-creds-file access_key.txt (Access ID/Access Key[/Session Token])"
        )
        return

    s3 = session.client("s3")

    
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
    parser.add_argument("--aws-region", type=str, default="us-west-1", help="AWS region (default: us-west-1)")
    parser.add_argument("--aws-profile", type=str, default=None, help="AWS profile name from ~/.aws/credentials")
    parser.add_argument(
        "--aws-creds-file",
        type=str,
        default=None,
        help='Optional creds file with "Access ID:" and "Access Key:" lines (and optional "Session Token:").',
    )
    parser.add_argument("--aws-access-key-id", type=str, default=None, help="Optional explicit AWS access key id")
    parser.add_argument("--aws-secret-access-key", type=str, default=None, help="Optional explicit AWS secret key")
    parser.add_argument("--aws-session-token", type=str, default=None, help="Optional AWS session token (temp creds)")
    args = parser.parse_args()

    download_sensors_direct(
        args.target_uids_file,
        args.output_dir,
        args.manifest_dir,
        aws_region=args.aws_region,
        aws_profile=args.aws_profile,
        aws_creds_file=args.aws_creds_file,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        aws_session_token=args.aws_session_token,
    )
    print("\n✓ Download complete!")
