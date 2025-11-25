import argparse
import pandas as pd
import boto3
import os
from pathlib import Path
from tqdm import tqdm
import sys

def download_sensors(target_uids_file, output_dir):
    target_uids_file = Path(target_uids_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load target UIDs
    df = pd.read_csv(target_uids_file)
    target_uids = set(df['video_uid'].tolist())
    print(f"Targeting {len(target_uids)} videos.")
    
    # Manifest paths (assuming they were downloaded by CLI previously)
    # If not, we might need to fetch them. But we saw them in data/ego4d_data/v2/
    # Let's assume standard structure or allow override.
    # For now, look in data/ego4d_data/v2/
    
    # We need to know where the manifests are. 
    # The previous CLI run put them in data/ego4d_data/v2/{dataset}/manifest.csv
    # If they don't exist, we should probably warn or try to download them (but CLI is hard to control).
    # Let's assume they exist for now as per previous steps.
    
    base_data_dir = Path("data/ego4d_data") # Default
    
    datasets = ['imu', 'gaze']
    
    session = boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name="us-west-1"
    )
    s3 = session.client('s3')
    
    for ds in datasets:
        manifest_path = base_data_dir / "v2" / ds / "manifest.csv"
        if not manifest_path.exists():
            print(f"Manifest for {ds} not found at {manifest_path}. Please run 'ego4d --datasets {ds} --output_directory data/ego4d_data' (it will download manifest first).")
            continue
            
        print(f"Processing {ds} manifest...")
        manifest = pd.read_csv(manifest_path)
        
        # Filter for target UIDs
        # Manifest columns: video_uid, type, s3_path
        filtered = manifest[manifest['video_uid'].isin(target_uids)]
        
        print(f"Found {len(filtered)} files for {ds}.")
        
        # Download
        ds_out_dir = output_dir / "v2" / ds
        ds_out_dir.mkdir(parents=True, exist_ok=True)
        
        for _, row in tqdm(filtered.iterrows(), total=len(filtered), desc=f"Downloading {ds}"):
            s3_url = row['s3_path']
            uid = row['video_uid']
            
            # Parse bucket and key
            # s3://bucket/key
            parts = s3_url.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1]
            
            # Output filename
            # Keep original filename from key
            filename = Path(key).name
            local_path = ds_out_dir / filename
            
            if local_path.exists():
                # Check size? Or just skip
                continue
                
            try:
                s3.download_file(bucket, key, str(local_path))
            except Exception as e:
                print(f"Failed to download {uid} ({ds}): {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-uids-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
        sys.exit(1)
        
    download_sensors(args.target_uids_file, args.output_dir)
