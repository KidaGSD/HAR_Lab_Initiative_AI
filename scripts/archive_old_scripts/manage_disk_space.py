import pandas as pd
from pathlib import Path
import random
import os

def manage_space(imu_dir, target_uids_file, keep_count=200):
    imu_dir = Path(imu_dir)
    target_uids_file = Path(target_uids_file)
    
    # Get list of downloaded IMU files
    downloaded_files = list(imu_dir.glob("*.csv"))
    downloaded_uids = [f.stem for f in downloaded_files]
    
    print(f"Found {len(downloaded_uids)} downloaded IMU files.")
    
    if len(downloaded_uids) == 0:
        print("No files found. Exiting.")
        return

    # Select subset
    if len(downloaded_uids) > keep_count:
        keep_uids = random.sample(downloaded_uids, keep_count)
    else:
        keep_uids = downloaded_uids
        
    print(f"Keeping {len(keep_uids)} UIDs.")
    
    # Delete others
    deleted_count = 0
    freed_space = 0
    for f in downloaded_files:
        if f.stem not in keep_uids:
            size = f.stat().st_size
            f.unlink()
            deleted_count += 1
            freed_space += size
            
    print(f"Deleted {deleted_count} files, freed {freed_space / 1024 / 1024:.2f} MB.")
    
    # Update target_uids.csv
    # Read original to preserve other columns if any (though we only have video_uid usually)
    # Actually we should filter the original dataframe
    if target_uids_file.exists():
        df = pd.read_csv(target_uids_file)
        # Filter df to keep only keep_uids
        new_df = df[df['video_uid'].isin(keep_uids)]
        new_df.to_csv(target_uids_file, index=False)
        print(f"Updated {target_uids_file} with {len(new_df)} UIDs.")
    else:
        # Create new
        df = pd.DataFrame({'video_uid': keep_uids})
        df.to_csv(target_uids_file, index=False)
        print(f"Created {target_uids_file} with {len(df)} UIDs.")

if __name__ == "__main__":
    manage_space("data/ego4d_data/v2/imu", "target_uids.csv", keep_count=200)
