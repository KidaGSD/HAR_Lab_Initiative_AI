import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import scipy.interpolate

def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def align_and_resample(imu_df, gaze_df=None, target_fps=50):
    """
    Aligns IMU and Gaze data to a common timeline at target_fps.
    Gaze is optional.
    """
    if imu_df is None:
        return None
        
    # Standardize column names if needed
    # Ego4D IMU: accelerometer_x, ..., gyroscope_x, ..., timestamp (sometimes 'canonical_timestamp_s')
    # Ego4D Gaze: canonical_timestamp_s, norm_pos_x, norm_pos_y
    
    # Check timestamps
    imu_ts_col = None
    if 'canonical_timestamp_s' in imu_df.columns:
        imu_ts_col = 'canonical_timestamp_s'
    elif 'canonical_timestamp_ms' in imu_df.columns:
        imu_ts_col = 'canonical_timestamp_ms'
        imu_df[imu_ts_col] = imu_df[imu_ts_col] / 1000.0
    elif 'timestamp' in imu_df.columns:
        imu_ts_col = 'timestamp'
    
    if imu_ts_col is None:
        print(f"Missing timestamp column in IMU. Columns: {imu_df.columns.tolist()}")
        return None
        
    # Sort by timestamp
    imu_df = imu_df.sort_values(imu_ts_col)
    
    t_start = imu_df[imu_ts_col].min()
    t_end = imu_df[imu_ts_col].max()
    
    if gaze_df is not None:
        gaze_ts_col = 'canonical_timestamp_s' if 'canonical_timestamp_s' in gaze_df.columns else 'timestamp'
        if gaze_ts_col in gaze_df.columns:
            gaze_df = gaze_df.sort_values(gaze_ts_col)
            # Intersect time range if gaze exists
            t_start = max(t_start, gaze_df[gaze_ts_col].min())
            t_end = min(t_end, gaze_df[gaze_ts_col].max())
    
    if t_end <= t_start:
        print("No valid time range")
        return None
        
    # Create target timeline
    target_times = np.arange(t_start, t_end, 1.0/target_fps)
    
    # Interpolate IMU
    # Columns: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    imu_cols = [c for c in imu_df.columns if 'accel' in c.lower() or 'accl' in c.lower() or 'gyro' in c.lower()]
    if not imu_cols:
        # Fallback to known names
        imu_cols = ['accelerometer_x', 'accelerometer_y', 'accelerometer_z', 'gyroscope_x', 'gyroscope_y', 'gyroscope_z']
        
    # Ensure columns exist
    valid_imu_cols = [c for c in imu_cols if c in imu_df.columns]
    if not valid_imu_cols:
        print("No valid IMU columns found")
        return None
        
    imu_interp = scipy.interpolate.interp1d(
        imu_df[imu_ts_col], 
        imu_df[valid_imu_cols], 
        axis=0, 
        kind='linear', 
        fill_value="extrapolate"
    )
    aligned_imu = imu_interp(target_times)
    
    aligned_gaze = None
    valid_gaze_cols = []
    
    if gaze_df is not None:
        # Interpolate Gaze
        gaze_cols = ['norm_pos_x', 'norm_pos_y']
        valid_gaze_cols = [c for c in gaze_cols if c in gaze_df.columns]
        
        if valid_gaze_cols:
            gaze_interp = scipy.interpolate.interp1d(
                gaze_df[gaze_ts_col], 
                gaze_df[valid_gaze_cols], 
                axis=0, 
                kind='linear', 
                fill_value="extrapolate"
            )
            aligned_gaze = gaze_interp(target_times)
    
    return {
        'timestamp': target_times,
        'imu': aligned_imu,
        'gaze': aligned_gaze,
        'imu_cols': valid_imu_cols,
        'gaze_cols': valid_gaze_cols
    }

def make_windows(aligned_data, window_sec=1.0, hop_sec=1.0, fps=50):
    """
    Slices aligned data into sliding windows.
    Returns:
        windows: dict with 'imu' and 'gaze' arrays of shape (N, window_len, channels)
    """
    window_len = int(window_sec * fps)
    hop_len = int(hop_sec * fps)
    
    imu = aligned_data['imu']
    gaze = aligned_data['gaze']
    n_samples = len(imu)
    
    if n_samples < window_len:
        return None
        
    n_windows = (n_samples - window_len) // hop_len + 1
    
    imu_windows = []
    gaze_windows = []
    
    for i in range(n_windows):
        start = i * hop_len
        end = start + window_len
        imu_windows.append(imu[start:end])
        if gaze is not None:
            gaze_windows.append(gaze[start:end])
        
    return {
        'imu': np.array(imu_windows),
        'gaze': np.array(gaze_windows) if gaze_windows else np.array([])
    }

def process_pipeline(target_uids_file, data_dir, output_dir):
    target_uids_file = Path(target_uids_file)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(target_uids_file)
    uids = df['video_uid'].tolist()
    
    print(f"Processing {len(uids)} videos...")
    
    for uid in tqdm(uids):
        # Look for files
        imu_path = list(data_dir.rglob(f"{uid}.csv"))
        imu_path = [p for p in imu_path if 'imu' in str(p)]
        
        gaze_path = list(data_dir.rglob(f"{uid}.csv"))
        gaze_path = [p for p in gaze_path if 'gaze' in str(p)]
        
        if not imu_path:
            continue
            
        imu_df = load_csv(imu_path[0])
        gaze_df = load_csv(gaze_path[0]) if gaze_path else None
        
        aligned = align_and_resample(imu_df, gaze_df)
        
        if aligned:
            # Use 1s window and 1s hop for labeling candidates (non-overlapping for efficiency/clarity)
            # Or 1s window 0.5s hop?
            # EgoCHARM uses 1s window.
            windows = make_windows(aligned, window_sec=1.0, hop_sec=1.0)
            if windows:
                uid_dir = output_dir / uid
                uid_dir.mkdir(parents=True, exist_ok=True)
                
                # Keys expected by EgoExoDataset: 'gaze', 'traj' (IMU)
                np.savez(uid_dir / "seq.npz", gaze=windows['gaze'], traj=windows['imu'])
            
    print("Pipeline complete.")

def main():
    parser = argparse.ArgumentParser(description="Run Ego4D processing pipeline")
    parser.add_argument("--target-uids-file", type=str, default="target_uids.csv")
    parser.add_argument("--data-dir", type=str, default="data/ego4d_data")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    
    args = parser.parse_args()
    
    process_pipeline(args.target_uids_file, args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()
