import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import os

def validate_data(processed_dir):
    processed_dir = Path(processed_dir)
    if not processed_dir.exists():
        print(f"Error: Directory {processed_dir} does not exist.")
        return

    files = list(processed_dir.rglob("seq.npz"))
    print(f"Found {len(files)} files to validate in {processed_dir}.")
    
    corrupted = []
    empty = []
    valid_count = 0
    
    for f in tqdm(files, desc="Validating"):
        try:
            data = np.load(f)
            
            # Check keys
            if 'traj' not in data:
                print(f"Missing 'traj' key: {f}")
                corrupted.append(f)
                continue
                
            traj = data['traj'] # (N, Window, Channels)
            
            if len(traj) == 0:
                empty.append(f)
                continue
                
            # Check for NaNs or Infs
            if np.isnan(traj).any():
                print(f"NaNs found in traj: {f}")
                corrupted.append(f)
                continue
                
            if np.isinf(traj).any():
                print(f"Infs found in traj: {f}")
                corrupted.append(f)
                continue
                
            # Check shape consistency (optional, but good)
            # Expecting (N, 50, 6)
            if traj.ndim != 3 or traj.shape[1] != 50 or traj.shape[2] != 6:
                print(f"Unexpected shape {traj.shape}: {f}")
                corrupted.append(f)
                continue
                
            valid_count += 1
                
        except Exception as e:
            print(f"Error reading {f}: {e}")
            corrupted.append(f)
            
    print(f"\n=== Validation Summary ===")
    print(f"Total Files Scanned: {len(files)}")
    print(f"Valid Files: {valid_count}")
    print(f"Empty Files: {len(empty)}")
    print(f"Corrupted Files: {len(corrupted)}")
    
    if corrupted:
        print("\n!!! Corrupted Files (Delete these or re-process) !!!")
        for c in corrupted:
            print(c)
            
    if valid_count == 0:
        print("\nWARNING: No valid data found. Training will fail.")
    else:
        print("\nData looks good! You can proceed to training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=str, default="data/processed_ego4d")
    args = parser.parse_args()
    
    validate_data(args.processed_dir)
