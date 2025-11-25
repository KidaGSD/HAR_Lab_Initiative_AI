#!/usr/bin/env python3
"""
Prepare video clips for VLM verification of Level 2 candidates.

Extracts key frames (start, middle, end) from high-scoring SVDD candidates
for GPT-4V analysis to verify if they represent "stuck/help needed" moments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import subprocess
import json

def extract_frames(video_path, timestamps, output_dir):
    """
    Extract frames at specific timestamps using ffmpeg.
    
    Args:
        video_path: Path to video file
        timestamps: List of timestamps in seconds
        output_dir: Directory to save frames
        
    Returns:
        List of paths to extracted frames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_paths = []
    video_name = Path(video_path).stem
    
    for i, ts in enumerate(timestamps):
        output_path = output_dir / f"{video_name}_frame_{i}_{ts:.2f}s.jpg"
        
        cmd = [
            'ffmpeg', '-ss', str(ts), '-i', str(video_path),
            '-vframes', '1', '-q:v', '2', str(output_path),
            '-y', '-loglevel', 'quiet'
        ]
        
        try:
            subprocess.run(cmd, check=True)
            frame_paths.append(output_path)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to extract frame at {ts}s: {e}")
    
    return frame_paths

def prepare_vlm_clips(args):
    """
    Main function to prepare clips for VLM verification.
    """
    # Load Level 2 candidates
    candidates_path = Path(args.candidates_file)
    if not candidates_path.exists():
        print(f"ERROR: Candidates file not found: {candidates_path}")
        print("Run: python scripts/generate_candidates.py first")
        return
    
    candidates_df = pd.read_csv(candidates_path)
    print(f"Loaded {len(candidates_df)} Level 2 candidates")
    
    # Filter to top candidates
    if args.top_k:
        candidates_df = candidates_df.nlargest(args.top_k, 'anomaly_score')
        print(f"Using top {args.top_k} candidates")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare VLM inputs
    vlm_inputs = []
    
    for idx, row in tqdm(candidates_df.iterrows(), total=len(candidates_df), desc="Preparing clips"):
        video_uid = row['video_uid']
        window_idx = row['window_idx']
        timestamp_start = row['timestamp_start']
        timestamp_end = row['timestamp_end']
        anomaly_score = row['anomaly_score']
        
        # Calculate key moments
        duration = timestamp_end - timestamp_start
        timestamps = [
            timestamp_start,
            timestamp_start + duration * 0.5,  # Middle
            timestamp_end
        ]
        
        # Video path (need to check if video is available)
        video_path = Path(args.video_dir) / f"{video_uid}.mp4"
        
        if not video_path.exists():
            print(f"Warning: Video not found: {video_path}")
            # Create entry without frames
            vlm_inputs.append({
                'video_uid': video_uid,
                'window_idx': window_idx,
                'timestamp_start': timestamp_start,
                'timestamp_end': timestamp_end,
                'anomaly_score': anomaly_score,
                'frames': [],
                'status': 'video_not_found'
            })
            continue
        
        # Extract frames
        window_frames_dir = frames_dir / f"{video_uid}_window{window_idx}"
        frame_paths = extract_frames(video_path, timestamps, window_frames_dir)
        
        vlm_inputs.append({
            'video_uid': video_uid,
            'window_idx': window_idx,
            'timestamp_start': timestamp_start,
            'timestamp_end': timestamp_end,
            'anomaly_score': anomaly_score,
            'frames': [str(p) for p in frame_paths],
            'status': 'ready' if frame_paths else 'extraction_failed'
        })
    
    # Save VLM input manifest
    manifest_path = output_dir / 'vlm_input_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(vlm_inputs, f, indent=2)
    
    # Summary
    ready_count = sum(1 for x in vlm_inputs if x['status'] == 'ready')
    
    print(f"\n{'='*50}")
    print(f"VLM Clip Preparation Results:")
    print(f"{'='*50}")
    print(f"Total candidates processed: {len(vlm_inputs)}")
    print(f"Ready for VLM analysis: {ready_count}")
    print(f"Videos not found: {sum(1 for x in vlm_inputs if x['status'] == 'video_not_found')}")
    print(f"Frame extraction failed: {sum(1 for x in vlm_inputs if x['status'] == 'extraction_failed')}")
    print(f"\nSaved manifest to: {manifest_path}")
    print(f"Frames saved to: {frames_dir}")
    print(f"{'='*50}")
    
    if ready_count == 0:
        print("\nWARNING: No clips ready for VLM analysis.")
        print("Videos may need to be downloaded first.")
        print("You can use: ego4d --output_directory data/ego4d_data --datasets full_scale --video_uids ...")

def main():
    parser = argparse.ArgumentParser(description="Prepare video clips for VLM verification")
    parser.add_argument("--candidates-file", type=str,
                        default="data/labels/level2_candidates.csv",
                        help="CSV file with Level 2 candidates")
    parser.add_argument("--video-dir", type=str,
                        default="data/ego4d_data/v2/full_scale",
                        help="Directory containing video files")
    parser.add_argument("--output-dir", type=str,
                        default="data/labels/vlm_input",
                        help="Output directory for frames and manifest")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Process only top K candidates (default: all)")
    args = parser.parse_args()
    
    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ffmpeg not found. Please install ffmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Linux: sudo apt-get install ffmpeg")
        return
    
    prepare_vlm_clips(args)

if __name__ == "__main__":
    main()
