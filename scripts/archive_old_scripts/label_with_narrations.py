#!/usr/bin/env python3
"""
Level 1 Labeling: Label Active/Normal moments using narrations.

Uses LLM to analyze narration text and map to IMU windows.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def analyze_narration_activity(narration_text):
    """
    Simple keyword-based activity classification.
    
    Returns:
        bool: True if narration implies active motion, False otherwise
    """
    # Keywords that imply active, coordinated motion
    active_keywords = [
        'cut', 'hammer', 'drill', 'saw', 'sand', 'paint', 'nail', 'screw',
        'walk', 'carry', 'lift', 'move', 'place', 'assemble', 'install',
        'measure', 'mark', 'clean', 'sweep', 'wipe', 'cook', 'chop', 'stir',
        'mix', 'pour', 'grab', 'take', 'put', 'open', 'close', 'turn', 'rotate'
    ]
    
    # Keywords that imply passive or stationary behavior
    passive_keywords = [
        'look', 'watch', 'observe', 'think', 'wait', 'pause', 'rest',
        'stand', 'sit', 'read', 'check'
    ]
    
    text_lower = narration_text.lower()
    
    # Check for active keywords
    active_count = sum(1 for kw in active_keywords if kw in text_lower)
    passive_count = sum(1 for kw in passive_keywords if kw in text_lower)
    
    # If more active keywords, classify as active
    if active_count > passive_count:
        return True
    elif passive_count > active_count:
        return False
    else:
        # Default to active if unclear
        return True

def label_with_narrations(args):
    """
    Main function to label Level 1 (Active/Normal) moments using narrations.
    """
    # Load narrations
    narrations_path = Path(args.narrations_file)
    if not narrations_path.exists():
        print(f"ERROR: Narrations file not found: {narrations_path}")
        print("Run: ego4d --output_directory data/ego4d_data --datasets annotations --yes")
        return
        
    with open(narrations_path) as f:
        narrations_data = json.load(f)
    
    # Load target UIDs
    target_df = pd.read_csv(args.target_uids_file)
    target_uids = set(target_df['video_uid'].tolist())
    
    print(f"Processing {len(target_uids)} videos...")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    level1_labels = []
    
    # Process each video
    for video in tqdm(narrations_data.get('videos', []), desc="Labeling"):
        video_uid = video.get('video_uid')
        
        if video_uid not in target_uids:
            continue
            
        # Get narrations for this video
        narration_clips = video.get('clips', [])
        
        for clip in narration_clips:
            narrations = clip.get('narrations', [])
            
            for narration in narrations:
                narration_text = narration.get('narration_text', '')
                timestamp = narration.get('timestamp_sec', 0)
                
                # Analyze if this narration implies active motion
                is_active = analyze_narration_activity(narration_text)
                
                if is_active:
                    level1_labels.append({
                        'video_uid': video_uid,
                        'timestamp': timestamp,
                        'narration': narration_text,
                        'label': 'Level_1_Active'
                    })
    
    # Save labels
    labels_df = pd.DataFrame(level1_labels)
    output_file = output_dir / 'level1_labels.csv'
    labels_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*50}")
    print(f"Level 1 Labeling Results:")
    print(f"{'='*50}")
    print(f"Total Active moments labeled: {len(labels_df)}")
    print(f"Unique videos labeled: {labels_df['video_uid'].nunique()}")
    print(f"Saved labels to: {output_file}")
    print(f"{'='*50}")
    
    # Show sample
    if len(labels_df) > 0:
        print("\nSample labels:")
        print(labels_df.head(10).to_string())

def main():
    parser = argparse.ArgumentParser(description="Level 1 Labeling using Narrations")
    parser.add_argument("--narrations-file", type=str, 
                        default="data/ego4d_data/v2/annotations/narration.json",
                        help="Path to narrations.json")
    parser.add_argument("--target-uids-file", type=str, 
                        default="target_uids.csv",
                        help="CSV file with target video UIDs")
    parser.add_argument("--output-dir", type=str, 
                        default="data/labels",
                        help="Output directory for labels")
    args = parser.parse_args()
    
    label_with_narrations(args)

if __name__ == "__main__":
    main()
