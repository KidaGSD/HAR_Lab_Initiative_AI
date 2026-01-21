import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Analyze Action Durations per Scenario")
    parser.add_argument("--scenario-labels", type=str, default="data/labels/scenario_labels.csv")
    parser.add_argument("--action-labels", type=str, default="data/labels/action_labels_4class.csv")
    args = parser.parse_args()
    
    # 1. Load Data
    print("Loading labels...")
    scenario_df = pd.read_csv(args.scenario_labels)
    action_df = pd.read_csv(args.action_labels)
    
    # Map video_uid to scenario
    video_to_scenario = dict(zip(scenario_df['video_uid'], scenario_df['scenario']))
    
    # Filter to train/val only
    train_val_uids = set(scenario_df[scenario_df['split'].isin(['train', 'val'])]['video_uid'])
    action_df = action_df[action_df['video_uid'].isin(train_val_uids)].copy()
    action_df['scenario'] = action_df['video_uid'].map(video_to_scenario)
    
    print(f"Analyzing {len(action_df)} action segments...")

    # 2. Calculate Durations
    # Assuming action_df has 'start_sec' and 'end_sec' OR consecutive timestamps
    # If the file format is per-second timestamps (as in previous discussions), we need to group consecutive rows.
    
    # Let's inspect columns first to be sure
    # If it is like: video_uid, timestamp_sec, action
    # We need to group consecutive actions to find duration.
    
    # Check if we need to group
    if 'start_sec' not in action_df.columns:
        print("Grouping consecutive timestamps to find durations...")
        # Sort by video and time
        action_df = action_df.sort_values(['video_uid', 'timestamp_sec'])
        
        # Identify change points
        # Compare current row with previous row
        # Group is new if: video changes OR action changes OR time gap > 1.5s (assuming 1s freq)
        
        action_df['prev_uid'] = action_df['video_uid'].shift(1)
        action_df['prev_action'] = action_df['action'].shift(1)
        action_df['prev_time'] = action_df['timestamp_sec'].shift(1)
        
        action_df['new_segment'] = (
            (action_df['video_uid'] != action_df['prev_uid']) | 
            (action_df['action'] != action_df['prev_action']) |
            (action_df['timestamp_sec'] - action_df['prev_time'] > 1.5)
        )
        
        # Assign segment IDs
        action_df['segment_id'] = action_df['new_segment'].cumsum()
        
        # Aggregation
        segments = action_df.groupby(['segment_id', 'video_uid', 'scenario', 'action'])['timestamp_sec'].agg(['min', 'max', 'count']).reset_index()
        segments['duration'] = segments['count'] # duration in seconds (assuming 1Hz labels)
        
        # If labels are 50Hz or other, count might need scaling. But previous scripts implied 1-second labels.
        # Let's assume duration = count * 1.0
        
    else:
        # If start/end exist
        segments = action_df.copy()
        segments['duration'] = segments['end_sec'] - segments['start_sec']

    print(f"Identified {len(segments)} distinct action segments.")
    
    # 3. Compute Statistics (The Matrix)
    # We want Mean and StdDev for each (Scenario, Action)
    stats = segments.groupby(['scenario', 'action'])['duration'].agg(['mean', 'std', 'max', 'count']).round(1)
    
    print("\n=== Duration Statistics (Seconds) ===")
    print(stats)
    
    # 4. Visualization (Boxplots)
    # We will make a grid of boxplots: One plot per Scenario
    scenarios = segments['scenario'].unique()
    num_scenarios = len(scenarios)
    cols = 2
    rows = (num_scenarios + 1) // 2
    
    plt.figure(figsize=(15, 5 * rows))
    
    for i, scen in enumerate(sorted(scenarios)):
        plt.subplot(rows, cols, i+1)
        subset = segments[segments['scenario'] == scen]
        
        # Boxplot
        sns.boxplot(data=subset, x='action', y='duration', showfliers=False) # Hide extreme outliers for readability
        plt.title(f"Action Durations in '{scen}'")
        plt.ylabel("Duration (s)")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plt.savefig('data/duration_distributions.png')
    print("\nPlots saved to data/duration_distributions.png")

if __name__ == "__main__":
    main()





