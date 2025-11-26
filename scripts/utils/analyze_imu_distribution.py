#!/usr/bin/env python3
"""Analyze distribution of videos with IMU data."""

import pandas as pd

# Load the filtered data
scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
master_df = pd.read_csv("data/labels/master_annotations.csv")

print("=" * 60)
print("DATASET SUMMARY (Videos with IMU Data)")
print("=" * 60)

print(f"\nTotal Videos with IMU: {len(scenario_df)}")
print(f"Total Labeled Windows: {len(master_df)}")
print(f"Videos with Action Labels: {master_df['video_uid'].nunique()}")

print("\n" + "=" * 60)
print("SCENARIO DISTRIBUTION (High-Level Labels)")
print("=" * 60)
scenario_counts = scenario_df['scenario'].value_counts()
print(scenario_counts)

print("\n" + "=" * 60)
print("SPLIT DISTRIBUTION")
print("=" * 60)
split_counts = scenario_df['split'].value_counts()
print(split_counts)

print("\n" + "=" * 60)
print("SCENARIOS BY SPLIT")
print("=" * 60)
split_scenario = pd.crosstab(scenario_df['scenario'], scenario_df['split'])
print(split_scenario)

print("\n" + "=" * 60)
print("ACTION DISTRIBUTION (Low-Level Labels)")
print("=" * 60)
if len(master_df) > 0:
    action_counts = master_df['action'].value_counts()
    print(action_counts)
    print(f"\nTotal action-labeled windows: {len(master_df)}")
    
    print("\n" + "=" * 60)
    print("WINDOWS PER SCENARIO (for videos with action labels)")
    print("=" * 60)
    scenario_windows = master_df['scenario'].value_counts()
    print(scenario_windows)
else:
    print("No action labels available")

# Save summary
with open("data/labels/dataset_summary.txt", "w") as f:
    f.write("FINAL DATASET SUMMARY (With IMU Data)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total Videos: 1,652\n")
    f.write(f"Videos with Action Labels: {master_df['video_uid'].nunique()}\n")
    f.write(f"Total Labeled Windows: {len(master_df)}\n\n")
    f.write("Scenario Counts:\n")
    f.write(scenario_counts.to_string())

print("\n✓ Summary saved to data/labels/dataset_summary.txt")
