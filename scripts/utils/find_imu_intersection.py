#!/usr/bin/env python3
"""Find intersection between target UIDs and available IMU data."""

import pandas as pd

# Load target UIDs
target_df = pd.read_csv("target_uids.csv")
target_uids = set(target_df['video_uid'].tolist())
print(f"Target UIDs: {len(target_uids)}")

# Load IMU manifest
imu_manifest = pd.read_csv("data/ego4d_data/v2/imu/manifest.csv")
available_uids = set(imu_manifest['video_uid'].tolist())
print(f"Available IMU UIDs: {len(available_uids)}")

# Find intersection
intersection = target_uids & available_uids
print(f"Intersection (videos with both scenario labels AND IMU): {len(intersection)}")

# Save the intersection
if len(intersection) > 0:
    intersection_df = pd.DataFrame({'video_uid': list(intersection)})
    intersection_df.to_csv("target_uids_with_imu.csv", index=False)
    print(f"Saved {len(intersection)} UIDs to target_uids_with_imu.csv")
    
    # Also update scenario and master labels to match
    scenario_df = pd.read_csv("data/labels/scenario_labels.csv")
    scenario_filtered = scenario_df[scenario_df['video_uid'].isin(intersection)]
    scenario_filtered.to_csv("data/labels/scenario_labels_with_imu.csv", index=False)
    print(f"Saved filtered scenario labels: {len(scenario_filtered)} videos")
else:
    print("ERROR: No overlap! Check if you're using the right manifest.")
