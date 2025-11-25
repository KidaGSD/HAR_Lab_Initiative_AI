import json
import pandas as pd
import os
from collections import Counter

# --- Configuration ---
METADATA_PATH = "data/ego4d_meta/ego4d.json"
OUTPUT_SCENARIO_LABELS = "data/labels/scenario_labels.csv"
OUTPUT_TARGET_UIDS = "target_uids.csv"

# Mapping Rules: Keyword -> Scenario Class
# Priority is determined by order if we want to be strict, but here we'll just take the first match.
SCENARIO_MAPPING = {
    "Cooking": ["cooking", "cook", "baking", "kitchen"],
    "Gardening": ["gardening", "garden", "plant", "weeding"],
    "Carpentry": ["carpenter", "construction", "woodworking", "carpentry", "building"],
    "Walking Outdoors": ["walking", "outdoors", "hike", "hiking", "street"],
    "Fitness/Workout": ["fitness", "workout", "gym", "exercise", "yoga", "running"],
    "Cleaning": ["cleaning", "laundry", "washing", "housework", "vacuuming"],
    "Desk Work": ["office", "desk", "computer", "laptop", "study", "working"],
    "Playing Instrument": ["instrument", "music", "piano", "guitar", "drum", "playing"],
    "Mechanical Repair": ["mechanic", "repair", "car", "bike", "fixing", "maintenance"]
}

def normalize_text(text):
    if not text:
        return ""
    return text.lower().strip()

def get_scenario_label(scenarios_list):
    """
    Maps a list of Ego4D scenario strings to one of our 9 target classes.
    Returns the first match found.
    """
    if not scenarios_list:
        return None
    
    # Combine all scenario descriptions into one string for searching
    full_text = " ".join([normalize_text(s) for s in scenarios_list])
    
    for target_class, keywords in SCENARIO_MAPPING.items():
        for keyword in keywords:
            # Simple substring match
            if keyword in full_text:
                return target_class
    
    return None

def main():
    print(f"Loading metadata from {METADATA_PATH}...")
    with open(METADATA_PATH, 'r') as f:
        data = json.load(f)
    
    videos = data.get("videos", [])
    print(f"Total videos in metadata: {len(videos)}")
    
    labeled_data = []
    
    for video in videos:
        uid = video.get("video_uid")
        scenarios = video.get("scenarios", [])
        split = video.get("subset", "unknown") # train/val/test
        
        label = get_scenario_label(scenarios)
        
        if label:
            labeled_data.append({
                "video_uid": uid,
                "scenario": label,
                "split": split,
                "original_scenarios": ";".join(scenarios) if scenarios else ""
            })
            
    # Create DataFrame
    df = pd.DataFrame(labeled_data)
    
    # Filter duplicates if any (keep first)
    df = df.drop_duplicates(subset=["video_uid"])
    
    print(f"Found {len(df)} matching videos.")
    print("\nDistribution:")
    print(df["scenario"].value_counts())
    
    # Save Scenario Labels
    os.makedirs(os.path.dirname(OUTPUT_SCENARIO_LABELS), exist_ok=True)
    df[["video_uid", "scenario", "split"]].to_csv(OUTPUT_SCENARIO_LABELS, index=False)
    print(f"\nSaved scenario labels to {OUTPUT_SCENARIO_LABELS}")
    
    # Save Target UIDs (just the column)
    df[["video_uid"]].to_csv(OUTPUT_TARGET_UIDS, index=False, header=False) # Headerless for easy consumption by some tools, or keep header? 
    # Let's keep header for clarity, but some scripts might expect raw list. 
    # The download script usually takes a CSV with a specific column or just a list.
    # Let's stick to a standard CSV with header 'video_uid'.
    df[["video_uid"]].to_csv(OUTPUT_TARGET_UIDS, index=False)
    print(f"Saved target UIDs to {OUTPUT_TARGET_UIDS}")

if __name__ == "__main__":
    main()
