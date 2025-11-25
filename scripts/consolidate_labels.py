import pandas as pd
import os

# --- Configuration ---
SCENARIO_LABELS_PATH = "data/labels/scenario_labels.csv"
ACTION_LABELS_PATH = "data/labels/action_labels.csv"
OUTPUT_MASTER_PATH = "data/labels/master_annotations.csv"

def main():
    print("Loading labels...")
    
    # 1. Load Scenario Labels (The "Filter")
    if not os.path.exists(SCENARIO_LABELS_PATH):
        print(f"Error: {SCENARIO_LABELS_PATH} not found. Run extract_scenario_labels.py first.")
        return
    
    df_scenarios = pd.read_csv(SCENARIO_LABELS_PATH)
    print(f"Loaded {len(df_scenarios)} videos with scenario labels.")
    
    # 2. Load Action Labels (The "Bulk Data")
    if not os.path.exists(ACTION_LABELS_PATH):
        print(f"Error: {ACTION_LABELS_PATH} not found.")
        return
        
    df_actions = pd.read_csv(ACTION_LABELS_PATH)
    print(f"Loaded {len(df_actions)} action windows.")
    
    # 3. Merge (Inner Join)
    # This keeps only actions that belong to our target videos
    print("Merging and filtering...")
    df_master = pd.merge(
        df_actions, 
        df_scenarios, 
        on="video_uid", 
        how="inner"
    )
    
    print(f"Resulting Master Dataset: {len(df_master)} windows.")
    print(f"Filtered out {len(df_actions) - len(df_master)} windows from non-target videos.")
    
    # 4. Reorder Columns
    # video_uid, timestamp_sec, scenario, action_label, narration_text, split
    cols = ["video_uid", "timestamp_sec", "scenario", "label", "narration_text", "split"]
    # Rename 'label' to 'action' for clarity if preferred, but let's stick to 'label' or 'action_label'
    # The current action_labels.csv has 'label'. Let's rename it to 'action' for clarity in master.
    df_master = df_master.rename(columns={"label": "action"})
    
    final_cols = ["video_uid", "timestamp_sec", "scenario", "action", "narration_text", "split"]
    # Ensure all cols exist
    for c in final_cols:
        if c not in df_master.columns:
            print(f"Warning: Column {c} missing. Available: {df_master.columns}")
            
    df_master = df_master[final_cols]
    
    # 5. Stats
    print("\n--- Master Dataset Statistics ---")
    print(f"Total Videos: {df_master['video_uid'].nunique()}")
    print("\nWindows per Scenario:")
    print(df_master["scenario"].value_counts())
    print("\nWindows per Action:")
    print(df_master["action"].value_counts())
    
    # 6. Save
    os.makedirs(os.path.dirname(OUTPUT_MASTER_PATH), exist_ok=True)
    df_master.to_csv(OUTPUT_MASTER_PATH, index=False)
    print(f"\nSaved master annotations to {OUTPUT_MASTER_PATH}")

if __name__ == "__main__":
    main()
