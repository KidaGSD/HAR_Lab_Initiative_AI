import pandas as pd
import os
import argparse

# --- Configuration ---
DEFAULT_SCENARIO_PATH = "data/labels/scenario_labels.csv"
DEFAULT_ACTION_PATH = "data/labels/action_labels.csv"
DEFAULT_OUTPUT_PATH = "data/labels/master_annotations.csv"

def main():
    parser = argparse.ArgumentParser(description="Merge scenario and action labels into master annotations.")
    parser.add_argument("--action-labels", default=DEFAULT_ACTION_PATH, help="Path to low-level action labels CSV")
    parser.add_argument("--scenario-labels", default=DEFAULT_SCENARIO_PATH, help="Path to high-level scenario labels CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path to output master CSV")
    args = parser.parse_args()

    print("Loading labels...")
    
    # 1. Load Scenario Labels (The "Filter")
    if not os.path.exists(args.scenario_labels):
        print(f"Error: {args.scenario_labels} not found. Run extract_scenario_labels.py first.")
        return
    
    df_scenarios = pd.read_csv(args.scenario_labels)
    print(f"Loaded {len(df_scenarios)} videos with scenario labels.")
    
    # 2. Load Action Labels (The "Bulk Data")
    if not os.path.exists(args.action_labels):
        print(f"Error: {args.action_labels} not found.")
        return
        
    df_actions = pd.read_csv(args.action_labels)
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
    # Check if 'label' or 'action' column exists
    if 'label' in df_master.columns:
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
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_master.to_csv(args.output, index=False)
    print(f"\nSaved master annotations to {args.output}")

if __name__ == "__main__":
    main()
