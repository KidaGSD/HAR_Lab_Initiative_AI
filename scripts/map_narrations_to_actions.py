import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
NARRATIONS_PATH = 'data/ego4d_data/v2/annotations/narration.json'
TARGET_UIDS_PATH = 'target_uids.csv'
OUTPUT_PATH = 'data/labels/action_labels.csv'

# --- 4-Class Taxonomy Keywords (Refined for Research Design) ---
# Priority: Locomotion > Manual Work > Scanning > Stationary
# 1. Locomotion: Global displacement.
# 2. Manual Work: Tool usage, hand-object interaction (implies vibration/jerk).
# 3. Scanning: Visual search, head rotation (implies gyro signal).
# 4. Stationary: Passive, low motion.

# 10 Scenarios
HIGHLEVL: ["cooking", "carpenter", "maintenance"]

KEYWORDS = {
    'Locomotion': [
        'walk', 'run', 'step', 'move to', 'enter', 'leave', 'climb', 
        'stand up', 'sit down', 'go to', 'approach', 'descend', 'ascend',
        'navigate', 'journey'
    ],
    'Manual Work': [
        'cut', 'saw', 'hammer', 'drill', 'sand', 'wipe', 'scrub', 'wash', 
        'mix', 'stir', 'pour', 'shake', 'rub', 'dig', 'push', 'pull', 
        'lift', 'throw', 'catch', 'carry', 'hold', 'grab', 'take', 'put',
        'open', 'close', 'turn', 'twist', 'screw', 'unscrew', 'adjust',
        'fix', 'repair', 'assemble', 'disassemble', 'install', 'remove',
        'tighten', 'loosen', 'wrench', 'solder', 'weld', 'paint', 'glue'
    ],
    'Scanning': [
        'look', 'search', 'check', 'inspect', 'find', 'scan', 'glance', 
        'stare', 'gaze', 'observe', 'watch', 'examine', 'seek', 'locate',
        'survey', 'view', 'spot'
    ],
    'Stationary': [
        'read', 'write', 'measure', 'think', 'pause', 'touch', 'point', 
        'gesture', 'talk', 'speak', 'listen', 'wait', 'rest', 'sleep',
        'sit', 'stand', 'kneel', 'crouch' # Postures without movement
    ]
}

def map_text_to_label(text):
    text = text.lower()
    
    # 1. Locomotion (Highest priority - body movement)
    if any(k in text for k in KEYWORDS['Locomotion']):
        return 'Locomotion'
        
    # 2. Manual Work (Tool usage / Hand interaction)
    if any(k in text for k in KEYWORDS['Manual Work']):
        return 'Manual Work'
        
    # 3. Scanning (Head movement focus)
    if any(k in text for k in KEYWORDS['Scanning']):
        return 'Scanning'
        
    # 4. Stationary (Default/Fallback)
    return 'Stationary'

def main():
    # 1. Load Target UIDs
    print(f"Loading target UIDs from {TARGET_UIDS_PATH}...")
    try:
        uids_df = pd.read_csv(TARGET_UIDS_PATH)
        target_uids = set(uids_df['video_uid'].dropna().tolist())
        print(f"Found {len(target_uids)} target UIDs.")
    except Exception as e:
        print(f"Error loading target UIDs: {e}")
        return

    # 2. Load Narrations
    print(f"Loading narrations from {NARRATIONS_PATH}...")
    try:
        with open(NARRATIONS_PATH, 'r') as f:
            narrations_data = json.load(f)
    except Exception as e:
        print(f"Error loading narrations: {e}")
        return

    # 3. Process Narrations
    print("Processing narrations...")
    labeled_data = []
    
    # narrations_data structure: keys are video_uids
    # But wait, Ego4D structure might be different. Let's handle both dict and list.
    # Usually it's {"key": "value"} where key is UID.
    
    count_processed = 0
    count_matched = 0
    
    for video_uid, video_data in tqdm(narrations_data.items()):
        if video_uid not in target_uids:
            continue
            
        count_matched += 1
        
        # Extract narrations from pass 1 (preferred) or pass 2
        narrations = []
        if isinstance(video_data, dict):
            if 'narration_pass_1' in video_data and 'narrations' in video_data['narration_pass_1']:
                narrations = video_data['narration_pass_1']['narrations']
            elif 'narration_pass_2' in video_data and 'narrations' in video_data['narration_pass_2']:
                narrations = video_data['narration_pass_2']['narrations']
        
        if not narrations:
            continue
        
        for entry in narrations:
            text = entry.get('narration_text') or entry.get('text')
            timestamp = entry.get('timestamp_sec') or entry.get('timestamp')
            
            if text is None or timestamp is None:
                continue
                
            label = map_text_to_label(text)
            
            labeled_data.append({
                'video_uid': video_uid,
                'timestamp_sec': timestamp,
                'narration_text': text,
                'label': label
            })
            
    print(f"Processed {count_matched} videos matching target UIDs.")
    print(f"Generated {len(labeled_data)} labeled timestamps.")
    
    # 4. Save to CSV
    output_df = pd.DataFrame(labeled_data)
    
    # Create output directory if needed
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved labels to {OUTPUT_PATH}")
    
    # 5. Visualize Distribution
    if not output_df.empty:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=output_df, x='label', order=['Stationary', 'Scanning', 'Locomotion', 'Manual Work'])
        plt.title('Distribution of Fundamental Action Labels')
        plt.xlabel('Action Class')
        plt.ylabel('Count')
        plt.savefig('data/labels/action_distribution.png')
        print("Saved distribution plot to data/labels/action_distribution.png")
        
        # Print stats
        print("\nLabel Distribution:")
        print(output_df['label'].value_counts(normalize=True))

if __name__ == "__main__":
    main()
