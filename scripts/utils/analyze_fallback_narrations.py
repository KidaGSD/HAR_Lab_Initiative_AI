import json
import pandas as pd
import random
from tqdm import tqdm

# --- Configuration ---
NARRATIONS_PATH = 'data/ego4d_data/v2/annotations/narration.json'
TARGET_UIDS_PATH = 'target_uids.csv'

# --- Current Keywords (to exclude) ---
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
        'sit', 'stand', 'kneel', 'crouch'
    ]
}

def is_matched(text):
    text = text.lower()
    for cat, kws in KEYWORDS.items():
        if any(k in text for k in kws):
            return True
    return False

def main():
    # Load Target UIDs
    uids_df = pd.read_csv(TARGET_UIDS_PATH)
    target_uids = set(uids_df['video_uid'].tolist())
    
    # Load Narrations
    with open(NARRATIONS_PATH, 'r') as f:
        data = json.load(f)
        
    unmatched_samples = []
    
    for uid, video_data in tqdm(data.items()):
        if uid not in target_uids:
            continue
            
        # Access nested list
        if 'narration_pass_1' in video_data and 'narrations' in video_data['narration_pass_1']:
            narrations_list = video_data['narration_pass_1']['narrations']
        elif 'narration_pass_2' in video_data and 'narrations' in video_data['narration_pass_2']:
            narrations_list = video_data['narration_pass_2']['narrations']
        else:
            continue
            
        for narr in narrations_list:
            text = narr['narration_text']
            if not is_matched(text):
                unmatched_samples.append(text)

    print(f"Total Unmatched: {len(unmatched_samples)}")
    
    # Sample 100
    if len(unmatched_samples) > 100:
        samples = random.sample(unmatched_samples, 100)
    else:
        samples = unmatched_samples
        
    print("\n--- Sample Unmatched Narrations ---")
    for s in samples:
        print(f"- {s}")
        
    # Save to file for review
    with open("unmatched_samples.txt", "w") as f:
        for s in samples:
            f.write(f"{s}\n")

if __name__ == "__main__":
    main()
