import json
import pandas as pd
from tqdm import tqdm
from collections import Counter

# --- Configuration ---
NARRATIONS_PATH = 'data/ego4d_data/v2/annotations/narration.json'
TARGET_UIDS_PATH = 'target_uids.csv' # Use the filtered one

# --- Same Keywords as map_narrations_to_actions.py ---
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

def analyze_text(text):
    text = text.lower()
    matches = []
    
    for category, keywords in KEYWORDS.items():
        matched_kws = [k for k in keywords if k in text]
        if matched_kws:
            matches.append((category, matched_kws))
            
    return matches

def main():
    # Load Target UIDs
    print("Loading target UIDs...")
    uids_df = pd.read_csv(TARGET_UIDS_PATH)
    target_uids = set(uids_df['video_uid'].tolist())
    
    # Load Narrations
    print("Loading narrations...")
    with open(NARRATIONS_PATH, 'r') as f:
        data = json.load(f)
        
    print(f"Analyzing narrations for {len(target_uids)} videos...")
    
    stats = {
        'total_narrations': 0,
        'explicit_matches': 0,
        'fallback_stationary': 0,
        'multi_label': 0,
        'category_counts': Counter(),
        'keyword_counts': Counter(),
        'overlap_counts': Counter() # e.g. Locomotion + Manual Work
    }
    
    for uid, video_data in tqdm(data.items()):
        if uid not in target_uids:
            continue
            
        # Access the nested list
        if 'narration_pass_1' in video_data and 'narrations' in video_data['narration_pass_1']:
            narrations_list = video_data['narration_pass_1']['narrations']
        elif 'narration_pass_2' in video_data and 'narrations' in video_data['narration_pass_2']:
            narrations_list = video_data['narration_pass_2']['narrations']
        else:
            continue
            
        for narr in narrations_list:
            text = narr['narration_text']
            stats['total_narrations'] += 1
            
            matches = analyze_text(text)
            
            if not matches:
                stats['fallback_stationary'] += 1
                stats['category_counts']['Fallback (Stationary)'] += 1
            else:
                stats['explicit_matches'] += 1
                
                # Count categories
                categories = [m[0] for m in matches]
                
                # Current Logic Simulation (Priority)
                if 'Locomotion' in categories:
                    final_label = 'Locomotion'
                elif 'Manual Work' in categories:
                    final_label = 'Manual Work'
                elif 'Scanning' in categories:
                    final_label = 'Scanning'
                elif 'Stationary' in categories:
                    final_label = 'Stationary'
                else:
                    final_label = 'Error'
                    
                stats['category_counts'][final_label] += 1
                
                # Multi-label analysis
                if len(categories) > 1:
                    stats['multi_label'] += 1
                    key = " + ".join(sorted(categories))
                    stats['overlap_counts'][key] += 1
                    
                # Keyword analysis (for the chosen label)
                # Find which keyword triggered the label
                for cat, kws in matches:
                    if cat == final_label:
                        for kw in kws:
                            stats['keyword_counts'][f"{cat}:{kw}"] += 1

    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print(f"Total Narrations: {stats['total_narrations']}")
    print(f"Explicit Matches: {stats['explicit_matches']} ({stats['explicit_matches']/stats['total_narrations']*100:.1f}%)")
    print(f"Fallback to Stationary: {stats['fallback_stationary']} ({stats['fallback_stationary']/stats['total_narrations']*100:.1f}%)")
    print(f"Multi-Label Candidates: {stats['multi_label']} ({stats['multi_label']/stats['total_narrations']*100:.1f}%)")
    
    print("\n--- Current Label Distribution (Simulated) ---")
    for cat, count in stats['category_counts'].most_common():
        print(f"{cat}: {count} ({count/stats['total_narrations']*100:.1f}%)")
        
    print("\n--- Overlaps (Concurrent Actions) ---")
    for combo, count in stats['overlap_counts'].most_common(5):
        print(f"{combo}: {count}")
        
    print("\n--- Top Keywords Triggering Labels ---")
    for k, v in stats['keyword_counts'].most_common(20):
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
