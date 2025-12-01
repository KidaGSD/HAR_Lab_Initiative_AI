"""
Inspect what the LLM is seeing and why it's giving low confidence.
Compare DistilBERT vs heavier model.
"""

import json
import pandas as pd
from pathlib import Path
import torch
from transformers import pipeline

torch.cuda.empty_cache()
torch.cuda.synchronize()

# Model configurations
MODELS = {
    'distilbert': {
        'name': "distilbert-base-uncased",
        'description': 'DistilBERT (66M params)'
    },
    'qwen': {
        'name': "Qwen/Qwen2-0.5B-Instruct",  # Or "Qwen/Qwen2-0.5B" for non-instruct
        'description': 'Qwen2-0.5B (500M params)'
    }
}

CLASSES = ['Locomotion', 'Manual Work', 'Scanning', 'Stationary']
CONFIDENCE_THRESHOLD = 0.5

# Load classifiers
print("Loading models...")
classifiers = {}
for model_key, model_info in MODELS.items():
    try:
        print(f"  Loading {model_info['description']}...")
        
        # Clear cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        classifiers[model_key] = pipeline(
            "zero-shot-classification",
            model=model_info['name'],
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Clear cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"  ✓ {model_info['description']} loaded!")
    except Exception as e:
        print(f"  ✗ Failed to load {model_info['description']}: {e}")
        # Clear cache on error too
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"    Trying alternative...")
        # Try non-instruct version
        if 'Instruct' in model_info['name']:
            try:
                alt_name = model_info['name'].replace('-Instruct', '')
                classifiers[model_key] = pipeline(
                    "zero-shot-classification",
                    model=alt_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                print(f"  ✓ Loaded {alt_name} instead!")
            except:
                pass

if not classifiers:
    print("ERROR: No models loaded!")
    exit(1)

print(f"\nLoaded {len(classifiers)} model(s)\n")

# Load narrations
NARRATIONS_PATH = 'data/ego4d_data/v2/annotations/narration.json'
TARGET_UIDS_PATH = 'target_uids.csv'

print(f"Loading narrations from {NARRATIONS_PATH}...")
with open(NARRATIONS_PATH, 'r') as f:
    narrations_data = json.load(f)

uids_df = pd.read_csv(TARGET_UIDS_PATH)
target_uids = set(uids_df['video_uid'].dropna().tolist())

print(f"\n{'='*100}")
print("COMPARING MODELS ON FIRST 20 NARRATIONS")
print(f"{'='*100}\n")

# Collect first 20 narrations
sample_narrations = []
count = 0
for video_uid, video_data in narrations_data.items():
    if video_uid not in target_uids:
        continue
    
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
        
        sample_narrations.append({
            'video_uid': video_uid,
            'timestamp': timestamp,
            'text': text
        })
        
        count += 1
        if count >= 20:
            break
    
    if count >= 20:
        break

# Classify with all models
results_by_model = {}
for model_key, classifier in classifiers.items():
    results_by_model[model_key] = []
    for sample in sample_narrations:
        result = classifier(sample['text'], CLASSES)
        results_by_model[model_key].append(result)

# Print comparison
for idx, sample in enumerate(sample_narrations, 1):
    print(f"{'='*100}")
    print(f"NARRATION #{idx}")
    print(f"{'='*100}")
    print(f"Video UID: {sample['video_uid']}")
    print(f"Timestamp: {sample['timestamp']:.2f}s")
    print(f"Text: '{sample['text']}'")
    print(f"Text length: {len(sample['text'])} chars\n")
    
    for model_key in classifiers.keys():
        result = results_by_model[model_key][idx-1]
        model_desc = MODELS[model_key]['description']
        
        print(f"--- {model_desc} ---")
        print(f"  {'Label':<20} {'Score':<12} {'%':<10}")
        print(f"  {'-'*42}")
        for label, score in zip(result['labels'], result['scores']):
            marker = " ← TOP" if label == result['labels'][0] else ""
            print(f"  {label:<20} {score:<12.4f} {score*100:>6.2f}%{marker}")
        
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        print(f"\n  → Predicted: {top_label} (confidence: {top_score:.4f})")
        if top_score < CONFIDENCE_THRESHOLD:
            print(f"  → Final Label: Unknown (below {CONFIDENCE_THRESHOLD} threshold)")
        else:
            print(f"  → Final Label: {top_label}")
        
        # Show confidence spread
        score_range = max(result['scores']) - min(result['scores'])
        print(f"  → Score spread: {score_range:.4f} (higher = more confident)")
        print()
    
    print()

# Summary statistics
print(f"\n{'='*100}")
print("SUMMARY STATISTICS")
print(f"{'='*100}\n")

for model_key in classifiers.keys():
    model_desc = MODELS[model_key]['description']
    results = results_by_model[model_key]
    
    top_confidences = [r['scores'][0] for r in results]
    avg_confidence = sum(top_confidences) / len(top_confidences)
    max_confidence = max(top_confidences)
    min_confidence = min(top_confidences)
    
    labels_predicted = [r['labels'][0] for r in results]
    label_counts = {}
    for label in labels_predicted:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    above_threshold = sum(1 for c in top_confidences if c >= CONFIDENCE_THRESHOLD)
    
    print(f"{model_desc}:")
    print(f"  Average top confidence: {avg_confidence:.4f}")
    print(f"  Min confidence: {min_confidence:.4f}")
    print(f"  Max confidence: {max_confidence:.4f}")
    print(f"  Above threshold ({CONFIDENCE_THRESHOLD}): {above_threshold}/20 ({above_threshold/20*100:.1f}%)")
    print(f"  Label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {label}: {count}/20 ({count/20*100:.1f}%)")
    print()
