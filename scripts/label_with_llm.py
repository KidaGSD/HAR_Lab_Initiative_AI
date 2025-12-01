"""
Label narrations using a lightweight LLM instead of keyword matching.
Uses DistilBERT for text classification.
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os

# Model configuration
MODEL_NAME = "distilbert-base-uncased"  # or "albert-base-v2" for smaller model
CLASSES = ['Locomotion', 'Manual Work', 'Scanning', 'Stationary']
CONFIDENCE_THRESHOLD = 0.5  # If confidence below this, label as "Unknown"

# Classification prompt template
PROMPT_TEMPLATE = """Classify the following action description into one of these categories:
- Locomotion: Body movement and displacement (walking, running, moving)
- Manual Work: Hand-object interaction and tool usage (cutting, hammering, mixing)
- Scanning: Visual search and head movement (looking, searching, inspecting)
- Stationary: Passive activities with minimal movement (reading, waiting, sitting)

Action: "{narration_text}"

Category:"""

def create_classifier(model_name=MODEL_NAME):
    """Create a zero-shot classification pipeline."""
    # Option 1: Zero-shot classification (no fine-tuning needed)
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1
    )
    return classifier

def classify_with_llm(classifier, text, classes, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Classify text using LLM. Returns 'Unknown' if confidence is too low."""
    # Zero-shot classification
    result = classifier(text, classes)
    
    # Get the highest confidence label and score
    top_label = result['labels'][0]
    top_confidence = result['scores'][0]
    
    # If confidence is below threshold, return "Unknown"
    if top_confidence < confidence_threshold:
        return 'Unknown', top_confidence
    
    return top_label, top_confidence

def print_statistics(df):
    """Print detailed statistics about label distribution."""
    print("\n" + "="*60)
    print("LABEL STATISTICS")
    print("="*60)
    
    # Count and percentage
    label_counts = df['label'].value_counts()
    label_percentages = df['label'].value_counts(normalize=True) * 100
    
    # Total count
    total = len(df)
    print(f"\nTotal labeled narrations: {total:,}")
    print(f"\nLabel Distribution:")
    print("-" * 60)
    print(f"{'Label':<20} {'Count':<15} {'Percentage':<15}")
    print("-" * 60)
    
    # Print each label with count and percentage
    for label in label_counts.index:
        count = label_counts[label]
        percentage = label_percentages[label]
        print(f"{label:<20} {count:<15,} {percentage:>6.2f}%")
    
    print("-" * 60)
    
    # Confidence statistics
    print(f"\nConfidence Statistics:")
    print(f"  Average Confidence: {df['confidence'].mean():.3f}")
    print(f"  Median Confidence: {df['confidence'].median():.3f}")
    print(f"  Min Confidence: {df['confidence'].min():.3f}")
    print(f"  Max Confidence: {df['confidence'].max():.3f}")
    
    # Confidence by label
    print(f"\nAverage Confidence by Label:")
    print("-" * 60)
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        avg_conf = label_df['confidence'].mean()
        print(f"  {label:<20} {avg_conf:.3f}")
    
    # Unknown label details
    if 'Unknown' in df['label'].values:
        unknown_df = df[df['label'] == 'Unknown']
        print(f"\nUnknown Label Details:")
        print(f"  Count: {len(unknown_df):,}")
        print(f"  Percentage: {len(unknown_df)/total*100:.2f}%")
        print(f"  Average Confidence: {unknown_df['confidence'].mean():.3f}")
        print(f"  (These were below {CONFIDENCE_THRESHOLD:.2f} confidence threshold)")
    
    print("="*60 + "\n")

def main():
    # Configuration
    NARRATIONS_PATH = 'data/ego4d_data/v2/annotations/narration.json'
    TARGET_UIDS_PATH = 'target_uids.csv'
    OUTPUT_PATH = 'data/labels/action_labels_llm.csv'
    
    print(f"Loading model: {MODEL_NAME}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD} (below this = 'Unknown')")
    classifier = create_classifier()
    print("Model loaded!")
    
    # Load target UIDs
    print(f"Loading target UIDs from {TARGET_UIDS_PATH}...")
    uids_df = pd.read_csv(TARGET_UIDS_PATH)
    target_uids = set(uids_df['video_uid'].dropna().tolist())
    print(f"Found {len(target_uids)} target UIDs.")
    
    # Load narrations
    print(f"Loading narrations from {NARRATIONS_PATH}...")
    with open(NARRATIONS_PATH, 'r') as f:
        narrations_data = json.load(f)
    
    # Process narrations
    print("Processing narrations with LLM...")
    labeled_data = []
    
    count_processed = 0
    count_matched = 0
    
    for video_uid, video_data in tqdm(narrations_data.items(), desc="Processing videos"):
        if video_uid not in target_uids:
            continue
            
        count_matched += 1
        
        # Extract narrations
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
            
            # Classify with LLM (includes Unknown if low confidence)
            label, confidence = classify_with_llm(classifier, text, CLASSES, CONFIDENCE_THRESHOLD)
            
            labeled_data.append({
                'video_uid': video_uid,
                'timestamp_sec': timestamp,
                'narration_text': text,
                'label': label,
                'confidence': confidence
            })
            
            count_processed += 1
    
    print(f"\nProcessed {count_matched} videos matching target UIDs.")
    print(f"Generated {len(labeled_data)} labeled timestamps.")
    
    # Save to CSV
    output_df = pd.DataFrame(labeled_data)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved labels to {OUTPUT_PATH}")
    
    # Print detailed statistics
    print_statistics(output_df)

if __name__ == "__main__":
    main()
