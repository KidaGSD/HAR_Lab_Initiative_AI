#!/usr/bin/env python3
"""
Create a deduplicated dataset based on "No Articles" normalization.
Keeps the first original row for each unique normalized narration.
"""

import pandas as pd
import re
import os

def clean_articles(text):
    """
    1. Lowercase
    2. Remove #tags (like #C, #unsure)
    3. Remove punctuation
    4. Remove articles (the, a, an, some)
    5. Strip whitespace
    """
    if not isinstance(text, str): return ""
    t = text.lower()
    t = re.sub(r'#\w+', '', t) # Remove hashtags
    t = re.sub(r'[^\w\s]', '', t) # Remove punctuation
    t = re.sub(r'\b(the|a|an|some)\b', '', t) # Remove specific articles
    return re.sub(r'\s+', ' ', t).strip()

def main():
    input_csv = "data/labels/action_labels_llm_clean_refined.csv"
    output_csv = "data/labels/action_labels_llm_clean_refined_no_articles.csv"
    
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    original_count = len(df)
    
    print("Normalizing narrations...")
    # Create a temporary column for deduplication
    df['temp_normalized'] = df['narration_text'].apply(clean_articles)
    
    print("Deduplicating...")
    # Drop duplicates based on the normalized text, keeping the first occurrence
    # This preserves the original 'narration_text' of the survivor
    df_deduped = df.drop_duplicates(subset=['temp_normalized'], keep='first')
    
    # Remove the temp column
    df_deduped = df_deduped.drop(columns=['temp_normalized'])
    
    final_count = len(df_deduped)
    reduction = original_count - final_count
    
    print(f"Original rows: {original_count:,}")
    print(f"Final rows:    {final_count:,}")
    print(f"Removed:       {reduction:,} ({(reduction/original_count)*100:.1f}%)")
    
    print(f"Saving to {output_csv}...")
    df_deduped.to_csv(output_csv, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
