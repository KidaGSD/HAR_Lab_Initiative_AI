#!/usr/bin/env python3
"""
Analyze narration duplication levels:
1. Raw
2. No Articles (regex based)
3. Nouns & Verbs only (NLP based) - Optimized with nlp.pipe
"""

import pandas as pd
import re
import sys
import time
from collections import Counter

# Try importing spacy
try:
    import spacy
except ImportError:
    print("Error: spacy not installed. Please run: pip install spacy")
    sys.exit(1)

# Load Spacy model
MODEL_NAME = "en_core_web_sm"
try:
    if not spacy.util.is_package(MODEL_NAME):
        print(f"Spacy model '{MODEL_NAME}' not found. Downloading...")
        from spacy.cli import download
        download(MODEL_NAME)
    nlp = spacy.load(MODEL_NAME, disable=["parser", "ner"]) # Disable parser/ner for speed
except Exception as e:
    print(f"Error loading spacy model: {e}")
    sys.exit(1)

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
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    total_rows = len(df)
    print(f"Total Rows: {total_rows:,}")

    # 1. Raw Analysis
    unique_raw = df['narration_text'].nunique()
    print(f"\n[1] Raw Unique Narrations: {unique_raw:,} (Reduction: 0%)")

    # 2. Article Removal Analysis
    print("\nProcessing Level 2: Removing articles & punctuation...")
    t0 = time.time()
    df['clean_articles'] = df['narration_text'].apply(clean_articles)
    unique_articles = df['clean_articles'].nunique()
    red_articles = (1 - unique_articles / unique_raw) * 100
    print(f"[2] No Articles Unique:    {unique_articles:,} (Reduction from Raw: {red_articles:.1f}%)")
    print(f"    Time: {time.time()-t0:.2f}s")

    # 3. Nouns/Verbs Analysis
    print("\nProcessing Level 3: NLP Extraction (Nouns & Verbs only)...")
    print("    (This uses spacy nlp.pipe for speed, might take ~1-2 mins)")
    t0 = time.time()
    
    # Pre-clean for NLP (remove hashtags)
    # Handle NaN/float by converting to string first, or using empty string for NaNs
    texts = df['narration_text'].fillna("").astype(str).apply(lambda x: re.sub(r'#\w+', '', x)).tolist()
    
    clean_nlp_list = []
    # Process in batches
    for doc in nlp.pipe(texts, batch_size=2000, n_process=1):
        # Keep Nouns, Proper Nouns, Verbs
        tokens = [token.lemma_.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN', 'VERB']]
        clean_nlp_list.append(" ".join(tokens))
    
    df['clean_nlp'] = clean_nlp_list
    unique_nlp = df['clean_nlp'].nunique()
    red_nlp = (1 - unique_nlp / unique_raw) * 100
    print(f"[3] Nouns/Verbs Unique:    {unique_nlp:,} (Reduction from Raw: {red_nlp:.1f}%)")
    print(f"    Time: {time.time()-t0:.2f}s")

    print("\n" + "="*60)
    print("TOP MERGES (Examples)")
    print("="*60)

    # Show examples where Level 2 merged things
    counts = df.groupby('clean_articles')['narration_text'].nunique()
    merged_articles = counts[counts > 1].sort_values(ascending=False).head(5)
    
    print("\n--- Top Merges by Removing Articles ---")
    for clean_text, count in merged_articles.items():
        examples = df[df['clean_articles'] == clean_text]['narration_text'].unique()[:3]
        print(f"\nResult: '{clean_text}' (Merges {count} variations)")
        print(f"  Sources: {list(examples)}")

    # Show examples where Level 3 merged things
    counts_nlp = df.groupby('clean_nlp')['narration_text'].nunique()
    merged_nlp = counts_nlp[counts_nlp > 1].sort_values(ascending=False).head(5)

    print("\n--- Top Merges by Keeping Only Nouns/Verbs ---")
    for clean_text, count in merged_nlp.items():
        examples = df[df['clean_nlp'] == clean_text]['narration_text'].unique()[:3]
        print(f"\nResult: '{clean_text}' (Merges {count} variations)")
        print(f"  Sources: {list(examples)}")

if __name__ == "__main__":
    main()
