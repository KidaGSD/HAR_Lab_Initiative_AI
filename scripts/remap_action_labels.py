#!/usr/bin/env python3
"""
Remap 6-class semantic actions to 4-class motion-based actions.

Input: action_labels_llm_validated.csv (6 classes)
Output: action_labels_4class.csv (4 classes)

Mapping:
  Stationary          -> Stationary (13.0%)
  Locomotion          -> Locomotion (9.9%)
  Object Transfer     -> Manipulation (42.4%)
  Essential Operation -> Manipulation (27.9%)
  Search              -> Search/Interrupt (6.1%)
  Error / Correction  -> Search/Interrupt (0.7%)
  Unknown/Uncertain   -> DROP
"""

import pandas as pd
from pathlib import Path

# Define mapping
MOTION_MAP = {
    'Stationary': 'Stationary',
    'Locomotion': 'Locomotion',
    'Object Transfer': 'Manipulation',
    'Essential Operation': 'Manipulation',
    'Search': 'Search_Interrupt',  # Use underscore for CSV compatibility
    'Error / Correction': 'Search_Interrupt',
}

# Class indices for model training
CLASS_TO_IDX = {
    'Stationary': 0,
    'Locomotion': 1,
    'Manipulation': 2,
    'Search_Interrupt': 3,
}

def main():
    # Paths
    input_path = Path('data/labels/action_labels_llm_validated.csv')
    output_path = Path('data/labels/action_labels_4class.csv')
    
    print(f"Reading from: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"\n=== BEFORE (6 classes) ===")
    print(df['action'].value_counts())
    print(f"Total: {len(df)}")
    
    # Apply mapping
    df['action_original'] = df['action']  # Keep original for reference
    df['action'] = df['action'].map(MOTION_MAP)
    
    # Drop unmapped rows (Unknown, Uncertain)
    n_before = len(df)
    df = df.dropna(subset=['action'])
    n_after = len(df)
    print(f"\nDropped {n_before - n_after} rows with Unknown/Uncertain labels")
    
    # Add class index column
    df['action_idx'] = df['action'].map(CLASS_TO_IDX)
    
    print(f"\n=== AFTER (4 classes) ===")
    print(df['action'].value_counts())
    print(f"Total: {len(df)}")
    
    # Show percentages
    print(f"\n=== PERCENTAGES ===")
    for action, count in df['action'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {action:20s}: {count:7d} ({pct:5.2f}%)")
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    # Also save a mapping file for reference
    mapping_path = Path('data/labels/action_mapping_4class.txt')
    with open(mapping_path, 'w') as f:
        f.write("# 4-Class Motion-Based Action Mapping\n\n")
        f.write("## Class Indices\n")
        for cls, idx in CLASS_TO_IDX.items():
            f.write(f"  {idx}: {cls}\n")
        f.write("\n## Original -> New Mapping\n")
        for orig, new in MOTION_MAP.items():
            f.write(f"  {orig} -> {new}\n")
    print(f"Saved mapping to: {mapping_path}")

if __name__ == '__main__':
    main()
