#!/usr/bin/env python3
"""
Refine Error/Correction labels by applying pattern-based rules.

This script conservatively relabels obvious cases where "drops" indicates
intentional placement rather than an error.

Pattern Rules:
1. "drops X on [surface]" -> Object Transfer (intentional placement)
2. "drops X" (no destination) -> Keep as Error/Correction (likely accident)
3. "adjusts X" -> Object Transfer (deliberate action)
"""

import pandas as pd
import re
from pathlib import Path
import argparse
from datetime import datetime

# Paths
INPUT_CSV = "data/labels/action_labels_llm_clean.csv"
OUTPUT_CSV = "data/labels/action_labels_llm_clean_refined.csv"
REPORT_FILE = "data/labels/refinement_report.txt"

# Pattern definitions for relabeling
PATTERNS = {
    "Object Transfer": [
        # "drops X on [surface]" - intentional placement
        r"drops? .+ (on|onto|in|into|beside|next to|by) (the |a )?(table|counter|floor|plate|bowl|sink|zinc|tray|shelf|rack|cutting board|chopping board|pan|pot)",
        
        # "puts/places X" - already correct, but included for clarity
        r"(puts?|places?) .+ (on|onto|in|into)",
        
        # "adjusts X" - deliberate positioning
        r"adjusts? (the |a )?(bowl|plate|cup|glass|pan|pot|knife|spoon|tool)",
        
        # "sets X down" - intentional placement
        r"sets? .+ down",
        
        # Additional intentional drops with direction
        r"drops? .+ (down|back|aside)",
    ]
}

# Validation patterns - do NOT relabel if these appear (genuine errors)
KEEP_AS_ERROR = [
    r"spills?",
    r"fumbles?",
    r"falls?",
    r"slips?",
    r"accidentally",
    r"drops? .+ glass",  # Dropped glass is often an accident
    r"breaks?",
]


def should_relabel(text: str, current_label: str) -> tuple[bool, str, str]:
    """
    Determine if a label should be changed based on text patterns.
    
    Returns:
        (should_change, new_label, reason)
    """
    if current_label != "Error / Correction":
        return False, current_label, "Not an error label"
    
    text_lower = text.lower()
    
    # First check if it should stay as error
    for pattern in KEEP_AS_ERROR:
        if re.search(pattern, text_lower):
            return False, current_label, f"Matched keep-as-error pattern: {pattern}"
    
    # Check for relabel patterns
    for new_label, patterns in PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True, new_label, f"Matched pattern: {pattern}"
    
    return False, current_label, "No pattern match"


def refine_labels(input_path: str, output_path: str, dry_run: bool = False) -> dict:
    """
    Refine error labels based on patterns.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        dry_run: If True, don't save changes
    
    Returns:
        Statistics dictionary
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Total rows: {len(df):,}")
    print(f"Error/Correction labels: {(df['action'] == 'Error / Correction').sum():,}")
    
    # Track changes
    changes = []
    
    # Process each row
    for idx, row in df.iterrows():
        if row['action'] == "Error / Correction":
            should_change, new_label, reason = should_relabel(
                row['narration_text'], 
                row['action']
            )
            
            if should_change:
                old_label = row['action']
                old_reasoning = row['reasoning']
                
                # Update label
                df.at[idx, 'action'] = new_label
                
                # Update reasoning to explain the change
                df.at[idx, 'reasoning'] = f"[Refined from Error] {reason}. Original: {old_reasoning}"
                
                changes.append({
                    'video_uid': row['video_uid'],
                    'timestamp': row['timestamp_sec'],
                    'narration': row['narration_text'],
                    'old_label': old_label,
                    'new_label': new_label,
                    'reason': reason,
                    'old_reasoning': old_reasoning
                })
    
    # Calculate statistics
    stats = {
        'total_rows': len(df),
        'original_errors': (pd.read_csv(input_path)['action'] == 'Error / Correction').sum(),
        'remaining_errors': (df['action'] == 'Error / Correction').sum(),
        'changes_made': len(changes),
        'new_label_distribution': {}
    }
    
    # Count new labels
    for change in changes:
        new_label = change['new_label']
        stats['new_label_distribution'][new_label] = stats['new_label_distribution'].get(new_label, 0) + 1
    
    # Save results
    if not dry_run:
        print(f"\nSaving refined labels to {output_path}...")
        df.to_csv(output_path, index=False)
        
        # Save change log
        if changes:
            print(f"Saving change report to {REPORT_FILE}...")
            with open(REPORT_FILE, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("LABEL REFINEMENT REPORT\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Total changes: {len(changes)}\n")
                f.write(f"Original Error labels: {stats['original_errors']:,}\n")
                f.write(f"Remaining Error labels: {stats['remaining_errors']:,}\n")
                f.write(f"Corrected: {stats['changes_made']:,} ({stats['changes_made']/stats['original_errors']*100:.1f}%)\n\n")
                
                f.write("New Label Distribution:\n")
                for label, count in stats['new_label_distribution'].items():
                    f.write(f"  {label}: {count:,}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("DETAILED CHANGES\n")
                f.write("=" * 80 + "\n\n")
                
                for i, change in enumerate(changes, 1):
                    f.write(f"{i}. {change['video_uid']} @ {change['timestamp']:.2f}s\n")
                    f.write(f"   Narration: {change['narration']}\n")
                    f.write(f"   {change['old_label']} -> {change['new_label']}\n")
                    f.write(f"   Reason: {change['reason']}\n")
                    f.write(f"   Old reasoning: {change['old_reasoning']}\n")
                    f.write("\n")
    
    return stats, changes


def main():
    parser = argparse.ArgumentParser(
        description="Refine Error/Correction labels using pattern matching"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default=INPUT_CSV,
        help="Input CSV file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=OUTPUT_CSV,
        help="Output CSV file"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be changed without saving"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Run refinement
    stats, changes = refine_labels(args.input, args.output, dry_run=args.dry_run)
    
    # Print summary
    print("\n" + "=" * 60)
    print("REFINEMENT SUMMARY")
    print("=" * 60)
    print(f"Total rows processed: {stats['total_rows']:,}")
    print(f"Original Error labels: {stats['original_errors']:,}")
    print(f"Remaining Error labels: {stats['remaining_errors']:,}")
    print(f"Labels corrected: {stats['changes_made']:,} ({stats['changes_made']/stats['original_errors']*100:.1f}%)")
    
    if stats['new_label_distribution']:
        print("\nCorrected to:")
        for label, count in sorted(stats['new_label_distribution'].items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"  {label}: {count:,}")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN - No files were modified")
        print("\nSample changes (first 5):")
        for change in changes[:5]:
            print(f"\n  {change['narration']}")
            print(f"  {change['old_label']} -> {change['new_label']}")
            print(f"  Reason: {change['reason']}")
    else:
        print(f"\n✅ Saved refined labels to: {args.output}")
        print(f"✅ Saved change report to: {REPORT_FILE}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
