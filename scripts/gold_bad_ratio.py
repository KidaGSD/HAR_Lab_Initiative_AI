#!/usr/bin/env python3
"""
Print gold/(gold+bad) ratio from the CSV "status" column (pre-annotation data).

NOTE: This reads the status column in the CSV, NOT Label Studio annotations.
For Gold/Bad counts from Label Studio (human clicks in the UI), use instead:
    python scripts/labelstudio_gold_bad_ratio.py
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/labels/action_labels_llm_clean_refined.csv")
    parser.add_argument("--include-silver", action="store_true",
                        help="Treat silver as bad: ratio = gold/(gold+silver+bad)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"[CSV status column] {args.csv}")
    counts = df["status"].str.strip().str.lower().value_counts()
    gold = counts.get("gold", 0)
    bad = counts.get("bad", 0)
    silver = counts.get("silver", 0)

    print(f"Gold: {gold:,} | Bad: {bad:,} | Silver: {silver:,}")
    print()

    denom = gold + bad
    if denom > 0:
        ratio = gold / denom
        print(f"gold/(gold+bad) = {gold}/{denom} = {ratio:.4f} ({ratio*100:.2f}%)")
    else:
        print("gold/(gold+bad): no gold or bad labels")

    if args.include_silver:
        denom_all = gold + bad + silver
        if denom_all > 0:
            ratio_all = gold / denom_all
            print(f"gold/(gold+silver+bad) = {gold}/{denom_all} = {ratio_all:.4f} ({ratio_all*100:.2f}%)")


if __name__ == "__main__":
    main()
