#!/usr/bin/env python3
"""
Check if validated narrations still appear in the deduped CSV.
Performs exact-match and normalized-match analysis.
"""

import pandas as pd
import re
from collections import Counter

# -- Paths ---------------------------------------------------------------
DEDUPED = (
    "/Users/huangjunda/Desktop/MIT 2.156/HAR/HAR_Lab_Initiative_AI"
    "/data/annotation_rounds/r001_deduped/round1_deduped.csv"
)
VALIDATED = (
    "/Users/huangjunda/Desktop/MIT 2.156/HAR/HAR_Lab_Initiative_AI"
    "/data/1000_validated/project-7-at-2026-02-22-01-41-8ea80ad7.csv"
)

# -- Load -----------------------------------------------------------------
deduped = pd.read_csv(DEDUPED)
validated = pd.read_csv(VALIDATED)

print("=" * 80)
print("FILE SUMMARY")
print("=" * 80)
print(f"Deduped rows:    {len(deduped):,}")
print(f"Validated rows:  {len(validated):,}")
print(f"\nDeduped columns:    {list(deduped.columns)}")
print(f"Validated columns:  {list(validated.columns)}")

# -- 1. Exact row matches ------------------------------------------------
print("\n" + "=" * 80)
print("1. EXACT ROW MATCHES  (video_uid, narration_text, batch)")
print("=" * 80)

deduped_exact_set = set(
    zip(
        deduped["video_uid"].astype(str),
        deduped["narration_text"].astype(str),
        deduped["batch"].astype(str),
    )
)

validated["_vid"] = validated["video_uid"].astype(str)
validated["_nar"] = validated["narration_text"].astype(str)
validated["_bat"] = validated["batch"].astype(str)

validated["exact_match"] = [
    (v, n, b) in deduped_exact_set
    for v, n, b in zip(validated["_vid"], validated["_nar"], validated["_bat"])
]

n_exact = validated["exact_match"].sum()
print(f"Validated rows with EXACT match in deduped: {n_exact} / {len(validated)}")
print(f"Validated rows WITHOUT exact match:         {len(validated) - n_exact}")

if n_exact > 0:
    print("\nSample exact matches (first 10):")
    sample = validated[validated["exact_match"]].head(10)
    for _, r in sample.iterrows():
        print(f"  video={r['_vid'][:20]}...  batch={r['_bat']}  narration={r['_nar'][:60]}")

# -- 2. Normalized narration matches -------------------------------------
print("\n" + "=" * 80)
print("2. NORMALIZED NARRATION MATCHES")
print("=" * 80)


def normalize(text):
    t = str(text).lower()
    t = re.sub(r"[.,]+$", "", t)
    t = re.sub(r"#\w+", "", t)
    t = re.sub(r"\bthe\b", "", t)
    t = re.sub(r"\ba\b", "", t)
    t = re.sub(r"\ban\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    words = t.split()
    if len(words) > 1 and words[-1].endswith("s") and len(words[-1]) > 3:
        words[-1] = words[-1].rstrip("s")
    return " ".join(words)


deduped["narration_norm"] = deduped["narration_text"].apply(normalize)
validated["narration_norm"] = validated["narration_text"].apply(normalize)

deduped_norm_counts = Counter(deduped["narration_norm"])

validated["norm_in_deduped"] = validated["narration_norm"].isin(
    set(deduped["narration_norm"])
)
validated["norm_count_in_deduped"] = validated["narration_norm"].map(
    lambda x: deduped_norm_counts.get(x, 0)
)

n_norm = validated["norm_in_deduped"].sum()
print(f"Validated rows whose normalized narration appears in deduped: {n_norm} / {len(validated)}")
print(f"Validated rows whose normalized narration is NOT in deduped:  {len(validated) - n_norm}")

matched_narrations = validated[validated["norm_in_deduped"]]["narration_norm"].unique()
print(f"\nUnique normalized narrations from validated found in deduped: {len(matched_narrations)}")

print("\nTop 30 validated narrations (normalized) by occurrence count in deduped:")
print("-" * 80)
top_narrations = (
    validated[validated["norm_in_deduped"]]
    .groupby("narration_norm")["norm_count_in_deduped"]
    .first()
    .sort_values(ascending=False)
    .head(30)
)
for i, (narr, cnt) in enumerate(top_narrations.items(), 1):
    print(f"  {i:2d}. [{cnt:3d}x in deduped] {narr[:80]}")

# -- 3. WHY are they still there? ----------------------------------------
print("\n" + "=" * 80)
print("3. WHY DO THESE MATCHES EXIST?  (categorization)")
print("=" * 80)

val_lookup = {}
for _, r in validated[validated["norm_in_deduped"]].iterrows():
    key = r["narration_norm"]
    val_lookup.setdefault(key, []).append(
        (str(r["video_uid"]), str(r["batch"]), r.get("timestamp_sec", None))
    )

ded_lookup = {}
for _, r in deduped[deduped["narration_norm"].isin(matched_narrations)].iterrows():
    key = r["narration_norm"]
    ded_lookup.setdefault(key, []).append(
        (str(r["video_uid"]), str(r["batch"]), r.get("timestamp_sec", None))
    )

categories = Counter()
detail_rows = []

for narr_norm in matched_narrations:
    val_entries = val_lookup.get(narr_norm, [])
    ded_entries = ded_lookup.get(narr_norm, [])

    val_vids = {v for v, b, t in val_entries}
    val_bats = {b for v, b, t in val_entries}

    for d_vid, d_bat, d_ts in ded_entries:
        same_batch = d_bat in val_bats
        same_video = d_vid in val_vids

        if same_batch and same_video:
            cat = "same_batch_same_video"
        elif same_batch and not same_video:
            cat = "same_batch_diff_video"
        elif not same_batch and same_video:
            cat = "diff_batch_same_video"
        else:
            cat = "diff_batch_diff_video"
        categories[cat] += 1
        detail_rows.append(
            {
                "narration_norm": narr_norm,
                "deduped_video": d_vid,
                "deduped_batch": d_bat,
                "category": cat,
            }
        )

print("\nCategory counts (each count = one deduped row matching a validated narration):")
print("-" * 60)
total_cat = sum(categories.values())
for cat in [
    "same_batch_same_video",
    "same_batch_diff_video",
    "diff_batch_same_video",
    "diff_batch_diff_video",
]:
    cnt = categories.get(cat, 0)
    pct = cnt / total_cat * 100 if total_cat else 0
    print(f"  {cat:30s}  {cnt:6,d}  ({pct:5.1f}%)")
print(f"  {'TOTAL':30s}  {total_cat:6,d}")

detail_df = pd.DataFrame(detail_rows)
for cat in [
    "same_batch_same_video",
    "same_batch_diff_video",
    "diff_batch_same_video",
    "diff_batch_diff_video",
]:
    subset = detail_df[detail_df["category"] == cat]
    if len(subset) == 0:
        continue
    print(f"\n  Examples for '{cat}' (up to 5):")
    for _, r in subset.head(5).iterrows():
        print(
            f"    narration: {r['narration_norm'][:55]:55s}  "
            f"deduped_video: {r['deduped_video'][:20]}...  "
            f"batch: {r['deduped_batch']}"
        )

# -- 4. Total impact -----------------------------------------------------
print("\n" + "=" * 80)
print("4. TOTAL IMPACT ON DEDUPED")
print("=" * 80)

deduped_rows_with_validated_narr = deduped[
    deduped["narration_norm"].isin(
        set(validated[validated["norm_in_deduped"]]["narration_norm"])
    )
]
n_affected = len(deduped_rows_with_validated_narr)
total_deduped = len(deduped)
pct = n_affected / total_deduped * 100

print(f"Total deduped rows:                                {total_deduped:,d}")
print(f"Deduped rows with a validated normalized narration: {n_affected:,d}")
print(f"Percentage of deduped affected:                    {pct:.2f}%")
print()

print("Breakdown by batch in deduped (rows matching validated narrations):")
print("-" * 50)
batch_breakdown = (
    deduped_rows_with_validated_narr.groupby("batch")
    .size()
    .sort_values(ascending=False)
)
for batch, cnt in batch_breakdown.items():
    batch_total = len(deduped[deduped["batch"] == batch])
    print(f"  batch {batch}: {cnt:,d} / {batch_total:,d} rows ({cnt/batch_total*100:.1f}%)")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
