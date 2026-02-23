#!/usr/bin/env python3
"""
dedup_validation_batches.py

Generates a deduplicated CSV for HAR annotation validation.

Steps:
  1. Load assigned set (30,000 rows) and validated set (1,165 rows).
  2. Normalize narration text for consistent matching.
  3. Remove already-validated rows from the assigned set.
  4. Deduplicate within each batch by (normalized_narration, action).
  5. Deduplicate across batches, keeping each narration in the batch with
     the fewest remaining items (for load-balancing).
  6. Write the final CSV and print a comprehensive summary table.
"""

import os
import re
import csv
import time
from collections import defaultdict, Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/Users/huangjunda/Desktop/MIT 2.156/HAR/HAR_Lab_Initiative_AI")
ASSIGNED_CSV = BASE / "data/annotation_rounds/r001_numerical/round1_assigned_only.csv"
VALIDATED_CSV = BASE / "data/1000_validated/project-7-at-2026-02-22-01-41-8ea80ad7.csv"
OUTPUT_DIR = BASE / "data/annotation_rounds/r001_deduped"
OUTPUT_CSV = OUTPUT_DIR / "round1_deduped.csv"

# Columns for the output (same as assigned)
OUT_COLS = [
    "video_uid", "timestamp_sec", "narration_text",
    "scenario", "action", "reasoning", "status", "batch", "round",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_narration(text: str) -> str:
    """Normalize a narration string for dedup matching.

    Levels applied:
      1. Lowercase
      2. Remove trailing punctuation
      3. Remove #hashtag tokens (#C, #unsure, etc.)
      4. Remove articles (the, a, an)
      5. Normalize simple plurals (trailing 's' on last word)
      6. Collapse spaces and strip
    """
    t = text.lower()                           # 1. lowercase
    t = re.sub(r'[.,]+$', '', t)               # 2. trailing punctuation
    t = re.sub(r'#\w+', '', t)                 # 3. remove #hashtag tokens
    t = re.sub(r'\bthe\b', '', t)              # 4a. remove "the"
    t = re.sub(r'\ba\b', '', t)                # 4b. remove "a"
    t = re.sub(r'\ban\b', '', t)               # 4c. remove "an"
    # 5. normalize simple plurals on last word
    t = re.sub(r'\s+', ' ', t).strip()
    words = t.split()
    if len(words) > 1 and words[-1].endswith('s') and len(words[-1]) > 3:
        words[-1] = words[-1].rstrip('s')
    t = ' '.join(words)
    return t


def load_csv(path: Path) -> list:
    """Read a CSV file into a list of dicts."""
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def fmt_time(seconds: float) -> str:
    """Format seconds into Xh Ym Zs."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m or h:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    assigned_rows = load_csv(ASSIGNED_CSV)
    validated_rows = load_csv(VALIDATED_CSV)
    print(f"Loaded {len(assigned_rows):,} assigned rows, "
          f"{len(validated_rows):,} validated rows.\n")

    # ------------------------------------------------------------------
    # Step 1 & 2: Build validated keys and normalize
    # ------------------------------------------------------------------
    # Build set of validated (video_uid, normalized_narration, batch) triples
    validated_keys = set()
    for row in validated_rows:
        vid = row["video_uid"].strip()
        narr_norm = normalize_narration(row["narration_text"])
        batch = str(row["batch"]).strip()
        validated_keys.add((vid, narr_norm, batch))

    # Enrich assigned rows with normalized narration
    for row in assigned_rows:
        row["_narr_norm"] = normalize_narration(row["narration_text"])

    # ------------------------------------------------------------------
    # Step 3: Remove validated rows
    # ------------------------------------------------------------------
    validated_removed = Counter()  # per batch
    remaining = []
    for row in assigned_rows:
        key = (row["video_uid"].strip(),
               row["_narr_norm"],
               str(row["batch"]).strip())
        if key in validated_keys:
            validated_removed[int(row["batch"])] += 1
            validated_keys.discard(key)  # remove so each match is one-to-one
        else:
            remaining.append(row)

    total_validated_removed = sum(validated_removed.values())
    print(f"Removed {total_validated_removed:,} validated rows.\n")

    # ------------------------------------------------------------------
    # Step 4: Deduplicate WITHIN each batch
    # ------------------------------------------------------------------
    # Safety guard: only merge rows with the same normalized narration
    # if they also share the same action label.  When article/plural
    # normalization collapses two narrations that have DIFFERENT action
    # labels, we keep both (use original narration_text as tiebreaker).
    within_batch_removed = Counter()
    after_within = []
    seen_within = defaultdict(set)  # batch -> set of dedup keys

    for row in remaining:
        batch = int(row["batch"])
        # Primary dedup key: (normalized narration, action)
        dup_key = (row["_narr_norm"], row["action"].strip())
        if dup_key in seen_within[batch]:
            within_batch_removed[batch] += 1
        else:
            seen_within[batch].add(dup_key)
            after_within.append(row)

    total_within = sum(within_batch_removed.values())
    print(f"Removed {total_within:,} within-batch duplicates.\n")

    # ------------------------------------------------------------------
    # Step 4b: Ambiguity guard for article/plural normalization
    # ------------------------------------------------------------------
    # Check: did normalization collapse narrations with different actions?
    # If so, they are genuinely different and should NOT be merged.
    # We handle this by using (narr_norm, action) as the key -- so two
    # rows with the same narr_norm but different actions both survive.
    ambiguous_narrations = defaultdict(set)
    for row in after_within:
        ambiguous_narrations[row["_narr_norm"]].add(row["action"].strip())
    ambiguous = {n: acts for n, acts in ambiguous_narrations.items()
                 if len(acts) > 1}
    if ambiguous:
        print(f"Note: {len(ambiguous)} normalized narrations map to multiple "
              f"action labels (kept separate by action).\n")

    # ------------------------------------------------------------------
    # Step 5: Deduplicate ACROSS batches (balance-based)
    # ------------------------------------------------------------------
    # Group rows by (narr_norm, action) -> list of (batch, row)
    narr_action_to_rows = defaultdict(list)
    for row in after_within:
        key = (row["_narr_norm"], row["action"].strip())
        batch = int(row["batch"])
        narr_action_to_rows[key].append((batch, row))

    # Identify keys that appear in more than one batch
    cross_dup_keys = {k: v for k, v in narr_action_to_rows.items()
                      if len({b for b, _ in v}) > 1}

    # Current batch sizes (after within-batch dedup)
    batch_sizes = Counter()
    for row in after_within:
        batch_sizes[int(row["batch"])] += 1

    # Sort cross-dup keys by how many batches they span (most first)
    sorted_cross_keys = sorted(
        cross_dup_keys.keys(),
        key=lambda k: len({b for b, _ in cross_dup_keys[k]}),
        reverse=True,
    )

    # For each cross-dup key, keep it in the batch with fewest items
    cross_batch_removed = Counter()
    rows_to_remove = set()  # id(row) of rows to remove

    for key in sorted_cross_keys:
        entries = narr_action_to_rows[key]
        batches_present = {b for b, _ in entries}
        if len(batches_present) <= 1:
            continue

        # Pick the batch with the smallest current count
        best_batch = min(batches_present, key=lambda b: batch_sizes[b])

        kept_one = False
        for batch, row in entries:
            if batch == best_batch and not kept_one:
                kept_one = True  # keep this one
                continue
            # Mark for removal
            rows_to_remove.add(id(row))
            cross_batch_removed[batch] += 1
            batch_sizes[batch] -= 1

    # Build final list
    final_rows = [row for row in after_within if id(row) not in rows_to_remove]

    total_cross = sum(cross_batch_removed.values())
    print(f"Removed {total_cross:,} cross-batch duplicates.\n")

    # ------------------------------------------------------------------
    # Step 6: Sort and write output
    # ------------------------------------------------------------------
    # Sort by batch (int), video_uid, timestamp_sec (float)
    final_rows.sort(key=lambda r: (
        int(r["batch"]),
        r["video_uid"],
        float(r["timestamp_sec"]),
    ))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Wrote {len(final_rows):,} rows to:\n  {OUTPUT_CSV}\n")

    # ------------------------------------------------------------------
    # Step 7: Summary table
    # ------------------------------------------------------------------
    all_batches = sorted(set(int(r["batch"]) for r in assigned_rows))

    # Original counts per batch
    orig_counts = Counter(int(r["batch"]) for r in assigned_rows)
    # Final counts per batch
    final_counts = Counter(int(r["batch"]) for r in final_rows)

    # Table header
    hdr = (f"{'Batch':>6} | {'Original':>8} | {'Validated':>9} | "
           f"{'Within-Dup':>10} | {'Cross-Dup':>9} | {'Final':>6}")
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    tot_orig = tot_val = tot_within_t = tot_cross_t = tot_final = 0
    for b in all_batches:
        o = orig_counts[b]
        v = validated_removed.get(b, 0)
        w = within_batch_removed.get(b, 0)
        c = cross_batch_removed.get(b, 0)
        fn = final_counts.get(b, 0)
        print(f"{b:>6} | {o:>8,} | {v:>9,} | {w:>10,} | {c:>9,} | {fn:>6,}")
        tot_orig += o
        tot_val += v
        tot_within_t += w
        tot_cross_t += c
        tot_final += fn

    print(sep)
    print(f"{'Total':>6} | {tot_orig:>8,} | {tot_val:>9,} | "
          f"{tot_within_t:>10,} | {tot_cross_t:>9,} | {tot_final:>6,}")
    print(sep)

    # Unique narrations in final output
    unique_narrations = set(row["_narr_norm"] for row in final_rows)
    print(f"\nUnique normalized narrations in final output: {len(unique_narrations):,}")

    # Verification: no duplicate (narr_norm, action) across any batches
    seen_global = set()
    dup_count = 0
    for row in final_rows:
        key = (row["_narr_norm"], row["action"].strip())
        if key in seen_global:
            dup_count += 1
        else:
            seen_global.add(key)

    if dup_count == 0:
        print("Verification PASSED: No duplicate (narration, action) pairs across batches.")
    else:
        print(f"Verification FAILED: {dup_count} duplicate (narration, action) pairs remain!")

    # Time estimate (5 seconds per row)
    est_sec = tot_final * 5
    print(f"\nEstimated annotation time at 5 s/row: {fmt_time(est_sec)} "
          f"({tot_final:,} rows x 5 s)")

    elapsed = time.time() - t0
    print(f"\nScript completed in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()
