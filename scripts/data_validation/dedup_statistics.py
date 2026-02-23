import pandas as pd
import re
import string
from collections import Counter, defaultdict
import math

# ============================================================
# LOAD DATA
# ============================================================
ROUND1 = "/Users/huangjunda/Desktop/MIT 2.156/HAR/HAR_Lab_Initiative_AI/data/annotation_rounds/r001_numerical/round1_full.csv"
VALIDATED = "/Users/huangjunda/Desktop/MIT 2.156/HAR/HAR_Lab_Initiative_AI/data/1000_validated/project-7-at-2026-02-22-01-41-8ea80ad7.csv"

df_full = pd.read_csv(ROUND1)
df_val = pd.read_csv(VALIDATED)

# ============================================================
# NORMALIZE NARRATIONS
# ============================================================
def normalize(text):
    """Lowercase, strip punctuation, remove #tags, collapse whitespace."""
    if pd.isna(text):
        return ""
    t = str(text).lower()
    t = re.sub(r'#\S+', '', t)           # remove hashtags
    t = t.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    t = re.sub(r'\s+', ' ', t).strip()    # collapse whitespace
    return t

df_full['norm_narr'] = df_full['narration_text'].apply(normalize)
df_val['norm_narr'] = df_val['narration_text'].apply(normalize)

# ============================================================
# SECTION 1: Cross-batch duplication in the 30K assigned set
# ============================================================
print("=" * 80)
print("SECTION 1: CROSS-BATCH DUPLICATION IN 30K ASSIGNED SET")
print("=" * 80)

df_assigned = df_full[df_full['batch'].notna()].copy()
df_assigned['batch'] = df_assigned['batch'].astype(int)

print(f"\nTotal rows in full dataset:       {len(df_full):,}")
print(f"Rows with batch assigned (30K):   {len(df_assigned):,}")
print(f"Rows without batch (unassigned):  {len(df_full) - len(df_assigned):,}")
print(f"Unique normalized narrations (assigned): {df_assigned['norm_narr'].nunique():,}")

# For each unique narration, how many batches does it appear in?
narr_batches = df_assigned.groupby('norm_narr')['batch'].apply(lambda x: set(x))
narr_batch_count = narr_batches.apply(len)

batch_dist = narr_batch_count.value_counts().sort_index()
print(f"\nDistribution: narrations appearing in N batches:")
print(f"  {'Batches':>8}  {'Narrations':>12}  {'Cumulative%':>12}")
total_unique = len(narr_batch_count)
cum = 0
for n_batches, count in batch_dist.items():
    cum += count
    pct = cum / total_unique * 100
    print(f"  {n_batches:>8}  {count:>12,}  {pct:>11.1f}%")

multi_batch_narrations = narr_batch_count[narr_batch_count > 1]
print(f"\nNarrations in >1 batch: {len(multi_batch_narrations):,} / {total_unique:,} ({len(multi_batch_narrations)/total_unique*100:.1f}%)")

# Count redundant rows
redundant = 0
for narr, batches in narr_batches.items():
    if len(batches) > 1:
        # count rows for this narration in each batch; all but one batch are redundant
        rows_per_batch = df_assigned[df_assigned['norm_narr'] == narr].groupby('batch').size()
        # keep the batch with the most rows, remove from others
        total_rows = rows_per_batch.sum()
        max_batch_rows = rows_per_batch.max()
        redundant += (total_rows - max_batch_rows)

# Simpler: total rows minus unique narrations (rough upper bound)
total_assigned_rows = len(df_assigned)
unique_narrs_assigned = df_assigned['norm_narr'].nunique()
total_dup_rows = total_assigned_rows - df_assigned.drop_duplicates('norm_narr').shape[0]

print(f"\nTotal assigned rows:                {total_assigned_rows:,}")
print(f"Rows if fully deduplicated:         {unique_narrs_assigned:,}")
print(f"Total duplicate rows (all copies):  {total_dup_rows:,}")
print(f"  - That's {total_dup_rows/total_assigned_rows*100:.1f}% of assigned work is redundant")

# ============================================================
# SECTION 2: Per-Batch Analysis
# ============================================================
print("\n" + "=" * 80)
print("SECTION 2: PER-BATCH ANALYSIS")
print("=" * 80)

print(f"\n{'Batch':>5} | {'Total':>6} | {'Unique':>6} | {'Within-dup':>10} | {'Cross-dup':>9} | {'Dup%':>5}")
print("-" * 60)

batch_stats = []
for b in sorted(df_assigned['batch'].unique()):
    bdf = df_assigned[df_assigned['batch'] == b]
    total = len(bdf)
    unique_in_batch = bdf['norm_narr'].nunique()
    within_dup = total - unique_in_batch
    
    # Cross-batch: narrations in this batch that also appear in other batches
    other_narrs = set(df_assigned[df_assigned['batch'] != b]['norm_narr'].unique())
    this_narrs = set(bdf['norm_narr'].unique())
    cross_overlap = len(this_narrs & other_narrs)
    
    print(f"{b:>5} | {total:>6,} | {unique_in_batch:>6,} | {within_dup:>10,} | {cross_overlap:>9,} | {(within_dup+cross_overlap)/total*100:>4.1f}%")
    batch_stats.append({
        'batch': b, 'total': total, 'unique': unique_in_batch,
        'within_dup': within_dup, 'cross_overlap': cross_overlap
    })

print("-" * 60)
totals = pd.DataFrame(batch_stats)
print(f"{'SUM':>5} | {totals['total'].sum():>6,} | {totals['unique'].sum():>6,} | {totals['within_dup'].sum():>10,} | {totals['cross_overlap'].sum():>9,} |")
print(f"\nNote: cross-dup counts each side, so the sum double-counts overlapping pairs.")

# ============================================================
# SECTION 3: Option A - Remove duplicates within each batch
# ============================================================
print("\n" + "=" * 80)
print("SECTION 3: OPTION A - REMOVE WITHIN-BATCH DUPLICATES")
print("=" * 80)
print("Keep first occurrence of each narration per batch.\n")

df_optA = df_assigned.drop_duplicates(subset=['batch', 'norm_narr'], keep='first')

print(f"{'Batch':>5} | {'Before':>7} | {'After':>7} | {'Removed':>7}")
print("-" * 40)
for b in sorted(df_assigned['batch'].unique()):
    before = len(df_assigned[df_assigned['batch'] == b])
    after = len(df_optA[df_optA['batch'] == b])
    print(f"{b:>5} | {before:>7,} | {after:>7,} | {before-after:>7,}")

total_before = len(df_assigned)
total_after = len(df_optA)
print("-" * 40)
print(f"{'TOTAL':>5} | {total_before:>7,} | {total_after:>7,} | {total_before-total_after:>7,}")

# Cross-batch redundancies remaining
optA_narr_batches = df_optA.groupby('norm_narr')['batch'].apply(lambda x: len(set(x)))
cross_remaining = (optA_narr_batches > 1).sum()
print(f"\nCross-batch overlapping narrations still remaining: {cross_remaining:,}")
print(f"Total unique narrations across all batches: {df_optA['norm_narr'].nunique():,}")
print(f"Sum of per-batch items: {len(df_optA):,}")
print(f"Cross-batch redundant rows: {len(df_optA) - df_optA['norm_narr'].nunique():,}")

# ============================================================
# SECTION 4: Option B - Full dedup across all batches
# ============================================================
print("\n" + "=" * 80)
print("SECTION 4: OPTION B - FULL DEDUP ACROSS ALL BATCHES")
print("=" * 80)
print("Each unique (normalized narration, action) pair appears exactly once.\n")

# Unique narration+action pairs
unique_na = df_assigned.drop_duplicates(subset=['norm_narr', 'action'])
unique_narr_only = df_assigned.drop_duplicates(subset=['norm_narr'])

print(f"Unique normalized narrations:                {df_assigned['norm_narr'].nunique():,}")
print(f"Unique (narration, action) pairs:            {len(unique_na):,}")
print(f"  -> Some narrations have multiple actions assigned")

# Check action agreement
multi_action = df_assigned.groupby('norm_narr')['action'].apply(lambda x: len(set(x)))
disagree = (multi_action > 1).sum()
print(f"  -> Narrations with >1 action label:        {disagree:,} ({disagree/len(multi_action)*100:.1f}%)")

n_batches = df_assigned['batch'].nunique()
per_batch = len(unique_narr_only) // n_batches
remainder = len(unique_narr_only) % n_batches
print(f"\nIf {len(unique_narr_only):,} unique narrations distributed evenly across {n_batches} batches:")
print(f"  -> ~{per_batch:,} per batch (+ {remainder} batches get 1 extra)")

# Class balance check
print(f"\nClass balance (action) in unique narrations:")
action_counts = unique_narr_only['action'].value_counts()
for act, cnt in action_counts.items():
    print(f"  {act:<20s} {cnt:>6,}  ({cnt/len(unique_narr_only)*100:.1f}%)")

# ============================================================
# SECTION 5: Option C - Hybrid: keep validated, dedup the rest
# ============================================================
print("\n" + "=" * 80)
print("SECTION 5: OPTION C - HYBRID (KEEP VALIDATED, DEDUP REST)")
print("=" * 80)

# What's validated?
print(f"\nValidated file rows:          {len(df_val):,}")
print(f"Unique annotators:            {df_val['annotator'].nunique()}")
print(f"Batches in validated:         {sorted(df_val['batch'].unique())}")

# How many validated rows are status_main != "Delete Row" or similar
if 'status_main' in df_val.columns:
    valid_statuses = df_val[df_val['status_main'].notna() & (df_val['status_main'] != '')]
    print(f"Validated with status_main:   {len(valid_statuses):,}")
    print(f"  Status distribution:")
    for s, c in df_val['status_main'].value_counts().items():
        print(f"    {s:<20s} {c:>5,}")
    if 'status_secondary' in df_val.columns:
        sec = df_val['status_secondary'].value_counts()
        print(f"  Secondary status:")
        for s, c in sec.items():
            if pd.notna(s) and s != '':
                print(f"    {s:<20s} {c:>5,}")

# Count per batch in validated
print(f"\nValidated rows per batch:")
val_per_batch = df_val.groupby('batch').size()
for b, cnt in val_per_batch.items():
    total_in_batch = len(df_assigned[df_assigned['batch'] == b])
    print(f"  Batch {b}: {cnt:>4,} validated out of {total_in_batch:>5,} ({cnt/total_in_batch*100:.1f}%)")

# Validated narrations
validated_narrs = set(df_val['norm_narr'].unique())
print(f"\nUnique validated narrations:   {len(validated_narrs):,}")

# Remove validated narrations from remaining batches
# Keep validated rows as-is, then dedup unvalidated across batches
df_unvalidated = df_assigned[~df_assigned['norm_narr'].isin(validated_narrs)]
print(f"Assigned rows NOT yet validated (by narration): {len(df_unvalidated):,}")

# Dedup unvalidated across all batches
df_unval_dedup = df_unvalidated.drop_duplicates(subset=['norm_narr'], keep='first')
print(f"After full dedup of unvalidated:  {len(df_unval_dedup):,}")

total_optC = len(validated_narrs) + len(df_unval_dedup)
print(f"Total work (validated + deduped): {total_optC:,}")

per_batch_C = len(df_unval_dedup) // n_batches
remainder_C = len(df_unval_dedup) % n_batches
print(f"\nRedistribute {len(df_unval_dedup):,} remaining items across {n_batches} batches:")
print(f"  -> ~{per_batch_C:,} per batch (+ {remainder_C} batches get 1 extra)")
print(f"  -> Plus each annotator keeps their already-validated items")

# Per annotator view for Option C
print(f"\nPer-annotator workload under Option C:")
for b in sorted(df_assigned['batch'].unique()):
    already_done = len(df_val[df_val['batch'] == b]) if b in val_per_batch.index else 0
    new_work = per_batch_C + (1 if b <= remainder_C else 0)
    print(f"  Batch {int(b):>2}: {already_done:>4} done + {new_work:>5,} new = {already_done + new_work:>5,} total")

# ============================================================
# SECTION 6: Comparison - rows per annotator under each option
# ============================================================
print("\n" + "=" * 80)
print("SECTION 6: COMPARISON - ROWS PER ANNOTATOR")
print("=" * 80)

print(f"\n{'Batch':>5} | {'Current':>8} | {'Opt A':>8} | {'Opt B':>8} | {'Opt C':>8}")
print(f"{'':>5} | {'(as-is)':>8} | {'(w/in)':>8} | {'(full)':>8} | {'(hybrid)':>8}")
print("-" * 52)

total_A = total_B = total_C_total = 0
for b in sorted(df_assigned['batch'].unique()):
    current = len(df_assigned[df_assigned['batch'] == b])
    optA = len(df_optA[df_optA['batch'] == b])
    optB = per_batch + (1 if b <= remainder else 0)
    already_done = len(df_val[df_val['batch'] == b]) if b in val_per_batch.index else 0
    optC_new = per_batch_C + (1 if b <= remainder_C else 0)
    optC_total_b = already_done + optC_new
    
    total_A += optA
    total_B += optB
    total_C_total += optC_total_b
    
    print(f"{int(b):>5} | {current:>8,} | {optA:>8,} | {optB:>8,} | {optC_total_b:>8,}")

print("-" * 52)
print(f"{'TOTAL':>5} | {len(df_assigned):>8,} | {total_A:>8,} | {total_B:>8,} | {total_C_total:>8,}")
print(f"{'UNIQ':>5} | {unique_narrs_assigned:>8,} | {df_optA['norm_narr'].nunique():>8,} | {len(unique_narr_only):>8,} | {total_optC:>8,}")

savings_A = len(df_assigned) - total_A
savings_B = len(df_assigned) - total_B
savings_C = len(df_assigned) - total_C_total
print(f"\nTotal rows saved vs current assignment:")
print(f"  Option A (within-batch dedup):     {savings_A:>6,} fewer rows ({savings_A/len(df_assigned)*100:.1f}%)")
print(f"  Option B (full dedup, even split):  {savings_B:>6,} fewer rows ({savings_B/len(df_assigned)*100:.1f}%)")
print(f"  Option C (hybrid, keep validated):  {savings_C:>6,} fewer rows ({savings_C/len(df_assigned)*100:.1f}%)")

print("\n" + "=" * 80)
print("KEY TAKEAWAY")
print("=" * 80)
print(f"""
Current assignment: {len(df_assigned):,} total rows across {n_batches} batches
Unique narrations:  {unique_narrs_assigned:,}
Redundancy:         {len(df_assigned) - unique_narrs_assigned:,} rows ({(len(df_assigned) - unique_narrs_assigned)/len(df_assigned)*100:.1f}%)

Already validated:  {len(df_val):,} rows ({len(validated_narrs):,} unique narrations)
Remaining work:     {unique_narrs_assigned - len(validated_narrs):,} unique narrations still need annotation

Recommendation: Option C preserves validated work and eliminates redundancy.
  Each annotator would handle ~{per_batch_C:,} new items + their already-validated ones.
""")

