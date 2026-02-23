import pandas as pd
import re
from collections import defaultdict

# ─── Load data ───────────────────────────────────────────────────────────────
CSV = "/Users/huangjunda/Desktop/MIT 2.156/HAR/HAR_Lab_Initiative_AI/data/annotation_rounds/r001_deduped/round1_deduped.csv"
df = pd.read_csv(CSV)
print(f"{'='*90}")
print(f"  NEAR-DUPLICATE ANALYSIS  —  round1_deduped.csv")
print(f"{'='*90}\n")

# ─── Current normalization (already applied during dedup) ────────────────────
def normalize_current(text):
    t = str(text).lower()
    t = re.sub(r'[.,]+$', '', t)
    t = re.sub(r'#\w+', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

df['norm_current'] = df['narration_text'].apply(normalize_current)

# ─── Additional normalizations ───────────────────────────────────────────────
def norm_articles(text):
    """Level A: remove standalone articles."""
    t = re.sub(r'\b(the|a|an)\b', '', text)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def norm_plural(text):
    """Level B: strip trailing 's'/'es' from the LAST word."""
    words = text.split()
    if not words:
        return text
    last = words[-1]
    if last.endswith('ies') and len(last) > 4:
        # e.g. "berries" → "berry"  (but keep short words)
        candidate = last[:-3] + 'y'
    elif last.endswith('shes') or last.endswith('ches') or last.endswith('xes') or last.endswith('ses') or last.endswith('zes'):
        candidate = last[:-2]
    elif last.endswith('s') and not last.endswith('ss') and len(last) > 2:
        candidate = last[:-1]
    else:
        candidate = last
    words[-1] = candidate
    return ' '.join(words)

def norm_tense(text):
    """Level C: strip trailing 's' from the FIRST verb (word after 'c ')."""
    words = text.split()
    if len(words) < 2:
        return text
    # The subject is typically "c" at index 0.  The verb is index 1.
    verb = words[1]
    if verb.endswith('ies') and len(verb) > 4:
        verb = verb[:-3] + 'y'
    elif verb.endswith('shes') or verb.endswith('ches') or verb.endswith('xes') or verb.endswith('ses') or verb.endswith('zes'):
        verb = verb[:-2]
    elif verb.endswith('s') and not verb.endswith('ss') and len(verb) > 2:
        verb = verb[:-1]
    words[1] = verb
    return ' '.join(words)

# ─── Build normalized columns ────────────────────────────────────────────────
df['norm_A'] = df['norm_current'].apply(norm_articles)
df['norm_B'] = df['norm_current'].apply(norm_plural)
df['norm_C'] = df['norm_current'].apply(norm_tense)
df['norm_ABC'] = df['norm_current'].apply(norm_articles).apply(norm_plural).apply(norm_tense)

# ─── Section 1: Current state ────────────────────────────────────────────────
print(f"{'─'*90}")
print(f"  1. CURRENT STATE OF THE DEDUPED FILE")
print(f"{'─'*90}")
total_rows = len(df)
unique_current = df.groupby(['norm_current', 'action']).ngroups
print(f"  Total rows:                              {total_rows:,}")
print(f"  Unique (narration_norm, action) pairs:   {unique_current:,}")
print(f"  Exact duplicates remaining:              {total_rows - unique_current:,}")
print()

# Check if any exact dupes remain under current norm
if total_rows != unique_current:
    dupes = df.groupby(['norm_current', 'action']).filter(lambda g: len(g) > 1)
    dupe_groups = dupes.groupby(['norm_current', 'action']).size().sort_values(ascending=False).head(10)
    print("  Top 10 exact-duplicate groups still present:")
    for (narr, act), cnt in dupe_groups.items():
        print(f"    [{cnt}x] action={act:20s}  \"{narr}\"")
    print()

# ─── Helper: analyze a normalization level ───────────────────────────────────
def analyze_level(label, norm_col, top_n=30):
    print(f"{'─'*90}")
    print(f"  2{label}. LEVEL {label.upper()} NORMALIZATION")
    print(f"{'─'*90}")

    # Group by (normalized_narration, action) under NEW normalization
    # But we need to find groups where DIFFERENT norm_current values collapse
    # into the SAME new normalized value + action
    groups = df.groupby([norm_col, 'action'])

    merge_data = []
    for (new_narr, action), grp in groups:
        distinct_originals = grp['norm_current'].unique()
        if len(distinct_originals) > 1:
            merge_data.append({
                'new_narr': new_narr,
                'action': action,
                'originals': list(distinct_originals),
                'count': len(grp),
                'n_originals': len(distinct_originals),
                'rows_saved': len(grp) - 1  # would collapse to 1
            })

    unique_new = groups.ngroups
    rows_after = unique_new  # if we kept one per group
    additional_saved = unique_current - unique_new

    print(f"  Unique (narration_norm, action) pairs:  {unique_new:,}")
    print(f"  Additional pairs collapsed:             {additional_saved:,}")
    print(f"  Merge groups found:                     {len(merge_data)}")
    print()

    if not merge_data:
        print("  No merges found.\n")
        return merge_data, unique_new, additional_saved

    # Sort by number of originals collapsing, then count
    merge_data.sort(key=lambda x: (-x['n_originals'], -x['count']))

    print(f"  Top {min(top_n, len(merge_data))} merge groups:")
    print(f"  {'No.':<5} {'Action':<25} {'#Orig':<7} {'#Rows':<7} Originals → Merged")
    print(f"  {'---':<5} {'---':<25} {'---':<7} {'---':<7} {'---'}")
    for i, mg in enumerate(merge_data[:top_n]):
        origs = ' | '.join(f'"{o}"' for o in mg['originals'][:5])
        if len(mg['originals']) > 5:
            origs += f" ... +{len(mg['originals'])-5} more"
        print(f"  {i+1:<5} {mg['action']:<25} {mg['n_originals']:<7} {mg['count']:<7} {origs}")
        print(f"  {'':5} {'':25} {'':7} {'':7} → \"{mg['new_narr']}\"")
    print()

    return merge_data, unique_new, additional_saved

# ─── Section 2: Each level ───────────────────────────────────────────────────
merges_A, uniq_A, saved_A = analyze_level('A', 'norm_A')
merges_B, uniq_B, saved_B = analyze_level('B', 'norm_B')
merges_C, uniq_C, saved_C = analyze_level('C', 'norm_C')

# ─── Section 3: Combined A+B+C ──────────────────────────────────────────────
print(f"{'─'*90}")
print(f"  3. COMBINED (A + B + C) NORMALIZATION")
print(f"{'─'*90}")

groups_abc = df.groupby(['norm_ABC', 'action'])
uniq_ABC = groups_abc.ngroups
saved_ABC = unique_current - uniq_ABC

# Find merge groups
merge_abc = []
for (new_narr, action), grp in groups_abc:
    distinct_originals = grp['norm_current'].unique()
    if len(distinct_originals) > 1:
        merge_abc.append({
            'new_narr': new_narr,
            'action': action,
            'originals': list(distinct_originals),
            'count': len(grp),
            'n_originals': len(distinct_originals),
        })

merge_abc.sort(key=lambda x: (-x['n_originals'], -x['count']))

print(f"  Unique (narration_norm, action) pairs:  {uniq_ABC:,}")
print(f"  Additional pairs collapsed:             {saved_ABC:,}")
print(f"  Merge groups:                           {len(merge_abc)}")
print()

print(f"  Top 30 merge groups (combined):")
print(f"  {'No.':<5} {'Action':<25} {'#Orig':<7} {'#Rows':<7} Originals → Merged")
print(f"  {'---':<5} {'---':<25} {'---':<7} {'---':<7} {'---'}")
for i, mg in enumerate(merge_abc[:30]):
    origs = ' | '.join(f'"{o}"' for o in mg['originals'][:5])
    if len(mg['originals']) > 5:
        origs += f" ... +{len(mg['originals'])-5} more"
    print(f"  {i+1:<5} {mg['action']:<25} {mg['n_originals']:<7} {mg['count']:<7} {origs}")
    print(f"  {'':5} {'':25} {'':7} {'':7} → \"{mg['new_narr']}\"")
print()

# ─── Check cross-action inconsistencies ──────────────────────────────────────
print(f"{'─'*90}")
print(f"  3b. INCONSISTENT MERGES (same norm_ABC narration, DIFFERENT actions)")
print(f"{'─'*90}")

# Find narrations that map to multiple actions under norm_ABC
narr_actions = df.groupby('norm_ABC')['action'].apply(lambda x: tuple(sorted(x.unique()))).reset_index()
inconsistent = narr_actions[narr_actions['action'].apply(len) > 1]

print(f"  Narrations mapping to multiple actions:  {len(inconsistent)}")
print()

if len(inconsistent) > 0:
    # Show details
    for i, row in inconsistent.head(30).iterrows():
        narr = row['norm_ABC']
        actions = row['action']
        subset = df[df['norm_ABC'] == narr]
        action_counts = subset['action'].value_counts()
        print(f"  \"{narr}\"")
        for act, cnt in action_counts.items():
            print(f"      → {act}: {cnt} rows")
        print()

# ─── Now check consistency WITHIN each level's merge groups ──────────────────
print(f"{'─'*90}")
print(f"  3c. CONSISTENCY CHECK: Do merged pairs have same action labels?")
print(f"{'─'*90}")
print()
print("  (Merges are done WITHIN each (narration, action) group, so by definition")
print("   the action is the same. The real risk is narrations that SHOULD differ.)")
print()
print("  Cross-narration ambiguity (same normalized text → different actions):")

for level_name, norm_col in [('A', 'norm_A'), ('B', 'norm_B'), ('C', 'norm_C'), ('A+B+C', 'norm_ABC')]:
    narr_acts = df.groupby(norm_col)['action'].nunique()
    ambiguous = narr_acts[narr_acts > 1]
    # Compare to current
    narr_acts_cur = df.groupby('norm_current')['action'].nunique()
    ambiguous_cur = narr_acts_cur[narr_acts_cur > 1]
    # NEW ambiguities introduced
    new_ambig_narrs = set(ambiguous.index) - set(ambiguous_cur.index)
    print(f"  Level {level_name:5s}: {len(ambiguous):4d} ambiguous narrations total, "
          f"{len(new_ambig_narrs):4d} NEW (introduced by normalization)")

    if new_ambig_narrs:
        print(f"    NEW ambiguous narrations (normalization made previously-distinct narrations collide):")
        for narr in sorted(new_ambig_narrs)[:10]:
            subset = df[df[norm_col] == narr]
            originals = subset['norm_current'].unique()
            action_map = subset.groupby('norm_current')['action'].apply(lambda x: sorted(x.unique())).to_dict()
            print(f"      \"{narr}\"")
            for orig, acts in action_map.items():
                print(f"        from \"{orig}\" → actions: {acts}")
        if len(new_ambig_narrs) > 10:
            print(f"        ... and {len(new_ambig_narrs)-10} more")
    print()

# ─── Section 4: Summary table ────────────────────────────────────────────────
print(f"{'='*90}")
print(f"  4. SUMMARY TABLE")
print(f"{'='*90}")
print()

def consistency_pct(norm_col):
    """% of merge groups where all merged rows have the same action."""
    groups = df.groupby(norm_col)
    total_groups = 0
    consistent_groups = 0
    for narr, grp in groups:
        total_groups += 1
        if grp['action'].nunique() == 1:
            consistent_groups += 1
    return consistent_groups / total_groups * 100 if total_groups > 0 else 100.0

# New ambiguities only
def new_ambiguity_count(norm_col):
    narr_acts = df.groupby(norm_col)['action'].nunique()
    ambiguous = set(narr_acts[narr_acts > 1].index)
    narr_acts_cur = df.groupby('norm_current')['action'].nunique()
    ambiguous_cur = set(narr_acts_cur[narr_acts_cur > 1].index)
    return len(ambiguous - ambiguous_cur)

con_cur = consistency_pct('norm_current')
con_A = consistency_pct('norm_A')
con_B = consistency_pct('norm_B')
con_C = consistency_pct('norm_C')
con_ABC = consistency_pct('norm_ABC')

new_amb_A = new_ambiguity_count('norm_A')
new_amb_B = new_ambiguity_count('norm_B')
new_amb_C = new_ambiguity_count('norm_C')
new_amb_ABC = new_ambiguity_count('norm_ABC')

header = f"  {'Normalization':<22} {'Unique pairs':>14} {'Addtl saved':>13} {'Consistency':>13} {'New ambig':>11}"
print(header)
print(f"  {'─'*20}  {'─'*14} {'─'*13} {'─'*13} {'─'*11}")
print(f"  {'Current only':<22} {unique_current:>14,} {'--':>13} {con_cur:>12.1f}% {'--':>11}")
print(f"  {'+ Articles (A)':<22} {uniq_A:>14,} {saved_A:>13,} {con_A:>12.1f}% {new_amb_A:>11,}")
print(f"  {'+ Plural (B)':<22} {uniq_B:>14,} {saved_B:>13,} {con_B:>12.1f}% {new_amb_B:>11,}")
print(f"  {'+ Tense (C)':<22} {uniq_C:>14,} {saved_C:>13,} {con_C:>12.1f}% {new_amb_C:>11,}")
print(f"  {'+ All (A+B+C)':<22} {uniq_ABC:>14,} {saved_ABC:>13,} {con_ABC:>12.1f}% {new_amb_ABC:>11,}")
print()

# ─── Bonus: show some specific examples ─────────────────────────────────────
print(f"{'='*90}")
print(f"  5. EXAMPLE NEAR-DUPLICATES (sample of what each level catches)")
print(f"{'='*90}")

for level_name, merges in [('A (Articles)', merges_A), ('B (Plural)', merges_B), ('C (Tense)', merges_C)]:
    print(f"\n  Level {level_name}:")
    if not merges:
        print("    (none)")
        continue
    for mg in merges[:5]:
        print(f"    action={mg['action']}")
        for o in mg['originals'][:4]:
            print(f"      \"{o}\"")
        print(f"      → merged as \"{mg['new_narr']}\"")
        print()

print(f"\n{'='*90}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*90}")
