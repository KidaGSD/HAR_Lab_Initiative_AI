#!/usr/bin/env python3
"""
Comprehensive Narration Pattern Analysis for HAR Dataset - v2 (optimized)
"""

import csv
import re
from collections import Counter, defaultdict
import time

CSV_PATH = "/Users/huangjunda/Desktop/MIT 2.156/HAR/HAR_Lab_Initiative_AI/data/annotation_rounds/r001_numerical/round1_full.csv"

def levenshtein(s1, s2):
    """Compute Levenshtein edit distance with early termination at threshold 3."""
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) > 2:
        return 3  # Over threshold
    if len1 < len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1
    if len2 == 0:
        return len1
    prev_row = list(range(len2 + 1))
    for i in range(len1):
        curr_row = [i + 1]
        min_val = i + 1
        for j in range(len2):
            ins = prev_row[j + 1] + 1
            dele = curr_row[j] + 1
            sub = prev_row[j] + (0 if s1[i] == s2[j] else 1)
            val = min(ins, dele, sub)
            curr_row.append(val)
            if val < min_val:
                min_val = val
        # Early termination: if minimum in row is already > 2, bail out
        if min_val > 2:
            return 3
        prev_row = curr_row
    return prev_row[-1]

def char_bigrams(s):
    """Return set of character bigrams for similarity filtering."""
    return set(s[i:i+2] for i in range(len(s)-1))

# ──────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────
print("=" * 90)
print("HAR NARRATION PATTERN ANALYSIS")
print("=" * 90)

t0 = time.time()
rows = []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

total_rows = len(rows)
narrations = [r["narration_text"] for r in rows]
actions = [r["action"] for r in rows]
scenarios = [r["scenario"] for r in rows]
print(f"\nLoaded {total_rows:,} rows in {time.time()-t0:.1f}s")


# ======================================================================
# STEP 1: Catalog all special tokens / tags in narrations
# ======================================================================
print("\n" + "=" * 90)
print("STEP 1: SPECIAL TOKENS AND TAGS IN NARRATIONS")
print("=" * 90)

hashtag_counter = Counter()
hashtag_examples = defaultdict(list)
for n in narrations:
    tags = re.findall(r"#\w+", n)
    for tag in tags:
        hashtag_counter[tag] += 1
        if len(hashtag_examples[tag]) < 3:
            hashtag_examples[tag].append(n)

print(f"\n--- All hashtag patterns found ({len(hashtag_counter)} unique) ---")
for tag, cnt in hashtag_counter.most_common():
    pct = cnt / total_rows * 100
    exs = hashtag_examples[tag]
    print(f"  {tag:<20s}  {cnt:>8,} rows ({pct:5.1f}%)  ex: {exs[0][:80]}")
    for ex in exs[1:]:
        print(f"  {'':<20s}  {'':<8s}         {ex[:80]}")

print(f"\n--- Unusual character analysis ---")
char_counter = Counter()
for n in narrations:
    for c in n:
        if not c.isalnum() and c not in " .,!?'\"#-":
            char_counter[c] += 1
if char_counter:
    print("  Characters outside [alnum, space, .,!?'\"#-]:")
    for ch, cnt in char_counter.most_common(30):
        print(f"    U+{ord(ch):04X} ({repr(ch):<8s})  {cnt:>8,} occurrences")

leading_space = sum(1 for n in narrations if n != n.lstrip())
trailing_space = sum(1 for n in narrations if n != n.rstrip())
multi_space = sum(1 for n in narrations if "  " in n)
trailing_period = sum(1 for n in narrations if n.rstrip().endswith("."))
trailing_comma = sum(1 for n in narrations if n.rstrip().endswith(","))
trailing_nothing = sum(1 for n in narrations if n.rstrip() and n.rstrip()[-1] not in ".!?,;:")
print(f"\n  Leading whitespace:   {leading_space:>8,} rows")
print(f"  Trailing whitespace:  {trailing_space:>8,} rows")
print(f"  Multiple spaces:      {multi_space:>8,} rows")
print(f"  Ending with period:   {trailing_period:>8,} rows")
print(f"  Ending with comma:    {trailing_comma:>8,} rows")
print(f"  Ending with no punct: {trailing_nothing:>8,} rows")

prefix_c = sum(1 for n in narrations if n.startswith("#C C "))
prefix_c_lower = sum(1 for n in narrations if n.lower().startswith("#c c "))
no_hash_c = sum(1 for n in narrations if not n.startswith("#C") and not n.startswith("#c"))
print(f"\n  Starts with '#C C ':  {prefix_c:>8,} rows")
print(f"  Starts with '#c c ' (case-insensitive): {prefix_c_lower:>8,} rows")
print(f"  Does NOT start with #C/#c: {no_hash_c:>8,} rows")

if no_hash_c > 0:
    no_hash_examples = [n for n in narrations if not n.startswith("#C") and not n.startswith("#c")][:20]
    print(f"\n  Examples of narrations without #C prefix:")
    for ex in no_hash_examples:
        print(f"    {repr(ex)[:100]}")


# ======================================================================
# STEP 2: Normalization-based deduplication
# ======================================================================
print("\n" + "=" * 90)
print("STEP 2: NORMALIZATION-BASED DEDUPLICATION")
print("=" * 90)

raw_counter = Counter(narrations)
unique_raw = len(raw_counter)
print(f"\n--- Level 0 (raw) ---")
print(f"  Unique narrations: {unique_raw:,}")
print(f"  Top 20 most common:")
for txt, cnt in raw_counter.most_common(20):
    print(f"    {cnt:>6,}x  {txt[:90]}")

def normalize_L1(text):
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    t = t.lower()
    t = re.sub(r"[.,;:!?]+$", "", t)
    return t.strip()

L1 = [normalize_L1(n) for n in narrations]
L1_counter = Counter(L1)
unique_L1 = len(L1_counter)
reduction_L1 = unique_raw - unique_L1
print(f"\n--- Level 1 (basic cleanup: strip, collapse spaces, lowercase, rm trailing punct) ---")
print(f"  Unique narrations: {unique_L1:,}  (reduced by {reduction_L1:,} = {reduction_L1/unique_raw*100:.1f}%)")
print(f"  Top 20 most common:")
for txt, cnt in L1_counter.most_common(20):
    print(f"    {cnt:>6,}x  {txt[:90]}")

print(f"\n  Narrations that merged (L0 -> L1) [top 30 by group size]:")
L0_to_L1 = defaultdict(set)
for n in narrations:
    L0_to_L1[normalize_L1(n)].add(n)
merged_L1 = {k: v for k, v in L0_to_L1.items() if len(v) > 1}
merged_L1_sorted = sorted(merged_L1.items(), key=lambda x: -len(x[1]))
for i, (norm, variants) in enumerate(merged_L1_sorted[:30]):
    print(f"    [{i+1}] Normalized: \"{norm[:70]}\"  ({len(variants)} variants)")
    for v in sorted(variants)[:5]:
        print(f"         - {repr(v)[:90]}")
    if len(variants) > 5:
        print(f"         ... and {len(variants)-5} more")

def normalize_L2(text):
    t = normalize_L1(text)
    t = re.sub(r"#\w+", "", t)
    return re.sub(r"\s+", " ", t).strip()

L2 = [normalize_L2(n) for n in narrations]
L2_counter = Counter(L2)
unique_L2 = len(L2_counter)
reduction_L2 = unique_L1 - unique_L2
print(f"\n--- Level 2 (remove all #tags) ---")
print(f"  Unique narrations: {unique_L2:,}  (reduced by {reduction_L2:,} = {reduction_L2/unique_L1*100:.1f}% from L1)")
print(f"  Top 20 most common:")
for txt, cnt in L2_counter.most_common(20):
    print(f"    {cnt:>6,}x  {txt[:90]}")

L1_to_L2 = defaultdict(set)
for n in L1:
    L1_to_L2[normalize_L2(n)].add(n)
merged_L2 = {k: v for k, v in L1_to_L2.items() if len(v) > 1}
merged_L2_sorted = sorted(merged_L2.items(), key=lambda x: -len(x[1]))
print(f"\n  Narrations that merged (L1 -> L2) [top 30 by group size]:")
for i, (norm, variants) in enumerate(merged_L2_sorted[:30]):
    print(f"    [{i+1}] Normalized: \"{norm[:70]}\"  ({len(variants)} variants)")
    for v in sorted(variants)[:5]:
        print(f"         - \"{v[:85]}\"")
    if len(variants) > 5:
        print(f"         ... and {len(variants)-5} more")

def normalize_L3(text):
    t = normalize_L2(text)
    if t.startswith("c "):
        t = t[2:]
    return t.strip()

L3 = [normalize_L3(n) for n in narrations]
L3_counter = Counter(L3)
unique_L3 = len(L3_counter)
reduction_L3 = unique_L2 - unique_L3
print(f"\n--- Level 3 (remove 'C' subject prefix) ---")
print(f"  Unique narrations: {unique_L3:,}  (reduced by {reduction_L3:,} = {reduction_L3/unique_L2*100:.1f}% from L2)")
print(f"  Top 20 most common:")
for txt, cnt in L3_counter.most_common(20):
    print(f"    {cnt:>6,}x  {txt[:90]}")

def extract_verb_template(text):
    t = normalize_L3(text)
    words = t.split()
    return words[0] if words else ""

L4 = [extract_verb_template(n) for n in narrations]
L4_counter = Counter(L4)
unique_L4 = len(L4_counter)
print(f"\n--- Level 4 (verb-only: first word after '#C C' removal) ---")
print(f"  Unique verb templates: {unique_L4:,}")
print(f"  Top 40 verbs:")
for verb, cnt in L4_counter.most_common(40):
    pct = cnt / total_rows * 100
    print(f"    {verb:<25s}  {cnt:>8,} rows ({pct:5.1f}%)")


# ======================================================================
# STEP 3: Specific pattern analysis
# ======================================================================
print("\n" + "=" * 90)
print("STEP 3: SPECIFIC PATTERN ANALYSIS")
print("=" * 90)

# 3.1 #unsure / #unknown
print(f"\n--- 3.1 #unsure / #unknown narrations ---")
unsure_rows_list = [(n, a) for n, a in zip(narrations, actions) if "#unsure" in n.lower() or "#unknown" in n.lower()]
unsure_narrations = Counter([n for n, _ in unsure_rows_list])
print(f"  Total rows with #unsure/#unknown: {len(unsure_rows_list):,}")
print(f"  Unique narrations:                {len(unsure_narrations):,}")
print(f"\n  Top 50 unique #unsure/#unknown narrations:")
for idx, (txt, cnt) in enumerate(unsure_narrations.most_common(50)):
    print(f"    {cnt:>5,}x  {txt[:100]}")
if len(unsure_narrations) > 50:
    print(f"    ... ({len(unsure_narrations) - 50} more not shown)")

# 3.2 Typo variants (Levenshtein distance <= 2) - OPTIMIZED
print(f"\n--- 3.2 Typo variants (Levenshtein distance <= 2) ---")
t_lev = time.time()

L1_counts = Counter(L1)
# Only narrations appearing 3+ times
freq_narrations = [(txt, cnt) for txt, cnt in L1_counts.items() if cnt >= 3]
freq_narrations.sort(key=lambda x: -x[1])
print(f"  Narrations appearing 3+ times: {len(freq_narrations):,}")

freq_count_map = {t: c for t, c in freq_narrations}
freq_texts = [t for t, c in freq_narrations]

# OPTIMIZATION: Build bigram index for fast candidate filtering
# Two strings with Lev dist <= 2 share at least (max(len)-3) bigrams
# We use a bigram inverted index to find candidates
print(f"  Building bigram index for {len(freq_texts):,} narrations...")
bigram_index = defaultdict(set)  # bigram -> set of indices
text_bigrams = {}
for idx, t in enumerate(freq_texts):
    bgs = char_bigrams(t)
    text_bigrams[idx] = bgs
    for bg in bgs:
        bigram_index[bg].add(idx)

# Group by length for the length filter
by_len = defaultdict(list)
for idx, t in enumerate(freq_texts):
    by_len[len(t)].append(idx)

print(f"  Finding near-duplicates...")
near_dupes = []
checked = 0
skipped = 0

for idx_i in range(len(freq_texts)):
    t1 = freq_texts[idx_i]
    len1 = len(t1)
    bgs1 = text_bigrams[idx_i]
    
    # Find candidate indices: same length +/- 2
    candidate_indices = set()
    for dl in range(-2, 3):
        target_len = len1 + dl
        if target_len in by_len:
            for idx_j in by_len[target_len]:
                if idx_j > idx_i:  # avoid self and duplicates
                    candidate_indices.add(idx_j)
    
    # Further filter by bigram overlap
    # For Lev dist <= 2, strings must share significant bigram overlap
    # Minimum shared bigrams: max(len1, len2) - 1 - 2*2 = max_len - 5
    # But this can be negative for short strings, so use max(1, ...)
    for idx_j in candidate_indices:
        t2 = freq_texts[idx_j]
        bgs2 = text_bigrams[idx_j]
        max_len = max(len(t1), len(t2))
        min_shared = max(1, max_len - 5)
        shared = len(bgs1 & bgs2)
        if shared < min_shared:
            skipped += 1
            continue
        
        checked += 1
        d = levenshtein(t1, t2)
        if d <= 2 and d > 0:
            if freq_count_map[t1] >= 5 or freq_count_map[t2] >= 5:
                near_dupes.append((t1, freq_count_map[t1], t2, freq_count_map[t2], d))

print(f"  Pairs checked (after bigram filter): {checked:,}  (skipped {skipped:,}) in {time.time()-t_lev:.1f}s")
near_dupes.sort(key=lambda x: -(x[1] + x[3]))
print(f"  Near-duplicate pairs found (dist<=2, at least one with 5+ occ): {len(near_dupes):,}")
print(f"\n  Top 50 near-duplicate pairs:")
for i, (t1, c1, t2, c2, d) in enumerate(near_dupes[:50]):
    print(f"    [{i+1:>2}] dist={d}  ({c1:>5,}x) \"{t1[:60]}\"")
    print(f"         {'':>7}  ({c2:>5,}x) \"{t2[:60]}\"")

# 3.3 Singular/plural variants
print(f"\n--- 3.3 Singular/plural variants ---")
L2_unique = list(L2_counter.keys())
L2_set = set(L2_unique)
L2_count_map = dict(L2_counter)
plural_pairs = []
seen_plural = set()
for txt in L2_unique:
    words = txt.split()
    for wi in range(len(words)):
        w = words[wi]
        if w.endswith("s") and len(w) > 2 and not w.endswith("ss"):
            new_words = words[:wi] + [w[:-1]] + words[wi+1:]
            candidate = " ".join(new_words)
            if candidate in L2_set and candidate != txt:
                pair_key = tuple(sorted([txt, candidate]))
                if pair_key not in seen_plural:
                    seen_plural.add(pair_key)
                    plural_pairs.append((txt, L2_count_map[txt], candidate, L2_count_map[candidate]))

plural_pairs.sort(key=lambda x: -(x[1] + x[3]))
print(f"  Singular/plural pairs found: {len(plural_pairs):,}")
print(f"\n  Top 40 singular/plural pairs:")
for i, (t1, c1, t2, c2) in enumerate(plural_pairs[:40]):
    print(f"    [{i+1:>2}] ({c1:>5,}x) \"{t1[:55]}\"")
    print(f"         ({c2:>5,}x) \"{t2[:55]}\"")

# 3.4 Article variants
print(f"\n--- 3.4 Article variants ---")
article_pairs = []
seen_article = set()
for txt in L2_unique:
    for art in [" the ", " a ", " an "]:
        if art in txt:
            candidate = txt.replace(art, " ", 1)
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if candidate in L2_set and candidate != txt:
                pair_key = tuple(sorted([txt, candidate]))
                if pair_key not in seen_article:
                    seen_article.add(pair_key)
                    article_pairs.append((txt, L2_count_map[txt], candidate, L2_count_map[candidate]))
    if " the " in txt:
        cand = txt.replace(" the ", " a ", 1)
        if cand in L2_set and cand != txt:
            pair_key = tuple(sorted([txt, cand]))
            if pair_key not in seen_article:
                seen_article.add(pair_key)
                article_pairs.append((txt, L2_count_map[txt], cand, L2_count_map[cand]))
    if " a " in txt:
        cand = txt.replace(" a ", " the ", 1)
        if cand in L2_set and cand != txt:
            pair_key = tuple(sorted([txt, cand]))
            if pair_key not in seen_article:
                seen_article.add(pair_key)
                article_pairs.append((txt, L2_count_map[txt], cand, L2_count_map[cand]))

article_pairs.sort(key=lambda x: -(x[1] + x[3]))
print(f"  Article variant pairs found: {len(article_pairs):,}")
print(f"\n  Top 40 article variant pairs:")
for i, (t1, c1, t2, c2) in enumerate(article_pairs[:40]):
    print(f"    [{i+1:>2}] ({c1:>5,}x) \"{t1[:60]}\"")
    print(f"         ({c2:>5,}x) \"{t2[:60]}\"")

# 3.5 Tense variants
print(f"\n--- 3.5 Tense variants ---")
verb_tense_map = defaultdict(lambda: defaultdict(int))
for txt, cnt in Counter(L3).items():
    words = txt.split()
    if not words:
        continue
    verb = words[0]
    rest = " ".join(words[1:])
    base = verb
    tense = "base"
    if verb.endswith("ing"):
        base = verb[:-3]
        if len(base) > 1 and len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        tense = "ing"
    elif verb.endswith("ed"):
        base = verb[:-2]
        tense = "ed"
    elif verb.endswith("s") and not verb.endswith("ss"):
        base = verb[:-1]
        tense = "s"
    key = (base, rest)
    verb_tense_map[key][(verb, tense)] += cnt

tense_variant_groups = {k: v for k, v in verb_tense_map.items() if len(v) > 1}
tense_sorted = sorted(tense_variant_groups.items(), key=lambda x: -sum(x[1].values()))
print(f"  Groups with multiple tense forms: {len(tense_variant_groups):,}")
print(f"\n  Top 40 tense variant groups:")
for i, ((base, rest), variants) in enumerate(tense_sorted[:40]):
    total = sum(variants.values())
    rest_short = rest[:45] if rest else "(no object)"
    print(f"    [{i+1:>2}] base=\"{base}\" + \"{rest_short}\"  total={total:,}")
    for (verb, tense), cnt in sorted(variants.items(), key=lambda x: -x[1]):
        print(f"         {verb:<20s} ({tense:<4s}) {cnt:>6,}x")

# 3.6 With/without #C C prefix
print(f"\n--- 3.6 With/without '#C C' prefix ---")
with_prefix_bodies = set()
without_prefix_set = set()
L1_set = set(L1_counter.keys())
for txt in L1_counter:
    if txt.startswith("#c c "):
        with_prefix_bodies.add(txt[5:])
    else:
        without_prefix_set.add(txt)

overlap = with_prefix_bodies & without_prefix_set
print(f"  Narrations appearing BOTH with and without '#C C' prefix: {len(overlap):,}")
if overlap:
    overlap_sorted = sorted(overlap, key=lambda x: -(L1_counter.get("#c c " + x, 0) + L1_counter.get(x, 0)))
    print(f"\n  Top 30 such narrations:")
    for i, body in enumerate(overlap_sorted[:30]):
        c_with = L1_counter.get("#c c " + body, 0)
        c_without = L1_counter.get(body, 0)
        print(f"    [{i+1:>2}] \"{body[:60]}\"")
        print(f"         with '#c c' prefix: {c_with:>6,}x    without: {c_without:>6,}x")

# 3.7 Trailing character differences
print(f"\n--- 3.7 Trailing character differences ---")
trailing_diffs = []
seen_trail = set()
for txt in L1_counter:
    if len(txt) > 1:
        candidate = txt[:-1].rstrip()
        if candidate in L1_set and candidate != txt:
            pair_key = tuple(sorted([txt, candidate]))
            if pair_key not in seen_trail:
                seen_trail.add(pair_key)
                trailing_diffs.append((txt, L1_counter[txt], candidate, L1_counter[candidate]))

trailing_diffs.sort(key=lambda x: -(x[1] + x[3]))
print(f"  Pairs differing by trailing character: {len(trailing_diffs):,}")
print(f"\n  Top 30:")
for i, (t1, c1, t2, c2) in enumerate(trailing_diffs[:30]):
    print(f"    [{i+1:>2}] ({c1:>5,}x) {repr(t1)[:65]}")
    print(f"         ({c2:>5,}x) {repr(t2)[:65]}")


# ======================================================================
# STEP 4: Verb pattern grouping
# ======================================================================
print("\n" + "=" * 90)
print("STEP 4: VERB PATTERN GROUPING")
print("=" * 90)

verb_data = []
for i, n in enumerate(narrations):
    l3 = normalize_L3(n)
    words = l3.split()
    verb = words[0] if words else ""
    verb_data.append((verb, l3, actions[i], scenarios[i]))

verb_counter = Counter(v[0] for v in verb_data)
verb_unique_narrations = defaultdict(set)
verb_action_dist = defaultdict(Counter)
for verb, l3, action, scenario in verb_data:
    verb_unique_narrations[verb].add(l3)
    verb_action_dist[verb][action] += 1

print(f"\n--- Top 50 verbs by row count ---")
print(f"  {'Verb':<25s}  {'Rows':>8s}  {'Unique':>8s}  {'Ratio':>6s}")
print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*6}")
for verb, cnt in verb_counter.most_common(50):
    uniq = len(verb_unique_narrations[verb])
    ratio = cnt / uniq if uniq > 0 else 0
    print(f"  {verb:<25s}  {cnt:>8,}  {uniq:>8,}  {ratio:>6.1f}")

print(f"\n--- Action label distribution for top 10 verbs ---")
for verb, cnt in verb_counter.most_common(10):
    print(f"\n  Verb: \"{verb}\" ({cnt:,} rows)")
    dist = verb_action_dist[verb]
    total = sum(dist.values())
    for action, acnt in dist.most_common():
        pct = acnt / total * 100
        print(f"    {action:<25s}  {acnt:>7,}  ({pct:5.1f}%)")


# ======================================================================
# STEP 5: Summary Statistics
# ======================================================================
print("\n" + "=" * 90)
print("STEP 5: SUMMARY STATISTICS")
print("=" * 90)

unsure_count = len(unsure_rows_list)

near_dupe_narrations = set()
for t1, c1, t2, c2, d in near_dupes:
    near_dupe_narrations.add(t1)
    near_dupe_narrations.add(t2)
near_dupe_rows = sum(L1_counts[n] for n in near_dupe_narrations)

estimated_truly_unique = unique_L2 - len(near_dupes)

print(f"""
  Total rows:                                {total_rows:>10,}
  Raw unique narrations (Level 0):           {unique_raw:>10,}
  After Level 1 (basic cleanup):             {unique_L1:>10,}  ({(unique_raw-unique_L1)/unique_raw*100:.1f}% reduction from raw)
  After Level 2 (remove #tags):              {unique_L2:>10,}  ({(unique_L1-unique_L2)/unique_L1*100:.1f}% reduction from L1)
  After Level 3 (remove 'C' subject):        {unique_L3:>10,}  ({(unique_L2-unique_L3)/unique_L2*100:.1f}% reduction from L2)
  Unique verb templates (Level 4):           {unique_L4:>10,}

  Narrations with #unsure/#unknown:          {unsure_count:>10,} rows
  Near-duplicate pairs (Lev<=2, freq>=5):    {len(near_dupes):>10,} pairs
  Near-duplicate rows coverage:              {near_dupe_rows:>10,} rows
  Singular/plural variant pairs:             {len(plural_pairs):>10,} pairs
  Article variant pairs:                     {len(article_pairs):>10,} pairs
  Tense variant groups:                      {len(tense_variant_groups):>10,} groups

  Estimated "truly unique" narrations:       ~{estimated_truly_unique:>9,} (L2 unique minus near-dupe pairs)
""")

print(f"--- Distribution of narration frequencies (L1 normalized) ---")
freq_dist = Counter()
for txt, cnt in L1_counter.items():
    if cnt == 1:
        freq_dist["1 (hapax)"] += 1
    elif cnt <= 5:
        freq_dist["2-5"] += 1
    elif cnt <= 10:
        freq_dist["6-10"] += 1
    elif cnt <= 50:
        freq_dist["11-50"] += 1
    elif cnt <= 100:
        freq_dist["51-100"] += 1
    elif cnt <= 500:
        freq_dist["101-500"] += 1
    else:
        freq_dist["500+"] += 1

print(f"  {'Frequency bucket':<20s}  {'Unique narrations':>20s}  {'% of unique':>12s}")
for bucket in ["1 (hapax)", "2-5", "6-10", "11-50", "51-100", "101-500", "500+"]:
    cnt = freq_dist.get(bucket, 0)
    pct = cnt / unique_L1 * 100
    print(f"  {bucket:<20s}  {cnt:>20,}  {pct:>11.1f}%")

hapax_unique = freq_dist.get("1 (hapax)", 0)
print(f"\n  Narrations appearing exactly once (hapax legomena):")
print(f"  {hapax_unique:,} / {unique_L1:,} = {hapax_unique/unique_L1*100:.1f}% of unique narrations")
print(f"  but only {hapax_unique:,} / {total_rows:,} = {hapax_unique/total_rows*100:.1f}% of total rows")

elapsed = time.time() - t0
print(f"\n{'='*90}")
print(f"Analysis complete in {elapsed:.1f}s")
print(f"{'='*90}")
