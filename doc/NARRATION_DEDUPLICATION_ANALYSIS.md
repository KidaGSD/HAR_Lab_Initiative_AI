# HAR Dataset Narration Deduplication Analysis

**Date:** 2026-02-22
**Dataset:** `action_labels_llm_clean_refined.csv` (355,580 rows, 1,491 videos)
**Validated sample:** 1,165 rows from first annotation session

---

## 1. Executive Summary

The 355K-row HAR dataset contains massive redundancy in narration text. After systematic analysis, we find:

| Metric | Count | % of 355K |
|--------|------:|----------:|
| Total rows | 355,580 | 100% |
| Raw unique `narration_text` | 143,860 | 40.5% |
| After basic normalization (case, punctuation) | 136,270 | 38.3% |
| After removing `#tags` | 135,963 | 38.2% |
| After removing subject prefix | 135,586 | 38.1% |
| Estimated truly unique (after near-duplicate removal) | ~132,000 | ~37.1% |

**Key finding:** ~62% of all rows are duplicates or near-duplicates of another row. For validation purposes, checking ~132K unique narrations covers the entire 355K dataset.

**For the 30K assigned validation set:**
- 20,126 unique (narration, action) pairs out of 30,000 rows
- ~10,000 rows (33%) are redundant
- Time savings: ~14 hours at 5 sec/task

---

## 2. Sources of Duplication

### 2.1 Exact Duplicates (211,721 rows = 59.5%)

Many narrations appear identically across different videos. The same action described the same way by different Ego4D annotators.

**Top 10 most repeated narrations:**

| Narration | Occurrences | # Action Labels |
|-----------|------------:|:---------------:|
| `#C C looks around` | 4,580 | 2 |
| `#C C walks around` | 2,093 | 1 |
| `#C C walks` | 1,411 | 1 |
| `#C C walks around the kitchen` | 1,035 | 1 |
| `#C C moves around` | 938 | 1 |
| `#C C turns around` | 746 | 1 |
| `#C C closes the tap` | 568 | 1 |
| `#C C opens the tap` | 564 | 1 |
| `#C C walks in the kitchen` | 547 | 1 |
| `#C C walks in the house` | 524 | 1 |

**Consistency:** Among 41,441 narrations appearing 2+ times, **91.9% have consistent LLM action labels** (same class every time). Only 3,360 narrations (8.1%) have conflicting labels across occurrences, affecting 44,432 rows.

### 2.2 Case and Punctuation Variants (7,590 narrations merged)

The same narration appears in multiple capitalization/punctuation forms:

```
"#C C closes the fridge"     (7 raw variants)
  - '#C C Closes the fridge'
  - '#C C closes the fridge'
  - '#C C closes the fridge .'
  - '#C C closes the fridge,'
  - '#C C closes the fridge.'
  - '#c C closes the fridge'
  - '#c c closes the fridge'
```

**Scale:** 7,590 unique narrations eliminated by basic normalization (lowercase + strip punctuation). This is 5.3% of all unique narrations.

**Trailing punctuation distribution:**
- 274,922 rows (77.3%): no trailing punctuation
- 80,556 rows (22.7%): trailing period
- 96 rows (0.0%): trailing comma

### 2.3 Hashtag Tag Variations (15,146 rows = 4.3%)

The Ego4D narrations use hashtag markers that create artificial variation:

**Subject markers (interchangeable):**

| Tag | Rows | Meaning |
|-----|-----:|---------|
| `#C` | 304,765 | Camera person (standard) |
| `#c` | 10,976 | Camera person (lowercase variant) |
| `#CC` | 624 | Camera person (malformed double) |
| `#cc` | 153 | Camera person (lowercase malformed) |
| `#cC`, `#Cc` | 6 | Camera person (mixed case) |
| `#O` | 35,714 | Other person |
| `#o` | 410 | Other person (lowercase) |
| `#OO` | 120 | Other person (malformed) |

**Uncertainty markers:**

| Tag | Rows | Note |
|-----|-----:|------|
| `#unsure` | 11,316 | Standard uncertainty tag |
| `#Unsure` | 3,809 | Capitalized variant |
| `#UNSURE` | 32 | All-caps variant |
| `#unknown` | 4 | Alternative tag |
| `#unsured` | 3 | Typo |
| `#Unssure` | 1 | Typo |
| `#ub=nsure` | 1 | Typo |
| `#unsurefrom` | 1 | Missing space |
| `#unsurewith` | 1 | Missing space |

**Merging examples when removing tags:**
```
"C picks"  <-- merges from:
  "#C C picks"
  "#C C picks #unknown"
  "#C C picks #unsure"
  "#c c picks #unsure"
```

**307 additional unique narrations** merged by removing hashtag tags.

### 2.4 Article Variants (13,658 pairs -- LARGEST source)

The single biggest source of near-duplication. The same action with `the`, `a`, or no article:

```
"C opens the tap"   (708 rows)
"C opens tap"       (237 rows)
"C opens a tap"     (196 rows)
```

```
"C opens the door"  (515 rows)
"C opens a door"    (234 rows)
"C opens door"      (150 rows)
```

```
"C walks in the room"   (677 rows)
"C walks in a room"     (106 rows)
"C walks in room"       (  3 rows)
```

**Doubled articles (typos):**
```
"C walks around the the kitchen"  (1 row)
"C closes the the tap"           (2 rows)
"C opens the the tap"            (1 row)
```

### 2.5 Singular/Plural Variants (3,876 pairs)

Object nouns appear in both singular and plural forms for the same action:

| Singular | Count | Plural | Count |
|----------|------:|--------|------:|
| `puts the card down` | 459 | `puts the cards down` | 57 |
| `moves hand` | 331 | `moves hands` | 228 |
| `picks card` | 203 | `picks cards` | 129 |
| `picks the card` | 114 | `picks the cards` | 169 |
| `washes hand` | 40 | `washes hands` | 190 |
| `picks a cloth` | 156 | `picks a clothe` | 40 |

### 2.6 Tense/Conjugation Variants (678 groups)

Most narrations use 3rd-person present (`picks`, `walks`), but bare infinitives and occasional past tense exist:

| Base Form | 3rd Person | Count | Bare Infinitive | Count | Past | Count |
|-----------|-----------|------:|-----------------|------:|------|------:|
| walk around | walks | 2,384 | walk | 197 | - | - |
| close the tap | closes | 718 | close | 50 | - | - |
| open the tap | opens | 709 | open | 58 | - | - |
| stand up | stands | 394 | stand | 27 | - | - |
| bend down | bends | 199 | bend | 18 | - | - |
| walk on the floor | walks | 444 | walk | 2 | walked | 1 |

### 2.7 Typo and Spacing Errors (3,691 near-duplicate pairs)

Levenshtein distance analysis (edit distance <= 2) reveals:

**Space insertion:**
```
"C looks around"   (5,105 rows)
"C looks a round"  (  161 rows)   <-- space inserted in "around"
```

**Missing third-person -s:**
```
"C walks around"  (2,381 rows)
"C walk around"   (  197 rows)
```

**Missing space in prefix:**
```
"# c c walks around"  (46 rows)   <-- extra space after #
"#cc walks around"    ( 4 rows)   <-- missing space between #C and C
```

**Contraction typos:**
```
"C walk's in the kitchen"  (5 rows)   <-- incorrect apostrophe
"C walks in the kitchen"   (615 rows)
```

**Malformed tags (missing spaces):**
```
"#OOpoints at the park"     <-- should be "#OO points"
"#CCtalks to #OO"           <-- should be "#CC talks"
"#OOstands sits again down" <-- should be "#OO stands"
```

### 2.8 Semantically False Near-Duplicates

Some Levenshtein-close pairs are actually **different actions** and must NOT be merged:

```
"C walks"  (1,799 rows) vs "C talks"  (310 rows)   -- dist=1, but different!
"C opens the tap" (708) vs "C opens the tin" (52)   -- dist=2, different object
"C opens the tap" (708) vs "C opens the bag" (40)   -- dist=2, different object
"C opens the tap" (708) vs "C opens the cap" (7)    -- dist=1, different object
```

**Implication:** Automated deduplication by edit distance alone is dangerous. Only merge when the difference is a tag, article, punctuation, or case -- not when it changes the verb or object.

---

## 3. Verb-Level Analysis

### 3.1 Action Label Consistency by Verb

Most verbs map strongly to a single action class:

| Verb | Rows | Dominant Class | % |
|------|-----:|---------------|--:|
| picks | 37,133 | Object Transfer | 98.2% |
| puts | 28,189 | Object Transfer | 93.2% |
| walks | 18,647 | Locomotion | 99.9% |
| holds | 12,666 | Stationary | 97.2% |
| drops | 12,097 | Object Transfer | 96.4% |
| places | 11,346 | Object Transfer | 90.6% |
| opens | 9,974 | Object Transfer | 96.1% |
| closes | 6,280 | Object Transfer | 95.5% |
| stares | 2,707 | Search | 94.5% |
| scrolls | 1,679 | Stationary | 95.8% |

### 3.2 Ambiguous Verbs

Two verbs have genuinely split action distributions:

**`moves` (13,163 rows) -- most ambiguous verb:**

| Action | Count | % |
|--------|------:|--:|
| Locomotion | 4,931 | 37.5% |
| Object Transfer | 4,086 | 31.0% |
| Stationary | 2,613 | 19.9% |
| Essential Operation | 1,430 | 10.9% |
| Search | 103 | 0.8% |

**`looks` (13,993 rows) -- Search vs Stationary split:**

| Action | Count | % |
|--------|------:|--:|
| Search | 12,144 | 86.8% |
| Stationary | 1,676 | 12.0% |
| Locomotion | 167 | 1.2% |

The `looks` split is a known issue -- `looks at phone` is often labeled Stationary rather than Search, while `looks around` is consistently Search.

### 3.3 Verb Repetitiveness

Some verbs are described in very few ways (high rows-to-unique ratio):

| Verb | Rows | Unique Narrations | Ratio |
|------|-----:|------------------:|------:|
| talks | 1,251 | 108 | 11.6x |
| scrolls | 1,679 | 164 | 10.2x |
| paints | 1,295 | 141 | 9.2x |
| looks | 13,993 | 1,554 | 9.0x |
| uses | 1,429 | 170 | 8.4x |
| walks | 18,647 | 2,359 | 7.9x |
| plays | 2,302 | 308 | 7.5x |

These high-ratio verbs benefit most from deduplication -- a small number of validated narrations covers many rows.

---

## 4. Validated Sample Analysis (1,165 rows)

### 4.1 Coverage

| Metric | Value |
|--------|------:|
| Validated rows | 1,165 |
| Unique narrations validated | 858 |
| Rows in full dataset these cover | 31,059 (8.7%) |
| Narrations with consistent labels | 765 (auto-covers 18,024 rows) |
| Narrations with inconsistent labels | 93 (affects 13,035 rows) |

### 4.2 Propagation Potential

If a validated narration has consistent LLM labels across the full dataset, validating one instance validates all. The 765 consistent narrations from the first 1,165 validated rows could propagate to cover 18,024 rows -- a **15.5x multiplier**.

The 93 inconsistent narrations (where the same text got different LLM labels in different rows) need special treatment -- these represent genuine ambiguity that should be resolved in annotation guidelines.

---

## 5. Recommended Normalization Pipeline

### Safe transformations (will not change meaning):

```
Level 1: lowercase + strip whitespace + remove trailing punctuation
Level 2: remove all #hashtag tokens (#C, #unsure, #O, etc.)
Level 3: remove subject prefix "c " (after tag removal)
Level 4: collapse doubled articles ("the the" -> "the")
```

### Risky transformations (may merge semantically different narrations):

```
- Article normalization (the/a/no article) -- SAFE for validation
- Singular/plural normalization -- SAFE for validation
- Tense normalization -- SAFE for validation
- Levenshtein merging -- DANGEROUS without manual review
```

### Estimated unique narrations after safe normalization:

```
Raw:                143,860
After safe (L1-L4): ~135,000
After article norm: ~125,000
After sing/plural:  ~122,000
After tense norm:   ~121,000
```

---

## 6. Impact on Validation Strategy

### 6.1 Current Approach (validate all 30K assigned rows)

- Time: ~41.7 hours (at 5 sec/task)
- Redundancy: ~33% of rows are duplicates within the assigned set
- Annotators see the same narration repeatedly across different videos

### 6.2 Proposed Approach (validate unique narration-action pairs)

**Step 1:** Normalize narrations (Levels 1-3)
**Step 2:** Group by (normalized_narration, action_label)
**Step 3:** Validate one representative per group
**Step 4:** Propagate validation result to all rows in the group

**For the full dataset:**

| Approach | Unique Items | Time (5s each) | Coverage |
|----------|------------:|---------------:|:--------:|
| All rows | 355,580 | 493.9 hrs | 100% |
| Unique narrations | 143,860 | 199.8 hrs | 100% |
| Unique (narration, action) pairs | 147,471 | 204.8 hrs | 100% |
| After normalization | ~121,000 | ~168 hrs | 100% |

**For the 30K assigned set:**

| Approach | Unique Items | Time | Savings |
|----------|------------:|-----:|--------:|
| All rows | 30,000 | 41.7 hrs | -- |
| Unique (narr, action) pairs | 20,126 | 28.0 hrs | 33% |
| After normalization | ~17,000 | ~23.6 hrs | 43% |

### 6.3 Priority Ordering for Validation

**Highest priority (validate first):**
1. Narrations with **inconsistent LLM labels** across occurrences (3,360 unique narrations, 44,432 rows) -- these reveal LLM confusion
2. Narrations with **ambiguous verbs** (`moves`, `looks`) -- highest error rate expected

**Medium priority:**
3. High-frequency narrations (100+ occurrences) -- each validation covers many rows
4. Narrations with `#unsure` tags -- uncertain object identity may affect classification

**Low priority (validate last or skip):**
5. Hapax legomena (narrations appearing exactly once) -- 96,347 unique narrations covering only 27.1% of rows. Low impact per validation effort.

---

## 7. Data Quality Issues Found

### 7.1 Malformed Narrations

| Issue | Examples | Count |
|-------|---------|------:|
| Missing space after tag | `#OOpoints at the park`, `#CCtalks to #OO` | ~20 |
| Typo in tag | `#unsured`, `#Unssure`, `#ub=nsure` | ~6 |
| Doubled articles | `the the kitchen`, `the the tap` | ~5 |
| Incorrect apostrophe | `walk's`, `hold's` | ~10 |
| Special characters | `@`, `$`, `+`, `=`, `\|` | ~30 |
| Empty object after verb | `#C C picks`, `#C C puts`, `#C C opens` | ~1,500 |

### 7.2 `#Summary` Rows (22 rows)

22 rows contain `#Summary` tags with paragraph-level descriptions rather than action narrations:
```
"#Summary c was in the kitchen. C walked out of the kitchen, poured salt on fish..."
```
These should be excluded from action classification.

### 7.3 Subject Ambiguity

The `#O` prefix (35,714 rows, 10%) describes actions of **other people**, not the camera wearer. These narrations use person-specific identifiers (`man K`, `lady Z`, `person X`) that add variation without changing the action semantics.

---

## 8. Appendix: Frequency Distribution

```
Frequency   Unique Narrations   % of Unique   Cumulative Rows
1 (hapax)          96,347          70.7%           96,347 (27.1%)
2-5                30,855          22.6%          ~170,000 (47.8%)
6-10                5,023           3.7%          ~210,000 (59.1%)
11-50               3,551           2.6%          ~280,000 (78.7%)
51-100                327           0.2%          ~305,000 (85.8%)
101-500               152           0.1%          ~335,000 (94.2%)
500+                   15           0.0%          ~355,580 (100%)
```

The top 15 narrations (500+ occurrences each) alone account for ~20,000 rows (5.6% of the dataset). Validating just these 15 covers more rows than validating 5,000 hapax narrations.
