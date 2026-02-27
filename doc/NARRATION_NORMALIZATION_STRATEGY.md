# Narration Normalization Strategy

**Goal:** Reduce the validation workload by deduplicating semantically identical narrations.

## The Strategy: "No Articles" Normalization

We observed that many narrations differ only by articles ("the", "a") or punctuation, but describe the exact same action. By normalizing these, we can validate one representative instance and propagate the result to all variations.

### Normalization Logic
1. **Lowercase** (`Opens` -> `opens`)
2. **Remove Hashtags** (`#C`, `#unsure` -> ``)
3. **Remove Punctuation** (`door.` -> `door`)
4. **Remove Articles** (`the`, `a`, `an`, `some`)
5. **Strip Whitespace**

### Example Merges
| Original Narrations | Normalized | Result |
|---------------------|------------|--------|
| `#C C opens the door.` | `c opens door` | **Merged** |
| `#C C Opens a door` | `c opens door` | **Merged** |
| `opens door` | `c opens door` | **Merged** |

## Alternative Strategy Considered: "Nouns & Verbs Only"

We also evaluated a more aggressive strategy using NLP to keep only nouns and verbs (e.g., "walks out of the room" -> "walk room").

- **Potential Reduction:** ~33.5% (95k unique rows).
- **Why Rejected:** It was too aggressive. It merged distinct actions with different meanings, losing critical directional context.
    - *Example:* "walks **to** the room" and "walks **out of** the room" both became "walk room".
    - *Example:* "moves **his** hand" and "moves hand **around**" both became "move hand".

We chose the "No Articles" strategy as the safer balance between efficiency and semantic preservation.

## Results

| Metric | Count |
|--------|------:|
| **Total Rows (Original)** | 355,580 |
| **Unique Raw Narrations** | 143,859 |
| **Unique "No Articles" Narrations** | **121,738** |

**Reduction:**
- We reduced the unique items to label from 143k to 121k (**~15% reduction**).
- Compared to labeling every row (355k), this is a **66% reduction**.

## Output File
**`data/labels/action_labels_llm_clean_refined_no_articles.csv`**

This file contains **121,738 rows**.
- Each row is a **real, original row** from the source dataset.
- It was selected as the *first* occurrence of that specific normalized narration group.
- It retains the original `narration_text` (with articles/punctuation intact).
