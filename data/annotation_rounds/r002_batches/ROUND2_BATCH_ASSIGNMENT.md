# Round 2 Batch Assignment (r002)

This note documents how Round 2 annotation batches were assigned.

## Goal

Assign exactly **18,000 rows** for Round 2:

- **12 batches**
- **1,500 rows per batch**
- batch labels: `B01` ... `B12`
- round label: `r002`

## Input and Output

- **Input pool CSV**
  - `data/labels/action_labels_llm_clean_refined_no_articles_round2_pool.csv`
- **Output full CSV**
  - `data/labels/action_labels_llm_clean_refined_no_articles_round2_assigned_alpha1_freq2.csv`
- **Output assigned-only CSV**
  - `data/labels/action_labels_llm_clean_refined_no_articles_round2_assigned_alpha1_freq2_assigned_only.csv`

## Assignment Logic

Script used:

- `scripts/assign_annotation_batches.py`

Main settings:

- `--num-batches 12`
- `--batch-size 1500`
- `--round-id r002`
- `--alpha 1.0` (proportional class allocation)
- `--prioritize-frequent`
- `--frequency-key nouns_verbs` (aggressive grouping by verb+noun lemmas)
- `--min-narration-frequency 2`

Interpretation:

1. Start from unassigned active-class rows.
2. Build narration frequency groups using aggressive `nouns_verbs` key.
3. Keep rows with frequency >= 2. Was not working for >=3
4. Compute class targets per batch (proportional, `alpha=1.0`).
5. Assign videos to batches (greedy, Search-aware).
6. Sample rows per class per batch, prioritizing high-frequency narrations first.
7. Write `batch` and `round` labels to output files.

## Exact Command

```bash
python scripts/assign_annotation_batches.py \
  --labels-csv data/labels/action_labels_llm_clean_refined_no_articles_round2_pool.csv \
  --num-batches 12 \
  --batch-size 1500 \
  --alpha 1.0 \
  --round-id r002 \
  --seed 2026 \
  --prioritize-frequent \
  --frequency-key nouns_verbs \
  --min-narration-frequency 2 \
  --output-csv data/labels/action_labels_llm_clean_refined_no_articles_round2_assigned_alpha1_freq2.csv
```

## Final Result

- Total assigned: **18,000**
- Total unassigned (in output full CSV): **94,631**
- Assigned-only file has **18,000 rows** (+ header line in CSV viewers).

