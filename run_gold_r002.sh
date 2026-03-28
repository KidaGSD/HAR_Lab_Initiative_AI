#!/usr/bin/env bash
set -euo pipefail

# Gold labels from Label Studio refined export.
GOLD_CSV="data/annotation_rounds/r002_gold_exports/r002_refined_combined/r002_refined.csv"
PROCESSED_DIR="data/processed_ego4d"
SCENARIO_CSV="data/labels/scenario_labels.csv"
OUT_ROOT="checkpoints/checkpoints/gold_r002"

# Set this to your previously trained best model for fine-tuning.
# Example:
# PRETRAINED_BEST="checkpoints/checkpoints/experiments_20251206_224347/beta_1.0/best_model.pth"
PRETRAINED_BEST="${PRETRAINED_BEST:-}"

echo "=== 1) Train from scratch on R002 gold action labels ==="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python scripts/train_hierarchical.py \
  --run-name "r002_gold_scratch" \
  --processed-dir "${PROCESSED_DIR}" \
  --scenario-labels-csv "${SCENARIO_CSV}" \
  --action-labels-csv "${GOLD_CSV}" \
  --use-corrected-actions \
  --output-dir "${OUT_ROOT}/scratch"

if [[ -n "${PRETRAINED_BEST}" ]]; then
  echo "=== 2) Fine-tune from previous best on R002 gold ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python scripts/train_hierarchical.py \
    --run-name "r002_gold_finetune" \
    --processed-dir "${PROCESSED_DIR}" \
    --scenario-labels-csv "${SCENARIO_CSV}" \
    --action-labels-csv "${GOLD_CSV}" \
    --use-corrected-actions \
    --init-checkpoint "${PRETRAINED_BEST}" \
    --output-dir "${OUT_ROOT}/finetune"
else
  echo "Skipping fine-tune: set PRETRAINED_BEST to your old best_model.pth path."
fi
