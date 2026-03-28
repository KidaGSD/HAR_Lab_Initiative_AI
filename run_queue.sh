#!/usr/bin/env bash
set -euo pipefail

echo "Starting command queue at $(date)"

# --- Put your commands below, one per line ---
python scripts/train_hierarchical.py --run-name "Baseline_355_beta_1" --output-dir "checkpoints/checkpoints/new_runs" --exclude-scenarios "Gardening" --cache-dir "data/cache/train_hierarchical"
python scripts/train_hierarchical.py --run-name "Baseline_355_beta_1_CV" --output-dir "checkpoints/checkpoints/new_runs" --exclude-scenarios "Gardening" --cv --n-folds 4 --cache-dir "data/cache/train_hierarchical"
#CUDA_VISIBLE_DEVICES=0,1 BATCH_SIZE=512 python scripts/train_hierarchical.py --run-name "run2" --output-dir "checkpoints/run2" --exclude-scenarios "Gardening"


echo "Finished command queue at $(date)"