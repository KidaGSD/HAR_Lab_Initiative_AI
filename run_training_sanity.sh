#!/bin/bash
# Single-split sanity training (no CV) for quick validation
# Usage: ./run_training_sanity.sh

set -e

echo "=========================================="
echo "  Hierarchical HAR Sanity Train (No CV)"
echo "=========================================="

# 1. Environment
if [ -n "$CONDA_PREFIX" ]; then
    echo "Conda already activated: $CONDA_PREFIX"
else
    echo "Activating ego4d_lab environment..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ego4d_lab
fi

# 2. WandB (optional)
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set; training will skip WandB logging if unavailable."
fi

# 3. GPU
export CUDA_VISIBLE_DEVICES=6
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# 4. Config
PROCESSED_DIR="data/processed_ego4d"
OUTPUT_DIR="checkpoints_sanity"
RUN_NAME="sanity_$(date +%Y%m%d_%H%M%S)"

echo "Processed data dir: $PROCESSED_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Run name: $RUN_NAME"

echo "Starting single-split training (train/val from scenario_labels.csv)" 
python scripts/train_hierarchical.py \
    --processed-dir "$PROCESSED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_NAME" \
    "$@" \
    2>&1 | tee training_sanity.log

echo "Sanity training complete. Check $OUTPUT_DIR and training_sanity.log"
