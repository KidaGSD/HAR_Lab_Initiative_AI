#!/bin/bash
# =============================================================================
# BASELINE MODELS TRAINING SCRIPT
# Trains all 4 baseline models sequentially
# =============================================================================
#
# Usage:
#   ./run_baselines.sh           # Default GPU 0
#   ./run_baselines.sh 2         # Use GPU 2
#
# =============================================================================

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="checkpoints/baselines_${TIMESTAMP}"
GPU=${1:-0}

echo "==========================================="
echo "  Baseline Models Training"
echo "==========================================="
echo "Started: $(date)"
echo "GPU: $GPU"
echo "Output: $OUTPUT_DIR"
echo ""

# Setup
export PYTHONPATH="$(pwd):$PYTHONPATH"
export WANDB_API_KEY="${WANDB_API_KEY:-e83326e014ad7a27c2a538f4e38b95bd11a161a0}"
export WANDB_RUN_GROUP="baselines_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

# Train all baselines
MODELS=("mlp_mlp" "cnn_mlp" "imu2clip" "cnn_lstm_gru")

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo ">> Training: $MODEL"
    echo "-------------------------------------------"
    
    CUDA_VISIBLE_DEVICES=$GPU python scripts/train_baselines.py \
        --model "$MODEL" \
        --config configs/beta_0.3.yaml \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "${OUTPUT_DIR}/${MODEL}.log"
    
    if [ $? -eq 0 ]; then
        echo "✓ $MODEL: SUCCESS"
    else
        echo "✗ $MODEL: FAILED"
    fi
done

echo ""
echo "==========================================="
echo "  COMPLETE"
echo "==========================================="
echo "Finished: $(date)"
echo "Results: $OUTPUT_DIR"
