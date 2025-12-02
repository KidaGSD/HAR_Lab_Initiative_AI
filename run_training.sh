#!/bin/bash
# Complete Training Pipeline for Hierarchical HAR
# Usage: ./run_training.sh

set -e

echo "=========================================="
echo "  Hierarchical HAR Training Pipeline"
echo "=========================================="

# ===========================
# 1. Environment Setup
# ===========================
echo ""
echo "=== Step 1: Environment Setup ==="

# Activate conda environment
if [ -n "$CONDA_PREFIX" ]; then
    echo "Conda already activated: $CONDA_PREFIX"
else
    echo "Activating ego4d_lab environment..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ego4d_lab
fi

# Ensure local src is importable
export PYTHONPATH="$(pwd):$PYTHONPATH"

# ===========================
# 2. WandB Setup
# ===========================
echo ""
echo "=== Step 2: WandB Configuration ==="

export WANDB_API_KEY=e83326e014ad7a27c2a538f4e38b95bd11a161a0
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set; will proceed without WandB logging if unavailable."
else
    wandb login $WANDB_API_KEY || echo "WandB login failed, continuing without WandB."
    echo "WandB login attempted"
fi

# ===========================
# 3. GPU Configuration
# ===========================
echo ""
echo "=== Step 3: GPU Configuration ==="

# Use a single GPU; auto-pick the freest meeting a min free threshold if not set
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    MIN_FREE_MB=${MIN_FREE_MB:-20000}
    PICKED=""
    while IFS=, read -r FREE IDX; do
        FREE=$(echo "$FREE" | xargs)
        IDX=$(echo "$IDX" | xargs)
        if [ "$FREE" -ge "$MIN_FREE_MB" ]; then
            PICKED=$IDX
            break
        fi
    done < <(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits | sort -nr)
    if [ -z "$PICKED" ]; then
        PICKED=$(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits | sort -nr | head -1 | awk -F',' '{print $2}' | xargs)
        echo "Warning: no GPU meets MIN_FREE_MB=${MIN_FREE_MB}MB, picking best available: $PICKED"
    fi
    export CUDA_VISIBLE_DEVICES=${PICKED:-0}
fi
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Display GPU info
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# ===========================
# 4. Training Configuration
# ===========================
echo "=== Step 4: Training Configuration ==="

PROCESSED_DIR="data/processed_ego4d"
OUTPUT_DIR="checkpoints"
RUN_NAME="hierarchical_cv_$(date +%Y%m%d_%H%M%S)"

echo "Processed data dir: $PROCESSED_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Run name: $RUN_NAME"
echo ""

# ===========================
# 5. Main Training (CV Mode)
# ===========================
echo "=== Step 5: Cross Validation Training ==="
echo "This will run 4-fold CV (~8-10 hours)"
echo ""

python train.py \
    --cv \
    --n-folds 4 \
    --processed-dir "$PROCESSED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_NAME" \
    2>&1 | tee training_cv.log

echo ""
echo "CV Training complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Logs saved to: training_cv.log"
echo ""

# ===========================
# 7. Summary
# ===========================
echo ""
echo "=========================================="
echo "  Training Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - CV checkpoints: $OUTPUT_DIR"
echo "  - CV logs: training_cv.log"
echo ""
echo "WandB Dashboard: https://wandb.ai/wandbleo/har-imu-training"
echo ""
echo "Next steps:"
echo "  1. Review WandB dashboard for CV results"
echo "  2. Compare fold performances"
echo "  3. Analyze confusion matrices"
echo "  4. Run action probe (if needed) via: python scripts/train_entry.py --probe --checkpoint <best_model.pth> --processed-dir $PROCESSED_DIR --output-dir ${OUTPUT_DIR}_probe --run-name ${RUN_NAME}_probe"
echo ""
