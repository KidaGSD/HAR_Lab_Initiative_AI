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

# ===========================
# 2. WandB Setup
# ===========================
echo ""
echo "=== Step 2: WandB Configuration ==="

export WANDB_API_KEY=e83326e014ad7a27c2a538f4e38b95bd11a161a0
wandb login $WANDB_API_KEY
echo "WandB logged in successfully"

# ===========================
# 3. GPU Configuration
# ===========================
echo ""
echo "=== Step 3: GPU Configuration ==="

# Use a single GPU (adjust index as needed)
export CUDA_VISIBLE_DEVICES=6
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

python scripts/train_hierarchical.py \
    --cv \
    --n-folds 2 \
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
# 6. Action Probing (Optional)
# ===========================
echo "=== Step 6: Action Probing ==="

read -p "Do you want to train action probe? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Training action probe on frozen embeddings..."
    
    # Find best checkpoint from CV
    BEST_CHECKPOINT=$(ls -t $OUTPUT_DIR/best_model.pth 2>/dev/null | head -1)
    
    if [ -z "$BEST_CHECKPOINT" ]; then
        echo "Warning: No checkpoint found in $OUTPUT_DIR"
        echo "Please specify checkpoint path:"
        read -p "Checkpoint path: " BEST_CHECKPOINT
    fi
    
    echo "Using checkpoint: $BEST_CHECKPOINT"
    
    python scripts/train_hierarchical.py \
        --probe \
        --checkpoint "$BEST_CHECKPOINT" \
        --processed-dir "$PROCESSED_DIR" \
        --output-dir "${OUTPUT_DIR}_probe" \
        --run-name "${RUN_NAME}_probe" \
        2>&1 | tee training_probe.log
    
    echo "Probe training complete!"
else
    echo "Skipping probe training"
fi

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
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  - Probe checkpoints: ${OUTPUT_DIR}_probe"
    echo "  - Probe logs: training_probe.log"
fi
echo ""
echo "WandB Dashboard: https://wandb.ai/wandbleo/har-imu-training"
echo ""
echo "Next steps:"
echo "  1. Review WandB dashboard for CV results"
echo "  2. Compare fold performances"
echo "  3. Analyze confusion matrices"
echo "  4. Run final test set evaluation (if needed)"
echo ""
