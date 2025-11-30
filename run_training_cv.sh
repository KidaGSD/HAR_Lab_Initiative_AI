#!/bin/bash
# Training script for 4-Fold Cross Validation
# Usage: ./run_training_cv.sh

set -e

echo "=== Starting Hierarchical IMU Training (4-Fold CV) ==="

# 1. Setup Environment
# --------------------
# Ensure we are in a tmux session (optional but recommended)
if [ -z "$TMUX" ]; then
    echo "WARNING: You are not in a tmux session."
    echo "Training takes ~10 hours. It is HIGHLY recommended to run this in tmux."
    echo "To start tmux: tmux new -s training"
    echo "Press Ctrl+C to cancel, or Enter to continue anyway..."
    read
fi

# Activate environment (adjust path if needed)
# source activate ego4d_lab 
# OR
# conda activate ego4d_lab

# 2. WandB Setup
# --------------
echo "Logging into WandB..."
export WANDB_API_KEY=e83326e014ad7a27c2a538f4e38b95bd11a161a0
wandb login $WANDB_API_KEY

# 3. GPU Configuration
# --------------------
# Use Single GPU for faster iteration (DataParallel has overhead for small models)
export CUDA_VISIBLE_DEVICES=6
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv

# 4. Run Training (CV)
# --------------------
echo "Starting 4-Fold Cross Validation..."
echo "Log file: training_cv.log"

python scripts/train_hierarchical_cv.py \
    --processed-dir data/processed_ego4d \
    --n-folds 4 \
    --output-dir cv_results \
    2>&1 | tee training_cv.log

echo "=== Training Complete ==="
echo "Results saved to cv_results/"
