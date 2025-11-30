#!/bin/bash
# Training script for GPU server
# Usage: ./run_training.sh

set -e

echo "=== Starting Hierarchical IMU Training ==="

# OPTIMIZED: Use single GPU (much faster than DataParallel with small model)
export CUDA_VISIBLE_DEVICES=6  # Use GPU 6 (or change to 7)

echo "GPU Configuration: Using GPU ${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Activate environment
source ego4d_lab/bin/activate  # or: conda activate ego4d_lab

# Run training with optimized settings
python scripts/train_hierarchical.py \
    --target-uids-file target_uids.csv \
    --processed-dir data/processed_ego4d \
    --output-dir checkpoints \
    --run-name egocharm_optimized

echo "=== Training Complete ==="
echo "Check checkpoints/best_model.pth for best model"
echo "Check wandb for training metrics"
# Script to run training on GPUs 6 and 7
# Usage: ./run_training.sh

echo "Starting training on GPUs 6 and 7..."
echo "Ideally, run this inside a tmux session!"

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=6,7

# Run training
# Using nohup is optional if inside tmux, but good for safety
python scripts/train_hierarchical.py --run-name final_run_v1
