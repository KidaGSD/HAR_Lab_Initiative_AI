#!/bin/bash
# Script to run training on GPUs 6 and 7
# Usage: ./run_training.sh

echo "Starting training on GPUs 6 and 7..."
echo "Ideally, run this inside a tmux session!"

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=6,7

# Run training
# Using nohup is optional if inside tmux, but good for safety
python scripts/train_hierarchical.py --run-name final_run_v1
