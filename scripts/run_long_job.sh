#!/bin/bash
# Script to run the labeling job in a detached tmux session
# This ensures the job keeps running even if you close the terminal.

SESSION_NAME="qwen_labeling"

# Check if session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
  # Create new session
  echo "Starting new tmux session: $SESSION_NAME"
  tmux new-session -d -s $SESSION_NAME
  
  # Send commands to the session
  # 1. Activate environment
  tmux send-keys -t $SESSION_NAME "conda activate ego4d_lab" C-m
  
  # 2. Run the script
  # We explicitly target GPUs 2 and 3 because 0 and 1 are busy
  tmux send-keys -t $SESSION_NAME "CUDA_VISIBLE_DEVICES=6,7 python scripts/label_with_qwen.py --gpus 2" C-m
  
  echo "Job started in background!"
  echo "To view the progress, run: tmux attach -t $SESSION_NAME"
  echo "To detach again (leave it running), press: Ctrl+B, then D"
  
else
  echo "Session $SESSION_NAME already exists."
  echo "Attach to it with: tmux attach -t $SESSION_NAME"
fi
