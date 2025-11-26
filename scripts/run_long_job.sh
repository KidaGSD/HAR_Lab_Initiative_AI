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
  # 1. Activate environment (if needed, assuming conda is auto-activated or in .bashrc)
  # tmux send-keys -t $SESSION_NAME "conda activate ego4d_lab" C-m
  
  # 2. Run the script
  # You can change --gpus to 2 or 4 if you have multiple GPUs!
  # You can change --limit to test, or remove it for full run.
  tmux send-keys -t $SESSION_NAME "python scripts/label_with_qwen.py --gpus 2" C-m
  
  echo "Job started in background!"
  echo "To view the progress, run: tmux attach -t $SESSION_NAME"
  echo "To detach again (leave it running), press: Ctrl+B, then D"
  
else
  echo "Session $SESSION_NAME already exists."
  echo "Attach to it with: tmux attach -t $SESSION_NAME"
fi
