#!/bin/bash
# Sync local code to remote GPU server
# Usage: ./sync_to_remote.sh

REMOTE="leopolddas@10.100.241.227"
REMOTE_DIR="~/HAR_Lab_Initiative_AI"
LOCAL_DIR="."

echo "🔄 Syncing local code to remote server..."
echo "   Remote: $REMOTE:$REMOTE_DIR"
echo ""

rsync -avz --progress \
    --exclude '*.ipynb_checkpoints' \
    --exclude '__pycache__' \
    --exclude '.git' \
    --exclude 'runs/' \
    --exclude 'data/raw/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude 'wandb/' \
    $LOCAL_DIR $REMOTE:$REMOTE_DIR

echo ""
echo "✅ Sync complete!"
echo ""
echo "Next steps:"
echo "  1. SSH to server: ssh $REMOTE"
echo "  2. cd ~/HAR_Lab_Initiative_AI"
echo "  3. Run your training!"

