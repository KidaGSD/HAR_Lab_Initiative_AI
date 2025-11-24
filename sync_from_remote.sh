#!/bin/bash
# Sync results back from remote GPU server
# Usage: ./sync_from_remote.sh

REMOTE="leopolddas@10.100.241.227"
REMOTE_DIR="~/HAR_Lab_Initiative_AI"
LOCAL_DIR="."

echo "🔄 Syncing results from remote server..."
echo "   Remote: $REMOTE:$REMOTE_DIR"
echo ""

# Sync runs directory (models, checkpoints)
if ssh $REMOTE "[ -d $REMOTE_DIR/runs ]"; then
    echo "📦 Syncing runs/ (models, checkpoints)..."
    rsync -avz --progress $REMOTE:$REMOTE_DIR/runs/ $LOCAL_DIR/runs/
fi

# Sync reports if they exist
if ssh $REMOTE "[ -d $REMOTE_DIR/reports ]"; then
    echo "📊 Syncing reports/..."
    rsync -avz --progress $REMOTE:$REMOTE_DIR/reports/ $LOCAL_DIR/reports/ 2>/dev/null || true
fi

# Sync wandb logs if they exist
if ssh $REMOTE "[ -d $REMOTE_DIR/wandb ]"; then
    echo "📈 Syncing wandb/..."
    rsync -avz --progress $REMOTE:$REMOTE_DIR/wandb/ $LOCAL_DIR/wandb/ 2>/dev/null || true
fi

echo ""
echo "✅ Sync complete!"
echo ""
echo "Results are now available locally!"

