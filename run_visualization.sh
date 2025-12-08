#!/bin/bash
# run_visualization.sh - Run embedding visualization on server

set -e

echo "🎨 Running Embedding Visualization on Server"
echo "=============================================="

# Configuration
MODEL_NAME="beta_1.0"
CHECKPOINT="checkpoints/checkpoints/experiments_20251206_224347/beta_1.0/best_model.pth"
CONFIG="configs/hierarchical.yaml"
OUTPUT_DIR="outputs/embeddings_viz"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found at $CHECKPOINT"
    echo "Available checkpoints:"
    find checkpoints/checkpoints/experiments_20251206_224347 -name "best_model.pth"
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

echo ""
echo "📊 Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Checkpoint: $CHECKPOINT"
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run PCA visualization (fast)
echo "1️⃣  Generating PCA plots (fast)..."
python scripts/visualize_embeddings.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --split val \
    --output-dir $OUTPUT_DIR

echo ""
echo "2️⃣  Generating t-SNE plots (slower, prettier)..."
python scripts/visualize_embeddings.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --split val \
    --use-tsne \
    --output-dir $OUTPUT_DIR

echo ""
echo "✅ Visualization complete!"
echo ""
echo "📁 Output files in: $OUTPUT_DIR/"
ls -lh $OUTPUT_DIR/

echo ""
echo "💡 To download to local machine:"
echo "   scp -r <server>:$(pwd)/$OUTPUT_DIR ./local_viz/"
