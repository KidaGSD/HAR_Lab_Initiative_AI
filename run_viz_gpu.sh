#!/bin/bash
# Manual GPU assignment for visualization

# Usage: ./run_viz_gpu.sh <gpu_id>
# Example: ./run_viz_gpu.sh 2  (use GPU 2)

if [ $# -eq 0 ]; then
    echo "Usage: $0 <gpu_id>"
    echo "Example: $0 2  (to use GPU 2)"
    echo ""
    echo "First, check available GPUs:"
    nvidia-smi
    exit 1
fi

GPU_ID=$1

echo "🎯 Using GPU $GPU_ID"
echo ""

# Configuration
MODEL_NAME="beta_1.0"
CHECKPOINT="checkpoints/checkpoints/experiments_20251206_224347/beta_1.0/best_model.pth"
CONFIG="configs/beta_1.0.yaml"
OUTPUT_DIR="outputs/embeddings_viz"

# Check checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

mkdir -p $OUTPUT_DIR

echo "📊 Configuration:"
echo "  GPU: $GPU_ID"
echo "  Model: $MODEL_NAME"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run with specified GPU
echo "1️⃣ Generating PCA plots..."
python scripts/visualize_embeddings.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --split val \
    --gpu $GPU_ID \
    --output-dir $OUTPUT_DIR

echo ""
echo "2️⃣ Generating t-SNE plots..."
python scripts/visualize_embeddings.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --split val \
    --gpu $GPU_ID \
    --use-tsne \
    --output-dir $OUTPUT_DIR

echo ""
echo "✅ Visualization complete!"
echo ""
echo "📁 Output files:"
ls -lh $OUTPUT_DIR/
