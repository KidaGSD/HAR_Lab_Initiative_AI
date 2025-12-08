#!/bin/bash
# run_visualization_safe.sh - GPU-safe visualization script

set -e

echo "🔍 Checking GPU availability..."

# Function to find free GPU
find_free_gpu() {
    python3 << 'EOF'
import subprocess
import sys

try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', 
                           '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True, check=True)
    
    gpus = []
    for line in result.stdout.strip().split('\n'):
        idx, used, total = map(float, line.split(','))
        free = total - used
        util = used / total * 100
        gpus.append((int(idx), free, util))
        print(f"GPU {int(idx)}: {free:.0f}MB free ({100-util:.1f}% available)")
    
    # Find GPU with most free memory
    if gpus:
        best_gpu = max(gpus, key=lambda x: x[1])
        if best_gpu[1] > 20000:  # Need at least 2GB free
            print(f"\n✓ Using GPU {best_gpu[0]} ({best_gpu[1]:.0f}MB free)")
            sys.exit(best_gpu[0])
        else:
            print("\n⚠ No GPU has enough free memory, will use CPU")
            sys.exit(255)
    else:
        print("⚠ No GPUs found, will use CPU")
        sys.exit(255)
        
except Exception as e:
    print(f"⚠ Error checking GPUs: {e}")
    print("Will use CPU")
    sys.exit(255)
EOF
}

# Get free GPU or use CPU
FREE_GPU=$(find_free_gpu)
GPU_STATUS=$?

if [ $GPU_STATUS -eq 255 ]; then
    echo ""
    echo "📍 Running on CPU (slower but safe)"
    export CUDA_VISIBLE_DEVICES=""
    DEVICE_INFO="CPU"
else
    echo ""
    export CUDA_VISIBLE_DEVICES=$FREE_GPU
    DEVICE_INFO="GPU $FREE_GPU"
fi

# Configuration
MODEL_NAME="beta_1.0"
CHECKPOINT="checkpoints/checkpoints/experiments_20251206_224347/beta_1.0/best_model.pth"
CONFIG="configs/hierarchical.yaml"
OUTPUT_DIR="outputs/embeddings_viz"
BATCH_SIZE=8  # Reduced for safety

# Check checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

mkdir -p $OUTPUT_DIR

echo ""
echo "📊 Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Device: $DEVICE_INFO"
echo "  Batch size: $BATCH_SIZE"
echo "  Output: $OUTPUT_DIR"
echo ""

# Temporarily modify config for smaller batch size
echo "🔧 Creating temporary config with batch_size=$BATCH_SIZE..."
cat $CONFIG | sed "s/batch_size: [0-9]*/batch_size: $BATCH_SIZE/" > /tmp/viz_config.yaml

echo ""
echo "1️⃣ Generating PCA plots..."
python scripts/visualize_embeddings.py \
    --config /tmp/viz_config.yaml \
    --checkpoint $CHECKPOINT \
    --split val \
    --output-dir $OUTPUT_DIR

echo ""
echo "2️⃣ Generating t-SNE plots (slower)..."
python scripts/visualize_embeddings.py \
    --config /tmp/viz_config.yaml \
    --checkpoint $CHECKPOINT \
    --split val \
    --use-tsne \
    --output-dir $OUTPUT_DIR

# Cleanup
rm /tmp/viz_config.yaml

echo ""
echo "✅ Visualization complete!"
echo ""
echo "📁 Output files:"
ls -lh $OUTPUT_DIR/*.png 2>/dev/null || echo "  (No PNG files found)"
ls -lh $OUTPUT_DIR/*.pdf 2>/dev/null || echo "  (No PDF files found)"

echo ""
echo "💡 To download to local machine:"
echo "   scp -r server:$(pwd)/$OUTPUT_DIR ./local_figures/"
