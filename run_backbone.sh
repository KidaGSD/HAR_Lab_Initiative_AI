#!/bin/bash
# =============================================================================
# BACKBONE TRAINING SCRIPT  
# Trains Hierarchical Models (β=0.0 and β=0.3) in PARALLEL + Auto Probe
# =============================================================================
#
# Usage:
#   ./run_backbone.sh              # Auto-detect GPUs
#   ./run_backbone.sh 2 3          # Use GPU 2 and GPU 3
#
# =============================================================================

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="checkpoints/backbone_${TIMESTAMP}"
LOG_DIR="${OUTPUT_BASE}/logs"

GPU0=${1:-0}
GPU1=${2:-1}

echo "==========================================="
echo "  Hierarchical Model Training"
echo "==========================================="
echo "Started: $(date)"
echo "Output: $OUTPUT_BASE"
echo ""
echo "GPU $GPU0: Backbone (β=0.0) - Scenario Only"
echo "GPU $GPU1: Joint (β=0.3) - Scenario + Action"
echo ""

# Setup
export PYTHONPATH="$(pwd):$PYTHONPATH"
export WANDB_API_KEY="${WANDB_API_KEY:-e83326e014ad7a27c2a538f4e38b95bd11a161a0}"
export WANDB_RUN_GROUP="backbone_${TIMESTAMP}"

mkdir -p "$OUTPUT_BASE" "$LOG_DIR"

# ============= PARALLEL TRAINING =============
echo ">> Starting PARALLEL training..."

# Backbone β=0.0
CUDA_VISIBLE_DEVICES=$GPU0 python train.py \
    --config configs/beta_0.0.yaml \
    --output-dir "${OUTPUT_BASE}/beta0" \
    2>&1 | tee "${LOG_DIR}/beta0.log" &
PID_BETA0=$!

# Joint β=0.3
CUDA_VISIBLE_DEVICES=$GPU1 python train.py \
    --config configs/beta_0.3.yaml \
    --output-dir "${OUTPUT_BASE}/beta03" \
    2>&1 | tee "${LOG_DIR}/beta03.log" &
PID_BETA03=$!

echo "Training started:"
echo "  β=0.0 PID: $PID_BETA0"
echo "  β=0.3 PID: $PID_BETA03"
echo ""
echo "Monitor: tail -f ${LOG_DIR}/*.log"
echo ""

# Wait for both
wait $PID_BETA0
EXIT_BETA0=$?
wait $PID_BETA03
EXIT_BETA03=$?

echo ""
echo "==========================================="
echo "  Training Results"
echo "==========================================="

# ============= AUTO PROBE =============
if [ $EXIT_BETA0 -eq 0 ] && [ -f "${OUTPUT_BASE}/beta0/best_model.pth" ]; then
    echo "✓ β=0.0 Training: SUCCESS"
    echo ">> Running Probe on β=0.0 backbone..."
    CUDA_VISIBLE_DEVICES=$GPU0 python train.py \
        --probe \
        --checkpoint "${OUTPUT_BASE}/beta0/best_model.pth" \
        --config configs/beta_0.0.yaml \
        --output-dir "${OUTPUT_BASE}/beta0_probe" \
        2>&1 | tee "${LOG_DIR}/beta0_probe.log"
    echo "✓ Probe complete"
else
    echo "✗ β=0.0 Training: FAILED"
fi

if [ $EXIT_BETA03 -eq 0 ] && [ -f "${OUTPUT_BASE}/beta03/best_model.pth" ]; then
    echo "✓ β=0.3 Training: SUCCESS"
    echo ">> Running Probe on β=0.3 joint model..."
    CUDA_VISIBLE_DEVICES=$GPU1 python train.py \
        --probe \
        --checkpoint "${OUTPUT_BASE}/beta03/best_model.pth" \
        --config configs/beta_0.3.yaml \
        --output-dir "${OUTPUT_BASE}/beta03_probe" \
        2>&1 | tee "${LOG_DIR}/beta03_probe.log"
    echo "✓ Probe complete"
else
    echo "✗ β=0.3 Training: FAILED"
fi

echo ""
echo "==========================================="
echo "  COMPLETE"
echo "==========================================="
echo "Finished: $(date)"
echo "Results: $OUTPUT_BASE"
