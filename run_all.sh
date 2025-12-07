#!/bin/bash
# =============================================================================
# PARALLEL TRAINING SCRIPT
# Runs Backbone (β=0.0) and Joint (β=0.3) in PARALLEL on different GPUs
# =============================================================================
#
# Usage:
#   ./run_all.sh              # Auto-detect GPUs, run parallel
#   ./run_all.sh --dry-run    # Preview mode
#
# =============================================================================

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="checkpoints/run_${TIMESTAMP}"
LOG_DIR="${OUTPUT_BASE}/logs"

DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo "[DRY RUN MODE]"
fi

# -----------------------------------------------------------------------------
# Find Available GPUs
# -----------------------------------------------------------------------------
find_available_gpus() {
    AVAILABLE_GPUS=()
    if ! command -v nvidia-smi &> /dev/null; then
        AVAILABLE_GPUS=(0 1)
        return
    fi
    while IFS=, read -r FREE IDX; do
        FREE=$(echo "$FREE" | xargs)
        IDX=$(echo "$IDX" | xargs)
        if [ "$FREE" -ge "15000" ]; then
            AVAILABLE_GPUS+=("$IDX")
        fi
    done < <(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits 2>/dev/null)
    
    if [ ${#AVAILABLE_GPUS[@]} -lt 2 ]; then
        echo "⚠ Need at least 2 GPUs with 15GB+ free for parallel training"
        echo "Found: ${AVAILABLE_GPUS[*]}"
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
setup() {
    find_available_gpus
    
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    export WANDB_API_KEY="${WANDB_API_KEY:-e83326e014ad7a27c2a538f4e38b95bd11a161a0}"
    export WANDB_RUN_GROUP="exp_${TIMESTAMP}"
    
    mkdir -p "$OUTPUT_BASE" "$LOG_DIR"
    
    echo "==========================================="
    echo "  Parallel Training Pipeline"
    echo "==========================================="
    echo "Started: $(date)"
    echo "Output: $OUTPUT_BASE"
    echo "W&B Group: $WANDB_RUN_GROUP"
    echo ""
    echo "GPU ${AVAILABLE_GPUS[0]}: Backbone (β=0.0)"
    echo "GPU ${AVAILABLE_GPUS[1]}: Joint (β=0.3)"
    echo ""
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    setup
    
    GPU0=${AVAILABLE_GPUS[0]}
    GPU1=${AVAILABLE_GPUS[1]}
    
    if [ "$DRY_RUN" == "true" ]; then
        echo "[DRY RUN] Would start backbone on GPU $GPU0"
        echo "[DRY RUN] Would start joint on GPU $GPU1"
        exit 0
    fi
    
    # Start BOTH trainings in parallel (background processes)
    echo ">> [Backbone β=0.0] Starting on GPU $GPU0..."
    CUDA_VISIBLE_DEVICES=$GPU0 python train.py \
        --config configs/beta_0.0.yaml \
        --output-dir "${OUTPUT_BASE}/backbone_beta0" \
        2>&1 | tee "${LOG_DIR}/backbone.log" &
    PID_BACKBONE=$!
    
    echo ">> [Joint β=0.3] Starting on GPU $GPU1..."
    CUDA_VISIBLE_DEVICES=$GPU1 python train.py \
        --config configs/beta_0.3.yaml \
        --output-dir "${OUTPUT_BASE}/joint_beta03" \
        2>&1 | tee "${LOG_DIR}/joint.log" &
    PID_JOINT=$!
    
    echo ""
    echo "Both trainings started in parallel!"
    echo "  Backbone PID: $PID_BACKBONE"
    echo "  Joint PID: $PID_JOINT"
    echo ""
    echo "Monitor with: tail -f ${LOG_DIR}/backbone.log ${LOG_DIR}/joint.log"
    echo ""
    
    # Wait for both to complete
    echo "Waiting for both trainings to complete..."
    wait $PID_BACKBONE
    BACKBONE_EXIT=$?
    wait $PID_JOINT  
    JOINT_EXIT=$?
    
    echo ""
    echo "==========================================="
    echo "  Training Complete"
    echo "==========================================="
    
    if [ $BACKBONE_EXIT -eq 0 ]; then
        echo "✓ Backbone: SUCCESS"
        # Run probe on backbone
        if [ -f "${OUTPUT_BASE}/backbone_beta0/best_model.pth" ]; then
            echo ">> Running Probe on Backbone..."
            CUDA_VISIBLE_DEVICES=$GPU0 python train.py \
                --probe \
                --checkpoint "${OUTPUT_BASE}/backbone_beta0/best_model.pth" \
                --config configs/beta_0.0.yaml \
                --output-dir "${OUTPUT_BASE}/backbone_beta0_probe" \
                2>&1 | tee "${LOG_DIR}/backbone_probe.log"
        fi
    else
        echo "✗ Backbone: FAILED (exit code $BACKBONE_EXIT)"
    fi
    
    if [ $JOINT_EXIT -eq 0 ]; then
        echo "✓ Joint: SUCCESS"
    else
        echo "✗ Joint: FAILED (exit code $JOINT_EXIT)"
    fi
    
    echo ""
    echo "Finished: $(date)"
    echo "Logs: $LOG_DIR"
}

main
