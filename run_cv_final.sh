#!/bin/bash
# =============================================================================
# CV + FINAL MODEL TRAINING SCRIPT
# =============================================================================
#
# This script:
#   1. Runs 4-fold CV for reliable metrics (mean±std)
#   2. Retrains final model on train+val combined
#   3. Runs probe on final model
#   4. Generates scientific benchmark table
#
# Usage:
#   ./run_cv_final.sh              # Run everything
#   ./run_cv_final.sh --dry-run    # Preview
#
# =============================================================================

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="checkpoints/cv_final_${TIMESTAMP}"
LOG_DIR="${OUTPUT_BASE}/logs"
RESULTS_DIR="${OUTPUT_BASE}/results"

DRY_RUN=false
[ "$1" == "--dry-run" ] && DRY_RUN=true && echo "[DRY RUN]"

# -----------------------------------------------------------------------------
# Find Available GPU
# -----------------------------------------------------------------------------
find_gpu() {
    while IFS=, read -r FREE IDX; do
        FREE=$(echo "$FREE" | xargs)
        IDX=$(echo "$IDX" | xargs)
        if [ "$FREE" -ge "30000" ]; then
            echo "$IDX"
            return
        fi
    done < <(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits 2>/dev/null)
    echo "0"
}

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
setup() {
    echo "=========================================="
    echo "  CV + Final Model Training"
    echo "  $(date)"
    echo "=========================================="
    
    GPU=$(find_gpu)
    echo "Using GPU: $GPU"
    export CUDA_VISIBLE_DEVICES=$GPU
    
    # Conda
    [ -z "$CONDA_PREFIX" ] && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ego4d_lab
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    export WANDB_API_KEY="${WANDB_API_KEY:-e83326e014ad7a27c2a538f4e38b95bd11a161a0}"
    wandb login $WANDB_API_KEY 2>/dev/null || echo "W&B offline"
    
    mkdir -p "$OUTPUT_BASE" "$LOG_DIR" "$RESULTS_DIR"
    cp configs/beta_0.3.yaml "$OUTPUT_BASE/"
    
    echo "" 
}

# -----------------------------------------------------------------------------
# Phase 1: 4-Fold CV (for reliable metrics)
# -----------------------------------------------------------------------------
run_cv_evaluation() {
    echo "=========================================="
    echo "  Phase 1: 4-Fold Cross-Validation"
    echo "=========================================="
    
    CV_DIR="${OUTPUT_BASE}/cv_evaluation"
    mkdir -p "$CV_DIR"
    
    [ "$DRY_RUN" == "true" ] && echo "[DRY RUN] python train.py --cv --n-folds 4" && return
    
    python train.py \
        --cv \
        --n-folds 4 \
        --config configs/beta_0.3.yaml \
        --output-dir "$CV_DIR" \
        --run-name "cv_4fold_beta03" \
        2>&1 | tee "${LOG_DIR}/cv_evaluation.log"
    
    # Extract CV summary
    echo ""
    echo "CV Results:"
    grep -A 10 "CV Summary" "${LOG_DIR}/cv_evaluation.log" | tee "${RESULTS_DIR}/cv_summary.txt"
    
    echo ""
}

# -----------------------------------------------------------------------------
# Phase 2: Final Model Training (train+val combined)
# -----------------------------------------------------------------------------
run_final_training() {
    echo "=========================================="
    echo "  Phase 2: Final Model (train+val)"
    echo "=========================================="
    
    FINAL_DIR="${OUTPUT_BASE}/final_model"
    mkdir -p "$FINAL_DIR"
    
    [ "$DRY_RUN" == "true" ] && echo "[DRY RUN] python train.py --use-all-data" && return
    
    python train.py \
        --use-all-data \
        --config configs/beta_0.3.yaml \
        --output-dir "$FINAL_DIR" \
        --run-name "final_model_beta03" \
        2>&1 | tee "${LOG_DIR}/final_training.log"
    
    if [ -f "${FINAL_DIR}/best_model.pth" ]; then
        echo "✓ Final model saved: ${FINAL_DIR}/best_model.pth"
        ls -lh "${FINAL_DIR}/best_model.pth"
    fi
    
    echo ""
}

# -----------------------------------------------------------------------------
# Phase 3: Probe on Final Model
# -----------------------------------------------------------------------------
run_final_probe() {
    echo "=========================================="
    echo "  Phase 3: Probe on Final Model"
    echo "=========================================="
    
    FINAL_CHECKPOINT="${OUTPUT_BASE}/final_model/best_model.pth"
    PROBE_DIR="${OUTPUT_BASE}/final_probe"
    mkdir -p "$PROBE_DIR"
    
    if [ ! -f "$FINAL_CHECKPOINT" ]; then
        echo "✗ Checkpoint not found, skipping probe"
        return
    fi
    
    [ "$DRY_RUN" == "true" ] && echo "[DRY RUN] python train.py --probe" && return
    
    python train.py \
        --probe \
        --checkpoint "$FINAL_CHECKPOINT" \
        --config configs/beta_0.3.yaml \
        --output-dir "$PROBE_DIR" \
        --run-name "final_probe" \
        2>&1 | tee "${LOG_DIR}/final_probe.log"
    
    # Extract probe F1
    PROBE_F1=$(grep "Probe Val Action F1" "${LOG_DIR}/final_probe.log" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
    echo "Final Probe Action F1: $PROBE_F1" | tee -a "${RESULTS_DIR}/final_metrics.txt"
    
    echo ""
}

# -----------------------------------------------------------------------------
# Generate Results Table
# -----------------------------------------------------------------------------
generate_results() {
    echo "=========================================="
    echo "  Results Summary"
    echo "=========================================="
    
    RESULTS_MD="${RESULTS_DIR}/final_benchmark.md"
    
    cat > "$RESULTS_MD" << 'EOF'
# Final Benchmark Results

## Cross-Validation Results (4-Fold)

EOF
    
    if [ -f "${RESULTS_DIR}/cv_summary.txt" ]; then
        cat "${RESULTS_DIR}/cv_summary.txt" >> "$RESULTS_MD"
    fi
    
    cat >> "$RESULTS_MD" << EOF

## Final Model Performance

$(cat "${RESULTS_DIR}/final_metrics.txt" 2>/dev/null || echo "N/A")

## Configuration
- Config: beta_0.3.yaml
- Focal Loss: gamma=2.0
- Class Weights: [5.0, 7.0, 0.5, 10.0]
- Label Smoothing: 0.1
- Augmentation: Jittering + Scaling

## Output
- CV Logs: ${LOG_DIR}/cv_evaluation.log
- Final Model: ${OUTPUT_BASE}/final_model/best_model.pth
- Generated: $(date)
EOF
    
    echo "Results saved to: $RESULTS_MD"
    cat "$RESULTS_MD"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    setup
    run_cv_evaluation
    run_final_training
    run_final_probe
    generate_results
    
    echo ""
    echo "=========================================="
    echo "  COMPLETE!"
    echo "=========================================="
    echo "Output: $OUTPUT_BASE"
    echo "Finished: $(date)"
}

main
