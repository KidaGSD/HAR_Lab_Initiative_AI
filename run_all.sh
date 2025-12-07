#!/bin/bash
# =============================================================================
# MASTER TRAINING SCRIPT
# Runs complete end-to-end research pipeline with all experiments
# =============================================================================
#
# This script runs:
#   1. Backbone training (beta=0.0) with augmentation + label smoothing
#   2. Probe training with Focal Loss + aggressive weights
#   3. Joint training (beta=0.3)
#   4. All 4 baseline models
#   5. Auto-generates analysis report
#
# All results saved to timestamped directory with full logging
#
# Usage:
#   ./run_all.sh              # Run everything
#   ./run_all.sh --dry-run    # Show what would run
#
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="checkpoints/run_${TIMESTAMP}"
LOG_DIR="${OUTPUT_BASE}/logs"
RESULTS_DIR="${OUTPUT_BASE}/results"

DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo "[DRY RUN MODE]"
fi

# -----------------------------------------------------------------------------
# Pre-flight Checks
# -----------------------------------------------------------------------------
preflight_check() {
    echo "=========================================="
    echo "  Pre-flight Checks"
    echo "=========================================="
    
    # Check CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        echo "✗ nvidia-smi not found. GPU required!"
        exit 1
    fi
    
    # Check GPU memory
    GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | xargs)
    if [ "$GPU_MEM" -lt 15000 ]; then
        echo "✗ GPU memory too low: ${GPU_MEM}MB (need 15GB+)"
        echo "  Close other processes and retry."
        exit 1
    fi
    echo "✓ GPU memory: ${GPU_MEM}MB available"
    
    # Check W&B
    if [ -z "$WANDB_API_KEY" ]; then
        export WANDB_API_KEY="e83326e014ad7a27c2a538f4e38b95bd11a161a0"
    fi
    wandb login $WANDB_API_KEY 2>/dev/null && echo "✓ W&B logged in" || echo "⚠ W&B login failed (will continue offline)"
    
    # Check data
    DATA_COUNT=$(ls -1 data/processed_ego4d/ 2>/dev/null | wc -l)
    if [ "$DATA_COUNT" -lt 50 ]; then
        echo "✗ Insufficient data: only $DATA_COUNT videos found"
        exit 1
    fi
    echo "✓ Data: $DATA_COUNT processed videos"
    
    # Check configs
    for config in beta_0.0 beta_0.3; do
        if [ ! -f "configs/${config}.yaml" ]; then
            echo "✗ Config missing: configs/${config}.yaml"
            exit 1
        fi
    done
    echo "✓ Configs present"
    
    # Check disk space
    DISK_FREE=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ "$DISK_FREE" -lt 10 ]; then
        echo "✗ Low disk space: ${DISK_FREE}GB (need 10GB+)"
        exit 1
    fi
    echo "✓ Disk space: ${DISK_FREE}GB available"
    
    echo ""
    echo "All pre-flight checks passed!"
    echo ""
}

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
setup_env() {
    echo "=========================================="
    echo "  Environment Setup"
    echo "=========================================="
    
    # Conda
    if [ -n "$CONDA_PREFIX" ]; then
        echo "Conda env: $CONDA_PREFIX"
    else
        source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
        conda activate ego4d_lab
    fi
    
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    export CUDA_VISIBLE_DEVICES=0
    
    # Create output directories
    mkdir -p "$OUTPUT_BASE" "$LOG_DIR" "$RESULTS_DIR"
    
    echo "Output directory: $OUTPUT_BASE"
    echo "Started: $(date)" | tee "${OUTPUT_BASE}/run_info.txt"
    echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')" >> "${OUTPUT_BASE}/run_info.txt"
    
    # Save configs for reproducibility
    cp configs/beta_0.0.yaml configs/beta_0.3.yaml "$OUTPUT_BASE/"
    
    echo ""
}

# -----------------------------------------------------------------------------
# Phase 1: Backbone Training
# -----------------------------------------------------------------------------
run_backbone() {
    echo "=========================================="
    echo "  Phase 1: Backbone Training (beta=0.0)"
    echo "=========================================="
    
    PHASE1_DIR="${OUTPUT_BASE}/phase1_backbone"
    mkdir -p "$PHASE1_DIR"
    
    if [ "$DRY_RUN" == "true" ]; then
        echo "[DRY RUN] Would run: python train.py --config configs/beta_0.0.yaml --output-dir $PHASE1_DIR"
        return
    fi
    
    python train.py \
        --config configs/beta_0.0.yaml \
        --output-dir "$PHASE1_DIR" \
        2>&1 | tee "${LOG_DIR}/phase1_backbone.log"
    
    # Save metrics
    echo "Phase 1 completed: $(date)" >> "${RESULTS_DIR}/timeline.txt"
    
    # Check if model was saved
    if [ -f "${PHASE1_DIR}/best_model.pth" ]; then
        echo "✓ Backbone model saved"
        MODEL_SIZE=$(ls -lh "${PHASE1_DIR}/best_model.pth" | awk '{print $5}')
        echo "  Model size: $MODEL_SIZE"
    else
        echo "✗ Backbone model not found!"
        exit 1
    fi
    
    echo ""
}

# -----------------------------------------------------------------------------
# Phase 2: Probe Training
# -----------------------------------------------------------------------------
run_probe() {
    echo "=========================================="
    echo "  Phase 2: Probe Training (Action Head)"
    echo "=========================================="
    
    PHASE1_DIR="${OUTPUT_BASE}/phase1_backbone"
    PHASE2_DIR="${OUTPUT_BASE}/phase2_probe"
    CHECKPOINT="${PHASE1_DIR}/best_model.pth"
    mkdir -p "$PHASE2_DIR"
    
    if [ ! -f "$CHECKPOINT" ]; then
        echo "✗ Checkpoint not found: $CHECKPOINT"
        echo "  Run backbone training first!"
        exit 1
    fi
    
    if [ "$DRY_RUN" == "true" ]; then
        echo "[DRY RUN] Would run: python train.py --probe --checkpoint $CHECKPOINT"
        return
    fi
    
    python train.py \
        --probe \
        --checkpoint "$CHECKPOINT" \
        --config configs/beta_0.0.yaml \
        --output-dir "$PHASE2_DIR" \
        2>&1 | tee "${LOG_DIR}/phase2_probe.log"
    
    echo "Phase 2 completed: $(date)" >> "${RESULTS_DIR}/timeline.txt"
    
    # Extract final metrics from log
    PROBE_F1=$(grep "Probe Val Action F1" "${LOG_DIR}/phase2_probe.log" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    echo "Probe Action F1: $PROBE_F1" | tee -a "${RESULTS_DIR}/metrics.txt"
    
    echo ""
}

# -----------------------------------------------------------------------------
# Phase 3: Joint Training
# -----------------------------------------------------------------------------
run_joint() {
    echo "=========================================="
    echo "  Phase 3: Joint Training (beta=0.3)"
    echo "=========================================="
    
    PHASE3_DIR="${OUTPUT_BASE}/phase3_joint"
    mkdir -p "$PHASE3_DIR"
    
    if [ "$DRY_RUN" == "true" ]; then
        echo "[DRY RUN] Would run: python train.py --config configs/beta_0.3.yaml --output-dir $PHASE3_DIR"
        return
    fi
    
    python train.py \
        --config configs/beta_0.3.yaml \
        --output-dir "$PHASE3_DIR" \
        2>&1 | tee "${LOG_DIR}/phase3_joint.log"
    
    echo "Phase 3 completed: $(date)" >> "${RESULTS_DIR}/timeline.txt"
    
    if [ -f "${PHASE3_DIR}/best_model.pth" ]; then
        echo "✓ Joint model saved"
    fi
    
    echo ""
}

# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------
run_baselines() {
    echo "=========================================="
    echo "  Baseline Models (4 models)"
    echo "=========================================="
    
    BASELINE_DIR="${OUTPUT_BASE}/baselines"
    mkdir -p "$BASELINE_DIR"
    
    if [ "$DRY_RUN" == "true" ]; then
        echo "[DRY RUN] Would run baseline models: mlp_mlp, cnn_mlp, imu2clip, cnn_lstm_gru"
        return
    fi
    
    for MODEL in mlp_mlp cnn_mlp imu2clip cnn_lstm_gru; do
        echo ">> Training: $MODEL"
        python scripts/train_baselines.py \
            --model "$MODEL" \
            --config configs/beta_0.3.yaml \
            --output-dir "$BASELINE_DIR" \
            2>&1 | tee "${LOG_DIR}/baseline_${MODEL}.log"
    done
    
    echo "Baselines completed: $(date)" >> "${RESULTS_DIR}/timeline.txt"
    
    # Copy results
    if [ -f "${BASELINE_DIR}/baseline_results.csv" ]; then
        cp "${BASELINE_DIR}/baseline_results.csv" "$RESULTS_DIR/"
        echo "✓ Baseline results saved"
    fi
    
    echo ""
}

# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
run_analysis() {
    echo "=========================================="
    echo "  Generating Analysis Report"
    echo "=========================================="
    
    if [ "$DRY_RUN" == "true" ]; then
        echo "[DRY RUN] Would generate analysis report"
        return
    fi
    
    python scripts/experiment_analysis.py 2>&1 | tee "${LOG_DIR}/analysis.log"
    
    # Copy report to results
    if [ -d "reports" ]; then
        cp -r reports/* "$RESULTS_DIR/" 2>/dev/null || true
        echo "✓ Reports copied to $RESULTS_DIR"
    fi
    
    echo ""
}

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print_summary() {
    echo "=========================================="
    echo "  COMPLETE! Summary"
    echo "=========================================="
    echo ""
    echo "Output: $OUTPUT_BASE"
    echo ""
    echo "Structure:"
    ls -la "$OUTPUT_BASE" 2>/dev/null || true
    echo ""
    
    if [ -f "${RESULTS_DIR}/metrics.txt" ]; then
        echo "Metrics:"
        cat "${RESULTS_DIR}/metrics.txt"
    fi
    
    echo ""
    echo "Timeline:"
    cat "${RESULTS_DIR}/timeline.txt" 2>/dev/null || true
    
    echo ""
    echo "Logs: $LOG_DIR"
    echo "Results: $RESULTS_DIR"
    echo ""
    echo "Finished: $(date)"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    preflight_check
    setup_env
    
    run_backbone
    run_probe
    run_joint
    run_baselines
    run_analysis
    
    print_summary
}

# Run
main
