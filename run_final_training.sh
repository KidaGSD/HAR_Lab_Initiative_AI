#!/bin/bash
# =============================================================================
# FINAL TRAINING PIPELINE
# End-to-end training following EgoCHARM methodology
# =============================================================================
#
# Pipeline:
#   Phase 1: Train backbone (beta=0.0) → LLE learns motion embeddings
#   Phase 2: Probe action head on frozen LLE
#   Phase 3: (Optional) Fine-tune with beta=0.3
#
# Usage:
#   ./run_final_training.sh           # Run full pipeline
#   ./run_final_training.sh phase1    # Backbone only
#   ./run_final_training.sh phase2    # Probe only
#   ./run_final_training.sh baselines # Baselines only
#
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="checkpoints/final_${TIMESTAMP}"
LOG_FILE="${OUTPUT_BASE}/training.log"

PHASE=$1
if [ -z "$PHASE" ]; then
    PHASE="all"
fi

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
echo "=========================================="
echo "  Final Training Pipeline"
echo "  Started: $(date)"
echo "=========================================="

# Conda
if [ -n "$CONDA_PREFIX" ]; then
    echo "Conda env: $CONDA_PREFIX"
else
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ego4d_lab
fi

export PYTHONPATH="$(pwd):$PYTHONPATH"
export WANDB_API_KEY=${WANDB_API_KEY:-e83326e014ad7a27c2a538f4e38b95bd11a161a0}
wandb login $WANDB_API_KEY 2>/dev/null || true

mkdir -p "$OUTPUT_BASE"
echo "Output: $OUTPUT_BASE"
echo ""

# GPU Detection
AVAILABLE_GPUS=()
while IFS=, read -r FREE IDX; do
    FREE=$(echo "$FREE" | xargs)
    IDX=$(echo "$IDX" | xargs)
    if [ "$FREE" -ge "30000" ]; then
        AVAILABLE_GPUS+=("$IDX")
    fi
done < <(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits 2>/dev/null || echo "0,0")

GPU=${AVAILABLE_GPUS[0]:-0}
echo "Using GPU: $GPU"

# -----------------------------------------------------------------------------
# Phase 1: Backbone Training (Scenario-Only)
# -----------------------------------------------------------------------------
run_phase1() {
    echo ""
    echo "=========================================="
    echo "  Phase 1: Backbone Training (beta=0.0)"
    echo "=========================================="
    
    PHASE1_DIR="${OUTPUT_BASE}/phase1_backbone"
    mkdir -p "$PHASE1_DIR"
    
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --config configs/beta_0.0.yaml \
        --output-dir "$PHASE1_DIR" \
        2>&1 | tee "${PHASE1_DIR}/training.log"
    
    echo "Phase 1 complete: $PHASE1_DIR"
    echo "Best model: ${PHASE1_DIR}/best_model.pth"
}

# -----------------------------------------------------------------------------
# Phase 2: Probe Training (Action Head on Frozen LLE)
# -----------------------------------------------------------------------------
run_phase2() {
    echo ""
    echo "=========================================="
    echo "  Phase 2: Probe Training (Action Head)"
    echo "=========================================="
    
    PHASE1_DIR="${OUTPUT_BASE}/phase1_backbone"
    PHASE2_DIR="${OUTPUT_BASE}/phase2_probe"
    mkdir -p "$PHASE2_DIR"
    
    # Find checkpoint
    CHECKPOINT="${PHASE1_DIR}/best_model.pth"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: Phase 1 checkpoint not found: $CHECKPOINT"
        echo "Run phase1 first!"
        exit 1
    fi
    
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --probe \
        --checkpoint "$CHECKPOINT" \
        --config configs/beta_0.0.yaml \
        --output-dir "$PHASE2_DIR" \
        2>&1 | tee "${PHASE2_DIR}/probe.log"
    
    echo "Phase 2 complete: $PHASE2_DIR"
}

# -----------------------------------------------------------------------------
# Phase 3: Joint Fine-tuning (Optional)
# -----------------------------------------------------------------------------
run_phase3() {
    echo ""
    echo "=========================================="
    echo "  Phase 3: Joint Fine-tuning (beta=0.3)"
    echo "=========================================="
    
    PHASE3_DIR="${OUTPUT_BASE}/phase3_finetune"
    mkdir -p "$PHASE3_DIR"
    
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --config configs/beta_0.3.yaml \
        --output-dir "$PHASE3_DIR" \
        2>&1 | tee "${PHASE3_DIR}/training.log"
    
    echo "Phase 3 complete: $PHASE3_DIR"
}

# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------
run_baselines() {
    echo ""
    echo "=========================================="
    echo "  Baseline Models"
    echo "=========================================="
    
    BASELINE_DIR="${OUTPUT_BASE}/baselines"
    
    # Run baselines script if exists
    if [ -f "./run_baselines.sh" ]; then
        OUTPUT_DIR="$BASELINE_DIR" ./run_baselines.sh
    else
        echo "Baselines script not found, running individually..."
        mkdir -p "$BASELINE_DIR"
        for MODEL in mlp_mlp cnn_mlp imu2clip cnn_lstm_gru; do
            echo "Training: $MODEL"
            CUDA_VISIBLE_DEVICES=$GPU python scripts/train_baselines.py \
                --model "$MODEL" \
                --output-dir "$BASELINE_DIR" \
                2>&1 | tee "${BASELINE_DIR}/${MODEL}.log" &
        done
        wait
    fi
}

# -----------------------------------------------------------------------------
# Generate Report
# -----------------------------------------------------------------------------
run_analysis() {
    echo ""
    echo "=========================================="
    echo "  Generating Analysis Report"
    echo "=========================================="
    
    python scripts/experiment_analysis.py
    
    echo "Report: reports/experiment_report.md"
}

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
case $PHASE in
    "phase1")
        run_phase1
        ;;
    "phase2")
        run_phase2
        ;;
    "phase3")
        run_phase3
        ;;
    "baselines")
        run_baselines
        ;;
    "analysis")
        run_analysis
        ;;
    "all")
        run_phase1
        run_phase2
        run_phase3
        run_baselines
        run_analysis
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: $0 [phase1|phase2|phase3|baselines|analysis|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "  Pipeline Complete!"
echo "  Finished: $(date)"
echo "  Output: $OUTPUT_BASE"
echo "=========================================="
