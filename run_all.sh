#!/bin/bash
# =============================================================================
# MASTER TRAINING SCRIPT (Multi-GPU Parallel)
# Runs complete research pipeline with parallel execution
# =============================================================================
#
# Experiments:
#   1. Backbone (beta=0.0) + Probe
#   2. Joint (beta=0.3) + Probe
#   3. 4 Baseline Models
#
# Output: Scientific benchmark comparison table
#
# Usage:
#   ./run_all.sh              # Run everything
#   ./run_all.sh --dry-run    # Preview mode
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
RESULTS_CSV="${RESULTS_DIR}/benchmark_results.csv"

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
    while IFS=, read -r FREE IDX; do
        FREE=$(echo "$FREE" | xargs)
        IDX=$(echo "$IDX" | xargs)
        if [ "$FREE" -ge "30000" ]; then
            AVAILABLE_GPUS+=("$IDX")
        fi
    done < <(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits 2>/dev/null)
}

# -----------------------------------------------------------------------------
# Pre-flight Checks
# -----------------------------------------------------------------------------
preflight_check() {
    echo "=========================================="
    echo "  Pre-flight Checks"
    echo "=========================================="
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo "✗ nvidia-smi not found!"
        exit 1
    fi
    
    find_available_gpus
    
    if [ ${#AVAILABLE_GPUS[@]} -eq 0 ]; then
        echo "✗ No GPU with 20GB+ free memory!"
        nvidia-smi --query-gpu=index,memory.free --format=csv
        exit 1
    fi
    echo "✓ Available GPUs: ${AVAILABLE_GPUS[*]}"
    
    # W&B
    export WANDB_API_KEY="${WANDB_API_KEY:-e83326e014ad7a27c2a538f4e38b95bd11a161a0}"
    wandb login $WANDB_API_KEY 2>/dev/null && echo "✓ W&B logged in" || echo "⚠ W&B offline"
    
    # Data
    DATA_COUNT=$(ls -1 data/processed_ego4d/ 2>/dev/null | wc -l)
    echo "✓ Data: $DATA_COUNT videos"
    
    echo ""
}

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
setup_env() {
    echo "=========================================="
    echo "  Setup"
    echo "=========================================="
    
    if [ -z "$CONDA_PREFIX" ]; then
        source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
        conda activate ego4d_lab 2>/dev/null || true
    fi
    
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    mkdir -p "$OUTPUT_BASE" "$LOG_DIR" "$RESULTS_DIR"
    
    # Initialize results CSV
    echo "Model,Scenario_F1,Action_F1,Params,Training_Time" > "$RESULTS_CSV"
    
    echo "Output: $OUTPUT_BASE"
    echo "Started: $(date)" | tee "${OUTPUT_BASE}/run_info.txt"
    
    cp configs/beta_0.0.yaml configs/beta_0.3.yaml "$OUTPUT_BASE/"
    echo ""
}

# -----------------------------------------------------------------------------
# Run Training
# -----------------------------------------------------------------------------
run_training() {
    local NAME="$1"
    local CONFIG="$2"
    local GPU="$3"
    local OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"
    
    mkdir -p "$OUTPUT_DIR"
    echo ">> [$NAME] GPU $GPU..."
    
    if [ "$DRY_RUN" == "true" ]; then
        echo "[DRY RUN] python train.py --config $CONFIG"
        return
    fi
    
    START_TIME=$(date +%s)
    
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --config "$CONFIG" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "${LOG_DIR}/${NAME}.log" &
    
    echo $! > "${OUTPUT_DIR}/pid.txt"
}

# -----------------------------------------------------------------------------
# Run Probe
# -----------------------------------------------------------------------------
run_probe() {
    local NAME="$1"
    local CHECKPOINT="$2"
    local GPU="$3"
    local OUTPUT_DIR="${OUTPUT_BASE}/${NAME}_probe"
    
    mkdir -p "$OUTPUT_DIR"
    echo ">> [$NAME Probe] GPU $GPU..."
    
    if [ "$DRY_RUN" == "true" ]; then
        echo "[DRY RUN] python train.py --probe"
        return
    fi
    
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --probe \
        --checkpoint "$CHECKPOINT" \
        --config configs/beta_0.0.yaml \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "${LOG_DIR}/${NAME}_probe.log"
}

# -----------------------------------------------------------------------------
# Run Baselines
# -----------------------------------------------------------------------------
run_baselines() {
    local GPU="$1"
    local BASELINE_DIR="${OUTPUT_BASE}/baselines"
    mkdir -p "$BASELINE_DIR"
    
    echo ">> [Baselines] GPU $GPU..."
    
    if [ "$DRY_RUN" == "true" ]; then
        echo "[DRY RUN] 4 baseline models"
        return
    fi
    
    for MODEL in mlp_mlp cnn_mlp imu2clip cnn_lstm_gru; do
        echo "  Training: $MODEL"
        CUDA_VISIBLE_DEVICES=$GPU python scripts/train_baselines.py \
            --model "$MODEL" \
            --config configs/beta_0.3.yaml \
            --output-dir "$BASELINE_DIR" \
            2>&1 | tee "${LOG_DIR}/baseline_${MODEL}.log"
    done
}

# -----------------------------------------------------------------------------
# Extract Metrics
# -----------------------------------------------------------------------------
extract_metrics() {
    local LOG="$1"
    local NAME="$2"
    
    if [ ! -f "$LOG" ]; then
        echo "$NAME,N/A,N/A,N/A,N/A" >> "$RESULTS_CSV"
        return
    fi
    
    # Extract scenario F1
    SCENARIO_F1=$(grep -oE "Val Scenario F1: [0-9]+\.[0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
    
    # Extract action F1
    ACTION_F1=$(grep -oE "(Val Action F1|Probe Val Action F1): [0-9]+\.[0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
    
    echo "$NAME,$SCENARIO_F1,$ACTION_F1,~1.5M,-" >> "$RESULTS_CSV"
}

# -----------------------------------------------------------------------------
# Generate Results Table
# -----------------------------------------------------------------------------
generate_results_table() {
    echo ""
    echo "=========================================="
    echo "  BENCHMARK RESULTS"
    echo "=========================================="
    echo ""
    
    # Extract metrics from logs
    extract_metrics "${LOG_DIR}/backbone_beta0.log" "Backbone (β=0.0)"
    extract_metrics "${LOG_DIR}/backbone_beta0_probe.log" "Backbone + Probe"
    extract_metrics "${LOG_DIR}/joint_beta03.log" "Joint (β=0.3)"
    extract_metrics "${LOG_DIR}/joint_beta03_probe.log" "Joint + Probe"
    extract_metrics "${LOG_DIR}/baseline_mlp_mlp.log" "MLP-MLP"
    extract_metrics "${LOG_DIR}/baseline_cnn_mlp.log" "CNN-MLP"
    extract_metrics "${LOG_DIR}/baseline_imu2clip.log" "IMU2CLIP"
    extract_metrics "${LOG_DIR}/baseline_cnn_lstm_gru.log" "CNN-LSTM-GRU"
    
    # Print table
    echo "| Model | Scenario F1 | Action F1 |"
    echo "|-------|-------------|-----------|"
    
    while IFS=, read -r MODEL SCENARIO ACTION PARAMS TIME; do
        if [ "$MODEL" != "Model" ]; then
            printf "| %-20s | %11s | %9s |\n" "$MODEL" "$SCENARIO" "$ACTION"
        fi
    done < "$RESULTS_CSV"
    
    echo ""
    
    # Save as markdown
    RESULTS_MD="${RESULTS_DIR}/benchmark_results.md"
    echo "# Benchmark Results" > "$RESULTS_MD"
    echo "" >> "$RESULTS_MD"
    echo "Generated: $(date)" >> "$RESULTS_MD"
    echo "" >> "$RESULTS_MD"
    echo "## Performance Comparison" >> "$RESULTS_MD"
    echo "" >> "$RESULTS_MD"
    echo "| Model | Scenario F1 | Action F1 |" >> "$RESULTS_MD"
    echo "|-------|-------------|-----------|" >> "$RESULTS_MD"
    
    while IFS=, read -r MODEL SCENARIO ACTION PARAMS TIME; do
        if [ "$MODEL" != "Model" ]; then
            echo "| $MODEL | $SCENARIO | $ACTION |" >> "$RESULTS_MD"
        fi
    done < "$RESULTS_CSV"
    
    echo "" >> "$RESULTS_MD"
    echo "## Configuration" >> "$RESULTS_MD"
    echo "- Focal Loss: gamma=2.0" >> "$RESULTS_MD"
    echo "- Class Weights: [5.0, 7.0, 0.5, 10.0]" >> "$RESULTS_MD"
    echo "- Label Smoothing: 0.1" >> "$RESULTS_MD"
    echo "- Augmentation: Jittering + Scaling" >> "$RESULTS_MD"
    
    echo "✓ Results saved to: $RESULTS_MD"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    preflight_check
    setup_env
    
    NUM_GPUS=${#AVAILABLE_GPUS[@]}
    GPU0=${AVAILABLE_GPUS[0]}
    GPU1=${AVAILABLE_GPUS[1]:-$GPU0}
    GPU2=${AVAILABLE_GPUS[2]:-$GPU0}
    
    echo "=========================================="
    echo "  Parallel Execution ($NUM_GPUS GPUs)"
    echo "=========================================="
    echo "GPU $GPU0: Backbone (β=0.0)"
    echo "GPU $GPU1: Joint (β=0.3)"
    echo "GPU $GPU2: Baselines"
    echo ""
    
    # Start parallel training
    run_training "backbone_beta0" "configs/beta_0.0.yaml" "$GPU0"
    run_training "joint_beta03" "configs/beta_0.3.yaml" "$GPU1"
    run_baselines "$GPU2" &
    PID_BASELINES=$!
    
    # Wait for backbone, then probe
    wait $(cat "${OUTPUT_BASE}/backbone_beta0/pid.txt" 2>/dev/null || echo "")
    echo "✓ Backbone complete"
    run_probe "backbone_beta0" "${OUTPUT_BASE}/backbone_beta0/best_model.pth" "$GPU0"
    
    # Wait for joint, then probe
    wait $(cat "${OUTPUT_BASE}/joint_beta03/pid.txt" 2>/dev/null || echo "")
    echo "✓ Joint complete"
    run_probe "joint_beta03" "${OUTPUT_BASE}/joint_beta03/best_model.pth" "$GPU1"
    
    # Wait for baselines
    wait $PID_BASELINES
    echo "✓ Baselines complete"
    
    # Generate results
    generate_results_table
    
    echo ""
    echo "=========================================="
    echo "  COMPLETE!"
    echo "=========================================="
    echo "Output: $OUTPUT_BASE"
    echo "Results: $RESULTS_CSV"
    echo "Report: ${RESULTS_DIR}/benchmark_results.md"
    echo "Finished: $(date)"
}

main
