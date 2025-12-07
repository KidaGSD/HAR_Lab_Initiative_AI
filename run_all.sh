#!/bin/bash
# =============================================================================
# MASTER TRAINING SCRIPT (Robust End-to-End Pipeline)
# Runs complete research pipeline with proper error handling
# =============================================================================
#
# Features improved:
#   - Proper checkpoint validation before probe
#   - Sequential execution with clear dependencies
#   - Better error handling and logging
#   - GPU memory check with fallback
#
# Usage:
#   ./run_all.sh              # Run everything
#   ./run_all.sh --dry-run    # Preview mode
#   ./run_all.sh --quick      # Quick test (10 epochs)
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
QUICK_MODE=false

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            echo "[DRY RUN MODE]"
            ;;
        --quick)
            QUICK_MODE=true
            echo "[QUICK MODE - 10 epochs]"
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Find Available GPUs (with fallback)
# -----------------------------------------------------------------------------
find_available_gpus() {
    AVAILABLE_GPUS=()
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo "⚠ nvidia-smi not found, using GPU 0"
        AVAILABLE_GPUS=(0)
        return
    fi
    
    while IFS=, read -r FREE IDX; do
        FREE=$(echo "$FREE" | xargs)
        IDX=$(echo "$IDX" | xargs)
        if [ "$FREE" -ge "20000" ]; then  # Reduced threshold to 20GB
            AVAILABLE_GPUS+=("$IDX")
        fi
    done < <(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits 2>/dev/null || echo "40000,0")
    
    if [ ${#AVAILABLE_GPUS[@]} -eq 0 ]; then
        echo "⚠ No GPU with 20GB+ free, using GPU 0"
        AVAILABLE_GPUS=(0)
    fi
}

# -----------------------------------------------------------------------------
# Pre-flight Checks
# -----------------------------------------------------------------------------
preflight_check() {
    echo "==========================================="
    echo "  Pre-flight Checks"
    echo "==========================================="
    
    find_available_gpus
    echo "✓ Available GPUs: ${AVAILABLE_GPUS[*]}"
    
    # W&B
    export WANDB_API_KEY="${WANDB_API_KEY:-e83326e014ad7a27c2a538f4e38b95bd11a161a0}"
    wandb login $WANDB_API_KEY 2>/dev/null && echo "✓ W&B logged in" || echo "⚠ W&B offline mode"
    
    # Set W&B group for all runs in this experiment
    export WANDB_RUN_GROUP="exp_${TIMESTAMP}"
    echo "✓ W&B group: ${WANDB_RUN_GROUP}"
    
    # Data check
    if [ -d "data/processed_ego4d" ]; then
        DATA_COUNT=$(ls -1 data/processed_ego4d/ 2>/dev/null | wc -l)
        echo "✓ Data: $DATA_COUNT videos"
    else
        echo "✗ data/processed_ego4d not found!"
        exit 1
    fi
    
    # Config check
    if [ ! -f "configs/beta_0.0.yaml" ] || [ ! -f "configs/beta_0.3.yaml" ]; then
        echo "✗ Config files missing!"
        exit 1
    fi
    echo "✓ Configs verified"
    
    echo ""
}

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
setup_env() {
    echo "==========================================="
    echo "  Setup"
    echo "==========================================="
    
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
    
    # Copy configs for reproducibility
    cp configs/beta_0.0.yaml configs/beta_0.3.yaml "$OUTPUT_BASE/"
    echo ""
}

# -----------------------------------------------------------------------------
# Run Training (with proper wait)
# -----------------------------------------------------------------------------
run_training() {
    local NAME="$1"
    local CONFIG="$2"
    local GPU="$3"
    local OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"
    local EPOCHS_OVERRIDE=""
    
    if [ "$QUICK_MODE" == "true" ]; then
        EPOCHS_OVERRIDE="--epochs 10"
    fi
    
    mkdir -p "$OUTPUT_DIR"
    echo ">> [$NAME] Starting on GPU $GPU..."
    
    if [ "$DRY_RUN" == "true" ]; then
        echo "[DRY RUN] python train.py --config $CONFIG --output-dir $OUTPUT_DIR $EPOCHS_OVERRIDE"
        touch "${OUTPUT_DIR}/best_model.pth"  # Create dummy checkpoint for dry run
        return 0
    fi
    
    local START_TIME=$(date +%s)
    
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --config "$CONFIG" \
        --output-dir "$OUTPUT_DIR" \
        $EPOCHS_OVERRIDE \
        2>&1 | tee "${LOG_DIR}/${NAME}.log"
    
    local EXIT_CODE=$?
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))
    
    echo "${NAME}_duration=${DURATION}s" >> "${OUTPUT_BASE}/timings.txt"
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "✗ [$NAME] Training failed with exit code $EXIT_CODE"
        return 1
    fi
    
    # Verify checkpoint exists
    if [ -f "${OUTPUT_DIR}/best_model.pth" ]; then
        echo "✓ [$NAME] Complete (${DURATION}s) - checkpoint saved"
        return 0
    else
        echo "⚠ [$NAME] Complete but no checkpoint saved"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Run Probe (with checkpoint validation)
# -----------------------------------------------------------------------------
run_probe() {
    local NAME="$1"
    local CHECKPOINT="$2"
    local GPU="$3"
    local OUTPUT_DIR="${OUTPUT_BASE}/${NAME}_probe"
    
    mkdir -p "$OUTPUT_DIR"
    
    # Validate checkpoint exists
    if [ ! -f "$CHECKPOINT" ]; then
        echo "⚠ [$NAME Probe] Skipped - checkpoint not found: $CHECKPOINT"
        return 1
    fi
    
    echo ">> [$NAME Probe] Starting on GPU $GPU..."
    
    if [ "$DRY_RUN" == "true" ]; then
        echo "[DRY RUN] python train.py --probe --checkpoint $CHECKPOINT"
        return 0
    fi
    
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --probe \
        --checkpoint "$CHECKPOINT" \
        --config configs/beta_0.0.yaml \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "${LOG_DIR}/${NAME}_probe.log"
    
    local EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "✗ [$NAME Probe] Failed"
        return 1
    fi
    
    echo "✓ [$NAME Probe] Complete"
    return 0
}

# -----------------------------------------------------------------------------
# Run Baselines (sequential for stability)
# -----------------------------------------------------------------------------
run_baselines() {
    local GPU="$1"
    local BASELINE_DIR="${OUTPUT_BASE}/baselines"
    mkdir -p "$BASELINE_DIR"
    
    echo ""
    echo "==========================================="
    echo "  Running Baselines on GPU $GPU"
    echo "==========================================="
    
    if [ "$DRY_RUN" == "true" ]; then
        echo "[DRY RUN] 4 baseline models"
        return 0
    fi
    
    local EPOCHS_OVERRIDE=""
    if [ "$QUICK_MODE" == "true" ]; then
        EPOCHS_OVERRIDE="--epochs 10"
    fi
    
    for MODEL in mlp_mlp cnn_mlp imu2clip cnn_lstm_gru; do
        echo ">> [Baseline: $MODEL] Training..."
        CUDA_VISIBLE_DEVICES=$GPU python scripts/train_baselines.py \
            --model "$MODEL" \
            --config configs/beta_0.3.yaml \
            --output-dir "$BASELINE_DIR" \
            $EPOCHS_OVERRIDE \
            2>&1 | tee "${LOG_DIR}/baseline_${MODEL}.log"
        
        if [ $? -eq 0 ]; then
            echo "✓ [Baseline: $MODEL] Complete"
        else
            echo "⚠ [Baseline: $MODEL] Failed (continuing...)"
        fi
    done
}

# -----------------------------------------------------------------------------
# Extract Metrics from Logs
# -----------------------------------------------------------------------------
extract_metrics() {
    local LOG="$1"
    local NAME="$2"
    
    if [ ! -f "$LOG" ]; then
        echo "$NAME,N/A,N/A,~1.5M,-" >> "$RESULTS_CSV"
        return
    fi
    
    # Extract best scenario F1 (more reliable than last epoch)
    SCENARIO_F1=$(grep -oE "Best model saved \(Scenario F1: [0-9]+\.[0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || \
                  grep -oE "Val Scenario F1: [0-9]+\.[0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
    
    # Extract action F1
    ACTION_F1=$(grep -oE "(Val Action F1|Probe Val Action F1): [0-9]+\.[0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
    
    echo "$NAME,$SCENARIO_F1,$ACTION_F1,~1.5M,-" >> "$RESULTS_CSV"
}

# -----------------------------------------------------------------------------
# Generate Results Table
# -----------------------------------------------------------------------------
generate_results_table() {
    echo ""
    echo "==========================================="
    echo "  BENCHMARK RESULTS"
    echo "==========================================="
    echo ""
    
    # Extract metrics from all logs
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
    cat > "$RESULTS_MD" << EOF
# Benchmark Results

**Generated**: $(date)
**Config**: EgoCHARM features (14 channels)

## Performance Comparison

| Model | Scenario F1 | Action F1 |
|-------|-------------|-----------|
EOF
    
    while IFS=, read -r MODEL SCENARIO ACTION PARAMS TIME; do
        if [ "$MODEL" != "Model" ]; then
            echo "| $MODEL | $SCENARIO | $ACTION |" >> "$RESULTS_MD"
        fi
    done < "$RESULTS_CSV"
    
    cat >> "$RESULTS_MD" << EOF

## Configuration
- **Features**: 14 channels (6 raw + 2 norms + 6 variance)
- **Focal Loss**: gamma=2.0
- **Class Weights**: [5.0, 7.0, 0.5, 10.0]
- **Label Smoothing**: 0.1
- **Augmentation**: Jittering + Scaling + Time Warp + Rotation
EOF
    
    echo "✓ Results saved to: $RESULTS_MD"
}

# -----------------------------------------------------------------------------
# Main (Sequential for Reliability)
# -----------------------------------------------------------------------------
main() {
    preflight_check
    setup_env
    
    GPU0=${AVAILABLE_GPUS[0]}
    GPU1=${AVAILABLE_GPUS[1]:-$GPU0}
    
    echo "==========================================="
    echo "  Training Pipeline"
    echo "==========================================="
    echo "Phase 1: Backbone (β=0.0) on GPU $GPU0"
    echo "Phase 2: Joint (β=0.3) on GPU $GPU1"
    echo "Phase 3: Probes and Baselines"
    echo ""
    
    # Phase 1: Backbone Training
    echo "--- Phase 1: Backbone (β=0.0) ---"
    if run_training "backbone_beta0" "configs/beta_0.0.yaml" "$GPU0"; then
        # Probe immediately after successful backbone
        run_probe "backbone_beta0" "${OUTPUT_BASE}/backbone_beta0/best_model.pth" "$GPU0" || true
    fi
    
    # Phase 2: Joint Training
    echo ""
    echo "--- Phase 2: Joint (β=0.3) ---"
    if run_training "joint_beta03" "configs/beta_0.3.yaml" "$GPU1"; then
        # Probe immediately after successful joint
        run_probe "joint_beta03" "${OUTPUT_BASE}/joint_beta03/best_model.pth" "$GPU1" || true
    fi
    
    # Phase 3: Baselines
    run_baselines "$GPU0"
    
    # Generate results
    generate_results_table
    
    echo ""
    echo "==========================================="
    echo "  PIPELINE COMPLETE!"
    echo "==========================================="
    echo "Output: $OUTPUT_BASE"
    echo "Results: $RESULTS_CSV"
    echo "Report: ${RESULTS_DIR}/benchmark_results.md"
    echo "Finished: $(date)"
}

main
