#!/bin/bash
# Master Experiment Runner - 2 Day Sprint
# Runs all experiments in parallel, auto-collects results, auto-generates report
# Usage: ./run_experiments.sh [day1|day2|all]

set -e

echo "==========================================="
echo "  HAR-IMU Master Experiment Runner"
echo "  2-Day Sprint - Research Grade Pipeline"
echo "==========================================="

# ===========================
# Configuration
# ===========================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="checkpoints/experiments_${TIMESTAMP}"
REPORT_DIR="reports"
LOG_FILE="${REPORT_DIR}/experiment_log_${TIMESTAMP}.txt"

# Experiment configs
DAY1_CONFIGS=("beta_0.5" "beta_0.7" "beta_1.0")
DAY2_CONFIGS=("beta_0.3_cv" "beta_0.5_cv")  # CV validation

# ===========================
# Setup
# ===========================
setup_environment() {
    echo ""
    echo "=== Environment Setup ==="
    
    if [ -n "$CONDA_PREFIX" ]; then
        echo "Conda: $CONDA_PREFIX"
    else
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate ego4d_lab
    fi
    
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    export WANDB_API_KEY=${WANDB_API_KEY:-e83326e014ad7a27c2a538f4e38b95bd11a161a0}
    wandb login $WANDB_API_KEY 2>/dev/null || true
    
    mkdir -p "$BASE_OUTPUT_DIR" "$REPORT_DIR"
    
    # Log start
    echo "Experiment started: $(date)" | tee "$LOG_FILE"
    echo "Output dir: $BASE_OUTPUT_DIR" | tee -a "$LOG_FILE"
}

# ===========================
# GPU Detection
# ===========================
detect_gpus() {
    echo ""
    echo "=== GPU Detection ==="
    
    AVAILABLE_GPUS=()
    
    while IFS=, read -r FREE IDX; do
        FREE=$(echo "$FREE" | xargs)
        IDX=$(echo "$IDX" | xargs)
        if [ "$FREE" -ge "15000" ]; then
            AVAILABLE_GPUS+=("$IDX")
            echo "  GPU $IDX: ${FREE}MB free - AVAILABLE"
        else
            echo "  GPU $IDX: ${FREE}MB free - skipped"
        fi
    done < <(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits)
    
    NUM_GPUS=${#AVAILABLE_GPUS[@]}
    echo ""
    echo "Using $NUM_GPUS GPUs: ${AVAILABLE_GPUS[*]}"
    
    if [ $NUM_GPUS -lt 1 ]; then
        echo "ERROR: No GPUs available!"
        exit 1
    fi
}

# ===========================
# Run Single Experiment
# ===========================
run_experiment() {
    local CONFIG_NAME=$1
    local GPU_ID=$2
    local USE_CV=$3
    
    local CONFIG="configs/${CONFIG_NAME}.yaml"
    local RUN_NAME="${CONFIG_NAME}_${TIMESTAMP}"
    local OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CONFIG_NAME}"
    
    mkdir -p "$OUTPUT_DIR"
    
    echo ">> [GPU $GPU_ID] Starting $CONFIG_NAME" | tee -a "$LOG_FILE"
    
    # Build command
    local CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python train.py --config $CONFIG --processed-dir data/processed_ego4d --output-dir $OUTPUT_DIR --run-name $RUN_NAME"
    
    if [ "$USE_CV" = "true" ]; then
        CMD="$CMD --cv --n-folds 4"
    fi
    
    # Run training
    eval $CMD > "${OUTPUT_DIR}/training.log" 2>&1
    local EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ">> [GPU $GPU_ID] ✓ $CONFIG_NAME completed" | tee -a "$LOG_FILE"
        
        # Auto-run probe if single split
        if [ "$USE_CV" != "true" ] && [ -f "${OUTPUT_DIR}/best_model.pth" ]; then
            echo ">> [GPU $GPU_ID] Running probe for $CONFIG_NAME" | tee -a "$LOG_FILE"
            mkdir -p "${OUTPUT_DIR}/probe"
            CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
                --probe \
                --checkpoint "${OUTPUT_DIR}/best_model.pth" \
                --config "$CONFIG" \
                --processed-dir data/processed_ego4d \
                --output-dir "${OUTPUT_DIR}/probe" \
                --run-name "probe_${RUN_NAME}" \
                > "${OUTPUT_DIR}/probe/probe.log" 2>&1 || true
        fi
    else
        echo ">> [GPU $GPU_ID] ✗ $CONFIG_NAME FAILED" | tee -a "$LOG_FILE"
    fi
}

# ===========================
# Run Analysis
# ===========================
run_analysis() {
    echo ""
    echo "=== Generating Analysis Report ===" | tee -a "$LOG_FILE"
    python scripts/experiment_analysis.py 2>&1 | tee -a "$LOG_FILE"
}

# ===========================
# Day 1: Beta Sweep
# ===========================
run_day1() {
    echo ""
    echo "==========================================="
    echo "  DAY 1: Extended Beta Sweep"
    echo "  Configs: ${DAY1_CONFIGS[*]}"
    echo "==========================================="
    
    PIDS=()
    
    for i in "${!DAY1_CONFIGS[@]}"; do
        CONFIG=${DAY1_CONFIGS[$i]}
        GPU_IDX=$((i % NUM_GPUS))
        GPU_ID=${AVAILABLE_GPUS[$GPU_IDX]}
        
        run_experiment "$CONFIG" "$GPU_ID" "false" &
        PIDS+=($!)
        sleep 5
    done
    
    echo ""
    echo "Day 1 experiments launched. Waiting..."
    echo "Monitor with: tail -f ${BASE_OUTPUT_DIR}/*/training.log"
    
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    
    echo "Day 1 experiments complete!" | tee -a "$LOG_FILE"
    run_analysis
}

# ===========================
# Day 2: CV Validation
# ===========================
run_day2() {
    echo ""
    echo "==========================================="
    echo "  DAY 2: Cross-Validation"
    echo "  Configs: ${DAY2_CONFIGS[*]}"
    echo "==========================================="
    
    PIDS=()
    
    for i in "${!DAY2_CONFIGS[@]}"; do
        CONFIG=${DAY2_CONFIGS[$i]/_cv/}  # Remove _cv suffix for config file
        GPU_IDX=$((i % NUM_GPUS))
        GPU_ID=${AVAILABLE_GPUS[$GPU_IDX]}
        
        run_experiment "$CONFIG" "$GPU_ID" "true" &
        PIDS+=($!)
        sleep 5
    done
    
    echo ""
    echo "Day 2 experiments launched. Waiting..."
    
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
    
    echo "Day 2 experiments complete!" | tee -a "$LOG_FILE"
    run_analysis
}

# ===========================
# Main
# ===========================
main() {
    local MODE=${1:-all}
    
    setup_environment
    detect_gpus
    
    case $MODE in
        day1)
            run_day1
            ;;
        day2)
            run_day2
            ;;
        all)
            run_day1
            echo ""
            echo "Day 1 complete. Proceeding to Day 2..."
            run_day2
            ;;
        analysis)
            run_analysis
            ;;
        *)
            echo "Usage: $0 [day1|day2|all|analysis]"
            exit 1
            ;;
    esac
    
    echo ""
    echo "==========================================="
    echo "  All Experiments Complete!"
    echo "==========================================="
    echo "Results: $BASE_OUTPUT_DIR"
    echo "Report: $REPORT_DIR/experiment_report.md"
    echo "Log: $LOG_FILE"
    echo "W&B: https://wandb.ai/wandbleo/har-imu-training"
}

main "$@"
