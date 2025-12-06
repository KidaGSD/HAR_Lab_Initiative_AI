#!/bin/bash
# Multi-Config Training Pipeline for Hierarchical HAR (Parallel Multi-GPU)
# Runs multiple beta configurations (0.0, 0.1, 0.3) in PARALLEL on separate GPUs
# Each config trains, then runs probe evaluation
# Results are logged to W&B for comparison

set -e

echo "==========================================="
echo "  Multi-Config Parallel Training Pipeline"
echo "  Beta values: 0.0, 0.1, 0.3"
echo "==========================================="

# ===========================
# 1. Environment Setup
# ===========================
echo ""
echo "=== Step 1: Environment Setup ==="

if [ -n "$CONDA_PREFIX" ]; then
    echo "Conda already activated: $CONDA_PREFIX"
else
    echo "Activating ego4d_lab environment..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ego4d_lab
fi

export PYTHONPATH="$(pwd):$PYTHONPATH"

# ===========================
# 2. WandB Setup
# ===========================
echo ""
echo "=== Step 2: WandB Configuration ==="

export WANDB_API_KEY=${WANDB_API_KEY:-e83326e014ad7a27c2a538f4e38b95bd11a161a0}
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set"
else
    wandb login $WANDB_API_KEY 2>/dev/null || echo "WandB login attempted"
fi

# ===========================
# 3. Training Configuration
# ===========================
echo ""
echo "=== Step 3: Configuration ==="

PROCESSED_DIR="data/processed_ego4d"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="checkpoints/sweep_${TIMESTAMP}"

# Beta values to sweep
BETAS=("0.0" "0.1" "0.3")

echo "Base output dir: $BASE_OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo "Configs to run: ${BETAS[*]}"
echo ""

mkdir -p "$BASE_OUTPUT_DIR"

# ===========================
# 4. Helper Function
# ===========================

run_experiment() {
    local BETA=$1
    local GPU_ID=$2
    
    local CONFIG="configs/beta_${BETA}.yaml"
    local RUN_NAME="beta_${BETA}_${TIMESTAMP}"
    local OUTPUT_DIR="${BASE_OUTPUT_DIR}/beta_${BETA}"
    
    mkdir -p "$OUTPUT_DIR"
    
    echo ">> [GPU $GPU_ID] Starting beta=$BETA"
    echo "   Config: $CONFIG"
    echo "   Log: ${OUTPUT_DIR}/training.log"
    
    # Run training
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        --config "$CONFIG" \
        --processed-dir "$PROCESSED_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --run-name "$RUN_NAME" \
        > "${OUTPUT_DIR}/training.log" 2>&1
        
    local TRAIN_EXIT=$?
    
    if [ $TRAIN_EXIT -eq 0 ]; then
        echo ">> [GPU $GPU_ID] Training finished for beta=$BETA"
        
        # Run probe
        if [ -f "${OUTPUT_DIR}/best_model.pth" ]; then
            echo ">> [GPU $GPU_ID] Running probe for beta=$BETA"
            local PROBE_OUTPUT_DIR="${OUTPUT_DIR}/probe"
            local PROBE_RUN_NAME="probe_${RUN_NAME}"
            
            CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
                --probe \
                --checkpoint "${OUTPUT_DIR}/best_model.pth" \
                --config "$CONFIG" \
                --processed-dir "$PROCESSED_DIR" \
                --output-dir "$PROBE_OUTPUT_DIR" \
                --run-name "$PROBE_RUN_NAME" \
                > "${PROBE_OUTPUT_DIR}/probe.log" 2>&1
                
            echo ">> [GPU $GPU_ID] Probe finished for beta=$BETA"
        else
            echo ">> [GPU $GPU_ID] WARNING: No best_model.pth found for beta=$BETA, skipping probe"
        fi
    else
        echo ">> [GPU $GPU_ID] ERROR: Training failed for beta=$BETA (Exit code $TRAIN_EXIT)"
        echo "   Check log: ${OUTPUT_DIR}/training.log"
    fi
}

# ===========================
# 5. Parallel Execution
# ===========================
echo "=== Step 4: Launching Parallel Jobs ==="

# Get available GPUs
# This gets a comma-separated list of indices, e.g. "0, 1, 2"
AVAILABLE_GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ','))
NUM_GPUS=${#AVAILABLE_GPUS[@]}

echo "Detected ${NUM_GPUS} GPUs: ${AVAILABLE_GPUS[*]}"

if [ $NUM_GPUS -lt 1 ]; then
    echo "Error: No GPUs detected!"
    exit 1
fi

PIDS=()

for i in "${!BETAS[@]}"; do
    BETA=${BETAS[$i]}
    # Round-robin assignment if fewer GPUs than jobs
    GPU_IDX=$((i % NUM_GPUS))
    GPU_ID=${AVAILABLE_GPUS[$GPU_IDX]}
    
    echo "Assigning beta=$BETA to GPU $GPU_ID"
    
    run_experiment "$BETA" "$GPU_ID" &
    PID=$!
    PIDS+=($PID)
    echo "Job launched with PID $PID"
    
    # stagger starts slightly to avoid race conditions on file creation etc
    sleep 5
done

echo ""
echo "All jobs launched. Waiting for completion..."
echo "Tail logs with:"
for BETA in "${BETAS[@]}"; do
    echo "  tail -f ${BASE_OUTPUT_DIR}/beta_${BETA}/training.log"
done
echo ""

# Wait for all background jobs
for PID in "${PIDS[@]}"; do
    wait $PID
done

# ===========================
# 6. Summary
# ===========================
echo ""
echo "==========================================="
echo "  Multi-Config Parallel Training Complete!"
echo "==========================================="
echo ""
echo "Results saved to: $BASE_OUTPUT_DIR"
echo ""
echo "Checkpoints:"
for BETA in "${BETAS[@]}"; do
    echo "  beta=$BETA: ${BASE_OUTPUT_DIR}/beta_${BETA}/best_model.pth"
done
echo ""
echo "W&B Dashboard: https://wandb.ai/wandbleo/har-imu-training"
echo ""
