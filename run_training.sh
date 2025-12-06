#!/bin/bash
# Multi-Config Training Pipeline for Hierarchical HAR (Parallel Multi-GPU)
# Runs multiple beta configurations (0.0, 0.1, 0.3) in PARALLEL on separate GPUs
# Usage: ./run_training.sh [GPU_ID_1] [GPU_ID_2] [GPU_ID_3]
# Example: ./run_training.sh 0 1 2
# If no GPUs provided, tries to auto-detect 3 GPUs with >20GB free.

set -e

echo "==========================================="
echo "  Multi-Config Parallel Training Pipeline"
echo "  Beta values: 0.0, 0.1, 0.3"
echo "==========================================="

# Parse optional GPU arguments
MANUAL_GPUS=("$@")

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
NUM_EXPERIMENTS=${#BETAS[@]}

echo "Base output dir: $BASE_OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo "Configs to run: ${BETAS[*]} ($NUM_EXPERIMENTS experiments)"
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
            mkdir -p "$PROBE_OUTPUT_DIR"
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
# 5. GPU Selection
# ===========================
echo "=== Step 4: GPU Selection ==="

AVAILABLE_GPUS=()

if [ ${#MANUAL_GPUS[@]} -gt 0 ]; then
    echo "Using manually specified GPUs: ${MANUAL_GPUS[*]}"
    AVAILABLE_GPUS=("${MANUAL_GPUS[@]}")
else
    # Auto-detect GPUs with >20GB free memory
    echo "Auto-detecting GPUs with >20GB free memory..."
    echo ""
    echo "Current GPU status:"
    nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv
    echo ""
    
    while IFS=, read -r FREE IDX; do
        FREE=$(echo "$FREE" | xargs)
        IDX=$(echo "$IDX" | xargs)
        if [ "$FREE" -ge "20000" ]; then
            AVAILABLE_GPUS+=("$IDX")
            echo "  GPU $IDX: ${FREE}MB free - SELECTED"
        else
            echo "  GPU $IDX: ${FREE}MB free - skipped (need >20GB)"
        fi
    done < <(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits)
fi

NUM_GPUS=${#AVAILABLE_GPUS[@]}
echo ""
echo "Found $NUM_GPUS GPUs with sufficient memory: ${AVAILABLE_GPUS[*]}"

# ===========================
# 6. Execution Mode Decision
# ===========================
echo ""
echo "=== Step 5: Execution Mode ==="

if [ $NUM_GPUS -ge $NUM_EXPERIMENTS ]; then
    echo "MODE: PARALLEL (each experiment gets its own GPU)"
    PARALLEL_MODE=true
elif [ $NUM_GPUS -ge 1 ]; then
    echo "MODE: SEQUENTIAL (not enough GPUs for parallel, will run one-by-one on GPU ${AVAILABLE_GPUS[0]})"
    PARALLEL_MODE=false
else
    echo "ERROR: No GPUs with >20GB free memory found!"
    echo "Please specify GPU IDs manually: ./run_training.sh 0 1 2"
    exit 1
fi

# ===========================
# 7. Launch Experiments
# ===========================
echo ""
echo "=== Step 6: Launching Experiments ==="

PIDS=()

if [ "$PARALLEL_MODE" = true ]; then
    # Parallel execution
    for i in "${!BETAS[@]}"; do
        BETA=${BETAS[$i]}
        GPU_ID=${AVAILABLE_GPUS[$i]}
        
        echo "Launching beta=$BETA on GPU $GPU_ID"
        
        run_experiment "$BETA" "$GPU_ID" &
        PID=$!
        PIDS+=($PID)
        echo "  PID: $PID"
        
        # Stagger starts slightly
        sleep 5
    done
    
    echo ""
    echo "All $NUM_EXPERIMENTS jobs launched in parallel."
    echo "Tail logs with:"
    for BETA in "${BETAS[@]}"; do
        echo "  tail -f ${BASE_OUTPUT_DIR}/beta_${BETA}/training.log"
    done
    echo ""
    echo "Waiting for all jobs to complete..."
    
    # Wait for all background jobs
    for PID in "${PIDS[@]}"; do
        wait $PID
    done
else
    # Sequential execution
    GPU_ID=${AVAILABLE_GPUS[0]}
    for BETA in "${BETAS[@]}"; do
        echo "Running beta=$BETA on GPU $GPU_ID (sequential)"
        run_experiment "$BETA" "$GPU_ID"
        echo ""
    done
fi

# ===========================
# 8. Summary
# ===========================
echo ""
echo "==========================================="
echo "  Multi-Config Training Complete!"
echo "==========================================="
echo ""
echo "Results saved to: $BASE_OUTPUT_DIR"
echo ""
echo "Checkpoints:"
for BETA in "${BETAS[@]}"; do
    if [ -f "${BASE_OUTPUT_DIR}/beta_${BETA}/best_model.pth" ]; then
        echo "  ✓ beta=$BETA: ${BASE_OUTPUT_DIR}/beta_${BETA}/best_model.pth"
    else
        echo "  ✗ beta=$BETA: FAILED (check log)"
    fi
done
echo ""
echo "W&B Dashboard: https://wandb.ai/wandbleo/har-imu-training"
echo ""
