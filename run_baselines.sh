#!/bin/bash
# Baseline Models Runner
# Trains all 4 baseline models in parallel on separate GPUs
# Usage: ./run_baselines.sh

set -e

echo "==========================================="
echo "  Baseline Models Training"
echo "  Models: MLP-MLP, CNN-MLP, IMU2CLIP, CNN-LSTM-GRU"
echo "==========================================="

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="checkpoints/baselines_${TIMESTAMP}"
CONFIG="configs/beta_0.3.yaml"
BASELINES=("mlp_mlp" "cnn_mlp" "imu2clip" "cnn_lstm_gru")

# Environment
if [ -n "$CONDA_PREFIX" ]; then
    echo "Conda: $CONDA_PREFIX"
else
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ego4d_lab
fi

export PYTHONPATH="$(pwd):$PYTHONPATH"
export WANDB_API_KEY=${WANDB_API_KEY:-e83326e014ad7a27c2a538f4e38b95bd11a161a0}
wandb login $WANDB_API_KEY 2>/dev/null || true

mkdir -p "$OUTPUT_DIR"

# Detect GPUs
AVAILABLE_GPUS=()
while IFS=, read -r FREE IDX; do
    FREE=$(echo "$FREE" | xargs)
    IDX=$(echo "$IDX" | xargs)
    if [ "$FREE" -ge "15000" ]; then
        AVAILABLE_GPUS+=("$IDX")
    fi
done < <(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits)

NUM_GPUS=${#AVAILABLE_GPUS[@]}
echo "Found $NUM_GPUS available GPUs: ${AVAILABLE_GPUS[*]}"

if [ $NUM_GPUS -lt 1 ]; then
    echo "No GPUs available, running sequentially on GPU 0"
    AVAILABLE_GPUS=(0)
    NUM_GPUS=1
fi

# Run baselines
run_baseline() {
    local MODEL=$1
    local GPU=$2
    
    echo ">> [GPU $GPU] Starting $MODEL"
    
    CUDA_VISIBLE_DEVICES=$GPU python scripts/train_baselines.py \
        --model "$MODEL" \
        --config "$CONFIG" \
        --output-dir "$OUTPUT_DIR" \
        --run-suffix "$TIMESTAMP" \
        > "${OUTPUT_DIR}/${MODEL}.log" 2>&1
    
    echo ">> [GPU $GPU] $MODEL completed"
}

PIDS=()

for i in "${!BASELINES[@]}"; do
    MODEL=${BASELINES[$i]}
    GPU_IDX=$((i % NUM_GPUS))
    GPU=${AVAILABLE_GPUS[$GPU_IDX]}
    
    run_baseline "$MODEL" "$GPU" &
    PIDS+=($!)
    sleep 3
done

echo ""
echo "All baselines launched. Waiting..."
echo "Monitor: tail -f ${OUTPUT_DIR}/*.log"

for PID in "${PIDS[@]}"; do
    wait $PID
done

echo ""
echo "==========================================="
echo "  Baseline Training Complete!"
echo "==========================================="
echo "Results: ${OUTPUT_DIR}/baseline_results.csv"
echo ""

# Display results
if [ -f "${OUTPUT_DIR}/baseline_results.csv" ]; then
    echo "Results:"
    cat "${OUTPUT_DIR}/baseline_results.csv"
fi
