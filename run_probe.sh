#!/bin/bash
# Action probe training on frozen encoder
# Usage: ./run_probe.sh --checkpoint <path> [--processed-dir ...] [--output-dir ...] [--run-name ...]

set -e

# Defaults
PROCESSED_DIR="data/processed_ego4d"
OUTPUT_DIR="checkpoints_probe"
RUN_NAME="probe_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT=""
CONFIG="configs/hierarchical.yaml"

# Parse args
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --checkpoint)
      CHECKPOINT="$2"; shift; shift;
      ;;
    --processed-dir)
      PROCESSED_DIR="$2"; shift; shift;
      ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift; shift;
      ;;
    --run-name)
      RUN_NAME="$2"; shift; shift;
      ;;
    --config)
      CONFIG="$2"; shift; shift;
      ;;
    *) shift; ;;
  esac
done

# Auto-pick latest best_model.pth if not provided
if [ -z "$CHECKPOINT" ]; then
  CHECKPOINT=$(ls -t checkpoints/best_model.pth checkpoints/fold*/best_model.pth 2>/dev/null | head -1)
  if [ -z "$CHECKPOINT" ]; then
    echo "No checkpoint found automatically; please specify --checkpoint"; exit 1; fi
  echo "Auto-selected checkpoint: $CHECKPOINT"
fi

# Ensure local src is importable
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Auto-pick GPU if not set
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  MIN_FREE_MB=${MIN_FREE_MB:-20000}
  PICKED=""
  while IFS=, read -r FREE IDX; do
    FREE=$(echo "$FREE" | xargs)
    IDX=$(echo "$IDX" | xargs)
    if [ "$FREE" -ge "$MIN_FREE_MB" ]; then
      PICKED=$IDX; break; fi
  done < <(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits | sort -nr)
  if [ -z "$PICKED" ]; then
    PICKED=$(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits | sort -nr | head -1 | awk -F',' '{print $2}' | xargs)
    echo "Warning: no GPU meets MIN_FREE_MB=${MIN_FREE_MB}MB, picking best available: $PICKED"
  fi
  export CUDA_VISIBLE_DEVICES=${PICKED:-0}
fi

echo "Using GPU: $CUDA_VISIBLE_DEVICES"

python train.py \
  --probe \
  --checkpoint "$CHECKPOINT" \
  --processed-dir "$PROCESSED_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --config "$CONFIG" \
  "$@"
