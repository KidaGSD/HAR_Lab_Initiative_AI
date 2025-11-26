#!/bin/bash
# Setup script for LLM Labeling (Qwen + vLLM)
set -e

echo "=== Setting up LLM Labeling Environment ==="

# 1. Install vLLM (Fast Inference Engine)
echo "Installing vLLM..."
pip install vllm

# 2. Install HuggingFace Hub (to download model)
pip install huggingface_hub

# 3. Download Qwen Model (Quantized for speed/memory)
# We use Qwen1.5-14B-Chat-AWQ (4-bit quantized)
# It fits easily on RTX 6000 (needs ~10GB VRAM) and is very fast.
echo "Downloading Qwen1.5-14B-Chat-AWQ..."
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen1.5-14B-Chat-AWQ')"

echo ""
echo "=== Setup Complete! ==="
echo "You can now run the labeling script:"
echo "python scripts/label_with_qwen.py --model Qwen/Qwen1.5-14B-Chat-AWQ --gpus 1"
