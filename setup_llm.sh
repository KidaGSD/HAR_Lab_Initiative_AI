#!/bin/bash
# Setup script for LLM Labeling (Qwen + vLLM)
set -e

echo "=== Setting up LLM Labeling Environment ==="

# 1. Install vLLM (Fast Inference Engine)
echo "Installing vLLM..."
pip install vllm

# 2. Install Python Dependencies
# Qwen3 requires latest transformers and vllm >= 0.8.5
echo "Installing/Updating dependencies..."
pip install --upgrade "vllm>=0.8.5" "transformers>=4.51.0" huggingface_hub

# 3. Download Qwen Model
# We use Qwen3-14B (Latest generation with thinking capabilities)
echo "Downloading Qwen3-14B..."
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-14B')"

echo ""
echo "=== Setup Complete! ==="
echo "You can now run the labeling script:"
echo "python scripts/label_with_qwen.py --model Qwen/Qwen3-14B --gpus 1"
