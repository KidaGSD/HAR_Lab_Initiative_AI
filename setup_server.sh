#!/bin/bash
# Server Setup Script for HAR Training
set -e

echo "=== Setting up HAR Training Environment ==="

# Clean up any previous failed installations
echo "Cleaning conda cache..."
conda clean --all -y

# Remove old environment if exists
conda env remove -n ego4d_lab -y 2>/dev/null || true

# Create fresh environment with Python 3.10
echo "Creating conda environment..."
conda create -n ego4d_lab python=3.10 -y

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ego4d_lab

# Install conda packages (non-PyTorch dependencies)
echo "Installing conda packages..."
conda install -y \
    pandas \
    numpy \
    scipy \
    tqdm \
    matplotlib \
    seaborn

# Install PyTorch with CUDA 12.1 via pip (more reliable)
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other pip packages
echo "Installing additional packages..."
pip install boto3 ego4d wandb scikit-learn

# Setup WandB
echo ""
echo "=== Setting up WandB ===\"
echo "WandB API Key: e83326e014ad7a27c2a538f4e38b95bd11a161a0"
export WANDB_API_KEY=e83326e014ad7a27c2a538f4e38b95bd11a161a0
wandb login $WANDB_API_KEY

# Verify installation
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU 0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Setup Complete! ==="
echo "To activate: conda activate ego4d_lab"
echo "To verify GPU: nvidia-smi"
