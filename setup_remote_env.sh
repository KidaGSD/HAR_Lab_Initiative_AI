#!/bin/bash
# Quick setup script - run this on the remote server

echo "=========================================="
echo "HAR Project - Remote Server Setup"
echo "=========================================="
echo ""

# Check if conda exists
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install conda first:"
    echo "   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "   bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"
echo ""

# Create environment
echo "📦 Creating conda environment 'har_gpu' with Python 3.10..."
conda create -n har_gpu python=3.10 -y

# Activate (need to source conda first)
eval "$(conda shell.bash hook)"
conda activate har_gpu

echo "✅ Environment created and activated"
echo ""

# Install PyTorch
echo "🔥 Installing PyTorch with CUDA support..."
echo "   (Assuming CUDA 11.8 - adjust if different)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "✅ PyTorch installed"
echo ""

# Install dependencies
echo "📚 Installing dependencies..."
pip install numpy pandas matplotlib seaborn tqdm jupyter ipykernel wandb pyyaml

echo "✅ Dependencies installed"
echo ""

# Setup Jupyter kernel
echo "🔧 Setting up Jupyter kernel..."
python -m ipykernel install --user --name har_gpu --display-name "Python (har_gpu)"

echo "✅ Jupyter kernel configured"
echo ""

# Test GPU
echo "🧪 Testing GPU access..."
python -c "import torch; print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To activate environment:"
echo "   conda activate har_gpu"
echo ""
echo "To start Jupyter:"
echo "   jupyter notebook --no-browser --port=8888"
echo ""
