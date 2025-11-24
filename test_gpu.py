#!/usr/bin/env python3
"""Quick GPU test script - run this on the GPU server to verify setup"""

import torch
import sys

print("=" * 60)
print("GPU Detection Test")
print("=" * 60)

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"\n✅ GPU Detected!")
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Count: {torch.cuda.device_count()}")
    print(f"   CUDA Version: {torch.version.cuda}")
    
    props = torch.cuda.get_device_properties(0)
    print(f"   GPU Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"   Compute Capability: {props.major}.{props.minor}")
    
    # Test GPU computation
    print(f"\n🧪 Testing GPU computation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("   ✅ GPU computation test passed!")
        print(f"   Result shape: {z.shape}")
    except Exception as e:
        print(f"   ❌ GPU computation failed: {e}")
else:
    print(f"\n❌ No GPU detected")
    print(f"\nTroubleshooting:")
    print(f"   1. Are you on the GPU server? (not your local Mac)")
    print(f"   2. Run: nvidia-smi (should show GPU info)")
    print(f"   3. Check PyTorch CUDA support:")
    print(f"      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print(f"   4. Verify CUDA version: nvcc --version")

print("=" * 60)

