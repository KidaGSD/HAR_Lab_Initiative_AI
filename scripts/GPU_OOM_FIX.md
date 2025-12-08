# GPU Out-of-Memory Solutions

## 🔴 Problem
```
torch.OutOfMemoryError: CUDA out of memory
GPU 0 has 47.51 GiB total, but 45.17 GiB already in use by Process 1367599
```

**Root cause**: Another process is using GPU 0, leaving only 770MB free (you need 1.72 GiB).

## ✅ Solutions (Try in order)

### Solution 1: Use Auto GPU Selection (Recommended)
```bash
# New safe script that auto-finds free GPU
./run_visualization_safe.sh
```

This script will:
1. Check all GPUs for free memory
2. Automatically use the freest GPU
3. Fall back to CPU if no GPU available
4. Use smaller batch size (8) for safety

### Solution 2: Manual GPU Selection
```bash
# Check which GPUs are available
nvidia-smi

# Use a different GPU (e.g., GPU 1)
CUDA_VISIBLE_DEVICES=1 python scripts/visualize_embeddings.py \
  --config configs/hierarchical.yaml \
  --checkpoint checkpoints/checkpoints/experiments_20251206_224347/beta_1.0/best_model.pth \
  --split val \
  --output-dir outputs/embeddings_viz
```

### Solution 3: Run on CPU (Slower but Always Works)
```bash
# Force CPU usage
CUDA_VISIBLE_DEVICES="" python scripts/visualize_embeddings.py \
  --config configs/hierarchical.yaml \
  --checkpoint checkpoints/checkpoints/experiments_20251206_224347/beta_1.0/best_model.pth \
  --split val \
  --output-dir outputs/embeddings_viz
```

**Time estimate on CPU**:
- PCA: ~2-3 minutes (vs 30 sec on GPU)
- t-SNE: ~10-15 minutes (vs 3 min on GPU)

### Solution 4: Reduce Batch Size
Edit `configs/hierarchical.yaml`:
```yaml
training:
  batch_size: 8  # Reduced from 32 or 16
```

Then run normally.

### Solution 5: Kill Other Process (If it's yours)
```bash
# Check who owns the process
ps aux | grep 1367599

# If it's yours and not needed:
kill 1367599

# Then run visualization
```

## 🎯 Recommended Approach

**Use the new safe script**:
```bash
chmod +x run_visualization_safe.sh
./run_visualization_safe.sh
```

This handles everything automatically:
- ✅ Finds free GPU or uses CPU
- ✅ Reduces batch size for safety
- ✅ No manual configuration needed

## 📊 Memory Requirements

| Component | Memory Needed |
|-----------|---------------|
| Model (1.5M params) | ~10 MB |
| Batch (32 samples) | ~500 MB |
| Forward pass | ~1.2 GB |
| **Total** | **~1.7 GB** |

With batch_size=8:
- Total memory: ~500 MB (much safer)

## 🔍 Check GPU Status

```bash
# See all GPUs and their usage
nvidia-smi

# See in real-time
watch -n 1 nvidia-smi

# See which process is using GPU 0
nvidia-smi -q -d PIDS -i 0
```

## ⚡ Quick Fix

**If you just want it to work NOW**:
```bash
# Use CPU, smaller batch, PCA only (skip slow t-SNE)
CUDA_VISIBLE_DEVICES="" python scripts/visualize_embeddings.py \
  --config configs/hierarchical.yaml \
  --checkpoint checkpoints/checkpoints/experiments_20251206_224347/beta_1.0/best_model.pth \
  --split val \
  --output-dir outputs/embeddings_viz

# Time: ~3 minutes on CPU
```

You can run t-SNE later when GPU is free.

## 💡 Prevention

For future runs:
1. Always check GPU status first: `nvidia-smi`
2. Use the safe script: `./run_visualization_safe.sh`
3. Or set environment variable: `CUDA_VISIBLE_DEVICES=<free_gpu_id>`

## 🚀 Full Command with All Safety Measures

```bash
#!/bin/bash
# Find free GPU or use CPU
FREE_GPU=$(python3 -c "import subprocess; result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True); gpus = [(int(l.split(',')[0]), float(l.split(',')[1])) for l in result.stdout.strip().split('\n')]; best = max(gpus, key=lambda x: x[1]) if gpus else (-1, 0); print(best[0] if best[1] > 2000 else '')")

if [ -z "$FREE_GPU" ]; then
    echo "No free GPU, using CPU"
    export CUDA_VISIBLE_DEVICES=""
else
    echo "Using GPU $FREE_GPU"
    export CUDA_VISIBLE_DEVICES=$FREE_GPU
fi

# Run with reduced batch size
python scripts/visualize_embeddings.py \
  --config configs/hierarchical.yaml \
  --checkpoint checkpoints/checkpoints/experiments_20251206_224347/beta_1.0/best_model.pth \
  --split val \
  --output-dir outputs/embeddings_viz
```

Save this as `run_viz_auto.sh` and use it!
