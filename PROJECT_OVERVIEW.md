# Project Overview: Hierarchical Activity Recognition

## Current Project Status

### Architecture
- **LLE (Low-Level Encoder)**: Multi-dilation CNN (dilations [1,2,4]) → GRU → 128-dim embedding
- **HLA (High-Level Architecture)**: Transformer Encoder (2 layers, 4 heads) → 7 scenario classes  
- **Input**: IMU data only (8 channels: 6 raw + 2 norm features)
- **Gaze Data**: Available in processed files but **NOT currently used**

### Data
- **Scenarios**: 7 classes (Gardening excluded)
- **Actions**: 6 classes (Stationary, Locomotion, Essential Operation, Object Transfer, Search, Error/Correction)
- **Videos**: ~1,652 with IMU data
- **Labeled Windows**: ~355,580 windows

---

## Checkpoint Files Overview

### Location: `checkpoints/`

```
checkpoints/
├── fold1/
│   ├── best_model.pth    # Best model (highest val F1)
│   └── last_model.pth    # Last epoch model
├── fold2/
│   ├── best_model.pth
│   └── last_model.pth
├── fold3/
│   ├── best_model.pth
│   └── last_model.pth
├── fold4/
│   ├── best_model.pth
│   └── last_model.pth
├── best_model.pth         # Single training run
└── last_model.pth         # Single training run
```

### What's Saved in Checkpoints

**best_model.pth / last_model.pth:**
- ✅ Model weights (`state_dict`) - All layer parameters
- ❌ Hyperparameters - Stored in WandB
- ❌ Training metrics - Stored in WandB  
- ❌ Training history - Stored in WandB

**What you CAN do:**
1. Load model for inference/evaluation
2. Continue training from checkpoint
3. Extract model architecture info

**What you CANNOT do:**
- See training loss curves (need WandB)
- See hyperparameters (need WandB)
- See validation metrics (need WandB)

---

## How to Access Results

### Option 1: WandB Dashboard (Recommended)

**Access:** https://wandb.ai/wandbleo/har-imu-training

**What you'll find:**
- ✅ Training/validation loss curves
- ✅ F1 scores, accuracy metrics
- ✅ Hyperparameters (learning rate, batch size, etc.)
- ✅ Confusion matrices
- ✅ Model architecture details
- ✅ Training logs

**Run names format:**
- `hierarchical-hierarchical_har_fold1`
- `hierarchical-hierarchical_har_fold2`
- etc.

### Option 2: Inspect Checkpoints Directly

**Script:** `scripts/inspect_checkpoints.py`

```bash
# Inspect all folds
python scripts/inspect_checkpoints.py --all-folds

# Inspect specific checkpoint
python scripts/inspect_checkpoints.py --checkpoint checkpoints/fold1/best_model.pth

# Get WandB results
python scripts/inspect_checkpoints.py --wandb --run-name "hierarchical_har_fold1"
```

**What it shows:**
- Model architecture (LLE/HLA structure)
- Parameter count
- Layer names and shapes
- Architecture type (multi-dilation CNN vs baseline)

### Option 3: Evaluate Models

**Script:** `scripts/test_model.py`

```bash
# Evaluate a checkpoint
python scripts/test_model.py --checkpoint checkpoints/fold1/best_model.pth

# This will show:
# - Scenario F1 (weighted & macro)
# - Action F1 (weighted & macro)  
# - Accuracy scores
# - Confusion matrices
# - Per-class metrics
```

### Option 4: Extract Metrics from WandB

**Script:** `scripts/compute_metrics_from_wandb.py`

```bash
# Get metrics for a specific run
python scripts/compute_metrics_from_wandb.py --run-name "hierarchical_har_fold1"

# Output: JSON file with all metrics
```

---

## Understanding Cross-Validation Folds

### What are the folds?

Your project uses **4-fold cross-validation**:
- **fold1/**: Trained on folds 2,3,4, validated on fold 1
- **fold2/**: Trained on folds 1,3,4, validated on fold 2
- **fold3/**: Trained on folds 1,2,4, validated on fold 3
- **fold4/**: Trained on folds 1,2,3, validated on fold 4

### Why cross-validation?

- More robust performance estimates
- Reduces overfitting to specific train/test split
- Better generalization assessment

### How to use fold results:

1. **Average performance**: Average metrics across all 4 folds
2. **Best fold**: Use the fold with highest validation F1
3. **Ensemble**: Combine predictions from all 4 folds

---

## Current Performance Issues

Based on your feedback:
- ❌ Low F1 scores
- ❌ Action confusion (especially Search vs Stationary)
- ❌ Overall performance needs improvement

### Why gaze integration will help:

1. **Search vs Stationary**: Gaze patterns are very different
   - Search: High dispersion, many saccades, scanning
   - Stationary: Low velocity, stable fixations

2. **Better context**: Gaze shows where attention is focused
   - Essential Operation: Focused fixations on work area
   - Object Transfer: Brief gaze shifts when picking/placing

3. **Temporal patterns**: Gaze velocity/acceleration capture eye movement dynamics

---

## Next Steps

1. **Inspect current results**: Use `inspect_checkpoints.py` and WandB
2. **Identify weak points**: Which actions/scenarios perform worst?
3. **Integrate gaze**: Follow the plan to add gaze data
4. **Re-train and compare**: See if gaze improves performance

---

## Quick Commands Reference

```bash
# Inspect checkpoints
python scripts/inspect_checkpoints.py --all-folds

# Evaluate model
python scripts/test_model.py --checkpoint checkpoints/fold1/best_model.pth

# Get WandB metrics
python scripts/compute_metrics_from_wandb.py --run-name "hierarchical_har_fold1"

# Train new model (with gaze)
python scripts/train_hierarchical.py --cv --n-folds 4
```

---

## File Structure Summary

```
HAR_Lab_Initiative_AI/
├── checkpoints/          # Model checkpoints (weights only)
├── data/
│   ├── labels/          # CSV files with scenario/action labels
│   └── processed_ego4d/ # Processed IMU + gaze data (npz files)
├── scripts/
│   ├── train_hierarchical.py    # Main training script
│   ├── test_model.py            # Evaluation script
│   ├── inspect_checkpoints.py   # Checkpoint inspector (NEW)
│   └── compute_metrics_from_wandb.py  # WandB metrics extractor
└── notebooks/
    └── wandb/           # WandB run data
```

---

## Important Notes

1. **Checkpoints only contain weights** - No hyperparameters or metrics
2. **WandB has everything** - Training history, metrics, hyperparameters
3. **Gaze data exists** - But model doesn't use it yet
4. **4-fold CV** - Each fold is a separate training run

