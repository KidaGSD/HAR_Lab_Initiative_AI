# Research Design: Hierarchical IMU-Based Human Activity Recognition

## 1. Objective

Develop a **hierarchical activity recognition system** using head-mounted IMU data that jointly learns:
- **High-Level (Scenario)**: Activity context classification (7 classes, 30-second windows)
- **Low-Level (Action)**: Motion primitive classification (4 classes, 1-second windows)

The system employs a **semi-supervised approach** where low-level motion representations emerge as a byproduct of scenario classification, eliminating the need for dense action annotations.

---

## 2. Architecture

### 2.1 Model Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL MODEL                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   IMU Input (1s window)                                         │
│   [8 channels × 50 Hz = 400 features]                           │
│           │                                                      │
│           ▼                                                      │
│   ┌───────────────────────┐                                     │
│   │  Low-Level Encoder    │  CNN-GRU with SE Attention          │
│   │  (LLE)                │  [32→64→128] conv + 256 GRU         │
│   │  ~85k params          │                                     │
│   └───────────┬───────────┘                                     │
│               │ 128-dim embedding                                │
│               ▼                                                  │
│   ┌───────────────────────┐                                     │
│   │  30 × LLE embeddings  │  Sequence of 30 seconds             │
│   └───────────┬───────────┘                                     │
│               │                                                  │
│               ▼                                                  │
│   ┌───────────────────────┐                                     │
│   │  High-Level Arch      │  Transformer (4 heads, 2 layers)    │
│   │  (HLA)                │                                     │
│   │  ~200k params         │                                     │
│   └───────────┬───────────┘                                     │
│               │                                                  │
│       ┌───────┴───────┐                                         │
│       ▼               ▼                                         │
│   ┌────────┐    ┌──────────┐                                    │
│   │Scenario│    │  Action  │  (via probe or joint training)     │
│   │ Head   │    │   Head   │                                    │
│   │7 class │    │ 4 class  │                                    │
│   └────────┘    └──────────┘                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Details

**Low-Level Encoder (LLE)**
- Input: 1-second IMU window (8 channels × 50 Hz)
  - 6 raw channels: accelerometer (x,y,z) + gyroscope (x,y,z)
  - 2 derived channels: acceleration norm + rotation norm
- CNN: 3-layer [32→64→128] with Squeeze-and-Excitation attention
- GRU: 2-layer, 256 hidden units
- Output: 128-dimensional motion embedding
- Parameters: ~85,000

**High-Level Architecture (HLA)**
- Input: Sequence of 30 LLE embeddings (30-second window)
- Architecture: 2-layer Transformer with 4 attention heads
- Dropout: 0.15
- Output: 7-class scenario prediction
- Parameters: ~200,000

### 2.3 Training Strategies

| Strategy | β Value | Description |
|----------|---------|-------------|
| **Backbone (β=0.0)** | 0.0 | Train on scenario labels only, probe for action |
| **Joint (β=0.3)** | 0.3 | Train with combined loss: L = α·L_scenario + β·L_action |

---

## 3. Classification Taxonomy

### 3.1 High-Level Scenarios (7 Classes)

| Scenario | Motion Signature | Data Available |
|:---------|:-----------------|:---------------|
| Cleaning | Intermittent walking, downward gaze, wiping cycles | 313 videos |
| Mechanical Repair | Static positioning, focused gaze, tool manipulation | 294 videos |
| Cooking | Minimal locomotion, downward focus, episodic arm motion | 267 videos |
| Walking Outdoors | Continuous locomotion, forward gaze, step periodicity | 228 videos |
| Carpentry | Static stance, stable gaze, sawing/hammering rhythm | 186 videos |
| Playing Instrument | Stationary, instrument focus, musical rhythm | 158 videos |
| Desk Work | Stationary, screen focus, minimal motion | 150 videos |

**Note**: Gardening (56 videos) excluded due to insufficient data.

### 3.2 Low-Level Actions (4 Classes)

| Action | Motion Characteristics | Distribution |
|:-------|:-----------------------|:-------------|
| **Stationary** | Low energy on all sensors | ~13% |
| **Locomotion** | High body acceleration, rhythmic step pattern | ~10% |
| **Manipulation** | Hand acceleration, irregular/complex patterns | ~70% |
| **Search/Interrupt** | High head rotation, low hand acceleration | ~7% |

---

## 4. Training Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Backbone Training (β=0.0)                         │
│  ├─ Train LLE + HLA on scenario labels only                 │
│  ├─ Data augmentation: Jittering + Scaling                  │
│  ├─ Label smoothing: 0.1                                    │
│  └─ Output: backbone_model.pth                              │
│                        ↓                                     │
│  Phase 2: Probe Training                                    │
│  ├─ Freeze LLE + HLA                                        │
│  ├─ Train linear action head                                │
│  ├─ Focal Loss (γ=2.0) for class imbalance                 │
│  ├─ Class weights: [5, 7, 0.5, 10]                         │
│  └─ Output: action probe metrics                            │
│                        ↓                                     │
│  Phase 3: Joint Training (β=0.3)                            │
│  ├─ Train all components with combined loss                 │
│  ├─ L = 1.0·L_scenario + 0.3·L_action                      │
│  └─ Output: joint_model.pth                                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 4.1 Training Techniques

| Technique | Implementation | Purpose |
|-----------|----------------|---------|
| **Data Augmentation** | Jittering (σ=0.02, p=0.5), Scaling (0.9-1.1x, p=0.5) | Regularization |
| **Focal Loss** | γ=2.0, α=class_weights | Handle class imbalance |
| **Label Smoothing** | 0.1 | Prevent overconfidence |
| **Warmup** | 5 epochs | Stable initial training |
| **Gradient Clipping** | 1.0 | Training stability |

### 4.2 Class Weights (Action)

| Class | Distribution | Weight | Rationale |
|-------|--------------|--------|-----------|
| Stationary | 13% | 5.0 | Minority, upweight |
| Locomotion | 10% | 7.0 | Minority, upweight |
| Manipulation | 70% | 0.5 | Majority, downweight |
| Search/Interrupt | 7% | 10.0 | Rarest, highest weight |

---

## 5. Experimental Design

### 5.1 Evaluation Protocol

**4-Fold Cross-Validation**
- Split by video to prevent data leakage
- Report mean ± std across folds
- Final model trained on train+val combined

### 5.2 Baseline Comparisons

| Model | LLE Architecture | HLA Architecture | Reference |
|:------|:-----------------|:-----------------|:----------|
| MLP-MLP | Hand-picked features + MLP | MLP | EgoCHARM Table 2 |
| CNN-MLP | 1D-CNN | MLP | EgoCHARM Table 2 |
| IMU2CLIP | CNN-GRU (large) | MLP | Moon et al. 2022 |
| CNN-LSTM-GRU | CNN-LSTM | GRU | EgoCHARM Table 2 |
| **Ours (β=0.0)** | CNN-GRU-SE | Transformer | Backbone only |
| **Ours (β=0.3)** | CNN-GRU-SE | Transformer | Joint training |

### 5.3 Key Research Questions

1. **Semi-Supervised Learning**: Can scenario-only training produce useful action representations?
2. **Joint vs. Probe**: Does joint training (β=0.3) outperform linear probing (β=0.0)?
3. **Architecture**: Do Transformer attention mechanisms improve temporal modeling over GRU?

---

## 6. Hyperparameters

### 6.1 Model Configuration

```yaml
# Low-Level Encoder
lle:
  in_channels: 8        # 6 raw + 2 norms
  cnn_filters: [32, 64, 128]
  gru_hidden: 256
  gru_layers: 2
  embedding_dim: 128
  se_reduction: 8       # SE attention reduction ratio

# High-Level Architecture
hla:
  hidden_dim: 128
  num_layers: 2
  num_classes: 7
  seq_len: 30
  nhead: 4              # Transformer heads
  dropout: 0.15
  type: transformer
```

### 6.2 Training Configuration

```yaml
training:
  batch_size: 256
  lr: 1.0e-4
  epochs: 60
  patience: 20          # Early stopping
  weight_decay: 1.0e-5
  grad_clip: 1.0
  warmup_epochs: 5
  label_smoothing: 0.1
  use_focal_loss: true
  use_action_class_weights: true

data:
  per_video_center: true    # Normalize per video
  add_norm_features: true   # Add magnitude channels
  augmentation: true
```

---

## 7. Data Processing

### 7.1 Input Features

| Channel | Description | Preprocessing |
|---------|-------------|---------------|
| acc_x, acc_y, acc_z | Accelerometer | Per-video centering |
| gyro_x, gyro_y, gyro_z | Gyroscope | Per-video centering |
| acc_norm | √(ax² + ay² + az²) | Derived |
| gyro_norm | √(gx² + gy² + gz²) | Derived |

### 7.2 Window Specifications

| Level | Window Size | Stride | Sampling Rate |
|-------|-------------|--------|---------------|
| Low-Level (Action) | 1 second | 1 second | 50 Hz |
| High-Level (Scenario) | 30 seconds | N/A | 1 embedding/sec |

### 7.3 Label Sources

- **Scenario Labels**: Extracted from Ego4D activity summaries (30s annotations)
- **Action Labels**: LLM-validated (Qwen-14B) from narration timestamps
  - v3 labeling strategy reduces false "Stationary" labels
  - Error validation reclassified 84% of false "Error" labels

---

## 8. Metrics

### 8.1 Scenario Classification
- **Macro F1-score**: Primary metric (handles class imbalance)
- **Per-class Recall**: Identify weak scenario detection
- **Confusion Matrix**: Analyze cross-scenario errors

### 8.2 Action Classification (Probe)
- **Macro F1-score**: Average across 4 classes
- **Per-class Precision/Recall**: Validate minority class performance

### 8.3 Targets

| Task | Baseline | Target |
|------|----------|--------|
| Scenario F1 | 0.57 | 0.60+ |
| Action F1 (Probe) | 0.26 | 0.40+ |
| Action F1 (Joint) | 0.38 | 0.45+ |

---

## 9. Scripts Reference

| Script | Purpose |
|--------|---------|
| `run_backbone.sh` | Train backbone (β=0.0) and joint (β=0.3) models in parallel |
| `run_baselines.sh` | Train all 4 baseline models sequentially |
| `run_cv_final.sh` | 4-fold CV + final model + probe |
| `configs/beta_0.0.yaml` | Backbone training configuration |
| `configs/beta_0.3.yaml` | Joint training configuration |

---

## 10. Key Innovations

1. **Hierarchical Semi-Supervised Learning**: Action representations emerge from scenario supervision
2. **Squeeze-and-Excitation CNN**: Channel attention for IMU feature recalibration
3. **Transformer HLA**: Self-attention captures long-range temporal dependencies
4. **Focal Loss + Aggressive Weights**: Combat severe class imbalance (70% manipulation)
5. **Multi-Phase Training**: Backbone → Probe → Joint allows controlled comparison
