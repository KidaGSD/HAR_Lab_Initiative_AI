# AR Glasses Anomaly Detection with IMU and Gaze

**Harvard AI and Robotics Labt** - Initiative AI assistance triggered by anomalous IMU and gaze patterns on AR glasses.

---

## Overview

Self-supervised anomaly detection system that:
- Processes egocentric IMU (25 Hz) and gaze (30 Hz) data from AR glasses
- Uses dual-branch CNN+GRU encoder with Deep SVDD for one-class anomaly detection
- Detects unusual behavior patterns at **<1% camera duty cycle** (~3 seconds/hour)
- Achieves **1.12 alerts/hour** on multi-scenario test data

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+
conda create -n har python=3.10
conda activate har

# Install dependencies
pip install -r requirements.txt

# Install EgoExo4D CLI (for data download)
pip install egoexo
```

### Required Files

Ensure these files exist before running:
- `state/active/candidate_takes.csv` - List of EgoExo4D takes (auto-copied to root if missing)
- `config/default.yaml` - Pipeline configuration (included)

---

## 📁 Project Structure

```
.
├── notebooks/
│   ├── 00_pipeline_runner.ipynb    # Data pipeline (download → windows)
│   └── 10_training.ipynb           # SSL training + SVDD anomaly detection
├── scripts/
│   ├── download_modalities.py      # Download IMU, gaze, video from EgoExo4D
│   ├── run_alignment.py            # Temporal alignment of sensors
│   └── make_windows.py             # Windowing + feature extraction
├── config/
│   └── default.yaml                # Pipeline parameters
├── state/active/
│   └── candidate_takes.csv         # Available EgoExo4D takes
└── runs/{RUN_ID}/                  # Output directory for each run
    ├── windows/                    # Processed window data
    ├── windows_normal.parquet      # Filtered "normal" windows
    ├── models/ssl/                 # Trained SSL models
    └── reports/                    # Training metrics and plots
```

---

## 🔧 Usage

### Step 1: Run Data Pipeline

Open `notebooks/00_pipeline_runner.ipynb` and configure:

```python
RUN_ID = '20251115_myrun'  # Change this for each run
RUN_DOWNLOAD = True        # Download data
RUN_ALIGNMENT = True       # Align sensors
RUN_WINDOWING = True       # Create windows
```

**Run sections in order**:
1. **Section 0**: Config (set RUN_ID)
2. **Section 2**: Download modalities (if needed)
3. **Section 3**: Alignment & QA
4. **Section 4**: Windowing
5. **Section 5**: Manifest generation
6. **Section 6-8**: QA & Coverage analysis
7. **Section 9**: Normal window filtering

**Output**: `runs/{RUN_ID}/windows_normal.parquet` (~20-30% of windows pass filtering)

---

### Step 2: Train Anomaly Detection Model

Open `notebooks/10_training.ipynb` and configure:

```python
CONFIG = {
    'run_id': '20251115_myrun',  # Match your RUN_ID from Step 1
    ...
}
```

**Run all sections**:
- **Sections 1-4**: Load data, create train/val/test splits
- **Sections 5-6**: SSL model training (~50 min for 20K windows, 20 epochs)
- **Sections 7-8**: Extract embeddings, fit Deep SVDD
- **Section 9**: Evaluate metrics (tiered thresholds: P85/P95/P99)

**Output**:
- `runs/{RUN_ID}/models/ssl/{timestamp}/` - Trained models
- `runs/{RUN_ID}/reports/training/{timestamp}/` - Metrics and plots

---

## 📊 Key Parameters

### Window Configuration (`config/default.yaml`)
```yaml
windowing:
  window_size_s: 3.0      # 3-second windows
  hop_size_s: 0.2         # 200ms hop (87% overlap)
  seq_len: 50             # 50 timesteps per window
```

### Normal Filtering (Section 9 in 00_pipeline)
```python
# Percentile-based thresholds
thresholds = {
    'gaze_missing_ratio': P30,  # Low data loss
    'traj_missing_ratio': P30,
    'eye_head_corr': P70        # Strong eye-head coordination
}
```

### Anomaly Detection (10_training Section 9)
```python
# Tiered alert levels
LOW_THRESH = P85    # ~15% of training windows
MID_THRESH = P95    # ~5% of training windows
HIGH_THRESH = P99   # ~1% of training windows (target)
```

---

## 🎯 Expected Results

### 10-Scenario Multi-Task Training
- **Training windows**: 21,853 normal windows
- **Training time**: ~50 minutes (M1 Mac, 20 epochs)
- **HIGH alerts**: 1.12 alerts/hour on test set
- **Duty cycle**: 0.093% (3.35 seconds/hour)

### Single Scenario (40-Bike Baseline)
- **Training windows**: 4,177 normal windows
- **Training time**: ~15 minutes (M1 Mac, 20 epochs)
- **HIGH alerts**: 3.0 alerts/hour (oversensitive)
- **Duty cycle**: 0.8% (28.8 seconds/hour)

**Conclusion**: Multi-scenario training reduces false positives by 87%.

---

## Troubleshooting

### Missing `candidate_takes.csv`
```bash
# The notebook auto-copies from state/active/, but if that's missing:
# Option 1: Restore from backup
cp archive/legacy_data/candidate_takes.csv state/active/

# Option 2: Generate from EgoExo4D metadata
# (requires EgoExo4D access and metadata query)
```

### Missing `windows_normal.parquet`
```
Run 00_pipeline_runner.ipynb Section 9 first to generate normal windows.
```

### Import Errors
```bash
# Ensure you're in the project root directory
cd "/Users/huangjunda/Desktop/MIT 2.156/Final Project"

# Reinstall dependencies
pip install -r requirements.txt
```

---

## ⚠️ Important Notes

1. **Data not included**: Raw EgoExo4D data (~140GB) and runs (~260MB) are gitignored. You need to download data using Section 2 of the pipeline.

2. **RUN_ID convention**: Use format `YYYYMMDD_description` (e.g., `20251115_10scenario`)

3. **Cross-machine compatibility**: All paths are relative. Run from project root directory.

4. **Credentials**: If downloading from EgoExo4D requires authentication, follow their CLI setup guide.

---

## Model Architecture

```
Input: 3s window (50 timesteps)
├─ IMU Branch (3 channels: ax, ay, az)
│  └─ 1D-CNN → BatchNorm → ReLU → GRU → FC
├─ Gaze Branch (4 channels: x, y, z, confidence)
│  └─ 1D-CNN → BatchNorm → ReLU → GRU → FC
└─ Fusion → LayerNorm → Linear → ReLU → 128-dim embedding

Anomaly Detection:
└─ Deep SVDD (nu=0.1) → Distance to hypersphere center
```

**Training Objective**: Mean reconstruction (MSE loss)
**Anomaly Score**: Euclidean distance from SVDD center

---

## Citation

```bibtex
@misc{har_ar_glasses_2025,
  title={Initiative AI Assistance Triggered by IMU and Gaze on AR Glasses},
  author={Chung-Ta Huang},
  year={2025}
}
```

---


**Last Updated**: 2025-11-14
