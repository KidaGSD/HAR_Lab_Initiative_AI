# HAR Lab Initiative: IMU-Based Activity Recognition

Hierarchical activity recognition system using head-mounted IMU data from Ego4D. Designed for resource-efficient deployment on smartglasses.

---

## Quick Start

### 1. Prerequisites
```bash
# Create conda environment
conda env create -f environment.yml
conda activate ego4d_lab

# Install Ego4D CLI
pip install ego4d
```

### 2. Data Overview (Crucial for Partners)

**Final Dataset:** 1,652 videos with IMU data from Ego4D

We use two levels of labels for our **Semi-Supervised** approach:

| File | Count | Description | Purpose |
|:-----|:------|:------------|:--------|
| `data/labels/scenario_labels.csv` | **1,652** videos | All videos with IMU data, labeled by scenario (Cooking, Carpentry, etc.). | Training High-Level Classifier (HLA). |
| `data/labels/master_annotations.csv` | **1,485** videos<br/>**306,473** windows | Subset with Low-Level Action labels. Currently using **Improved Keyword Matching** (v2). | Training Motion Encoder (LLE) probes. |

**Labeling Strategy:**
- **v1 (Legacy)**: Simple keyword matching (Archived).
- **v2 (Current)**: Context-aware keyword matching with expanded vocabulary and strict validation (no "Stationary" fallback).
- **v3 (Planned)**: LLM-based labeling using Qwen-14B (`scripts/label_with_qwen.py`) for handling ambiguity.

**Key Statistics (v2 Labels):**
- **Scenario Distribution**: Cleaning (313), Mechanical Repair (294), Cooking (267), Walking (228), Carpentry (186), Instruments (158), Desk Work (150), Gardening (56)
- **Train/Val/Test Split**: 1,112 / 184 / 180 videos (67% / 11% / 11%, plus 176 multi-task)
- **Action Labels**: Manual Work (73%), Locomotion (15%), Scanning (6%), Stationary (6%)

**Note**: Only videos with available IMU sensor data are included. The "Fitness/Workout" scenario was removed due to insufficient samples.

### 3. How to Download Data

**Required credentials**: Ego4D access (request at [ego4d-data.org](https://ego4d-data.org))

```bash
# Step 1: Set up AWS credentials (provided by Ego4D)
export AWS_ACCESS_KEY_ID="<your_key>"
export AWS_SECRET_ACCESS_KEY="<your_secret>"

# Step 2: Download the 1,652 target videos (~15 GB IMU data)
# This script uses target_uids.csv to filter the download
python scripts/download_sensors_direct.py --target-uids-file target_uids.csv --output-dir data/ego4d_data/v2 --manifest-dir data/ego4d_data/v2
```

---

## Training Setup

### Architecture
We use a **Hierarchical Model**:
1.  **LLE (Low-Level Encoder)**: CNN-GRU. Takes 1s IMU window -> Outputs Motion Embedding.
2.  **HLA (High-Level Architecture)**: GRU. Takes sequence of 30 LLE embeddings -> Outputs Scenario Class.

### Strategy: Masked Loss
Since only 938 videos have action labels, we use a **Masked Loss** to train on the full dataset:
*   **Labeled Windows**: Loss = `Loss_Scenario` + `Loss_Action`
*   **Unlabeled Windows**: Loss = `Loss_Scenario` + `0` (Action loss is masked)

### Running Training
1.  **Process Data** (Aligns IMU to 50Hz windows):
    ```bash
    python scripts/process_imu_data.py --data-dir data/ego4d_data --output-dir data/processed_ego4d
    ```
2.  **Train Model** (Batch Size 128):
    ```bash
    python scripts/train_hierarchical.py --target-uids-file target_uids.csv --processed-dir data/processed_ego4d
    ```

---

## Repository Structure

```
Final Project/
├── README.md                      # This file
├── RESEARCH_DESIGN.md             # Technical design document
### 4. LLM Labeling Setup (New)

To run the AI labeling pipeline (v3):

```bash
# 1. Install dependencies and download model (Qwen-14B)
./setup_llm.sh

# 2. Run the labeling script (Labels all 300k narrations)
# This takes ~2 hours on a single GPU
python scripts/label_with_qwen.py --model Qwen/Qwen1.5-14B-Chat-AWQ
```

### 5. Repository Structure

```
├── data/
│   ├── ego4d_data/       # Raw IMU/Gaze/Narrations
│   ├── processed_ego4d/  # Processed .npz files (Windowed)
│   ├── labels/           # Final CSVs for training
│   │   ├── scenario_labels.csv      # High-Level Labels
│   │   ├── master_annotations.csv   # Low-Level Labels (v2)
│   │   ├── action_labels_llm.csv    # LLM Labels (v3 - Output)
│   └── intermediate/     # Old/Backup files
├── scripts/
│   ├── download_sensors_direct.py   # Download IMU data
│   ├── process_imu_data.py          # Process raw -> npz
│   ├── train_hierarchical.py        # Main Training Script
│   ├── label_with_qwen.py           # LLM Labeling Script
│   ├── extract_scenario_labels.py   # (Setup) Extract scenarios
│   ├── consolidate_labels.py        # (Setup) Merge labels
│   └── utils/                       # Analysis & Helper scripts
├── setup_server.sh       # Setup Conda Env
├── setup_llm.sh          # Setup vLLM & Model
└── environment.yml       # Dependencies
```

---

See `RESEARCH_DESIGN.md` for technical details.
