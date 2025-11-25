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

We use two levels of labels for our **Semi-Supervised** approach:

| File | Count | Description | Purpose |
|:-----|:------|:------------|:--------|
| `data/labels/scenario_labels.csv` | **6,147** videos | **Superset**: All videos matching our 8 scenarios (Cooking, Carpentry, etc.). | Validating the High-Level Classifier (HLA). |
| `data/labels/master_annotations.csv` | **938** videos | **Subset**: Videos that *also* have Low-Level Action labels (Walking, Stationary) derived from narrations. | Training the Motion Encoder (LLE). |

**Note**: The "Fitness/Workout" scenario was removed due to insufficient data.

### 3. How to Download Data

**Required credentials**: Ego4D access (request at [ego4d-data.org](https://ego4d-data.org))

```bash
# Step 1: Set up AWS credentials (provided by Ego4D)
export AWS_ACCESS_KEY_ID="<your_key>"
export AWS_SECRET_ACCESS_KEY="<your_secret>"

# Step 2: Download the 6,147 target videos (~50 GB)
# This script uses target_uids.csv to filter the download
python scripts/download_sensors_direct.py --uids target_uids.csv --output data/ego4d_data/v2/imu
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
├── environment.yml                # Conda dependencies
├── target_uids.csv                # List of 6,147 target videos
│
├── data/                          # All datasets
│   ├── ego4d_data/v2/             # Raw Ego4D downloads (GITIGNORED)
│   ├── labels/                    # Generated labels (SHARED IN GIT)
│   │   ├── master_annotations.csv # Training data (Action + Scenario)
│   │   └── scenario_labels.csv    # Full dataset (Scenario only)
│   └── processed_ego4d/           # Aligned .npz files (GITIGNORED)
│
├── scripts/                       # Active scripts
│   ├── process_imu_data.py        # Pre-processing (IMU -> .npz)
│   ├── train_hierarchical.py      # Main training script
│   ├── download_sensors_direct.py # Download utility
│   └── ...
```

---

See `RESEARCH_DESIGN.md` for technical details.
