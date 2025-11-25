# HAR Lab Initiative: IMU-Based Activity Recognition

Hierarchical activity recognition system using head-mounted IMU data from Ego4D. Designed for resource-efficient deployment on smartglasses.

---

## Quick Start

### 1. Prerequisites
```bash
# Clone repository
git clone <repo_url>
cd "Final Project"

# Create conda environment
conda env create -f environment.yml
conda activate ego4d_lab

# Install Ego4D CLI
pip install ego4d
```

### 2. Data Sharing Guide (Crucial for Partners)

**What is in this Repo (Git):**
- `target_uids.csv`: List of 1,200 videos we are using
- `data/labels/action_labels.csv`: Our generated labels (15 MB)
- `data/labels/scenario_labels.csv`: Scenario labels (100 KB)

**What is NOT in this Repo (You must download):**
- `data/ego4d_data/`: Raw video/IMU data (50GB+)
- `data/processed/`: Training tensors (20GB+)

### 3. How to Download Data

**Required credentials**: Ego4D access (request at [ego4d-data.org](https://ego4d-data.org))

```bash
# Step 1: Set up AWS credentials (provided by Ego4D)
export AWS_ACCESS_KEY_ID="<your_key>"
export AWS_SECRET_ACCESS_KEY="<your_secret>"

# Step 2: Download ONLY the 1,200 videos we need (~50 GB)
# This script uses target_uids.csv to filter the download
python scripts/download_sensors_direct.py --uids target_uids.csv --output data/ego4d_data/v2/imu
```

---

## Repository Structure

```
Final Project/
├── README.md                      # This file
├── RESEARCH_DESIGN.md             # Technical design document
├── environment.yml                # Conda dependencies
│
├── data/                          # All datasets
│   ├── ego4d_data/v2/             # Raw Ego4D downloads (GITIGNORED)
│   │   └── imu/                   # IMU CSV files (6-axis, 50Hz)
│   ├── labels/                    # Generated labels (SHARED IN GIT)
│   │   ├── action_labels.csv      # 230k LL action labels
│   │   └── scenario_labels.csv    # HL scenario labels
│   └── processed/                 # Model-ready tensors (GITIGNORED)
│
├── scripts/                       # Active scripts
│   ├── download_ego4d_metadata.py # Download annotations
│   ├── download_sensors_direct.py # Download IMU CSVs
│   ├── map_narrations_to_actions.py # Generate LL labels
│   └── archive_old_scripts/       # Unused legacy code
│
├── models/                        # Model architectures
│   ├── lle.py                     # Low-Level Encoder
│   └── hla.py                     # High-Level Architecture
```

---

## Active Pipeline

| Step | Script | Input | Output |
|:-----|:-------|:------|:-------|
| 1. Metadata | `download_ego4d_metadata.py` | AWS Creds | `narration.json` |
| 2. LL Labels | `map_narrations_to_actions.py` | `narration.json` | `action_labels.csv` |
| 3. IMU Data | `download_sensors_direct.py` | `target_uids.csv` | `data/ego4d_data/v2/imu/` |
| 4. Training | `train_hierarchical.py` (TODO) | Labels + IMU | `checkpoints/` |

---

## Questions?

See `RESEARCH_DESIGN.md` for technical details.
