#!/bin/bash
# Setup script for GPU server training
# Usage: ./setup_training.sh

set -e

echo "=== 1. Setting up Environment ==="
# Assuming conda is installed and environment 'ego4d_lab' exists or will be created
# If not, uncomment:
# conda env create# 1. Install Dependencies
echo "Installing specific dependencies..."
pip install wandb scikit-learn tqdm pandas numpy scipy boto3
# conda activate ego4d_lab

pip install wandb scikit-learn

echo "\n=== 2. Downloading Sensor Data ==="
# Ensure AWS credentials are set in environment
if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "Error: AWS_ACCESS_KEY_ID not set"
    exit 1
fi

python scripts/download_sensors_direct.py \
    --target-uids-file target_uids.csv \
    --output-dir data/ego4d_data/v2 \
    --manifest-dir data/ego4d_data/v2

echo "\n=== 3. Processing IMU Data ==="
python scripts/process_imu_data.py \
    --target-uids-file target_uids.csv \
    --data-dir data/ego4d_data/v2 \
    --output-dir data/processed_ego4d

echo "\n=== 4. Setup Complete ==="
echo "You can now run the training using ./run_training.sh"
