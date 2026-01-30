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
# Credentials may come from AWS env vars, AWS_PROFILE, or the default boto3 chain.
# If you use an AWS profile, export AWS_PROFILE=<profile_name> before running.

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
