# AR Glasses Deployment Guide

This guide explains how to deploy the hierarchical activity recognition model to AR Meta Research glasses, starting with simulation and then adapting to the actual hardware platform.

## Overview

The deployment pipeline consists of:
1. **Model Export**: Convert PyTorch checkpoint to ONNX/TorchScript format
2. **Inference Engine**: Real-time preprocessing and model inference
3. **Simulation**: Test with mock sensor data before deploying to hardware
4. **Platform Adaptation**: Adapt to specific glasses SDK/API (to be determined)

## Architecture

```
AR Glasses Sensors (IMU + Gaze @ 50Hz)
  ↓
Sliding Window Buffer (30s context)
  ↓
Preprocessing (Normalization + Feature Augmentation)
  ↓
Model Inference (ONNX/TorchScript)
  ↓
Post-processing (Scenario + Action Predictions)
```

## Quick Start

### 1. Export Model

Export a trained checkpoint to deployment format:

```bash
python scripts/export_model.py \
    --checkpoint checkpoints/fold1/best_model.pth \
    --output-dir deployment \
    --export-format both
```

This creates:
- `deployment/model.onnx` - ONNX model (recommended for cross-platform)
- `deployment/model.pt` - TorchScript model (backup)
- `deployment/normalization_stats.json` - Normalization statistics and config

### 2. Run Simulation

Test the inference pipeline with mock sensor data:

```bash
python scripts/simulate_glasses.py \
    --model deployment/model.onnx \
    --stats deployment/normalization_stats.json \
    --inference-interval 1.0 \
    --duration 60.0 \
    --output simulation/simulation_results.json
```

This will:
- Load a test video's sensor data
- Stream it at 50Hz (simulating real-time)
- Run inference every 1 second
- Display predictions and save results

### 3. Use Inference Engine in Your Code

```python
from inference_engine import InferenceEngine

# Initialize engine
engine = InferenceEngine(
    model_path='deployment/model.onnx',
    stats_path='deployment/normalization_stats.json',
    use_gaze=True,
    device='cpu'  # or 'cuda' if available
)

# Add sensor samples (called at 50Hz from glasses)
imu_sample = np.array([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
gaze_sample = np.array([norm_pos_x, norm_pos_y])  # or None if missing

engine.add_sample(imu_sample, gaze_sample)

# Run inference (call periodically, e.g., every 1 second)
if engine.buffer.is_full():
    scenario_pred, action_preds, scenario_probs, action_probs_list = engine.infer()
    print(f"Scenario: {scenario_pred}")
    print(f"Actions: {action_preds}")
```

## File Structure

```
deployment/
├── README.md                    # This file
├── model.onnx                   # Exported ONNX model
├── model.pt                     # Exported TorchScript model (optional)
└── normalization_stats.json     # Normalization statistics and config

scripts/
├── export_model.py              # Export PyTorch model to ONNX/TorchScript
├── inference_engine.py          # Real-time inference engine
├── simulate_glasses.py          # Simulation environment
└── test_inference_engine.py     # Unit tests
```

## Model Input/Output

### Input Format

- **Shape**: `(batch_size, seq_len, window_size, in_channels)`
  - `batch_size`: 1 for real-time inference
  - `seq_len`: 30 (30 seconds of context)
  - `window_size`: 50 (1 second at 50Hz)
  - `in_channels`: 10 (6 IMU + 2 gaze + 2 norms) or 8 (6 IMU + 2 gaze) or 6 (IMU only)

### Output Format

- **Scenario logits**: `(batch_size, num_scenarios)` - 7 scenario classes
- **Action logits**: `(batch_size, seq_len, num_actions)` - 6 action classes per window

### Scenario Classes

- Cleaning
- Mechanical Repair
- Cooking
- Walking
- Carpentry
- Instruments
- Desk Work

### Action Classes

- Stationary
- Locomotion
- Essential Operation
- Object Transfer
- Search
- Error / Correction

## Preprocessing Pipeline

1. **Normalization**: Separate z-score normalization for IMU and gaze using saved statistics
2. **Concatenation**: Combine IMU (6 channels) and gaze (2 channels) → 8 channels
3. **Feature Augmentation**: Add accel_norm and gyro_norm → 10 channels
4. **Windowing**: Extract last 30 windows (1s each, stride=5) from 30s buffer

## Performance Considerations

### Latency Targets

- **Preprocessing**: < 10ms
- **Model Inference**: < 100ms (for 30s context)
- **Total**: < 150ms per prediction

### Memory Usage

- **Buffer**: 30s × 50Hz × 8 channels × 4 bytes ≈ 48 KB
- **Model**: ~5-10 MB (depending on quantization)
- **Total**: < 50 MB

### Optimization Options

1. **Quantization**: Convert to INT8 for mobile deployment (future work)
2. **Model Pruning**: Remove redundant weights (future work)
3. **Reduced Context**: Use shorter sequence length (e.g., 20s instead of 30s) for lower latency

## Platform Adaptation

### Android (if applicable)

1. Convert ONNX to TensorFlow Lite or use ONNX Runtime Mobile
2. Create Android service for sensor data collection
3. Integrate inference engine into Android app

### Python SDK (if applicable)

1. Wrap `InferenceEngine` in Python API
2. Handle sensor data callbacks from glasses SDK
3. Return predictions via callback or polling

### C++ API (if applicable)

1. Port preprocessing to C++
2. Use ONNX Runtime C++ API
3. Create C++ wrapper for inference

## Testing

Run unit tests:

```bash
python scripts/test_inference_engine.py
```

Test with simulation:

```bash
python scripts/simulate_glasses.py \
    --model deployment/model.onnx \
    --stats deployment/normalization_stats.json \
    --data data/processed_ego4d/<video_uid>/seq.npz
```

## Troubleshooting

### Buffer Not Full Enough

If you see "Buffer too small" errors, ensure you're adding samples at 50Hz for at least 30 seconds before running inference.

### Model Format Issues

- **ONNX**: Requires `onnxruntime` (`pip install onnxruntime`)
- **TorchScript**: Requires PyTorch (`pip install torch`)

### Normalization Stats Mismatch

Ensure the normalization stats match the training configuration. Check `normalization_stats.json` for:
- `use_gaze`: Should match training config
- `add_norm_features`: Should match training config
- `in_channels`: Should match model input channels

### Missing Gaze Data

If gaze data is missing, the engine will zero-fill automatically (if `use_gaze=True`). For better results, ensure gaze data is available or set `use_gaze=False` during export.

## Next Steps

1. **Research Meta Research Glasses SDK**: Determine platform/API (Android, Python, C++, etc.)
2. **Adapt Inference Engine**: Modify to work with glasses SDK
3. **Handle Sensor Data Collection**: Integrate with glasses sensor APIs
4. **Optimize for Platform**: Apply quantization/pruning if needed
5. **Deploy to Hardware**: Test on actual glasses

## References

- Model architecture: `scripts/train_hierarchical.py`
- Data processing: `scripts/process_imu_data.py`
- Model evaluation: `scripts/test_model.py`

