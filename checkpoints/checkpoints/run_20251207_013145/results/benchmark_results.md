# Benchmark Results

Generated: Sun Dec  7 09:00:21 AM EST 2025

## Performance Comparison

| Model | Scenario F1 | Action F1 |
|-------|-------------|-----------|
| Backbone (β=0.0) | 0.5725 | N/A |
| Backbone + Probe | N/A | N/A |
| Joint (β=0.3) | N/A | N/A |
| Joint + Probe | N/A | N/A |
| MLP-MLP | N/A | N/A |
| CNN-MLP | N/A | N/A |
| IMU2CLIP | N/A | N/A |
| CNN-LSTM-GRU | N/A | N/A |

## Configuration
- Focal Loss: gamma=2.0
- Class Weights: [5.0, 7.0, 0.5, 10.0]
- Label Smoothing: 0.1
- Augmentation: Jittering + Scaling
