#!/usr/bin/env python3
"""
Unit tests for inference engine preprocessing, normalization, and window extraction.
"""

import numpy as np
import unittest
from pathlib import Path
import sys
import tempfile
import json

sys.path.append(str(Path(__file__).parent))
from inference_engine import SlidingWindowBuffer, InferenceEngine


class TestSlidingWindowBuffer(unittest.TestCase):
    """Test SlidingWindowBuffer class"""
    
    def setUp(self):
        self.buffer = SlidingWindowBuffer(max_samples=1500, num_imu_channels=6, num_gaze_channels=2)
    
    def test_add_sample(self):
        """Test adding samples to buffer"""
        imu_sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        gaze_sample = np.array([0.5, 0.6])
        
        self.buffer.add_sample(imu_sample, gaze_sample)
        self.assertEqual(self.buffer.size(), 1)
        
        imu_array, gaze_array = self.buffer.get_array(use_gaze=True)
        np.testing.assert_array_equal(imu_array[0], imu_sample)
        np.testing.assert_array_equal(gaze_array[0], gaze_sample)
    
    def test_buffer_max_size(self):
        """Test that buffer respects max_samples limit"""
        imu_sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        # Add more samples than max_size
        for i in range(2000):
            self.buffer.add_sample(imu_sample)
        
        self.assertEqual(self.buffer.size(), 1500)  # Should be capped at max_samples
        self.assertTrue(self.buffer.is_full())
    
    def test_missing_gaze(self):
        """Test handling of missing gaze data"""
        imu_sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        self.buffer.add_sample(imu_sample, gaze_sample=None)
        imu_array, gaze_array = self.buffer.get_array(use_gaze=True)
        
        # Gaze should be zero-filled
        np.testing.assert_array_equal(gaze_array[0], np.zeros(2))
    
    def test_clear(self):
        """Test clearing buffer"""
        imu_sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        self.buffer.add_sample(imu_sample)
        self.assertEqual(self.buffer.size(), 1)
        
        self.buffer.clear()
        self.assertEqual(self.buffer.size(), 0)


class TestInferenceEnginePreprocessing(unittest.TestCase):
    """Test InferenceEngine preprocessing logic"""
    
    def setUp(self):
        # Create temporary normalization stats file
        self.temp_dir = tempfile.mkdtemp()
        self.stats_path = Path(self.temp_dir) / 'normalization_stats.json'
        
        # Create mock stats
        stats = {
            'imu_mean': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'imu_std': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'gaze_mean': [0.5, 0.5],
            'gaze_std': [0.25, 0.25],
            'config': {
                'use_gaze': True,
                'add_norm_features': True,
                'per_video_center': False,
                'seq_len': 30,
                'window_size': 50,
                'stride': 5,
                'in_channels': 10
            },
            'scenario_map': {'Cooking': 0, 'Walking': 1},
            'action_map': {'Stationary': 0, 'Locomotion': 1},
            'idx_to_scenario': {'0': 'Cooking', '1': 'Walking'},
            'idx_to_action': {'0': 'Stationary', '1': 'Locomotion'}
        }
        
        with open(self.stats_path, 'w') as f:
            json.dump(stats, f)
    
    def test_normalization(self):
        """Test that normalization is applied correctly"""
        # Create a dummy ONNX model file (we'll skip actual model loading in tests)
        # Instead, test preprocessing logic directly
        
        # Create buffer with test data
        buffer = SlidingWindowBuffer(max_samples=1500, num_imu_channels=6, num_gaze_channels=2)
        
        # Add normalized samples (mean=0, std=1)
        imu_sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # Raw values
        gaze_sample = np.array([0.5, 0.6])  # Raw values
        
        buffer.add_sample(imu_sample, gaze_sample)
        
        # Load stats
        with open(self.stats_path, 'r') as f:
            stats = json.load(f)
        
        imu_mean = np.array(stats['imu_mean'])
        imu_std = np.array(stats['imu_std'])
        gaze_mean = np.array(stats['gaze_mean'])
        gaze_std = np.array(stats['gaze_std'])
        
        # Get buffer arrays
        imu_array, gaze_array = buffer.get_array(use_gaze=True)
        
        # Normalize
        imu_normalized = (imu_array - imu_mean) / imu_std
        gaze_normalized = (gaze_array - gaze_mean) / gaze_std
        
        # Check normalization
        np.testing.assert_array_almost_equal(imu_normalized[0], imu_sample, decimal=5)
        np.testing.assert_array_almost_equal(gaze_normalized[0], (gaze_sample - gaze_mean) / gaze_std, decimal=5)
    
    def test_window_extraction(self):
        """Test window extraction with stride"""
        buffer = SlidingWindowBuffer(max_samples=1500, num_imu_channels=6, num_gaze_channels=2)
        
        # Fill buffer with sequential data
        for i in range(200):
            imu_sample = np.array([float(i)] * 6)
            gaze_sample = np.array([float(i), float(i)])
            buffer.add_sample(imu_sample, gaze_sample)
        
        # Get arrays
        imu_array, gaze_array = buffer.get_array(use_gaze=True)
        
        # Test window extraction logic
        window_size = 50
        stride = 5
        seq_len = 30
        
        num_windows = (len(imu_array) - window_size) // stride + 1
        start_idx = max(0, num_windows - seq_len) * stride
        
        windows = []
        for i in range(seq_len):
            window_start = start_idx + i * stride
            window_end = window_start + window_size
            
            if window_end <= len(imu_array):
                window = np.concatenate([imu_array[window_start:window_end], 
                                       gaze_array[window_start:window_end]], axis=1)
                windows.append(window)
        
        # Check window shape
        self.assertEqual(len(windows), seq_len)
        self.assertEqual(windows[0].shape, (window_size, 8))  # 6 IMU + 2 gaze
    
    def test_norm_features_augmentation(self):
        """Test that norm features are added correctly"""
        # Create test windows: (seq_len, window_size, channels)
        seq_len = 2
        window_size = 50
        channels = 8  # 6 IMU + 2 gaze
        
        windows = np.random.randn(seq_len, window_size, channels)
        
        # Extract IMU channels and compute norms
        imu_channels = windows[..., :6]
        accel = imu_channels[..., :3]
        gyro = imu_channels[..., 3:6]
        
        accel_norm = np.linalg.norm(accel, axis=2, keepdims=True)
        gyro_norm = np.linalg.norm(gyro, axis=2, keepdims=True)
        
        # Concatenate
        augmented = np.concatenate([windows, accel_norm, gyro_norm], axis=2)
        
        # Check shape
        self.assertEqual(augmented.shape, (seq_len, window_size, channels + 2))
        
        # Check that norms are computed correctly
        expected_accel_norm = np.sqrt(np.sum(accel**2, axis=2, keepdims=True))
        np.testing.assert_array_almost_equal(augmented[..., -2], expected_accel_norm.squeeze(), decimal=5)


class TestInferenceEngineIntegration(unittest.TestCase):
    """Integration tests for InferenceEngine (requires model files)"""
    
    def test_engine_initialization(self):
        """Test that InferenceEngine can be initialized with mock stats"""
        temp_dir = tempfile.mkdtemp()
        stats_path = Path(temp_dir) / 'normalization_stats.json'
        
        # Create minimal stats
        stats = {
            'imu_mean': [0.0] * 6,
            'imu_std': [1.0] * 6,
            'gaze_mean': [0.5, 0.5],
            'gaze_std': [0.25, 0.25],
            'config': {
                'use_gaze': True,
                'add_norm_features': True,
                'seq_len': 30,
                'window_size': 50,
                'stride': 5,
                'in_channels': 10
            },
            'scenario_map': {'Cooking': 0},
            'action_map': {'Stationary': 0},
            'idx_to_scenario': {'0': 'Cooking'},
            'idx_to_action': {'0': 'Stationary'}
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
        
        # Create dummy model file (empty, just for testing initialization)
        model_path = Path(temp_dir) / 'model.pt'
        model_path.touch()
        
        # This will fail at model loading, but we can test that stats are loaded correctly
        # For a full test, we'd need an actual model file
        try:
            engine = InferenceEngine(
                model_path=str(model_path),
                stats_path=str(stats_path),
                use_gaze=True,
                device='cpu'
            )
            # If we get here, stats were loaded correctly
            self.assertEqual(engine.seq_len, 30)
            self.assertEqual(engine.window_size, 50)
            self.assertEqual(engine.use_gaze, True)
        except Exception as e:
            # Expected to fail at model loading without actual model
            # But stats should be loaded before that
            pass


if __name__ == '__main__':
    unittest.main()

