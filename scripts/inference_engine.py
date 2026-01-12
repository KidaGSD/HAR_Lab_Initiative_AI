#!/usr/bin/env python3
"""
Real-time inference engine for hierarchical activity recognition model.
Handles sliding window buffer, preprocessing, and model inference.

Usage:
    from inference_engine import InferenceEngine
    
    engine = InferenceEngine(
        model_path='deployment/model.onnx',
        stats_path='deployment/normalization_stats.json'
    )
    
    # Add sensor samples
    engine.add_sample(imu_sample, gaze_sample)
    
    # Run inference
    scenario_pred, action_preds = engine.infer()
"""

import numpy as np
import json
from pathlib import Path
from collections import deque
import torch

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("WARNING: onnxruntime not installed. Install with: pip install onnxruntime")


class SlidingWindowBuffer:
    """Maintains a sliding window buffer for sensor data"""
    
    def __init__(self, max_samples, num_imu_channels=6, num_gaze_channels=2):
        """
        Args:
            max_samples: Maximum number of samples to store (e.g., 30s * 50Hz = 1500)
            num_imu_channels: Number of IMU channels (6)
            num_gaze_channels: Number of gaze channels (2)
        """
        self.max_samples = max_samples
        self.num_imu_channels = num_imu_channels
        self.num_gaze_channels = num_gaze_channels
        
        # Use deque for efficient append/pop operations
        self.imu_buffer = deque(maxlen=max_samples)
        self.gaze_buffer = deque(maxlen=max_samples)
        
    def add_sample(self, imu_sample, gaze_sample=None):
        """
        Add a new sample to the buffer.
        
        Args:
            imu_sample: numpy array of shape (num_imu_channels,) or (6,)
            gaze_sample: numpy array of shape (num_gaze_channels,) or (2,), or None
        """
        # Validate shapes
        if imu_sample.shape != (self.num_imu_channels,):
            raise ValueError(f"IMU sample shape {imu_sample.shape} != ({self.num_imu_channels},)")
        
        self.imu_buffer.append(imu_sample.copy())
        
        if gaze_sample is not None:
            if gaze_sample.shape != (self.num_gaze_channels,):
                raise ValueError(f"Gaze sample shape {gaze_sample.shape} != ({self.num_gaze_channels},)")
            self.gaze_buffer.append(gaze_sample.copy())
        else:
            # Zero-fill if gaze is missing
            self.gaze_buffer.append(np.zeros(self.num_gaze_channels))
    
    def get_array(self, use_gaze=True):
        """
        Get current buffer as numpy arrays.
        
        Returns:
            imu_array: (N, num_imu_channels) where N is current buffer size
            gaze_array: (N, num_gaze_channels) or None if use_gaze=False
        """
        imu_array = np.array(self.imu_buffer)
        
        if use_gaze and len(self.gaze_buffer) > 0:
            gaze_array = np.array(self.gaze_buffer)
            return imu_array, gaze_array
        else:
            return imu_array, None
    
    def is_full(self):
        """Check if buffer has reached maximum capacity"""
        return len(self.imu_buffer) >= self.max_samples
    
    def size(self):
        """Get current buffer size"""
        return len(self.imu_buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.imu_buffer.clear()
        self.gaze_buffer.clear()


class InferenceEngine:
    """Real-time inference engine for hierarchical activity recognition"""
    
    def __init__(self, model_path, stats_path, use_gaze=None, device='cpu'):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to ONNX or TorchScript model
            stats_path: Path to normalization_stats.json
            use_gaze: Whether to use gaze data (None = auto-detect from stats.json)
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_path = Path(model_path)
        self.stats_path = Path(stats_path)
        self.device = device
        
        # Load normalization statistics
        with open(self.stats_path, 'r') as f:
            self.stats = json.load(f)
        
        self.imu_mean = np.array(self.stats['imu_mean'])
        self.imu_std = np.array(self.stats['imu_std'])
        self.gaze_mean = np.array(self.stats['gaze_mean'])
        self.gaze_std = np.array(self.stats['gaze_std'])
        
        # Get config from stats (this determines the actual model architecture)
        self.config = self.stats['config']
        self.seq_len = self.config['seq_len']
        self.window_size = self.config['window_size']
        self.stride = self.config.get('stride', 5)
        self.add_norm_features = self.config.get('add_norm_features', True)
        self.in_channels = self.config['in_channels']
        
        # Auto-detect use_gaze from config if not explicitly provided
        if use_gaze is None:
            self.use_gaze = self.config.get('use_gaze', False)
            print(f"⚠️  use_gaze not specified, auto-detected from stats.json: {self.use_gaze}")
        else:
            self.use_gaze = use_gaze
            # Warn if there's a mismatch
            config_use_gaze = self.config.get('use_gaze', False)
            if self.use_gaze != config_use_gaze:
                print(f"⚠️  WARNING: use_gaze={self.use_gaze} doesn't match model config (use_gaze={config_use_gaze})")
                print(f"   Model expects {self.in_channels} channels. This may cause errors!")
        
        # Load label mappings
        self.scenario_map = self.stats['scenario_map']
        self.action_map = self.stats['action_map']
        self.idx_to_scenario = self.stats['idx_to_scenario']
        self.idx_to_action = self.stats['idx_to_action']
        
        # Initialize sliding window buffer (30 seconds at 50Hz)
        buffer_size = self.seq_len * self.window_size  # 30 * 50 = 1500 samples
        self.buffer = SlidingWindowBuffer(
            max_samples=buffer_size,
            num_imu_channels=6,
            num_gaze_channels=2
        )
        
        # Load model
        self.model = self._load_model()
        
        print(f"✓ InferenceEngine initialized")
        print(f"   Model: {self.model_path}")
        print(f"   Sequence length: {self.seq_len}")
        print(f"   Window size: {self.window_size}")
        print(f"   Use gaze: {self.use_gaze}")
    
    def _load_model(self):
        """Load ONNX or TorchScript model"""
        if self.model_path.suffix == '.onnx':
            if not ONNX_AVAILABLE:
                raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")
            
            # Import here to avoid issues when ONNX is not available
            import onnxruntime as ort
            
            # Create ONNX Runtime session
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            session = ort.InferenceSession(str(self.model_path), providers=providers)
            print(f"✓ Loaded ONNX model with providers: {providers}")
            return session
            
        elif self.model_path.suffix == '.pt':
            # Load TorchScript model
            model = torch.jit.load(str(self.model_path), map_location=self.device)
            model.eval()
            print(f"✓ Loaded TorchScript model")
            return model
        else:
            raise ValueError(f"Unsupported model format: {self.model_path.suffix}")
    
    def add_sample(self, imu_sample, gaze_sample=None):
        """
        Add a new sensor sample to the buffer.
        
        Args:
            imu_sample: numpy array of shape (6,) [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
            gaze_sample: numpy array of shape (2,) [norm_pos_x, norm_pos_y] or None
        """
        if gaze_sample is None and self.use_gaze:
            # Zero-fill missing gaze
            gaze_sample = np.zeros(2)
        
        self.buffer.add_sample(imu_sample, gaze_sample if self.use_gaze else None)
    
    def _preprocess(self):
        """
        Preprocess buffer data: normalize, augment, and create windows.
        
        Returns:
            windows: numpy array of shape (1, seq_len, window_size, in_channels)
        """
        # Get buffer arrays
        imu_array, gaze_array = self.buffer.get_array(use_gaze=self.use_gaze)
        
        if len(imu_array) < self.window_size:
            raise ValueError(f"Buffer too small: {len(imu_array)} < {self.window_size}")
        
        # Normalize IMU
        imu_normalized = (imu_array - self.imu_mean) / self.imu_std
        
        # Normalize gaze if enabled AND available
        if self.use_gaze:
            if gaze_array is not None:
                gaze_normalized = (gaze_array - self.gaze_mean) / self.gaze_std
                # Concatenate IMU and gaze
                combined = np.concatenate([imu_normalized, gaze_normalized], axis=1)  # (N, 8)
            else:
                # Gaze enabled but missing - zero-fill
                gaze_normalized = np.zeros((len(imu_normalized), 2))
                combined = np.concatenate([imu_normalized, gaze_normalized], axis=1)  # (N, 8)
        else:
            # Gaze disabled - use IMU only
            combined = imu_normalized  # (N, 6)
        
        # Create windows with stride
        windows = []
        num_windows = (len(combined) - self.window_size) // self.stride + 1
        
        # Extract last seq_len windows
        start_idx = max(0, num_windows - self.seq_len) * self.stride
        
        for i in range(self.seq_len):
            window_start = start_idx + i * self.stride
            window_end = window_start + self.window_size
            
            if window_end <= len(combined):
                window = combined[window_start:window_end]  # (window_size, channels)
                windows.append(window)
            else:
                # Pad with last sample if needed
                last_window = combined[-self.window_size:]
                windows.append(last_window)
        
        windows = np.array(windows)  # (seq_len, window_size, channels)
        
        # Add norm features if enabled
        if self.add_norm_features:
            windows = self._augment_features(windows)
        
        # Add batch dimension: (1, seq_len, window_size, in_channels)
        windows = windows[np.newaxis, ...]
        
        return windows
    
    def _augment_features(self, windows):
        """
        Add norm features (accel_norm, gyro_norm) to windows.
        
        Args:
            windows: (seq_len, window_size, channels) where channels = 6 (IMU) or 8 (IMU+gaze)
        
        Returns:
            augmented: (seq_len, window_size, channels+2)
        """
        # Extract IMU channels (first 6 channels)
        imu_channels = windows[..., :6]  # (seq_len, window_size, 6)
        accel = imu_channels[..., :3]  # (seq_len, window_size, 3)
        gyro = imu_channels[..., 3:6]  # (seq_len, window_size, 3)
        
        # Compute norms
        accel_norm = np.linalg.norm(accel, axis=2, keepdims=True)  # (seq_len, window_size, 1)
        gyro_norm = np.linalg.norm(gyro, axis=2, keepdims=True)  # (seq_len, window_size, 1)
        
        # Concatenate
        augmented = np.concatenate([windows, accel_norm, gyro_norm], axis=2)
        return augmented
    
    def infer(self):
        """
        Run inference on current buffer.
        
        Returns:
            scenario_pred: str, predicted scenario name
            action_preds: list of str, predicted actions for each window
            scenario_probs: dict, probability for each scenario
            action_probs: list of dict, probabilities for each action window
        """
        if self.buffer.size() < self.window_size:
            raise ValueError(f"Buffer too small: {self.buffer.size()} < {self.window_size}")
        
        # Preprocess
        windows = self._preprocess()  # (1, seq_len, window_size, in_channels)
        
        # Run inference
        # Check if model is ONNX Runtime session (check by type name to avoid import issues)
        model_type_name = type(self.model).__name__
        if ONNX_AVAILABLE and model_type_name == 'InferenceSession':
            # ONNX Runtime
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: windows.astype(np.float32)})
            scenario_logits = outputs[0]  # (1, num_scenarios)
            action_logits = outputs[1]  # (1, seq_len, num_actions)
        else:
            # TorchScript (or any other PyTorch model)
            with torch.no_grad():
                windows_tensor = torch.from_numpy(windows).float()
                if self.device == 'cuda':
                    windows_tensor = windows_tensor.cuda()
                outputs = self.model(windows_tensor)
                scenario_logits = outputs[0].cpu().numpy()
                action_logits = outputs[1].cpu().numpy()
        
        # Post-process: convert logits to predictions
        scenario_probs = self._softmax(scenario_logits[0])
        scenario_idx = np.argmax(scenario_logits[0])
        scenario_pred = self.idx_to_scenario[str(scenario_idx)]
        
        # Action predictions (per window)
        action_preds = []
        action_probs_list = []
        for i in range(action_logits.shape[1]):
            action_probs = self._softmax(action_logits[0, i])
            action_idx = np.argmax(action_logits[0, i])
            action_pred = self.idx_to_action[str(action_idx)]
            action_preds.append(action_pred)
            action_probs_list.append(action_probs)
        
        return scenario_pred, action_preds, scenario_probs, action_probs_list
    
    def _softmax(self, logits):
        """Compute softmax probabilities"""
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return {self.idx_to_scenario[str(i)]: float(probs[i]) for i in range(len(probs))}
    
    def reset(self):
        """Reset the buffer"""
        self.buffer.clear()
    
    def get_buffer_status(self):
        """Get current buffer status"""
        return {
            'size': self.buffer.size(),
            'max_size': self.buffer.max_samples,
            'is_full': self.buffer.is_full(),
            'percent_full': self.buffer.size() / self.buffer.max_samples * 100
        }

