import numpy as np
import pandas as pd
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


class HierarchicalDataset(Dataset):
    def __init__(self, take_uids, processed_dir, scenario_labels_path, action_labels_path, config, training=True):
        self.samples = []
        self.config = config
        self.training = training  # For augmentation
        self.augmentation_enabled = config['data'].get('augmentation', False)
        
        # Load Labels
        print("Loading label files...")
        self.scenario_df = pd.read_csv(scenario_labels_path).set_index('video_uid')
        self.action_df = pd.read_csv(action_labels_path)

        # Scenarios to exclude (e.g., insufficient training data)
        self.excluded_scenarios = set(config['data'].get('excluded_scenarios', []))
        if self.excluded_scenarios:
            print(f"Excluding scenarios: {self.excluded_scenarios}")

        # Map Scenario Names to Integers (excluding specified scenarios)
        all_scenarios = sorted(self.scenario_df['scenario'].unique())
        valid_scenarios = [s for s in all_scenarios if s not in self.excluded_scenarios]
        self.scenario_map = {name: i for i, name in enumerate(valid_scenarios)}
        self.num_scenarios = len(self.scenario_map)
        self.idx_to_scenario = {v: k for k, v in self.scenario_map.items()}
        print(f"Scenarios ({self.num_scenarios} classes): {self.scenario_map}")
        
        # Map Action Names to Integers (4 Classes - Motion-based)
        # Remapped from 6 semantic classes to 4 IMU-distinguishable classes
        self.action_map = {
            'Stationary': 0, 
            'Locomotion': 1, 
            'Manipulation': 2, 
            'Search_Interrupt': 3,
        }
        self.num_action_classes = len(self.action_map)
        self.idx_to_action = {v: k for k, v in self.action_map.items()}
        # Keep only clean action labels (drop Unknown/Uncertain)
        self.action_df = self.action_df[self.action_df['action'].isin(self.action_map.keys())]
        self.action_pad = self.config['data'].get('action_label_pad', 0.5)
        self.per_video_center = self.config['data'].get('per_video_center', True)
        self.add_norm_features = self.config['data'].get('add_norm_features', True)

        # === GLOBAL NORMALIZATION (with optional feature augmentation) ===
        print("\nComputing global normalization statistics...")
        all_data = []
        valid_uids = []
        
        for uid in tqdm(take_uids, desc='Collecting normalization data'):
            seq_path = Path(processed_dir) / uid / 'seq.npz'
            if not seq_path.exists():
                continue
            try:
                data = np.load(seq_path)
                traj = data['traj']  # (N, 50, 6)
                traj = self._augment_traj(traj)  # add norms if enabled
                if len(traj) > 0 and uid in self.scenario_df.index:
                    all_data.append(traj)
                    valid_uids.append(uid)
            except Exception:
                continue
        
        if len(all_data) == 0:
            raise ValueError("No valid data found for normalization!")
            
        all_data = np.concatenate(all_data, axis=0)  # (Total_Windows, 50, C)
        self.global_mean = all_data.mean(axis=(0, 1))  # (C,) - mean per channel
        self.global_std = all_data.std(axis=(0, 1)) + 1e-6  # (C,) - std per channel
        
        print(f"Global statistics computed from {len(valid_uids)} videos:")
        print(f"  Mean: {self.global_mean}")
        print(f"  Std:  {self.global_std}")
        print(f"  Data range: [{all_data.min():.2f}, {all_data.max():.2f}]")
        # ==========================================
        
        # Iterate Videos
        for uid in tqdm(take_uids, desc='Loading Data'):
            seq_path = Path(processed_dir) / uid / 'seq.npz'
            if not seq_path.exists():
                continue
                
            try:
                data = np.load(seq_path)
                traj = data['traj'] # (N, 50, 6)
                traj = self._augment_traj(traj)
                timestamps = data['timestamp'] # (N, 50)
                
                # GLOBAL normalization + optional per-video centering
                if len(traj) > 0:
                    traj = (traj - self.global_mean) / self.global_std
                    if self.per_video_center:
                        traj = traj - traj.mean(axis=(0, 1), keepdims=True)
                
                if len(traj) == 0:
                    continue
                
                # Get Scenario Label (Global for video)
                if uid not in self.scenario_df.index:
                    continue
                scenario_name = self.scenario_df.loc[uid, 'scenario']
                if scenario_name not in self.scenario_map:
                    continue 
                scenario_label = self.scenario_map[scenario_name]
                
                # Get Action Labels for this video
                video_actions = self.action_df[self.action_df['video_uid'] == uid]
                
                # Create Windows
                seq_len = self.config['hla']['seq_len']
                stride = 5  # denser stride for more supervision
                
                num_seqs = (len(traj) - seq_len) // stride + 1
                if num_seqs <= 0:
                    continue
                
                for i in range(num_seqs):
                    start_idx = i * stride
                    end_idx = start_idx + seq_len
                    
                    # Window Sequence
                    window_seq = traj[start_idx:end_idx] # (Seq, 50, C)
                    ts_windows = timestamps[start_idx:end_idx] # (Seq, 50)
                    
                    action_labels_seq = []
                    
                    for w_idx in range(seq_len):
                        w_ts = ts_windows[w_idx]
                        w_start = w_ts[0]
                        w_end = w_ts[-1]
                        pad = self.action_pad
                        
                        # Find actions within padded window
                        match = video_actions[
                            (video_actions['timestamp_sec'] >= w_start - pad) & 
                            (video_actions['timestamp_sec'] <= w_end + pad)
                        ]
                        
                        if not match.empty:
                            # Majority vote within the window
                            counts = match['action'].value_counts()
                            act_name = counts.idxmax()
                            if act_name in self.action_map:
                                action_labels_seq.append(self.action_map[act_name])
                            else:
                                action_labels_seq.append(-1) # Unknown class
                        else:
                            action_labels_seq.append(-1) # No label in this window
                    
                    # Guard against variable-length windows
                    if window_seq.shape[0] != seq_len or len(action_labels_seq) != seq_len:
                        continue
                    
                    self.samples.append({
                        'video_uid': uid,
                        'inputs': torch.tensor(np.ascontiguousarray(window_seq), dtype=torch.float32), # (Seq, 50, C)
                        'scenario_label': torch.tensor(scenario_label, dtype=torch.long),
                        'action_labels': torch.tensor(action_labels_seq, dtype=torch.long) # (Seq,)
                    })
                
            except Exception as e:
                print(f"Error loading {uid}: {e}")
                
    def _augment_traj(self, traj):
        """Add EgoCHARM-style statistical features to trajectory.
        
        Input: traj (N, 50, 6) - raw accel (3) + gyro (3)
        Output: (N, 50, C) where C includes:
            - Original 6 channels
            - Accel/Gyro norms (2 channels)
            - Window-level variance broadcast (6 channels) - EgoCHARM inspired
        
        Total: 14 channels (if all enabled)
        """
        if not self.add_norm_features:
            return traj
        
        # Original IMU data
        accel = traj[..., :3]  # (N, 50, 3)
        gyro = traj[..., 3:6]  # (N, 50, 3)
        
        # Norm features (existing)
        accel_norm = np.linalg.norm(accel, axis=2, keepdims=True)  # (N, 50, 1)
        gyro_norm = np.linalg.norm(gyro, axis=2, keepdims=True)    # (N, 50, 1)
        
        # EgoCHARM statistical features: variance per window (broadcast to all timesteps)
        # This helps distinguish static vs dynamic activities
        var_per_window = traj.var(axis=1, keepdims=True)  # (N, 1, 6)
        var_broadcast = np.broadcast_to(var_per_window, traj.shape)  # (N, 50, 6)
        
        # Concatenate all features: 6 + 2 + 6 = 14 channels
        return np.concatenate([traj, accel_norm, gyro_norm, var_broadcast], axis=2)
    
    def _apply_augmentation(self, x):
        """Apply comprehensive time-series augmentation during training.
        x: Tensor of shape (seq_len, window_size, channels)
        
        Augmentation techniques (Tier 2 improvements):
        1. Jittering - add random noise
        2. Scaling - random magnitude scaling  
        3. Time Warping - slight temporal distortion
        4. Rotation - rotate accel/gyro vectors (first 6 channels only)
        """
        if not self.training or not self.augmentation_enabled:
            return x
        
        # 1. Jittering: add small random noise
        if random.random() < 0.5:
            noise = torch.randn_like(x) * 0.02
            x = x + noise
        
        # 2. Scaling: random magnitude scaling
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            x = x * scale
        
        # 3. Time Warping: slightly stretch/compress time axis
        if random.random() < 0.3:
            x = self._time_warp(x)
        
        # 4. Rotation: apply small rotation to IMU vectors (first 6 channels)
        if random.random() < 0.3:
            x = self._rotate_imu(x)
        
        return x
    
    def _time_warp(self, x, sigma=0.1):
        """Apply smooth time warping to the sequence."""
        seq_len, window_size, channels = x.shape
        
        # Create smooth warping curve using cumulative sum of random values
        warp_steps = torch.cumsum(torch.abs(torch.randn(window_size)) + 1, dim=0)
        warp_steps = warp_steps / warp_steps[-1] * (window_size - 1)
        warp_steps = warp_steps.clamp(0, window_size - 1).long()
        
        # Apply warping to each sequence in the batch
        warped = x.clone()
        for s in range(seq_len):
            warped[s] = x[s, warp_steps]
        
        return warped
    
    def _rotate_imu(self, x, max_angle_deg=10):
        """Apply small 3D rotation to accelerometer and gyroscope channels."""
        # Only rotate the first 6 channels (accel + gyro in raw form)
        # Derived features (norms, variance) are rotation-invariant or recomputed
        
        angle = random.uniform(-max_angle_deg, max_angle_deg) * (3.14159 / 180)
        
        # Simple rotation around Z-axis for efficiency  
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=x.dtype)
        
        rotated = x.clone()
        seq_len, window_size, channels = x.shape
        
        # Rotate accel (channels 0-2) and gyro (channels 3-5)
        if channels >= 6:
            for i in range(seq_len):
                for j in range(window_size):
                    # Rotate accelerometer
                    accel = x[i, j, :3]
                    rotated[i, j, :3] = rot_matrix @ accel
                    # Rotate gyroscope
                    gyro = x[i, j, 3:6]
                    rotated[i, j, 3:6] = rot_matrix @ gyro
        
        return rotated
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Apply augmentation if enabled
        if self.training and self.augmentation_enabled:
            inputs = self._apply_augmentation(sample['inputs'].clone())
            return {
                'inputs': inputs,
                'scenario_label': sample['scenario_label'],
                'action_labels': sample['action_labels'],
            }
        
        return sample

