import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


class HierarchicalDataset(Dataset):
    def __init__(self, take_uids, processed_dir, scenario_labels_path, action_labels_path, config):
        self.samples = []
        self.config = config
        
        # Load Labels
        print("Loading label files...")
        self.scenario_df = pd.read_csv(scenario_labels_path).set_index('video_uid')
        self.action_df = pd.read_csv(action_labels_path)
        
        # Map Scenario Names to Integers
        self.scenario_map = {name: i for i, name in enumerate(sorted(self.scenario_df['scenario'].unique()))}
        self.num_scenarios = len(self.scenario_map)
        self.idx_to_scenario = {v: k for k, v in self.scenario_map.items()}
        print(f"Scenarios: {self.scenario_map}")
        
        # Map Action Names to Integers (6 Classes)
        self.action_map = {
            'Stationary': 0, 
            'Locomotion': 1, 
            'Essential Operation': 2, 
            'Object Transfer': 3,
            'Search': 4,
            'Error / Correction': 5
        }
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
        # traj: (N, 50, 6)
        if not self.add_norm_features:
            return traj
        accel = traj[..., :3]
        gyro = traj[..., 3:6]
        accel_norm = np.linalg.norm(accel, axis=2, keepdims=True)
        gyro_norm = np.linalg.norm(gyro, axis=2, keepdims=True)
        return np.concatenate([traj, accel_norm, gyro_norm], axis=2)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
