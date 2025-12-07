"""
Baseline Models for HAR-IMU Comparison
Implements 4 baseline architectures for comparison with our Hierarchical model.

Baselines:
1. MLP-MLP: Hand-picked features + MLP (simplest)
2. CNN-MLP: Simple 1D-CNN + MLP  
3. IMU2CLIP: Large CNN-GRU (comparable to our model size)
4. CNN-LSTM-GRU: CNN-LSTM + GRU (recurrent baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP_LLE(nn.Module):
    """Hand-picked features + MLP for LLE (for MLP-MLP baseline)"""
    def __init__(self, config):
        super().__init__()
        in_ch = config['in_channels']
        window_size = config['window_size']  # 50 timesteps
        
        # Extract hand-picked features: mean, std, min, max, energy, zero-crossing rate per channel
        # 6 features per channel × in_ch channels = feature_dim
        self.feature_dim = 6 * in_ch
        
        # MLP: feature_dim -> hidden -> embedding_dim
        hidden_dim = 64  # Small to match ~5k parameters
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, config['embedding_dim'])
        )
    
    def extract_features(self, x):
        """Extract hand-picked statistical features from IMU window
        x: (B*Seq, 50, C)
        Returns: (B*Seq, feature_dim)
        """
        # Mean per channel
        mean = x.mean(dim=1)  # (B*Seq, C)
        
        # Std per channel
        std = x.std(dim=1)  # (B*Seq, C)
        
        # Min per channel
        min_val = x.min(dim=1)[0]  # (B*Seq, C)
        
        # Max per channel
        max_val = x.max(dim=1)[0]  # (B*Seq, C)
        
        # Energy (sum of squares) per channel
        energy = (x ** 2).sum(dim=1) / x.shape[1]  # (B*Seq, C)
        
        # Zero-crossing rate per channel
        zcr = torch.zeros_like(mean)
        for c in range(x.shape[2]):
            x_channel = x[:, :, c]  # (B*Seq, 50)
            sign_changes = (x_channel[:, 1:] * x_channel[:, :-1] < 0).sum(dim=1)
            zcr[:, c] = sign_changes.float() / (x_channel.shape[1] - 1)
        
        # Concatenate all features: (B*Seq, 6*C)
        features = torch.cat([mean, std, min_val, max_val, energy, zcr], dim=1)
        return features
    
    def forward(self, x):
        # x: (B*Seq, 50, C)
        features = self.extract_features(x)  # (B*Seq, feature_dim)
        return self.mlp(features)  # (B*Seq, embedding_dim)


class CNN_LLE_Simple(nn.Module):
    """Simple 1D-CNN for LLE (for CNN-MLP baseline)"""
    def __init__(self, config):
        super().__init__()
        filters = config['cnn_filters']
        in_ch = config['in_channels']
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch if i==0 else filters[i-1], filters[i], 3, padding=1)
            for i in range(len(filters))
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(f) for f in filters])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(filters[-1], config['embedding_dim'])
    
    def forward(self, x):
        # x: (B*Seq, 50, C) -> (B*Seq, C, 50)
        x = x.transpose(1, 2)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
        x = self.pool(x).squeeze(-1)  # (B*Seq, C)
        return self.fc(x)


class IMU2CLIP_LLE(nn.Module):
    """CNN-GRU (large) for LLE (for IMU2CLIP baseline)"""
    def __init__(self, config):
        super().__init__()
        filters = config['cnn_filters']
        in_ch = config['in_channels']
        
        # Larger CNN filters for IMU2CLIP
        large_filters = [f * 2 for f in filters]  # Double the filters: [64, 128, 256]
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch if i==0 else large_filters[i-1], large_filters[i], 3, padding=1)
            for i in range(len(large_filters))
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(f) for f in large_filters])
        # Larger GRU for IMU2CLIP
        gru_hidden = config['gru_hidden'] * 2  # 512 instead of 256
        self.gru = nn.GRU(large_filters[-1], gru_hidden, config['gru_layers'], batch_first=True)
        self.fc = nn.Linear(gru_hidden, config['embedding_dim'])
    
    def forward(self, x):
        # x: (B*Seq, 50, C) -> (B*Seq, C, 50)
        x = x.transpose(1, 2)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
        x = x.transpose(1, 2)  # (B*Seq, 50, C)
        _, h = self.gru(x)
        return self.fc(h[-1])


class CNN_LSTM_LLE(nn.Module):
    """CNN-LSTM for LLE (for CNN-LSTM-GRU baseline)"""
    def __init__(self, config):
        super().__init__()
        filters = config['cnn_filters']
        in_ch = config['in_channels']
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch if i==0 else filters[i-1], filters[i], 3, padding=1)
            for i in range(len(filters))
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(f) for f in filters])
        self.lstm = nn.LSTM(filters[-1], config['gru_hidden'], config['gru_layers'], batch_first=True)
        self.fc = nn.Linear(config['gru_hidden'], config['embedding_dim'])
    
    def forward(self, x):
        # x: (B*Seq, 50, C) -> (B*Seq, C, 50)
        x = x.transpose(1, 2)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
        x = x.transpose(1, 2)  # (B*Seq, 50, C)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class MLP_HLA(nn.Module):
    """Simple MLP for HLA (for CNN-MLP and MLP-MLP baselines)"""
    def __init__(self, config, input_dim):
        super().__init__()
        hidden_dim = config['hidden_dim']
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * config['seq_len'], hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(hidden_dim, config['num_classes'])
        )
    
    def forward(self, x):
        # x: (B, Seq, Emb) -> (B, Seq*Emb)
        b, seq, emb = x.shape
        x = x.view(b, -1)
        return self.mlp(x)


class GRU_HLA(nn.Module):
    """GRU for HLA (for CNN-LSTM-GRU baseline)"""
    def __init__(self, config, input_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, config['hidden_dim'], config['num_layers'], batch_first=True)
        self.head = nn.Linear(config['hidden_dim'], config['num_classes'])
    
    def forward(self, x):
        # x: (B, Seq, Emb)
        _, h = self.gru(x)
        return self.head(h[-1])


class BaselineModel(nn.Module):
    """Wrapper model for baseline LLE + HLA combinations"""
    def __init__(self, lle, hla, config):
        super().__init__()
        self.lle = lle
        self.hla = hla
        # Use configurable num_action_classes (default 4)
        num_action_classes = config.get('lla', {}).get('num_classes', 4)
        self.num_action_classes = num_action_classes
        self.action_head = nn.Linear(config['lle']['embedding_dim'], num_action_classes)
    
    def forward(self, x):
        b, s, w, c = x.shape
        x_flat = x.view(b*s, w, c)
        embeddings = self.lle(x_flat)
        action_logits = self.action_head(embeddings).view(b, s, self.num_action_classes)
        embeddings_seq = embeddings.view(b, s, -1)
        scenario_logits = self.hla(embeddings_seq)
        return scenario_logits, action_logits


def create_baseline_model(model_name, config):
    """Factory function to create baseline models
    
    Args:
        model_name: One of 'mlp_mlp', 'cnn_mlp', 'imu2clip', 'cnn_lstm_gru'
        config: Model configuration dict
    
    Returns:
        BaselineModel instance
    """
    if model_name == 'mlp_mlp':
        lle = MLP_LLE(config['lle'])
        hla = MLP_HLA(config['hla'], config['lle']['embedding_dim'])
    elif model_name == 'cnn_mlp':
        lle = CNN_LLE_Simple(config['lle'])
        hla = MLP_HLA(config['hla'], config['lle']['embedding_dim'])
    elif model_name == 'imu2clip':
        lle = IMU2CLIP_LLE(config['lle'])
        hla = MLP_HLA(config['hla'], config['lle']['embedding_dim'])
    elif model_name == 'cnn_lstm_gru':
        lle = CNN_LSTM_LLE(config['lle'])
        hla = GRU_HLA(config['hla'], config['lle']['embedding_dim'])
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: mlp_mlp, cnn_mlp, imu2clip, cnn_lstm_gru")
    
    return BaselineModel(lle, hla, config)


# Model descriptions for documentation
BASELINE_DESCRIPTIONS = {
    'mlp_mlp': 'Hand-picked statistical features (mean, std, min, max, energy, ZCR) + MLP',
    'cnn_mlp': 'Simple 3-layer 1D-CNN + Global Average Pooling + MLP',
    'imu2clip': 'Large CNN-GRU (2x filters, 2x GRU hidden) inspired by IMU2CLIP',
    'cnn_lstm_gru': 'CNN-LSTM for window encoding + GRU for sequence modeling',
}
