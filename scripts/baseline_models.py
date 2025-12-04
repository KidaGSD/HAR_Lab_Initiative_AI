import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Simple MLP for HLA (for CNN-MLP baseline)"""
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
    """GRU for HLA (for EgoCHARM baseline)"""
    def __init__(self, config, input_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, config['hidden_dim'], config['num_layers'], batch_first=True)
        self.head = nn.Linear(config['hidden_dim'], config['num_classes'])
    
    def forward(self, x):
        # x: (B, Seq, Emb)
        _, h = self.gru(x)
        return self.head(h[-1])

def create_baseline_model(model_name, config):
    """Factory function to create baseline models"""
    if model_name == 'cnn_mlp':
        lle = CNN_LLE_Simple(config['lle'])
        hla = MLP_HLA(config['hla'], config['lle']['embedding_dim'])
    elif model_name == 'egocharm':
        # Use your existing LLE, but with GRU HLA
        from train_hierarchical import LLE
        lle = LLE(config['lle'])
        hla = GRU_HLA(config['hla'], config['lle']['embedding_dim'])
    elif model_name == 'cnn_lstm_gru':
        lle = CNN_LSTM_LLE(config['lle'])
        hla = GRU_HLA(config['hla'], config['lle']['embedding_dim'])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create wrapper model
    class BaselineModel(nn.Module):
        def __init__(self, lle, hla):
            super().__init__()
            self.lle = lle
            self.hla = hla
            self.action_head = nn.Linear(config['lle']['embedding_dim'], 6)
        
        def forward(self, x):
            b, s, w, c = x.shape
            x_flat = x.view(b*s, w, c)
            embeddings = self.lle(x_flat)
            action_logits = self.action_head(embeddings).view(b, s, 6)
            embeddings_seq = embeddings.view(b, s, -1)
            scenario_logits = self.hla(embeddings_seq)
            return scenario_logits, action_logits
    
    return BaselineModel(lle, hla)