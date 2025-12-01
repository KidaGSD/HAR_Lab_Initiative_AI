import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: (B, C, T)
        b, c, t = x.shape
        se = x.mean(dim=2)  # (B, C)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).view(b, c, 1)
        return x * se


class LLE(nn.Module):
    """Low-Level Encoder with Variable Dilation CNNs + SE channel attention."""
    def __init__(self, config):
        super().__init__()
        filters = config['cnn_filters']
        in_ch = config['in_channels']
        self.se_reduction = config.get('se_reduction', 8)
        
        # Variable dilations to capture different periodicities [1, 2, 4]
        self.dilations = [1, 2, 4]
        
        # Create parallel multi-dilation conv blocks
        self.conv_blocks = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        
        for i in range(len(filters)):
            in_channels = in_ch if i == 0 else filters[i-1]
            
            # Each dilation gets equal share of output filters
            filters_per_dilation = filters[i] // len(self.dilations)
            remainder = filters[i] % len(self.dilations)
            
            # Create parallel convolutions with different dilations
            parallel_convs = nn.ModuleList()
            for j, dilation in enumerate(self.dilations):
                # Give remainder filters to first convolution
                out_ch = filters_per_dilation + (remainder if j == 0 else 0)
                parallel_convs.append(
                    nn.Conv1d(in_channels, out_ch, kernel_size=3, 
                             padding=dilation, dilation=dilation)
                )
            
            self.conv_blocks.append(parallel_convs)
            self.bns.append(nn.BatchNorm1d(filters[i]))
            self.se_blocks.append(SqueezeExcite(filters[i], reduction=self.se_reduction))
        
        self.gru = nn.GRU(filters[-1], config['gru_hidden'], config['gru_layers'], batch_first=True)
        self.fc = nn.Linear(config['gru_hidden'], config['embedding_dim'])
        
    def forward(self, x):
        # x: (B*Seq, 50, C) -> (B*Seq, C, 50)
        x = x.transpose(1, 2)
        
        # Apply multi-dilation conv blocks + SE channel attention
        for conv_block, bn, se in zip(self.conv_blocks, self.bns, self.se_blocks):
            conv_outputs = [conv(x) for conv in conv_block]
            x = torch.cat(conv_outputs, dim=1)  # Concat along channel dimension
            x = F.relu(bn(x))
            x = se(x)
        
        x = x.transpose(1, 2)
        _, h = self.gru(x)
        return self.fc(h[-1])


class HLA(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        if config.get('type', 'transformer') == 'transformer':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=config['nhead'],
                dim_feedforward=config['hidden_dim'] * 4,
                dropout=config.get('dropout', 0.1),
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
            self.pos_embed = nn.Parameter(torch.zeros(config['seq_len'] + 1, input_dim))
            self.head = nn.Linear(input_dim, config['num_classes'])
        else:
            self.gru = nn.GRU(input_dim, config['hidden_dim'], config['num_layers'], batch_first=True)
            self.head = nn.Linear(config['hidden_dim'], config['num_classes'])
        
    def forward(self, x):
        # x: (B, Seq, Emb)
        if hasattr(self, 'encoder'):
            seq_len = x.size(1)
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B,1,E)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, Seq+1, E)
            pos = self.pos_embed[:seq_len + 1, :].unsqueeze(0).to(x.device)
            x = x + pos
            enc = self.encoder(x)  # (B, Seq+1, Emb)
            cls_out = enc[:, 0, :]
            return self.head(cls_out)
        else:
            _, h = self.gru(x)
            return self.head(h[-1])


class HierarchicalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lle = LLE(config['lle'])
        self.hla = HLA(config['hla'], config['lle']['embedding_dim'])
        
        # Probing Head for LLE (Action Classification)
        self.action_head = nn.Linear(config['lle']['embedding_dim'], 6) 
        
    def forward(self, x):
        # x: (B, Seq, 50, C)
        b, s, w, c = x.shape
        
        # Flatten for LLE
        x_flat = x.view(b*s, w, c)
        
        # LLE Forward
        embeddings = self.lle(x_flat) # (B*S, Emb)
        
        # Action Logits (for Probing/Auxiliary Loss)
        action_logits = self.action_head(embeddings) # (B*S, 6)
        action_logits = action_logits.view(b, s, 6)
        
        # Reshape for HLA
        embeddings_seq = embeddings.view(b, s, -1)
        
        # HLA Forward
        scenario_logits = self.hla(embeddings_seq) # (B, NumScenarios)
        
        return scenario_logits, action_logits
