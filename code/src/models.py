"""
Neural Network Models for EC Prediction
=======================================
All model architectures for the EC benchmark.

Recommended architectures (best performance):
- MLP: Simple, fast, best accuracy
- DeepMLP: Slightly deeper, similar performance
- WideMLP: More parameters, marginal improvement
- AttentionMLP: Self-attention layer, good for some cases

Not recommended:
- Transformer: Fails to converge with default hyperparameters

Author: Rajesh Sathyamoorthy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class MLP(nn.Module):
    """
    Multi-layer perceptron with BatchNorm and Dropout.
    
    This is the recommended architecture for EC prediction from PLM embeddings.
    Simple, fast, and achieves the best performance.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class DeepMLP(nn.Module):
    """Deep MLP (1024-512-256)."""
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


class WideMLP(nn.Module):
    """Wide MLP (2048-1024-512)."""
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


class AttentionMLP(nn.Module):
    """MLP with self-attention layer."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 num_classes: int, n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        
        self.proj = nn.Linear(input_dim, hidden_dims[0])
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dims[0],
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dims[0])
        
        layers = []
        prev = hidden_dims[0]
        for h in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out).squeeze(1)
        return self.mlp(x)


class CNN1D(nn.Module):
    """1D CNN on embedding chunks."""
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.chunk = 64
        self.n_pos = input_dim // self.chunk
        
        self.conv = nn.Sequential(
            nn.Conv1d(self.chunk, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        b = x.size(0)
        x = x[:, :self.n_pos * self.chunk].view(b, self.n_pos, self.chunk).permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class ResidualBlock1D(nn.Module):
    """1D Residual block with skip connection."""
    
    def __init__(self, channels: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = x + residual
        return F.relu(x)


class ResNet1D(nn.Module):
    """1D ResNet with skip connections."""
    
    def __init__(self, input_dim: int, num_classes: int, 
                 n_blocks: int = 3, channels: int = 256, dropout: float = 0.3):
        super().__init__()
        self.chunk = 64
        self.n_pos = input_dim // self.chunk
        
        self.conv_in = nn.Sequential(
            nn.Conv1d(self.chunk, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )
        
        self.res_blocks = nn.Sequential(*[
            ResidualBlock1D(channels, dropout) for _ in range(n_blocks)
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        b = x.size(0)
        x = x[:, :self.n_pos * self.chunk].view(b, self.n_pos, self.chunk).permute(0, 2, 1)
        x = self.conv_in(x)
        x = self.res_blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling for sequence aggregation."""
    
    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        b, s, h = x.shape
        
        q = self.query(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, s, h)
        out = self.out(out)
        
        return out.mean(dim=1)


class MultiHeadAttentionMLP(nn.Module):
    """MLP with multi-head attention pooling."""
    
    def __init__(self, input_dim: int, num_classes: int, 
                 hidden: int = 512, n_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.chunk = 64
        self.n_pos = max(1, input_dim // self.chunk)
        
        self.proj = nn.Linear(self.chunk, hidden)
        self.pos = nn.Parameter(torch.randn(1, self.n_pos, hidden) * 0.02)
        
        self.attn_pool = MultiHeadAttentionPooling(hidden, n_heads, dropout)
        self.norm = nn.LayerNorm(hidden)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        b = x.size(0)
        x = x[:, :self.n_pos * self.chunk].view(b, self.n_pos, self.chunk)
        x = self.proj(x) + self.pos
        x = self.attn_pool(x)
        x = self.norm(x)
        return self.fc(x)


class HybridCNNTransformer(nn.Module):
    """Hybrid CNN + Transformer architecture."""
    
    def __init__(self, input_dim: int, num_classes: int,
                 cnn_channels: int = 256, transformer_hidden: int = 256,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.chunk = 64
        self.n_pos = input_dim // self.chunk
        
        # CNN feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(self.chunk, cnn_channels, 3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, 3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )
        
        # Transformer
        self.proj = nn.Linear(cnn_channels, transformer_hidden)
        self.pos = nn.Parameter(torch.randn(1, self.n_pos, transformer_hidden) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden,
            nhead=n_heads,
            dim_feedforward=transformer_hidden * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(transformer_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        b = x.size(0)
        x = x[:, :self.n_pos * self.chunk].view(b, self.n_pos, self.chunk).permute(0, 2, 1)
        x = self.cnn(x).permute(0, 2, 1)
        x = self.proj(x) + self.pos
        x = self.transformer(x).mean(dim=1)
        return self.fc(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder on embedding chunks.
    
    WARNING: This architecture often fails to converge with default hyperparameters.
    If using, consider: lower learning rate (1e-5), warmup schedule, more epochs.
    """
    
    def __init__(self, input_dim: int, num_classes: int, 
                 n_heads: int = 8, n_layers: int = 2, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.chunk = 64
        self.n_pos = max(1, input_dim // self.chunk)
        
        self.proj = nn.Linear(self.chunk, hidden)
        self.pos = nn.Parameter(torch.randn(1, self.n_pos, hidden) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
    
    def forward(self, x):
        b = x.size(0)
        x = x[:, :self.n_pos * self.chunk].view(b, self.n_pos, self.chunk)
        x = self.proj(x) + self.pos
        x = self.transformer(x).mean(dim=1)
        return self.fc(x)


# =============================================================================
# MODEL FACTORY
# =============================================================================

def get_model(arch: str, input_dim: int, num_classes: int, 
              dropout: float = 0.3, device: str = 'cuda') -> nn.Module:
    """
    Create model by architecture name.
    
    Args:
        arch: Architecture name (mlp, deep_mlp, wide_mlp, etc.)
        input_dim: Input embedding dimension
        num_classes: Number of output classes
        dropout: Dropout rate
        device: Device to place model on
    
    Returns:
        Initialized model on specified device
    """
    if arch == 'mlp':
        model = MLP(input_dim, [512, 256], num_classes, dropout)
    elif arch == 'deep_mlp':
        model = DeepMLP(input_dim, num_classes, dropout)
    elif arch == 'wide_mlp':
        model = WideMLP(input_dim, num_classes, dropout)
    elif arch == 'attention_mlp':
        model = AttentionMLP(input_dim, [512, 256], num_classes, dropout=dropout)
    elif arch == 'cnn':
        model = CNN1D(input_dim, num_classes, dropout)
    elif arch == 'resnet':
        model = ResNet1D(input_dim, num_classes, dropout=dropout)
    elif arch == 'multihead_attn':
        model = MultiHeadAttentionMLP(input_dim, num_classes, dropout=dropout)
    elif arch == 'hybrid_cnn_transformer':
        model = HybridCNNTransformer(input_dim, num_classes, dropout=dropout)
    elif arch == 'transformer':
        model = TransformerEncoder(input_dim, num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
