
import torch
import torch.nn as nn
import torch.nn.functional as F

class GridFeatureEnhancer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024):
        super().__init__()
        self.linear_proj = nn.Linear(d_model, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, V_lb):
        B, C, H, W = V_lb.shape
        M = H * W
        feat = V_lb.view(B, C, M).permute(0, 2, 1)
        G0 = self.linear_proj(feat)
        G = self.encoder(G0)
        return G
