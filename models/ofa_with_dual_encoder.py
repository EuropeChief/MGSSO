
import torch.nn as nn
from region_encoder import DeformableRegionEncoder
from grid_encoder import GridFeatureEnhancer
from fusion_module import RegionGridFusion

class OFAWithDualEncoder(nn.Module):
    def __init__(self, backbone, d_model=256):
        super().__init__()
        self.backbone = backbone  # Swin Transformer 输出 multiscale_feats
        self.region_encoder = DeformableRegionEncoder(d_model=d_model)
        self.grid_encoder = GridFeatureEnhancer(d_model=d_model)
        self.fusion_module = RegionGridFusion(d_model=d_model)

    def forward(self, x, geom_adj_matrix):
        multiscale_feats, V_lb = self.backbone(x)
        region_feats = self.region_encoder(multiscale_feats)
        grid_feats = self.grid_encoder(V_lb)
        Rout, Gout = self.fusion_module(region_feats, grid_feats, geom_adj_matrix)
        return Rout, Gout
