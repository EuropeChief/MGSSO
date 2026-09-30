
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableRegionEncoder(nn.Module):
    def __init__(self, num_layers=6, num_queries=100, d_model=256, num_scales=4, sample_points=4):
        super().__init__()
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.sample_points = sample_points
        self.d_model = d_model
        self.num_scales = num_scales

        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model))
        self.scale_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_scales)])
        self.offset_attention = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, sample_points * 2 + sample_points)
        ) for _ in range(num_layers)])

        self.bbox_head = nn.Linear(d_model, 4)
        self.cls_head = nn.Linear(d_model, 1)
        self.attr_head = nn.Linear(d_model, 10)

    def forward(self, multiscale_feats):
        B = multiscale_feats[0].shape[0]
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        for layer_idx in range(self.num_layers):
            all_features = []
            for scale_idx in range(self.num_scales):
                feat = multiscale_feats[scale_idx].permute(0, 2, 3, 1)
                H, W = feat.shape[1:3]
                feat_flat = feat.view(B, -1, self.d_model)
                proj_feat = self.scale_proj[scale_idx](feat_flat)
                offsets_attn = self.offset_attention[layer_idx](queries)
                offsets, attn = offsets_attn.split([self.sample_points * 2, self.sample_points], dim=-1)
                offsets = offsets.view(B, -1, self.sample_points, 2)
                attn = F.softmax(attn.view(B, -1, self.sample_points), dim=-1)
                coords = torch.sigmoid(queries[:, :, :2]).unsqueeze(2).expand(-1, -1, self.sample_points, -1)
                sample_coords = coords + offsets / torch.tensor([W, H], device=queries.device)
                sample_coords = torch.clamp(sample_coords, 0, 1)
                sample_coords = sample_coords * torch.tensor([W, H], device=queries.device)
                sample_coords = sample_coords.view(B, -1, 2)
                sampled_feats = self.bilinear_sample(feat, sample_coords)
                weighted = (sampled_feats * attn.unsqueeze(-1)).sum(dim=2)
                all_features.append(weighted)
            queries = sum(all_features)
        return queries

    def bilinear_sample(self, feat_map, coords):
        B, H, W, C = feat_map.shape
        coords = coords.clone()
        coords[..., 0] = coords[..., 0] / (W - 1) * 2 - 1
        coords[..., 1] = coords[..., 1] / (H - 1) * 2 - 1
        coords = coords.view(B, 1, -1, 2)
        feat_map = feat_map.permute(0, 3, 1, 2)
        sampled = F.grid_sample(feat_map, coords, align_corners=True)
        sampled = sampled.permute(0, 2, 3, 1).view(B, -1, self.sample_points, C)
        return sampled
