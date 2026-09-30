
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegionGridFusion(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.offset_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)
        )
        self.gate_r = nn.Linear(d_model * 2, d_model)
        self.gate_g = nn.Linear(d_model * 2, d_model)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, R, G, A):
        B, N, d = R.shape
        M = G.shape[1]
        r_proj = self.wq(R)
        g_proj = self.wk(G)
        v_proj = self.wv(G)
        attn_logits = torch.matmul(r_proj, g_proj.transpose(1, 2)) / (d ** 0.5)
        attn_mask = A.bool()
        attn_logits[~attn_mask] = float('-inf')
        attn = F.softmax(attn_logits, dim=-1)
        offset = self.offset_mlp(torch.cat([R.unsqueeze(2).expand(-1, -1, M, -1),
                                            G.unsqueeze(1).expand(-1, N, -1, -1)], dim=-1))
        R_agg = torch.bmm(attn, v_proj)
        R_fused = R + self.gamma * R_agg
        G_attn = F.softmax(torch.matmul(g_proj, g_proj.transpose(1, 2)) / (d ** 0.5), dim=-1)
        G_agg = torch.bmm(G_attn, v_proj)
        G_fused = G + self.gamma * G_agg
        alpha = torch.sigmoid(self.gate_r(torch.cat([R_fused, R_agg], dim=-1)))
        Rout = R + self.gamma * (alpha * F.layer_norm(R_agg, (d,)) + (1 - alpha) * F.layer_norm(G_agg.mean(dim=1, keepdim=True).expand_as(R), (d,)))
        beta = torch.sigmoid(self.gate_g(torch.cat([G_fused, G_agg], dim=-1)))
        Gout = G + self.gamma * (beta * F.layer_norm(G_agg, (d,)) + (1 - beta) * F.layer_norm(R.mean(dim=1, keepdim=True).expand_as(G), (d,)))
        return Rout, Gout
