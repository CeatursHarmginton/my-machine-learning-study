import torch
import torch.nn as nn
import torch.nn.functional as F

class BiaffineSpanHead(nn.Module):
    """Ma trận đoạn Biaffine (i, j) trong 22 dòng — bắt trọn mọi thực thể lồng nhau với độ dài <= max_span."""
    def __init__(self, d_model: int, n_classes: int, proj: int = 192, max_span: int = 32):
        super().__init__()
        self.max_span = max_span
        self.mlp_s = nn.Sequential(nn.Linear(d_model, proj), nn.GELU())
        self.mlp_e = nn.Sequential(nn.Linear(d_model, proj), nn.GELU())
        self.u_mat = nn.Parameter(torch.randn(n_classes, proj, proj) * 0.02)
        self.w_lin = nn.Linear(proj * 2, n_classes)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hs, he, seq_len = self.mlp_s(h), self.mlp_e(h), h.size(1)
        pair = torch.cat([hs.unsqueeze(2).expand(-1, -1, seq_len, -1), he.unsqueeze(1).expand(-1, seq_len, -1, -1)], -1)
        logits = torch.einsum("bid,cdk,bjk->bijc", hs, self.u_mat, he) + self.w_lin(pair)
        idx = torch.arange(seq_len, device=h.device)
        dist = idx[None, :] - idx[:, None]
        valid = (dist >= 0) & (dist < self.max_span) & mask.bool()[:, :, None] & mask.bool()[:, None, :]
        return logits, valid

def biaffine_span_loss(logits: torch.Tensor, gold_grid: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[valid], gold_grid[valid])
