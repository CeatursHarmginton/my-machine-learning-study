import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SubCenterArcFace(nn.Module):
    """Deng et al. (ECCV 2020): K tâm phụ mỗi lớp tự gom nhãn nhiễu + ép biên góc m trên mặt cầu."""
    def __init__(self, in_dim: int, num_classes: int, k: int = 3, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.c, self.k, self.s = num_classes, k, s
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th, self.mm = math.cos(math.pi - m), math.sin(math.pi - m) * m
        self.weight = nn.Parameter(torch.empty(num_classes * k, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        z, w = F.normalize(emb.float(), dim=1), F.normalize(self.weight.float(), dim=1)
        cos = F.linear(z, w).view(-1, self.c, self.k).max(dim=2).values
        if target is None:
            return cos * self.s
        sin = (1.0 - cos.pow(2)).clamp_min(1e-7).sqrt()
        phi = torch.where(cos > self.th, cos * self.cos_m - sin * self.sin_m, cos - self.mm)
        oh = F.one_hot(target, self.c).float()
        return self.s * (oh * phi + (1.0 - oh) * cos)

def batch_hard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """Đào mẫu Dương xa nhất & mẫu Âm gần nhất ngay trong batch (chặn 1e-12 trước căn bậc hai!)."""
    z = F.normalize(emb.float(), dim=1)
    dist = (2.0 - 2.0 * (z @ z.T)).clamp_min(1e-12).sqrt()
    same = labels[:, None] == labels[None, :]
    pos = same & (~torch.eye(len(labels), dtype=torch.bool, device=z.device))
    hard_p = dist.masked_fill(~pos, -1e9).max(dim=1).values
    hard_n = dist.masked_fill(same, 1e9).min(dim=1).values
    ok = pos.any(1) & (~same).any(1)
    return F.relu(hard_p[ok] - hard_n[ok] + margin).mean()
