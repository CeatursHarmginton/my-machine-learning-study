import torch
import torch.nn as nn
import torch.nn.functional as F

class GeMPooling(nn.Module):
    """Generalized Mean Pooling học số mũ p: tự động hội tụ vào vùng vật thể nổi bật, lờ đi nền nhiễu."""
    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.float().clamp(1.0, 10.0)
        return x.float().clamp_min(self.eps).pow(p).mean(dim=(-2, -1)).pow(1.0 / p).to(x.dtype)

class DifferentiableNetVLAD(nn.Module):
    """Gom phần dư mềm tới K từ điển trực quan + Chuẩn hóa L2 kép (Intra-Norm & Global Norm)."""
    def __init__(self, dim: int, num_clusters: int = 16, out_dim: int = 512):
        super().__init__()
        self.k = num_clusters
        self.assign = nn.Conv2d(dim, num_clusters, kernel_size=1)
        self.centroids = nn.Parameter(torch.randn(num_clusters, dim) * 0.05)
        self.proj = nn.Linear(num_clusters * dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        xn = F.normalize(x.float(), dim=1)
        a = F.softmax(self.assign(xn).view(b, self.k, -1), dim=1)
        vlad = torch.bmm(a, xn.view(b, c, -1).transpose(1, 2)) - a.sum(-1, keepdim=True) * self.centroids[None]
        vlad = F.normalize(F.normalize(vlad, dim=2).flatten(1), dim=1)
        return F.normalize(self.proj(vlad), dim=1)
