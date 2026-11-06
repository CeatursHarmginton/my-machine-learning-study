import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    """7x7 Depthwise Conv + Inverted Bottleneck 4x + LayerScale (nhẹ tham số, trường tiếp nhận rộng)."""
    def __init__(self, dim: int, layer_scale: float = 1e-6):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.mlp = nn.Sequential(nn.LayerNorm(dim, eps=1e-6), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp(self.dw(x).permute(0, 2, 3, 1))
        return x + (self.gamma * h).permute(0, 3, 1, 2)

class Res2NetSplitBlock(nn.Module):
    """Tách kênh thành s nhánh nối tầng y_i = K_i(x_i + y_{i-1}) bắt đa kích thước đối tượng trong 1 khối."""
    def __init__(self, dim: int, scales: int = 4):
        super().__init__()
        w = dim // scales
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True))
            for _ in range(scales - 1)
        ])
        self.fuse = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = torch.chunk(x, self.scales, dim=1)
        ys = [xs[0]]
        for i in range(1, self.scales):
            ys.append(self.convs[i - 1](xs[i] if i == 1 else xs[i] + ys[-1]))
        return torch.relu(x + self.fuse(torch.cat(ys, dim=1)))
