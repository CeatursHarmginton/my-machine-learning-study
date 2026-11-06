import torch
import torch.nn as nn
import torch.nn.functional as F

class TopDownFPNHead(nn.Module):
    """Hợp nhất tháp [C2, C3, C4, C5] từ TorchvisionPyramidBackbone(return_pyramid=True) cho M lớp bất kỳ."""
    def __init__(self, in_channels: list[int], num_classes: int = 1, dim: int = 128):
        super().__init__()
        self.lats = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in in_channels])
        self.smooths = nn.ModuleList([nn.Conv2d(dim, dim, 3, padding=1) for _ in in_channels])
        self.head = nn.Conv2d(dim * len(in_channels), num_classes, 1)

    def forward(self, feats: list[torch.Tensor], out_hw: tuple[int, int]) -> torch.Tensor:
        ps = [l(f) for l, f in zip(self.lats, feats)]
        for i in range(len(ps) - 2, -1, -1):
            ps[i] = ps[i] + F.interpolate(ps[i + 1], size=ps[i].shape[-2:], mode="bilinear", align_corners=False)
        ps = [s(p) for s, p in zip(self.smooths, ps)]
        h2, w2 = ps[0].shape[-2:]
        cat = torch.cat([p if p.shape[-2:] == (h2, w2) else F.interpolate(p, (h2, w2), mode="bilinear", align_corners=False) for p in ps], dim=1)
        return F.interpolate(self.head(cat), size=out_hw, mode="bilinear", align_corners=False)

def edge_weighted_bce_iou(logits: torch.Tensor, target: torch.Tensor, gain: float = 5.0, k: int = 15) -> torch.Tensor:
    """Nhân trọng số gấp (1 + gain) dọc sát biên đối tượng mảnh bằng bộ lọc trung bình cục bộ."""
    t = target.float()
    w = 1.0 + gain * (F.avg_pool2d(t, k, stride=1, padding=k // 2) - t).abs()
    wbce = (w * F.binary_cross_entropy_with_logits(logits, t, reduction="none")).sum((2, 3)) / w.sum((2, 3)).clamp_min(1e-6)
    p = torch.sigmoid(logits)
    inter = (w * p * t).sum((2, 3))
    wiou = 1.0 - (inter + 1.0) / ((w * (p + t)).sum((2, 3)) - inter + 1.0)
    return (wbce + wiou).mean()
