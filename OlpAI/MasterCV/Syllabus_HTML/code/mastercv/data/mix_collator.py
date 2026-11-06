import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def mixup_cutmix_batch(x: torch.Tensor, y: torch.Tensor, num_classes: int, alpha: float = 1.0, smooth: float = 0.05):
    """Trộn MixUp / CutMix tổng quát tính chính xác tỷ lệ diện tích hộp cắt lam_exact."""
    b, _, h, w = x.shape
    y_soft = F.one_hot(y, num_classes).float() * (1.0 - smooth) + smooth / num_classes
    if b < 2:
        return x, y_soft
    perm = torch.randperm(b, device=x.device)
    lam = float(np.random.beta(alpha, alpha))
    if np.random.rand() < 0.5:
        return lam * x + (1.0 - lam) * x[perm], lam * y_soft + (1.0 - lam) * y_soft[perm]
    rx, ry = np.random.randint(w), np.random.randint(h)
    rw, rh = int(w * np.sqrt(1.0 - lam)), int(h * np.sqrt(1.0 - lam))
    x1, x2 = max(0, rx - rw // 2), min(w, rx + rw // 2)
    y1, y2 = max(0, ry - rh // 2), min(h, ry + rh // 2)
    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam_exact = 1.0 - ((x2 - x1) * (y2 - y1)) / float(h * w)
    return x_mix, lam_exact * y_soft + (1.0 - lam_exact) * y_soft[perm]

@torch.no_grad()
def multiscale_tta_predict(model: nn.Module, x: torch.Tensor, scales: tuple = (1.0, 1.15), hflip: bool = True) -> torch.Tensor:
    model.eval()
    h, w = x.shape[-2:]
    views = []
    for s in scales:
        xs = x if s == 1.0 else F.interpolate(x, size=(int(h * s), int(w * s)), mode="bilinear", align_corners=False)
        views.extend([xs, torch.flip(xs, [-1])] if hflip else [xs])
    return torch.stack([F.softmax(model(v), dim=-1) for v in views]).mean(0)
