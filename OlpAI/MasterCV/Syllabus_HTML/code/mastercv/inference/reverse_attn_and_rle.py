import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReverseAttentionRefiner(nn.Module):
    """Cổng chú ý ngược (1 - sigmoid(coarse)) xóa vùng lõi đã biết, ép mạng gọt sắc ranh giới mờ."""
    def __init__(self, feat_channels: int, num_classes: int = 1):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(feat_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, low_level_feat: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(coarse_logits, size=low_level_feat.shape[-2:], mode="bilinear", align_corners=False)
        gate = 1.0 - torch.sigmoid(up.detach())
        return up + self.refine(low_level_feat * gate)

def rle_encode(mask: np.ndarray, column_major: bool = True) -> str:
    """Mã hóa Run-Length Encoding 1-indexed vector hóa hoàn toàn bằng NumPy (không vòng lặp)."""
    px = np.concatenate([[0], mask.flatten(order="F" if column_major else "C").astype(np.uint8), [0]])
    runs = np.where(px[1:] != px[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))

def rle_decode(rle: str, shape: tuple[int, int], column_major: bool = True) -> np.ndarray:
    if not rle or str(rle) == "nan":
        return np.zeros(shape, dtype=np.uint8)
    s = np.asarray(rle.split(), dtype=int)
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, length in zip(s[0::2] - 1, s[1::2]):
        flat[lo : lo + length] = 1
    return flat.reshape(shape, order="F" if column_major else "C")
