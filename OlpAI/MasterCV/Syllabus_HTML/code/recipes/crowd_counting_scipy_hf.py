import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from transformers import SegformerForSemanticSegmentation

# 1. Tạo nhanh bản đồ mật độ bằng bộ lọc Gaussian của SciPy kèm chuẩn hóa bảo toàn tổng số lượng:
def fast_scipy_density_map(img_hw: tuple[int, int], points_xy: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    imp = np.zeros(img_hw, dtype=np.float32)
    for x, y in points_xy:
        imp[np.clip(int(round(y)), 0, img_hw[0] - 1), np.clip(int(round(x)), 0, img_hw[1] - 1)] += 1.0
    den = gaussian_filter(imp, sigma=sigma, mode="constant", cval=0.0)
    return den * (len(points_xy) / den.sum()) if len(points_xy) > 0 and den.sum() > 1e-8 else den

# 2. Đầu hồi quy mật độ không âm trên nền SegFormer chuẩn của transformers (không cần cài smp!):
class SegformerDensityRegressor(nn.Module):
    def __init__(self, ckpt: str = "nvidia/mit-b0"):
        super().__init__()
        self.net = SegformerForSemanticSegmentation.from_pretrained(ckpt, num_labels=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(pixel_values=x).logits)
