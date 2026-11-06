import numpy as np
import torch
import torch.nn.functional as F

def integral_density_map(img_hw: tuple[int, int], points_xy: np.ndarray, stride: int = 4, sigma: float = 4.0) -> np.ndarray:
    """Chuẩn hóa tích phân = 1.0 trên TỪNG hạt nhân riêng biệt — không mất mát cá thể nằm sát mép ảnh."""
    ho, wo = img_hw[0] // stride, img_hw[1] // stride
    den = np.zeros((ho, wo), dtype=np.float32)
    rad = int(np.ceil(3.0 * sigma))
    for x, y in points_xy:
        cx, cy = np.clip(int(round(x / stride)), 0, wo - 1), np.clip(int(round(y / stride)), 0, ho - 1)
        x1, x2, y1, y2 = max(0, cx - rad), min(wo, cx + rad + 1), max(0, cy - rad), min(ho, cy + rad + 1)
        yy, xx = np.ogrid[y1 - cy : y2 - cy, x1 - cx : x2 - cx]
        k = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
        if k.sum() > 0:
            den[y1:y2, x1:x2] += k / k.sum()
    return den

def multiscale_count_loss(pred_scaled: torch.Tensor, gt_den: torch.Tensor, scale: float = 100.0, w_cnt: float = 0.05) -> torch.Tensor:
    """MSE mức điểm ảnh (nhân scale=100 tránh underflow) + MAE tổng tích phân trên lưới 1x1, 2x2, 4x4."""
    mse = F.mse_loss(pred_scaled, gt_den * scale)
    pred = pred_scaled / scale
    area = pred.size(-2) * pred.size(-1)
    mae = sum(
        F.l1_loss(F.adaptive_avg_pool2d(pred, g) * (area / (g * g)), F.adaptive_avg_pool2d(gt_den, g) * (area / (g * g)))
        for g in (1, 2, 4)
    )
    return mse + w_cnt * mae
