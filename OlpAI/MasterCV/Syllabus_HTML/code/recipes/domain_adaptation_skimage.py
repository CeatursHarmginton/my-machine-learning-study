import numpy as np
import torch
import torch.nn as nn
from skimage.exposure import match_histograms

# 1. Khớp phân phối màu sắc miền đích bằng scikit-image (có sẵn trong OlpAI'25) & Fourier Domain Adaptation:
def match_domain_style_skimage(src_rgb: np.ndarray, ref_rgb: np.ndarray) -> np.ndarray:
    return match_histograms(src_rgb, ref_rgb, channel_axis=-1).astype(np.uint8)

def fourier_domain_adapt(src_rgb: np.ndarray, ref_rgb: np.ndarray, beta: float = 0.05) -> np.ndarray:
    """Hoán đổi biên độ tần số thấp (ánh sáng / cảm biến) miền đích trong 7 dòng FFT NumPy thuần."""
    f_src, f_ref = np.fft.fftshift(np.fft.fft2(src_rgb, axes=(0, 1))), np.fft.fftshift(np.fft.fft2(ref_rgb, axes=(0, 1)))
    amp, pha = np.abs(f_src), np.angle(f_src)
    h, w, _ = src_rgb.shape
    cy, cx, bh, bw = h // 2, w // 2, max(1, int(h * beta)), max(1, int(w * beta))
    amp[cy - bh : cy + bh, cx - bw : cx + bw] = np.abs(f_ref)[cy - bh : cy + bh, cx - bw : cx + bw]
    out = np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * pha)), axes=(0, 1)).real
    return np.clip(out, 0, 255).astype(np.uint8)

# 2. AdaBN (Adaptive BatchNorm): cập nhật lại running_mean / running_var trên lô ảnh miền mới không cần nhãn:
@torch.no_grad()
def adapt_batchnorm_stats(model: nn.Module, test_loader, device: str = "cuda") -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.train()
    for b in test_loader:
        model(b["image"].to(device))
