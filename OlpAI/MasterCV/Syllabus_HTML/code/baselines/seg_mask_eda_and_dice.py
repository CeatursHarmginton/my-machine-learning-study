"""EDA Mặt Nạ Phân Đoạn Ngắn Gọn & Mạng MiniUNet 24 Dòng Tự Viết Bằng PyTorch Thuần."""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def quick_mask_eda(mask_paths: list[str | Path], sample_k: int = 200) -> dict:
    """Đo tỷ lệ diện tích vật thể tiền cảnh (foreground %) và đếm số mặt nạ rỗng hoàn toàn."""
    fg_ratios, empty_cnt = [], 0
    for p in list(mask_paths)[:sample_k]:
        m = np.asarray(Image.open(p).convert("L")) > 127
        ratio = float(m.mean()) * 100.0
        fg_ratios.append(ratio)
        empty_cnt += int(ratio == 0.0)
    info = {
        "sampled_masks": len(fg_ratios),
        "empty_masks": empty_cnt,
        "fg_pct_min_med_max": (round(float(np.min(fg_ratios)), 2), round(float(np.median(fg_ratios)), 2), round(float(np.max(fg_ratios)), 2)) if fg_ratios else (),
    }
    print("[Mask EDA]", info)
    return info


def _double_conv(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )


class MiniUNet(nn.Module):
    """Kiến trúc U-Net 3 tầng thu phóng đối xứng kèm Skip-Connections tự viết chỉ trong 20 dòng."""
    def __init__(self, in_ch: int = 3, num_classes: int = 1, base: int = 32):
        super().__init__()
        self.enc1, self.enc2 = _double_conv(in_ch, base), _double_conv(base, base * 2)
        self.bot = _double_conv(base * 2, base * 4)
        self.dec2, self.dec1 = _double_conv(base * 4 + base * 2, base * 2), _double_conv(base * 2 + base, base)
        self.head = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bot(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))
        return self.head(d1)
