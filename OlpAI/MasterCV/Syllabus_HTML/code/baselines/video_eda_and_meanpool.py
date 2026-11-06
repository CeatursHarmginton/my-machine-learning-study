"""Bộ hàm EDA Video ngắn gọn & Mô hình Gộp Trung Bình Khung Hình tối giản."""
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def get_video_meta(video_path: str | Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return n_frames, fps


def quick_video_eda(video_paths: list[str | Path], sample_k: int = 100) -> dict:
    counts, broken = [], 0
    for p in list(video_paths)[:sample_k]:
        n_f, _ = get_video_meta(p)
        if n_f > 0:
            counts.append(n_f)
        else:
            broken += 1
    info = {
        "total_videos": len(video_paths),
        "broken_in_sample": broken,
        "frames_min_med_p95": (int(np.min(counts)), int(np.median(counts)), int(np.percentile(counts, 95))) if counts else (),
    }
    print("[Video EDA]", info)
    return info


def build_2d_trunk(pretrained: bool = False) -> nn.Sequential:
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    return nn.Sequential(*list(resnet.children())[:-1])


class FrameMeanBaseline(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        self.backbone = build_2d_trunk(pretrained=pretrained)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        feat = self.backbone(x.view(b * t, c, h, w)).view(b, t, 512)
        return self.fc(feat.mean(dim=1))
