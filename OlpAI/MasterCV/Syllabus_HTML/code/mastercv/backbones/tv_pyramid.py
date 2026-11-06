import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

class TorchvisionPyramidBackbone(nn.Module):
    """Trích xuất đa tỷ lệ [C2, C3, C4, C5] chuẩn torchvision — bật return_pyramid=True cho GĐ 3 & 4."""
    STAGE_NODES = {
        "resnet34": ["layer1", "layer2", "layer3", "layer4"],
        "resnet50": ["layer1", "layer2", "layer3", "layer4"],
        "convnext_small": ["features.1", "features.3", "features.5", "features.7"],
        "convnext_base": ["features.1", "features.3", "features.5", "features.7"],
    }

    def __init__(self, arch: str = "convnext_small", return_pyramid: bool = False):
        super().__init__()
        base = getattr(models, arch)(weights="DEFAULT")
        return_nodes = {node: f"c{i + 2}" for i, node in enumerate(self.STAGE_NODES[arch])}
        self.extractor = create_feature_extractor(base, return_nodes=return_nodes)
        self.return_pyramid = return_pyramid

    def forward(self, x: torch.Tensor):
        feats = list(self.extractor(x).values())
        return feats if self.return_pyramid else feats[-1]
