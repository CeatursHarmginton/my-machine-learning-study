import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MultiTaskVisionModel(nn.Module):
    """Kiến trúc đa tác vụ tổng quát trên nền torchvision: truyền bất kỳ dict task_dims nào (VD: Bengali, Thuộc tính)."""
    def __init__(self, task_dims: dict[str, int], arch: str = "convnext_small"):
        super().__init__()
        base = getattr(models, arch)(weights="DEFAULT")
        in_dim = base.classifier[-1].in_features if hasattr(base, "classifier") else base.fc.in_features
        if hasattr(base, "classifier"):
            base.classifier[-1] = nn.Identity()
        else:
            base.fc = nn.Identity()
        self.backbone = base
        self.heads = nn.ModuleDict({name: nn.Linear(in_dim, out_c) for name, out_c in task_dims.items()})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return {name: head(feat) for name, head in self.heads.items()}

# Ví dụ dùng cho Bengali.AI (task_dims={"root": 168, "vowel": 11, "cons": 7}, weights={"root": 0.5, ...}):
def weighted_mtl_loss(outs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    return sum(w * F.cross_entropy(outs[k], batch[k]) for k, w in weights.items())
