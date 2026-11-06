import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

# 1. Bộ trộn MixUp + CutMix chuẩn của torchvision.transforms.v2 (chạy mượt cả batch chẵn lẫn lẻ!):
def build_mix_augmenter(num_classes: int):
    return v2.RandomChoice([
        v2.MixUp(num_classes=num_classes, alpha=0.4),
        v2.CutMix(num_classes=num_classes, alpha=1.0),
    ])

# Lưu ý: nn.CrossEntropyLoss() của PyTorch hiện đại nhận trực tiếp nhãn mềm xác suất (B, C) từ v2.MixUp/CutMix!
soft_ce_criterion = nn.CrossEntropyLoss()

# 2. Suy luận TTA lật ngang 4 dòng bằng PyTorch thuần (không cần thư viện ngoài ttach):
@torch.no_grad()
def predict_hflip_tta(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    p_orig = F.softmax(model(x), dim=-1)
    p_flip = F.softmax(model(torch.flip(x, dims=[-1])), dim=-1)
    return 0.5 * (p_orig + p_flip)
