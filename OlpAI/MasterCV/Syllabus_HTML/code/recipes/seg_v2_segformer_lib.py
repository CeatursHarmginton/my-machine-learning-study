import torch
import torch.nn.functional as F
from torchvision import tv_tensors
from torchvision.transforms import v2
from transformers import SegformerForSemanticSegmentation

# 1. Đồng bộ biến đổi hình học cho cả (Image, Mask) bằng torchvision.transforms.v2 (không cần Albumentations!):
seg_train_tf = v2.Compose([
    v2.RandomResizedCrop(size=(512, 512), scale=(0.75, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.15, contrast=0.15),  # Tự động chỉ biến đổi tv_tensors.Image, giữ nguyên Mask!
    v2.ToDtype(torch.float32, scale=True),
])
# Cách gọi: img_t, mask_t = seg_train_tf(tv_tensors.Image(img), tv_tensors.Mask(mask))

# 2. Khởi tạo SegFormer đa tỷ lệ từ transformers (thay thế hoàn toàn gói ngoài smp):
def build_segformer(num_classes: int = 1):
    return SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2", num_labels=num_classes, ignore_mismatched_sizes=True
    )

def bce_dice_loss(logits: torch.Tensor, mask: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    p, t = torch.sigmoid(logits), mask.float()
    dice = 1.0 - (2.0 * (p * t).sum((2, 3)) + eps) / (p.sum((2, 3)) + t.sum((2, 3)) + eps)
    return F.binary_cross_entropy_with_logits(logits, t) + dice.mean()
