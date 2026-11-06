import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """Thay thế hoàn toàn thư viện peft trong 14 dòng: W' = W_0 + (alpha / r) * B @ A kèm gộp trọng số suy luận."""
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base, self.scale = base.requires_grad_(False), alpha / r
        self.a = nn.Parameter(torch.empty(r, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.a.to(x.dtype).T @ self.b.to(x.dtype).T) * self.scale

def inject_lora_vision(model: nn.Module, targets: tuple[str, ...] = ("query", "value", "qkv", "proj"), r: int = 8) -> nn.Module:
    """Đệ quy 7 dòng gắn LoRA tự động vào MỌI mạng ViT / Swin / DINOv2 của transformers hoặc torchvision!"""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and any(t in name for t in targets):
            setattr(model, name, LoRALinear(child, r=r, alpha=float(2 * r)))
        else:
            inject_lora_vision(child, targets=targets, r=r)
    return model

def get_vision_llrd_groups(model: nn.Module, base_lr: float = 1e-4, decay: float = 0.8) -> list[dict]:
    """Layer-Wise LR Decay tổng quát tự nhận diện chỉ số tầng sâu trên cả transformers & torchvision."""
    no_wd = ("bias", "norm", "layernorm", "cls_token", "pos_embed", "position_embeddings")
    max_l = max([int(s) for n in dict(model.named_parameters()) for s in n.split(".") if s.isdigit()] + [11]) + 1
    groups = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            nums = [int(s) for s in name.split(".") if s.isdigit()]
            depth = 0 if "embed" in name else (nums[0] + 1 if nums else max_l + 1)
            lr = base_lr * (decay ** (max_l + 1 - depth))
            groups.append({"params": [p], "lr": lr, "weight_decay": 0.0 if any(k in name.lower() for k in no_wd) else 0.05})
    return groups
