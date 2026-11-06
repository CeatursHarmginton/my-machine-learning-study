import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """Thay thế hoàn toàn thư viện peft chỉ bằng 12 dòng PyTorch thuần: W' = W_0 + (alpha / r) * B @ A."""
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_a.to(x.dtype).T @ self.lora_b.to(x.dtype).T) * self.scale

def inject_lora(model: nn.Module, targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"), r: int = 8) -> nn.Module:
    """Hàm đệ quy 8 dòng gắn LoRA vào BẤT KỲ mô hình nào trong transformers hoặc torchvision!"""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and any(t in name for t in targets):
            setattr(model, name, LoRALinear(child, r=r, alpha=float(2 * r)))
        else:
            inject_lora(child, targets=targets, r=r)
    return model

def apply_rope(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Mã hóa vị trí quay RoPE trong 8 dòng: quay từng cặp chiều chẵn-lẻ góc m * theta_k."""
    half = x.size(-1) // 2
    inv_f = 1.0 / (base ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    ang = pos.float().unsqueeze(-1) * inv_f
    c, s = ang.cos().unsqueeze(1).type_as(x), ang.sin().unsqueeze(1).type_as(x)
    xe, xo = x[..., 0::2], x[..., 1::2]
    return torch.stack([xe * c - xo * s, xe * s + xo * c], dim=-1).flatten(-2)

def mask_prompt_labels(input_ids: torch.Tensor, prompt_lens: list[int]) -> torch.Tensor:
    """Gán -100 cho toàn bộ phần Prompt để LLM chỉ học sinh câu trả lời."""
    labels = input_ids.clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
    return labels
