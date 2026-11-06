import torch
import torch.nn as nn

def ccc_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Concordance Correlation Loss (7 dòng): giữ nguyên phương sai & tương quan cho bài toán chấm điểm / cảm xúc."""
    dp, dt = pred - pred.mean(0), target - target.mean(0)
    cov = (dp * dt).mean(0)
    ccc = (2.0 * cov) / ((dp ** 2).mean(0) + (dt ** 2).mean(0) + (pred.mean(0) - target.mean(0)) ** 2 + eps)
    return (1.0 - ccc).mean()

def asymmetric_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                          g_neg: float = 4.0, g_pos: float = 1.0, clip: float = 0.05) -> torch.Tensor:
    """Asymmetric Loss (8 dòng): triệt tiêu nhiễu từ hàng chục nhãn âm dễ mà không làm suy yếu nhãn dương hiếm."""
    p = torch.sigmoid(logits)
    p_neg = (1.0 - p + clip).clamp(max=1.0)
    log_loss = targets * p.clamp(min=1e-8).log() + (1.0 - targets) * p_neg.clamp(min=1e-8).log()
    pt = targets * p + (1.0 - targets) * p_neg
    w = (1.0 - pt).pow(g_pos * targets + g_neg * (1.0 - targets)).detach()
    return -(w * log_loss).mean()

class FGMAttack:
    """Đối kháng Fast Gradient Method (12 dòng) trên Word Embeddings — tăng 0.005-0.015 điểm ổn định mọi đề NLP."""
    def __init__(self, model: nn.Module, eps: float = 0.5, key: str = "word_embeddings"):
        self.model, self.eps, self.key, self.backup = model, eps, key, {}

    def attack(self) -> None:
        for name, p in self.model.named_parameters():
            if p.requires_grad and self.key in name and p.grad is not None:
                self.backup[name] = p.data.clone()
                p.data.add_(self.eps * p.grad / p.grad.norm().clamp(min=1e-8))

    def restore(self) -> None:
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup.pop(name))
