import torch
import torch.nn.functional as F

def distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                 labels: torch.Tensor, temp: float = 2.5, alpha: float = 0.7) -> torch.Tensor:
    """Chưng cất tri thức 6 dòng: nén dàn Ensemble đa mô hình xuống 1 mô hình suy luận nhanh."""
    kl = F.kl_div(F.log_softmax(student_logits / temp, dim=-1), F.softmax(teacher_logits / temp, dim=-1), reduction="batchmean")
    return alpha * (temp ** 2) * kl + (1.0 - alpha) * F.cross_entropy(student_logits, labels)
