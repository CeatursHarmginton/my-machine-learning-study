import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

def logit_adjusted_ce(logits: torch.Tensor, targets: torch.Tensor, class_counts: list[int], tau: float = 1.0) -> torch.Tensor:
    """Menon et al. (ICLR 2021): cộng bù tau * log(pi_c) tối ưu Bayes cho phân phối nhãn đuôi dài."""
    priors = torch.tensor(class_counts, dtype=logits.dtype, device=logits.device)
    priors = priors / priors.sum()
    return F.cross_entropy(logits + tau * torch.log(priors.clamp_min(1e-12)), targets)

def symmetric_ce_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0, beta: float = 0.5) -> torch.Tensor:
    """Kết hợp CE + Reverse CE (chặn log(1e-4)) kháng nhãn nhiễu thực địa trên mọi số lớp C."""
    y_oh = F.one_hot(targets, num_classes=logits.size(-1)).float().clamp_min(1e-4)
    probs = F.softmax(logits.float(), dim=-1).clamp(1e-7, 1.0)
    rce = (-torch.sum(probs * torch.log(y_oh), dim=-1)).mean()
    return alpha * F.cross_entropy(logits, targets) + beta * rce

def kd_distill_loss(s_logits: torch.Tensor, t_logits: torch.Tensor, targets: torch.Tensor,
                    temp: float = 4.0, alpha: float = 0.5, smooth: float = 0.05) -> torch.Tensor:
    """Chưng cất tri thức (Knowledge Distillation) + Label Smoothing nén ensemble lớn vào 1 mạng nhẹ chạy kịp giờ thi."""
    ce = F.cross_entropy(s_logits, targets, label_smoothing=smooth)
    kl = F.kl_div(F.log_softmax(s_logits / temp, dim=-1), F.softmax(t_logits.detach() / temp, dim=-1), reduction="batchmean") * (temp ** 2)
    return (1.0 - alpha) * ce + alpha * kl

def tune_oof_multipliers(oof_probs: np.ndarray, y_true: np.ndarray, n_rounds: int = 3) -> np.ndarray:
    """Coordinate Ascent 12 dòng tối ưu trực tiếp Macro-F1 mà không cần đạo hàm."""
    mult = np.ones(oof_probs.shape[1], dtype=np.float64)
    best = f1_score(y_true, oof_probs.argmax(1), average="macro")
    for _ in range(n_rounds):
        for c in range(len(mult)):
            for m in np.linspace(0.4, 2.4, 50):
                cand = mult.copy()
                cand[c] = m
                sc = f1_score(y_true, (oof_probs * cand[None, :]).argmax(1), average="macro")
                if sc > best:
                    best, mult[c] = sc, m
    return mult
