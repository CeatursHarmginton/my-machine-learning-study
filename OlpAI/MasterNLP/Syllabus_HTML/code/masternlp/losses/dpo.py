import numpy as np
import torch
import torch.nn.functional as F

def seq_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Tính tổng log P(y | x) trên các token trả lời hợp lệ (bỏ qua vị trí -100) trong 5 dòng."""
    lg, lb = logits[:, :-1, :], labels[:, 1:]
    mask = lb != -100
    tok_lp = F.log_softmax(lg, dim=-1).gather(-1, lb.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (tok_lp * mask.float()).sum(dim=-1)

def dpo_loss(pi_win_lp: torch.Tensor, pi_lose_lp: torch.Tensor,
             ref_win_lp: torch.Tensor, ref_lose_lp: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """Nghiệm đóng Direct Preference Optimization (DPO) trong 3 dòng — không cần cài đặt thư viện trl!"""
    margin = (pi_win_lp - ref_win_lp) - (pi_lose_lp - ref_lose_lp)
    return -F.logsigmoid(beta * margin).mean()

def swap_order_debias(probs_ab: np.ndarray, probs_ba: np.ndarray) -> np.ndarray:
    """Khử thiên kiến vị trí A/B (cột 0: A thắng, cột 1: B thắng, cột 2: Hòa) chỉ bằng 1 dòng hoán vị."""
    return 0.5 * (probs_ab + probs_ba[:, [1, 0, 2]])
