import random
import torch
import torch.nn as nn

class UncertaintyMTLLoss(nn.Module):
    """Kendall et al. (CVPR 2018): Tự động cân bằng các nhánh phân loại & hồi quy qua phương sai đồng phương."""
    def __init__(self, is_regression: list[bool]):
        super().__init__()
        self.is_reg = is_regression
        self.log_vars = nn.Parameter(torch.zeros(len(is_regression)))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        return sum((0.5 if reg else 1.0) * torch.exp(-lv) * l + 0.5 * lv
                   for l, lv, reg in zip(losses, self.log_vars, self.is_reg))

def pcgrad_step(losses: list[torch.Tensor], shared_params: list[nn.Parameter]) -> None:
    """Chiếu trực giao triệt tiêu xung đột đạo hàm (dot < 0) giữa K tác vụ trên Backbone dùng chung."""
    grads = []
    for k, lk in enumerate(losses):
        gs = torch.autograd.grad(lk, shared_params, retain_graph=(k < len(losses) - 1), allow_unused=True)
        grads.append(torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, shared_params)]))
    proj = [g.clone() for g in grads]
    for i in range(len(losses)):
        for j in random.sample(range(len(losses)), len(losses)):
            dot = torch.dot(proj[i], grads[j])
            if i != j and dot < 0:
                proj[i] = proj[i] - (dot / (grads[j].norm().pow(2) + 1e-8)) * grads[j]
    merged, off = torch.stack(proj).sum(0), 0
    for p in shared_params:
        p.grad = merged[off : off + p.numel()].view_as(p).clone()
        off += p.numel()

def enforce_hierarchy_probs(p_coarse: torch.Tensor, p_fine: torch.Tensor, fine_to_coarse: torch.Tensor) -> torch.Tensor:
    joint = p_fine * p_coarse[:, fine_to_coarse]
    return joint / joint.sum(dim=-1, keepdim=True).clamp_min(1e-8)
