import random
import torch
import torch.nn as nn

def pcgrad_step(task_losses: list[torch.Tensor], shared_params: list[nn.Parameter]) -> None:
    """Chiếu trực giao đạo hàm xung đột PCGrad trong 18 dòng — dùng cho mọi mạng Đa tác vụ (Multi-Task)."""
    params = [p for p in shared_params if p.requires_grad]
    grads = []
    for k, loss in enumerate(task_losses):
        gs = torch.autograd.grad(loss, params, retain_graph=(k < len(task_losses) - 1), allow_unused=True)
        grads.append(torch.cat([(g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)) for g, p in zip(gs, params)]))

    proj = [g.clone() for g in grads]
    for i in range(len(grads)):
        for j in random.sample(range(len(grads)), len(grads)):
            dot = torch.dot(proj[i], grads[j])
            if i != j and dot < 0:
                proj[i] = proj[i] - (dot / grads[j].pow(2).sum().clamp(min=1e-12)) * grads[j]

    merged, idx = torch.stack(proj).sum(0), 0
    for p in params:
        n = p.numel()
        p.grad = merged[idx : idx + n].view_as(p).clone()
        idx += n
