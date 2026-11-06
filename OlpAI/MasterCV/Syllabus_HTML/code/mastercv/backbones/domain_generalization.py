import torch
import torch.nn as nn
import torch.nn.functional as F

class MixStyle(nn.Module):
    """Zhou et al. (ICLR 2021): Nội suy thống kê (mu, sigma) ở tầng nông lúc Train để kháng lệch màu thiết bị."""
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p, self.eps = p, eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or torch.rand(1).item() > self.p:
            return x
        mu = x.mean(dim=(2, 3), keepdim=True)
        sig = (x.var(dim=(2, 3), unbiased=False, keepdim=True) + self.eps).sqrt()
        lam = self.beta.sample((x.size(0), 1, 1, 1)).to(x.device, x.dtype)
        perm = torch.randperm(x.size(0), device=x.device)
        return ((x - mu) / sig) * (lam * sig.detach() + (1.0 - lam) * sig.detach()[perm]) + (lam * mu.detach() + (1.0 - lam) * mu.detach()[perm])

class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return -ctx.lambd * grad_out, None

def deep_coral_loss(f_src: torch.Tensor, f_tgt: torch.Tensor) -> torch.Tensor:
    """Căn chỉnh ma trận hiệp phương sai bậc hai giữa miền nguồn và miền đích trong 4 dòng."""
    cs, ct = f_src.float() - f_src.float().mean(0), f_tgt.float() - f_tgt.float().mean(0)
    cov_s, cov_t = (cs.T @ cs) / max(1, len(cs) - 1), (ct.T @ ct) / max(1, len(ct) - 1)
    return ((cov_s - cov_t) ** 2).sum() / (4.0 * (f_src.size(1) ** 2))

def tent_adapt_step(model: nn.Module, opt: torch.optim.Optimizer, x_test: torch.Tensor) -> torch.Tensor:
    opt.zero_grad(set_to_none=True)
    logits = model(x_test)
    (-(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(-1).mean()).backward()
    opt.step()
    return logits.detach()
