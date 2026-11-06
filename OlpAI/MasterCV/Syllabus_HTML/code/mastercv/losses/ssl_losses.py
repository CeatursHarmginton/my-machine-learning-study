import torch
import torch.nn as nn
import torch.nn.functional as F

def simclr_infonce_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """Đối tương phản 2 góc nhìn biến đổi của cùng lô ảnh (B, D) trong 7 dòng."""
    b = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0).float(), dim=1)
    sim = (z @ z.T) / temp
    sim.masked_fill_(torch.eye(2 * b, dtype=torch.bool, device=z.device), -1e9)
    targets = torch.cat([torch.arange(b, 2 * b, device=z.device), torch.arange(0, b, device=z.device)])
    return F.cross_entropy(sim, targets)

class DINOCenteringLoss(nn.Module):
    """Chống sụp đổ về vector hằng không cần mẫu âm nhờ Centering + Sharpening (tau_t < tau_s)."""
    def __init__(self, out_dim: int, tau_s: float = 0.1, tau_t: float = 0.04, mom: float = 0.9):
        super().__init__()
        self.tau_s, self.tau_t, self.mom = tau_s, tau_t, mom
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, s_out: torch.Tensor, t_out: torch.Tensor) -> torch.Tensor:
        pt = F.softmax((t_out.detach() - self.center) / self.tau_t, dim=-1)
        loss = -(pt * F.log_softmax(s_out / self.tau_s, dim=-1)).sum(-1).mean()
        with torch.no_grad():
            self.center.mul_(self.mom).add_((1.0 - self.mom) * t_out.mean(0, keepdim=True))
        return loss
