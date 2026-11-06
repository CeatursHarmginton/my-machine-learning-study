import torch
import torch.nn as nn
import torch.nn.functional as F

def helmholtz_energy_ood_score(logits: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """Liu et al. (NeurIPS 2020): Điểm năng lượng tự do E(x) càng lớn -> xác suất là vật thể lạ OOD càng cao."""
    return -temp * torch.logsumexp(logits.float() / temp, dim=-1)

class EnergyBoundedMarginLoss(nn.Module):
    """Ép năng lượng mẫu quen xuống dưới m_in (-25) và đẩy mẫu nhiễu ngoại lai lên trên m_out (-7)."""
    def __init__(self, m_in: float = -25.0, m_out: float = -7.0, weight: float = 0.1):
        super().__init__()
        self.m_in, self.m_out, self.weight = m_in, m_out, weight

    def forward(self, logits_in: torch.Tensor, y_in: torch.Tensor, logits_out: torch.Tensor) -> torch.Tensor:
        e_in, e_out = helmholtz_energy_ood_score(logits_in), helmholtz_energy_ood_score(logits_out)
        reg = F.relu(e_in - self.m_in).pow(2).mean() + F.relu(self.m_out - e_out).pow(2).mean()
        return F.cross_entropy(logits_in, y_in) + self.weight * reg
