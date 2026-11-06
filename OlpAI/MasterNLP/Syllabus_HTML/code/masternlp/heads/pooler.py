import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskPooler(nn.Module):
    """Bộ gộp đa năng (MeanMax / Attention + Pha trộn K tầng cuối), tự động che bỏ hoàn toàn vị trí [PAD]."""
    def __init__(self, d_model: int, out_dim: int, mode: str = "attn", last_k: int = 4):
        super().__init__()
        self.mode, self.last_k = mode, last_k
        self.layer_w = nn.Parameter(torch.zeros(last_k))
        self.attn = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
        in_proj = d_model * 2 if mode == "mean_max" else d_model
        self.fc = nn.Linear(in_proj, out_dim)

    def forward(self, enc_out, attention_mask: torch.Tensor) -> torch.Tensor:
        stack = torch.stack(enc_out.hidden_states[-self.last_k:], dim=0)
        h = (F.softmax(self.layer_w, dim=0).view(-1, 1, 1, 1) * stack).sum(dim=0)
        m = attention_mask.unsqueeze(-1).type_as(h)
        if self.mode == "mean_max":
            mean_v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
            max_v = h.masked_fill(m == 0, -1e4).max(1).values
            feat = torch.cat([mean_v, max_v], dim=-1)
        else:
            w = F.softmax(self.attn(h).squeeze(-1).masked_fill(attention_mask == 0, -1e4), dim=-1)
            feat = (w.unsqueeze(-1) * h).sum(1)
        return self.fc(feat)
