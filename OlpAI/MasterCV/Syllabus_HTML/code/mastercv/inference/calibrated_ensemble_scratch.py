import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperatureLogOddsBlender(nn.Module):
    """Học đồng thời nhiệt độ hiệu chuẩn T_m và trọng số pha trộn alpha_m giữa M họ mô hình trên tập OOF."""
    def __init__(self, num_models: int):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(num_models))
        self.raw_w = nn.Parameter(torch.zeros(num_models))

    def forward(self, logits_stack: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.raw_w, dim=0).view(-1, 1, 1)
        t = self.log_t.exp().view(-1, 1, 1)
        return (w * F.log_softmax(logits_stack / t, dim=-1)).sum(dim=0)

    def fit_oof(self, oof_logits: list[np.ndarray], y_true: np.ndarray, steps: int = 150) -> "TemperatureLogOddsBlender":
        x = torch.tensor(np.stack(oof_logits, axis=0), dtype=torch.float32)
        y = torch.tensor(y_true, dtype=torch.long)
        opt = torch.optim.LBFGS(self.parameters(), lr=0.25, max_iter=steps)
        opt.step(lambda: (opt.zero_grad(), loss := F.nll_loss(self(x), y), loss.backward(), loss)[-1])
        return self

def rank_average_ensemble(preds_list: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Hòa trộn thứ hạng bách phân vị [0, 1] bằng argsort kép — bất biến với mọi lệch thang xác suất (AUC / AP)."""
    w = np.full(len(preds_list), 1.0 / len(preds_list)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = np.zeros_like(preds_list[0], dtype=np.float64)
    for wi, p in zip(w, preds_list):
        ranks = np.argsort(np.argsort(p.reshape(len(p), -1), axis=0), axis=0) / max(1, len(p) - 1)
        out += wi * ranks.reshape(p.shape)
    return out
