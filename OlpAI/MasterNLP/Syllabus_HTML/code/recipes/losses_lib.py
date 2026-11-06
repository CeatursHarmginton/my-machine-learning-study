import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

def make_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """Tự động tính trọng số bù lệch nhãn dương/âm cho mọi bài toán Đa nhãn (BCEWithLogitsLoss)."""
    pos = y_train.sum(axis=0).clip(min=1.0)
    neg = len(y_train) - pos
    return torch.tensor(np.clip(neg / pos, 1.0, 20.0), dtype=torch.float32)

multilabel_loss = nn.BCEWithLogitsLoss()
reg_loss = nn.SmoothL1Loss(beta=0.5)

def mean_spearman(preds: np.ndarray, targets: np.ndarray) -> float:
    vals = [spearmanr(preds[:, j], targets[:, j]).correlation for j in range(preds.shape[1])]
    return float(np.nanmean(vals))
