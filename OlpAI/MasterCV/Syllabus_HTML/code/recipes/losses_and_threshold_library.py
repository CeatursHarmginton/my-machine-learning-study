import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from torchvision.ops import sigmoid_focal_loss

# 1. Kháng nhãn nhiễu nhẹ bằng tham số label_smoothing tích hợp sẵn trong PyTorch:
ce_smooth_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# 2. Sigmoid Focal Loss chuẩn của torchvision (tự suy ra số lớp C từ chiều cuối của logits!):
def focal_multiclass_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    y_oh = F.one_hot(targets, num_classes=logits.size(-1)).float()
    return sigmoid_focal_loss(logits, y_oh, alpha=alpha, gamma=gamma, reduction="mean")

# 3. Tối ưu trọng số nhân từng lớp trên OOF cho bất kỳ số lớp C nào bằng Powell (SciPy):
def fit_oof_multipliers_scipy(oof_probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    loss_fn = lambda w: -float(f1_score(y_true, (oof_probs * w[None, :]).argmax(1), average="macro"))
    return minimize(loss_fn, np.ones(oof_probs.shape[1]), method="Powell").x
