import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score

def tune_thresholds(oof_probs: np.ndarray, y_true: np.ndarray, metric_fn=f1_score) -> np.ndarray:
    """Tối ưu ngưỡng quyết định tổng quát cho mọi độ đo rời rạc (F1, F-beta, MCC) bằng Powell của SciPy."""
    if oof_probs.ndim == 1:
        oof_probs, y_true = oof_probs[:, None], y_true[:, None]
    ths = np.full(oof_probs.shape[1], 0.5)
    for j in range(oof_probs.shape[1]):
        loss_fn = lambda t: -float(metric_fn(y_true[:, j], (oof_probs[:, j] >= np.clip(t[0], 0.05, 0.95)).astype(int)))
        ths[j] = np.clip(minimize(loss_fn, x0=[0.5], method="Powell").x[0], 0.05, 0.95)
    return ths
