import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss

def optimize_oof_blend_weights(oof_list: list[np.ndarray], y_true: np.ndarray, loss_metric=log_loss) -> np.ndarray:
    """Tìm trọng số lồi w_m >= 0, sum(w_m) = 1 tối ưu trên OOF cho BẤT KỲ hàm loss_metric nào bằng SLSQP."""
    m = len(oof_list)
    obj = lambda w: float(loss_metric(y_true, np.clip(sum(wi * p for wi, p in zip(w, oof_list)), 1e-15, 1.0 - 1e-15)))
    res = minimize(obj, np.full(m, 1.0 / m), method="SLSQP", bounds=[(0.0, 1.0)] * m,
                   constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    return res.x / np.sum(res.x)

def verify_submission_integrity(sub_df: pd.DataFrame, sample_df: pd.DataFrame) -> None:
    """Chốt chặn 30 giây cuối giờ: ngăn chặn mất điểm 0 vì sai tên cột, lệch thứ tự ID hoặc dính NaN!"""
    assert list(sub_df.columns) == list(sample_df.columns), f"Sai cột: {sub_df.columns}"
    assert len(sub_df) == len(sample_df), f"Sai số dòng: {len(sub_df)} vs {len(sample_df)}"
    assert (sub_df.iloc[:, 0].to_numpy() == sample_df.iloc[:, 0].to_numpy()).all(), "Xáo trộn thứ tự ID!"
    assert not sub_df.isna().any().any(), "Có giá trị NaN/Null!"
