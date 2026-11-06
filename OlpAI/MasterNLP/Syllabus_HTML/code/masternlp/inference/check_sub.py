import numpy as np
import pandas as pd

def check_submission(sub: pd.DataFrame, sample: pd.DataFrame, target_cols: list[str], is_prob: bool = False) -> None:
    """Chốt chặn 15 dòng bắt trọn 6 lỗi gây 0 điểm oan trước khi nộp file cuối cùng trong phòng thi!"""
    assert list(sub.columns) == list(sample.columns) and len(sub) == len(sample), "Lệch cột hoặc số dòng!"
    assert (sub.iloc[:, 0].values == sample.iloc[:, 0].values).all(), "Bị đảo thứ tự cột ID so với file mẫu!"
    for col in target_cols:
        s = sub[col]
        assert not s.isna().any(), f"Cột {col} chứa NaN!"
        if np.issubdtype(s.dtype, np.number):
            v = s.to_numpy(dtype=np.float64)
            assert np.isfinite(v).all() and np.std(v) > 1e-7, f"Cột {col} chứa Inf hoặc sụp đổ thành hằng số!"
            assert (not is_prob) or (v.min() >= 0.0 and v.max() <= 1.0), f"Xác suất {col} vượt ngoài [0, 1]!"
        else:
            assert (s.astype(str).str.strip() == "").mean() < 0.95, f"Cột chuỗi {col} bị rỗng bất thường!"
