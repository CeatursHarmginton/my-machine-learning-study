"""Thiết kế Validation chống rò rỉ dữ liệu (Anti-Data Leakage): Stratified, Group, Domain & Perceptual Hash."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold


def compute_ahash_groups(img_paths: list[str | Path], hash_size: int = 8) -> np.ndarray:
    """Gom các ảnh có cùng mã băm thị giác Average Hash (8x8) vào chung một nhóm để chặn rò rỉ ảnh trùng."""
    hashes = []
    for p in img_paths:
        with Image.open(p) as im:
            arr = np.asarray(im.convert("L").resize((hash_size, hash_size)), dtype=np.float32)
            hashes.append(int("".join("1" if b else "0" for b in (arr >= arr.mean()).ravel()), 2))
    _, group_ids = np.unique(hashes, return_inverse=True)
    return group_ids


def make_cv_folds(
    df: pd.DataFrame,
    y_col: str,
    group_col: str | None = None,
    domain_col: str | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tự động chọn chế độ chia Validation chuẩn xác và kiểm chứng điều kiện G_train ∩ G_val = ∅."""
    y = df[y_col].to_numpy()
    if domain_col is not None:
        g = df[domain_col].to_numpy()
        splits = list(LeaveOneGroupOut().split(df, y, g))
    elif group_col is not None:
        g = df[group_col].to_numpy()
        splits = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(df, y, g))
    else:
        g = None
        splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(df, y))

    if g is not None:
        for tr_idx, va_idx in splits:
            assert set(g[tr_idx]).isdisjoint(set(g[va_idx])), "Phát hiện rò rỉ nhóm giữa Train và Val!"
    return splits
