"""Thiết kế Validation NLP chống rò rỉ văn bản gần trùng lặp (Near-Duplicate Group Split)."""
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold


def make_group_folds(
    texts: list[str],
    y: np.ndarray,
    n_splits: int = 5,
    sim_th: float = 0.88,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Gom cụm câu có độ tương đồng char n-gram >= sim_th và chia StratifiedGroupKFold (G_train ∩ G_val = ∅)."""
    mat = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(texts)
    _, groups = connected_components((mat @ mat.T) >= sim_th, directed=False)
    folds = np.zeros(len(texts), dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (tr_idx, val_idx) in enumerate(sgkf.split(texts, y, groups)):
        assert set(groups[tr_idx]).isdisjoint(set(groups[val_idx])), "Rò rỉ nhóm văn bản trùng giữa Train và Val!"
        folds[val_idx] = f
    return folds, groups
