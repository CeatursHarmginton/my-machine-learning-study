import re
import unicodedata
import numpy as np
from sklearn.covariance import LedoitWolf

ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")

def clean_unicode(text: str) -> str:
    """Chuẩn hóa NFKC + xóa ký tự tàng hình Zero-Width trong 2 dòng bằng thư viện chuẩn Python."""
    return ZERO_WIDTH_RE.sub("", unicodedata.normalize("NFKC", text))

def fit_mahalanobis_ood(in_dist_feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ước lượng tâm và nghịch đảo hiệp phương sai co rút Ledoit-Wolf để bắt mẫu lạ OOD."""
    est = LedoitWolf().fit(in_dist_feats)
    return est.location_, est.precision_
