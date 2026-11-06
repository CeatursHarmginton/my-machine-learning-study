import numpy as np

def char_span_ensemble(text_len: int, model_preds: list[tuple[np.ndarray, list[tuple[int, int]], float]],
                       th: float = 0.5) -> list[tuple[int, int]]:
    """Hợp nhất xác suất cấp độ KÝ TỰ (20 dòng) giữa mọi kiến trúc khác bộ tách từ (DeBERTa + RoBERTa + PhoBERT)."""
    p_sum = np.zeros(text_len, dtype=np.float64)
    w_sum = np.zeros(text_len, dtype=np.float64)
    for probs, offsets, weight in model_preds:
        for p, (cs, ce) in zip(probs, offsets):
            if ce > cs:
                p_sum[max(0, cs) : min(text_len, ce)] += float(p) * weight
                w_sum[max(0, cs) : min(text_len, ce)] += weight

    mask = (p_sum / np.maximum(w_sum, 1e-8)) >= th
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))
