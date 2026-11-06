import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata
from xgboost import XGBClassifier

# 1. Pha trộn Log-Odds (tối ưu cho Log-Loss) & Rank Percentile (tối ưu cho ROC-AUC / Spearman):
def blend_logit(probs: list[np.ndarray], weights: list[float]) -> np.ndarray:
    return expit(sum(w * logit(np.clip(p, 1e-6, 1 - 1e-6)) for p, w in zip(probs, weights)))

def blend_rank(preds: list[np.ndarray], weights: list[float]) -> np.ndarray:
    return sum(w * (rankdata(p) / len(p)) for p, w in zip(preds, weights))

# 2. Meta-Stacker bằng XGBoost (có sẵn trong OlpAI'25) kết hợp OOF của TF-IDF + DeBERTa + XLM-R + thống kê độ dài:
meta_stacker = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=42)
