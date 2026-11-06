import numpy as np
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Tải cặp Policy LM và Reference LM (đóng băng) bằng trực tiếp transformers:
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
policy_lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ref_lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").eval().requires_grad_(False)

# 2. Ước lượng điểm sức mạnh Bradley-Terry từ bảng kết quả thắng/thua cặp bằng LogisticRegression chuẩn:
def fit_bradley_terry_scores(n_models: int, pairs: np.ndarray, a_wins: np.ndarray) -> np.ndarray:
    x_mat = np.zeros((len(pairs), n_models))
    x_mat[np.arange(len(pairs)), pairs[:, 0]] = 1.0
    x_mat[np.arange(len(pairs)), pairs[:, 1]] = -1.0
    return LogisticRegression(C=1.0, fit_intercept=False).fit(x_mat, a_wins).coef_[0]
