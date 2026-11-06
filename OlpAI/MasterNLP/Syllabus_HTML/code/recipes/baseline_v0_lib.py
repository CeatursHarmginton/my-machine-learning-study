import os, random
import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_group_folds(texts: list[str], y: np.ndarray, n_splits: int = 5, sim_th: float = 0.88) -> np.ndarray:
    """Gom cụm văn bản gần trùng lặp bằng char n-gram TF-IDF + Đồ thị liên thông trước khi chia Fold."""
    mat = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(texts)
    _, groups = connected_components((mat @ mat.T) >= sim_th, directed=False)
    folds = np.zeros(len(texts), dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for f, (_, val_idx) in enumerate(sgkf.split(texts, y, groups)):
        folds[val_idx] = f
    return folds

# 1. Baseline v0.1 trong 5 giây (kiểm chứng luồng dữ liệu & metric):
tfidf = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, max_features=60000)
fast_clf = RidgeClassifier(alpha=1.0)

# 2. Huấn luyện bộ tách từ BPE mới cho ngôn ngữ lạ (IOAI 2024) trực tiếp từ transformers (chỉ 1 dòng!):
# custom_tok = AutoTokenizer.from_pretrained("gpt2").train_new_from_iterator(texts, vocab_size=8000)

# 3. Baseline v0.2 Neural chuẩn mực với Dynamic Padding (nhanh gấp 4 lần đệm tĩnh):
tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
pad_collate = DataCollatorWithPadding(tok, pad_to_multiple_of=8)
model_v0 = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", num_labels=2)
