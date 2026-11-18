import os, random
import datasets, evaluate
import numpy as np
import pandas as pd
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

# 2. Tận dụng bộ đôi thư viện chuẩn datasets + evaluate sẵn có trong OlpAI'25:
def build_hf_dataset_and_metric(df: pd.DataFrame, tok, text_col: str = "text", max_len: int = 256):
    ds = datasets.Dataset.from_pandas(df).map(lambda b: tok(b[text_col], truncation=True, max_length=max_len), batched=True)
    f1_metric = evaluate.load("f1")
    return ds, f1_metric

# 3. Baseline v0.2 Neural chuẩn mực với Dynamic Padding (nhanh gấp 4 lần đệm tĩnh):
def build_deberta_v0(ckpt: str = "microsoft/deberta-v3-small", num_labels: int = 2):
    tok = AutoTokenizer.from_pretrained(ckpt)
    pad_collate = DataCollatorWithPadding(tok, pad_to_multiple_of=8)
    model_v0 = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=num_labels)
    return tok, pad_collate, model_v0
