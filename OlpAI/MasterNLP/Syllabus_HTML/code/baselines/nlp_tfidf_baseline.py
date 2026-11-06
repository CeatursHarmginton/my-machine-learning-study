"""Bộ hàm Làm Sạch, EDA Ngắn Gọn Đa Năng & Baseline TF-IDF Chia Nhỏ Từng Bước."""
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip()


def quick_text_eda(df: pd.DataFrame, text_col: str = "text", label_col: str | None = "label") -> dict:
    s = df[text_col].fillna("").astype(str).map(clean_text)
    lens = s.map(lambda x: len(x.split()))
    info = {
        "rows": len(df),
        "empty_or_null": int((s == "").sum()),
        "duplicates": int(s.duplicated().sum()),
        "words_min_p50_p95_max": (int(lens.min()), int(np.percentile(lens, 50)), int(np.percentile(lens, 95)), int(lens.max())),
    }
    if label_col and label_col in df.columns:
        info["label_counts"] = df[label_col].value_counts().to_dict()
    print("[NLP EDA]", info)
    return info


def split_stratified_text(df: pd.DataFrame, text_col: str = "text", label_col: str = "label", val_ratio: float = 0.2, seed: int = 42):
    x_all = df[text_col].fillna("").map(clean_text)
    y_all = df[label_col]
    return train_test_split(x_all, y_all, test_size=val_ratio, random_state=seed, stratify=y_all)


def build_tfidf_pipeline(max_features: int = 50000, seed: int = 42) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    ])


def eval_cls_metrics(y_true, y_pred) -> dict[str, float]:
    metrics = {
        "acc": round(100.0 * accuracy_score(y_true, y_pred), 2),
        "macro_f1": round(100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0), 2),
    }
    print("[TF-IDF Val]", metrics)
    return metrics


def run_tfidf_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, text_col: str = "text", label_col: str = "label", id_col: str = "id", out_csv: str = "./outputs/submission.csv") -> pd.DataFrame:
    x_tr, x_va, y_tr, y_va = split_stratified_text(train_df, text_col, label_col)
    pipe = build_tfidf_pipeline()
    pipe.fit(x_tr, y_tr)
    eval_cls_metrics(y_va, pipe.predict(x_va))

    preds = pipe.predict(test_df[text_col].fillna("").map(clean_text))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({id_col: test_df[id_col], label_col: preds})
    sub.to_csv(out_csv, index=False)
    return sub
