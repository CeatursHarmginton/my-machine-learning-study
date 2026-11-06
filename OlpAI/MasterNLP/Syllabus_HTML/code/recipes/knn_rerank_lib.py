import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Truy xuất Top-K vector Cosine bằng NearestNeighbors của scikit-learn (thay thế hoàn toàn FAISS):
def build_knn_retriever(doc_vecs: np.ndarray, top_k: int = 50) -> NearestNeighbors:
    return NearestNeighbors(n_neighbors=top_k, metric="cosine", n_jobs=-1).fit(doc_vecs)

# 2. Tái xếp hạng chú ý chéo (Cross-Encoder Reranker) bằng trực tiếp AutoModelForSequenceClassification:
@torch.no_grad()
def cross_rerank(query: str, cand_docs: list[str], tok, clf_model) -> np.ndarray:
    pairs = tok([query] * len(cand_docs), cand_docs, padding=True, truncation=True, max_length=512, return_tensors="pt")
    logits = clf_model(**{k: v.to(clf_model.device) for k, v in pairs.items()}).logits
    return (logits[:, -1] if logits.ndim == 2 and logits.size(-1) > 1 else logits.squeeze(-1)).cpu().numpy()
