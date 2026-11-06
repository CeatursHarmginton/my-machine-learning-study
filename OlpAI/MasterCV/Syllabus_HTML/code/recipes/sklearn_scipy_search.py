import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

def search_topk_sklearn(queries: np.ndarray, gallery: np.ndarray, topk: int = 50):
    """Truy xuất Top-K Cosine chuẩn xác bằng scikit-learn (có sẵn 100% trong OlpAI'25, không cần FAISS)."""
    qn, gn = normalize(queries.astype(np.float32)), normalize(gallery.astype(np.float32))
    nn_index = NearestNeighbors(n_neighbors=min(topk, len(gn)), metric="cosine", n_jobs=-1).fit(gn)
    dists, indices = nn_index.kneighbors(qn)
    return 1.0 - dists, indices

def pairwise_cosine_scipy(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Tính nhanh ma trận độ tương đồng Cosine bằng SciPy C-backend cho tập vừa và nhỏ."""
    return 1.0 - cdist(queries, gallery, metric="cosine")
