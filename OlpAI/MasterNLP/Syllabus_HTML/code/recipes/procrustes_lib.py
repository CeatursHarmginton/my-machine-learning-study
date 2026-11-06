import numpy as np
from scipy.linalg import orthogonal_procrustes

def align_embeddings_scipy(corrupted_x: np.ndarray, clean_y: np.ndarray) -> np.ndarray:
    """Giải bài toán sửa chữa không gian nhúng (IOAI 2025 Broken BERT) bằng 2 dòng scipy.linalg."""
    mx, my = corrupted_x.mean(0, keepdims=True), clean_y.mean(0, keepdims=True)
    omega, _ = orthogonal_procrustes(corrupted_x - mx, clean_y - my)
    return (corrupted_x - mx) @ omega + my
