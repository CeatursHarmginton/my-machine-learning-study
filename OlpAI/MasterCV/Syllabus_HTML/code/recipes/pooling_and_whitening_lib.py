import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

class CatAvgMaxNeck(nn.Module):
    """Nối song song AdaptiveAvgPool2d & AdaptiveMaxPool2d bằng torch.nn chuẩn + BNNeck ổn định góc."""
    def __init__(self, in_channels: int, emb_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels * 2, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        avg_f = feat_map.mean(dim=(-2, -1))
        max_f = feat_map.amax(dim=(-2, -1))
        return self.proj(torch.cat([avg_f, max_f], dim=1))

def pca_whiten_embeddings(train_embs: np.ndarray, test_embs: np.ndarray, out_dim: int = 256) -> np.ndarray:
    """Nén chiều & khử tương quan chéo (PCA Whitening) bằng scikit-learn chuẩn."""
    pca = PCA(n_components=out_dim, whiten=True, random_state=42).fit(normalize(train_embs))
    return normalize(pca.transform(normalize(test_embs)))
