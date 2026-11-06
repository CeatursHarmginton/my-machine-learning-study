import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor

# PatchCore bằng 100% thư viện có sẵn trong OlpAI'25 (torchvision + scikit-learn, không cần cài anomalib!):
def build_patch_extractor():
    base = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT).eval()
    return create_feature_extractor(base, return_nodes={"layer2": "p2", "layer3": "p3"})

@torch.no_grad()
def extract_patch_vectors(extractor, x: torch.Tensor) -> np.ndarray:
    out = extractor(x)
    p2 = F.avg_pool2d(out["p2"], 3, stride=1, padding=1)
    p3 = F.interpolate(F.avg_pool2d(out["p3"], 3, stride=1, padding=1), size=p2.shape[-2:], mode="bilinear", align_corners=False)
    patches = F.normalize(torch.cat([p2, p3], dim=1).permute(0, 2, 3, 1).flatten(0, 2), dim=-1)
    return patches.cpu().numpy()

jl_proj = SparseRandomProjection(n_components=128, random_state=42)
patch_knn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
