from collections import defaultdict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler

class PKSampler(Sampler):
    """Bộ lấy mẫu chuẩn cho Metric Learning: mỗi batch chọn P nhãn, mỗi nhãn lấy K ảnh."""
    def __init__(self, labels: list[int], p: int = 16, k: int = 4):
        self.p, self.k = p, k
        self.by_cls = defaultdict(list)
        for idx, y in enumerate(labels):
            self.by_cls[int(y)].append(idx)
        self.classes = list(self.by_cls.keys())

    def __iter__(self):
        random.shuffle(self.classes)
        for i in range(0, len(self.classes) - self.p + 1, self.p):
            for c in self.classes[i : i + self.p]:
                pts = self.by_cls[c]
                yield from (random.sample(pts, self.k) if len(pts) >= self.k else random.choices(pts, k=self.k))

    def __len__(self) -> int:
        return (len(self.classes) // self.p) * self.p * self.k

# Hàm mất mát khoảng cách Cosine tích hợp sẵn trong torch.nn (0 phụ thuộc thư viện ngoài!):
cosine_triplet_fn = nn.TripletMarginWithDistanceLoss(
    distance_function=lambda a, b: 1.0 - F.cosine_similarity(a, b), margin=0.3
)
