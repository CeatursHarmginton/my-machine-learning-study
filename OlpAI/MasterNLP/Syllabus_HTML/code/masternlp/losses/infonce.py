import math
from collections import Counter
import torch
import torch.nn.functional as F

class SimpleBM25:
    """Okapi BM25 tự viết trong 17 dòng Python thuần — cực nhanh, dễ nhớ và tùy biến tham số k1, b."""
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.lens = [len(d) for d in docs]
        self.avgdl = sum(self.lens) / max(1, self.n)
        self.tfs = [Counter(d) for d in docs]
        df = Counter(t for tf in self.tfs for t in tf)
        self.idf = {t: math.log(1.0 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query: list[str]) -> list[float]:
        out = [0.0] * self.n
        for t in query:
            if t in self.idf:
                for i, tf in enumerate(self.tfs):
                    f = tf.get(t, 0)
                    if f:
                        out[i] += self.idf[t] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl))
        return out

def infonce_loss(q_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor | None = None, tau: float = 0.05) -> torch.Tensor:
    """Hàm mất mát đối tương phản SimCSE / DPR (7 dòng): trải đều vector lên mặt cầu đơn vị S^(d-1)."""
    q, p = F.normalize(q_emb, dim=-1), F.normalize(pos_emb, dim=-1)
    keys = torch.cat([p, F.normalize(neg_emb, dim=-1)], dim=0) if neg_emb is not None else p
    return F.cross_entropy((q @ keys.T) / tau, torch.arange(q.size(0), device=q.device))
