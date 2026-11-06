import torch
import torch.nn.functional as F

@torch.no_grad()
def chunked_cosine_topk(queries: torch.Tensor, gallery: torch.Tensor, topk: int = 50, chunk: int = 2048):
    """Truy xuất nhân ma trận theo khối thẳng trên PyTorch — nhanh tương đương FAISS IndexFlatIP!"""
    qn, gn = F.normalize(queries.float(), dim=1), F.normalize(gallery.float(), dim=1)
    k = min(topk, len(gn))
    res = [torch.topk(qn[i : i + chunk] @ gn.T, k=k, dim=1) for i in range(0, len(qn), chunk)]
    return torch.cat([r.values.cpu() for r in res]), torch.cat([r.indices.cpu() for r in res])

@torch.no_grad()
def alpha_query_expansion(queries: torch.Tensor, gallery: torch.Tensor, topk: int = 5, alpha: float = 3.0) -> torch.Tensor:
    qn, gn = F.normalize(queries.float(), dim=1), F.normalize(gallery.float(), dim=1)
    sc, ix = chunked_cosine_topk(qn, gn, topk=topk)
    w = sc.to(qn.device).clamp_min(0.0).pow(alpha).unsqueeze(-1)
    return F.normalize(qn + (w * gn[ix.to(gn.device)]).sum(dim=1), dim=1)

@torch.no_grad()
def reciprocal_jaccard_rerank(embs: torch.Tensor, k1: int = 20, lam: float = 0.3) -> torch.Tensor:
    """Tái xếp hạng láng giềng thuận nghịch hai chiều R(i, k1) khử nhiễu hub trong 10 dòng."""
    z = F.normalize(embs.float(), dim=1)
    dist = (1.0 - z @ z.T).clamp_min(0.0)
    adj = torch.zeros_like(dist).scatter_(1, torch.argsort(dist, dim=1)[:, :k1], 1.0)
    v = (adj * adj.T) * torch.exp(-dist)
    inter = torch.minimum(v[:, None, :], v[None, :, :]).sum(-1)
    union = torch.maximum(v[:, None, :], v[None, :, :]).sum(-1).clamp_min(1e-6)
    return (1.0 - lam) * (1.0 - inter / union) + lam * dist
