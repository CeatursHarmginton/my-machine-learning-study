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
    """Tái xếp hạng láng giềng thuận nghịch hai chiều R(i, k1) khử nhiễu hub (tính theo hàng, không tràn RAM)."""
    z = F.normalize(embs.float(), dim=1)
    dist = (1.0 - z @ z.T).clamp_min(0.0)
    adj = torch.zeros_like(dist).scatter_(1, torch.argsort(dist, dim=1)[:, :k1], 1.0)
    v = (adj * adj.T) * torch.exp(-dist)
    jaccard = torch.stack([
        1.0 - torch.minimum(v[i : i + 1], v).sum(-1) / torch.maximum(v[i : i + 1], v).sum(-1).clamp_min(1e-6)
        for i in range(len(v))
    ])
    return (1.0 - lam) * jaccard + lam * dist
