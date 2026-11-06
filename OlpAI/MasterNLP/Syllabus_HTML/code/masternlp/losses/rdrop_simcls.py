import torch
import torch.nn.functional as F

def rdrop_kl_loss(logits_1: torch.Tensor, logits_2: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Điều hòa R-Drop KL đối xứng trong 7 dòng — dùng được cho cả Phân loại, NER lẫn Seq2Seq!"""
    lp1, lp2 = F.log_softmax(logits_1, dim=-1), F.log_softmax(logits_2, dim=-1)
    kl = 0.5 * ((lp1.exp() * (lp1 - lp2)).sum(-1) + (lp2.exp() * (lp2 - lp1)).sum(-1))
    return (kl * mask.float()).sum() / mask.float().sum().clamp(min=1.0) if mask is not None else kl.mean()

def simcls_ranking_loss(doc_v: torch.Tensor, gold_v: torch.Tensor, cand_vs: torch.Tensor, margin: float = 0.01) -> torch.Tensor:
    """SimCLS vector hóa 10 dòng (không vòng lặp!): xếp hạng ứng viên cand_vs [B, K, D] theo đúng thứ tự ROUGE."""
    d = F.normalize(doc_v, dim=-1).unsqueeze(1)
    g_sim = (d * F.normalize(gold_v, dim=-1).unsqueeze(1)).sum(-1)
    c_sim = (d * F.normalize(cand_vs, dim=-1)).sum(-1)
    k = c_sim.size(1)
    idx = torch.arange(k, device=c_sim.device)
    rank_gap = (idx[None, :] - idx[:, None]).clamp(min=0).float() * margin
    pair_mat = F.relu(c_sim[:, None, :] - c_sim[:, :, None] + rank_gap[None, :, :])
    tri_mask = idx[None, :] > idx[:, None]
    return F.relu(c_sim - g_sim).mean() + pair_mat[:, tri_mask].mean()
