import torch
import torch.nn.functional as F

@torch.no_grad()
def beam_search_wu(step_fn, bos_id: int, eos_id: int, k: int = 4, max_len: int = 64, alpha: float = 0.8) -> list[int]:
    """Beam Search có chuẩn hóa độ dài ((5 + |Y|)/6)^alpha cô đọng trong 19 dòng dễ nhớ."""
    beams: list[tuple[list[int], float, bool]] = [([bos_id], 0.0, False)]
    norm = lambda lp, n: lp / (((5.0 + n) / 6.0) ** alpha)
    for _ in range(max_len):
        cands = []
        for seq, lp, done in beams:
            if done:
                cands.append((seq, lp, True))
                continue
            vals, ids = torch.topk(step_fn(torch.tensor([seq]))[0], k)
            cands.extend((seq + [i], lp + v, i == eos_id) for v, i in zip(vals.tolist(), ids.tolist()))
        beams = sorted(cands, key=lambda x: norm(x[1], len(x[0])), reverse=True)[:k]
        if all(d for _, _, d in beams):
            break
    return beams[0][0]

def top_p_filter(logits: torch.Tensor, top_p: float = 0.9, temp: float = 0.8) -> torch.Tensor:
    """Nucleus Sampling (8 dòng): cắt bỏ đuôi phân phối xác suất thấp tránh sinh từ lạc đề."""
    z = logits / max(1e-5, temp)
    s_val, s_idx = torch.sort(z, descending=True, dim=-1)
    drop = torch.cumsum(F.softmax(s_val, dim=-1), dim=-1) > top_p
    drop[..., 1:], drop[..., 0] = drop[..., :-1].clone(), False
    mask = torch.zeros_like(z, dtype=torch.bool).scatter_(-1, s_idx, drop)
    return z.masked_fill(mask, -1e9)
