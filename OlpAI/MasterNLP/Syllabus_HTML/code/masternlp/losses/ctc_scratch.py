import torch

def ctc_greedy_decode(token_ids: list[int], blank_id: int = 0) -> list[int]:
    """Ánh xạ gộp CTC B(pi) trong 4 dòng: gộp ký hiệu trùng liên tiếp trước -> lọc bỏ blank sau."""
    dedup = [tok for i, tok in enumerate(token_ids) if i == 0 or tok != token_ids[i - 1]]
    return [tok for tok in dedup if tok != blank_id]

def ctc_forward_loss(log_probs: torch.Tensor, targets: torch.Tensor, blank_id: int = 0) -> torch.Tensor:
    """Quy hoạch động Forward CTC trên chuỗi mở rộng [blank, y1, blank, y2, ...] dài 2U + 1 (18 dòng)."""
    t_steps, ext_len = log_probs.size(0), 2 * targets.numel() + 1
    ext = torch.full((ext_len,), blank_id, dtype=torch.long, device=log_probs.device)
    ext[1::2] = targets
    alpha = torch.full((t_steps, ext_len), -1e9, device=log_probs.device)
    alpha[0, 0] = log_probs[0, ext[0]]
    if ext_len > 1:
        alpha[0, 1] = log_probs[0, ext[1]]
    for t in range(1, t_steps):
        for s in range(ext_len):
            prev = [alpha[t - 1, s]] + ([alpha[t - 1, s - 1]] if s > 0 else [])
            if s > 1 and ext[s] != blank_id and ext[s] != ext[s - 2]:
                prev.append(alpha[t - 1, s - 2])
            alpha[t, s] = torch.logsumexp(torch.stack(prev), dim=0) + log_probs[t, ext[s]]
    end_pair = alpha[-1, -1:] if ext_len == 1 else torch.stack([alpha[-1, -1], alpha[-1, -2]])
    return -torch.logsumexp(end_pair, dim=0)
