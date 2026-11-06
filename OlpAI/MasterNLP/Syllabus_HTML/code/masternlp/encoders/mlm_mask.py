import torch

def dynamic_mlm_mask(input_ids: torch.Tensor, special_ids: list[int],
                     mask_id: int, vocab_size: int, prob: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    """Che mặt nạ động 80-10-10 chuẩn BERT trong 15 dòng (dùng khi tự dựng Transformer cho ngôn ngữ mật mã lạ)."""
    x, labels = input_ids.clone(), input_ids.clone()
    p_mat = torch.full_like(x, prob, dtype=torch.float32)
    for sid in special_ids:
        p_mat.masked_fill_(x == sid, 0.0)
    masked = torch.bernoulli(p_mat).bool()
    labels[~masked] = -100

    rnd = torch.rand_like(p_mat)
    x[masked & (rnd < 0.8)] = mask_id
    rand_pos = masked & (rnd >= 0.8) & (rnd < 0.9)
    x[rand_pos] = torch.randint(0, vocab_size, x.shape, device=x.device)[rand_pos]
    return x, labels
