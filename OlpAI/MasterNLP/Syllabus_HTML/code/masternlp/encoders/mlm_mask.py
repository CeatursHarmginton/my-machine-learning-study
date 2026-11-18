import random
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
    x[rand_pos] = torch.randint(0, vocab_size, (int(rand_pos.sum().item()),), device=x.device)
    return x, labels

def eda_augment_text(text: str, p_del: float = 0.1, p_swap: float = 0.1, seed: int | None = None) -> str:
    """Tăng cường dữ liệu văn bản EDA (Random Deletion + Word Swap) 10 dòng chống quá khớp khi tập Train ít mẫu."""
    rng = random.Random(seed)
    words = text.split()
    if len(words) <= 2:
        return text
    kept = [w for w in words if rng.random() > p_del] or [rng.choice(words)]
    for _ in range(max(1, int(len(kept) * p_swap))):
        if len(kept) >= 2:
            i, j = rng.sample(range(len(kept)), 2)
            kept[i], kept[j] = kept[j], kept[i]
    return " ".join(kept)
