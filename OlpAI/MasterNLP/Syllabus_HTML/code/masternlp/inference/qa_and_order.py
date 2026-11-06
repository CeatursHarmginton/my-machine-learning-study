import torch
import torch.nn.functional as F

def decode_qa_null_margin(s_log: torch.Tensor, e_log: torch.Tensor, ctx_mask: torch.Tensor,
                          max_len: int = 30, margin: float = 0.0) -> tuple[int, int]:
    """Giải mã cặp (start, end) hợp lệ tối ưu kèm kiểm định vô nghiệm SQuAD v2 chỉ trong 12 dòng."""
    null_s = float(s_log[0] + e_log[0])
    joint = s_log.masked_fill(~ctx_mask, -1e9)[:, None] + e_log.masked_fill(~ctx_mask, -1e9)[None, :]
    idx = torch.arange(s_log.size(0), device=s_log.device)
    valid = (idx[None, :] >= idx[:, None]) & ((idx[None, :] - idx[:, None]) < max_len)
    joint = joint.masked_fill(~valid, -1e9)
    flat = int(joint.view(-1).argmax())
    bi, bj = divmod(flat, s_log.size(0))
    return (0, 0) if (float(joint[bi, bj]) - null_s) < margin else (bi, bj)

def kendall_order_loss(pred: torch.Tensor, gold: torch.Tensor, alpha: float = 0.5, margin: float = 0.05) -> torch.Tensor:
    """Hàm mất mát kép L1 + Pairwise Ranking cho bài Sắp xếp thứ tự văn bản / mã nguồn (AI4Code & IOAI 2026)."""
    dp, dg = pred[:, None] - pred[None, :], (gold[:, None] - gold[None, :] > 0).float()
    pair_loss = (F.relu(margin - dp) * dg).sum() / dg.sum().clamp(min=1.0)
    return F.l1_loss(pred, gold) + alpha * pair_loss
