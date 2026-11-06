from collections import Counter
import torch
import torch.nn as nn

def train_bpe_merges(texts: list[str], n_merges: int = 300) -> dict[tuple[int, int], int]:
    """Thuật toán BPE cốt lõi trong 11 dòng: lặp lại việc gộp cặp byte kề nhau xuất hiện nhiều nhất."""
    ids, merges = list(" ".join(texts).encode("utf-8")), {}
    for step in range(n_merges):
        counts = Counter(zip(ids, ids[1:]))
        if not counts:
            break
        pair = counts.most_common(1)[0][0]
        merges[pair], nxt, i = 256 + step, [], 0
        while i < len(ids):
            cond = i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair
            nxt.append(256 + step if cond else ids[i])
            i += 2 if cond else 1
        ids = nxt
    return merges

def get_llrd_groups(model: nn.Module, lr: float = 2e-5, head_lr: float = 1e-4, decay: float = 0.9) -> list[dict]:
    """Layer-Wise LR Decay tổng quát cho mọi mạng BERT / RoBERTa / DeBERTa trong 10 dòng."""
    no_wd = ("bias", "LayerNorm.weight", "layernorm.weight")
    max_l = max([int(s) for n in dict(model.named_parameters()) for s in n.split(".") if s.isdigit()] + [11]) + 1
    groups = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            nums = [int(s) for s in name.split(".") if s.isdigit()]
            cur_lr = head_lr if ("classifier" in name or "head" in name) else lr * (decay ** (max_l + 1 if "embed" in name else max_l - (nums[0] if nums else 0)))
            groups.append({"params": [p], "lr": cur_lr, "weight_decay": 0.0 if any(k in name for k in no_wd) else 0.01})
    return groups

def train_epoch(model: nn.Module, loader, optimizer, loss_fn, scheduler=None, device: str = "cuda") -> float:
    """Vòng lặp huấn luyện vạn năng 14 dòng: dùng chung cho Phân loại, Hồi quy, NER, QA, Retrieval & Seq2Seq."""
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    total_loss = 0.0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            loss = loss_fn(model, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))
