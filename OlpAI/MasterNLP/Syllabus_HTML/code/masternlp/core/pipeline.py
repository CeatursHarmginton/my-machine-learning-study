from collections import Counter
import copy
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

class ModelEMA:
    """Trung bình trượt trọng số hàm mũ (EMA) 10 dòng — ổn định mạnh mẽ cho DeBERTa / RoBERTa."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for es, ms in zip(self.ema.state_dict().values(), model.state_dict().values()):
            es.copy_(self.decay * es + (1.0 - self.decay) * ms if es.is_floating_point() else ms)

def train_epoch(model: nn.Module, loader, optimizer, loss_fn, scheduler=None,
                grad_accum: int = 1, ema: ModelEMA | None = None, device: str = "cuda") -> float:
    """Vòng lặp huấn luyện vạn năng hỗ trợ AMP, Gradient Accumulation, Clipping & EMA cho mọi bài NLP."""
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for step, batch in enumerate(loader, 1):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            loss = loss_fn(model, batch) / grad_accum
        scaler.scale(loss).backward()
        if step % grad_accum == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            if scheduler is not None:
                scheduler.step()
        total_loss += loss.item() * grad_accum
    return total_loss / max(1, len(loader))
