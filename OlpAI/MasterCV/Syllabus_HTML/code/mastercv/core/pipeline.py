import copy
import numpy as np
import torch
import torch.nn as nn

class VisionSystem(nn.Module):
    """Tách biệt rõ Tầng 02 (backbone) và Tầng 03 (head) xuyên suốt 6 giai đoạn."""
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x: torch.Tensor, **kwargs):
        return self.head(self.backbone(x), **kwargs)

class ModelEMA:
    """Trung bình trượt hàm mũ (EMA) 10 dòng — tăng 0.5–1.5% F1/mAP miễn phí mà không tốn thêm thời gian suy luận."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for es, ms in zip(self.ema.state_dict().values(), model.state_dict().values()):
            es.copy_(self.decay * es + (1.0 - self.decay) * ms if es.is_floating_point() else ms)

def train_epoch(model: nn.Module, loader, optimizer, loss_fn, scheduler=None,
                grad_accum: int = 1, ema: ModelEMA | None = None, device: str = "cuda") -> float:
    """Vòng lặp huấn luyện vạn năng hỗ trợ Mixed Precision AMP, Gradient Accumulation, Clipping & EMA."""
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for step, batch in enumerate(loader, 1):
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            loss = loss_fn(model, batch) / grad_accum
        scaler.scale(loss).backward()
        if step % grad_accum == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            if scheduler is not None:
                scheduler.step()
        total_loss += loss.item() * grad_accum
    return total_loss / max(1, len(loader))

@torch.no_grad()
def eval_epoch(model: nn.Module, loader, metric_fn, pred_key: str = "logits", device: str = "cuda") -> float:
    model.eval()
    preds, targets = [], []
    for b in loader:
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out = model(b["image"].to(device, non_blocking=True))
        p = out[pred_key] if isinstance(out, dict) else out
        preds.append(p.float().cpu().numpy())
        targets.append(b["target"].numpy())
    return float(metric_fn(np.concatenate(targets), np.concatenate(preds)))
