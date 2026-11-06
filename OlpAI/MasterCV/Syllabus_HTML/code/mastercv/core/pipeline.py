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

def train_epoch(model: nn.Module, loader, optimizer, loss_fn, scheduler=None, device: str = "cuda") -> float:
    """Vòng lặp huấn luyện vạn năng 14 dòng với Mixed Precision AMP & Gradient Clipping."""
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    total_loss = 0.0
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            loss = loss_fn(model, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
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
