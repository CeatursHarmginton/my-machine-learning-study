import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def greedy_kcenter_coreset(embs: torch.Tensor, ratio: float = 0.05) -> torch.Tensor:
    """Chọn lặp điểm xa nhất giữ trọn biên giới hình học của mẫu bình thường chỉ với 5% bộ nhớ."""
    n, d = embs.shape
    proj = embs @ F.normalize(torch.randn(d, min(128, d), device=embs.device), dim=0)
    sel = [int(torch.randint(0, n, (1,)).item())]
    min_d = torch.cdist(proj, proj[sel[0] : sel[0] + 1]).squeeze(1)
    for _ in range(1, max(1, int(n * ratio))):
        nxt = int(min_d.argmax().item())
        sel.append(nxt)
        min_d = torch.minimum(min_d, torch.cdist(proj, proj[nxt : nxt + 1]).squeeze(1))
    return embs[sel]

class PatchCoreDetector(nn.Module):
    def __init__(self, pyramid_backbone: nn.Module):
        super().__init__()
        self.backbone = pyramid_backbone.eval().requires_grad_(False)
        self.bank = None

    @torch.no_grad()
    def _patches(self, x: torch.Tensor):
        _, c3, c4, _ = self.backbone(x)
        p3 = F.avg_pool2d(c3, 3, 1, 1)
        p4 = F.interpolate(F.avg_pool2d(c4, 3, 1, 1), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        cat = torch.cat([p3, p4], dim=1)
        return F.normalize(cat.permute(0, 2, 3, 1).flatten(0, 2).float(), dim=1), cat.shape[-2:]

    @torch.no_grad()
    def fit(self, normal_loader, ratio: float = 0.05, device: str = "cuda") -> None:
        all_p = torch.cat([self._patches(b["image"].to(device))[0] for b in normal_loader])
        self.bank = greedy_kcenter_coreset(all_p, ratio=ratio)

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        p, (h8, w8) = self._patches(x)
        d = torch.cdist(p, self.bank).min(1).values.view(x.size(0), 1, h8, w8)
        return d.flatten(1).max(1).values, F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
