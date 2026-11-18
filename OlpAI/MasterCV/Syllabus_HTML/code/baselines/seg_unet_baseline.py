"""Baseline Phân Đoạn Ảnh (Segmentation) Khép Kín A–Z: SegDataset -> FCN/U-Net -> BCE+Dice -> RLE CSV."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import functional as TF


class SegDataset(Dataset):
    """Đọc cặp (ảnh, mặt nạ nhị phân) và co giãn đồng bộ về đúng (img_size, img_size)."""
    def __init__(self, img_paths: list[Path], mask_paths: list[Path] | None = None, img_size: int = 256):
        self.img_paths, self.mask_paths, self.img_size = list(img_paths), mask_paths, img_size

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        p = self.img_paths[idx]
        img = TF.to_tensor(TF.resize(Image.open(p).convert("RGB"), [self.img_size, self.img_size]))
        if self.mask_paths is None:
            return img, p.stem
        m_pil = Image.open(self.mask_paths[idx]).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)
        mask = (torch.from_numpy(np.array(m_pil, dtype=np.float32)) > 127.0).float().unsqueeze(0)
        return img, mask


def build_seg_baseline(num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    model = fcn_resnet50(weights="DEFAULT" if pretrained else None)
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    return model


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(2, 3))
    dice = (2.0 * inter + smooth) / (probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)
    return 0.5 * bce + 0.5 * (1.0 - dice.mean())


def _forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out["out"] if isinstance(out, dict) else out


def fit_segmenter(model: nn.Module, tr_loader, va_loader, epochs: int = 15, lr: float = 3e-4,
                  patience: int = 4, device: str = "cuda", ckpt_path: str = "best_seg.pth") -> float:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_dice, wait = -1.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        tr_l = 0.0
        for x, m in tr_loader:
            x, m = x.to(device), m.to(device)
            opt.zero_grad()
            loss = bce_dice_loss(_forward_logits(model, x), m)
            loss.backward()
            opt.step()
            tr_l += loss.item()
        model.eval()
        dices = []
        with torch.no_grad():
            for x, m in va_loader:
                p = (torch.sigmoid(_forward_logits(model, x.to(device))) > 0.5).float().cpu()
                inter = (p * m).sum(dim=(1, 2, 3))
                dices.extend(((2.0 * inter + 1e-6) / (p.sum(dim=(1, 2, 3)) + m.sum(dim=(1, 2, 3)) + 1e-6)).tolist())
        va_dice = 100.0 * float(np.mean(dices))
        print(f"Epoch {ep:02d} | TrLoss: {tr_l / max(len(tr_loader), 1):.4f} | Val Dice: {va_dice:.2f}%")
        if va_dice > best_dice:
            best_dice, wait = va_dice, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1
            if wait >= patience:
                break
    return best_dice


def rle_encode_mask(mask: np.ndarray) -> str:
    pixels = np.concatenate([[0], mask.T.flatten(), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


@torch.no_grad()
def predict_seg_rle_csv(model: nn.Module, te_loader, thr: float = 0.5, device: str = "cuda",
                        ckpt_path: str = "best_seg.pth", out_csv: str = "./outputs/submission.csv") -> pd.DataFrame:
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()
    rows = []
    for x, ids in te_loader:
        preds = (torch.sigmoid(_forward_logits(model, x.to(device))) >= thr).cpu().numpy().astype(np.uint8)
        rows.extend({"id": img_id, "rle": rle_encode_mask(p[0])} for img_id, p in zip(ids, preds))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame(rows)
    sub.to_csv(out_csv, index=False)
    return sub
