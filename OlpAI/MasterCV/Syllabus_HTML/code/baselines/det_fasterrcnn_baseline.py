"""Baseline Phát Hiện Vật Thể Chia Nhỏ Từng Hàm: Scale -> Dataset -> Model -> Train/Val -> CSV."""
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF


def scale_boxes(boxes: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
    out = boxes.clone()
    out[:, [0, 2]] *= sx
    out[:, [1, 3]] *= sy
    return out


class DetectionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, img_size: int = 512):
        self.img_dir, self.img_size = Path(img_dir), img_size
        self.groups = list(df.groupby("image_id"))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        image_id, rows = self.groups[idx]
        with Image.open(self.img_dir / f"{image_id}.jpg") as raw:
            rgb = raw.convert("RGB")
            w0, h0 = rgb.size
        img_t = TF.to_tensor(TF.resize(rgb, [self.img_size, self.img_size]))
        raw_boxes = torch.tensor(rows[["x_min", "y_min", "x_max", "y_max"]].values, dtype=torch.float32)
        boxes = scale_boxes(raw_boxes, self.img_size / w0, self.img_size / h0)
        labels = torch.tensor(rows["class_id"].values, dtype=torch.int64) if "class_id" in rows else torch.ones((len(rows),), dtype=torch.int64)
        return img_t, {"boxes": boxes, "labels": labels}


def det_collate_fn(batch):
    return tuple(zip(*batch))


def build_fasterrcnn(num_classes: int = 2, pretrained: bool = True):
    try:
        model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None)
    except TypeError:
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    return model


def train_det_epoch(model, loader, optimizer, device: str) -> float:
    model.train()
    loss_sum = 0.0
    for imgs, targets in loader:
        imgs = [im.to(device) for im in imgs]
        t_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        loss = sum(model(imgs, t_dev).values())
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum / max(len(loader), 1)


@torch.no_grad()
def eval_det_loss(model, loader, device: str) -> float:
    model.train()
    loss_sum = 0.0
    for imgs, targets in loader:
        imgs = [im.to(device) for im in imgs]
        t_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_sum += sum(model(imgs, t_dev).values()).item()
    return loss_sum / max(len(loader), 1)


def fit_detector(model, tr_loader, va_loader, epochs: int = 10, lr: float = 1e-4, device: str = "cuda", ckpt_path: str = "best_det.pth") -> float:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_loss = float("inf")
    for ep in range(1, epochs + 1):
        tr_l = train_det_epoch(model, tr_loader, opt, device)
        va_l = eval_det_loss(model, va_loader, device)
        print(f"Epoch {ep:02d} | TrLoss: {tr_l:.4f} | VaLoss: {va_l:.4f}")
        if va_l < best_loss:
            best_loss = va_l
            torch.save(model.state_dict(), ckpt_path)
    return best_loss


def format_prediction_string(boxes: list, scores: list, sx: float, sy: float, score_thr: float = 0.5) -> str:
    parts = [
        f"{sc:.4f} {b[0]*sx:.1f} {b[1]*sy:.1f} {(b[2]-b[0])*sx:.1f} {(b[3]-b[1])*sy:.1f}"
        for b, sc in zip(boxes, scores)
        if sc >= score_thr
    ]
    return " ".join(parts)


@torch.no_grad()
def predict_boxes_csv(model, test_paths: list[Path], img_size: int = 512, score_thr: float = 0.5, device: str = "cuda", ckpt_path: str = "best_det.pth", out_csv: str = "./outputs/submission.csv") -> pd.DataFrame:
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()
    rows = []
    for p in test_paths:
        with Image.open(p) as raw:
            rgb = raw.convert("RGB")
            w0, h0 = rgb.size
        out = model([TF.to_tensor(TF.resize(rgb, [img_size, img_size])).to(device)])[0]
        pred_str = format_prediction_string(out["boxes"].cpu().tolist(), out["scores"].cpu().tolist(), w0 / img_size, h0 / img_size, score_thr)
        rows.append({"image_id": p.stem, "PredictionString": pred_str})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame(rows)
    sub.to_csv(out_csv, index=False)
    return sub
