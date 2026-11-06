"""Baseline Video Chia Nhỏ Từng Hàm (Chuẩn OlpAI SLR): Read -> Sample -> Normalize -> CRNN -> Fit -> ZIP."""
import csv
from pathlib import Path
import zipfile
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from torchvision import models

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def read_video_frames(video_path: str, img_size: int = 224) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(cv2.resize(frame, (img_size, img_size)), cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return torch.zeros((1, img_size, img_size, 3), dtype=torch.uint8)
    return torch.from_numpy(np.stack(frames, axis=0))


def sample_or_pad_frames(frames: torch.Tensor, target_frames: int = 16) -> torch.Tensor:
    n = frames.shape[0]
    if n < target_frames:
        return torch.cat([frames, frames[-1:].repeat(target_frames - n, 1, 1, 1)], dim=0)
    if n > target_frames:
        return frames[torch.linspace(0, n - 1, target_frames).long()]
    return frames


def normalize_video_clip(frames: torch.Tensor) -> torch.Tensor:
    return (frames.permute(0, 3, 1, 2).float() / 255.0 - MEAN) / STD


def load_video_tensor(video_path: str, target_frames: int = 16, img_size: int = 224) -> torch.Tensor:
    raw = read_video_frames(video_path, img_size=img_size)
    return normalize_video_clip(sample_or_pad_frames(raw, target_frames=target_frames))


class VideoDataset(Dataset):
    def __init__(self, root_dir: str, label_to_idx: dict[str, int], target_frames: int = 16):
        self.target_frames = target_frames
        self.samples = [
            (vid, label_to_idx[cls_name])
            for cls_name in label_to_idx
            for vid in sorted((Path(root_dir) / cls_name).glob("*.mp4"))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, y = self.samples[idx]
        return load_video_tensor(str(vid), self.target_frames), y


class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden: int = 256, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(512, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        seq = self.cnn(x.view(b * t, c, h, w)).view(b, t, 512)
        out, _ = self.lstm(seq)
        return self.fc(out[:, -1, :])


def train_video_epoch(model: nn.Module, loader, criterion, optimizer, device: str) -> float:
    model.train()
    loss_sum = 0.0
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x.to(device)), y.to(device))
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum / max(len(loader), 1)


@torch.no_grad()
def validate_video(model: nn.Module, loader, criterion, device: str) -> tuple[float, float]:
    model.eval()
    loss_sum, preds, golds = 0.0, [], []
    for x, y in loader:
        logits = model(x.to(device))
        loss_sum += criterion(logits, y.to(device)).item()
        preds.extend(logits.argmax(1).cpu().tolist())
        golds.extend(y.tolist())
    f1 = 100.0 * f1_score(golds, preds, average="macro", zero_division=0)
    return loss_sum / max(len(loader), 1), f1


def fit_video(model: nn.Module, tr_loader, va_loader, epochs: int = 15, lr: float = 1e-4, device: str = "cuda", ckpt_path: str = "best_video.pth") -> float:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best_f1 = -1.0
    for ep in range(1, epochs + 1):
        tr_loss = train_video_epoch(model, tr_loader, crit, opt, device)
        va_loss, va_f1 = validate_video(model, va_loader, crit, device)
        print(f"Epoch {ep:02d} | TrLoss: {tr_loss:.4f} | VaLoss: {va_loss:.4f} | Macro-F1: {va_f1:.2f}%")
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), ckpt_path)
    return best_f1


@torch.no_grad()
def predict_video_zip(model: nn.Module, test_dir: str, idx_to_label: dict[int, str], target_frames: int = 16, device: str = "cuda", ckpt_path: str = "best_video.pth", csv_path: str = "submission.csv", zip_path: str = "submission.zip") -> None:
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "label"])
        for vp in sorted(Path(test_dir).glob("*.mp4")):
            clip = load_video_tensor(str(vp), target_frames=target_frames).unsqueeze(0).to(device)
            pred = int(model(clip).argmax(1).item())
            w.writerow([vp.name, idx_to_label[pred]])
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=Path(csv_path).name)
