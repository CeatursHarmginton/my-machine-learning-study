"""Baseline Phân Loại Ảnh Cơ Bản Chia Nhỏ Từng Hàm: Transforms -> Loaders -> Model -> Train/Val -> CSV."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class TransformDataset(Dataset):
    def __init__(self, subset, tf):
        self.subset, self.tf = subset, tf

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, y = self.subset[idx]
        return self.tf(img), y


class TestDataset(Dataset):
    def __init__(self, test_dir: str, tf):
        self.files = sorted(p for p in Path(test_dir).iterdir() if p.is_file())
        self.tf = tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        return self.tf(Image.open(p).convert("RGB")), p.stem


def get_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, val_tf


def get_dataloaders(cfg):
    raw_ds = datasets.ImageFolder(cfg.TRAIN_DIR)
    val_len = int(len(raw_ds) * cfg.VAL_RATIO)
    gen = torch.Generator().manual_seed(cfg.SEED)
    tr_sub, va_sub = random_split(raw_ds, [len(raw_ds) - val_len, val_len], generator=gen)

    train_tf, val_tf = get_transforms(cfg.IMG_SIZE)
    tr_loader = DataLoader(TransformDataset(tr_sub, train_tf), batch_size=cfg.BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(TransformDataset(va_sub, val_tf), batch_size=cfg.BATCH_SIZE, shuffle=False)
    te_loader = DataLoader(TestDataset(cfg.TEST_DIR, val_tf), batch_size=cfg.BATCH_SIZE, shuffle=False)
    return raw_ds.classes, raw_ds.targets, tr_sub.indices, tr_loader, va_loader, te_loader


def get_class_weights(all_targets: list[int], train_indices: list[int], num_classes: int, device: str) -> torch.Tensor:
    counts = np.bincount([all_targets[i] for i in train_indices], minlength=num_classes)
    weights = len(train_indices) / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.fc.in_features, num_classes))
    return model


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, device: str) -> tuple[float, float]:
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)


@torch.no_grad()
def validate(model: nn.Module, loader, criterion, device: str) -> tuple[float, float, float]:
    model.eval()
    loss_sum, preds, golds = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * x.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        golds.extend(y.cpu().tolist())
    total = max(len(golds), 1)
    acc = 100.0 * sum(p == g for p, g in zip(preds, golds)) / total
    macro_f1 = 100.0 * f1_score(golds, preds, average="macro", zero_division=0)
    return loss_sum / total, acc, macro_f1


def fit(model: nn.Module, tr_loader, va_loader, criterion, cfg, ckpt_path: str = "best_model.pth") -> float:
    model.to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    best_f1, wait = -1.0, 0

    for ep in range(1, cfg.EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, criterion, optimizer, cfg.DEVICE)
        va_loss, va_acc, va_f1 = validate(model, va_loader, criterion, cfg.DEVICE)
        print(f"Epoch {ep:02d} | TrLoss: {tr_loss:.4f} ({tr_acc:.1f}%) | VaLoss: {va_loss:.4f} | Acc: {va_acc:.2f}% | F1: {va_f1:.2f}%")
        if va_f1 > best_f1:
            best_f1, wait = va_f1, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1
            if wait >= cfg.PATIENCE:
                break
    return best_f1


@torch.no_grad()
def predict_csv(model: nn.Module, te_loader, classes: list[str], device: str, ckpt_path: str = "best_model.pth", out_csv: str = "./outputs/submission.csv") -> pd.DataFrame:
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()
    rows = []
    for x, ids in te_loader:
        preds = model(x.to(device)).argmax(1).cpu().tolist()
        rows.extend({"id": sample_id, "label": classes[p]} for sample_id, p in zip(ids, preds))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame(rows)
    sub.to_csv(out_csv, index=False)
    return sub
