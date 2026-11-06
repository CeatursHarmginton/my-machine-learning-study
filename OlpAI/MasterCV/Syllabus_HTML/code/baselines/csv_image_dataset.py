"""Bộ hàm nạp ảnh từ bảng CSV (thay thế get_dataloaders khi đề không chia sẵn thư mục lớp)."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def build_label_map(df: pd.DataFrame, label_col: str = "label") -> tuple[list[str], dict[str, int]]:
    classes = sorted(df[label_col].unique().tolist())
    return classes, {c: i for i, c in enumerate(classes)}


class CSVImageDataset(Dataset):
    def __init__(self, names: list[str], labels: list[int] | None, img_dir: str, tf=None):
        self.img_dir = Path(img_dir)
        self.names = list(names)
        self.labels = list(labels) if labels is not None else None
        self.tf = tf

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = str(self.names[idx])
        path = self.img_dir / (name if "." in name else f"{name}.jpg")
        x = Image.open(path).convert("RGB")
        x = self.tf(x) if self.tf else x
        return (x, self.labels[idx]) if self.labels is not None else (x, Path(name).stem)


def get_csv_dataloaders(train_df: pd.DataFrame, test_df: pd.DataFrame, train_tf, val_tf, cfg, img_col: str = "image_id", label_col: str = "label"):
    classes, c2i = build_label_map(train_df, label_col)
    y_all = train_df[label_col].map(c2i).tolist()
    tr_idx, va_idx = train_test_split(np.arange(len(train_df)), test_size=cfg.VAL_RATIO, random_state=cfg.SEED, stratify=y_all)

    tr_ds = CSVImageDataset(train_df[img_col].iloc[tr_idx], [y_all[i] for i in tr_idx], cfg.TRAIN_DIR, train_tf)
    va_ds = CSVImageDataset(train_df[img_col].iloc[va_idx], [y_all[i] for i in va_idx], cfg.TRAIN_DIR, val_tf)
    te_ds = CSVImageDataset(test_df[img_col], None, cfg.TEST_DIR, val_tf)

    tr_loader = DataLoader(tr_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    return classes, y_all, tr_idx.tolist(), tr_loader, va_loader, te_loader


def print_cls_summary(y_true: list[int], y_pred: list[int], class_names: list[str] | None = None) -> str:
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print(rep)
    return rep
