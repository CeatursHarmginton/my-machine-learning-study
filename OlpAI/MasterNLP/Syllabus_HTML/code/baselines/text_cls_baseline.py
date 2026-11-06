"""Baseline Phân Loại Văn Bản Chia Nhỏ Từng Hàm: Loaders -> Weights -> Model -> Train/Val -> CSV."""
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

os.environ.setdefault("HF_HOME", "/opt/hf_models")


@dataclass
class TextCFG:
    SEED: int = 42
    MODEL_NAME: str = "xlm-roberta-base"
    MAX_LEN: int = 256
    BATCH_SIZE: int = 16
    EPOCHS: int = 4
    LR: float = 2e-5
    VAL_RATIO: float = 0.2
    PATIENCE: int = 2
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int] | None, tokenizer, max_len: int):
        self.texts, self.labels, self.tok, self.max_len = list(texts), labels, tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.tok(str(self.texts[idx]), truncation=True, max_length=self.max_len)
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


def get_text_dataloaders(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: TextCFG, text_col: str = "text", label_col: str = "label"):
    tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    classes = sorted(train_df[label_col].unique().tolist())
    c2i = {c: i for i, c in enumerate(classes)}
    y_all = train_df[label_col].map(c2i).tolist()

    tr_idx, va_idx = train_test_split(np.arange(len(train_df)), test_size=cfg.VAL_RATIO, random_state=cfg.SEED, stratify=y_all)
    col = DataCollatorWithPadding(tok)
    tr_loader = DataLoader(TextDataset(train_df[text_col].iloc[tr_idx], [y_all[i] for i in tr_idx], tok, cfg.MAX_LEN), batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=col)
    va_loader = DataLoader(TextDataset(train_df[text_col].iloc[va_idx], [y_all[i] for i in va_idx], tok, cfg.MAX_LEN), batch_size=cfg.BATCH_SIZE * 2, collate_fn=col)
    te_loader = DataLoader(TextDataset(test_df[text_col], None, tok, cfg.MAX_LEN), batch_size=cfg.BATCH_SIZE * 2, collate_fn=col)
    return classes, y_all, tr_idx.tolist(), tr_loader, va_loader, te_loader


def get_class_weights(y_all: list[int], train_indices: list[int], num_classes: int, device: str) -> torch.Tensor:
    counts = np.bincount([y_all[i] for i in train_indices], minlength=num_classes)
    weights = len(train_indices) / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_text_classifier(model_name: str, num_classes: int):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, device: str) -> float:
    model.train()
    loss_sum, total = 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        y = b.pop("labels")
        optimizer.zero_grad()
        loss = criterion(model(**b).logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        total += y.size(0)
    return loss_sum / max(total, 1)


@torch.no_grad()
def validate_text(model: nn.Module, loader, criterion, device: str) -> tuple[float, float, float]:
    model.eval()
    loss_sum, preds, golds = 0.0, [], []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        y = b.pop("labels")
        logits = model(**b).logits
        loss_sum += criterion(logits, y).item() * y.size(0)
        preds.extend(logits.argmax(-1).cpu().tolist())
        golds.extend(y.cpu().tolist())
    total = max(len(golds), 1)
    acc = 100.0 * sum(p == g for p, g in zip(preds, golds)) / total
    macro_f1 = 100.0 * f1_score(golds, preds, average="macro", zero_division=0)
    return loss_sum / total, acc, macro_f1


def fit_text(model: nn.Module, tr_loader, va_loader, criterion, cfg: TextCFG, ckpt_path: str = "best_nlp.pth") -> float:
    model.to(cfg.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=0.01)
    best_f1, wait = -1.0, 0
    for ep in range(1, cfg.EPOCHS + 1):
        tr_loss = train_one_epoch(model, tr_loader, criterion, opt, cfg.DEVICE)
        va_loss, va_acc, va_f1 = validate_text(model, va_loader, criterion, cfg.DEVICE)
        print(f"Epoch {ep:02d} | TrLoss: {tr_loss:.4f} | VaLoss: {va_loss:.4f} | Acc: {va_acc:.2f}% | F1: {va_f1:.2f}%")
        if va_f1 > best_f1:
            best_f1, wait = va_f1, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1
            if wait >= cfg.PATIENCE:
                break
    return best_f1


@torch.no_grad()
def predict_text_csv(model: nn.Module, te_loader, ids: list, classes: list, device: str, ckpt_path: str = "best_nlp.pth", out_csv: str = "./outputs/submission.csv") -> pd.DataFrame:
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()
    preds = []
    for b in te_loader:
        preds.extend(model(**{k: v.to(device) for k, v in b.items()}).logits.argmax(-1).cpu().tolist())
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id": ids, "label": [classes[p] for p in preds]})
    sub.to_csv(out_csv, index=False)
    return sub
