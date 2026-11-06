"""Tách riêng hàm Masked Mean Pooling, lớp MeanPoolClassifier & hàm lọc câu sai tự tin cao."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from transformers import AutoConfig, AutoModel


def masked_mean_pooling(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    m = attention_mask.unsqueeze(-1).float()
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)


class MeanPoolClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=cfg)
        self.fc = nn.Linear(cfg.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.fc(masked_mean_pooling(h, attention_mask))


def find_top_wrong_samples(texts: list[str], y_true: list[int], probs: np.ndarray, top_k: int = 5) -> list[dict]:
    preds, conf = probs.argmax(axis=1), probs.max(axis=1)
    wrong = sorted([i for i in range(len(y_true)) if preds[i] != y_true[i]], key=lambda i: conf[i], reverse=True)
    return [{"text": texts[i][:120], "true": int(y_true[i]), "pred": int(preds[i]), "conf": round(float(conf[i]), 4)} for i in wrong[:top_k]]


def quick_cls_report(y_true: list[int], y_pred: list[int], names: list[str] | None = None) -> str:
    rep = classification_report(y_true, y_pred, target_names=names, digits=4, zero_division=0)
    print(rep)
    return rep
