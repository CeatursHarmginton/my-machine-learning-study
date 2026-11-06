"""Baseline Gán Nhãn Token Chia Nhỏ Từng Hàm: EDA -> Align -100 -> Model -> Train/Val -> Predict."""
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification


def quick_ner_eda(word_seqs: list[list[str]], tag_seqs: list[list[str]] | None = None) -> dict:
    lens = [len(s) for s in word_seqs]
    info = {
        "sentences": len(word_seqs),
        "len_min_p50_p95_max": (int(min(lens)), int(np.percentile(lens, 50)), int(np.percentile(lens, 95)), int(max(lens))) if lens else (),
    }
    if tag_seqs is not None:
        info["entity_tags"] = dict(Counter(t for seq in tag_seqs for t in seq if t != "O"))
    print("[NER EDA]", info)
    return info


def align_word_labels(word_ids: list[int | None], word_tags: list[int]) -> list[int]:
    """Gán nhãn thật cho mảnh đầu của từ và -100 cho token đặc biệt / mảnh nối đuôi."""
    aligned, prev_id = [], None
    for wid in word_ids:
        aligned.append(-100 if (wid is None or wid == prev_id) else int(word_tags[wid]))
        prev_id = wid
    return aligned


class TokenClsDataset(Dataset):
    def __init__(self, word_seqs: list[list[str]], tag_ids: list[list[int]] | None, tokenizer, max_len: int = 256):
        self.words, self.tags, self.tok, self.max_len = word_seqs, tag_ids, tokenizer, max_len

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        enc = self.tok(self.words[idx], is_split_into_words=True, truncation=True, max_length=self.max_len)
        if self.tags is not None:
            enc["labels"] = align_word_labels(enc.word_ids(), self.tags[idx])
        return enc


def build_ner_baseline(model_name: str, tag_list: list[str], tokenizer):
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(tag_list))
    return model, DataCollatorForTokenClassification(tokenizer)


def train_ner_epoch(model, loader, optimizer, device: str) -> float:
    model.train()
    loss_sum = 0.0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        optimizer.zero_grad()
        loss = model(**b).loss
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum / max(len(loader), 1)


@torch.no_grad()
def eval_ner_macro_f1(model, loader, device: str) -> tuple[float, float]:
    model.eval()
    loss_sum, preds, golds = 0.0, [], []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        out = model(**b)
        loss_sum += out.loss.item()
        p, y = out.logits.argmax(-1).cpu(), b["labels"].cpu()
        valid = y != -100
        preds.extend(p[valid].tolist())
        golds.extend(y[valid].tolist())
    f1 = 100.0 * f1_score(golds, preds, average="macro", zero_division=0)
    return loss_sum / max(len(loader), 1), f1


@torch.no_grad()
def predict_word_tags(model, words: list[str], tokenizer, id2tag: dict[int, str], device: str, max_len: int = 256) -> list[str]:
    model.to(device).eval()
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_len, return_tensors="pt")
    pred_ids = model(**{k: v.to(device) for k, v in enc.items()}).logits[0].argmax(-1).cpu().tolist()
    out_tags, prev_id = [], None
    for wid, pid in zip(enc.word_ids(batch_index=0), pred_ids):
        if wid is not None and wid != prev_id:
            out_tags.append(id2tag[pid])
        prev_id = wid
    return out_tags
