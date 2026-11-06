"""Baseline Seq2Seq Chia Nhỏ Từng Hàm: EDA -> Dataset -> Train/Val -> Generate -> CSV."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq


def quick_seq2seq_eda(src_texts: list[str], tgt_texts: list[str] | None = None) -> dict:
    src_l = [len(str(s).split()) for s in src_texts]
    info = {"pairs": len(src_texts), "src_p50_p95_max": (int(np.percentile(src_l, 50)), int(np.percentile(src_l, 95)), int(max(src_l)))}
    if tgt_texts is not None:
        tgt_l = [len(str(t).split()) for t in tgt_texts]
        info["tgt_p50_p95_max"] = (int(np.percentile(tgt_l, 50)), int(np.percentile(tgt_l, 95)), int(max(tgt_l)))
    print("[Seq2Seq EDA]", info)
    return info


class Seq2SeqDataset(Dataset):
    def __init__(self, src_texts: list[str], tgt_texts: list[str] | None, tokenizer, max_src: int = 256, max_tgt: int = 128):
        self.src, self.tgt, self.tok, self.max_src, self.max_tgt = list(src_texts), tgt_texts, tokenizer, max_src, max_tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        item = self.tok(str(self.src[idx]), truncation=True, max_length=self.max_src)
        if self.tgt is not None:
            item["labels"] = self.tok(text_target=str(self.tgt[idx]), truncation=True, max_length=self.max_tgt)["input_ids"]
        return item


def build_seq2seq_baseline(model_name: str, tokenizer):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)
    return model, collator


def train_seq2seq_epoch(model, loader, optimizer, device: str) -> float:
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
def eval_seq2seq_loss(model, loader, device: str) -> float:
    model.eval()
    loss_sum = sum(model(**{k: v.to(device) for k, v in b.items()}).loss.item() for b in loader)
    return loss_sum / max(len(loader), 1)


def fit_seq2seq(model, tr_loader, va_loader, epochs: int = 5, lr: float = 3e-5, device: str = "cuda", ckpt_path: str = "best_seq2seq.pth") -> float:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_loss = float("inf")
    for ep in range(1, epochs + 1):
        tr_l = train_seq2seq_epoch(model, tr_loader, opt, device)
        va_l = eval_seq2seq_loss(model, va_loader, device)
        print(f"Epoch {ep:02d} | TrLoss: {tr_l:.4f} | VaLoss: {va_l:.4f}")
        if va_l < best_loss:
            best_loss = va_l
            torch.save(model.state_dict(), ckpt_path)
    return best_loss


@torch.no_grad()
def generate_texts(model, tokenizer, loader, device: str = "cuda", max_new_tokens: int = 128) -> list[str]:
    model.to(device).eval()
    preds = []
    for b in loader:
        gen = model.generate(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device), max_new_tokens=max_new_tokens)
        preds.extend(s.strip() for s in tokenizer.batch_decode(gen, skip_special_tokens=True))
    return preds


def predict_seq2seq_csv(model, tokenizer, loader, ids: list, device: str = "cuda", max_new_tokens: int = 128, out_csv: str = "./outputs/submission.csv") -> pd.DataFrame:
    preds = generate_texts(model, tokenizer, loader, device=device, max_new_tokens=max_new_tokens)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id": ids, "prediction": preds})
    sub.to_csv(out_csv, index=False)
    return sub
