"""Bộ hàm Xây Dựng Vocab, Đệm Chuỗi, Gộp Mặt Nạ & Mạng BiLSTM Cơ Bản."""
from collections import Counter
import torch
import torch.nn as nn


def tokenize_words(text: str) -> list[str]:
    return text.lower().split()


def build_vocab(texts: list[str], max_vocab: int = 20000, min_freq: int = 2) -> dict[str, int]:
    cnt = Counter(w for line in texts for w in tokenize_words(line))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in cnt.most_common(max_vocab):
        if c >= min_freq and w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def encode_and_pad(text: str, vocab: dict[str, int], max_len: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [vocab.get(w, 1) for w in tokenize_words(text)[:max_len]]
    mask = [1.0] * len(ids) + [0.0] * (max_len - len(ids))
    ids = ids + [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)


def masked_mean_pool(seq_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (seq_out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.lstm = nn.LSTM(dim, dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(self.emb(input_ids))
        return self.fc(masked_mean_pool(out, mask))
