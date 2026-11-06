"""
Xây dữ liệu demo web: số đếm MLE N-gram toàn Emma + 3 đoạn hiển thị.

Xác suất trên giao diện lấy từ toàn corpus; các đoạn chỉ để trực quan hóa tương tác.

Số đếm lưu lồng theo lịch sử để file nhỏ hơn:
  cont[historyKey] = {"__c": context_count, "word": ngram_count, ...}
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import nltk

nltk.download("gutenberg", quiet=True)
from nltk.corpus import gutenberg

SEP = "\u0001"
OUT_JSON = Path(__file__).with_name("data.json")
OUT_JS = Path(__file__).with_name("corpus-data.js")


def normalize_token(tok: str) -> str:
    return tok.lower().strip()


def build_nested(sentences: list[list[str]], order: int) -> dict[str, dict[str, int]]:
    """historyKey -> {"__c": C(h), word: C(h,w), ...}."""
    cont: dict[str, dict[str, int]] = {}

    for sent in sentences:
        if order == 1:
            hist_key = ""
            bucket = cont.setdefault(hist_key, {"__c": 0})
            for w in sent:
                bucket["__c"] += 1
                bucket[w] = bucket.get(w, 0) + 1
            continue

        if len(sent) < order:
            continue
        for i in range(order - 1, len(sent)):
            hist = sent[i - (order - 1) : i]
            word = sent[i]
            hist_key = SEP.join(hist)
            bucket = cont.setdefault(hist_key, {"__c": 0})
            bucket["__c"] += 1
            bucket[word] = bucket.get(word, 0) + 1

    return cont


def main():
    raw_sents = list(gutenberg.sents("austen-emma.txt"))
    sentences: list[list[str]] = []
    surface: dict[str, Counter] = defaultdict(Counter)

    for sent in raw_sents:
        norm = []
        for tok in sent:
            t = tok.strip()
            if not t:
                continue
            key = normalize_token(t)
            if not key:
                continue
            norm.append(key)
            surface[key][t] += 1
        if norm:
            sentences.append(norm)

    total_tokens = sum(len(s) for s in sentences)
    display_form = {
        k: max(forms.items(), key=lambda kv: kv[1])[0] for k, forms in surface.items()
    }

    models = {}
    for order in (1, 2, 3, 4):
        cont = build_nested(sentences, order)
        n_ctx = len(cont)
        n_ng = sum(len(v) - 1 for v in cont.values())
        models[str(order)] = {"cont": cont}
        print(f"{order}-gram: {n_ng:,} ngrams, {n_ctx:,} contexts")

    raw = gutenberg.raw("austen-emma.txt")
    paras = [re.sub(r"\s+", " ", p.strip()) for p in raw.split("\n\n") if p.strip()]
    candidates = [
        p
        for p in paras
        if 400 < len(p) < 1200
        and not p.startswith("CHAPTER")
        and not p.startswith("VOLUME")
    ]
    selected = [
        ("Đoạn 1 — Emma Woodhouse", candidates[0]),
        ("Đoạn 2 — Miss Taylor", candidates[1]),
        ("Đoạn 3 — Mr. Knightley", candidates[7]),
    ]

    paragraphs = []
    for title, text in selected:
        tokens = nltk.word_tokenize(text)
        paragraphs.append({"title": title, "tokens": tokens})

    # Chỉ giữ display_form cho khóa dạng từ dùng trong gợi ý (nhỏ hơn)
    display_small = {
        k: v for k, v in display_form.items() if any(ch.isalpha() for ch in k)
    }

    payload = {
        "source": "austen-emma.txt (NLTK Gutenberg) — số đếm MLE toàn corpus",
        "total_tokens": total_tokens,
        "num_sentences": len(sentences),
        "sep": SEP,
        "display_form": display_small,
        "models": models,
        "paragraphs": paragraphs,
        "note": (
            "Số đếm N-gram ước lượng từ toàn bộ tiểu thuyết Emma. "
            "Ba đoạn chỉ là mẫu trực quan hóa."
        ),
    }

    raw_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    OUT_JSON.write_text(raw_json, encoding="utf-8")
    OUT_JS.write_text(
        "var CORPUS_DATA="
        + raw_json
        + ';\nif(typeof window!=="undefined")window.CORPUS_DATA=CORPUS_DATA;\n',
        encoding="utf-8",
    )

    size_mb = OUT_JS.stat().st_size / (1024 * 1024)
    print(f"Đã ghi {OUT_JS.name} ({size_mb:.2f} MB), tokens={total_tokens:,}")


if __name__ == "__main__":
    main()
