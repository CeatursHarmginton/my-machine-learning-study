"""Tách nhỏ các hàm Tách Câu, Gom Cụm Từ & Dịch Văn Bản Dài Cơ Bản."""
import re


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", str(text).strip()) if s.strip()]


def pack_sentences(sentences: list[str], max_words: int = 180) -> list[str]:
    chunks, cur, cur_words = [], [], 0
    for s in sentences:
        w = len(s.split())
        if cur and cur_words + w > max_words:
            chunks.append(" ".join(cur))
            cur, cur_words = [s], w
        else:
            cur.append(s)
            cur_words += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def split_long_text(text: str, max_words: int = 180) -> list[str]:
    return pack_sentences(split_sentences(text), max_words=max_words)


def translate_long_text(text: str, batch_translate_fn, max_words: int = 180) -> str:
    chunks = split_long_text(text, max_words=max_words)
    return " ".join(batch_translate_fn(chunks)) if chunks else ""
