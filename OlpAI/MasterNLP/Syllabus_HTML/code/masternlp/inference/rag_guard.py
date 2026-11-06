import re
import torch
import torch.nn.functional as F

def structured_chunk(text: str, doc_title: str, max_words: int = 180, overlap: int = 35,
                     split_pat: str = r"\n(?=(?:Điều\s+\d+|Section\s+\d+|#{1,3}\s+))") -> list[str]:
    """Phân mảnh tài liệu giữ nguyên tiêu đề phân cấp [Tài liệu | Điều/Mục] gắn vào đầu từng mảnh cắt."""
    chunks, step = [], max(1, max_words - overlap)
    for sec in re.split(split_pat, text.strip()):
        lines = [ln.strip() for ln in sec.splitlines() if ln.strip()]
        if not lines:
            continue
        head, words = lines[0], " ".join(lines[1:] or lines).split()
        for s in range(0, len(words), step):
            chunks.append(f"[{doc_title} | {head}] {' '.join(words[s : s + max_words])}")
            if s + max_words >= len(words):
                break
    return chunks

@torch.no_grad()
def nli_support_score(chunks: list[str], claim: str, tok, nli_model, entail_id: int = 0) -> float:
    """Trả về xác suất cao nhất mệnh đề claim được suy ra hợp lệ từ các đoạn bằng chứng (Chống ảo giác)."""
    enc = tok(chunks, [claim] * len(chunks), padding=True, truncation=True, max_length=512, return_tensors="pt")
    probs = F.softmax(nli_model(**{k: v.to(nli_model.device) for k, v in enc.items()}).logits, dim=-1)
    return float(probs[:, entail_id].max().item())
