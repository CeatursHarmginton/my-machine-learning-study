import torch

HOMOGLYPH_TABLE = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
})

def clean_homoglyphs(text: str) -> str:
    """Bảo toàn 100% độ dài chuỗi ký tự (an toàn tuyệt đối cho cả bài chấm vị trí NER/QA)."""
    return text.translate(HOMOGLYPH_TABLE)

def deep_coral_loss(src_h: torch.Tensor, tgt_h: torch.Tensor) -> torch.Tensor:
    """Căn chỉnh hiệp phương sai bậc hai giữa miền Train có nhãn và miền Test không nhãn trong 5 dòng."""
    sc, tc = src_h - src_h.mean(0, keepdim=True), tgt_h - tgt_h.mean(0, keepdim=True)
    cs = (sc.T @ sc) / max(1, src_h.size(0) - 1)
    ct = (tc.T @ tc) / max(1, tgt_h.size(0) - 1)
    return ((cs - ct) ** 2).mean() / 4.0

def free_energy_ood(logits: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """Điểm năng lượng tự do 1 dòng: giá trị càng lớn càng chắc chắn mẫu nằm ngoài phân phối đã học."""
    return -temp * torch.logsumexp(logits / temp, dim=-1)
