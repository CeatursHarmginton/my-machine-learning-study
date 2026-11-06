from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def linearize_tuples(items: list[tuple[str, ...]], tags: tuple[str, ...] = ("[A]", "[C]", "[O]", "[S]")) -> str:
    """Hàm tổng quát biến mọi bộ quan hệ (Cặp 2, Bộ 3 ASTE, Bộ 4 ACOS) thành chuỗi đích chuẩn hóa cho T5 / ViT5."""
    return " ; ".join(" ".join(f"{t} {v}" for t, v in zip(tags, tup)) for tup in items)

t5_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
t5_net = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
