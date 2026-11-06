from transformers import AutoTokenizer

def sliding_window_tokenize(tok, text: str, max_len: int = 512, stride: int = 128):
    """Cắt tài liệu dài hàng nghìn từ thành các cửa sổ chồng lấp kèm offset_mapping bằng transformers chuẩn."""
    return tok(
        text, max_length=max_len, stride=stride, truncation=True,
        return_overflowing_tokens=True, return_offsets_mapping=True,
    )
