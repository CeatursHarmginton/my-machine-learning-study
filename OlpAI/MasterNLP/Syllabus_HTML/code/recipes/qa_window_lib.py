from transformers import AutoModelForQuestionAnswering, AutoTokenizer

def tokenize_qa_pairs(tok, question: str, context: str, max_len: int = 384, stride: int = 128):
    """Chỉ trượt cửa sổ trên phần ngữ cảnh (only_second), giữ nguyên câu hỏi ở đầu mọi cửa sổ."""
    return tok(
        question, context, max_length=max_len, stride=stride,
        truncation="only_second", return_overflowing_tokens=True, return_offsets_mapping=True,
    )

qa_tok = AutoTokenizer.from_pretrained("deepset/xlm-roberta-base-squad2")
qa_net = AutoModelForQuestionAnswering.from_pretrained("deepset/xlm-roberta-base-squad2")
