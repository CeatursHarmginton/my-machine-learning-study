from transformers import AutoModelForMultipleChoice, AutoModelForSequenceClassification, AutoTokenizer

# 1. Đầu đọc trắc nghiệm khoa học / pháp luật có ngữ cảnh (chuẩn Top Kaggle LLM Science Exam):
def load_mc_reader(ckpt: str = "microsoft/deberta-v3-base"):
    return AutoTokenizer.from_pretrained(ckpt), AutoModelForMultipleChoice.from_pretrained(ckpt)

# 2. Mô hình suy luận ngôn ngữ tự nhiên (NLI) kiểm chứng câu trả lời có bám sát bằng chứng hay không:
def load_nli_verifier(ckpt: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
    return AutoTokenizer.from_pretrained(ckpt), AutoModelForSequenceClassification.from_pretrained(ckpt)
