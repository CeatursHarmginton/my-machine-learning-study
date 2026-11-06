from transformers import AutoModelForMultipleChoice, AutoModelForSequenceClassification, AutoTokenizer

# 1. Đầu đọc trắc nghiệm khoa học / pháp luật có ngữ cảnh (chuẩn Top Kaggle LLM Science Exam):
mc_tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
mc_net = AutoModelForMultipleChoice.from_pretrained("microsoft/deberta-v3-base")

# 2. Mô hình suy luận ngôn ngữ tự nhiên (NLI) kiểm chứng câu trả lời có bám sát bằng chứng hay không:
nli_tok = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
nli_net = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
