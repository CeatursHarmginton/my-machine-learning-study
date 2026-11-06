from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

def build_tapt_and_classifier(model_name: str = "FacebookAI/xlm-roberta-base", num_labels: int = 3):
    """Khởi tạo cặp mô hình Tiền huấn luyện thích nghi miền (TAPT MLM) -> Phân loại dùng chung mọi họ Encoder."""
    tok = AutoTokenizer.from_pretrained(model_name)
    mlm_collator = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=0.15)
    mlm_net = AutoModelForMaskedLM.from_pretrained(model_name)
    clf_net = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    # Sau khi train mlm_net vài epoch trên tập văn bản đề thi, chép trọng số sang clf_net bằng base_model:
    clf_net.base_model.load_state_dict(mlm_net.base_model.state_dict(), strict=False)
    return tok, mlm_collator, mlm_net, clf_net
