from transformers import AutoConfig, AutoModel

def build_encoder(model_name: str = "microsoft/deberta-v3-base", zero_dropout: bool = False):
    """Khởi tạo bất kỳ Encoder nào (DeBERTa, PhoBERT, XLM-R); bật zero_dropout=True khi giải bài Hồi quy."""
    cfg = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    if zero_dropout:
        cfg.update({"hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0})
    return AutoModel.from_pretrained(model_name, config=cfg)
