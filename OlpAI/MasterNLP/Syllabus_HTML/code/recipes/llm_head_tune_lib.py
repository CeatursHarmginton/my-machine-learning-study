import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_efficient_llm_clf(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", num_labels: int = 3):
    """Tải LLM bằng transformers thuần (BF16 + Gradient Checkpointing + mở khóa Head & 2 tầng cuối)."""
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.pad_token or tok.eos_token
    net = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, torch_dtype=torch.bfloat16)
    net.config.pad_token_id = tok.pad_token_id
    net.gradient_checkpointing_enable()
    net.requires_grad_(False)
    for name, p in net.named_parameters():
        if any(k in name for k in ("score", "classifier", "norm", "layers.26.", "layers.27.")):
            p.requires_grad = True
    return tok, net
