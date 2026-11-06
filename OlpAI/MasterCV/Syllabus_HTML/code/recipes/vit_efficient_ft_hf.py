import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

def load_efficient_vit_classifier(num_classes: int, ckpt: str = "facebook/dinov2-base"):
    """Fine-tune tiết kiệm tham số bằng transformers chuẩn: bật Gradient Checkpointing + mở khóa Head & 2 tầng cuối."""
    proc = AutoImageProcessor.from_pretrained(ckpt)
    net = AutoModelForImageClassification.from_pretrained(ckpt, num_labels=num_classes, ignore_mismatched_sizes=True)
    net.gradient_checkpointing_enable()
    net.requires_grad_(False)
    for name, p in net.named_parameters():
        if any(k in name for k in ("classifier", "layernorm", "layer.10.", "layer.11.")):
            p.requires_grad = True
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01)
    return proc, net, opt
