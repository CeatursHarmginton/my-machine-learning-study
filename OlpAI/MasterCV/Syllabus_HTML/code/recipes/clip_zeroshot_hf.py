import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

def load_clip_hf(ckpt: str = "openai/clip-vit-base-patch16", device: str = "cuda"):
    """Sử dụng trực tiếp CLIPModel & CLIPProcessor có sẵn trong transformers (không cần gói ngoài open_clip!)."""
    return CLIPProcessor.from_pretrained(ckpt), CLIPModel.from_pretrained(ckpt).to(device).eval()

@torch.no_grad()
def build_zeroshot_weights_hf(clip_model: CLIPModel, clip_proc: CLIPProcessor,
                              class_names: list[str], templates: list[str], device: str = "cuda") -> torch.Tensor:
    """Chuẩn hóa L2 kép: (1) từng mẫu câu nhắc -> lấy trung bình -> (2) chuẩn hóa L2 lại tâm lớp!"""
    weights = []
    for cname in class_names:
        toks = clip_proc(text=[t.format(cname) for t in templates], return_tensors="pt", padding=True).to(device)
        t_emb = F.normalize(clip_model.get_text_features(**toks).float(), p=2, dim=-1)
        weights.append(F.normalize(t_emb.mean(dim=0), p=2, dim=-1))
    return torch.stack(weights, dim=0)  # (C, D)
