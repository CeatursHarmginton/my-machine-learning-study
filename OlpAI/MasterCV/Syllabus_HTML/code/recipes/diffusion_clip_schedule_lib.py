import torch
from transformers import CLIPTextModel, CLIPTokenizer

def load_sd_text_encoder(ckpt: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
    """Bảng OlpAI'25 có sẵn transformers (chứa CLIPTextModel cung cấp ngữ nghĩa văn bản cho U-Net Diffusion)."""
    return CLIPTokenizer.from_pretrained(ckpt), CLIPTextModel.from_pretrained(ckpt).to(device).eval()

@torch.no_grad()
def encode_concept_pair(sd_tok: CLIPTokenizer, sd_txt_enc: CLIPTextModel,
                        src_prompt: str, tgt_prompt: str, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Trích xuất cặp chuỗi nhúng (77, d_txt) của khái niệm nguồn & đích phục vụ chỉnh sửa Cross-Attention."""
    inp = sd_tok([src_prompt, tgt_prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
    hidden = sd_txt_enc(**inp).last_hidden_state
    return hidden[0], hidden[1]

def build_cosine_alpha_bars(steps: int = 1000, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, steps, steps + 1) / steps
    f = torch.cos((t + s) / (1.0 + s) * (torch.pi * 0.5)) ** 2
    return (f[1:] / f[0]).clamp(1e-4, 0.9999)
