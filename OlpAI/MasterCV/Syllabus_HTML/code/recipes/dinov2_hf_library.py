import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, Dinov2Model

# Tải trực tiếp DINOv2 từ thư viện chuẩn transformers của OlpAI'25 (không cần cài timm!):
dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-base").cuda().eval()

@torch.no_grad()
def extract_dinov2_cls_and_patch(pixel_values: torch.Tensor) -> torch.Tensor:
    """Ghép nối CLS token toàn cục và trung bình các Patch tokens cục bộ rồi chuẩn hóa L2."""
    hidden = dinov2(pixel_values=pixel_values).last_hidden_state  # (B, 1 + N_patches, 768)
    cls_tok = hidden[:, 0]
    patch_mean = hidden[:, 1:].mean(dim=1)
    return F.normalize(torch.cat([cls_tok, patch_mean], dim=-1), p=2, dim=-1)
