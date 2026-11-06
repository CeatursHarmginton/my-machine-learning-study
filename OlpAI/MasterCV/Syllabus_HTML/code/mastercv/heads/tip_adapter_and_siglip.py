import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TipAdapterClassifier(nn.Module):
    """Few-shot không cần huấn luyện Backbone: pha trộn Zero-Shot Text Weights & Kho nhớ ảnh mẫu K-shot."""
    def __init__(self, text_w: torch.Tensor, cache_k: torch.Tensor, cache_y: torch.Tensor,
                 num_classes: int, alpha: float = 1.5, beta: float = 5.5, tune_keys: bool = False):
        super().__init__()
        self.register_buffer("w_txt", F.normalize(text_w.float(), dim=-1))
        self.keys = nn.Parameter(F.normalize(cache_k.float(), dim=-1), requires_grad=tune_keys)
        self.register_buffer("vals", F.one_hot(cache_y, num_classes).float())
        self.alpha, self.beta = alpha, beta

    def forward(self, img_f: torch.Tensor) -> torch.Tensor:
        f, k = F.normalize(img_f.float(), dim=-1), F.normalize(self.keys, dim=-1)
        return 100.0 * (f @ self.w_txt.T) + self.alpha * (torch.exp(-self.beta * (1.0 - f @ k.T)) @ self.vals)

class SigLIPLoss(nn.Module):
    """Zhai et al. (ICCV 2023): Sigmoid từng cặp với bias khởi tạo = -10.0 ổn định tuyệt đối ở batch nhỏ."""
    def __init__(self, init_t: float = 10.0, init_b: float = -10.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(math.log(init_t)))
        self.bias = nn.Parameter(torch.tensor(init_b))

    def forward(self, img_z: torch.Tensor, txt_z: torch.Tensor) -> torch.Tensor:
        zi, zt = F.normalize(img_z.float(), dim=-1), F.normalize(txt_z.float(), dim=-1)
        logits = (zi @ zt.T) * self.log_t.exp() + self.bias
        labels = 2.0 * torch.eye(zi.size(0), device=zi.device) - 1.0
        return -F.logsigmoid(labels * logits).mean()
