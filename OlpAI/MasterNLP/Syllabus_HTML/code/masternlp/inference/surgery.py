import torch
import torch.nn as nn

def procrustes_svd(src_x: torch.Tensor, tgt_y: torch.Tensor) -> torch.Tensor:
    """Nghiệm đóng min_{Omega^T Omega = I} || X @ Omega - Y ||_F bằng SVD trong 5 dòng PyTorch."""
    xc = src_x - src_x.mean(0, keepdim=True)
    yc = tgt_y - tgt_y.mean(0, keepdim=True)
    u_mat, _, vh_mat = torch.linalg.svd(xc.T @ yc, full_matrices=False)
    return u_mat @ vh_mat

def attach_steering_hook(layer: nn.Module, pos_acts: torch.Tensor, neg_acts: torch.Tensor, coeff: float = 1.5):
    """Contrastive Activation Addition (CAA): tiêm vector điều khiển hành vi vào Residual Stream trong 9 dòng."""
    vec = pos_acts.mean(0) - neg_acts.mean(0)
    vec = vec / vec.norm(p=2).clamp(min=1e-8)
    def _hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        steered = h + coeff * vec.to(h.device, h.dtype)
        return (steered,) + out[1:] if isinstance(out, tuple) else steered
    return layer.register_forward_hook(_hook)
