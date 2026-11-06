import torch

@torch.no_grad()
def closed_form_cross_attention_edit(w_old: torch.Tensor, c_src: torch.Tensor, c_tgt: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Nghiệm đóng IOAI 2024: giải hệ (C_src^T C_src + lam*I) bẻ hướng khái niệm nguồn -> đích trong 0.1 giây."""
    w, cs, ct = w_old.float(), c_src.float(), c_tgt.float()
    rhs = (w @ ct.T) @ cs + lam * w
    gram = cs.T @ cs + lam * torch.eye(w.size(1), device=w.device)
    return torch.linalg.solve(gram.T, rhs.T).T.to(w_old.dtype)

@torch.no_grad()
def steer_unet_cross_attention(unet, c_src: torch.Tensor, c_tgt: torch.Tensor, lam: float = 1.0) -> int:
    edited = 0
    for name, mod in unet.named_modules():
        if name.endswith("attn2") and hasattr(mod, "to_k") and hasattr(mod, "to_v"):
            mod.to_k.weight.data = closed_form_cross_attention_edit(mod.to_k.weight.data, c_src, c_tgt, lam)
            mod.to_v.weight.data = closed_form_cross_attention_edit(mod.to_v.weight.data, c_src, c_tgt, lam)
            edited += 2
    return edited

@torch.no_grad()
def ddim_deterministic_step(x_t: torch.Tensor, eps_pred: torch.Tensor, ab_t: float, ab_prev: float) -> torch.Tensor:
    """Bước khử nhiễu tất định DDIM (eta = 0) bằng PyTorch thuần (không cần gói ngoài diffusers!)."""
    at, ap = torch.tensor(ab_t, device=x_t.device, dtype=x_t.dtype), torch.tensor(ab_prev, device=x_t.device, dtype=x_t.dtype)
    x0 = (x_t - (1.0 - at).sqrt() * eps_pred) / at.sqrt()
    return ap.sqrt() * x0 + (1.0 - ap).sqrt() * eps_pred
