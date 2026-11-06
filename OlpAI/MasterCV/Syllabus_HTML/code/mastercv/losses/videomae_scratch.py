import torch

def generate_tube_mask(batch_size: int, t_cubes: int, grid_hw: tuple[int, int], ratio: float = 0.90, device: str = "cpu") -> torch.Tensor:
    """Mặt nạ ống xuyên thời gian (B, T' * H' * W'): che cùng tọa độ (h, w) trên mọi khung hình chống rò rỉ!"""
    n_sp = grid_hw[0] * grid_hw[1]
    ids = torch.argsort(torch.rand(batch_size, n_sp, device=device), dim=1)
    m2d = torch.zeros(batch_size, n_sp, dtype=torch.bool, device=device).scatter_(1, ids[:, : int(round(n_sp * ratio))], True)
    return m2d[:, None, :].expand(-1, t_cubes, -1).reshape(batch_size, t_cubes * n_sp)

def normalized_tube_mse_loss(pred_cubes: torch.Tensor, target_cubes: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Chuẩn hóa trung bình - phương sai cục bộ từng khối 3D trước khi tính MSE trên đúng 90% khối bị che."""
    mean = target_cubes.mean(dim=-1, keepdim=True)
    var = target_cubes.var(dim=-1, unbiased=False, keepdim=True)
    norm_tgt = (target_cubes - mean) / (var + eps).sqrt()
    mse = ((pred_cubes - norm_tgt) ** 2).mean(dim=-1)
    return (mse * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
