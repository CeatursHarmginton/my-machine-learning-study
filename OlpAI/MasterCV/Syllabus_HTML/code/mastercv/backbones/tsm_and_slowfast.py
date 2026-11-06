import torch
import torch.nn as nn

def temporal_shift(x: torch.Tensor, n_seg: int = 8, fold_div: int = 8) -> torch.Tensor:
    """Zero-FLOP TSM: biến mọi mạng 2D (B*T, C, H, W) thành mạng Video bằng phép dịch kênh quá khứ/tương lai."""
    bt, c, h, w = x.shape
    x5 = x.view(bt // n_seg, n_seg, c, h, w)
    f = c // fold_div
    out = torch.zeros_like(x5)
    out[:, :-1, :f] = x5[:, 1:, :f]
    out[:, 1:, f : 2 * f] = x5[:, :-1, f : 2 * f]
    out[:, :, 2 * f :] = x5[:, :, 2 * f :]
    return out.view(bt, c, h, w)

class TimeToChannelLateral(nn.Module):
    """Cầu nối ngang SlowFast truyền đặc trưng chuyển động nhanh (alpha*T) sang nhánh ngữ nghĩa chậm (T)."""
    def __init__(self, slow_c: int, alpha: int = 8, beta_inv: int = 8):
        super().__init__()
        fc = slow_c // beta_inv
        self.t2c = nn.Sequential(
            nn.Conv3d(fc, fc * 2, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0), bias=False),
            nn.BatchNorm3d(fc * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_slow: torch.Tensor, x_fast: torch.Tensor) -> torch.Tensor:
        return torch.cat([x_slow, self.t2c(x_fast)], dim=1)
