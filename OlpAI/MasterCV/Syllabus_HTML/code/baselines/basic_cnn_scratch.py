"""Mạng SimpleCNN cơ bản tách hàm tạo khối conv_bn_relu & hàm kiểm tra chiều đầu ra."""
import torch
import torch.nn as nn


def conv_bn_relu(in_c: int, out_c: int, pool: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_chans: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu(in_chans, 32, pool=True),
            conv_bn_relu(32, 64, pool=True),
            conv_bn_relu(64, 128, pool=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x).flatten(1))


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sanity_check_model(model: nn.Module, batch_shape: tuple[int, int, int, int] = (2, 3, 224, 224)) -> dict:
    x = torch.randn(*batch_shape)
    out = model(x)
    return {"output_shape": tuple(out.shape), "trainable_params": count_trainable_params(model)}
