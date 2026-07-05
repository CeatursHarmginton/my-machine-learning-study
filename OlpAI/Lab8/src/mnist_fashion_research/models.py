from __future__ import annotations

import torch
from torch import nn


class MLPBaseline(nn.Module):
    def __init__(self, num_classes: int = 10, hidden_layer_sizes: tuple[int, ...] | int = (256, 128), dropout: float = 0.2):
        super().__init__()
        if isinstance(hidden_layer_sizes, int):
            hidden_layer_sizes = (hidden_layer_sizes,)

        layers: list[nn.Module] = [nn.Flatten()]
        in_features = 28 * 28
        for hidden_size in hidden_layer_sizes:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden_size
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class LeNet5(nn.Module):
    """LeNet-5 adapted to 28x28 grayscale inputs and 10 classes."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class AlexNetMini(nn.Module):
    """A compact AlexNet-style CNN that fits 28x28 grayscale images."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class FlexibleLeNet(nn.Module):
    """LeNet-style CNN with configurable convolution channel widths."""

    def __init__(
        self,
        conv_channels: tuple[int, ...] = (4, 8),
        num_classes: int = 10,
        classifier_hidden: tuple[int, int] = (120, 84),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
                    nn.Tanh(),
                    nn.AvgPool2d(2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        flatten_features = self._infer_flatten_features()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_features, classifier_hidden[0]),
            nn.Tanh(),
            nn.Linear(classifier_hidden[0], classifier_hidden[1]),
            nn.Tanh(),
            nn.Linear(classifier_hidden[1], num_classes),
        )

    def _infer_flatten_features(self) -> int:
        with torch.no_grad():
            output = self.features(torch.zeros(1, 1, 28, 28))
        return int(output.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class FlexibleAlexNetMini(nn.Module):
    """Compact AlexNet-style CNN with configurable convolution channel widths."""

    def __init__(
        self,
        conv_channels: tuple[int, ...] = (4, 8),
        num_classes: int = 10,
        classifier_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for index, out_channels in enumerate(conv_channels):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
            if index < 2 or index == len(conv_channels) - 1:
                layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        flatten_features = self._infer_flatten_features()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flatten_features, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def _infer_flatten_features(self) -> int:
        with torch.no_grad():
            output = self.features(torch.zeros(1, 1, 28, 28))
        return int(output.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


MODEL_REGISTRY = {
    "mlp": MLPBaseline,
    "simple_cnn": SimpleCNN,
    "lenet": LeNet5,
    "alexnet_mini": AlexNetMini,
    "flexible_lenet": FlexibleLeNet,
    "flexible_alexnet_mini": FlexibleAlexNetMini,
}


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def layer_shape_table(model: nn.Module, input_shape: tuple[int, int, int, int] = (1, 1, 28, 28)):
    rows = []
    hooks = []

    def register(name: str, module: nn.Module):
        def hook(_, __, output):
            if isinstance(output, torch.Tensor):
                rows.append(
                    {
                        "layer": name,
                        "type": module.__class__.__name__,
                        "output_shape": tuple(output.shape),
                        "parameters": sum(p.numel() for p in module.parameters()),
                    }
                )

        if not list(module.children()):
            hooks.append(module.register_forward_hook(hook))

    for name, module in model.named_modules():
        if name:
            register(name, module)

    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape))
    for hook in hooks:
        hook.remove()
    return rows
