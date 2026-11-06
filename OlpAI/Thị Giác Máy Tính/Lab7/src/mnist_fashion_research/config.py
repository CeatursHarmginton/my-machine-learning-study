from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    slug: str
    display_name: str
    class_names: tuple[str, ...]
    image_size: tuple[int, int] = (28, 28)
    channels: int = 1
    canonical_mean: float | None = None
    canonical_std: float | None = None


DATASET_SPECS: dict[str, DatasetSpec] = {
    "MNIST": DatasetSpec(
        name="MNIST",
        slug="mnist",
        display_name="MNIST",
        class_names=tuple(str(i) for i in range(10)),
        canonical_mean=0.1307,
        canonical_std=0.3081,
    ),
    "FashionMNIST": DatasetSpec(
        name="FashionMNIST",
        slug="fashionmnist",
        display_name="FashionMNIST",
        class_names=(
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ),
        canonical_mean=0.2860,
        canonical_std=0.3530,
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    if name not in DATASET_SPECS:
        choices = ", ".join(DATASET_SPECS)
        raise KeyError(f"Unknown dataset {name!r}. Choose one of: {choices}")
    return DATASET_SPECS[name]


def safe_label_name(label: str) -> str:
    return (
        label.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )

