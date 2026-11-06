from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import DATASET_SPECS, get_dataset_spec
from .utils import project_root


TORCHVISION_DATASETS = {
    "MNIST": datasets.MNIST,
    "FashionMNIST": datasets.FashionMNIST,
}


class TensorImageDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform=None):
        self.images = images
        self.labels = labels.long()
        self.transform = transform

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        image = self.images[index]
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = image.float().unsqueeze(0) / 255.0
        return image, label


def processed_dir(dataset_name: str, root: str | Path | None = None) -> Path:
    spec = get_dataset_spec(dataset_name)
    return project_root(root) / "data" / "processed" / spec.slug


def load_tensor_split(
    dataset_name: str,
    split: str,
    root: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    folder = processed_dir(dataset_name, root)
    images_path = folder / f"{split}_images.pt"
    labels_path = folder / f"{split}_labels.pt"
    if not images_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Missing processed files for {dataset_name} {split}. "
            "Run: python scripts/prepare_data.py"
        )
    return torch.load(images_path), torch.load(labels_path)


def load_numpy_split(
    dataset_name: str,
    split: str,
    root: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    npz_path = processed_dir(dataset_name, root) / f"{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}. Run scripts/prepare_data.py")
    data = np.load(npz_path)
    return data["images"], data["labels"]


def torchvision_transform(
    dataset_name: str,
    augment: bool = False,
    rotation_degrees: float = 10,
    affine_degrees: float = 0,
    affine_translate: tuple[float, float] | None = (0.08, 0.08),
    affine_scale: tuple[float, float] | None = None,
    affine_shear: float | tuple[float, float] | tuple[float, float, float, float] | None = None,
):
    spec = get_dataset_spec(dataset_name)
    mean = spec.canonical_mean if spec.canonical_mean is not None else 0.0
    std = spec.canonical_std if spec.canonical_std is not None else 1.0
    steps = []
    if augment:
        if rotation_degrees:
            steps.append(transforms.RandomRotation(rotation_degrees))
        if any(value is not None for value in [affine_translate, affine_scale, affine_shear]):
            steps.append(
                transforms.RandomAffine(
                    degrees=affine_degrees,
                    translate=affine_translate,
                    scale=affine_scale,
                    shear=affine_shear,
                )
            )
    steps.extend([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
    return transforms.Compose(steps)


def get_torchvision_dataset(
    dataset_name: str,
    train: bool,
    transform=None,
    download: bool = False,
    root: str | Path | None = None,
):
    dataset_cls = TORCHVISION_DATASETS[dataset_name]
    data_root = project_root(root) / "data" / "raw"
    return dataset_cls(root=data_root, train=train, download=download, transform=transform)


def make_dataloaders(
    dataset_name: str,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    seed: int = 42,
    augment: bool = False,
    stratify: bool = True,
    augmentation_config: dict | None = None,
    num_workers: int = 0,
    root: str | Path | None = None,
):
    augmentation_config = augmentation_config or {}
    train_dataset = get_torchvision_dataset(
        dataset_name,
        train=True,
        transform=torchvision_transform(dataset_name, augment=augment, **augmentation_config),
        download=False,
        root=root,
    )
    val_dataset = get_torchvision_dataset(
        dataset_name,
        train=True,
        transform=torchvision_transform(dataset_name, augment=False),
        download=False,
        root=root,
    )
    test_dataset = get_torchvision_dataset(
        dataset_name,
        train=False,
        transform=torchvision_transform(dataset_name, augment=False),
        download=False,
        root=root,
    )

    targets = np.asarray(train_dataset.targets)
    if stratify:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros_like(targets), targets))
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros_like(targets)))

    loaders = {
        "train": DataLoader(
            Subset(train_dataset, train_idx.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            Subset(val_dataset, val_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    return loaders


def dataset_card(dataset_name: str) -> dict[str, object]:
    spec = get_dataset_spec(dataset_name)
    return {
        **asdict(spec),
        "task": "single-label image classification",
        "pixel_range": "0-255 before normalization",
        "recommended_loss": "CrossEntropyLoss",
    }
