from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .utils import device as default_device
from .utils import ensure_dir, set_seed


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_count += int(labels.size(0))
    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_count += int(labels.size(0))
    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


def fit(
    model: nn.Module,
    loaders: dict,
    epochs: int = 5,
    lr: float = 1e-3,
    seed: int = 42,
    checkpoint_path: str | Path | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> list[dict[str, float]]:
    set_seed(seed)
    run_device = default_device()
    model = model.to(run_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    best_val_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, criterion, run_device, max_train_batches
        )
        val_metrics = evaluate(model, loaders["val"], criterion, run_device, max_eval_batches)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": lr,
        }
        history.append(row)
        if checkpoint_path is not None and val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            checkpoint_path = Path(checkpoint_path)
            ensure_dir(checkpoint_path.parent)
            torch.save({"model_state_dict": model.state_dict(), "history": history}, checkpoint_path)
    return history


def save_history(history: list[dict[str, float]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def load_history(path: str | Path) -> list[dict[str, float]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

