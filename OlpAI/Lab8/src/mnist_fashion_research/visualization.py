from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML
from matplotlib import animation


def to_numpy_image(image) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3:
            image = image.squeeze(0)
        image = image.numpy()
    image = np.asarray(image)
    return image


def plot_image_grid(images, labels=None, class_names=None, ncols: int = 10, title: str | None = None):
    n_images = len(images)
    nrows = math.ceil(n_images / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.4 * ncols, 1.6 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for ax, image_index in zip(axes, range(n_images)):
        ax.imshow(to_numpy_image(images[image_index]), cmap="gray")
        ax.axis("off")
        if labels is not None:
            label = int(labels[image_index])
            label_text = class_names[label] if class_names is not None else str(label)
            ax.set_title(label_text, fontsize=9)
    for ax in axes[n_images:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(matrix, class_names, title: str = "Confusion matrix"):
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def animate_samples(images, labels, class_names, n_frames: int = 30, interval: int = 350):
    images = [to_numpy_image(image) for image in images[:n_frames]]
    labels = list(labels[:n_frames])
    fig, ax = plt.subplots(figsize=(3, 3))
    display = ax.imshow(images[0], cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    title = ax.set_title("")

    def update(frame):
        display.set_data(images[frame])
        label = int(labels[frame])
        title.set_text(f"{frame + 1}/{len(images)}: {class_names[label]}")
        return display, title

    anim = animation.FuncAnimation(fig, update, frames=len(images), interval=interval, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def animate_cumulative_mean(images, label_name: str, max_frames: int = 60, interval: int = 180):
    array = np.asarray([to_numpy_image(image) for image in images[:max_frames]], dtype=np.float32)
    cumulative = np.cumsum(array, axis=0) / np.arange(1, len(array) + 1)[:, None, None]
    fig, ax = plt.subplots(figsize=(3, 3))
    display = ax.imshow(cumulative[0], cmap="gray")
    ax.axis("off")
    title = ax.set_title("")

    def update(frame):
        display.set_data(cumulative[frame])
        title.set_text(f"Mean {label_name}: first {frame + 1} images")
        return display, title

    anim = animation.FuncAnimation(fig, update, frames=len(cumulative), interval=interval, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def animate_augmentation(image, transform, n_frames: int = 24, interval: int = 250):
    frames = []
    for _ in range(n_frames):
        augmented = transform(image)
        if isinstance(augmented, torch.Tensor):
            augmented = augmented.detach().cpu().squeeze().numpy()
        frames.append(augmented)
    fig, ax = plt.subplots(figsize=(3, 3))
    display = ax.imshow(frames[0], cmap="gray")
    ax.axis("off")
    title = ax.set_title("Augmentation samples")

    def update(frame):
        display.set_data(frames[frame])
        title.set_text(f"Augmentation sample {frame + 1}/{len(frames)}")
        return display, title

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def animate_training_curve(history, interval: int = 500):
    epochs = [row["epoch"] for row in history]
    train = [row["train_accuracy"] for row in history]
    val = [row["val_accuracy"] for row in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(1, max(epochs))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    train_line, = ax.plot([], [], marker="o", label="train")
    val_line, = ax.plot([], [], marker="o", label="validation")
    ax.legend()

    def update(frame):
        end = frame + 1
        train_line.set_data(epochs[:end], train[:end])
        val_line.set_data(epochs[:end], val[:end])
        ax.set_title(f"Training progress through epoch {epochs[frame]}")
        return train_line, val_line

    anim = animation.FuncAnimation(fig, update, frames=len(epochs), interval=interval, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())

