"""Capture and visualize how a LeNet-5 changes during training.

This module trains LeNet-5 for a fixed number of epochs while saving a
checkpoint after every epoch (plus the untrained ``epoch_00`` state) and the
per-epoch combined absolute gradient of each convolution layer. The saved
artifacts power three notebook visualizations:

1. an epoch animation of the first two convolution layers' feature maps,
2. an epoch animation of the two convolution layers' kernels together with a
   comparison plot of their combined absolute gradients, and
3. an interactive inference explorer (class dropdown + epoch slider).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML
from matplotlib import animation, gridspec
from torch import nn

from .config import get_dataset_spec
from .data import make_dataloaders
from .models import LeNet5
from .training import evaluate
from .utils import device as default_device
from .utils import ensure_dir, project_root, set_seed

# features[0] is the first convolution, features[3] is the second convolution.
CONV_LAYER_INDICES = (0, 3)


# ---------------------------------------------------------------------------
# Training and artifact IO
# ---------------------------------------------------------------------------
def evolution_dir(dataset_name: str, root: str | Path | None = None) -> Path:
    spec = get_dataset_spec(dataset_name)
    return project_root(root) / "outputs" / "lenet_evolution" / spec.slug


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_lenet_evolution(
    dataset_name: str,
    root: str | Path | None = None,
    epochs: int = 10,
    lr: float = 1e-3,
    seed: int = 42,
    batch_size: int = 128,
    augment: bool = False,
) -> Path:
    """Train LeNet-5 and save a checkpoint plus gradient stats per epoch."""
    set_seed(seed)
    run_device = default_device()
    loaders = make_dataloaders(
        dataset_name,
        batch_size=batch_size,
        val_fraction=0.1,
        seed=seed,
        augment=augment,
        root=root,
    )

    model = LeNet5().to(run_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir = ensure_dir(evolution_dir(dataset_name, root))
    torch.save(_cpu_state(model), out_dir / "epoch_00.pt")

    history: list[dict[str, float]] = []
    grad_history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        grad_abs = {index: 0.0 for index in CONV_LAYER_INDICES}
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for images, labels in loaders["train"]:
            images = images.to(run_device)
            labels = labels.to(run_device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            for index in CONV_LAYER_INDICES:
                conv = model.features[index]
                step_grad = conv.weight.grad.abs().sum().item()
                if conv.bias is not None and conv.bias.grad is not None:
                    step_grad += conv.bias.grad.abs().sum().item()
                grad_abs[index] += step_grad

            optimizer.step()

            batch = labels.size(0)
            total_loss += float(loss.item()) * batch
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_count += batch

        val_metrics = evaluate(model, loaders["val"], criterion, run_device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_count, 1),
                "train_accuracy": total_correct / max(total_count, 1),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        grad_history.append(
            {
                "epoch": epoch,
                "conv1_abs_grad": grad_abs[CONV_LAYER_INDICES[0]],
                "conv2_abs_grad": grad_abs[CONV_LAYER_INDICES[1]],
            }
        )
        torch.save(_cpu_state(model), out_dir / f"epoch_{epoch:02d}.pt")
        print(
            f"[{dataset_name}] epoch {epoch:>2}/{epochs} "
            f"train_acc={history[-1]['train_accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

    meta = {
        "dataset": dataset_name,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "batch_size": batch_size,
        "augment": augment,
        "conv_layer_indices": list(CONV_LAYER_INDICES),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "gradients.json").write_text(json.dumps(grad_history, indent=2), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def load_lenet_evolution(dataset_name: str, root: str | Path | None = None) -> dict:
    """Load the per-epoch checkpoints and gradient/accuracy history."""
    out_dir = evolution_dir(dataset_name, root)
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No LeNet evolution artifacts for {dataset_name} in {out_dir}. "
            "Run: python scripts/train_lenet_evolution.py"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    states = [
        torch.load(out_dir / f"epoch_{epoch:02d}.pt", map_location="cpu")
        for epoch in range(meta["epochs"] + 1)
    ]
    return {
        "states": states,
        "gradients": json.loads((out_dir / "gradients.json").read_text(encoding="utf-8")),
        "history": json.loads((out_dir / "history.json").read_text(encoding="utf-8")),
        "meta": meta,
    }


def _artifacts_complete(dataset_name: str, root: str | Path | None = None, epochs: int = 10) -> bool:
    out_dir = evolution_dir(dataset_name, root)
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("epochs") != epochs:
        return False
    for epoch in range(meta["epochs"] + 1):
        if not (out_dir / f"epoch_{epoch:02d}.pt").exists():
            return False
    return (out_dir / "gradients.json").exists() and (out_dir / "history.json").exists()


def ensure_lenet_evolution(
    dataset_name: str,
    root: str | Path | None = None,
    epochs: int = 10,
    **kwargs,
) -> dict:
    """Load cached artifacts, or train them first if they are missing."""
    if not _artifacts_complete(dataset_name, root, epochs):
        print(f"No cached artifacts for {dataset_name}; training {epochs} epochs...")
        train_lenet_evolution(dataset_name, root=root, epochs=epochs, **kwargs)
    return load_lenet_evolution(dataset_name, root)


# ---------------------------------------------------------------------------
# Model / activation helpers
# ---------------------------------------------------------------------------
def lenet_from_state(state: dict[str, torch.Tensor]) -> LeNet5:
    model = LeNet5()
    model.load_state_dict(state)
    model.eval()
    return model


def prepare_input(image, spec) -> torch.Tensor:
    """Turn a uint8 (28, 28) image into a normalized (1, 1, 28, 28) tensor."""
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    image = image.float() / 255.0
    mean = spec.canonical_mean if spec.canonical_mean is not None else 0.0
    std = spec.canonical_std if spec.canonical_std is not None else 1.0
    image = (image - mean) / std
    return image.unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def lenet_activations(model: LeNet5, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return activations for every panel of the inference explorer."""
    features = model.features
    classifier = model.classifier

    x = features[0](input_tensor)
    conv1 = x
    x = features[1](x)
    x = features[2](x)
    x = features[3](x)
    conv2 = x
    x = features[4](x)
    x = features[5](x)

    flatten = classifier[0](x)
    fc1 = classifier[1](flatten)
    fc1_act = classifier[2](fc1)
    fc2 = classifier[3](fc1_act)
    fc2_act = classifier[4](fc2)
    logits = classifier[5](fc2_act)

    return {
        "conv1": conv1.squeeze(0),
        "conv2": conv2.squeeze(0),
        "flatten": flatten.squeeze(0),
        "fc1": fc1.squeeze(0),
        "fc2": fc2.squeeze(0),
        "logits": logits.squeeze(0),
    }


# ---------------------------------------------------------------------------
# Animation 1: feature maps of the first two conv layers, by epoch
# ---------------------------------------------------------------------------
def animate_feature_maps_by_epoch(
    states,
    image,
    spec,
    interval: int = 700,
    title: str = "LeNet feature maps during training",
) -> HTML:
    """6 first-layer maps on top of 16 second-layer maps, animated by epoch."""
    input_tensor = prepare_input(image, spec)
    conv1_frames, conv2_frames = [], []
    for state in states:
        model = lenet_from_state(state)
        acts = lenet_activations(model, input_tensor)
        conv1_frames.append(acts["conv1"].numpy())
        conv2_frames.append(acts["conv2"].numpy())

    conv1_vmin = float(np.min([f.min() for f in conv1_frames]))
    conv1_vmax = float(np.max([f.max() for f in conv1_frames]))
    conv2_vmin = float(np.min([f.min() for f in conv2_frames]))
    conv2_vmax = float(np.max([f.max() for f in conv2_frames]))

    fig = plt.figure(figsize=(9, 4.6))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.45)
    top = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[0], wspace=0.15)
    bottom = gridspec.GridSpecFromSubplotSpec(2, 8, subplot_spec=outer[1], wspace=0.15, hspace=0.15)

    conv1_images = []
    for i in range(6):
        ax = fig.add_subplot(top[i])
        im = ax.imshow(conv1_frames[0][i], cmap="magma", vmin=conv1_vmin, vmax=conv1_vmax)
        ax.axis("off")
        if i == 0:
            ax.set_title("conv1: 6 maps", loc="left", fontsize=9)
        conv1_images.append(im)

    conv2_images = []
    for j in range(16):
        ax = fig.add_subplot(bottom[j // 8, j % 8])
        im = ax.imshow(conv2_frames[0][j], cmap="viridis", vmin=conv2_vmin, vmax=conv2_vmax)
        ax.axis("off")
        if j == 0:
            ax.set_title("conv2: 16 maps", loc="left", fontsize=9)
        conv2_images.append(im)

    suptitle = fig.suptitle(f"{title}\nepoch 0 (untrained)", fontsize=11)

    def update(frame):
        for i, im in enumerate(conv1_images):
            im.set_data(conv1_frames[frame][i])
        for j, im in enumerate(conv2_images):
            im.set_data(conv2_frames[frame][j])
        label = "0 (untrained)" if frame == 0 else str(frame)
        suptitle.set_text(f"{title}\nepoch {label}")
        return conv1_images + conv2_images + [suptitle]

    anim = animation.FuncAnimation(fig, update, frames=len(states), interval=interval, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


# ---------------------------------------------------------------------------
# Animation 2: conv kernels by epoch + combined absolute gradient comparison
# ---------------------------------------------------------------------------
def _conv2_montage(weight: np.ndarray) -> np.ndarray:
    """Tile a (16, 6, 5, 5) kernel into a (6*6, 16*6) montage with NaN borders."""
    out_channels, in_channels, kh, kw = weight.shape
    tile_h, tile_w = kh + 1, kw + 1
    montage = np.full((in_channels * tile_h - 1, out_channels * tile_w - 1), np.nan, dtype=np.float32)
    for o in range(out_channels):
        for i in range(in_channels):
            row = i * tile_h
            col = o * tile_w
            montage[row : row + kh, col : col + kw] = weight[o, i]
    return montage


def animate_kernels_and_gradients(
    states,
    gradients,
    interval: int = 700,
    title: str = "LeNet kernels and gradients during training",
) -> HTML:
    """Kernels of both conv layers (animated by epoch) above a gradient plot."""
    conv1_key = "features.0.weight"
    conv2_key = "features.3.weight"
    conv1_weights = [state[conv1_key].numpy() for state in states]
    conv2_weights = [state[conv2_key].numpy() for state in states]

    conv1_scale = float(np.max([np.abs(w).max() for w in conv1_weights]))
    conv2_scale = float(np.max([np.abs(w).max() for w in conv2_weights]))
    conv2_montages = [_conv2_montage(w) for w in conv2_weights]

    epochs = [row["epoch"] for row in gradients]
    conv1_grad = [row["conv1_abs_grad"] for row in gradients]
    conv2_grad = [row["conv2_abs_grad"] for row in gradients]
    grad_max = max(max(conv1_grad), max(conv2_grad)) * 1.1

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white")

    fig = plt.figure(figsize=(9, 7.5))
    outer = gridspec.GridSpec(3, 1, height_ratios=[1, 2.2, 2.4], hspace=0.45)
    top = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[0], wspace=0.2)

    conv1_images = []
    for i in range(6):
        ax = fig.add_subplot(top[i])
        im = ax.imshow(conv1_weights[0][i, 0], cmap=cmap, vmin=-conv1_scale, vmax=conv1_scale)
        ax.axis("off")
        if i == 0:
            ax.set_title("conv1 kernels (6)", loc="left", fontsize=9)
        conv1_images.append(im)

    ax_conv2 = fig.add_subplot(outer[1])
    conv2_image = ax_conv2.imshow(conv2_montages[0], cmap=cmap, vmin=-conv2_scale, vmax=conv2_scale)
    ax_conv2.set_title("conv2 kernels (16 filters x 6 input channels)", fontsize=9)
    ax_conv2.set_xticks([])
    ax_conv2.set_yticks([])

    ax_grad = fig.add_subplot(outer[2])
    ax_grad.set_xlim(1, max(epochs))
    ax_grad.set_ylim(0, grad_max)
    ax_grad.set_xlabel("epoch")
    ax_grad.set_ylabel("combined |gradient|")
    ax_grad.set_title("Combined absolute gradient per conv layer", fontsize=10)
    conv1_line, = ax_grad.plot([], [], marker="o", color="#D62728", label="conv1")
    conv2_line, = ax_grad.plot([], [], marker="o", color="#1F77B4", label="conv2")
    ax_grad.legend(loc="upper right")

    suptitle = fig.suptitle(f"{title}\nepoch 0 (untrained)", fontsize=11)

    def update(frame):
        for i, im in enumerate(conv1_images):
            im.set_data(conv1_weights[frame][i, 0])
        conv2_image.set_data(conv2_montages[frame])
        # epoch 0 is the untrained state; gradients exist for epochs >= 1.
        count = frame
        conv1_line.set_data(epochs[:count], conv1_grad[:count])
        conv2_line.set_data(epochs[:count], conv2_grad[:count])
        label = "0 (untrained)" if frame == 0 else str(frame)
        suptitle.set_text(f"{title}\nepoch {label}")
        return conv1_images + [conv2_image, conv1_line, conv2_line, suptitle]

    anim = animation.FuncAnimation(fig, update, frames=len(states), interval=interval, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def plot_gradient_comparison(
    gradients,
    title: str = "Combined absolute gradient per conv layer",
):
    """Static comparison of per-epoch combined |gradient| for both conv layers."""
    epochs = [row["epoch"] for row in gradients]
    conv1_grad = [row["conv1_abs_grad"] for row in gradients]
    conv2_grad = [row["conv2_abs_grad"] for row in gradients]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, conv1_grad, marker="o", color="#D62728", label="conv1")
    ax.plot(epochs, conv2_grad, marker="o", color="#1F77B4", label="conv2")
    ax.set_xlabel("epoch")
    ax.set_ylabel("combined |gradient|")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interactive inference explorer (class dropdown + epoch slider)
# ---------------------------------------------------------------------------
def _representative_indices(labels: np.ndarray, num_classes: int = 10) -> list[int]:
    indices = []
    for label in range(num_classes):
        matches = np.where(labels == label)[0]
        indices.append(int(matches[0]) if len(matches) else 0)
    return indices


def inference_explorer(
    states,
    images,
    labels,
    class_names,
    spec,
    figsize=(16, 4.2),
):
    """Build an ipywidgets UI to step through the inference pipeline.

    Left to right: input image, the 6 first-layer feature maps, the 16
    second-layer feature maps, the 256-d flatten output, and the outputs of
    the first, second, and last classifier layers. A class dropdown (top left)
    picks the image and an epoch slider picks the checkpoint.
    """
    import ipywidgets as widgets

    if isinstance(images, torch.Tensor):
        labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    else:
        labels_np = np.asarray(labels)

    rep_indices = _representative_indices(labels_np, len(class_names))
    n_epochs = len(states) - 1

    class_dropdown = widgets.Dropdown(
        options=[(name, idx) for idx, name in enumerate(class_names)],
        value=0,
        description="Class:",
        layout=widgets.Layout(width="220px"),
    )
    epoch_slider = widgets.IntSlider(
        value=n_epochs,
        min=0,
        max=n_epochs,
        step=1,
        description="Epoch:",
        continuous_update=False,
        layout=widgets.Layout(width="360px"),
    )
    output = widgets.Output()

    def render(class_id: int, epoch: int):
        image = images[rep_indices[class_id]]
        input_tensor = prepare_input(image, spec)
        model = lenet_from_state(states[epoch])
        acts = lenet_activations(model, input_tensor)

        conv1 = acts["conv1"].numpy()
        conv2 = acts["conv2"].numpy()
        flatten = acts["flatten"].numpy()
        fc1 = acts["fc1"].numpy()
        fc2 = acts["fc2"].numpy()
        logits = acts["logits"].numpy()
        probabilities = torch.softmax(acts["logits"], dim=0).numpy()
        predicted = int(np.argmax(logits))

        raw = image.numpy() if isinstance(image, torch.Tensor) else np.asarray(image)

        fig = plt.figure(figsize=figsize)
        widths = [1.1, 1.6, 1.6, 0.7, 0.7, 0.7, 0.9]
        outer = gridspec.GridSpec(1, 7, width_ratios=widths, wspace=0.35)

        ax_in = fig.add_subplot(outer[0])
        ax_in.imshow(raw, cmap="gray")
        ax_in.set_title(f"input: {class_names[class_id]}", fontsize=9)
        ax_in.axis("off")

        grid1 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
        for i in range(6):
            ax = fig.add_subplot(grid1[i // 3, i % 3])
            ax.imshow(conv1[i], cmap="magma")
            ax.axis("off")
            if i == 0:
                ax.set_title("conv1: 6 maps", loc="left", fontsize=9)

        grid2 = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outer[2], wspace=0.1, hspace=0.1)
        for j in range(16):
            ax = fig.add_subplot(grid2[j // 4, j % 4])
            ax.imshow(conv2[j], cmap="viridis")
            ax.axis("off")
            if j == 0:
                ax.set_title("conv2: 16 maps", loc="left", fontsize=9)

        ax_flat = fig.add_subplot(outer[3])
        ax_flat.imshow(flatten.reshape(16, 16), cmap="cividis")
        ax_flat.set_title("flatten\n(256)", fontsize=9)
        ax_flat.axis("off")

        ax_fc1 = fig.add_subplot(outer[4])
        ax_fc1.barh(np.arange(fc1.shape[0]), fc1, color="#4C78A8")
        ax_fc1.set_title("fc1\n(120)", fontsize=9)
        ax_fc1.set_xticks([])
        ax_fc1.set_yticks([])

        ax_fc2 = fig.add_subplot(outer[5])
        ax_fc2.barh(np.arange(fc2.shape[0]), fc2, color="#54A24B")
        ax_fc2.set_title("fc2\n(84)", fontsize=9)
        ax_fc2.set_xticks([])
        ax_fc2.set_yticks([])

        ax_out = fig.add_subplot(outer[6])
        colors = ["#BBBBBB"] * len(class_names)
        colors[predicted] = "#F58518"
        ax_out.barh(np.arange(len(class_names)), probabilities, color=colors)
        ax_out.set_yticks(np.arange(len(class_names)))
        ax_out.set_yticklabels(class_names, fontsize=7)
        ax_out.invert_yaxis()
        ax_out.set_xlim(0, 1)
        ax_out.set_title(f"output (10)\npred: {class_names[predicted]}", fontsize=9)

        epoch_label = "0 (untrained)" if epoch == 0 else str(epoch)
        fig.suptitle(f"LeNet inference — epoch {epoch_label}", fontsize=12, y=1.05)
        fig.tight_layout()
        plt.show()

    def on_change(_change=None):
        with output:
            output.clear_output(wait=True)
            render(class_dropdown.value, epoch_slider.value)

    class_dropdown.observe(on_change, names="value")
    epoch_slider.observe(on_change, names="value")
    on_change()

    controls = widgets.HBox([class_dropdown, epoch_slider])
    return widgets.VBox([controls, output])
