from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mnist_fashion_research.config import DATASET_SPECS, get_dataset_spec
from mnist_fashion_research.data import load_tensor_split, make_dataloaders
from mnist_fashion_research.evaluation import predict_batches
from mnist_fashion_research.models import (
    FlexibleAlexNetMini,
    FlexibleLeNet,
    MLPBaseline,
    count_parameters,
)
from mnist_fashion_research.training import fit, save_history
from mnist_fashion_research.utils import device, ensure_dir, set_seed


SEED = 42
ROTATION_DEGREES = [2, 5, 10, 15]
AFFINE_POLICIES = [
    {
        "affine_name": "translate_4pct",
        "affine_translate": (0.04, 0.04),
        "affine_scale": None,
        "affine_shear": None,
    },
    {
        "affine_name": "translate_8pct",
        "affine_translate": (0.08, 0.08),
        "affine_scale": None,
        "affine_shear": None,
    },
    {
        "affine_name": "zoom_10pct",
        "affine_translate": (0.04, 0.04),
        "affine_scale": (0.90, 1.10),
        "affine_shear": None,
    },
    {
        "affine_name": "shear_8deg",
        "affine_translate": (0.04, 0.04),
        "affine_scale": None,
        "affine_shear": (-8, 8, -8, 8),
    },
]
STRATIFY_OPTIONS = [True, False]
MLP_HIDDEN_LAYER_SIZES = [64, 128, 256]
LENET_CONV_VARIANTS = [(4, 8), (4, 8, 16), (4, 8, 16, 32)]
ALEXNET_CONV_VARIANTS = [(4, 8), (4, 8, 16), (4, 8, 16, 32)]
EPOCHS = 4
LR_VALUES = [5e-4, 1e-3, 2e-3]
BATCH_SIZE_VALUES = [64, 128]
WEIGHT_DECAY_VALUES = [0.0, 1e-4]
HYPERPARAMETER_GRID = [
    {
        "name": f"adam_lr{lr:g}_bs{batch_size}_wd{weight_decay:g}",
        "epochs": EPOCHS,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
    }
    for lr, batch_size, weight_decay in itertools.product(
        LR_VALUES,
        BATCH_SIZE_VALUES,
        WEIGHT_DECAY_VALUES,
    )
]
assert len(HYPERPARAMETER_GRID) == 12


def augmentation_config(rotation_degrees: int, affine_policy: dict) -> dict:
    return {
        "rotation_degrees": rotation_degrees,
        "affine_translate": affine_policy["affine_translate"],
        "affine_scale": affine_policy["affine_scale"],
        "affine_shear": affine_policy["affine_shear"],
    }


def channel_variant_name(channels: tuple[int, ...]) -> str:
    return "conv_" + "_".join(str(channel) for channel in channels)


def model_variants() -> list[dict]:
    variants = []
    for hidden_size in MLP_HIDDEN_LAYER_SIZES:
        variants.append(
            {
                "model_family": "MLP",
                "variant": f"hidden_{hidden_size}",
                "hidden_layer_sizes": (hidden_size,),
                "conv_channels": None,
            }
        )
    for channels in LENET_CONV_VARIANTS:
        variants.append(
            {
                "model_family": "LeNet",
                "variant": channel_variant_name(channels),
                "hidden_layer_sizes": None,
                "conv_channels": channels,
            }
        )
    for channels in ALEXNET_CONV_VARIANTS:
        variants.append(
            {
                "model_family": "AlexNet",
                "variant": channel_variant_name(channels),
                "hidden_layer_sizes": None,
                "conv_channels": channels,
            }
        )
    return variants


def make_model(experiment: dict) -> torch.nn.Module:
    family = experiment["model_family"]
    if family == "MLP":
        return MLPBaseline(hidden_layer_sizes=tuple(experiment["hidden_layer_sizes"]), dropout=0.2)
    if family == "LeNet":
        return FlexibleLeNet(conv_channels=tuple(experiment["conv_channels"]))
    if family == "AlexNet":
        return FlexibleAlexNetMini(conv_channels=tuple(experiment["conv_channels"]), dropout=0.3)
    raise ValueError(f"Unknown model family: {family}")


def experiment_record(
    model_variant: dict,
    rotation: int,
    affine_policy: dict,
    stratify: bool,
    hparams: dict,
    purpose: str,
) -> dict:
    experiment_id = "_".join(
        [
            model_variant["model_family"].lower(),
            model_variant["variant"],
            f"rot{rotation}",
            affine_policy["affine_name"],
            "stratified" if stratify else "randomsplit",
            hparams["name"],
        ]
    )
    return {
        "experiment_id": experiment_id,
        "purpose": purpose,
        "model_family": model_variant["model_family"],
        "variant": model_variant["variant"],
        "hidden_layer_sizes": model_variant["hidden_layer_sizes"],
        "conv_channels": model_variant["conv_channels"],
        "rotation_degrees": rotation,
        "affine_name": affine_policy["affine_name"],
        "affine_policy": affine_policy,
        "stratify": stratify,
        **hparams,
    }


def dedupe_experiments(experiments: list[dict]) -> list[dict]:
    deduped = {}
    for experiment in experiments:
        deduped.setdefault(experiment["experiment_id"], experiment)
    return list(deduped.values())


def build_experiments(run_mode: str) -> list[dict]:
    variants = model_variants()
    base_rotation = 10
    base_affine_policy = AFFINE_POLICIES[1]
    base_stratify = True
    base_hparams = HYPERPARAMETER_GRID[0]
    base_model_variant = next(
        variant
        for variant in variants
        if variant["model_family"] == "LeNet" and variant["variant"] == "conv_4_8"
    )

    full_experiments = [
        experiment_record(model_variant, rotation, affine_policy, stratify, hparams, "full_grid")
        for model_variant in variants
        for rotation in ROTATION_DEGREES
        for affine_policy in AFFINE_POLICIES
        for stratify in STRATIFY_OPTIONS
        for hparams in HYPERPARAMETER_GRID
    ]
    if run_mode == "full_grid":
        return full_experiments

    balanced_experiments = []
    balanced_experiments.extend(
        experiment_record(base_model_variant, rotation, base_affine_policy, base_stratify, base_hparams, "rotation_sweep")
        for rotation in ROTATION_DEGREES
    )
    balanced_experiments.extend(
        experiment_record(base_model_variant, base_rotation, affine_policy, base_stratify, base_hparams, "affine_sweep")
        for affine_policy in AFFINE_POLICIES
    )
    balanced_experiments.extend(
        experiment_record(base_model_variant, base_rotation, base_affine_policy, stratify, base_hparams, "stratification_sweep")
        for stratify in STRATIFY_OPTIONS
    )
    balanced_experiments.extend(
        experiment_record(model_variant, base_rotation, base_affine_policy, base_stratify, hparams, "model_hparam_sweep")
        for model_variant in variants
        for hparams in HYPERPARAMETER_GRID
    )
    return dedupe_experiments(balanced_experiments)


def verify_processed_data(dataset_name: str) -> None:
    for split in ["train", "test"]:
        load_tensor_split(dataset_name, split, root=ROOT)


def train_dataset(
    dataset_name: str,
    run_mode: str,
    max_train_batches: int | None,
    max_eval_batches: int | None,
    limit: int | None,
) -> Path:
    verify_processed_data(dataset_name)
    spec = get_dataset_spec(dataset_name)
    experiments = build_experiments(run_mode)
    if limit is not None:
        experiments = experiments[:limit]

    checkpoint_dir = ROOT / "outputs" / "models" / spec.slug / "combined_pipeline"
    report_dir = ROOT / "outputs" / "reports" / spec.slug
    ensure_dir(checkpoint_dir)
    ensure_dir(report_dir)

    run_device = device()
    results = []
    print(f"{dataset_name}: training {len(experiments)} experiments on {run_device}")
    for run_index, experiment in enumerate(experiments, start=1):
        set_seed(SEED + run_index)
        print(f"[{run_index:03d}/{len(experiments):03d}] {experiment['experiment_id']}", flush=True)
        loaders = make_dataloaders(
            dataset_name,
            batch_size=experiment["batch_size"],
            val_fraction=0.1,
            seed=SEED,
            augment=True,
            stratify=experiment["stratify"],
            augmentation_config=augmentation_config(experiment["rotation_degrees"], experiment["affine_policy"]),
            root=ROOT,
        )
        model = make_model(experiment)
        trainable_parameters = count_parameters(model)
        checkpoint_path = checkpoint_dir / f"{experiment['experiment_id']}.pt"
        history_path = report_dir / f"{experiment['experiment_id']}_history.json"

        history = fit(
            model,
            loaders,
            epochs=experiment["epochs"],
            lr=experiment["lr"],
            weight_decay=experiment["weight_decay"],
            seed=SEED + run_index,
            checkpoint_path=checkpoint_path,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
        )
        save_history(history, history_path)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["experiment"] = {
            key: value
            for key, value in experiment.items()
            if key not in {"affine_policy"}
        }
        checkpoint["trainable_parameters"] = trainable_parameters
        torch.save(checkpoint, checkpoint_path)

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(run_device)
        predictions_bundle = predict_batches(model, loaders["test"], run_device, max_batches=max_eval_batches)
        labels = predictions_bundle["labels"].numpy()
        predictions = predictions_bundle["predictions"].numpy()

        results.append(
            {
                "experiment_id": experiment["experiment_id"],
                "purpose": experiment["purpose"],
                "model_family": experiment["model_family"],
                "variant": experiment["variant"],
                "rotation_degrees": experiment["rotation_degrees"],
                "affine_name": experiment["affine_name"],
                "stratify": experiment["stratify"],
                "epochs": experiment["epochs"],
                "lr": experiment["lr"],
                "batch_size": experiment["batch_size"],
                "weight_decay": experiment["weight_decay"],
                "trainable_parameters": trainable_parameters,
                "best_val_accuracy": max(row["val_accuracy"] for row in history),
                "final_val_accuracy": history[-1]["val_accuracy"],
                "test_accuracy": float((predictions == labels).mean()),
                "test_macro_f1": f1_score(labels, predictions, average="macro"),
                "checkpoint_path": checkpoint_path.relative_to(ROOT).as_posix(),
                "history_path": history_path.relative_to(ROOT).as_posix(),
            }
        )

    results_df = pd.DataFrame(results).sort_values("best_val_accuracy", ascending=False)
    results_path = report_dir / f"{spec.slug}_combined_pipeline_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"{dataset_name}: saved {results_path.relative_to(ROOT).as_posix()}")
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the combined MNIST/FashionMNIST pipelines.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_SPECS), choices=list(DATASET_SPECS))
    parser.add_argument("--run-mode", default="balanced", choices=["balanced", "full_grid"])
    parser.add_argument("--max-train-batches", type=int, default=40)
    parser.add_argument("--max-eval-batches", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None, help="Optional experiment limit for a smoke run.")
    args = parser.parse_args()

    max_train_batches = None if args.max_train_batches <= 0 else args.max_train_batches
    max_eval_batches = None if args.max_eval_batches <= 0 else args.max_eval_batches
    for dataset_name in args.datasets:
        train_dataset(
            dataset_name=dataset_name,
            run_mode=args.run_mode,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
