from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mnist_fashion_research.config import DATASET_SPECS, safe_label_name
from mnist_fashion_research.utils import ensure_dir, project_root


TORCHVISION_DATASETS = {
    "MNIST": datasets.MNIST,
    "FashionMNIST": datasets.FashionMNIST,
}


def class_folder(label_id: int, label_name: str) -> str:
    return f"{label_id}_{safe_label_name(label_name)}"


def save_training_formats(dataset_name: str, split: str, images: torch.Tensor, labels: torch.Tensor, out_dir: Path):
    ensure_dir(out_dir)
    torch.save(images, out_dir / f"{split}_images.pt")
    torch.save(labels.long(), out_dir / f"{split}_labels.pt")
    np.savez_compressed(
        out_dir / f"{split}.npz",
        images=images.numpy().astype(np.uint8),
        labels=labels.numpy().astype(np.int64),
    )


def export_images(
    dataset_name: str,
    split: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    root: Path,
    image_limit: int | None,
    force: bool,
) -> list[dict[str, object]]:
    spec = DATASET_SPECS[dataset_name]
    rows: list[dict[str, object]] = []
    image_root = root / "data" / "images" / spec.slug / split
    total = len(labels) if image_limit is None else min(len(labels), image_limit)
    for index in range(total):
        label_id = int(labels[index])
        label_name = spec.class_names[label_id]
        folder = ensure_dir(image_root / class_folder(label_id, label_name))
        filename = f"{index:05d}_label-{label_id}.png"
        path = folder / filename
        if force or not path.exists():
            Image.fromarray(images[index].numpy().astype(np.uint8), mode="L").save(path)
        rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "index": index,
                "label_id": label_id,
                "label_name": label_name,
                "relative_path": path.relative_to(root).as_posix(),
            }
        )
        if (index + 1) % 5000 == 0:
            print(f"{dataset_name} {split}: exported {index + 1:,}/{total:,} images")
    return rows


def prepare_dataset(
    dataset_name: str,
    root: Path,
    export_png: bool,
    image_limit: int | None,
    force: bool,
) -> dict[str, object]:
    spec = DATASET_SPECS[dataset_name]
    dataset_cls = TORCHVISION_DATASETS[dataset_name]
    raw_root = root / "data" / "raw"
    processed_root = root / "data" / "processed" / spec.slug
    image_rows: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}

    for split, train in [("train", True), ("test", False)]:
        dataset = dataset_cls(root=raw_root, train=train, download=True)
        images = dataset.data.cpu()
        labels = dataset.targets.cpu().long()
        save_training_formats(dataset_name, split, images, labels, processed_root)
        split_counts[split] = int(len(labels))
        if export_png:
            image_rows.extend(export_images(dataset_name, split, images, labels, root, image_limit, force))

    label_map = {str(i): name for i, name in enumerate(spec.class_names)}
    (processed_root / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    if export_png:
        labels_csv = root / "data" / "images" / spec.slug / "labels.csv"
        ensure_dir(labels_csv.parent)
        with labels_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["dataset", "split", "index", "label_id", "label_name", "relative_path"],
            )
            writer.writeheader()
            writer.writerows(image_rows)

    return {
        "name": dataset_name,
        "slug": spec.slug,
        "class_names": list(spec.class_names),
        "splits": split_counts,
        "processed_dir": str(processed_root.relative_to(root)),
        "images_exported": export_png,
        "image_limit": image_limit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and export MNIST/FashionMNIST data.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_SPECS), choices=list(DATASET_SPECS))
    parser.add_argument("--no-images", action="store_true", help="Skip PNG image export.")
    parser.add_argument("--image-limit", type=int, default=None, help="Limit PNG export per split for quick tests.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing PNG files.")
    args = parser.parse_args()

    root = project_root(Path(__file__).resolve())
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "datasets": {},
    }
    for dataset_name in args.datasets:
        print(f"Preparing {dataset_name}")
        manifest["datasets"][dataset_name] = prepare_dataset(
            dataset_name=dataset_name,
            root=root,
            export_png=not args.no_images,
            image_limit=args.image_limit,
            force=args.force,
        )
    manifest_path = root / "data" / "manifest.json"
    ensure_dir(manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
