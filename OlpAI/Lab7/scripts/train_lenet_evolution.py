"""Train LeNet-5 for both datasets while saving per-epoch artifacts.

For each dataset the script saves an untrained ``epoch_00`` checkpoint, a
checkpoint after every training epoch, the per-epoch combined absolute
gradient of each convolution layer, and the train/validation history. These
artifacts drive the epoch animations in notebook 04.

Usage:
    python scripts/train_lenet_evolution.py
    python scripts/train_lenet_evolution.py --datasets MNIST --epochs 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mnist_fashion_research.config import DATASET_SPECS
from mnist_fashion_research.feature_evolution import train_lenet_evolution


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LeNet-5 and capture per-epoch artifacts.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_SPECS), choices=list(DATASET_SPECS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--augment", action="store_true", help="Apply mild training augmentation.")
    args = parser.parse_args()

    for dataset_name in args.datasets:
        print(f"=== Training LeNet evolution for {dataset_name} ===")
        out_dir = train_lenet_evolution(
            dataset_name,
            root=ROOT,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            batch_size=args.batch_size,
            augment=args.augment,
        )
        try:
            shown = out_dir.relative_to(ROOT)
        except ValueError:
            shown = out_dir
        print(f"Saved artifacts to {shown}")


if __name__ == "__main__":
    main()
