from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mnist_fashion_research.config import DATASET_SPECS


def main() -> None:
    errors: list[str] = []
    manifest_path = ROOT / "data" / "manifest.json"
    if not manifest_path.exists():
        errors.append("Missing data/manifest.json. Run scripts/prepare_data.py.")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for dataset_name, spec in DATASET_SPECS.items():
            if dataset_name not in manifest.get("datasets", {}):
                errors.append(f"Manifest missing {dataset_name}.")
            folder = ROOT / "data" / "processed" / spec.slug
            for split in ["train", "test"]:
                for suffix in ["images.pt", "labels.pt", "npz"]:
                    if suffix == "npz":
                        path = folder / f"{split}.npz"
                    else:
                        path = folder / f"{split}_{suffix}"
                    if not path.exists():
                        errors.append(f"Missing {path.relative_to(ROOT)}")
            labels_csv = ROOT / "data" / "images" / spec.slug / "labels.csv"
            if not labels_csv.exists():
                errors.append(f"Missing {labels_csv.relative_to(ROOT)}")

    notebooks = sorted((ROOT / "notebooks").glob("*/*/*.ipynb"))
    if len(notebooks) < 16:
        errors.append(f"Expected at least 16 notebooks, found {len(notebooks)}.")
    for notebook in notebooks:
        try:
            nbformat.read(notebook, as_version=4)
        except Exception as exc:
            errors.append(f"Invalid notebook {notebook.relative_to(ROOT)}: {exc}")

    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"Validation passed: {len(notebooks)} notebooks and prepared data are present.")


if __name__ == "__main__":
    main()

