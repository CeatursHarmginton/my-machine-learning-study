from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mnist_fashion_research.config import DATASET_SPECS


def is_notebook_checkpoint(path: Path) -> bool:
    return ".ipynb_checkpoints" in path.parts


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
            dataset_manifest = manifest.get("datasets", {}).get(dataset_name, {})
            if dataset_manifest.get("images_exported", False):
                labels_csv = ROOT / "data" / "images" / spec.slug / "labels.csv"
                if not labels_csv.exists():
                    errors.append(f"Missing {labels_csv.relative_to(ROOT)}")

    expected_notebooks = [
        ROOT / "notebooks" / dataset_name / f"combined_pipeline_{dataset_name}.ipynb"
        for dataset_name in DATASET_SPECS
    ]
    notebooks = [
        path
        for path in sorted((ROOT / "notebooks").glob("*/*.ipynb"))
        if not is_notebook_checkpoint(path)
    ]
    legacy_notebooks = [
        path
        for path in sorted((ROOT / "notebooks").glob("*/*/*.ipynb"))
        if not is_notebook_checkpoint(path)
    ]
    for notebook in expected_notebooks:
        if not notebook.exists():
            errors.append(f"Missing {notebook.relative_to(ROOT)}")
    unexpected_notebooks = [path for path in notebooks if path not in expected_notebooks]
    for notebook in unexpected_notebooks:
        errors.append(f"Unexpected notebook {notebook.relative_to(ROOT)}")
    for notebook in legacy_notebooks:
        errors.append(f"Unexpected legacy step notebook {notebook.relative_to(ROOT)}")
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
    print(f"Validation passed: {len(notebooks)} combined notebooks and prepared data are present.")


if __name__ == "__main__":
    main()
