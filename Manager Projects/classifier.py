"""
File classifier — decides whether each file is SOURCE, DATASET, or MODEL.

Priority chain:
  1. Model extension  → MODEL
  2. Inside MODEL_DIRS with ambiguous ext → MODEL
  3. Dataset extension → DATASET
  4. Inside DATASET_DIRS → DATASET
  5. Ambiguous ext outside model dirs → MODEL (safer to keep off GitHub)
  6. Everything else → SOURCE
"""
from pathlib import Path

from config import (
    FileType,
    ClassifiedFiles,
    DATASET_DIRS,
    DATASET_EXTS,
    MODEL_EXTS,
    MODEL_DIRS,
    AMBIGUOUS_MODEL_EXTS,
    IGNORED_DIRS,
)


def classify_file(filepath: Path, project_root: Path) -> FileType:
    """Classify a single file relative to its project root."""
    try:
        rel = filepath.relative_to(project_root)
    except ValueError:
        return FileType.SOURCE

    parts_lower = [p.lower() for p in rel.parts]
    ext = filepath.suffix.lower()

    # ── 1. Definite model extensions ────────────────────────────
    if ext in MODEL_EXTS:
        return FileType.MODEL

    # ── 2. Ambiguous ext inside model directories → MODEL ──────
    if ext in AMBIGUOUS_MODEL_EXTS:
        if any(p in MODEL_DIRS for p in parts_lower[:-1]):
            return FileType.MODEL
        # Outside model dirs → still treat as model (keep off GitHub)
        return FileType.MODEL

    # ── 3. Definite dataset extensions ──────────────────────────
    if ext in DATASET_EXTS:
        return FileType.DATASET

    # ── 4. Inside dataset directories ───────────────────────────
    if any(p in DATASET_DIRS for p in parts_lower[:-1]):
        return FileType.DATASET
    # Also match if the file is directly inside a dataset dir
    if len(parts_lower) >= 2 and parts_lower[0] in DATASET_DIRS:
        return FileType.DATASET

    # ── 5. Default → SOURCE ─────────────────────────────────────
    return FileType.SOURCE


def classify_project(project_path: Path, root_path: Path) -> ClassifiedFiles:
    """Walk the project and classify files into source / dataset / model.

    Uses os.walk with directory pruning so we never descend into dataset
    directories (which can contain tens of thousands of image files).
    Instead, dataset dirs are handled by a fast file-count helper.
    """
    import os

    result = ClassifiedFiles()
    ignored_lower = {d.lower() for d in IGNORED_DIRS}
    dataset_dirs_lower = {d.lower() for d in DATASET_DIRS}

    # ── Phase 1: collect dataset files from dataset directories ──
    def _collect_dataset_dir(dpath: Path):
        """Recursively list every file inside a dataset directory."""
        try:
            for item in os.scandir(dpath):
                if item.is_file(follow_symlinks=False):
                    fp = Path(item.path)
                    rel_str = str(fp.relative_to(root_path)).replace("\\", "/")
                    result.dataset.append(rel_str)
                elif item.is_dir(follow_symlinks=False):
                    if item.name.lower() not in ignored_lower:
                        _collect_dataset_dir(fp := Path(item.path))
        except PermissionError:
            pass

    # ── Phase 2: walk non-dataset directories for source & model ─
    for dirpath, dirnames, filenames in os.walk(project_path):
        dp = Path(dirpath)

        # Separate dataset dirs from the rest
        dataset_subdirs = []
        keep_dirs = []
        for d in dirnames:
            dl = d.lower()
            if dl in ignored_lower or d.startswith("."):
                continue          # skip entirely
            if dl in dataset_dirs_lower:
                dataset_subdirs.append(d)
            else:
                keep_dirs.append(d)

        # Prune in-place: only descend into non-dataset, non-ignored dirs
        dirnames[:] = keep_dirs

        # Collect dataset dirs without full traversal overhead
        for dd in dataset_subdirs:
            _collect_dataset_dir(dp / dd)

        # Classify regular files in this directory
        for fname in filenames:
            fp = dp / fname
            file_type = classify_file(fp, project_path)
            rel_str = str(fp.relative_to(root_path)).replace("\\", "/")

            if file_type == FileType.SOURCE:
                result.source.append(rel_str)
            elif file_type == FileType.DATASET:
                result.dataset.append(rel_str)
            elif file_type == FileType.MODEL:
                result.model.append(rel_str)

    return result

