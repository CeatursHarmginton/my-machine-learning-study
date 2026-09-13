"""
Project discovery and change detection.

Scans the monorepo recursively, identifies leaf project directories,
and compares file hashes against the persisted state to detect new or
changed projects.
"""
from pathlib import Path

from config import (
    ProjectInfo,
    ProjectStatus,
    IGNORED_DIRS,
    PROJECT_MARKERS,
    MODEL_EXTS,
    AMBIGUOUS_MODEL_EXTS,
    DATASET_EXTS,
    DATASET_DIRS,
    ROOT_PATH,
)
from classifier import classify_project
from state import MonorepoState


# ── Helpers ───────────────────────────────────────────────────────────

_IGNORED_LOWER = {d.lower() for d in IGNORED_DIRS}
_DATASET_DIRS_LOWER = {d.lower() for d in DATASET_DIRS}
_DISCOVERY_SKIP_LOWER = _IGNORED_LOWER | _DATASET_DIRS_LOWER


def _should_skip(dirname: str) -> bool:
    """Return True if this directory name should be skipped."""
    return dirname.lower() in _DISCOVERY_SKIP_LOWER or dirname.startswith(".")


def _is_python_package(path: Path) -> bool:
    """True if the directory is a Python package (has __init__.py)."""
    return (path / "__init__.py").exists()


def is_project_dir(path: Path) -> bool:
    """A directory is a project if it directly contains source files or markers.

    Directories that are pure Python packages (__init__.py with no other
    top-level scripts) are NOT treated as standalone projects — they are
    sub-packages of a parent project.
    """
    has_init = False
    has_other_source = False

    try:
        for item in path.iterdir():
            if not item.is_file():
                continue
            name = item.name
            if name in PROJECT_MARKERS:
                return True
            if name == "__init__.py":
                has_init = True
                continue
            if item.suffix.lower() in {".py", ".ipynb"}:
                has_other_source = True
    except PermissionError:
        pass

    # A package that only has __init__.py → not a project
    if has_init and not has_other_source:
        return False

    return has_other_source


# ── Discovery ─────────────────────────────────────────────────────────

def discover_projects(root: Path) -> list[Path]:
    """Recursively discover project directories.

    Strategy (bottom-up):
    1. Skip ignored directories and Python packages (__init__.py).
    2. If a directory is a project and has NO child projects → leaf, add it.
    3. If a directory is a project and HAS child projects → keep the PARENT
       and discard the children (absorb them).  This prevents Lab7 from
       exploding into 14 sub-projects when each notebook sits in its own
       folder.
    4. If a directory owns a dataset dir plus child projects → keep the parent.
    5. If a directory is NOT a project but children were found → keep children.
    """

    def _scan(directory: Path) -> list[Path]:
        """Return the list of projects discovered under *directory*."""
        if directory != root and _should_skip(directory.name):
            return []

        # Skip pure Python packages — they belong to the parent project
        if directory != root and _is_python_package(directory):
            return []

        # Enumerate child directories
        subdirs: list[Path] = []
        has_dataset_dir = False
        try:
            for item in sorted(directory.iterdir()):
                if not item.is_dir():
                    continue
                name_lower = item.name.lower()
                if name_lower in _DATASET_DIRS_LOWER:
                    has_dataset_dir = True
                elif not _should_skip(item.name):
                    subdirs.append(item)
        except PermissionError:
            return []

        # Recurse into children and collect their projects
        child_projects: list[Path] = []
        for subdir in subdirs:
            child_projects.extend(_scan(subdir))

        # Root is never a project itself
        if directory == root:
            return child_projects

        this_is_project = is_project_dir(directory)

        if (this_is_project or has_dataset_dir) and child_projects:
            # Parent owns project-level data; keep it with child source files.
            return [directory]
        elif this_is_project:
            # Leaf project, no children
            return [directory]
        else:
            # Not a project — pass child projects upward
            return child_projects

    return sorted(_scan(root))


# ── Change detection ──────────────────────────────────────────────────

# Directories to completely skip when walking (never enter them)
_SKIP_WALK_DIRS = _IGNORED_LOWER | {d.lower() for d in DATASET_DIRS}


def _walk_source_files(project_path: Path, root: Path):
    """Yield (Path, rel_str) for *source* files only.

    Uses os.walk with in-place directory pruning so we never descend
    into data / dataset / images / train / test / … directories,
    avoiding the cost of touching tens of thousands of image files.
    """
    import os

    for dirpath, dirnames, filenames in os.walk(project_path):
        # Prune ignored & dataset directories IN-PLACE so os.walk skips them
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in _SKIP_WALK_DIRS and not d.startswith(".")
        ]

        dp = Path(dirpath)
        for fname in filenames:
            fp = dp / fname
            ext = fp.suffix.lower()

            # Skip HF-bound files — they are not tracked in source state.
            if ext in MODEL_EXTS or ext in AMBIGUOUS_MODEL_EXTS or ext in DATASET_EXTS:
                continue

            rel_str = str(fp.relative_to(root)).replace("\\", "/")
            yield fp, rel_str


def detect_changes(
    project_path: Path,
    rel_path: str,
    state: MonorepoState,
    root: Path,
) -> tuple[ProjectStatus, list[str], dict[str, str]]:
    """Compare current *source* file hashes with saved state.

    Only SOURCE files are hashed — dataset & model files are skipped
    during change detection for performance (they can contain tens of
    thousands of small images).

    Returns (status, list_of_changed_source_files, current_hashes).
    """
    old_hashes = state.get_file_hashes(rel_path)
    current_hashes: dict[str, str] = {}
    changed_files: list[str] = []

    for filepath, rel_str in _walk_source_files(project_path, root):
        file_hash = MonorepoState.compute_hash(filepath)
        current_hashes[rel_str] = file_hash

        if rel_str not in old_hashes:
            changed_files.append(f"+ {rel_str}")
        elif old_hashes[rel_str] != file_hash:
            changed_files.append(f"~ {rel_str}")

    # Detect deleted files (only among previously hashed source files)
    for old_file in old_hashes:
        if _is_tracked_source_path(old_file) and old_file not in current_hashes:
            changed_files.append(f"- {old_file}")

    # Determine status
    if not old_hashes:
        status = ProjectStatus.NEW
    elif changed_files:
        status = ProjectStatus.CHANGED
    else:
        status = ProjectStatus.UNCHANGED

    return status, changed_files, current_hashes


def _is_tracked_source_path(rel_path: str) -> bool:
    """True for paths that belong in source hash state."""
    parts = [p.lower() for p in rel_path.replace("\\", "/").split("/")]
    if any(part in _SKIP_WALK_DIRS or part.startswith(".") for part in parts[:-1]):
        return False
    ext = Path(parts[-1]).suffix.lower()
    return ext not in MODEL_EXTS and ext not in AMBIGUOUS_MODEL_EXTS and ext not in DATASET_EXTS


# ── Public API ────────────────────────────────────────────────────────

def scan_projects(root: Path, state: MonorepoState) -> list[ProjectInfo]:
    """Discover all projects, classify their files, and detect changes."""
    project_dirs = discover_projects(root)
    projects: list[ProjectInfo] = []

    for proj_path in project_dirs:
        rel_path = str(proj_path.relative_to(root)).replace("\\", "/")
        name = proj_path.name

        status, changed_files, source_hashes = detect_changes(proj_path, rel_path, state, root)
        files = classify_project(proj_path, root)

        # Retrieve existing HF repo mappings from state
        proj_state = state.get_project(rel_path)
        hf_dataset = proj_state.get("hf_dataset_repo") if proj_state else None
        hf_model = proj_state.get("hf_model_repo") if proj_state else None

        if status == ProjectStatus.UNCHANGED:
            if files.dataset_count and not hf_dataset:
                status = ProjectStatus.CHANGED
                changed_files.append("~ HuggingFace dataset upload missing")
            if files.model_count and not hf_model:
                status = ProjectStatus.CHANGED
                changed_files.append("~ HuggingFace model upload missing")

        projects.append(ProjectInfo(
            name=name,
            rel_path=rel_path,
            abs_path=proj_path,
            status=status,
            files=files,
            changed_files=changed_files,
            source_hashes=source_hashes,
            hf_dataset_repo=hf_dataset,
            hf_model_repo=hf_model,
        ))

    return projects


def materialize_project_files(projects: list[ProjectInfo], root: Path) -> None:
    """Populate full dataset/model file lists after the user selects projects."""
    for project in projects:
        project.files = classify_project(project.abs_path, root, materialize_hf=True)
