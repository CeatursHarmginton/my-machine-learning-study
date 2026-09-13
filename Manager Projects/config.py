"""
Configuration, shared types, and constants for AI Monorepo Manager.
"""
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ──────────────────────────── Enums ────────────────────────────

class FileType(Enum):
    """Classification type for a file."""
    SOURCE = "source"
    DATASET = "dataset"
    MODEL = "model"


class ProjectStatus(Enum):
    """Status of a project relative to last publish."""
    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


# ──────────────────────────── Data Classes ────────────────────────────

@dataclass
class ClassifiedFiles:
    """Files in a project grouped by type."""
    source: list = field(default_factory=list)    # relative paths (str)
    dataset: list = field(default_factory=list)
    model: list = field(default_factory=list)
    source_count: int = 0
    dataset_count: int = 0
    model_count: int = 0
    materialized: bool = False


@dataclass
class ProjectInfo:
    """Metadata for a discovered project."""
    name: str                                       # Display name (e.g., "Number Recognition")
    rel_path: str                                   # Relative to ROOT_PATH (e.g., "More Project/Number Recognition")
    abs_path: Path                                  # Absolute path on disk
    status: ProjectStatus                           # NEW / CHANGED / UNCHANGED
    files: ClassifiedFiles = field(default_factory=ClassifiedFiles)
    changed_files: list = field(default_factory=list)  # list of changed relative paths
    source_hashes: dict = field(default_factory=dict)
    hf_dataset_repo: Optional[str] = None
    hf_model_repo: Optional[str] = None


# ──────────────────────────── Paths ────────────────────────────

ROOT_PATH = Path(r"D:\AI\Exercise")
MANAGER_DIR = ROOT_PATH / "Manager Projects"
STATE_FILE = MANAGER_DIR / ".monorepo_state.json"


# ──────────────────────────── File Classification Patterns ────────────────────────────

# Directories whose *contents* are treated as dataset files
DATASET_DIRS = {
    "data", "dataset", "datasets", "images",
    "train", "test", "val", "validation",
}

# File extensions always treated as dataset
DATASET_EXTS = {
    ".csv", ".tsv", ".parquet", ".arrow",
    ".npy", ".npz", ".hdf5",
}

# File extensions always treated as model weights
MODEL_EXTS = {
    ".pth", ".pt", ".onnx", ".ckpt", ".safetensors",
    ".pb", ".tflite", ".keras", ".h5", ".bin",
}

# Ambiguous extensions — classified as model only inside MODEL_DIRS, otherwise dataset
AMBIGUOUS_MODEL_EXTS = {".pkl", ".joblib"}

# Directories whose contents are treated as model files
MODEL_DIRS = {"models", "model", "checkpoints", "weights"}

# Recognised source-code / doc extensions (everything else defaults to SOURCE too)
SOURCE_EXTS = {
    ".py", ".ipynb", ".md", ".txt", ".rst",
    ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".sh", ".bat", ".ps1",
    ".html", ".css", ".js", ".jsx", ".ts", ".tsx",
    ".json", ".xml", ".r", ".rmd",
    ".pdf", ".docx", ".xlsx", ".xlsm", ".xls", ".pptx",
}


# ──────────────────────────── Git LFS ────────────────────────────

# Extensions that always use Git LFS when committed to GitHub
LFS_ALWAYS_EXTS = {".ipynb", ".pdf", ".docx", ".xlsx", ".xlsm", ".xls", ".pptx"}

# Source files larger than this threshold also use LFS (50 MB)
LFS_SIZE_THRESHOLD = 50 * 1024 * 1024

# Upload to HuggingFace as zip by default (True = zip, False = individual files)
ZIP_UPLOAD_DEFAULT = True


# ──────────────────────────── Scanning ────────────────────────────

# Directories to skip during project discovery
IGNORED_DIRS = {
    ".venv", "venv", "env",
    "__pycache__", ".ipynb_checkpoints",
    ".git", "node_modules",
    "outputs", "results",
    "Manager Projects", ".agents", ".monorepo_manager",
    ".cache", "wandb", "runs", "logs",
}

# Files whose presence signals "this directory is a project"
PROJECT_MARKERS = {
    "requirements.txt", "setup.py", "setup.cfg",
    "pyproject.toml", "Makefile", "Dockerfile",
    "app.py", "main.py",
}


# ──────────────────────────── .gitignore managed section ────────────────────────────

GITIGNORE_MARKER_START = "# === [Monorepo Manager] Auto-managed patterns ==="
GITIGNORE_MARKER_END   = "# === [/Monorepo Manager] ==="

MANAGED_GITIGNORE_PATTERNS = [
    "# Dataset files",
    "*.csv",
    "*.tsv",
    "*.parquet",
    "*.arrow",
    "*.npy",
    "*.npz",
    "*.hdf5",
    "**/data/",
    "**/dataset/",
    "**/datasets/",
    "**/images/",
    "**/train/",
    "**/test/",
    "**/val/",
    "**/validation/",
    "",
    "# Model files",
    "*.pth",
    "*.pt",
    "*.onnx",
    "*.ckpt",
    "*.safetensors",
    "*.pb",
    "*.tflite",
    "*.keras",
    "*.h5",
    "*.bin",
    "*.pkl",
    "*.joblib",
    "",
    "# Cache & temporary",
    "**/__pycache__/",
    "**/.ipynb_checkpoints/",
    "**/outputs/",
    "**/results/",
    "**/.cache/",
    "**/wandb/",
    "**/runs/",
    "**/logs/",
    ".venv/",
    "venv/",
    "env/",
    "",
    "# Manager state",
    "Manager Projects/.monorepo_state.json",
]


# ──────────────────────────── .gitattributes managed patterns ────────────────────────────

# LFS rules for source files only (datasets & models go to HF, not GitHub)
MANAGED_LFS_PATTERNS = [
    "*.ipynb filter=lfs diff=lfs merge=lfs -text",
    "*.pdf filter=lfs diff=lfs merge=lfs -text",
    "*.docx filter=lfs diff=lfs merge=lfs -text",
    "*.xlsx filter=lfs diff=lfs merge=lfs -text",
    "*.xlsm filter=lfs diff=lfs merge=lfs -text",
    "*.xls filter=lfs diff=lfs merge=lfs -text",
    "*.pptx filter=lfs diff=lfs merge=lfs -text",
]
