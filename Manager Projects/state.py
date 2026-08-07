"""
Persistent state management for AI Monorepo Manager.

Tracks per-project file hashes, HuggingFace repo mappings, and publish history.
State is stored as JSON in .monorepo_state.json alongside the tool.
"""
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional


class MonorepoState:
    """Manages persistent state across tool runs."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.data: dict = {
            "last_updated": None,
            "projects": {},
            "settings": {},
        }
        self.load()

    # ── Load / Save ──────────────────────────────────────────────

    def load(self):
        """Load state from disk. Initialise empty state if file missing."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as fh:
                    self.data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass  # keep default empty state
        # Ensure required keys
        self.data.setdefault("projects", {})
        self.data.setdefault("settings", {})

    def save(self):
        """Persist current state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, ensure_ascii=False)

    # ── Project-level accessors ──────────────────────────────────

    def get_project(self, rel_path: str) -> Optional[dict]:
        """Return the stored state dict for a project, or None."""
        return self.data["projects"].get(rel_path)

    def update_project(
        self,
        rel_path: str,
        file_hashes: dict,
        hf_dataset_repo: Optional[str] = None,
        hf_model_repo: Optional[str] = None,
    ):
        """Update (or create) state for a project after publishing."""
        if rel_path not in self.data["projects"]:
            self.data["projects"][rel_path] = {}

        proj = self.data["projects"][rel_path]
        proj["file_hashes"] = file_hashes
        proj["last_published"] = datetime.now().isoformat()

        if hf_dataset_repo is not None:
            proj["hf_dataset_repo"] = hf_dataset_repo
        if hf_model_repo is not None:
            proj["hf_model_repo"] = hf_model_repo

    def get_file_hashes(self, rel_path: str) -> dict:
        """Return {relative_path: md5_hash} for a project's last-known state."""
        proj = self.get_project(rel_path)
        return proj.get("file_hashes", {}) if proj else {}

    # ── Hashing utility ──────────────────────────────────────────

    @staticmethod
    def compute_hash(filepath: Path) -> str:
        """Compute MD5 hex-digest of a file (empty string on error)."""
        md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    md5.update(chunk)
            return md5.hexdigest()
        except (OSError, IOError):
            return ""
