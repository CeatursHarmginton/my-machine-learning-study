"""
HuggingFace upload manager for datasets and models.

Creates HF repos on-demand and uploads files while preserving the
project's internal directory structure.  Supports uploading as a
single zip archive (default) or individual files.
"""
import os
import re
import zipfile
import tempfile
from pathlib import Path
from typing import Optional

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    FileSizeColumn,
    TransferSpeedColumn,
)

from config import ProjectInfo, ZIP_UPLOAD_DEFAULT
from state import MonorepoState


# File types that are already compressed — ZIP_STORED skips futile
# re-compression, giving a 10–50× speedup for image-heavy datasets.
_PRECOMPRESSED_EXTS = frozenset({
    # Images
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".ico", ".tiff",
    # Video / audio
    ".mp4", ".avi", ".mov", ".mkv", ".mp3", ".wav", ".flac", ".ogg",
    # Archives
    ".zip", ".gz", ".bz2", ".xz", ".7z", ".rar", ".tar",
    # ML model weights (already dense binary)
    ".pth", ".pt", ".onnx", ".safetensors", ".bin", ".ckpt",
    # Binary data formats
    ".npy", ".npz", ".parquet", ".arrow", ".hdf5", ".h5",
    ".pkl", ".joblib",
})


class HFUploader:
    """Handles dataset and model uploads to Hugging Face Hub."""

    def __init__(self, root_path: Path, state: MonorepoState):
        self.root = root_path
        self.state = state
        self._api = None
        self._username: Optional[str] = None

    # ── Lazy API initialisation ───────────────────────────────────

    @property
    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi()
        return self._api

    # ── Authentication ────────────────────────────────────────────

    def check_auth(self) -> bool:
        """Verify that we have a valid HuggingFace token."""
        try:
            info = self.api.whoami()
            self._username = info.get("name", info.get("fullname", "unknown"))
            return True
        except Exception:
            return False

    def get_username(self) -> str:
        if not self._username:
            self.check_auth()
        return self._username or "unknown"

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _slugify(name: str) -> str:
        """Convert a project name to a URL-safe slug."""
        slug = re.sub(r"[^\w\s-]", "", name.lower())
        slug = re.sub(r"[\s_]+", "-", slug)
        return slug.strip("-") or "project"

    def _ensure_repo(self, repo_id: str, repo_type: str) -> str:
        """Create the HF repo if it doesn't already exist."""
        try:
            self.api.repo_info(repo_id=repo_id, repo_type=repo_type)
        except Exception:
            self.api.create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                exist_ok=True,
            )
        return repo_id

    # ── Zip helper ────────────────────────────────────────────────

    def _create_zip(
        self,
        file_list: list[str],
        project: ProjectInfo,
        label: str,
        console=None,
    ) -> Optional[Path]:
        """Create a zip archive of files.

        Strategy:
          1. Try native ``tar`` (Windows 10+ built-in) — 10–50× faster
             than Python for many small files.
          2. Fall back to Python ``zipfile`` with smart compression.
        """
        slug = self._slugify(project.name)
        tmp_dir = tempfile.mkdtemp(prefix="monorepo_")
        zip_path = Path(tmp_dir) / f"{slug}-{label}.zip"

        total = len(file_list)
        if total == 0:
            os.rmdir(tmp_dir)
            return None

        # Resolve absolute paths and archive names
        entries: list[tuple[Path, str]] = []
        for rel_path in file_list:
            filepath = self.root / rel_path
            if not filepath.exists():
                continue
            try:
                arcname = str(
                    filepath.relative_to(project.abs_path)
                ).replace("\\", "/")
            except ValueError:
                arcname = filepath.name
            entries.append((filepath, arcname))

        if not entries:
            os.rmdir(tmp_dir)
            return None
        total = len(entries)

        # ── Try native zip (via tar.exe or 7z) ───────────────────
        if total >= 100:
            native_ok = self._try_native_zip(
                entries, project.abs_path, zip_path, label, total, console
            )
            if native_ok:
                return zip_path

        # ── Fallback: Python zipfile ──────────────────────────────
        return self._python_zip(entries, zip_path, label, total, console)

    def _try_native_zip(
        self,
        entries: list[tuple[Path, str]],
        project_root: Path,
        zip_path: Path,
        label: str,
        total: int,
        console=None,
    ) -> bool:
        """Attempt to create zip using native tar.exe (Windows 10+).

        Returns True if successful, False to trigger fallback.
        """
        import subprocess
        import shutil

        tar_bin = shutil.which("tar")
        if not tar_bin:
            return False

        if console:
            console.print(
                f"  [dim]⚡ Using native tar for {total:,} files "
                f"(~10× faster)[/dim]"
            )

        try:
            # Write a file list (manifest) to avoid command-line length limits
            manifest = zip_path.parent / "_manifest.txt"
            with open(manifest, "w", encoding="utf-8") as fh:
                for filepath, arcname in entries:
                    # tar needs paths relative to the -C directory
                    fh.write(arcname + "\n")

            # tar -a -cf output.zip -C <project_root> -T manifest.txt
            #   -a  = auto-detect archive format from extension (.zip)
            #   -C  = change to project dir (so paths are relative)
            #   -T  = read file list from manifest
            with console.status(
                f"[bold cyan]📦 Zipping {label} ({total:,} files) via tar…[/bold cyan]"
            ) if console else _nullcontext():
                result = subprocess.run(
                    [
                        tar_bin, "-a", "-cf", str(zip_path),
                        "-C", str(project_root),
                        "-T", str(manifest),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 min max
                )

            manifest.unlink(missing_ok=True)

            if result.returncode != 0:
                if console:
                    console.print(
                        f"  [yellow]⚠ tar failed, falling back to Python zip[/yellow]"
                    )
                zip_path.unlink(missing_ok=True)
                return False

            if zip_path.exists() and zip_path.stat().st_size > 0:
                size_mb = zip_path.stat().st_size / (1024 * 1024)
                if console:
                    console.print(
                        f"  [green]✅ Zip ready: {zip_path.name} "
                        f"({total:,} files, {size_mb:.1f} MB)[/green]"
                    )
                return True

            return False

        except Exception:
            zip_path.unlink(missing_ok=True)
            return False

    def _python_zip(
        self,
        entries: list[tuple[Path, str]],
        zip_path: Path,
        label: str,
        total: int,
        console=None,
    ) -> Optional[Path]:
        """Create zip using Python zipfile with smart compression + progress."""
        progress = Progress(
            SpinnerColumn("dots"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=35, complete_style="bright_green"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )

        count = 0
        with progress:
            task = progress.add_task(
                f"📦 Zipping {label} ({total:,} files)", total=total
            )

            with zipfile.ZipFile(zip_path, "w") as zf:
                for filepath, arcname in entries:
                    ext = filepath.suffix.lower()
                    method = (
                        zipfile.ZIP_STORED
                        if ext in _PRECOMPRESSED_EXTS
                        else zipfile.ZIP_DEFLATED
                    )
                    zf.write(filepath, arcname, compress_type=method)
                    count += 1
                    progress.advance(task)

        if count == 0:
            zip_path.unlink(missing_ok=True)
            os.rmdir(zip_path.parent)
            return None

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        if console:
            console.print(
                f"  [green]✅ Zip ready: {zip_path.name} "
                f"({count:,} files, {size_mb:.1f} MB)[/green]"
            )

        return zip_path

    @staticmethod
    def _cleanup_zip(zip_path: Optional[Path]):
        """Delete the zip file and its temp directory."""
        if zip_path and zip_path.exists():
            parent = zip_path.parent
            zip_path.unlink(missing_ok=True)
            try:
                os.rmdir(parent)
            except OSError:
                pass

    # ── Upload datasets ───────────────────────────────────────────

    def upload_datasets(
        self,
        project: ProjectInfo,
        console=None,
        use_zip: bool = ZIP_UPLOAD_DEFAULT,
    ) -> Optional[str]:
        """Upload dataset files for a project. Returns the repo ID or None."""
        if not project.files.dataset:
            return None

        username = self.get_username()
        slug = self._slugify(project.name)

        repo_id = project.hf_dataset_repo or f"{username}/{slug}-dataset"
        repo_id = self._ensure_repo(repo_id, "dataset")

        if use_zip:
            return self._upload_as_zip(
                project, repo_id, "dataset", project.files.dataset, console
            )
        else:
            return self._upload_individually(
                project, repo_id, "dataset", project.files.dataset, console
            )

    # ── Upload models ─────────────────────────────────────────────

    def upload_models(
        self,
        project: ProjectInfo,
        console=None,
        use_zip: bool = ZIP_UPLOAD_DEFAULT,
    ) -> Optional[str]:
        """Upload model files for a project. Returns the repo ID or None."""
        if not project.files.model:
            return None

        username = self.get_username()
        slug = self._slugify(project.name)

        repo_id = project.hf_model_repo or f"{username}/{slug}-model"
        repo_id = self._ensure_repo(repo_id, "model")

        if use_zip:
            return self._upload_as_zip(
                project, repo_id, "model", project.files.model, console
            )
        else:
            return self._upload_individually(
                project, repo_id, "model", project.files.model, console
            )

    # ── Internal upload strategies ────────────────────────────────

    def _upload_as_zip(
        self,
        project: ProjectInfo,
        repo_id: str,
        repo_type: str,
        file_list: list[str],
        console=None,
    ) -> Optional[str]:
        """Zip all files → upload single zip → delete zip."""
        zip_path = None
        try:
            # Phase 1: Create zip (has its own progress bar)
            zip_path = self._create_zip(file_list, project, repo_type, console)
            if not zip_path:
                return repo_id

            zip_name = zip_path.name
            size_mb = zip_path.stat().st_size / (1024 * 1024)

            # Phase 2: Upload with spinner
            if console:
                with console.status(
                    f"[bold cyan]📤 Uploading {zip_name} ({size_mb:.1f} MB)…[/bold cyan]"
                ):
                    self.api.upload_file(
                        path_or_fileobj=str(zip_path),
                        path_in_repo=zip_name,
                        repo_id=repo_id,
                        repo_type=repo_type,
                        commit_message=f"Upload {zip_name}",
                    )
                console.print(
                    f"  [green]✅ Uploaded {zip_name} ({size_mb:.1f} MB)[/green]"
                )
            else:
                self.api.upload_file(
                    path_or_fileobj=str(zip_path),
                    path_in_repo=zip_name,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    commit_message=f"Upload {zip_name}",
                )

            return repo_id

        except Exception as exc:
            if console:
                console.print(f"  [red]⚠ Zip upload failed: {exc}[/red]")
            raise

        finally:
            self._cleanup_zip(zip_path)
            if console and zip_path:
                console.print(f"  [dim]🗑️  Cleaned up temp zip[/dim]")

    def _upload_individually(
        self,
        project: ProjectInfo,
        repo_id: str,
        repo_type: str,
        file_list: list[str],
        console=None,
    ) -> Optional[str]:
        """Upload each file individually with progress."""
        total = len(file_list)

        progress = Progress(
            SpinnerColumn("dots"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=35, complete_style="bright_cyan"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) if console else None

        ctx = progress if progress else _nullcontext()
        failures: list[str] = []

        with ctx:
            task = progress.add_task(
                f"📤 Uploading {repo_type} files", total=total
            ) if progress else None

            for rel_path in file_list:
                filepath = self.root / rel_path
                if not filepath.exists():
                    if progress:
                        progress.advance(task)
                    continue

                try:
                    path_in_repo = str(
                        filepath.relative_to(project.abs_path)
                    ).replace("\\", "/")
                except ValueError:
                    path_in_repo = filepath.name

                if progress:
                    progress.update(task, description=f"📤 {path_in_repo}")

                try:
                    self.api.upload_file(
                        path_or_fileobj=str(filepath),
                        path_in_repo=path_in_repo,
                        repo_id=repo_id,
                        repo_type=repo_type,
                        commit_message=f"Update {path_in_repo}",
                    )
                except Exception as exc:
                    failures.append(f"{path_in_repo}: {exc}")
                    if console:
                        console.print(
                            f"  [red]⚠ Failed: {path_in_repo} — {exc}[/red]"
                        )

                if progress:
                    progress.advance(task)

        if failures:
            raise RuntimeError(f"{len(failures)} {repo_type} upload(s) failed")

        return repo_id


class _nullcontext:
    """Minimal no-op context manager for Python 3.10 compatibility."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
