"""
Git operations for AI Monorepo Manager.

Handles:
  • .gitignore — adds managed section for dataset / model / cache patterns
  • .gitattributes — keeps LFS only for source docs, removes model/dataset LFS
  • Selective staging — only source files of selected projects
  • Commit & push
  • Untracking files that became gitignored
"""
import subprocess
from pathlib import Path

from config import (
    ProjectInfo,
    GITIGNORE_MARKER_START,
    GITIGNORE_MARKER_END,
    MANAGED_GITIGNORE_PATTERNS,
    MANAGED_LFS_PATTERNS,
)


class GitManager:
    """Manages all Git interactions for the monorepo."""

    def __init__(self, root_path: Path):
        self.root = root_path
        self.gitignore_path = root_path / ".gitignore"
        self.gitattributes_path = root_path / ".gitattributes"

    # ── Internal helpers ──────────────────────────────────────────

    def _run(self, *args, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command in the repo root."""
        return subprocess.run(
            ["git"] + list(args),
            cwd=str(self.root),
            capture_output=True,
            text=True,
            check=check,
            encoding="utf-8",
            errors="replace",
        )

    # ── .gitignore ────────────────────────────────────────────────

    def update_gitignore(self):
        """Insert or replace the managed section in .gitignore."""
        existing = ""
        if self.gitignore_path.exists():
            existing = self.gitignore_path.read_text(encoding="utf-8")

        # Strip old managed section if present
        if GITIGNORE_MARKER_START in existing:
            before = existing[: existing.index(GITIGNORE_MARKER_START)]
            if GITIGNORE_MARKER_END in existing:
                after_idx = existing.index(GITIGNORE_MARKER_END) + len(GITIGNORE_MARKER_END)
                after = existing[after_idx:]
            else:
                after = ""
            existing = before.rstrip("\n") + after.lstrip("\n")

        # Build the new managed block
        managed_block = "\n".join(
            ["", GITIGNORE_MARKER_START]
            + MANAGED_GITIGNORE_PATTERNS
            + [GITIGNORE_MARKER_END, ""]
        )

        new_content = existing.rstrip("\n") + "\n" + managed_block
        self.gitignore_path.write_text(new_content, encoding="utf-8")

    # ── .gitattributes ────────────────────────────────────────────

    def update_gitattributes(self):
        """Rewrite .gitattributes with LFS rules for source docs only.

        Model / dataset extensions are removed from LFS because they
        will no longer be pushed to GitHub.
        """
        content = "\n".join(MANAGED_LFS_PATTERNS) + "\n"
        self.gitattributes_path.write_text(content, encoding="utf-8")

    # ── Untrack now-ignored files ─────────────────────────────────

    def untrack_ignored_files(self) -> list[str]:
        """Remove from the Git index any tracked files that are now gitignored.

        Returns a list of files that were untracked.
        """
        result = self._run("ls-files", "-ci", "--exclude-standard", check=False)
        removed: list[str] = []

        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                fname = line.strip()
                if fname:
                    self._run("rm", "--cached", "--", fname, check=False)
                    removed.append(fname)

        return removed

    # ── Staging ───────────────────────────────────────────────────

    def stage_files(self, projects: list[ProjectInfo]):
        """Stage source files of selected projects + .gitignore + .gitattributes."""
        # Always stage the config files we updated
        self._run("add", "--", ".gitignore", check=False)
        self._run("add", "--", ".gitattributes", check=False)

        # Stage each project's source files
        for proj in projects:
            for src_file in proj.files.source:
                src_path = self.root / src_file
                if src_path.exists():
                    self._run("add", "--", src_file, check=False)

        # Stage the Manager Projects tool itself (except state file)
        manager_rel = "Manager Projects"
        result = self._run(
            "ls-files", "--others", "--exclude-standard", "--", manager_rel,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                fname = line.strip()
                if fname and ".monorepo_state.json" not in fname:
                    self._run("add", "--", fname, check=False)

        # Also re-add modified tracked files in Manager Projects
        result = self._run(
            "ls-files", "--modified", "--", manager_rel,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                fname = line.strip()
                if fname and ".monorepo_state.json" not in fname:
                    self._run("add", "--", fname, check=False)

    # ── Commit & push ─────────────────────────────────────────────

    def commit_and_push(
        self,
        projects: list[ProjectInfo],
        message: str | None = None,
    ) -> bool:
        """Commit staged changes and push to origin.

        Returns True if a commit was created.
        """
        # Build default commit message
        if not message:
            names = ", ".join(p.name for p in projects[:3])
            if len(projects) > 3:
                names += f" (+{len(projects) - 3} more)"
            message = f"[publish] Update {names}"

        # Bail out if nothing staged
        diff = self._run("diff", "--cached", "--name-only", check=False)
        if not diff.stdout.strip():
            return False

        self._run("commit", "-m", message)
        self._run("push", "origin", check=False)
        return True
