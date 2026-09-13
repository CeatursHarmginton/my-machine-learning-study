"""
AI Monorepo Manager — main entry point.

Orchestrates the full publish flow:
  1.  Load persistent state
  2.  Scan for projects & detect changes
  3.  Show project table
  4.  Let user select projects to publish
  5.  Show file classification per project
  6.  Confirm
  7.  Upload datasets → HuggingFace Dataset Repo
  8.  Upload models  → HuggingFace Model Repo
  9.  Update .gitignore & .gitattributes
  10. Stage source files, commit & push → GitHub
  11. Save state
  12. Show summary
"""
import sys
from pathlib import Path

# Ensure our package is importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ROOT_PATH, STATE_FILE, ProjectStatus
from state import MonorepoState
from scanner import scan_projects, materialize_project_files
from git_manager import GitManager
from hf_uploader import HFUploader
from ui import MonorepoUI


def main():
    ui = MonorepoUI()

    try:
        # ── 1. Header ────────────────────────────────────────────
        ui.show_header()

        # ── 2. Load state ────────────────────────────────────────
        ui.log_info("Loading state…")
        state = MonorepoState(STATE_FILE)

        # ── 3. Scan projects ─────────────────────────────────────
        ui.log_step("🔍 Scanning Projects")
        projects = scan_projects(ROOT_PATH, state)

        if not projects:
            ui.log_warning("No projects found in the monorepo!")
            return

        # ── 4. Show table ────────────────────────────────────────
        ui.show_projects_table(projects)

        publishable = [p for p in projects if p.status != ProjectStatus.UNCHANGED]
        if not publishable:
            ui.log_success("All projects are up to date — nothing to publish.")
            return

        ui.log_info(
            f"Found {len(publishable)} new/changed project(s) "
            f"out of {len(projects)} total."
        )

        # ── 5. Select projects ───────────────────────────────────
        selected = ui.select_projects(projects)
        if not selected:
            ui.log_warning("No projects selected. Exiting.")
            return

        # ── 6. Ask upload mode (zip vs individual) ───────────────
        has_hf_files = any(
            p.files.dataset_count or p.files.model_count for p in selected
        )
        use_zip = True
        if has_hf_files:
            use_zip = ui.ask_zip_mode()

        materialize_project_files(selected, ROOT_PATH)

        # ── 6b. Show classification ──────────────────────────────
        ui.log_step("📋 File Classification")
        ui.show_classification(selected)

        # ── 7. Confirm ───────────────────────────────────────────
        if not ui.confirm_publish(selected, use_zip=use_zip):
            ui.log_warning("Cancelled by user.")
            return

        # ── 8. Initialise managers ───────────────────────────────
        git = GitManager(ROOT_PATH)
        hf = HFUploader(ROOT_PATH, state)
        results: dict[str, dict] = {}

        # ── 9. HuggingFace authentication ────────────────────────
        hf_authenticated = False

        if has_hf_files:
            ui.log_step("🔑 HuggingFace Authentication")
            if hf.check_auth():
                ui.log_success(f"Authenticated as: {hf.get_username()}")
                hf_authenticated = True
            else:
                ui.log_warning(
                    "Not authenticated with HuggingFace. "
                    "Dataset/model uploads will be skipped."
                )
                ui.log_info(
                    "Run [bold]huggingface-cli login[/bold] to set up your token."
                )

        # ── 10. Upload to HuggingFace ─────────────────────────────
        if has_hf_files and hf_authenticated:
            ui.log_step("☁️  Uploading to HuggingFace")

            for i, proj in enumerate(selected, 1):
                results[proj.rel_path] = {}

                # Datasets
                if proj.files.dataset:
                    ui.log_info(
                        f"[{i}/{len(selected)}] {proj.name} — "
                        f"📊 {len(proj.files.dataset):,} dataset file(s)"
                    )
                    try:
                        repo_id = hf.upload_datasets(proj, ui.console, use_zip=use_zip)
                        if repo_id:
                            results[proj.rel_path]["hf_dataset"] = repo_id
                            proj.hf_dataset_repo = repo_id
                            ui.log_success(
                                f"Datasets → [link=https://huggingface.co/datasets/{repo_id}]{repo_id}[/link]"
                            )
                    except Exception as exc:
                        ui.log_error(
                            f"Dataset upload failed for {proj.name}: {exc}"
                        )

                # Models
                if proj.files.model:
                    ui.log_info(
                        f"[{i}/{len(selected)}] {proj.name} — "
                        f"🧠 {len(proj.files.model):,} model file(s)"
                    )
                    try:
                        repo_id = hf.upload_models(proj, ui.console, use_zip=use_zip)
                        if repo_id:
                            results[proj.rel_path]["hf_model"] = repo_id
                            proj.hf_model_repo = repo_id
                            ui.log_success(
                                f"Models → [link=https://huggingface.co/{repo_id}]{repo_id}[/link]"
                            )
                    except Exception as exc:
                        ui.log_error(
                            f"Model upload failed for {proj.name}: {exc}"
                        )

        else:
            for proj in selected:
                results[proj.rel_path] = {}

        # ── 11. Git operations ────────────────────────────────────
        ui.log_step("📦 Publishing to GitHub")

        ui.log_info("Updating .gitignore…")
        git.update_gitignore()

        ui.log_info("Updating .gitattributes (LFS for source docs only)…")
        git.update_gitattributes()

        ui.log_info("Removing tracked files that are now gitignored…")
        removed = git.untrack_ignored_files()
        if removed:
            ui.log_info(f"Untracked {len(removed)} file(s) from Git index.")

        ui.log_info("Staging source files…")
        git.stage_files(selected)

        ui.log_info("Committing & pushing…")
        pushed = git.commit_and_push(selected)

        for proj in selected:
            results.setdefault(proj.rel_path, {})["github"] = pushed

        if pushed:
            ui.log_success("Changes committed and pushed to GitHub! 🎉")
        else:
            ui.log_warning("No source changes to commit.")

        # ── 12. Save state ────────────────────────────────────────
        ui.log_step("💾 Saving State")

        for proj in selected:
            state.update_project(
                proj.rel_path,
                proj.source_hashes,
                hf_dataset_repo=proj.hf_dataset_repo,
                hf_model_repo=proj.hf_model_repo,
            )

        state.save()
        ui.log_success("State saved.")

        # ── 13. Summary ──────────────────────────────────────────
        ui.show_summary(results)

    except KeyboardInterrupt:
        ui.console.print("\n[yellow]⏹ Interrupted by user.[/yellow]")
        sys.exit(1)
    except Exception as exc:
        ui.log_error(f"Unexpected error: {exc}")
        ui.console.print_exception(show_locals=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
