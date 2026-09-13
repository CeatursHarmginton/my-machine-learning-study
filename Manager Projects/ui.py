"""
Rich terminal UI for AI Monorepo Manager.

Provides:
  • Styled header, project table, classification tree
  • Interactive multi-select via InquirerPy
  • Progress bars with spinners
  • Colour-coded logging helpers
  • Final summary table
"""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.tree import Tree
from rich import box

from config import ProjectInfo, ProjectStatus


console = Console()


class MonorepoUI:
    """All terminal I/O goes through this class."""

    def __init__(self):
        self.console = console

    # ── Header ────────────────────────────────────────────────────

    def show_header(self):
        title = Text()
        title.append("🚀 ", style="bold")
        title.append("AI Monorepo Manager", style="bold bright_cyan")
        self.console.print(
            Panel(
                title,
                subtitle="[dim]Publish · Sync · Organize[/dim]",
                border_style="bright_cyan",
                padding=(1, 4),
            )
        )
        self.console.print()

    # ── Project table ─────────────────────────────────────────────

    def show_projects_table(self, projects: list[ProjectInfo]):
        table = Table(
            title="📂 Discovered Projects",
            box=box.ROUNDED,
            title_style="bold magenta",
            header_style="bold cyan",
            show_lines=True,
            expand=True,
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Status", width=12)
        table.add_column("Project", style="bold white", ratio=3)
        table.add_column("Source", justify="right", style="green", width=8)
        table.add_column("Dataset", justify="right", style="yellow", width=9)
        table.add_column("Model", justify="right", style="red", width=8)
        table.add_column("Changes", justify="right", style="cyan", width=9)

        for idx, proj in enumerate(projects, 1):
            if proj.status == ProjectStatus.NEW:
                status_text = Text("🆕 NEW", style="bold green")
            elif proj.status == ProjectStatus.CHANGED:
                status_text = Text("📝 CHANGED", style="bold yellow")
            else:
                status_text = Text("✅ OK", style="dim")

            table.add_row(
                str(idx),
                status_text,
                proj.rel_path,
                str(proj.files.source_count),
                str(proj.files.dataset_count),
                str(proj.files.model_count),
                str(len(proj.changed_files)),
            )

        self.console.print(table)
        self.console.print()

    # ── Interactive project selection ─────────────────────────────

    def select_projects(self, projects: list[ProjectInfo]) -> list[ProjectInfo]:
        """Present checkboxes for new/changed projects. Returns selected list."""
        from InquirerPy import inquirer

        publishable = [p for p in projects if p.status != ProjectStatus.UNCHANGED]

        if not publishable:
            self.console.print("[yellow]ℹ️  No new or changed projects.[/yellow]")
            return []

        choices = []
        for p in publishable:
            icon = "🆕" if p.status == ProjectStatus.NEW else "📝"
            label = (
                f"{icon} {p.rel_path}  "
                f"({p.files.source_count} src, "
                f"{p.files.dataset_count} data, "
                f"{p.files.model_count} model)"
            )
            choices.append({"name": label, "value": p.rel_path})

        selected_paths: list[str] = inquirer.checkbox(
            message="Select projects to publish (Space = toggle, Enter = confirm):",
            choices=choices,
            validate=lambda r: len(r) > 0,
            invalid_message="Please select at least one project.",
        ).execute()

        return [p for p in publishable if p.rel_path in selected_paths]

    # ── Classification tree ───────────────────────────────────────

    def show_classification(self, projects: list[ProjectInfo]):
        """Print a tree view showing how each file is classified."""
        for proj in projects:
            tree = Tree(f"📁 [bold]{proj.rel_path}[/bold]")

            if proj.files.source:
                branch = tree.add(
                    f"[green]📄 Source ({proj.files.source_count} files) → GitHub[/green]"
                )
                for f in proj.files.source[:5]:
                    branch.add(f"[dim]{f}[/dim]")
                if proj.files.source_count > 5:
                    branch.add(f"[dim]… +{proj.files.source_count - 5} more[/dim]")

            if proj.files.dataset:
                branch = tree.add(
                    f"[yellow]📊 Dataset ({proj.files.dataset_count} files) → HuggingFace Dataset[/yellow]"
                )
                for f in proj.files.dataset[:5]:
                    branch.add(f"[dim]{f}[/dim]")
                if proj.files.dataset_count > 5:
                    branch.add(f"[dim]… +{proj.files.dataset_count - 5} more[/dim]")

            if proj.files.model:
                branch = tree.add(
                    f"[red]🧠 Model ({proj.files.model_count} files) → HuggingFace Model[/red]"
                )
                for f in proj.files.model[:5]:
                    branch.add(f"[dim]{f}[/dim]")
                if proj.files.model_count > 5:
                    branch.add(f"[dim]… +{proj.files.model_count - 5} more[/dim]")

            self.console.print(tree)
            self.console.print()

    # ── Upload mode selection ────────────────────────────────────

    def ask_zip_mode(self) -> bool:
        """Ask whether to upload as zip or individual files. Default: zip."""
        from InquirerPy import inquirer

        return inquirer.confirm(
            message="Upload to HuggingFace as ZIP? (No = upload individual files)",
            default=True,
        ).execute()

    # ── Confirm publish ───────────────────────────────────────────

    def confirm_publish(self, projects: list[ProjectInfo], use_zip: bool = True) -> bool:
        from InquirerPy import inquirer

        total_src = sum(p.files.source_count for p in projects)
        total_data = sum(p.files.dataset_count for p in projects)
        total_model = sum(p.files.model_count for p in projects)
        upload_mode = "📦 ZIP archive" if use_zip else "📄 Individual files"

        summary = (
            f"[bold]Publish Plan[/bold]\n"
            f"  📦 Projects:              {len(projects)}\n"
            f"  📄 Source files → GitHub:  {total_src}\n"
            f"  📊 Dataset files → HF:    {total_data}\n"
            f"  🧠 Model files → HF:      {total_model}\n"
            f"  ☁️  Upload mode:            {upload_mode}"
        )
        self.console.print(
            Panel(summary, border_style="bright_cyan", padding=(1, 2))
        )
        self.console.print()

        return inquirer.confirm(
            message="Proceed with publishing?", default=True
        ).execute()

    # ── Progress bar ──────────────────────────────────────────────

    def create_progress(self) -> Progress:
        return Progress(
            SpinnerColumn("dots"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40, complete_style="bright_cyan"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        )

    # ── Final summary ─────────────────────────────────────────────

    def show_summary(self, results: dict):
        table = Table(
            title="✨ Publish Summary",
            box=box.DOUBLE_EDGE,
            title_style="bold green",
            header_style="bold",
            show_lines=True,
        )
        table.add_column("Project", style="bold white")
        table.add_column("GitHub", justify="center", width=8)
        table.add_column("HF Dataset", style="yellow", min_width=20)
        table.add_column("Download", style="cyan", min_width=30)
        table.add_column("HF Model", style="red", min_width=20)

        for proj_name, info in results.items():
            table.add_row(
                proj_name,
                "✅" if info.get("github") else "⏭️",
                info.get("hf_dataset") or "—",
                info.get("hf_dataset_download") or "—",
                info.get("hf_model") or "—",
            )

        self.console.print()
        self.console.print(table)
        self.console.print()
        self.console.print("[bold green]🎉 All done![/bold green]")
        self.console.print()

    # ── Logging helpers ───────────────────────────────────────────

    def log_info(self, msg: str):
        self.console.print(f"[cyan]ℹ️  {msg}[/cyan]")

    def log_success(self, msg: str):
        self.console.print(f"[green]✅ {msg}[/green]")

    def log_warning(self, msg: str):
        self.console.print(f"[yellow]⚠️  {msg}[/yellow]")

    def log_error(self, msg: str):
        self.console.print(f"[bold red]❌ {msg}[/bold red]")

    def log_step(self, msg: str):
        self.console.print()
        self.console.rule(f"[bold bright_cyan]{msg}[/bold bright_cyan]", style="bright_cyan")
        self.console.print()
