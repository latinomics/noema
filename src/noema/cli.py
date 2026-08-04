"""NOEMA CLI (BLUEPRINT §6). Every command is resumable via the manifest
cache; --force busts it. Pipeline commands land with their milestones."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from noema.kernel.config import Settings
from noema.kernel.manifest import Manifest

app = typer.Typer(
    name="noema",
    help="NOEMA - books -> domain ontologies -> agents with a point of view.",
    no_args_is_help=True,
)
console = Console()


def _not_yet(command: str, milestone: str) -> None:
    console.print(
        f"[yellow]`noema {command}` lands with {milestone}. The kernel (M0) is live; "
        f"see BLUEPRINT.md for the build order.[/yellow]"
    )
    raise typer.Exit(code=1)


@app.command()
def status(limit: int = typer.Option(50, help="Max stage rows to show.")) -> None:
    """Lineage view: what's cached, what's stale, spend."""
    settings = Settings.load()
    settings.paths.ensure()
    manifest = Manifest(settings)
    info = manifest.status(limit=limit)

    table = Table(title="stage cache", show_lines=False)
    for col in ("stage", "key", "status", "created_at", "duration_s"):
        table.add_column(col)
    for row in info["stages"]:
        table.add_row(
            str(row["stage"]),
            str(row["key"]),
            str(row["status"]),
            str(row["created_at"])[:19],
            f"{row['duration_s']:.2f}" if row["duration_s"] is not None else "-",
        )
    if info["stages"]:
        console.print(table)
    else:
        console.print("[dim]no stages recorded yet[/dim]")
    console.print(
        f"llm calls: [bold]{info['llm_calls']}[/bold]   "
        f"spend: [bold]${info['spend_usd']:.4f}[/bold]"
    )


@app.command()
def ingest(pdfs: list[Path]) -> None:
    """Register PDFs + run P1 (DOCFORGE)."""
    _not_yet("ingest", "M1 (DOCFORGE)")


@app.command()
def extract(src_id: str) -> None:
    """Run P2 (ONTOGEN) for one source."""
    _not_yet("extract", "M2 (ONTOGEN)")


@app.command()
def synthesize(
    frames: int = typer.Option(3, help="Number of synthetic_polymath frames."),
    authors: bool = typer.Option(False, "--authors", help="Also build author simulacra."),
) -> None:
    """Run P3 (NOESIS) fusion + intentional stance modeling."""
    _not_yet("synthesize", "M3/M4 (NOESIS)")


@app.command()
def gym(frame_id: str) -> None:
    """Interrogate a compiled frame (consistency / humility / transfer / tension)."""
    _not_yet("gym", "M4 (gym)")


@app.command()
def export(
    frame_id: str,
    target: str = typer.Option("claude-agent-sdk", help="claude-agent-sdk | langgraph | mcp"),
) -> None:
    """Export a frame bundle for an agent runtime."""
    _not_yet("export", "M4 (frame compiler)")


if __name__ == "__main__":
    app()
