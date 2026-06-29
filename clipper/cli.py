"""clipper CLI (§5).

    clipper run <youtube_url> --i-have-rights [--max-clips N] [--min-score 0.6]
                              [--formats both|vertical|horizontal] [--no-upload]
    clipper auth          # one-time YouTube OAuth
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import Enum

import typer
from rich.console import Console

from .config import get_config, run_path
from .logging_utils import setup_run_logging

app = typer.Typer(add_completion=False, help="Long-form YouTube -> short clip pipeline.")
console = Console()


class Formats(str, Enum):
    both = "both"
    vertical = "vertical"
    horizontal = "horizontal"


def _make_run_id(url: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{stamp}-{h}"


def _looks_like_url(url: str) -> bool:
    return bool(re.match(r"^https?://", url.strip()))


@app.command()
def run(
    youtube_url: str = typer.Argument(..., help="YouTube video URL to clip."),
    i_have_rights: bool = typer.Option(
        False, "--i-have-rights",
        help="Confirm you own / are licensed for / have fair-use rights to this content (§1).",
    ),
    max_clips: int = typer.Option(None, "--max-clips", help="Override max clips (config default)."),
    min_score: float = typer.Option(None, "--min-score", help="Override score threshold."),
    formats: Formats = typer.Option(None, "--formats", help="both | vertical | horizontal."),
    no_upload: bool = typer.Option(
        False, "--no-upload", help="Stop after rendering; do not upload (§4.8)."
    ),
):
    """Run the full pipeline on a YouTube URL."""
    if not _looks_like_url(youtube_url):
        console.print("[red]Error:[/red] argument does not look like an http(s) URL.")
        raise typer.Exit(2)

    # Ingest gate (§1): explicit flag OR interactive y/N confirmation. Abort if absent.
    rights = i_have_rights
    if not rights:
        console.print(
            "[yellow]Rights check (§9):[/yellow] Only process content you own, are "
            "licensed for, or that is fair use with added transformative commentary."
        )
        rights = typer.confirm("Do you have the rights to process this video?", default=False)
    if not rights:
        console.print("[red]Aborted:[/red] rights not confirmed.")
        raise typer.Exit(1)

    # Apply CLI overrides onto the loaded config (CLI wins over config.yaml, §6).
    cfg = get_config()
    if max_clips is not None:
        cfg.highlight.max_clips = max_clips
    if min_score is not None:
        cfg.highlight.min_score = min_score
    fmt_value = formats.value if formats is not None else cfg.output.formats

    run_id = _make_run_id(youtube_url)
    rdir = run_path(run_id)
    setup_run_logging(rdir)
    console.print(f"[bold green]clipper[/bold green] run_id=[cyan]{run_id}[/cyan]")
    console.print(f"artifacts: {rdir}")

    from .graph import get_graph

    initial_state = {
        "run_id": run_id,
        "source_url": youtube_url,
        "rights_confirmed": True,
        "formats": fmt_value,
        "no_upload": no_upload,
        "clips": [],
        "errors": [],
    }

    graph = get_graph()
    final_state = graph.invoke(initial_state, config={"recursion_limit": 200})

    clips = final_state.get("clips", [])
    console.print(f"\n[bold]Done.[/bold] {len(clips)} clip(s).")
    console.print(f"Report: {rdir / 'report.md'}")
    uploaded = sum(1 for c in clips if c.get("upload_status") == "uploaded")
    if uploaded:
        console.print(f"[green]{uploaded}[/green] uploaded as PRIVATE drafts — review before publishing.")


@app.command()
def auth():
    """One-time YouTube OAuth (§5). Caches a token for `upload_drafts`."""
    from .youtube.auth import run_oauth_flow

    console.print("Starting YouTube OAuth installed-app flow...")
    run_oauth_flow()
    console.print("[green]Authorized.[/green] Token cached. You can now upload private drafts.")


def main():  # console-script entry point
    app()


if __name__ == "__main__":
    main()
