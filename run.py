"""One-file runner for clipper -- no CLI fiddling needed.

HOW TO USE
==========
1. Edit the SETTINGS block below (paste your link, tweak the options).
2. Run it:  double-click `run.bat`  (or:  .venv\\Scripts\\python.exe run.py)

Outputs land in  runs/<run_id>/  -- the rendered clips + report.md.
"""

# ============================ SETTINGS ============================
# Paste the full YouTube link between the quotes:
YOUTUBE_URL = "https://www.youtube.com/watch?v=zTxkGUG4TVI"

# How many clips to produce (max):
MAX_CLIPS = 3

# Only keep highlights scored at least this (0.0 - 1.0):
MIN_SCORE = 0.6

# Which renders to make:  "both" | "vertical" | "horizontal"
FORMATS = "both"

# False = just make the clip files (no YouTube, no login needed).
# True  = also upload as PRIVATE drafts (requires `clipper auth` first).
UPLOAD = False

# Rights gate (§9). You MUST set this to True to confirm you own / are licensed
# for / have fair-use rights to this content. The run aborts otherwise.
I_HAVE_RIGHTS = False
# ==================================================================


def main() -> int:
    from rich.console import Console

    from clipper.cli import _looks_like_url, _make_run_id
    from clipper.config import get_config, run_path
    from clipper.logging_utils import setup_run_logging

    console = Console()

    # --- validate settings --------------------------------------------------
    url = YOUTUBE_URL.strip().strip('"').strip("'")
    if not url or url == "PASTE_YOUR_LINK_HERE":
        console.print("[red]Edit run.py first:[/red] set YOUTUBE_URL to your video link.")
        return 2
    if not _looks_like_url(url):
        console.print(f"[red]That doesn't look like an http(s) URL:[/red] {url}")
        return 2
    if not I_HAVE_RIGHTS:
        console.print(
            "[red]Aborted (rights not confirmed).[/red] Open run.py and set "
            "I_HAVE_RIGHTS = True only if you own / are licensed for / have "
            "fair-use rights to this content (see README -> Operator responsibility)."
        )
        return 1
    if FORMATS not in ("both", "vertical", "horizontal"):
        console.print(f"[red]FORMATS must be both | vertical | horizontal, got:[/red] {FORMATS}")
        return 2

    # --- apply settings onto config -----------------------------------------
    cfg = get_config()
    cfg.highlight.max_clips = MAX_CLIPS
    cfg.highlight.min_score = MIN_SCORE

    run_id = _make_run_id(url)
    rdir = run_path(run_id)
    setup_run_logging(rdir)

    console.print(f"[bold green]clipper[/bold green] run_id=[cyan]{run_id}[/cyan]")
    console.print(f"URL: {url}")
    console.print(f"artifacts: {rdir}")
    if not UPLOAD:
        console.print("[dim]render-only (UPLOAD=False) -- no YouTube login needed[/dim]")

    from clipper.graph import get_graph

    initial_state = {
        "run_id": run_id,
        "source_url": url,
        "rights_confirmed": True,
        "formats": FORMATS,
        "no_upload": not UPLOAD,
        "clips": [],
        "errors": [],
    }

    final_state = get_graph().invoke(initial_state, config={"recursion_limit": 200})

    clips = final_state.get("clips", [])
    console.print(f"\n[bold]Done.[/bold] {len(clips)} clip(s).")
    console.print(f"Report: {rdir / 'report.md'}")
    uploaded = sum(1 for c in clips if c.get("upload_status") == "uploaded")
    if uploaded:
        console.print(f"[green]{uploaded}[/green] uploaded as PRIVATE drafts -- review before publishing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
