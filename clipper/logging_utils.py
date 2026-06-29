"""Structured, per-run logging (§3: 'each node with retry + structured logging').

Every node emits JSON lines to runs/<run_id>/run.log *and* a human-readable
line to the console via rich. Call `setup_run_logging(run_id)` once at the start
of a run, then `log_event(...)` from anywhere.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()
_logger = logging.getLogger("clipper")


def setup_run_logging(run_dir: Path, verbose: bool = True) -> None:
    """Attach a JSONL file handler scoped to this run."""
    _logger.setLevel(logging.DEBUG)
    # Drop any handlers from a previous run in the same process.
    for h in list(_logger.handlers):
        _logger.removeHandler(h)

    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(fh)
    _logger.propagate = False


def log_event(node: str, event: str, level: str = "info", **fields: Any) -> None:
    """Emit one structured event to the run log + a console line."""
    record = {"ts": round(time.time(), 3), "node": node, "event": event, **fields}
    line = json.dumps(record, default=str)
    getattr(_logger, level if level in {"debug", "info", "warning", "error"} else "info")(line)

    color = {"info": "cyan", "warning": "yellow", "error": "red", "debug": "dim"}.get(level, "white")
    extra = " ".join(f"{k}={v}" for k, v in fields.items() if k not in {"traceback"})
    console.print(f"[{color}]\\[{node}][/{color}] {event} {extra}".rstrip())
