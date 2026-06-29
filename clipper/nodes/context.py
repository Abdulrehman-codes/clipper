"""Immutable per-run context shared by the per-clip nodes during fan-out.

The fan-out target (process_clip) receives one segment plus this context. Keeping
the heavy shared data here (instead of copying it into every Send payload) keeps
the parallel map lean.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config


@dataclass
class RenderContext:
    run_id: str
    run_dir: Path
    source_path: str
    transcript: list[dict]
    formats: str          # both | vertical | horizontal
    no_upload: bool
    config: Config

    def wants_vertical(self) -> bool:
        return self.formats in ("both", "vertical")

    def wants_horizontal(self) -> bool:
        return self.formats in ("both", "horizontal")

    def words_in(self, start: float, end: float) -> list[dict]:
        return [w for w in self.transcript if w["end"] > start and w["start"] < end]
