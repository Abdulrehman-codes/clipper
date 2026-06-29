"""reframe node (§4.5) -- single-clip stage in the per-segment fan-out.

Derives the 9:16 vertical render from the base cut by following the dominant
face (smoothed crop). Skipped when --formats horizontal.
"""

from __future__ import annotations

from ..logging_utils import log_event
from ..types import Clip
from ..video.reframe import reframe_to_vertical
from .context import RenderContext


def reframe_clip(ctx: RenderContext, clip: Clip) -> Clip:
    if not ctx.wants_vertical():
        return clip
    base = clip.renders.get("base")
    if not base:
        raise ValueError("reframe_clip: missing base render (cut must run first)")
    vertical = ctx.run_dir / f"clip_{clip.index:02d}_vertical.mp4"
    reframe_to_vertical(base, vertical, ctx.config.reframe, ctx.config.encode)
    clip.renders["vertical"] = str(vertical)
    log_event("reframe", "done", index=clip.index, path=str(vertical))
    return clip
