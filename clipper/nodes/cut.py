"""cut node (§4.4) -- single-clip stage used inside the per-segment fan-out.

Produces the base cut (16:9). The horizontal output IS this cut (§4.5: "emit the
16:9 version -- just the cut, no crop"); reframe derives the vertical from it.
We re-encode here for frame-accurate boundaries before reframing/captioning.
"""

from __future__ import annotations

from ..logging_utils import log_event
from ..types import Clip
from ..video.cut import cut_segment
from .context import RenderContext


def cut_clip(ctx: RenderContext, clip: Clip) -> Clip:
    seg = clip.segment
    base = ctx.run_dir / f"clip_{clip.index:02d}_base.mp4"
    # Re-encode (copy=False) so the cut lands exactly on the snapped boundaries.
    cut_segment(ctx.source_path, base, seg.start, seg.end, ctx.config.encode, copy=False)
    clip.renders["base"] = str(base)
    if ctx.wants_horizontal():
        clip.renders["horizontal"] = str(base)
    log_event("cut", "done", index=clip.index, start=seg.start, end=seg.end, path=str(base))
    return clip
