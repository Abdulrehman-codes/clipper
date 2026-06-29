"""caption node (§4.6) -- single-clip stage in the per-segment fan-out.

Builds one ASS file from the clip's words (rebased to clip-local time) and burns
it into every requested render. Captioned files are what gets uploaded.
"""

from __future__ import annotations

from ..logging_utils import log_event
from ..types import Clip
from ..video.caption import build_ass, burn_captions
from .context import RenderContext


def caption_clip(ctx: RenderContext, clip: Clip) -> Clip:
    seg = clip.segment
    words = ctx.words_in(seg.start, seg.end)
    if not words:
        log_event("caption", "no_words", level="warning", index=clip.index)

    ass_text = build_ass(words, ctx.config.caption, clip_start=seg.start)
    ass_path = ctx.run_dir / f"clip_{clip.index:02d}.ass"
    ass_path.write_text(ass_text, encoding="utf-8")

    for fmt in ("vertical", "horizontal"):
        src = clip.renders.get(fmt)
        if not src:
            continue
        dst = ctx.run_dir / f"clip_{clip.index:02d}_{fmt}_captioned.mp4"
        burn_captions(src, dst, ass_path, ctx.config.encode)
        clip.captioned[fmt] = str(dst)
        log_event("caption", "burned", index=clip.index, fmt=fmt, path=str(dst))
    return clip
