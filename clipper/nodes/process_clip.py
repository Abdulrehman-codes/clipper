"""process_clip -- the per-segment fan-out target (§3, §4.4-4.7).

LangGraph `Send`s one of these per selected segment so clips render in parallel.
It runs the four single-clip stages in order: cut -> reframe -> caption ->
metadata, and returns a single-element `clips` update (merged via the list
reducer in the graph state). Per-clip failures are captured on the clip instead
of aborting the whole run.
"""

from __future__ import annotations

from ..llm.grok import GrokClient, GrokError
from ..logging_utils import log_event
from ..types import Clip, Segment
from .caption import caption_clip
from .context import RenderContext
from .cut import cut_clip
from .metadata import metadata_clip
from .reframe import reframe_clip


def process_clip(payload: dict) -> dict:
    ctx: RenderContext = payload["ctx"]
    seg = Segment(**payload["segment"])
    clip = Clip(index=seg.index if seg.index >= 0 else payload.get("index", 0), segment=seg)

    log_event("process_clip", "start", index=clip.index, hook=seg.hook_title)
    try:
        cut_clip(ctx, clip)
        reframe_clip(ctx, clip)
        caption_clip(ctx, clip)

        try:
            grok = GrokClient(config=ctx.config)
        except GrokError:
            grok = None  # metadata node falls back gracefully
        metadata_clip(ctx, clip, grok)

        log_event("process_clip", "done", index=clip.index)
    except Exception as exc:  # noqa: BLE001 -- isolate per-clip failure
        clip.upload_status = "error"
        clip.error = str(exc)
        log_event("process_clip", "failed", level="error", index=clip.index, error=str(exc))

    return {"clips": [clip.model_dump()]}
