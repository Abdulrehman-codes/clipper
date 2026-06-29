"""metadata node (§4.7) -- single-clip stage in the per-segment fan-out.

Asks Grok for title (<=60 chars), description, 3-5 hashtags, and a suggested
filename. Falls back to the hook title if the LLM is unavailable, so rendering
never blocks on metadata.
"""

from __future__ import annotations

import re

from ..llm.grok import GrokClient
from ..logging_utils import log_event
from ..types import Clip, ClipMetadata
from .context import RenderContext


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "clip"


def metadata_clip(ctx: RenderContext, clip: Clip, grok: GrokClient | None) -> Clip:
    seg = clip.segment
    words = ctx.words_in(seg.start, seg.end)
    clip_text = " ".join(w["word"] for w in words).strip()

    if grok is not None:
        try:
            data = grok.generate_metadata(clip_text, seg.hook_title)
            title = str(data.get("title", "") or seg.hook_title)[:60]
            hashtags = [str(h).lstrip("#") for h in data.get("hashtags", [])][:5]
            clip.metadata = ClipMetadata(
                title=title,
                description=str(data.get("description", "")),
                hashtags=hashtags,
                filename=_slugify(str(data.get("filename") or title or seg.hook_title)),
            )
            log_event("metadata", "done", index=clip.index, title=title)
            return clip
        except Exception as exc:  # noqa: BLE001 -- degrade gracefully
            log_event("metadata", "grok_failed", level="warning", index=clip.index, error=str(exc))

    # Fallback metadata derived from the hook.
    title = (seg.hook_title or clip_text[:57])[:60]
    clip.metadata = ClipMetadata(
        title=title,
        description=seg.rationale or clip_text[:200],
        hashtags=["shorts"],
        filename=_slugify(title),
    )
    log_event("metadata", "fallback", index=clip.index, title=title)
    return clip
