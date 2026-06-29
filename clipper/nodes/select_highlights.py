"""select_highlights node (§4.3).

Sends the full timestamped transcript to Grok, parses the forced-JSON response,
then filters (duration window, score threshold, max clips) and snaps each
boundary to the nearest transcript word boundary. Caches to segments.json.
"""

from __future__ import annotations

import json

from ..config import get_config, run_path
from ..llm.grok import GrokClient
from ..logging_utils import log_event
from ..types import ClipState, Segment


def _snap_start(t: float, words: list[dict]) -> float:
    """Snap a start time to the start of the nearest word (§4.3)."""
    if not words:
        return t
    best = min(words, key=lambda w: abs(w["start"] - t))
    return float(best["start"])


def _snap_end(t: float, words: list[dict]) -> float:
    """Snap an end time to the end of the nearest word (§4.3)."""
    if not words:
        return t
    best = min(words, key=lambda w: abs(w["end"] - t))
    return float(best["end"])


def select_highlights(state: ClipState) -> dict:
    cfg = get_config().highlight
    run_id = state["run_id"]
    transcript = state.get("transcript") or []
    if not transcript:
        raise ValueError("select_highlights: empty transcript")

    segments_file = run_path(run_id) / "segments.json"
    if segments_file.exists():
        cached = json.loads(segments_file.read_text(encoding="utf-8"))
        log_event("select_highlights", "reuse_cached", count=len(cached))
        return {"segments": cached}

    context_bits = [
        f"title={state.get('source_title')}",
        f"channel={state.get('source_channel')}",
        f"duration={state.get('source_duration')}s",
    ]
    context = ", ".join(b for b in context_bits if "None" not in b)

    log_event("select_highlights", "calling_grok", model=get_config().llm.highlight_model)
    client = GrokClient()
    raw_segments = client.select_highlights(transcript, context=context)
    log_event("select_highlights", "grok_returned", count=len(raw_segments))

    # Snap, filter, sort, cap.
    snapped: list[Segment] = []
    for seg in raw_segments:
        start = _snap_start(seg.start, transcript)
        end = _snap_end(seg.end, transcript)
        if end <= start:
            continue
        seg.start, seg.end = start, end
        snapped.append(seg)

    filtered = [
        s for s in snapped
        if cfg.min_duration_s <= s.duration <= cfg.max_duration_s and s.score >= cfg.min_score
    ]
    filtered.sort(key=lambda s: s.score, reverse=True)
    filtered = filtered[: cfg.max_clips]
    for i, s in enumerate(filtered):
        s.index = i

    payload = [s.model_dump() for s in filtered]
    segments_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_event(
        "select_highlights", "done",
        selected=len(payload), rejected=len(snapped) - len(filtered),
        file=str(segments_file),
    )
    return {"segments": payload}
