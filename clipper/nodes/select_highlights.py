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


def _expand_to_min(seg: Segment, words: list[dict], min_dur: float, max_dur: float) -> None:
    """Grow a too-short segment toward `min_dur` by extending to neighbouring word
    boundaries (split evenly both sides, clamped to the transcript). Rescues strong
    picks that fall just under the minimum instead of discarding them."""
    if seg.duration >= min_dur or not words:
        return
    lo, hi = float(words[0]["start"]), float(words[-1]["end"])
    deficit = min_dur - seg.duration
    ns = _snap_start(max(lo, seg.start - deficit / 2), words)
    ne = _snap_end(min(hi, seg.end + deficit / 2), words)
    # If one side hit a transcript edge, push the other to make up the length.
    if ne - ns < min_dur:
        ne = min(hi, _snap_end(ns + min_dur, words))
    if ne - ns < min_dur:
        ns = max(lo, _snap_start(ne - min_dur, words))
    if ne - ns > max_dur:
        ne = _snap_end(ns + max_dur, words)
    seg.start, seg.end = ns, ne


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
    raw_segments = client.select_highlights(
        transcript, context=context,
        min_duration_s=cfg.min_duration_s, max_duration_s=cfg.max_duration_s,
    )
    log_event("select_highlights", "grok_returned", count=len(raw_segments))

    # Snap, filter, sort, cap -- logging WHY anything is dropped (debuggability).
    snapped: list[Segment] = []
    for seg in raw_segments:
        start = _snap_start(seg.start, transcript)
        end = _snap_end(seg.end, transcript)
        if end <= start:
            continue
        seg.start, seg.end = start, end
        # Rescue strong-but-short picks by padding up to the minimum duration.
        _expand_to_min(seg, transcript, cfg.min_duration_s, cfg.max_duration_s)
        snapped.append(seg)

    filtered: list[Segment] = []
    too_short = too_long = low_score = 0
    for s in snapped:
        if s.duration < cfg.min_duration_s:
            too_short += 1
        elif s.duration > cfg.max_duration_s:
            too_long += 1
        elif s.score < cfg.min_score:
            low_score += 1
        else:
            filtered.append(s)

    filtered.sort(key=lambda s: s.score, reverse=True)
    filtered = filtered[: cfg.max_clips]
    for i, s in enumerate(filtered):
        s.index = i

    if not filtered and snapped:
        log_event(
            "select_highlights", "all_rejected", level="warning",
            too_short=too_short, too_long=too_long, low_score=low_score,
            hint=f"adjust config.yaml highlight.min_duration_s/min_score "
                 f"(current min_duration_s={cfg.min_duration_s}, min_score={cfg.min_score})",
        )

    payload = [s.model_dump() for s in filtered]
    segments_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_event(
        "select_highlights", "done",
        selected=len(payload), rejected=len(snapped) - len(filtered),
        file=str(segments_file),
    )
    return {"segments": payload}
