"""upload_drafts node (§4.8).

Uploads each rendered clip as a PRIVATE draft, rate-limited by the per-run quota
budget. Resumable + backoff per video. On a quota error it stops, leaves files on
disk, and marks the rest so a later run can resume. `--no-upload` skips entirely.
"""

from __future__ import annotations

from ..config import get_config
from ..logging_utils import log_event
from ..types import Clip, ClipState
from ..youtube.auth import YouTubeAuthError, get_youtube_service
from ..youtube.upload import QuotaExceeded, build_body, upload_video


def _pick_upload_file(clip: Clip) -> str | None:
    """Prefer the captioned vertical Short; fall back sensibly."""
    for key in ("vertical", "horizontal"):
        if clip.captioned.get(key):
            return clip.captioned[key]
    for key in ("vertical", "horizontal", "base"):
        if clip.renders.get(key):
            return clip.renders[key]
    return None


def upload_drafts(state: ClipState) -> dict:
    cfg = get_config().upload
    clips = [Clip(**c) for c in state.get("clips", [])]

    if state.get("no_upload"):
        for c in clips:
            if c.upload_status == "pending":
                c.upload_status = "skipped"
        log_event("upload_drafts", "skipped_no_upload", clips=len(clips))
        return {"clips": [c.model_dump() for c in clips]}

    # Quota budget: how many inserts fit in the daily quota (§4.8).
    budget = max(0, cfg.daily_quota_units // max(1, cfg.insert_cost_units))
    log_event("upload_drafts", "budget", uploads_allowed=budget, cost_each=cfg.insert_cost_units)

    try:
        service = get_youtube_service()
    except YouTubeAuthError as exc:
        log_event("upload_drafts", "not_authorized", level="warning", error=str(exc))
        for c in clips:
            if c.upload_status == "pending":
                c.upload_status = "skipped"
                c.error = "not authorized (run `clipper auth`)"
        return {"clips": [c.model_dump() for c in clips]}

    uploaded = 0
    for clip in clips:
        if clip.upload_status not in ("pending",):
            continue
        if uploaded >= budget:
            clip.upload_status = "quota_exceeded"
            clip.error = "per-run quota budget reached; rerun tomorrow to resume"
            continue
        file_path = _pick_upload_file(clip)
        if not file_path:
            clip.upload_status = "error"
            clip.error = "no rendered file to upload"
            continue

        body = build_body(
            clip.metadata.title, clip.metadata.description, clip.metadata.hashtags, cfg
        )
        try:
            log_event("upload_drafts", "uploading", index=clip.index, file=file_path)
            vid = upload_video(service, file_path, body, cfg)
            clip.video_id = vid
            clip.upload_status = "uploaded"
            uploaded += 1
            log_event("upload_drafts", "uploaded", index=clip.index, video_id=vid,
                      url=f"https://youtube.com/watch?v={vid}")
        except QuotaExceeded as exc:
            # Stop gracefully; files remain on disk for a resumable rerun (§4.8).
            clip.upload_status = "quota_exceeded"
            clip.error = str(exc)
            log_event("upload_drafts", "quota_exceeded", level="warning", index=clip.index)
            for rest in clips:
                if rest.upload_status == "pending":
                    rest.upload_status = "quota_exceeded"
                    rest.error = "stopped after quota error"
            break
        except Exception as exc:  # noqa: BLE001
            clip.upload_status = "error"
            clip.error = str(exc)
            log_event("upload_drafts", "error", level="error", index=clip.index, error=str(exc))

    return {"clips": [c.model_dump() for c in clips]}
