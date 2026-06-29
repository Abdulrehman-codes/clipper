"""Resumable, quota-aware private-draft upload (§4.8).

Hard constraint (§1): privacyStatus is always "private" -- there is no publish
path. Each videos.insert costs ~1600 units against a 10k/day default quota
(~6 uploads/day), so we budget per run and, on a quota error, save state locally
and exit gracefully rather than hammering the API.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..config import UploadConfig
from ..logging_utils import log_event


class QuotaExceeded(RuntimeError):
    pass


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "quota" in text or "quotaexceeded" in text or "dailylimitexceeded" in text


def build_body(title: str, description: str, hashtags: list[str], cfg: UploadConfig) -> dict:
    tags = [h.lstrip("#") for h in hashtags]
    # Append hashtags to description so they render under the Short.
    desc = description
    if tags:
        desc = (desc + "\n\n" + " ".join(f"#{t}" for t in tags)).strip()
    return {
        "snippet": {
            "title": title[:100],
            "description": desc[:4900],
            "tags": tags,
            "categoryId": cfg.category_id,
        },
        "status": {
            "privacyStatus": cfg.privacy_status,          # ALWAYS private (§1)
            "selfDeclaredMadeForKids": cfg.made_for_kids,
        },
    }


def upload_video(service, file_path: str | Path, body: dict, cfg: UploadConfig) -> str:
    """Resumable upload with exponential backoff. Returns the new video id.

    Raises QuotaExceeded on quota errors so the caller can stop gracefully."""
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(str(file_path), chunksize=-1, resumable=True)
    request = service.videos().insert(part="snippet,status", body=body, media_body=media)

    attempt = 0
    response = None
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                log_event("upload_drafts", "progress", pct=int(status.progress() * 100))
        except HttpError as exc:
            if _is_quota_error(exc):
                raise QuotaExceeded(str(exc)) from exc
            attempt += 1
            if attempt > cfg.max_retries:
                raise
            sleep_s = cfg.backoff_base_s * (2 ** (attempt - 1))
            log_event("upload_drafts", "retry", level="warning", attempt=attempt, sleep_s=sleep_s)
            time.sleep(sleep_s)
    return response["id"]
