"""download node (§4.1).

yt-dlp -> runs/<id>/source.mp4 (bestvideo+bestaudio, remux to mp4) and extract a
16kHz mono wav for Whisper. Capture source title/channel/duration into state for
later metadata context. Idempotent: reuses an existing source.mp4 (§1).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import run_path
from ..ffmpeg_utils import ensure_ffmpeg, extract_audio_wav, probe_duration
from ..logging_utils import log_event
from ..types import ClipState


def _download_with_ytdlp(url: str, out_tmpl: str) -> dict:
    """Use the yt-dlp python API; bestvideo+bestaudio remuxed to mp4 (§4.1)."""
    import yt_dlp  # imported lazily so the package imports without yt-dlp present

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": out_tmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"},
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return info


def download(state: ClipState) -> dict:
    ensure_ffmpeg()
    run_id = state["run_id"]
    url = state["source_url"]
    rdir = run_path(run_id)

    source_path = rdir / "source.mp4"
    audio_path = rdir / "audio.wav"
    info_path = rdir / "source_info.json"

    # Idempotency (§1): reuse a previously downloaded source.
    if source_path.exists() and info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        log_event("download", "reuse_cached", path=str(source_path))
    else:
        log_event("download", "start", url=url)
        info = _download_with_ytdlp(url, str(rdir / "source.%(ext)s"))
        # yt-dlp may emit source.mkv/webm before remux; normalise to source.mp4.
        if not source_path.exists():
            candidates = sorted(rdir.glob("source.*"))
            mp4s = [c for c in candidates if c.suffix == ".mp4"]
            chosen = mp4s[0] if mp4s else (candidates[0] if candidates else None)
            if chosen is None:
                raise FileNotFoundError("yt-dlp produced no source file")
            if chosen != source_path:
                chosen.rename(source_path)
        # Persist a trimmed info blob for idempotent reuse.
        slim = {
            "title": info.get("title"),
            "channel": info.get("channel") or info.get("uploader"),
            "duration": info.get("duration"),
            "id": info.get("id"),
            "webpage_url": info.get("webpage_url", url),
        }
        info_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
        info = slim
        log_event("download", "downloaded", path=str(source_path), title=info.get("title"))

    if not audio_path.exists():
        extract_audio_wav(source_path, audio_path)
        log_event("download", "audio_extracted", path=str(audio_path))

    duration = info.get("duration")
    if not duration:
        duration = probe_duration(source_path)

    return {
        "source_path": str(source_path),
        "audio_path": str(audio_path),
        "source_title": info.get("title"),
        "source_channel": info.get("channel"),
        "source_duration": float(duration) if duration else None,
    }
