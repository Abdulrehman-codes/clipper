"""download node (§4.1).

yt-dlp -> cache/<video>/source.mp4 (bestvideo+bestaudio, remux to mp4) and a
16kHz mono wav for Whisper. Capture source title/channel/duration into state for
later metadata context.

Idempotent (§1): the source + audio live in a per-video cache keyed by the URL,
so re-running the SAME video never re-downloads -- even across different run_ids.
If a previous (timestamped) run already downloaded the file, we adopt it into the
cache instead of fetching again.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..config import cache_path, runs_dir, video_key
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


def _adopt_from_prior_runs(url: str, dst: Path, info_dst: Path) -> bool:
    """If an earlier run already downloaded this video, move it into the cache
    instead of re-downloading. Matches by video id (or url) in source_info.json."""
    key = video_key(url)
    for info_file in runs_dir().glob("*/source_info.json"):
        try:
            meta = json.loads(info_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        same = meta.get("id") == key or url in (meta.get("webpage_url") or "")
        src = info_file.parent / "source.mp4"
        if same and src.exists():
            shutil.copy2(src, dst)
            info_dst.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            # Adopt sibling audio.wav / transcript.json too, if present, so we
            # also skip re-extracting audio and re-transcribing.
            for name in ("audio.wav", "transcript.json"):
                sib = info_file.parent / name
                if sib.exists():
                    shutil.copy2(sib, dst.parent / name)
            log_event("download", "adopted_from_run", source=str(src), cache=str(dst))
            return True
    return False


def download(state: ClipState) -> dict:
    ensure_ffmpeg()
    url = state["source_url"]
    cdir = cache_path(url)

    source_path = cdir / "source.mp4"
    audio_path = cdir / "audio.wav"
    info_path = cdir / "source_info.json"

    # Idempotency (§1): reuse cached source; else adopt from a prior run; else fetch.
    if source_path.exists() and info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        log_event("download", "reuse_cached", path=str(source_path))
    elif _adopt_from_prior_runs(url, source_path, info_path):
        info = json.loads(info_path.read_text(encoding="utf-8"))
    else:
        log_event("download", "start", url=url)
        info = _download_with_ytdlp(url, str(cdir / "source.%(ext)s"))
        # yt-dlp may emit source.mkv/webm before remux; normalise to source.mp4.
        if not source_path.exists():
            candidates = sorted(cdir.glob("source.*"))
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
