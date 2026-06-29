"""Thin ffmpeg/ffprobe helpers used across the video nodes (§2: ffmpeg on PATH)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise FFmpegError(
            "ffmpeg/ffprobe not found on PATH. Install ffmpeg (see README) and retry."
        )


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run an ffmpeg-family command, raising FFmpegError with stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-2000:]}"
        )
    return proc


def probe_duration(path: str | Path) -> float:
    proc = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ])
    data = json.loads(proc.stdout)
    return float(data.get("format", {}).get("duration", 0.0))


def probe_dimensions(path: str | Path) -> tuple[int, int]:
    proc = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(path),
    ])
    stream = json.loads(proc.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def extract_audio_wav(src: str | Path, dst: str | Path, sample_rate: int = 16000) -> Path:
    """Extract mono PCM wav at the given sample rate (Whisper wants 16kHz mono, §4.1)."""
    dst = Path(dst)
    run([
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", str(dst),
    ])
    return dst
