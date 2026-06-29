"""Cut a segment out of the source video (§4.4).

Stream-copy where possible (fast, lossless); re-encode only when a clean cut on
the requested boundaries isn't achievable via copy. We place -ss before -i for a
fast seek and -to/-t for the segment end.
"""

from __future__ import annotations

from pathlib import Path

from ..config import EncodeConfig
from ..ffmpeg_utils import run


def cut_segment(
    src: str | Path,
    dst: str | Path,
    start: float,
    end: float,
    encode: EncodeConfig,
    copy: bool = True,
) -> Path:
    """Cut [start, end] from src into dst.

    copy=True attempts stream-copy (keyframe-accurate-ish, very fast). If the
    caller needs frame-accurate cuts (e.g. before reframing) pass copy=False to
    re-encode.
    """
    dst = Path(dst)
    duration = max(0.01, end - start)

    if copy:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-i", str(src),
            "-t", f"{duration:.3f}",
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            str(dst),
        ]
        try:
            run(cmd)
            return dst
        except Exception:
            # Fall through to re-encode if copy produced an error (e.g. odd codec).
            pass

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c:v", encode.video_codec, "-crf", str(encode.crf), "-preset", encode.preset,
        "-c:a", encode.audio_codec, "-b:a", encode.audio_bitrate,
        "-movflags", "+faststart",
        str(dst),
    ]
    run(cmd)
    return dst
