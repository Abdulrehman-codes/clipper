"""Word-level ASS karaoke captions, burned in via ffmpeg (§4.6).

Builds an ASS subtitle file from the clip's word timestamps with `\\k` karaoke
timing so the active word highlights. 1-2 lines, safe-area aware for 9:16. Style
(font/size/position/colors) comes from config.yaml -> caption.*.
"""

from __future__ import annotations

from pathlib import Path

from ..config import CaptionConfig
from ..ffmpeg_utils import run

# ASS reference canvas. We author against a 1080-tall canvas and let the
# subtitles filter scale to the actual render resolution.
PLAY_RES_X = 1080
PLAY_RES_Y = 1920


def _fmt_ts(seconds: float) -> str:
    """ASS timestamp H:MM:SS.cc (centiseconds)."""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _group_words(words: list[dict], max_chars: int, max_lines: int) -> list[list[dict]]:
    """Group consecutive words into caption events of <= max_lines lines."""
    events: list[list[dict]] = []
    current: list[dict] = []
    line_len = 0
    lines_in_event = 1
    for w in words:
        token = w["word"].strip()
        if not token:
            continue
        add = len(token) + (1 if line_len else 0)
        if line_len + add > max_chars:
            lines_in_event += 1
            line_len = len(token)
            if lines_in_event > max_lines:
                events.append(current)
                current = []
                lines_in_event = 1
        else:
            line_len += add
        current.append(w)
    if current:
        events.append(current)
    return events


def _event_text(words: list[dict], max_chars: int, highlight_color: str) -> str:
    """Render one event as karaoke text with per-word \\k durations + line wraps."""
    parts: list[str] = []
    line_len = 0
    for i, w in enumerate(words):
        token = w["word"].strip()
        dur_cs = max(1, int(round((w["end"] - w["start"]) * 100)))
        add = len(token) + (1 if line_len else 0)
        if i > 0 and line_len + add > max_chars:
            parts.append("\\N")  # hard line break
            line_len = 0
        # \kf = sweep highlight; recolor the active word.
        parts.append(f"{{\\kf{dur_cs}\\c{highlight_color}}}{token} ")
        line_len += add
    return "".join(parts).strip()


def build_ass(words: list[dict], cfg: CaptionConfig, clip_start: float) -> str:
    """Build a full .ass document for one clip. `words` are absolute-time; we
    rebase them to clip-local time using clip_start."""
    margin_v = int(PLAY_RES_Y * (1.0 - cfg.vertical_anchor))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PLAY_RES_X}
PlayResY: {PLAY_RES_Y}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{cfg.font},{cfg.font_size * 4},{cfg.primary_color},{cfg.highlight_color},&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{cfg.outline},{cfg.shadow},2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header]
    for event in _group_words(words, cfg.max_chars_per_line, cfg.max_lines):
        if not event:
            continue
        start = event[0]["start"] - clip_start
        end = event[-1]["end"] - clip_start
        text = _event_text(event, cfg.max_chars_per_line, cfg.highlight_color)
        lines.append(
            f"Dialogue: 0,{_fmt_ts(start)},{_fmt_ts(end)},Default,,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"


def _escape_filter_path(path: Path) -> str:
    """Escape a path for the ffmpeg subtitles= filter (Windows-aware)."""
    s = str(path).replace("\\", "/")
    # Escape the drive-letter colon (C: -> C\:) which ffmpeg filter parsing needs.
    s = s.replace(":", "\\:")
    return s


def burn_captions(src: str | Path, dst: str | Path, ass_path: str | Path, encode) -> Path:
    """Burn an .ass file into src -> dst via the subtitles filter (§4.6)."""
    dst = Path(dst)
    vf = f"subtitles='{_escape_filter_path(Path(ass_path))}'"
    run([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", vf,
        "-c:v", encode.video_codec, "-crf", str(encode.crf), "-preset", encode.preset,
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(dst),
    ])
    return dst
