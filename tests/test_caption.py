from clipper.config import CaptionConfig
from clipper.video.caption import _fmt_ts, _group_words, build_ass


def _words():
    return [
        {"word": "Hello", "start": 1.0, "end": 1.4},
        {"word": "there", "start": 1.4, "end": 1.9},
        {"word": "friends", "start": 1.9, "end": 2.6},
    ]


def test_fmt_ts():
    assert _fmt_ts(0) == "0:00:00.00"
    assert _fmt_ts(75.5) == "0:01:15.50"


def test_group_words_respects_char_limit():
    cfg = CaptionConfig(max_chars_per_line=6, max_lines=1)
    events = _group_words(_words(), cfg.max_chars_per_line, cfg.max_lines)
    # Each word is its own event because line is capped at 6 chars, 1 line.
    assert len(events) == 3


def test_build_ass_rebases_to_clip_local_time():
    cfg = CaptionConfig()
    ass = build_ass(_words(), cfg, clip_start=1.0)
    assert "[Events]" in ass
    assert "Dialogue:" in ass
    # First word started at 1.0 absolute -> 0.0 local.
    assert "0:00:00.00" in ass
    # karaoke tag present
    assert "\\kf" in ass
