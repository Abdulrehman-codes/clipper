from clipper.llm.grok import (
    _chunk_lines,
    _dedupe_segments,
    _estimate_tokens,
    _format_transcript_lines,
    _is_rate_limit,
    _retry_after,
)
from clipper.types import Segment


def _transcript(n):
    return [{"word": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.5} for i in range(n)]


def test_format_groups_words_one_timestamp_per_line():
    lines = _format_transcript_lines(_transcript(25), words_per_line=12)
    assert len(lines) == 3  # 12 + 12 + 1
    assert lines[0].startswith("[0.00-")
    # one timestamp marker per line
    assert lines[0].count("[") == 1


def test_compact_format_is_far_fewer_tokens_than_per_word():
    t = _transcript(300)
    compact = "\n".join(_format_transcript_lines(t, 12))
    per_word = " ".join(f'{w["word"]} [{w["start"]:.2f}-{w["end"]:.2f}]' for w in t)
    assert _estimate_tokens(compact) < _estimate_tokens(per_word) / 2


def test_chunk_lines_respects_budget():
    lines = _format_transcript_lines(_transcript(600), words_per_line=12)
    chunks = _chunk_lines(lines, max_tokens=200)
    assert len(chunks) > 1
    for c in chunks:
        assert _estimate_tokens("\n".join(c)) <= 200 + 50  # within a line's slack


def test_dedupe_drops_near_duplicate_windows_keeping_higher_score():
    segs = [
        Segment(start=10.0, end=20.0, score=0.7),
        Segment(start=10.2, end=20.1, score=0.9),  # near-dup of the first
        Segment(start=50.0, end=60.0, score=0.6),
    ]
    out = _dedupe_segments(segs)
    assert len(out) == 2
    assert out[0].score == 0.9  # higher-scored survivor


def test_rate_limit_detection():
    class E(Exception):
        status_code = 429

    assert _is_rate_limit(E("rate_limit_exceeded"))
    assert _is_rate_limit(Exception("Request too large for model"))
    assert not _is_rate_limit(Exception("some other error"))


def test_retry_after_parses_hint():
    assert _retry_after(Exception("Please try again in 7.5s")) == 7.5
    assert _retry_after(Exception("no hint here")) is None
