from clipper.nodes.select_highlights import _expand_to_min
from clipper.types import Segment


def _transcript(n=400, step=0.5):
    # words at 0.0-0.5, 0.5-1.0, ... covering up to n*step seconds
    return [{"word": f"w{i}", "start": i * step, "end": i * step + step} for i in range(n)]


def test_expand_grows_short_segment_to_minimum():
    t = _transcript()
    seg = Segment(start=50.0, end=58.0, score=0.9)  # 8s, under 15s min
    _expand_to_min(seg, t, min_dur=15.0, max_dur=90.0)
    assert seg.duration >= 15.0 - 0.5  # within one word boundary of the target


def test_expand_leaves_long_enough_segment_untouched():
    t = _transcript()
    seg = Segment(start=30.0, end=60.0, score=0.8)  # 30s, fine
    before = (seg.start, seg.end)
    _expand_to_min(seg, t, min_dur=15.0, max_dur=90.0)
    assert (seg.start, seg.end) == before


def test_expand_respects_max_duration():
    t = _transcript()
    seg = Segment(start=100.0, end=101.0, score=0.7)
    _expand_to_min(seg, t, min_dur=80.0, max_dur=90.0)
    assert seg.duration <= 90.0 + 0.5


def test_expand_clamps_to_transcript_start():
    t = _transcript()
    seg = Segment(start=1.0, end=4.0, score=0.9)  # near the very start
    _expand_to_min(seg, t, min_dur=15.0, max_dur=90.0)
    assert seg.start >= 0.0
    assert seg.duration >= 15.0 - 0.5
