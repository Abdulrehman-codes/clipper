from clipper.nodes.select_highlights import _snap_end, _snap_start


def _transcript():
    return [
        {"word": "a", "start": 0.0, "end": 0.5},
        {"word": "b", "start": 0.5, "end": 1.2},
        {"word": "c", "start": 1.2, "end": 2.0},
        {"word": "d", "start": 2.0, "end": 2.9},
    ]


def test_snap_start_to_nearest_word_start():
    assert _snap_start(0.6, _transcript()) == 0.5
    assert _snap_start(1.9, _transcript()) == 2.0


def test_snap_end_to_nearest_word_end():
    assert _snap_end(1.1, _transcript()) == 1.2
    assert _snap_end(2.85, _transcript()) == 2.9


def test_snap_empty_transcript_is_identity():
    assert _snap_start(5.0, []) == 5.0
