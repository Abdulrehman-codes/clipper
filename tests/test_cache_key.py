from clipper.config import video_key

VID = "zTxkGUG4TVI"


def test_watch_and_short_and_shorts_share_one_key():
    forms = [
        f"https://www.youtube.com/watch?v={VID}",
        f"https://youtu.be/{VID}",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://www.youtube.com/watch?v={VID}&t=42s",
        f"https://m.youtube.com/watch?feature=share&v={VID}",
    ]
    keys = {video_key(f) for f in forms}
    assert keys == {VID}, keys


def test_non_youtube_url_falls_back_to_hash():
    k = video_key("https://example.com/some/video")
    assert k != "" and "/" not in k and len(k) == 11
