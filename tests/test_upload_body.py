from clipper.config import UploadConfig
from clipper.nodes.metadata import _slugify
from clipper.youtube.upload import build_body


def test_build_body_is_always_private():
    """§1 hard constraint: there is no publish path; privacy is always private."""
    cfg = UploadConfig()
    body = build_body("My Title", "A description.", ["foo", "#bar"], cfg)
    assert body["status"]["privacyStatus"] == "private"
    # hashtags normalised (no leading '#') in tags, appended to description.
    assert body["snippet"]["tags"] == ["foo", "bar"]
    assert "#foo #bar" in body["snippet"]["description"]


def test_title_is_truncated_for_api():
    cfg = UploadConfig()
    body = build_body("x" * 200, "", [], cfg)
    assert len(body["snippet"]["title"]) <= 100


def test_slugify():
    assert _slugify("Hello, World!") == "hello-world"
    assert _slugify("") == "clip"
