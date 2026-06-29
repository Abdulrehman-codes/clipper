from clipper.config import Config


def test_config_loads_defaults_from_yaml():
    cfg = Config.load()
    assert cfg.upload.privacy_status == "private"  # §1 hard constraint
    assert cfg.highlight.min_duration_s == 15
    assert cfg.highlight.max_duration_s == 90
    assert cfg.output.formats in {"both", "vertical", "horizontal"}


def test_config_falls_back_to_model_defaults(tmp_path):
    missing = tmp_path / "nope.yaml"
    cfg = Config.load(missing)
    assert cfg.llm.base_url == "https://api.x.ai/v1"
    assert cfg.upload.privacy_status == "private"
