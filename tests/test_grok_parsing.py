from clipper.llm.grok import _coerce_array, _parse_json_loose


def test_parse_plain_json_array():
    out = _parse_json_loose('[{"start": 1.0, "end": 2.0}]')
    assert out == [{"start": 1.0, "end": 2.0}]


def test_parse_json_with_fences():
    text = "```json\n{\"a\": 1}\n```"
    assert _parse_json_loose(text) == {"a": 1}


def test_parse_json_embedded_in_prose():
    text = 'Here you go: [{"start": 3}] hope that helps'
    assert _parse_json_loose(text) == [{"start": 3}]


def test_coerce_array_unwraps_keyed_object():
    assert _coerce_array({"highlights": [{"start": 1}]}) == [{"start": 1}]
    assert _coerce_array({"start": 1, "end": 2}) == [{"start": 1, "end": 2}]
    assert _coerce_array("nonsense") == []
