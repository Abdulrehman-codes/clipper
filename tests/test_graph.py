"""Smoke test: the LangGraph wiring compiles and exposes the expected nodes."""


def test_graph_compiles_with_expected_nodes():
    from clipper.graph import build_graph

    graph = build_graph()
    nodes = set(graph.get_graph().nodes)
    for expected in {
        "ingest_gate", "download", "transcribe", "select_highlights",
        "process_clip", "upload_drafts", "report",
    }:
        assert expected in nodes
