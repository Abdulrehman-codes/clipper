"""LangGraph wiring (§3).

Flow:
    ingest_gate -> download -> transcribe -> select_highlights
        -> (fan-out: process_clip per segment) -> upload_drafts -> report

Per-segment work fans out via `Send` so clips render in parallel. Each node has a
retry policy; per-clip failures are isolated inside process_clip so one bad clip
never aborts the run.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, START, StateGraph

try:  # Send / RetryPolicy moved to langgraph.types in recent versions.
    from langgraph.types import RetryPolicy, Send
except ImportError:  # pragma: no cover - older langgraph
    from langgraph.pregel import RetryPolicy  # type: ignore
    from langgraph.constants import Send  # type: ignore

from .config import get_config, run_path
from .nodes.context import RenderContext
from .nodes.download import download
from .nodes.ingest_gate import ingest_gate
from .nodes.process_clip import process_clip
from .nodes.report import report
from .nodes.select_highlights import select_highlights
from .nodes.transcribe import transcribe
from .nodes.upload_drafts import upload_drafts
from .types import ClipState

# Network-bound nodes get more retries; deterministic local ones get fewer.
_NET_RETRY = RetryPolicy(max_attempts=3)
_LOCAL_RETRY = RetryPolicy(max_attempts=2)


def _dispatch_clips(state: ClipState):
    """Conditional edge after select_highlights: Send one process_clip per segment.

    With no segments, route straight to upload_drafts (nothing to render)."""
    segments = state.get("segments") or []
    if not segments:
        return "upload_drafts"

    ctx = RenderContext(
        run_id=state["run_id"],
        run_dir=run_path(state["run_id"]),
        source_path=state["source_path"],
        transcript=state.get("transcript") or [],
        formats=state.get("formats", get_config().output.formats),
        no_upload=bool(state.get("no_upload")),
        config=get_config(),
    )
    return [
        Send("process_clip", {"segment": seg, "ctx": ctx, "index": i})
        for i, seg in enumerate(segments)
    ]


def build_graph():
    g = StateGraph(ClipState)

    g.add_node("ingest_gate", ingest_gate, retry_policy=_LOCAL_RETRY)
    g.add_node("download", download, retry_policy=_NET_RETRY)
    g.add_node("transcribe", transcribe, retry_policy=_LOCAL_RETRY)
    g.add_node("select_highlights", select_highlights, retry_policy=_NET_RETRY)
    g.add_node("process_clip", process_clip, retry_policy=_LOCAL_RETRY)
    g.add_node("upload_drafts", upload_drafts, retry_policy=_NET_RETRY)
    g.add_node("report", report)

    g.add_edge(START, "ingest_gate")
    g.add_edge("ingest_gate", "download")
    g.add_edge("download", "transcribe")
    g.add_edge("transcribe", "select_highlights")
    # Fan-out per segment.
    g.add_conditional_edges("select_highlights", _dispatch_clips, ["process_clip", "upload_drafts"])
    # Fan-in: all process_clip branches converge on upload_drafts.
    g.add_edge("process_clip", "upload_drafts")
    g.add_edge("upload_drafts", "report")
    g.add_edge("report", END)

    return g.compile()


# Module-level compiled graph for convenience (`from clipper.graph import GRAPH`).
GRAPH = None


def get_graph():
    global GRAPH
    if GRAPH is None:
        GRAPH = build_graph()
    return GRAPH
