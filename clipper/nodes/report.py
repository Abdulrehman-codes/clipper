"""report node (§4.9).

Writes runs/<id>/report.md: every clip with path, title, score, rationale,
upload status, and video id. Also drops a machine-readable report.json.
"""

from __future__ import annotations

import json

from ..config import run_path
from ..logging_utils import log_event
from ..types import Clip, ClipState


def report(state: ClipState) -> dict:
    run_id = state["run_id"]
    rdir = run_path(run_id)
    clips = sorted((Clip(**c) for c in state.get("clips", [])), key=lambda c: c.index)

    lines: list[str] = []
    lines.append(f"# clipper report — run `{run_id}`\n")
    lines.append(f"- **Source:** {state.get('source_title') or state.get('source_url')}")
    lines.append(f"- **Channel:** {state.get('source_channel')}")
    lines.append(f"- **Source URL:** {state.get('source_url')}")
    lines.append(f"- **Clips produced:** {len(clips)}")
    errors = state.get("errors") or []
    if errors:
        lines.append(f"- **Run errors:** {len(errors)}")
    lines.append("\n> Default visibility is **private**. Review every clip before publishing.\n")

    for c in clips:
        seg = c.segment
        lines.append(f"## Clip {c.index:02d} — {c.metadata.title or seg.hook_title}\n")
        lines.append(f"- **Window:** {seg.start:.2f}s → {seg.end:.2f}s ({seg.duration:.1f}s)")
        lines.append(f"- **Score:** {seg.score:.2f}")
        lines.append(f"- **Hook:** {seg.hook_title}")
        lines.append(f"- **Rationale:** {seg.rationale}")
        if c.metadata.description:
            lines.append(f"- **Description:** {c.metadata.description}")
        if c.metadata.hashtags:
            lines.append(f"- **Hashtags:** {' '.join('#' + h for h in c.metadata.hashtags)}")
        for fmt, path in c.captioned.items():
            lines.append(f"- **{fmt} (captioned):** `{path}`")
        for fmt, path in c.renders.items():
            if fmt == "base":
                continue
            lines.append(f"- **{fmt} (raw):** `{path}`")
        lines.append(f"- **Upload status:** {c.upload_status}")
        if c.video_id:
            lines.append(f"- **Video:** https://youtube.com/watch?v={c.video_id} (private)")
        if c.error:
            lines.append(f"- **Error:** {c.error}")
        lines.append("")

    report_md = rdir / "report.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")

    report_json = rdir / "report.json"
    report_json.write_text(
        json.dumps({
            "run_id": run_id,
            "source_url": state.get("source_url"),
            "source_title": state.get("source_title"),
            "clips": [c.model_dump() for c in clips],
            "errors": errors,
        }, indent=2),
        encoding="utf-8",
    )

    log_event("report", "written", file=str(report_md), clips=len(clips))
    return {}
