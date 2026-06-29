"""Shared typed data structures for the pipeline (§3).

The LangGraph state (`ClipState`) is a TypedDict exactly as specified. The
payload structures (Word/Segment/Clip) are pydantic models so we get validation
for free when parsing LLM output and serialising artifacts.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from pydantic import BaseModel, Field


class Word(BaseModel):
    """A single transcribed word with timestamps (§3, §4.2)."""

    word: str
    start: float
    end: float
    # Per-word confidence if the ASR backend supplies it.
    score: Optional[float] = None


class Segment(BaseModel):
    """An LLM-selected highlight (§4.3).

    Raw LLM output carries start/end/hook_title/rationale/score; `index` and the
    snapped boundaries are filled in by `select_highlights` post-processing.
    """

    start: float
    end: float
    hook_title: str = ""
    rationale: str = ""
    score: float = 0.0
    index: int = -1

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class ClipMetadata(BaseModel):
    """Per-clip publishing metadata produced by the `metadata` node (§4.7)."""

    title: str = ""
    description: str = ""
    hashtags: list[str] = Field(default_factory=list)
    filename: str = ""


class Clip(BaseModel):
    """A rendered clip plus everything we know about it (§3 `clips`)."""

    index: int
    segment: Segment
    # Rendered output paths keyed by format name: "vertical" / "horizontal".
    renders: dict[str, str] = Field(default_factory=dict)
    captioned: dict[str, str] = Field(default_factory=dict)
    metadata: ClipMetadata = Field(default_factory=ClipMetadata)
    # Upload outcome (filled by upload_drafts / report).
    upload_status: str = "pending"   # pending | uploaded | skipped | quota_exceeded | error
    video_id: Optional[str] = None
    error: Optional[str] = None


# --- LangGraph state (§3) -------------------------------------------------

class ClipState(TypedDict, total=False):
    run_id: str
    source_url: str
    rights_confirmed: bool
    source_path: Optional[str]          # downloaded mp4
    audio_path: Optional[str]           # extracted 16kHz mono wav
    source_title: Optional[str]
    source_channel: Optional[str]
    source_duration: Optional[float]
    transcript: Optional[list[dict]]    # list[Word-as-dict]
    segments: Optional[list[dict]]      # list[Segment-as-dict]
    # `clips` and `errors` use list-append reducers so parallel per-segment
    # fan-out (LangGraph Send) merges cleanly instead of clobbering (§3).
    clips: Annotated[list[dict], operator.add]      # list[Clip-as-dict]
    errors: Annotated[list[str], operator.add]
    # Runtime flag threaded through to upload_drafts (§4.8 --no-upload).
    no_upload: bool
