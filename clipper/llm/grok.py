"""xAI Grok client + prompts + JSON parsing (§2, §4.3, §4.7).

Grok is OpenAI-SDK-compatible, so we drive it through the `openai` client with
`base_url="https://api.x.ai/v1"`. Two jobs live here:

  * highlight selection (§4.3) -- pick the strongest segments, JSON only.
  * metadata generation (§4.7) -- per-clip title/description/hashtags/filename.

NOTE: model strings (grok-4.1-fast / grok-4.3) change -- verify in the xAI
console. They are config-driven (config.yaml -> llm.*), not hard-coded here.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..config import Config, Settings, get_config, get_settings
from ..types import Segment, Word

# --- prompts --------------------------------------------------------------

HIGHLIGHT_SYSTEM_PROMPT = """\
You are an expert short-form video editor. You are given the full word-level \
timestamped transcript of a long-form video. Find the strongest self-contained \
moments to cut into shorts.

Each selected moment MUST:
- have a strong HOOK within its first 3 seconds,
- contain a clear PAYOFF (insight, punchline, emotional beat, or resolution),
- start and end on clean sentence boundaries (do not cut mid-sentence),
- stand alone without external context.

Return ONLY a JSON array. No prose, no markdown fences, no commentary. Schema:
[{"start": <float seconds>, "end": <float seconds>, "hook_title": <string>, \
"rationale": <string>, "score": <float 0.0-1.0>}]

`score` is your confidence that this is a great standalone short. Prefer fewer, \
higher-quality picks over many mediocre ones."""

METADATA_SYSTEM_PROMPT = """\
You are a YouTube Shorts metadata specialist. Given a short clip's transcript \
text and its hook, produce publishing metadata.

Return ONLY a JSON object. No prose, no markdown fences. Schema:
{"title": <string, <=60 chars, hook-style, no clickbait lies>, \
"description": <string, 1-3 sentences>, \
"hashtags": [<3-5 strings, no '#' prefix>], \
"filename": <string, lowercase-kebab, no extension>}"""


class GrokError(RuntimeError):
    pass


class GrokClient:
    def __init__(self, settings: Optional[Settings] = None, config: Optional[Config] = None):
        self.settings = settings or get_settings()
        self.config = config or get_config()
        if not self.settings.xai_api_key:
            raise GrokError(
                "XAI_API_KEY is not set. Add it to .env (see .env.example)."
            )
        from openai import OpenAI  # lazy import

        self._client = OpenAI(
            api_key=self.settings.xai_api_key,
            base_url=self.config.llm.base_url,
            timeout=self.config.llm.request_timeout_s,
        )

    # --- low-level call --------------------------------------------------
    def _chat_json(self, model: str, system: str, user: str) -> Any:
        resp = self._client.chat.completions.create(
            model=model,
            temperature=self.config.llm.temperature,
            response_format={"type": "json_object"},  # force JSON (§4.3)
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or ""
        return _parse_json_loose(content)

    # --- highlight selection (§4.3) -------------------------------------
    def select_highlights(self, transcript: list[dict], context: str = "") -> list[Segment]:
        transcript_text = _format_transcript(transcript)
        user = (
            (f"Video context: {context}\n\n" if context else "")
            + "Transcript (word: [start-end]):\n"
            + transcript_text
            + "\n\nReturn the JSON array of highlights now."
        )
        data = self._chat_json(self.config.llm.highlight_model, HIGHLIGHT_SYSTEM_PROMPT, user)
        items = _coerce_array(data)
        segments: list[Segment] = []
        for item in items:
            try:
                segments.append(Segment(
                    start=float(item["start"]),
                    end=float(item["end"]),
                    hook_title=str(item.get("hook_title", "")),
                    rationale=str(item.get("rationale", "")),
                    score=float(item.get("score", 0.0)),
                ))
            except (KeyError, TypeError, ValueError):
                continue  # skip malformed entries rather than fail the whole run
        return segments

    # --- metadata (§4.7) -------------------------------------------------
    def generate_metadata(self, clip_text: str, hook_title: str) -> dict:
        user = (
            f"Hook: {hook_title}\n\nClip transcript:\n{clip_text}\n\n"
            "Return the JSON metadata object now."
        )
        data = self._chat_json(self.config.llm.metadata_model, METADATA_SYSTEM_PROMPT, user)
        if not isinstance(data, dict):
            raise GrokError("metadata response was not a JSON object")
        return data


# --- helpers --------------------------------------------------------------

def _format_transcript(transcript: list[dict]) -> str:
    """Compact word: [start-end] lines. 2M ctx usually fits a full transcript (§4.3)."""
    lines = []
    for w in transcript:
        lines.append(f'{w["word"]} [{w["start"]:.2f}-{w["end"]:.2f}]')
    return " ".join(lines)


def _parse_json_loose(text: str) -> Any:
    """Parse JSON even if the model wrapped it in fences despite instructions."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: grab the first {...} or [...] block.
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise GrokError(f"could not parse JSON from model output: {text[:300]!r}")


def _coerce_array(data: Any) -> list[dict]:
    """json_object mode returns an object; highlights may be under a key."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("highlights", "segments", "clips", "results", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        # Single object describing one highlight.
        if "start" in data and "end" in data:
            return [data]
    return []
