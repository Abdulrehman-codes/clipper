"""LLM client + prompts + JSON parsing (§2, §4.3, §4.7).

Provider-agnostic: any OpenAI-SDK-compatible endpoint (Groq by default, xAI Grok,
...) via `base_url` + model strings from config.yaml -> llm.*. Two jobs:

  * highlight selection (§4.3) -- pick the strongest segments, JSON only.
  * metadata generation (§4.7) -- per-clip title/description/hashtags/filename.

To respect provider rate limits (e.g. Groq free tier = 12k tokens/min), the
transcript is compacted (one timestamp per line, not per word), split into
requests under `max_input_tokens`, and paced under `tpm_limit`. Rate-limit
errors (429/413) are retried with exponential backoff (§4.3 "chunk if >context").
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from ..config import Config, Settings, get_config, get_settings
from ..logging_utils import log_event
from ..types import Segment, Word

# --- prompts --------------------------------------------------------------

def highlight_system_prompt(min_s: float, max_s: float) -> str:
    target = int((min_s + max_s) / 2)
    return f"""\
You are an expert short-form video editor. You are given the full word-level \
timestamped transcript of a long-form video. Find the strongest self-contained \
segments to cut into shorts.

CRITICAL LENGTH REQUIREMENT:
- Each highlight MUST be between {int(min_s)} and {int(max_s)} seconds long \
(end - start). Aim for around {target} seconds.
- NEVER return a clip shorter than {int(min_s)} seconds. A single sentence or \
one-liner is too short -- EXPAND the window to include the full surrounding \
exchange: the setup/build-up AND the payoff, so it stands on its own.
- If a great moment is short, extend `start` earlier and `end` later (to clean \
sentence boundaries) until it reaches at least {int(min_s)} seconds.

Each highlight MUST also:
- have a strong HOOK within its first 3 seconds,
- contain a clear PAYOFF (insight, punchline, emotional beat, or resolution),
- start and end on clean sentence boundaries (do not cut mid-sentence),
- stand alone without external context.

Return ONLY a JSON object with a "highlights" array. No prose, no markdown \
fences, no commentary. Schema:
{{"highlights": [{{"start": <float seconds>, "end": <float seconds>, \
"hook_title": <string>, "rationale": <string>, "score": <float 0.0-1.0>}}]}}

`score` is your confidence that this is a great standalone short. Prefer fewer, \
higher-quality picks over many mediocre ones, but every pick MUST satisfy the \
length requirement above."""

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
        api_key = self.settings.resolved_llm_key
        if not api_key:
            raise GrokError(
                "No LLM API key set. Add GROQ_API_KEY (or XAI_API_KEY / LLM_API_KEY) "
                "to .env (see .env.example)."
            )
        from openai import OpenAI  # lazy import

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.config.llm.base_url,
            timeout=self.config.llm.request_timeout_s,
        )

    # --- low-level call (with rate-limit backoff) ------------------------
    def _chat_json(self, model: str, system: str, user: str) -> Any:
        llm = self.config.llm
        last_exc: Optional[Exception] = None
        for attempt in range(llm.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    temperature=llm.temperature,
                    response_format={"type": "json_object"},  # force JSON (§4.3)
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                content = resp.choices[0].message.content or ""
                return _parse_json_loose(content)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _is_rate_limit(exc) or attempt >= llm.max_retries:
                    raise
                wait = _retry_after(exc) or llm.backoff_base_s * (2 ** attempt)
                log_event(
                    "llm", "rate_limited", level="warning",
                    attempt=attempt + 1, wait_s=round(wait, 1),
                )
                time.sleep(wait)
        raise last_exc  # pragma: no cover

    # --- highlight selection (§4.3) -------------------------------------
    def select_highlights(
        self,
        transcript: list[dict],
        context: str = "",
        min_duration_s: Optional[float] = None,
        max_duration_s: Optional[float] = None,
    ) -> list[Segment]:
        llm = self.config.llm
        # Tell the model the target length so it returns full-length clips, not
        # one-liners we'd later reject (the duration filter lives downstream).
        hl = self.config.highlight
        min_s = min_duration_s if min_duration_s is not None else hl.min_duration_s
        max_s = max_duration_s if max_duration_s is not None else hl.max_duration_s
        system = highlight_system_prompt(min_s, max_s)
        # Compact, line-grouped transcript -> far fewer tokens than per-word.
        lines = _format_transcript_lines(transcript, llm.words_per_line)
        chunks = _chunk_lines(lines, llm.max_input_tokens)

        if len(chunks) > 1:
            log_event("llm", "transcript_chunked", chunks=len(chunks),
                      budget_tokens=llm.max_input_tokens)

        all_segments: list[Segment] = []
        for i, chunk in enumerate(chunks):
            chunk_text = "\n".join(chunk)
            user = (
                (f"Video context: {context}\n\n" if context else "")
                + (f"(Transcript part {i + 1} of {len(chunks)}.)\n" if len(chunks) > 1 else "")
                + "Transcript ([start-end] text):\n"
                + chunk_text
                + "\n\nReturn the JSON object of highlights now."
            )
            data = self._chat_json(llm.highlight_model, system, user)
            all_segments.extend(_parse_segments(_coerce_array(data)))

            # Pace under the TPM budget before the next request (§4.3).
            if i < len(chunks) - 1:
                est = _estimate_tokens(chunk_text) + 600  # + system/output overhead
                pause = min(60.0, 60.0 * est / max(1, llm.tpm_limit))
                if pause > 0:
                    log_event("llm", "pacing", sleep_s=round(pause, 1))
                    time.sleep(pause)

        return _dedupe_segments(all_segments)

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

def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Good enough for budgeting."""
    return max(1, len(text) // 4)


def _format_transcript_lines(transcript: list[dict], words_per_line: int) -> list[str]:
    """Group words into `[start-end] text` lines -- one timestamp per group keeps
    enough timing for boundary selection at a fraction of the per-word tokens."""
    lines: list[str] = []
    for i in range(0, len(transcript), max(1, words_per_line)):
        group = transcript[i : i + words_per_line]
        if not group:
            continue
        start = group[0]["start"]
        end = group[-1]["end"]
        text = " ".join(w["word"].strip() for w in group)
        lines.append(f"[{start:.2f}-{end:.2f}] {text}")
    return lines


def _chunk_lines(lines: list[str], max_tokens: int) -> list[list[str]]:
    """Split transcript lines into chunks that each stay under `max_tokens`."""
    chunks: list[list[str]] = []
    current: list[str] = []
    running = 0
    for line in lines:
        t = _estimate_tokens(line) + 1
        if current and running + t > max_tokens:
            chunks.append(current)
            current = []
            running = 0
        current.append(line)
        running += t
    if current:
        chunks.append(current)
    return chunks or [[]]


def _parse_segments(items: list[dict]) -> list[Segment]:
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


def _dedupe_segments(segments: list[Segment]) -> list[Segment]:
    """Drop near-duplicate windows that can appear across overlapping chunks."""
    out: list[Segment] = []
    for seg in sorted(segments, key=lambda s: s.score, reverse=True):
        if any(abs(seg.start - k.start) < 1.0 and abs(seg.end - k.end) < 1.0 for k in out):
            continue
        out.append(seg)
    return out


def _is_rate_limit(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in (429, 413):
        return True
    text = str(exc).lower()
    return "rate_limit" in text or "tokens per minute" in text or "request too large" in text


def _retry_after(exc: Exception) -> Optional[float]:
    """Honour a Retry-After header / 'try again in Xs' hint if present."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            ra = resp.headers.get("retry-after")
            if ra:
                return float(ra)
        except Exception:  # noqa: BLE001
            pass
    m = re.search(r"try again in ([0-9.]+)s", str(exc))
    if m:
        return float(m.group(1))
    return None


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
