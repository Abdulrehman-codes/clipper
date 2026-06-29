# Project Spec: `clipper` — Long-form → Short-clip Pipeline

> **Paste this into Claude Code as the build brief.** It is a planning + implementation spec, not a one-shot prompt. Build incrementally, node by node, with a working CLI at every stage.

---

## 0. Goal

A LangGraph-orchestrated pipeline that takes a YouTube URL, transcribes it, uses an LLM to pick the strongest segments, cuts them, reframes to vertical + horizontal, burns in captions, generates metadata, and uploads to YouTube **as private drafts for human review**.

The engine is content-agnostic. **Rights/eligibility of the source is the operator's responsibility** — see §9.

---

## 1. Hard constraints (do not skip)

- **Default upload visibility = `private`.** Never auto-publish. There is no "publish" code path in the MVP at all.
- **Ingest gate:** before processing, require an explicit `--i-have-rights` flag (or interactive `y/N` confirm). Abort if absent. Log the URL + timestamp + confirmation to `runs/<run_id>/rights.json`.
- **Idempotent runs:** every run gets a `run_id`; all artifacts live under `runs/<run_id>/`. Re-running with the same URL reuses cached transcription.
- **No secrets in code.** All keys via `.env` (`XAI_API_KEY`, YouTube OAuth client secrets path).

---

## 2. Stack

| Concern | Choice |
|---|---|
| Orchestration | **LangGraph** (StateGraph, one node per stage, retry policy per node) |
| Download | **yt-dlp** (bestvideo+bestaudio, remux to mp4) |
| Transcription | **WhisperX** (word-level timestamps + alignment) — fallback `faster-whisper` |
| LLM (highlight + metadata) | **Grok via xAI**, OpenAI-SDK-compatible: `base_url="https://api.x.ai/v1"`. Use **`grok-4.1-fast`** for highlight selection (cheap, 2M ctx); `grok-4.3` optional for metadata polish. **Verify exact model strings in the xAI console — they change.** |
| Cutting / encode | **ffmpeg** (via `ffmpeg-python` or subprocess) |
| Reframe | **MediaPipe** face/pose for active-speaker crop; OpenCV for compositing |
| Captions | Word-level **ASS** subtitles (karaoke `\k` timing), burned in via ffmpeg |
| Upload | **YouTube Data API v3** (`google-api-python-client`, OAuth 2.0 installed-app flow) |
| Config | `pydantic-settings` + a `config.yaml` for tunables |

Python 3.11, `uv` or `pip`, Docker optional. Assume ffmpeg present on PATH.

---

## 3. LangGraph state

```python
class ClipState(TypedDict):
    run_id: str
    source_url: str
    rights_confirmed: bool
    source_path: str | None          # downloaded mp4
    transcript: list[Word] | None     # {word, start, end}
    segments: list[Segment] | None    # LLM-selected highlights
    clips: list[Clip]                 # rendered outputs + metadata
    errors: list[str]
```

Nodes (each with retry + structured logging):
`ingest_gate → download → transcribe → select_highlights → for each segment: (cut → reframe → caption → metadata) → upload_drafts → report`

Fan-out the per-segment work (LangGraph `Send` / map pattern) so clips render in parallel.

---

## 4. Node specs

### 4.1 `download`
- `yt-dlp` → `runs/<id>/source.mp4` (+ extract 16kHz mono wav for Whisper).
- Capture source title/channel/duration into state for later metadata context.

### 4.2 `transcribe`
- WhisperX, word-level timestamps, language auto-detect.
- Cache to `runs/<id>/transcript.json`; skip if present.

### 4.3 `select_highlights` (Grok)
- Send the **full timestamped transcript** (chunk if >context, but 2M ctx usually fits).
- System prompt: act as a short-form editor; find self-contained moments with a hook in the first 3s, a payoff, and a clean start/end on sentence boundaries.
- **Force JSON output** (no prose, no markdown fences). Parse and validate:
```json
[{"start": 12.4, "end": 48.9, "hook_title": "...", "rationale": "...", "score": 0.0-1.0}]
```
- Filter: 15–90s duration, score ≥ threshold (config), max N clips (config). Snap start/end to nearest word boundary from the transcript.

### 4.4 `cut`
- ffmpeg stream-copy where possible, re-encode only if needed. `-ss`/`-to` on snapped boundaries.

### 4.5 `reframe` (the hard one — keep MVP lean)
- **MVP:** run MediaPipe face detection per frame; compute a smoothed (EMA) crop box that follows the dominant face; produce a 9:16 crop. Letterbox/blur-fill if no face found.
- Also emit the **16:9** version (just the cut, no crop) since output = both.
- Smoothing matters more than accuracy — jittery crops look broken. Cap crop velocity.
- *Defer:* multi-speaker switching (diarization-driven crop). Leave a clean seam for it.

### 4.6 `caption`
- Build ASS from word timestamps, karaoke highlight, 1–2 lines, safe-area aware for 9:16.
- Burn into both renders via ffmpeg `subtitles=` filter. Font/size/position in `config.yaml`.

### 4.7 `metadata` (Grok)
- Per clip: title (≤60 chars, hook-style), description, 3–5 hashtags, suggested filename. JSON out.

### 4.8 `upload_drafts`
- YouTube Data API `videos.insert`, `privacyStatus="private"`.
- **Quota reality:** each insert ≈ 1600 units; default 10k/day ≈ 6 uploads/day. Make upload a separate, rate-limited, resumable step (queue + backoff). On quota error, save locally and exit gracefully.
- **Audit warning to surface in README:** API uploads from an un-audited project may be restricted to private regardless — fine here since we want private anyway, but the operator must complete OAuth + verification.
- `--no-upload` flag → stop after rendering, just write files.

### 4.9 `report`
- Write `runs/<id>/report.md`: every clip with path, title, score, rationale, upload status, video ID.

---

## 5. CLI

```
clipper run <youtube_url> --i-have-rights [--max-clips N] [--min-score 0.6] \
                          [--formats both|vertical|horizontal] [--no-upload]
clipper auth          # one-time YouTube OAuth
```

## 6. Config (`config.yaml`)
clip duration bounds, score threshold, max clips, caption style, crop smoothing factor, Grok model strings, output formats.

## 7. Repo layout
```
clipper/
  graph.py          # LangGraph wiring
  nodes/            # one file per node
  llm/grok.py       # xAI client + prompts + JSON parsing
  video/            # cut, reframe, caption
  youtube/          # auth + upload
  config.py
  cli.py
runs/               # gitignored artifacts
.env.example
README.md
```

## 8. Build order (ship working slices)
1. `download` + `transcribe` → dump transcript. **Test on a 5-min video.**
2. `select_highlights` (Grok) → print chosen segments. Tune the prompt here.
3. `cut` → produce raw clips.
4. `caption` → captioned 16:9 clips (skip reframe first).
5. `reframe` → add 9:16.
6. `metadata` + `report`.
7. `auth` + `upload_drafts` last.
Wrap in LangGraph only once steps 1–3 work as plain functions.

## 9. Operator responsibility (put in README, verbatim-ish)
> This tool downloads and re-encodes video. Only run it on content you own, are licensed to use, or that constitutes fair use **with added transformative commentary**. Uploading clips of third-party videos with no added value is copyright infringement and violates YouTube's reused/inauthentic-content policy; enforcement is at the channel level (termination). Default visibility is private; you review every clip before it goes anywhere. Your inputs, your liability.

## 10. Deferred (v2)
Diarization-driven speaker switching, B-roll insertion, commentary/voiceover layer (the transformative path), multi-channel scheduling, web UI.
