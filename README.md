# clipper — long-form → short-clip pipeline

A [LangGraph](https://langchain-ai.github.io/langgraph/)-orchestrated pipeline that takes a YouTube URL, transcribes it, uses an LLM (Grok via xAI) to pick the strongest segments, cuts them, reframes to vertical + horizontal, burns in word-level captions, generates metadata, and uploads to YouTube **as private drafts for human review**.

The engine is content-agnostic. **Rights/eligibility of the source is the operator's responsibility** — see [Operator responsibility](#operator-responsibility).

---

## ⚠️ Operator responsibility

> This tool downloads and re-encodes video. Only run it on content you own, are licensed to use, or that constitutes fair use **with added transformative commentary**. Uploading clips of third-party videos with no added value is copyright infringement and violates YouTube's reused/inauthentic-content policy; enforcement is at the channel level (termination). Default visibility is private; you review every clip before it goes anywhere. Your inputs, your liability.

### YouTube API audit warning

API uploads from an un-audited Google Cloud project may be **restricted to private regardless** of what you request. That's fine here — we *only* upload private drafts — but to upload at all the operator must complete the OAuth consent flow (`clipper auth`) and, for anything beyond testing, Google's API verification.

---

## Hard constraints (by design)

- **Default upload visibility = `private`.** There is **no publish code path** in this MVP. `privacyStatus` is hard-wired to `private` in [`clipper/youtube/upload.py`](clipper/youtube/upload.py).
- **Ingest gate.** Processing aborts unless you pass `--i-have-rights` (or answer `y` to the interactive prompt). The URL + timestamp + confirmation are logged to `runs/<run_id>/rights.json`.
- **Idempotent runs.** Every run gets a `run_id`; all artifacts live under `runs/<run_id>/`. Re-running reuses cached download/transcription/segments.
- **No secrets in code.** All keys come from `.env` (`XAI_API_KEY`, YouTube OAuth client-secret path). `.env` and token files are gitignored.

---

## Stack

| Concern | Choice |
|---|---|
| Orchestration | **LangGraph** — `StateGraph`, one node per stage, retry policy per node, `Send` fan-out |
| Download | **yt-dlp** (bestvideo+bestaudio, remux to mp4) |
| Transcription | **WhisperX** (word-level timestamps + alignment); fallback **faster-whisper** |
| LLM | **Grok via xAI** (OpenAI-SDK-compatible, `base_url=https://api.x.ai/v1`) |
| Cutting / encode | **ffmpeg** (subprocess) |
| Reframe | **MediaPipe** face detection + OpenCV for the 9:16 active-speaker crop |
| Captions | Word-level **ASS** karaoke subtitles, burned in via ffmpeg |
| Upload | **YouTube Data API v3** (`google-api-python-client`, OAuth 2.0 installed-app flow) |
| Config | `pydantic-settings` + `config.yaml` |

Python ≥ 3.11. **ffmpeg must be on PATH.**

> **Model strings change.** `grok-4.1-fast` (highlight selection) and `grok-4.3` (metadata) are set in [`config.yaml`](config.yaml) → `llm.*`. **Verify the exact strings in the [xAI console](https://console.x.ai)** before a real run.

---

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   *nix: source .venv/bin/activate

pip install -e .                 # core: graph + CLI + download + Grok
pip install -e ".[transcribe]"   # + WhisperX / faster-whisper
pip install -e ".[reframe]"      # + MediaPipe / OpenCV
pip install -e ".[youtube]"      # + YouTube Data API client
pip install -e ".[all]"          # everything + dev tools
```

Heavy model dependencies (WhisperX, MediaPipe) are isolated into extras so the
lighter pipeline slices run without installing everything. If WhisperX/MediaPipe
are absent, the pipeline degrades gracefully (faster-whisper fallback; static
center-crop reframe).

Copy `.env.example` → `.env` and fill in your keys:

```bash
cp .env.example .env
```

---

## Usage

```bash
# Full pipeline (renders + uploads private drafts):
clipper run "https://www.youtube.com/watch?v=..." --i-have-rights

# Tune selection, render only (no upload):
clipper run "<url>" --i-have-rights --max-clips 5 --min-score 0.7 \
                    --formats both --no-upload

# One-time YouTube OAuth (needed before uploads):
clipper auth
```

| Flag | Meaning |
|---|---|
| `--i-have-rights` | Skips the interactive rights prompt (still logged). Required to proceed. |
| `--max-clips N` | Cap the number of clips (overrides `config.yaml`). |
| `--min-score 0.6` | Minimum highlight score (0–1). |
| `--formats both\|vertical\|horizontal` | Which renders to produce. |
| `--no-upload` | Stop after rendering; just write files. |

### Output layout

```
runs/<run_id>/
  rights.json            # ingest-gate audit record
  source.mp4  audio.wav  # download + extracted 16kHz mono audio
  transcript.json        # word-level timestamps (cached)
  segments.json          # LLM-selected highlights (cached)
  clip_00_base.mp4              # the cut (16:9)
  clip_00_vertical.mp4          # 9:16 reframe
  clip_00_*_captioned.mp4       # caption-burned renders (what gets uploaded)
  clip_00.ass                   # karaoke subtitle source
  report.md  report.json        # per-clip summary + upload status
  run.log                       # structured JSONL events
```

---

## Pipeline

```
ingest_gate → download → transcribe → select_highlights
   → ⇉ fan-out per segment: (cut → reframe → caption → metadata)
   → upload_drafts → report
```

Per-segment work fans out via LangGraph `Send` so clips render in parallel. Each
node carries a retry policy; per-clip failures are isolated so one bad clip never
aborts the run.

### Quota reality

Each `videos.insert` costs ≈ **1600 units** against a default **10k/day** quota
≈ **6 uploads/day**. `upload_drafts` budgets per run, uploads resumably with
exponential backoff, and on a quota error **saves everything locally and exits
gracefully** so a later run resumes where it left off.

---

## Development

```bash
pip install -e ".[dev]"
pytest        # unit tests (no network / no heavy models required)
ruff check .
```

The build follows the spec's ship-working-slices order: download+transcribe →
highlight selection → cut → caption → reframe → metadata+report → auth+upload,
wired into LangGraph once the stages worked as plain functions.

---

## Deferred (v2)

Diarization-driven speaker switching, B-roll insertion, commentary/voiceover
layer (the transformative path), multi-channel scheduling, web UI.

## License

MIT — see [LICENSE](LICENSE).
