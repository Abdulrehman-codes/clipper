"""transcribe node (§4.2).

WhisperX (word-level timestamps + alignment), language auto-detect, with a
documented fallback to faster-whisper. Cached to runs/<id>/transcript.json and
skipped if present (§1 idempotency).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..config import cache_path
from ..logging_utils import log_event
from ..types import ClipState, Word


def _transcribe_whisperx(audio_path: str, model_size: str = "base") -> list[Word]:
    import whisperx  # lazy import

    device = "cuda" if os.environ.get("CLIPPER_DEVICE") == "cuda" else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)
    language = result["language"]

    # Word-level alignment.
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )

    words: list[Word] = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            if w.get("start") is None or w.get("end") is None:
                continue
            words.append(Word(
                word=str(w.get("word", "")).strip(),
                start=float(w["start"]),
                end=float(w["end"]),
                score=w.get("score"),
            ))
    return words


def _transcribe_faster_whisper(audio_path: str, model_size: str = "base") -> list[Word]:
    """Fallback (§2). word_timestamps=True gives word-level timing."""
    from faster_whisper import WhisperModel  # lazy import

    device = "cuda" if os.environ.get("CLIPPER_DEVICE") == "cuda" else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, _info = model.transcribe(audio_path, word_timestamps=True)
    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            words.append(Word(
                word=str(w.word).strip(),
                start=float(w.start),
                end=float(w.end),
                score=float(getattr(w, "probability", 0.0)) or None,
            ))
    return words


def transcribe(state: ClipState) -> dict:
    audio_path = state.get("audio_path")
    if not audio_path:
        raise ValueError("transcribe: no audio_path in state (download must run first)")

    # Cache the transcript next to the cached source (per-video, reused across runs).
    transcript_file = cache_path(state["source_url"]) / "transcript.json"
    if transcript_file.exists():
        words = json.loads(transcript_file.read_text(encoding="utf-8"))
        log_event("transcribe", "reuse_cached", words=len(words), file=str(transcript_file))
        return {"transcript": words}

    model_size = os.environ.get("CLIPPER_WHISPER_MODEL", "base")
    log_event("transcribe", "start", backend="whisperx", model=model_size)
    try:
        words = _transcribe_whisperx(audio_path, model_size)
        backend = "whisperx"
    except Exception as exc:  # noqa: BLE001 -- documented fallback path (§2)
        log_event("transcribe", "whisperx_failed", level="warning", error=str(exc))
        words = _transcribe_faster_whisper(audio_path, model_size)
        backend = "faster_whisper"

    payload = [w.model_dump() for w in words]
    transcript_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_event("transcribe", "done", backend=backend, words=len(payload), file=str(transcript_file))
    return {"transcript": payload}
