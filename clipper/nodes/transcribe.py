"""transcribe node (§4.2).

WhisperX (word-level timestamps + alignment), with a documented fallback to
faster-whisper. Model size / language / task come from config.yaml -> transcribe
(env overrides supported). Cached per-video and skipped if present (§1).

For non-English audio, use a larger model (medium / large-v3) and ideally set
`transcribe.language` -- a small model with auto-detect produces garbage.
"""

from __future__ import annotations

import json
import os

from ..config import TranscribeConfig, cache_path, get_config
from ..logging_utils import log_event
from ..types import ClipState, Word


def _cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def _resolve(cfg: TranscribeConfig, force_device: str | None = None) -> tuple[str, str, str, str | None]:
    """Return (model, device, compute_type, language) honouring env overrides."""
    model = os.environ.get("CLIPPER_WHISPER_MODEL", cfg.model)
    device = force_device or os.environ.get("CLIPPER_DEVICE", cfg.device)
    if device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    compute_type = cfg.compute_type or ("float16" if device == "cuda" else "int8")
    language = os.environ.get("CLIPPER_LANGUAGE", cfg.language or "") or None
    return model, device, compute_type, language


def _transcribe_whisperx(audio_path: str, cfg: TranscribeConfig) -> list[Word]:
    import whisperx  # lazy import

    model_size, device, compute_type, language = _resolve(cfg)
    model = whisperx.load_model(
        model_size, device, compute_type=compute_type, language=language,
        asr_options={"task": cfg.task},
    )
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16, language=language)
    lang = result["language"]

    # Word-level alignment (alignment models are language-specific).
    align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
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


def _transcribe_faster_whisper(audio_path: str, cfg: TranscribeConfig,
                               force_device: str | None = None) -> list[Word]:
    """Fallback (§2). word_timestamps=True gives word-level timing."""
    from faster_whisper import WhisperModel  # lazy import

    model_size, device, compute_type, language = _resolve(cfg, force_device)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        audio_path, word_timestamps=True, language=language, task=cfg.task,
    )
    log_event("transcribe", "language", detected=getattr(info, "language", language),
              forced=language, task=cfg.task)
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

    cfg = get_config().transcribe
    model_size, device, _ct, language = _resolve(cfg)

    # Cache key includes model+task+language so changing transcription settings
    # produces a FRESH transcript instead of reusing a worse one (e.g. the garbled
    # output from a tiny auto-detect model). Per-video, reused across runs.
    sig = f"{model_size}_{cfg.task}_{language or 'auto'}".replace("/", "-")
    transcript_file = cache_path(state["source_url"]) / f"transcript_{sig}.json"
    if transcript_file.exists():
        words = json.loads(transcript_file.read_text(encoding="utf-8"))
        log_event("transcribe", "reuse_cached", words=len(words), file=str(transcript_file))
        return {"transcript": words}

    log_event("transcribe", "start", backend="whisperx", model=model_size,
              device=device, language=language or "auto", task=cfg.task)
    try:
        words = _transcribe_whisperx(audio_path, cfg)
        backend = "whisperx"
    except Exception as exc:  # noqa: BLE001 -- documented fallback path (§2)
        log_event("transcribe", "whisperx_failed", level="warning", error=str(exc))
        try:
            words = _transcribe_faster_whisper(audio_path, cfg)
            backend = "faster_whisper"
        except Exception as gpu_exc:  # noqa: BLE001 -- GPU kernel/lib failure -> CPU
            if device == "cuda":
                log_event("transcribe", "cuda_failed", level="warning",
                          error=str(gpu_exc), retry="cpu")
                words = _transcribe_faster_whisper(audio_path, cfg, force_device="cpu")
                backend = "faster_whisper_cpu"
            else:
                raise

    payload = [w.model_dump() for w in words]
    transcript_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_event("transcribe", "done", backend=backend, words=len(payload), file=str(transcript_file))
    return {"transcript": payload}
