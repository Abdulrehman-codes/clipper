"""Configuration: secrets via env (.env) + tunables via config.yaml (§2, §6).

`Settings` (pydantic-settings) holds *secrets / paths* read from the environment.
`Config` holds *tunables* read from config.yaml. Keeping them separate enforces
the §1 constraint: no secrets in config files that get committed.
"""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
RUNS_DIR = REPO_ROOT / "runs"
# Per-video cache for reusable heavy artifacts (download, audio, transcript) so
# re-running the SAME URL never re-downloads or re-transcribes (§1 idempotency).
CACHE_DIR = REPO_ROOT / "cache"

# Load .env once at import so os.environ is populated for Settings.
load_dotenv(REPO_ROOT / ".env")


class Settings(BaseSettings):
    """Secrets and credential paths -- never written to disk by us (§1)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM key. The pipeline is OpenAI-SDK-compatible, so any compatible provider
    # works (xAI Grok, Groq, ...). We accept several env var names and resolve in
    # priority order so an existing key just works regardless of provider.
    xai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    llm_api_key: Optional[str] = None
    youtube_client_secrets: str = "client_secret.json"
    youtube_token_path: str = "youtube_token.json"

    @property
    def resolved_llm_key(self) -> Optional[str]:
        return self.llm_api_key or self.groq_api_key or self.xai_api_key


# --- config.yaml sections (§6) -------------------------------------------

class HighlightConfig(BaseModel):
    min_duration_s: float = 15
    max_duration_s: float = 90
    min_score: float = 0.6
    max_clips: int = 8


class LLMConfig(BaseModel):
    # provider is informational; behaviour is driven by base_url + model strings.
    provider: str = "groq"
    base_url: str = "https://api.x.ai/v1"
    highlight_model: str = "grok-4.1-fast"
    metadata_model: str = "grok-4.3"
    temperature: float = 0.4
    request_timeout_s: float = 120
    # --- rate-limit / context handling (§4.3 "chunk if >context") ----------
    # Max transcript tokens to send per highlight request. Keep under the
    # provider's tokens-per-minute (TPM) limit. Groq free tier = 12k TPM, so a
    # conservative budget here lets a long transcript be chunked + paced.
    max_input_tokens: int = 5000
    # Provider TPM budget, used to pace requests so we don't trip 429s.
    tpm_limit: int = 12000
    # Retries + base backoff (seconds) for rate-limit (429 / 413) errors.
    max_retries: int = 6
    backoff_base_s: float = 5
    # Words grouped per transcript line (fewer timestamps = far fewer tokens).
    words_per_line: int = 12


class CaptionConfig(BaseModel):
    font: str = "Arial"
    font_size: int = 18
    primary_color: str = "&H00FFFFFF"
    highlight_color: str = "&H0000F0FF"
    outline: int = 3
    shadow: int = 1
    max_chars_per_line: int = 28
    max_lines: int = 2
    vertical_anchor: float = 0.78


class ReframeConfig(BaseModel):
    target_vertical: str = "9:16"
    target_horizontal: str = "16:9"
    smoothing: float = 0.15
    max_velocity: float = 0.04
    detection_confidence: float = 0.5
    fallback_fill: str = "blur"


class EncodeConfig(BaseModel):
    video_codec: str = "libx264"
    crf: int = 20
    preset: str = "veryfast"
    audio_codec: str = "aac"
    audio_bitrate: str = "160k"
    fps: int = 30


class TranscribeConfig(BaseModel):
    # base|small|medium|large-v3 -- bigger = far better (esp. non-English), slower.
    model: str = "small"
    # ISO code to force a language ("ur", "hi", "en", ...); None = auto-detect.
    language: Optional[str] = None
    # "transcribe" (keep original language) | "translate" (to English).
    task: str = "transcribe"
    device: str = "auto"         # auto | cpu | cuda
    compute_type: Optional[str] = None  # None = auto (int8 cpu / float16 cuda)


class OutputConfig(BaseModel):
    formats: str = "both"  # both | vertical | horizontal


class UploadConfig(BaseModel):
    privacy_status: str = "private"
    category_id: str = "22"
    made_for_kids: bool = False
    insert_cost_units: int = 1600
    daily_quota_units: int = 10000
    max_retries: int = 5
    backoff_base_s: float = 2


class Config(BaseModel):
    highlight: HighlightConfig = HighlightConfig()
    llm: LLMConfig = LLMConfig()
    transcribe: TranscribeConfig = TranscribeConfig()
    caption: CaptionConfig = CaptionConfig()
    reframe: ReframeConfig = ReframeConfig()
    encode: EncodeConfig = EncodeConfig()
    output: OutputConfig = OutputConfig()
    upload: UploadConfig = UploadConfig()

    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config.load()


def runs_dir() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR


def run_path(run_id: str) -> Path:
    p = runs_dir() / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def video_key(url: str) -> str:
    """Stable cache key for a video URL.

    Uses the YouTube video id when we can extract one (so youtu.be/<id>,
    watch?v=<id>, and /shorts/<id> for the same video share a cache), else a
    hash of the URL."""
    url = url.strip()
    patterns = [
        r"[?&]v=([\w-]{11})",                              # watch?v=<id> / &v=<id>
        r"youtu\.be/([\w-]{11})",                          # youtu.be/<id>
        r"youtube\.com/(?:shorts|embed|v)/([\w-]{11})",    # /shorts|embed|v/<id>
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:11]


def cache_path(url: str) -> Path:
    """Directory holding cached source.mp4 / audio.wav / transcript.json for a URL."""
    p = CACHE_DIR / video_key(url)
    p.mkdir(parents=True, exist_ok=True)
    return p
