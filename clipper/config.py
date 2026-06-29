"""Configuration: secrets via env (.env) + tunables via config.yaml (§2, §6).

`Settings` (pydantic-settings) holds *secrets / paths* read from the environment.
`Config` holds *tunables* read from config.yaml. Keeping them separate enforces
the §1 constraint: no secrets in config files that get committed.
"""

from __future__ import annotations

import os
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

# Load .env once at import so os.environ is populated for Settings.
load_dotenv(REPO_ROOT / ".env")


class Settings(BaseSettings):
    """Secrets and credential paths -- never written to disk by us (§1)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    xai_api_key: Optional[str] = None
    youtube_client_secrets: str = "client_secret.json"
    youtube_token_path: str = "youtube_token.json"


# --- config.yaml sections (§6) -------------------------------------------

class HighlightConfig(BaseModel):
    min_duration_s: float = 15
    max_duration_s: float = 90
    min_score: float = 0.6
    max_clips: int = 8


class LLMConfig(BaseModel):
    base_url: str = "https://api.x.ai/v1"
    highlight_model: str = "grok-4.1-fast"
    metadata_model: str = "grok-4.3"
    temperature: float = 0.4
    request_timeout_s: float = 120


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
