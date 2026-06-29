"""YouTube OAuth 2.0 installed-app flow (§4.8, §5 `clipper auth`).

Runs the one-time consent flow using the client-secret JSON, caches the
resulting credentials to YOUTUBE_TOKEN_PATH, and refreshes them on demand. No
secrets are written to the repo -- token path is gitignored (§1).
"""

from __future__ import annotations

import os
from pathlib import Path

from ..config import get_settings

# Upload scope only -- least privilege.
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


class YouTubeAuthError(RuntimeError):
    pass


def _load_credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    settings = get_settings()
    token_path = Path(settings.youtube_token_path)
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json(), encoding="utf-8")
        return creds
    return None


def run_oauth_flow():
    """Interactive consent -> cache token. Called by `clipper auth`."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    settings = get_settings()
    secrets_path = Path(settings.youtube_client_secrets)
    if not secrets_path.exists():
        raise YouTubeAuthError(
            f"OAuth client secrets not found at '{secrets_path}'. Download a Desktop "
            "OAuth client JSON from Google Cloud Console and set YOUTUBE_CLIENT_SECRETS."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), SCOPES)
    creds = flow.run_local_server(port=0)
    Path(settings.youtube_token_path).write_text(creds.to_json(), encoding="utf-8")
    return creds


def get_youtube_service():
    """Build an authorized youtube API client, or raise if not yet authed."""
    from googleapiclient.discovery import build

    creds = _load_credentials()
    if creds is None:
        raise YouTubeAuthError(
            "Not authorized. Run `clipper auth` first to complete the YouTube OAuth flow."
        )
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def is_authorized() -> bool:
    settings = get_settings()
    return os.path.exists(settings.youtube_token_path)
