# SPDX-License-Identifier: MIT
"""
clawmeets/mcp/auth/google_oauth.py

Google "installed application" OAuth flow for MCP servers.

Runs entirely on the runner machine: opens the default browser to Google's
consent screen, listens on an ephemeral localhost port for the redirect, and
writes the token to disk at mode 0600. Tokens never transit the ClawMeets
server.
"""
from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path
from typing import Optional

logger = logging.getLogger("clawmeets.runner.mcp.auth")

DEFAULT_CLIENT_SECRETS = Path.home() / ".clawmeets" / "google_oauth_client.json"


class GoogleOAuthError(RuntimeError):
    pass


def _resolve_client_secrets(explicit: Optional[Path]) -> Path:
    """Find the Google OAuth installed-app client secrets file.

    Precedence:
      1. --credentials flag (explicit)
      2. CLAWMEETS_GOOGLE_OAUTH_CREDENTIALS env var
      3. ~/.clawmeets/google_oauth_client.json
    """
    if explicit:
        return explicit
    env = os.environ.get("CLAWMEETS_GOOGLE_OAUTH_CREDENTIALS")
    if env:
        return Path(env).expanduser()
    return DEFAULT_CLIENT_SECRETS


def run_installed_flow(
    scopes: list[str],
    token_path: Path,
    client_secrets: Optional[Path] = None,
) -> None:
    """Run Google's installed-app OAuth flow and write the token to disk.

    Args:
        scopes: OAuth scopes to request (e.g. ["https://www.googleapis.com/auth/gmail.modify"]).
        token_path: Absolute path where the resulting token JSON will be saved.
            Parent directories are created; file is chmod 0600 after write.
        client_secrets: Path to the installed-app client-secrets JSON downloaded
            from Google Cloud Console. If None, resolve via env + default path.

    Raises:
        GoogleOAuthError: if dependencies aren't installed or the client secrets
            file is missing.
    """
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise GoogleOAuthError(
            "google-auth-oauthlib is required for MCP OAuth flows but missing — "
            "the clawmeets runner should bundle it by default. "
            "Try: pip install --upgrade clawmeets"
        ) from exc

    secrets_path = _resolve_client_secrets(client_secrets)
    if not secrets_path.exists():
        raise GoogleOAuthError(
            f"Google OAuth client secrets not found at {secrets_path}.\n"
            f"Create an OAuth client (Desktop app) in Google Cloud Console, "
            f"download the credentials JSON, and save it to that path (or pass "
            f"--credentials)."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), scopes)
    # port=0 asks the OS for an ephemeral port; Google's Desktop OAuth client
    # accepts any loopback redirect.
    creds = flow.run_local_server(port=0, open_browser=True)

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    os.chmod(token_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    logger.info(f"Wrote Google OAuth token to {token_path} (0600)")


def load_credentials(token_path: Path, scopes: list[str]):
    """Load and refresh cached credentials. Returns a google.oauth2.Credentials."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError as exc:
        raise GoogleOAuthError(
            "google-auth is required for MCP OAuth flows but missing — "
            "the clawmeets runner should bundle it by default. "
            "Try: pip install --upgrade clawmeets"
        ) from exc

    if not token_path.exists():
        raise GoogleOAuthError(
            f"No cached token at {token_path}. "
            f"Run `clawmeets mcp auth <name>` to authenticate first."
        )

    data = json.loads(token_path.read_text())
    creds = Credentials.from_authorized_user_info(data, scopes=scopes)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json())
        os.chmod(token_path, stat.S_IRUSR | stat.S_IWUSR)
    return creds
