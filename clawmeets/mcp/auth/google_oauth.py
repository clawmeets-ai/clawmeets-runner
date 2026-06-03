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

# Relax oauthlib's strict scope-equality check on token exchange. Google
# legitimately returns a broader scope set than requested when the same
# Google account has prior incremental grants to the same OAuth client
# (e.g. user installs Gmail, consents, then installs Calendar — Google
# bundles both into the second token). Without this, fetch_token raises
# `Scope has changed from "gmail.modify" to "calendar gmail.modify"`.
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")


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


def build_authorization_url(
    scopes: list[str],
    redirect_uri: str,
    state: str,
    client_secrets: Optional[Path] = None,
) -> tuple[str, str]:
    """Build a Google OAuth consent URL for the relay flow.

    Unlike ``run_installed_flow``, this does NOT bind a local callback server
    or open a browser. The runner calls this to mint a URL the user opens in
    their own browser; Google redirects the user to ``redirect_uri`` (the
    ClawMeets server's ``/oauth/mcp/callback``) carrying the same ``state``
    token. The runner later receives the auth code over its WebSocket and
    finishes the exchange via ``exchange_code``.

    Returns ``(auth_url, code_verifier)``. Google's ``Flow.authorization_url``
    enables PKCE by default — it generates a fresh ``code_verifier`` and
    embeds the corresponding ``code_challenge`` in the URL. The token
    exchange must present the same ``code_verifier``, so the caller is
    responsible for stashing it on the runner side and passing it back to
    ``exchange_code``. The verifier is a client-side secret and must not
    cross the relay server.

    The OAuth client at ``client_secrets`` must be a Google "Web application"
    client with ``redirect_uri`` registered as an authorized redirect URI.
    """
    try:
        from google_auth_oauthlib.flow import Flow
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
            f"For the relay flow, create an OAuth client (Web application) in "
            f"Google Cloud Console with the ClawMeets server's "
            f"/oauth/mcp/callback registered as an authorized redirect URI, "
            f"then save the credentials JSON to that path."
        )

    flow = Flow.from_client_secrets_file(str(secrets_path), scopes=scopes)
    flow.redirect_uri = redirect_uri
    # Don't pass include_granted_scopes — it asks Google to fold in any other
    # scopes this OAuth client has already obtained for this Google account,
    # which makes the returned token broader than what we requested and trips
    # oauthlib's strict scope check at fetch_token time. Each MCP install
    # should yield a self-scoped token.
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        state=state,
    )
    code_verifier = flow.code_verifier or ""
    return auth_url, code_verifier


def exchange_code(
    code: str,
    scopes: list[str],
    redirect_uri: str,
    token_path: Path,
    code_verifier: str | None = None,
    client_secrets: Optional[Path] = None,
) -> None:
    """Exchange an authorization code for tokens and write them to disk.

    Counterpart to ``build_authorization_url``: called by the runner once the
    server forwards the OAuth ``code`` over WebSocket. Writes the resulting
    token JSON at ``token_path`` with mode 0600. The server never sees the
    access or refresh tokens.

    ``code_verifier`` must be the same value returned from the matching
    ``build_authorization_url`` call (PKCE). Required when the upstream
    URL was built with PKCE enabled (Google's default).
    """
    try:
        from google_auth_oauthlib.flow import Flow
    except ImportError as exc:
        raise GoogleOAuthError(
            "google-auth-oauthlib is required for MCP OAuth flows but missing — "
            "the clawmeets runner should bundle it by default. "
            "Try: pip install --upgrade clawmeets"
        ) from exc

    secrets_path = _resolve_client_secrets(client_secrets)
    if not secrets_path.exists():
        raise GoogleOAuthError(
            f"Google OAuth client secrets not found at {secrets_path}."
        )

    flow = Flow.from_client_secrets_file(str(secrets_path), scopes=scopes)
    flow.redirect_uri = redirect_uri
    if code_verifier:
        flow.code_verifier = code_verifier
    flow.fetch_token(code=code)
    creds = flow.credentials

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    os.chmod(token_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    logger.info(f"Wrote Google OAuth token to {token_path} (0600) via relay flow")


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
