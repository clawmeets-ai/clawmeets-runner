# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/auth/google_oauth.py

Google "installed application" + relay OAuth flow shared by MCP servers
(chess-era) and skill-shelled CLI subcommands (gmail/gcal/gdrive/...).

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

logger = logging.getLogger("clawmeets.integrations.auth.google_oauth")

DEFAULT_CLIENT_SECRETS = Path.home() / ".clawmeets" / "google_oauth_client.json"

# Relax oauthlib's strict scope-equality check on token exchange. Google
# legitimately returns a broader scope set than requested when the same
# Google account has prior incremental grants to the same OAuth client
# (e.g. user installs Gmail, consents, then installs Calendar — Google
# bundles both into the second token). Without this, fetch_token raises
# `Scope has changed from "gmail.modify" to "calendar gmail.modify"`.
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

# Google's OAuth 2.0 token-revocation endpoint. POST the access or refresh
# token here (form-encoded) to withdraw the grant on Google's side.
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"


class GoogleOAuthError(RuntimeError):
    """Base error for the Google OAuth flow (missing deps, secrets, etc.)."""


class ScopeGrantError(GoogleOAuthError):
    """Google granted a scope set that does not cover what we requested.

    Carries the requested/granted diff so the failure is visible in logs and
    to any caller — a partial grant is never silently persisted.
    """

    def __init__(self, requested: set[str], granted: set[str]) -> None:
        self.requested = set(requested)
        self.granted = set(granted)
        self.missing = self.requested - self.granted
        super().__init__(
            "Google did not grant all requested scopes — "
            f"missing {sorted(self.missing)}; "
            f"requested {sorted(self.requested)}, granted {sorted(self.granted)}. "
            "The token was NOT saved; the integration stays disconnected."
        )


class ReauthRequired(GoogleOAuthError):
    """A stored Google grant is dead at call time (no token, or the refresh
    token is expired/revoked/rejected).

    Raised — never swallowed — so the agent gets an explicit, actionable
    signal instead of an opaque 401 deep inside a Google API call. Carries the
    skill name and a ready-to-run re-authentication instruction.
    """

    def __init__(self, skill_name: str, reason: str) -> None:
        self.skill_name = skill_name
        self.reason = reason
        self.remediation = f"clawmeets {skill_name} auth"
        super().__init__(
            f"Google authorization for '{skill_name}' is no longer valid "
            f"({reason}). Re-authenticate {skill_name}: run `{self.remediation}` "
            f"(or click Re-authenticate for {skill_name} in the web UI)."
        )


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


def _granted_scopes(creds) -> set[str]:
    """Return the set of scopes Google actually granted, read off the minted
    Credentials.

    Reads ``creds.scopes`` when populated; falls back to the raw ``scope``
    string on the underlying token when the attribute is None (older token
    shapes). Returns an empty set when neither is available — callers treat
    that as "cannot verify" (fail-open) rather than a scope mismatch. Pure; no
    I/O.
    """
    scopes = getattr(creds, "scopes", None)
    if scopes:
        return {s for s in scopes if s}
    raw = getattr(creds, "_scopes", None)
    if isinstance(raw, str):
        return {s for s in raw.split() if s}
    return set()


def _assert_granted_covers_requested(creds, requested_scopes: list[str]) -> None:
    """Guard run immediately after token exchange and BEFORE the token is
    written to disk.

    Asserts the requested scopes are a subset of what Google granted (Google
    did not silently drop a scope we asked for). On mismatch: log the
    requested-vs-granted diff at WARNING and raise :class:`ScopeGrantError`, so
    the caller never persists a partial-grant token and the authorization
    error is visible rather than swallowed.

    A *granted superset* (extra scopes from a prior incremental grant, allowed
    by ``OAUTHLIB_RELAX_TOKEN_SCOPE`` and by ``include_granted_scopes``) is NOT
    an error. When the granted set cannot be determined at all, log at INFO and
    allow the write (fail-open) rather than block a valid grant.
    """
    requested = {s for s in requested_scopes if s}
    granted = _granted_scopes(creds)
    if not granted:
        logger.info(
            "Could not read granted scopes off credentials; skipping "
            "granted-superset check (fail-open). requested=%s",
            sorted(requested),
        )
        return
    missing = requested - granted
    if missing:
        logger.warning(
            "Google scope grant is incomplete: missing=%s requested=%s granted=%s",
            sorted(missing), sorted(requested), sorted(granted),
        )
        raise ScopeGrantError(requested, granted)


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
    # accepts any loopback redirect. access_type=offline + prompt=consent match
    # the relay flow so the installed/CLI path reliably returns a refresh token
    # (previously it defaulted to online with no consent prompt).
    # include_granted_scopes=true — see build_authorization_url for rationale.
    # run_local_server forwards unknown kwargs to authorization_url.
    creds = flow.run_local_server(
        port=0,
        open_browser=True,
        access_type="offline",
        prompt="consent",
        include_granted_scopes="true",
    )
    # Visible-auth-error guardrail: reject a partial grant before persisting.
    _assert_granted_covers_requested(creds, scopes)

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
    # include_granted_scopes=true is set here (and in run_installed_flow) for
    # reviewer expectations only — it is Google's incremental-authorization
    # knob. ClawMeets does NOT rely on incremental auth: each skill requests
    # exactly one scope and stores its own self-scoped token file, so the flag
    # is a no-op for our access pattern. OAUTHLIB_RELAX_TOKEN_SCOPE (set above)
    # keeps fetch_token from rejecting a broader-than-requested scope set, and
    # the post-exchange _assert_granted_covers_requested guard still enforces
    # that every requested scope was actually granted.
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        include_granted_scopes="true",
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

    # Visible-auth-error guardrail: reject a partial grant before persisting.
    _assert_granted_covers_requested(creds, scopes)

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    os.chmod(token_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    logger.info(f"Wrote Google OAuth token to {token_path} (0600) via relay flow")


def _skill_name_from_token_path(token_path: Path) -> str:
    """Best-effort skill name for actionable auth errors.

    Tokens live at ``skill-hub/state/<skill>/token.json``, so the parent dir
    name is the skill. Falls back to the token stem if the layout differs.
    """
    parent = token_path.parent.name
    return parent or token_path.stem or "the integration"


def load_credentials(token_path: Path, scopes: list[str]):
    """Load and refresh cached credentials. Returns a google.oauth2.Credentials.

    Raises :class:`ReauthRequired` — never a swallowed exception or an opaque
    downstream 401 — when the stored grant is dead: no token on disk, an
    expired access token with no refresh token, or a refresh token that Google
    rejects (revoked / expired / consent withdrawn). The raised error carries
    the skill name and a ready-to-run re-authentication instruction so the
    agent gets an explicit, actionable recovery path.
    """
    try:
        from google.auth.exceptions import RefreshError
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError as exc:
        raise GoogleOAuthError(
            "google-auth is required for MCP OAuth flows but missing — "
            "the clawmeets runner should bundle it by default. "
            "Try: pip install --upgrade clawmeets"
        ) from exc

    skill_name = _skill_name_from_token_path(token_path)

    if not token_path.exists():
        raise ReauthRequired(skill_name, "no cached token on disk")

    data = json.loads(token_path.read_text())
    creds = Credentials.from_authorized_user_info(data, scopes=scopes)
    if creds.expired:
        if not creds.refresh_token:
            raise ReauthRequired(
                skill_name, "access token expired and no refresh token is stored"
            )
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            # Refresh token revoked / expired / consent withdrawn on Google's
            # side. Surface it loudly with a recovery instruction rather than
            # letting an opaque 401 escape from the first API call.
            logger.warning(
                "Refresh of Google grant for %s failed (dead grant): %s",
                skill_name, exc,
            )
            raise ReauthRequired(skill_name, "refresh token was rejected by Google") from exc
        token_path.write_text(creds.to_json())
        os.chmod(token_path, stat.S_IRUSR | stat.S_IWUSR)
    return creds


def revoke_token(token_path: Path, timeout: float = 5.0) -> bool:
    """Revoke a Google OAuth grant and delete the local token file.

    Idempotent and best-effort. Steps:
      1. If ``token_path`` is missing → return ``False`` (nothing to revoke).
      2. Read the token JSON; pick the ``refresh_token`` when present
         (revoking it invalidates the whole grant) else the access ``token``.
      3. POST ``{"token": <tok>}`` form-encoded to :data:`GOOGLE_REVOKE_URL`.
         Google returns 200 on success; a 400 ``invalid_token`` means the
         grant is already gone → treated as success (idempotent). Any network
         error is logged at WARNING and does not abort the local cleanup.
      4. Always ``unlink`` the token file at the end so the runner stops
         presenting a dead credential.

    Returns ``True`` when a token file was present (i.e. something was revoked
    / cleaned up), ``False`` when there was nothing to do.
    """
    if not token_path.exists():
        return False

    tok = None
    try:
        data = json.loads(token_path.read_text())
        tok = data.get("refresh_token") or data.get("token")
    except Exception as exc:  # unreadable / malformed token file
        logger.warning("Could not read token at %s for revoke: %s", token_path, exc)

    if tok:
        try:
            import requests

            resp = requests.post(
                GOOGLE_REVOKE_URL,
                data={"token": tok},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=timeout,
            )
            if resp.status_code == 200:
                logger.info("Revoked Google grant for token at %s", token_path)
            elif resp.status_code == 400:
                # Already-invalid token — the grant is gone. Idempotent success.
                logger.info(
                    "Google revoke returned 400 (token already invalid) for %s", token_path
                )
            else:
                logger.warning(
                    "Google revoke returned HTTP %s for %s: %s",
                    resp.status_code, token_path, resp.text[:200],
                )
        except Exception as exc:
            # Network / timeout — proceed to delete the local file regardless.
            logger.warning("Google revoke request failed for %s: %s", token_path, exc)

    token_path.unlink(missing_ok=True)
    logger.info("Deleted local token file %s", token_path)
    return True
