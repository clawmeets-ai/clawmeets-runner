# SPDX-License-Identifier: MIT
"""
clawmeets/cli_oauth.py

Client-side driver for ``clawmeets user login --provider google|github``: the
CLI counterpart to
the browser OAuth flow the web app already uses. Pure stdlib (http.server,
webbrowser, secrets, hashlib, base64, threading) + the caller's httpx.Client —
no new dependency.

Two shapes, both ending in the same ``POST /auth/oauth/exchange`` that the SPA
uses (so the CLI receives the byte-identical login body: access + refresh token
pair + user/assistant fields):

  - ``login_via_browser`` — the ``gh auth login`` / ``gcloud`` pattern. Bind a
    throwaway loopback listener on 127.0.0.1, ask the server for an authorize
    URL whose redirect points back at that listener, open the browser, and
    block until the provider callback delivers the one-time handoff code to the
    loopback. Exchange it (with the PKCE verifier) for tokens.
  - ``login_via_paste`` — headless/SSH fallback (``--no-browser``): redirect is
    the ``oob`` sentinel; the browser page displays the handoff code and the
    user pastes it back into the terminal.

PKCE (RFC-7636, S256) binds the final exchange to the CLI that started the flow:
the verifier never leaves this process; only its sha256 challenge is sent to the
server at start, and the server requires the matching verifier at exchange.
"""
from __future__ import annotations

import base64
import hashlib
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import httpx

# How long to wait for the user to complete the browser consent + redirect.
BROWSER_TIMEOUT_SECONDS = 180

_CLOSE_PAGE = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<title>clawmeets CLI login</title></head>"
    "<body style='font-family:system-ui;max-width:30rem;margin:3rem auto;text-align:center'>"
    "<h2>{title}</h2><p>{body}</p><p>You can close this tab and return to your terminal.</p>"
    "</body></html>"
)


class CliOAuthError(Exception):
    """A CLI OAuth flow failed. ``str(exc)`` is a user-facing message."""


def make_pkce() -> tuple[str, str]:
    """Return an (verifier, challenge) PKCE S256 pair.

    verifier: 43-char URL-safe random (token_urlsafe(32)); challenge:
    base64url-no-pad(sha256(verifier)) — the exact transform the server re-runs.
    """
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


class _LoopbackHandler(BaseHTTPRequestHandler):
    """Captures the single provider-callback hit on the loopback listener.

    The server 302s the browser to ``http://127.0.0.1:PORT/cb?code=<handoff>``
    (or ``?error=<CODE>``). We record whichever arrived on the server instance
    and show the user a close-this-tab page. Non-/cb paths (e.g. favicon) get a
    bare 404 so they don't consume the result.
    """

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        parsed = urlparse(self.path)
        if parsed.path != "/cb":
            self.send_response(404)
            self.end_headers()
            return
        params = parse_qs(parsed.query)
        code = (params.get("code") or [None])[0]
        error = (params.get("error") or [None])[0]
        self.server.oauth_result = {"code": code, "error": error}  # type: ignore[attr-defined]
        if code:
            page = _CLOSE_PAGE.format(title="Login complete", body="clawmeets received your login.")
        else:
            page = _CLOSE_PAGE.format(title="Login failed", body=f"The server reported: {error or 'unknown error'}.")
        body = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.server.oauth_done.set()  # type: ignore[attr-defined]

    def log_message(self, *args) -> None:  # silence stdlib request logging
        pass


def _cli_start(client: httpx.Client, provider: str, redirect: str,
               challenge: str, invite: str | None, username: str | None) -> dict:
    """POST /auth/oauth/{provider}/cli/start; return {authorize_url, state} or raise."""
    payload: dict = {"redirect": redirect, "code_challenge": challenge}
    if invite:
        payload["invite_code"] = invite
    if username:
        payload["username"] = username
    try:
        resp = client.post(f"/auth/oauth/{provider}/cli/start", json=payload)
    except httpx.RequestError as e:
        raise CliOAuthError(
            f"could not reach clawmeets server at {client.base_url} ({type(e).__name__})."
        ) from e
    if resp.status_code != 200:
        raise CliOAuthError(_server_error_message(resp))
    return resp.json()


def _exchange(client: httpx.Client, code: str, verifier: str) -> dict:
    """POST /auth/oauth/exchange {code, code_verifier}; return the login body or raise."""
    try:
        resp = client.post("/auth/oauth/exchange", json={"code": code, "code_verifier": verifier})
    except httpx.RequestError as e:
        raise CliOAuthError(f"could not reach clawmeets server ({type(e).__name__}).") from e
    if resp.status_code != 200:
        raise CliOAuthError(_server_error_message(resp))
    return resp.json()


def _server_error_message(resp: httpx.Response) -> str:
    """Best-effort human message from the {error:{code,message}} envelope."""
    try:
        err = resp.json().get("error") or {}
        code = err.get("code")
        message = err.get("message")
        if code or message:
            return f"{code or 'error'}: {message or resp.text}"
    except Exception:
        pass
    return f"server returned {resp.status_code}: {resp.text}"


def login_via_browser(client: httpx.Client, provider: str, invite: str | None,
                      port: int, username: str | None = None) -> dict:
    """Full browser + loopback login. Returns the exchange login body.

    Binds 127.0.0.1:``port`` (0 = ephemeral), starts the CLI flow with a loopback
    redirect + PKCE challenge, opens the authorize URL, and blocks (up to
    BROWSER_TIMEOUT_SECONDS) for the callback to deliver the handoff code, then
    exchanges it. Raises CliOAuthError with a user-facing message on any failure.
    """
    verifier, challenge = make_pkce()
    try:
        httpd = HTTPServer(("127.0.0.1", port), _LoopbackHandler)
    except OSError as e:
        raise CliOAuthError(f"could not bind loopback port {port} ({e}). Try --port <n> or --no-browser.") from e
    httpd.oauth_result = None  # type: ignore[attr-defined]
    httpd.oauth_done = threading.Event()  # type: ignore[attr-defined]
    bound_port = httpd.server_address[1]
    redirect = f"http://127.0.0.1:{bound_port}/cb"

    try:
        start = _cli_start(client, provider, redirect, challenge, invite, username)
        authorize_url = start.get("authorize_url")
        if not authorize_url:
            raise CliOAuthError("server did not return an authorize URL.")

        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        opened = webbrowser.open(authorize_url)
        if not opened:
            # No browser could be launched (e.g. headless) — show the URL so the
            # user can open it manually; the loopback still catches the redirect.
            print(f"Open this URL in your browser to continue:\n  {authorize_url}")

        if not httpd.oauth_done.wait(timeout=BROWSER_TIMEOUT_SECONDS):  # type: ignore[attr-defined]
            raise CliOAuthError(
                f"timed out after {BROWSER_TIMEOUT_SECONDS}s waiting for the browser login. "
                "Try again, or use --no-browser on a headless machine."
            )
        result = httpd.oauth_result or {}  # type: ignore[attr-defined]
    finally:
        httpd.shutdown()
        httpd.server_close()

    if result.get("error"):
        raise CliOAuthError(f"login was rejected ({result['error']}).")
    code = result.get("code")
    if not code:
        raise CliOAuthError("no handoff code received from the browser redirect.")
    return _exchange(client, code, verifier)


def login_via_paste(client: httpx.Client, provider: str, invite: str | None,
                     username: str | None = None, *, prompt=input,
                     echo=print) -> dict:
    """Headless (--no-browser) login: server shows the handoff code, user pastes it.

    ``prompt`` / ``echo`` are injectable for testing (default to input/print).
    Returns the exchange login body; raises CliOAuthError on failure.
    """
    verifier, challenge = make_pkce()
    start = _cli_start(client, provider, "oob", challenge, invite, username)
    authorize_url = start.get("authorize_url")
    if not authorize_url:
        raise CliOAuthError("server did not return an authorize URL.")
    echo(
        "Open this URL in a browser, complete the login, then paste the code shown:\n"
        f"  {authorize_url}\n"
    )
    code = (prompt("Handoff code: ") or "").strip()
    if not code:
        raise CliOAuthError("no code entered.")
    return _exchange(client, code, verifier)
