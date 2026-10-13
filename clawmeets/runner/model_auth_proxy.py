# SPDX-License-Identifier: MIT
"""
clawmeets/runner/model_auth_proxy.py

Inference-only model-auth proxy (plan 4.3, hardening item 2).

The provider credential never enters a scoped turn. The turn gets a base URL
with a per-turn nonce path (served by the egress proxy's local handler:
``http://127.0.0.1:<port>/m/<nonce>`` on macOS, the netshim address on Linux)
and a placeholder key. This proxy accepts only that nonce, only while the turn
is open, and only the (method, host, path) triples pinned in
:data:`INFERENCE_ROUTES` for the provider and login type. It strips any
client-supplied ``Authorization``, ``x-api-key``, ``x-goog-api-key`` and ``key=``
before attaching the real credential. Everything else gets 403.

A (provider, login) pair absent from :data:`INFERENCE_ROUTES` gets no
capability. :data:`UNCONFIRMED_ROUTES` records the paths we believe those CLIs
use but could not confirm, or that need more than inference to run a turn
(account / onboarding calls); they are documentation only and never served.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit

from ..utils import env_store

if TYPE_CHECKING:
    import http.server

    from .turn_home import TurnHome

logger = logging.getLogger(__name__)

PREFIX = "/m/"

_ANTHROPIC = "api.anthropic.com"
_OPENAI = "api.openai.com"
_GEMINI = "generativelanguage.googleapis.com"

_ANTHROPIC_ROUTES = frozenset({
    ("POST", _ANTHROPIC, "/v1/messages"),
    ("POST", _ANTHROPIC, "/v1/messages/count_tokens"),
})
_OPENAI_ROUTES = frozenset({
    ("POST", _OPENAI, "/v1/responses"),
    ("POST", _OPENAI, "/v1/chat/completions"),
})
_GEMINI_ROUTES = frozenset({
    ("POST", _GEMINI, "/v1beta/models/{model}:generateContent"),
    ("POST", _GEMINI, "/v1beta/models/{model}:streamGenerateContent"),
    ("POST", _GEMINI, "/v1beta/models/{model}:countTokens"),
})

# (provider, login) -> allowed (method, host, path pattern). Anything else: 403.
# ``{model}`` matches one path segment of model-name characters.
INFERENCE_ROUTES: Mapping[tuple[str, str], frozenset[tuple[str, str, str]]] = {
    ("claude", "api_key"): _ANTHROPIC_ROUTES,
    ("claude", "oauth"): _ANTHROPIC_ROUTES,
    ("codex", "api_key"): _OPENAI_ROUTES,
    ("gemini", "api_key"): _GEMINI_ROUTES,
    ("opencode", "anthropic"): _ANTHROPIC_ROUTES,
    ("opencode", "openai"): _OPENAI_ROUTES,
}

# Pinned for the record, NEVER served (no capability):
# - codex ChatGPT login: the CLI's inference endpoint on the ChatGPT backend;
#   the login also refreshes tokens against auth.openai.com and sends a
#   chatgpt-account-id header. Not confirmed from source in this commit.
# - gemini Google login (Code Assist): the CLI calls loadCodeAssist /
#   onboardUser before inference, i.e. more than inference to run a turn.
UNCONFIRMED_ROUTES: Mapping[tuple[str, str], frozenset[tuple[str, str, str]]] = {
    ("codex", "chatgpt"): frozenset({("POST", "chatgpt.com", "/backend-api/codex/responses")}),
    ("gemini", "google_login"): frozenset({
        ("POST", "cloudcode-pa.googleapis.com", "/v1internal:generateContent"),
        ("POST", "cloudcode-pa.googleapis.com", "/v1internal:streamGenerateContent"),
        ("POST", "cloudcode-pa.googleapis.com", "/v1internal:countTokens"),
    }),
}

# Header names a client may never pass through (its own auth) or that the
# proxy recomputes.
_STRIP_HEADERS = {
    "authorization", "x-api-key", "x-goog-api-key", "proxy-authorization", "cookie",
    "host", "content-length", "connection", "keep-alive", "transfer-encoding", "te",
    "trailer", "upgrade", "proxy-connection", "accept-encoding",
}
_STRIP_QUERY = {"key", "api_key", "access_token"}
_HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
        "proxy-authenticate", "content-length"}

PLACEHOLDER_KEY = "clawmeets-turn-placeholder"


def _pattern(path: str) -> re.Pattern[str]:
    return re.compile("^" + re.escape(path).replace(re.escape("{model}"), r"[A-Za-z0-9._\-]+") + "$")


def route_allowed(provider: str, login: str, method: str, path: str) -> Optional[str]:
    """The upstream host for an allowed request, or ``None`` (=> 403)."""
    for m, host, pat in INFERENCE_ROUTES.get((provider, login), ()):
        if m == method.upper() and _pattern(pat).match(path):
            return host
    return None


def capable(provider: str, login: Optional[str]) -> bool:
    return login is not None and (provider, login) in INFERENCE_ROUTES


# ---------------------------------------------------------------------------
# Credentials (held in the runner only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Credential:
    login: str
    headers: Mapping[str, str]


CredentialSource = Callable[[str], Optional[Credential]]


def _keychain_secret(service: str) -> Optional[str]:
    if sys.platform != "darwin":
        return None
    try:
        r = subprocess.run(["/usr/bin/security", "find-generic-password", "-s", service, "-w"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def _claude_oauth_token() -> Optional[str]:
    raw = _keychain_secret("Claude Code-credentials")
    if raw is None:
        path = Path.home() / ".claude" / ".credentials.json"
        try:
            raw = path.read_text()
        except OSError:
            return None
    try:
        tok = json.loads(raw).get("claudeAiOauth", {}).get("accessToken")
    except (ValueError, AttributeError):
        return None
    return tok or None


class DefaultCredentials:
    """Resolve each provider's credential and login type on the runner side.

    API keys come from the agent's env store, then the runner environment.
    ``local_settings.llm_api_key`` is never used: for these plain CLI providers
    it is only set as a local gateway's bearer token (with ``llm_base_url``).
    Claude falls back to its OAuth login. Codex ChatGPT login and Gemini Google
    login are detected (so the reason for "no capability" is accurate) but
    carry no credential.

    An agent pointed at its own endpoint (``llm_base_url``, e.g. a local model)
    gets no credential, so no capability: the proxy only forwards to the
    providers' public APIs, and a scoped turn there would send the stranger's
    content, and the user's key or login, to the cloud against that choice.
    """

    def __init__(self, agent_dir: Optional[Path] = None, settings: Optional[dict] = None) -> None:
        self._agent_dir = agent_dir
        self._settings = settings or {}

    def _key(self, *names: str) -> Optional[str]:
        store = env_store.load(self._agent_dir) if self._agent_dir else {}
        for n in names:
            if store.get(n):
                return store[n]
        for n in names:
            if os.environ.get(n):
                return os.environ[n]
        return None

    def __call__(self, provider: str) -> Optional[Credential]:
        if self._settings.get("llm_base_url"):
            return None
        if provider == "claude":
            k = self._key("ANTHROPIC_API_KEY")
            if k:
                return Credential("api_key", {"x-api-key": k})
            tok = _claude_oauth_token()
            if tok:
                return Credential("oauth", {"Authorization": f"Bearer {tok}",
                                            "anthropic-beta": "oauth-2025-04-20"})
            return None
        if provider == "codex":
            k = self._key("OPENAI_API_KEY")
            if k:
                return Credential("api_key", {"Authorization": f"Bearer {k}"})
            if (Path.home() / ".codex" / "auth.json").exists():
                return Credential("chatgpt", {})
            return None
        if provider == "gemini":
            k = self._key("GEMINI_API_KEY", "GOOGLE_API_KEY")
            if k:
                return Credential("api_key", {"x-goog-api-key": k})
            return Credential("google_login", {})
        if provider == "opencode":
            model = str(self._settings.get("llm_model") or "")
            family = model.split("/", 1)[0] if "/" in model else ""
            if family == "anthropic":
                k = self._key("ANTHROPIC_API_KEY")
                return Credential("anthropic", {"x-api-key": k}) if k else None
            if family == "openai":
                k = self._key("OPENAI_API_KEY")
                return Credential("openai", {"Authorization": f"Bearer {k}"}) if k else None
            return None  # unknown underlying provider: no capability
        return None


# ---------------------------------------------------------------------------
# The proxy
# ---------------------------------------------------------------------------


@dataclass
class _Turn:
    provider: str
    credential: Credential


UpstreamFactory = Callable[[str], http.client.HTTPConnection]


def _default_upstream(host: str) -> http.client.HTTPConnection:
    return http.client.HTTPSConnection(host, 443, timeout=600)


class ModelAuthProxy:
    """The egress proxy's local handler for ``/m/<nonce>/...`` requests."""

    def __init__(self, credentials: CredentialSource, *, upstream: Optional[UpstreamFactory] = None) -> None:
        self._credentials = credentials
        self._upstream = upstream or _default_upstream
        self._turns: dict[str, _Turn] = {}
        self._lock = threading.Lock()

    # -- per turn ---------------------------------------------------------------

    def login_for(self, provider: str) -> Optional[str]:
        cred = self._credentials(provider)
        return cred.login if cred else None

    def open_turn(self, provider: str, login: str, proxy_url: str) -> tuple[str, str]:
        """Register a turn; returns ``(base_url_with_nonce, placeholder_key)``.

        Raises ``PermissionError`` when (provider, login) has no pinned routes
        or the current credential is of another login type.
        """
        cred = self._credentials(provider)
        if cred is None or cred.login != login or not capable(provider, login):
            raise PermissionError(f"no inference routes for ({provider}, {login})")
        nonce = secrets.token_urlsafe(24)
        with self._lock:
            self._turns[nonce] = _Turn(provider, cred)
        return f"{proxy_url.rstrip('/')}{PREFIX}{nonce}", PLACEHOLDER_KEY

    def close_turn(self, nonce: str) -> None:
        with self._lock:
            self._turns.pop(nonce, None)

    def attach_turn(self, home: "TurnHome", provider: str) -> dict[str, str]:
        """Open a nonce for ``home`` and wire the provider at it: returns the env
        additions and writes provider config into the throwaway home. Stores
        the nonce on ``home.model_nonce`` for :meth:`close_turn`."""
        login = self.login_for(provider)
        if login is None or not capable(provider, login):
            raise PermissionError(f"{provider}: login {login!r} has no inference capability")
        base, placeholder = self.open_turn(provider, login, home.egress_url or "")
        home.model_nonce = base.rsplit("/", 1)[-1]  # type: ignore[attr-defined]
        return provider_wiring(home, provider, login, base, placeholder)

    # -- LocalHandler ------------------------------------------------------------

    def handles(self, path: str) -> bool:
        return path.startswith(PREFIX)

    def _parse(self, path: str) -> tuple[Optional[_Turn], str, str]:
        rest = path[len(PREFIX):]
        nonce, _, tail = rest.partition("/")
        with self._lock:
            turn = self._turns.get(nonce)
        parts = urlsplit("/" + tail)
        return turn, parts.path, parts.query

    def handle(self, req: "http.server.BaseHTTPRequestHandler") -> None:
        turn, path, query = self._parse(req.path)
        # Drain the body first so a refusal never resets the connection.
        try:
            body = _read_body(req)
        except (ValueError, OSError):
            return _reply(req, 400, "bad request body")
        if turn is None:
            return _reply(req, 403, "unknown or closed turn")
        host = route_allowed(turn.provider, turn.credential.login, req.command, path)
        if host is None:
            logger.info(f"model-auth: refused {req.command} {path} for {turn.provider}/{turn.credential.login}")
            return _reply(req, 403, "not an inference route")
        headers = {k: v for k, v in req.headers.items() if k.lower() not in _STRIP_HEADERS}
        headers.update(turn.credential.headers)
        headers["Host"] = host
        headers["Content-Length"] = str(len(body))
        q = [(k, v) for k, v in parse_qsl(query, keep_blank_values=True) if k.lower() not in _STRIP_QUERY]
        target = path + (f"?{urlencode(q)}" if q else "")
        try:
            conn = self._upstream(host)
            conn.request(req.command, target, body=body, headers=headers)
            resp = conn.getresponse()
        except OSError as e:
            return _reply(req, 502, f"upstream error: {e}")
        try:
            req.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() not in _HOP:
                    req.send_header(k, v)
            req.send_header("Connection", "close")
            req.end_headers()
            while True:
                chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                if not chunk:
                    break
                req.wfile.write(chunk)
                req.wfile.flush()
        except OSError:
            pass
        finally:
            conn.close()
            req.close_connection = True


def _reply(req, code: int, text: str) -> None:
    body = (text + "\n").encode()
    req.send_response(code)
    req.send_header("Content-Type", "text/plain; charset=utf-8")
    req.send_header("Content-Length", str(len(body)))
    req.send_header("Connection", "close")
    req.end_headers()
    req.wfile.write(body)
    req.close_connection = True


def _read_body(req) -> bytes:
    if "chunked" in (req.headers.get("Transfer-Encoding") or "").lower():
        out = b""
        while True:
            size = int(req.rfile.readline().split(b";")[0].strip() or b"0", 16)
            if size == 0:
                req.rfile.readline()
                return out
            out += req.rfile.read(size)
            req.rfile.readline()
    n = int(req.headers.get("Content-Length") or 0)
    return req.rfile.read(n) if n else b""


# ---------------------------------------------------------------------------
# Pointing each CLI at the proxy
# ---------------------------------------------------------------------------


def provider_wiring(home: "TurnHome", provider: str, login: str, base: str, placeholder: str) -> dict[str, str]:
    """Env additions (and, for codex/opencode, config in the turn home) that
    point the provider at ``base``. The real credential is never written."""
    if provider == "claude":
        return {"ANTHROPIC_BASE_URL": base, "ANTHROPIC_API_KEY": placeholder}
    if provider == "codex":
        cfg = home.home / ".codex" / "config.toml"
        existing = cfg.read_text() if cfg.exists() else ""
        cfg.write_text(
            'model_provider = "clawmeets"\n' + existing + "\n"
            "[model_providers.clawmeets]\n"
            'name = "clawmeets"\n'
            f"base_url = {json.dumps(base + '/v1')}\n"
            'env_key = "OPENAI_API_KEY"\n'
            'wire_api = "responses"\n'
        )
        return {"OPENAI_API_KEY": placeholder}
    if provider == "gemini":
        return {"GOOGLE_GEMINI_BASE_URL": base, "GEMINI_API_KEY": placeholder}
    if provider == "opencode":
        family = login  # "anthropic" | "openai"
        cfg = home.home / ".config" / "opencode" / "opencode.json"
        data = json.loads(cfg.read_text()) if cfg.exists() else {}
        data.setdefault("provider", {})[family] = {
            "options": {"baseURL": base + "/v1", "apiKey": placeholder}
        }
        cfg.write_text(json.dumps(data))
        return {}
    return {}
