# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/http_api/_client.py

General-purpose HTTP client backing ``clawmeets http-api {get,post,put,patch,
delete}``. Independent of the warehouse ``sync`` driver in ``_lib.py``.

Design contract (see skills/http-api/SKILL.md):

* **Redirects off by default.** Cookie apps signal auth state via 3xx — a
  ``302`` or ``303`` carrying ``Location: /`` = ok, ``/login`` = fail;
  auto-following would mask a failed login as a final ``200 OK``. ``--follow``
  opts in.
* **Expectations are declared, never inferred.** A generic client cannot know
  what success means for an arbitrary endpoint — a wrong password is a perfectly
  ordinary ``303`` with its own ``Set-Cookie``. ``--expect-location`` /
  ``--expect-status`` let the caller say what "worked" looks like; an unmet
  expectation exits 23.
* **The jar is written only on exit 0.** ``--save-session`` persists cookies
  only when the command succeeds outright — no unmet expectation, no ``--fail``
  status, no render error. A failed login therefore cannot overwrite a working
  session jar with the login page's anonymous cookie.
* **Session jar is cookies-only.** ``--save-session`` writes ``{"cookies": {…}}``
  at ``0600``; ``--session`` seeds a live ``httpx.Cookies`` jar that httpx
  updates from *every* response — including a 302's own ``Set-Cookie`` — so the
  session cookie handed back on a redirect is captured with or without
  following. Legacy ``{"cookies", "headers"}`` jars still load (headers ignored).
* **Auth is explicit.** There is no auth subcommand. Pass
  ``-H "Authorization: Bearer $ICS_TOKEN"``; ``$VAR`` / ``${VAR}`` in a header
  value is expanded from the environment (unset → empty string, shell parity),
  so the secret never appears in argv, logs, or the jar.
* **HTTP status never fails the process.** 4xx/5xx return exit 0 with the body
  so structured error payloads stay readable. ``--fail`` opts into curl-style
  non-zero (22) on status ≥ 400. Non-zero codes are reserved for usage / jar /
  network faults.
* **Credentials never logged.** Request headers/cookies/body are never echoed;
  only the response the caller asked for is printed.
"""
from __future__ import annotations

import json
import os
import re
import stat
import sys
import tempfile
from fnmatch import fnmatchcase
from urllib.parse import urlencode, urljoin, urlsplit

import httpx

# Exit codes — stable contract, documented in SKILL.md.
EXIT_OK = 0        # response received (ANY HTTP status, incl. 4xx/5xx)
EXIT_USAGE = 2     # bad -H / --query / --data pair, both body flags, bad --output/--json
EXIT_SESSION = 6   # session/jar I/O error (unreadable jar, unwritable path)
EXIT_NETWORK = 7   # network error (DNS, connect, timeout, TLS, bad URL)
EXIT_HTTP_FAIL = 22  # HTTP status >= 400 AND --fail was passed
EXIT_EXPECT = 23   # a declared --expect-* did not match (jar NOT written)

# Methods that may carry a request body.
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

_ENV_RE = re.compile(r"\$(?:\{(\w+)\}|(\w+))")

# One --expect-status token: 3 chars, digits or 'x' wildcards ("200", "2xx").
_STATUS_TOKEN_RE = re.compile(r"^[1-5x][0-9x]{2}$")

# Hop-by-hop / body-derived headers are never persisted; we keep the jar to
# cookies only, but this guards the header-expansion path too.
_ENV_UNSET = ""


def _expand_env(value: str) -> str:
    """Expand ``$VAR`` / ``${VAR}`` in a header value from ``os.environ``.

    Unset variables become the empty string (shell parity) so a missing token
    surfaces the API's own 401 rather than leaking which env vars exist.
    """
    def sub(m: "re.Match[str]") -> str:
        name = m.group(1) or m.group(2)
        return os.environ.get(name, _ENV_UNSET)
    return _ENV_RE.sub(sub, value)


def _parse_headers(items: list[str]) -> tuple[dict[str, str], str | None]:
    """Parse repeated ``"Name: value"`` into a dict (last wins on duplicates).

    Values are env-expanded. Returns ``(headers, error|None)``.
    """
    out: dict[str, str] = {}
    for raw in items:
        if ":" not in raw:
            return {}, f"bad header (want 'Name: value'): {raw!r}"
        name, _, value = raw.partition(":")
        name = name.strip()
        if not name:
            return {}, f"bad header (empty name): {raw!r}"
        out[name] = _expand_env(value.strip())
    return out, None


def _parse_pairs(items: list[str], kind: str) -> tuple[list[tuple[str, str]], str | None]:
    """Parse repeated ``"k=v"`` into ordered pairs (duplicate keys preserved).

    Used for both ``--query`` and ``--data``. Returns ``(pairs, error|None)``.
    """
    out: list[tuple[str, str]] = []
    for raw in items:
        if "=" not in raw:
            return [], f"bad {kind} (want 'key=value'): {raw!r}"
        key, _, value = raw.partition("=")
        if not key:
            return [], f"bad {kind} (empty key): {raw!r}"
        out.append((key, value))
    return out, None


def _read_source(spec: str) -> tuple[str, str | None]:
    """Resolve a ``--json`` chunk: ``@-`` = stdin, ``@path`` = file, else literal.

    Returns ``(text, error|None)``.
    """
    if spec == "@-":
        return sys.stdin.read(), None
    if spec.startswith("@"):
        path = spec[1:]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read(), None
        except OSError as exc:
            return "", f"cannot read --json file {path!r}: {exc}"
    return spec, None


def _load_jar(path: str) -> tuple[dict[str, str], str | None]:
    """Load cookies from a jar file. ``--session`` on a missing/bad jar errors.

    Accepts legacy ``{"cookies", "headers"}`` jars (headers ignored). Returns
    ``(cookies, error|None)``.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
    except FileNotFoundError:
        return {}, f"session jar not found: {path}"
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"cannot read session jar {path!r}: {exc}"
    if not isinstance(blob, dict):
        return {}, f"session jar {path!r} must be a JSON object"
    cookies = blob.get("cookies", {})
    if not isinstance(cookies, dict):
        return {}, f"session jar {path!r}: 'cookies' must be an object"
    return {str(k): str(v) for k, v in cookies.items()}, None


def _save_jar(path: str, cookies: httpx.Cookies) -> str | None:
    """Persist ``{"cookies": {…}}`` atomically at ``0600``.

    Cookies come from the live jar *after* the request, so a ``Set-Cookie``
    handed back on a 302 is captured whether or not redirects were followed.
    Domain/path scoping is flattened to name→value (same-host replay).
    Returns ``error|None``.
    """
    jar = {c.name: c.value for c in cookies.jar}
    payload = json.dumps({"cookies": jar}, ensure_ascii=False, indent=2)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".jar-", suffix=".tmp")
        try:
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # 0600 before it has a name
        os.replace(tmp, path)
    except OSError as exc:
        return f"cannot write session jar {path!r}: {exc}"
    return None


def _resolve_body(
    method: str,
    data_pairs: list[tuple[str, str]],
    json_parts: list[str],
) -> tuple[bytes | None, dict[str, str], str | None]:
    """Resolve the request body and its implied headers.

    ``--json`` (curl semantics: chunks concatenated raw, ``@file`` / ``@-``
    supported) sets both ``Content-Type`` and ``Accept: application/json``.
    ``--data`` urlencodes the form pairs as ``x-www-form-urlencoded``. The two
    are mutually exclusive. Returns ``(content|None, header_patch, error|None)``.
    """
    if json_parts and data_pairs:
        return None, {}, "use either --json or --data, not both"
    if json_parts:
        chunks: list[str] = []
        for part in json_parts:
            text, err = _read_source(part)
            if err:
                return None, {}, err
            chunks.append(text)
        body = "".join(chunks)
        patch = {"Content-Type": "application/json", "Accept": "application/json"}
        return body.encode("utf-8"), patch, None
    if data_pairs:
        body = urlencode(data_pairs)
        patch = {"Content-Type": "application/x-www-form-urlencoded"}
        return body.encode("utf-8"), patch, None
    return None, {}, None


def _parse_status_spec(spec: str) -> tuple[list[str], str | None]:
    """Split ``"200,3xx"`` into normalized 3-char patterns (digits or ``x``).

    Parsed before the request is sent so a typo costs a usage error rather than
    a live call whose verdict is meaningless. Returns ``(patterns, error|None)``.
    """
    patterns: list[str] = []
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if not _STATUS_TOKEN_RE.match(token):
            return [], (
                f"bad --expect-status token {raw.strip()!r} "
                "(want a 3-char code with 'x' wildcards, e.g. '200', '2xx', '200,303')"
            )
        patterns.append(token)
    if not patterns:
        return [], f"bad --expect-status {spec!r} (no status codes given)"
    return patterns, None


def _match_status(code: int, patterns: list[str]) -> bool:
    """True if the 3-digit status matches any pattern, ``x`` matching any digit."""
    text = f"{code:03d}"
    return any(
        all(want in ("x", got) for want, got in zip(pattern, text))
        for pattern in patterns
    )


def _resolve_landing(resp: httpx.Response) -> str:
    """Where did this request end up? — the absolute-ized ``Location`` if there
    is one (so ``/x``, ``https://h/x`` and ``../x`` all normalize), else the
    response's own URL.

    The else-branch is what makes ``--expect-location`` behave identically under
    ``--follow``, where the final response carries no ``Location`` at all.
    """
    location = resp.headers.get("location")
    if location:
        return urljoin(str(resp.url), location)
    return str(resp.url)


def _match_location(landing: str, pattern: str) -> bool:
    """Glob the landing URL. A pattern starting with ``/`` matches the URL *path*
    (``path?query`` when the pattern itself contains ``?``), so ``'/'`` is exact
    and ``'/dashboard*'`` is a prefix; anything else matches the whole absolute
    URL. Substring matching is deliberately not used — ``/`` would match
    ``/login``. Case-sensitive: URL paths are.
    """
    if not pattern.startswith("/"):
        return fnmatchcase(landing, pattern)
    parts = urlsplit(landing)
    target = parts.path or "/"
    if "?" in pattern and parts.query:
        target = f"{target}?{parts.query}"
    return fnmatchcase(target, pattern)


def _check_expectations(
    resp: httpx.Response,
    expect_location: str,
    status_patterns: list[str],
) -> str | None:
    """Evaluate every declared expectation (AND). ``None`` when all pass, else a
    one-line reason for stderr, e.g. ``expectation failed: landed at '/login'
    (want '/')``.
    """
    if status_patterns and not _match_status(resp.status_code, status_patterns):
        return (
            f"expectation failed: status {resp.status_code} "
            f"(want {','.join(status_patterns)})"
        )
    if expect_location:
        landing = _resolve_landing(resp)
        if not _match_location(landing, expect_location):
            return (
                f"expectation failed: landed at {landing!r} "
                f"(want {expect_location!r})"
            )
    return None


def _render(resp: httpx.Response, output: str) -> tuple[str, str | None]:
    """Render the response per ``--output`` (``body`` | ``json`` | ``full``).

    ``full`` is a JSON envelope exposing status + ``Location`` + redirect history
    so a login flow can read the auth signal off a 302. Never includes request
    creds. Returns ``(text, error|None)``.
    """
    if output == "body":
        return resp.text, None
    if output == "json":
        try:
            parsed = resp.json()
        except (json.JSONDecodeError, ValueError):
            return "", "response body is not valid JSON (use --output body/full)"
        return json.dumps(parsed, ensure_ascii=False, indent=2), None
    if output == "full":
        envelope = {
            "status": resp.status_code,
            "reason": resp.reason_phrase,
            "url": str(resp.url),
            "location": resp.headers.get("location"),
            "history": [r.status_code for r in resp.history],
            "headers": dict(resp.headers),
            "body": resp.text,
        }
        return json.dumps(envelope, ensure_ascii=False, indent=2), None
    return "", f"bad --output {output!r} (want body|json|full)"


def run(
    *,
    method: str,
    url: str,
    header: list[str],
    query: list[str],
    data: list[str],
    json_body: list[str],
    session: str,
    save_session: str,
    follow: bool,
    output: str,
    timeout: float,
    fail: bool,
    expect_location: str = "",
    expect_status: str = "",
) -> int:
    """Orchestrate one request. Returns a process exit code (see module top).

    Precedence: body-implied headers < explicit ``-H`` (explicit wins). A single
    ``httpx.Cookies`` jar is seeded from ``--session`` and captures ``Set-Cookie``
    across the redirect chain for ``--save-session``.

    Tail ordering matters: expectations are evaluated, the response is printed
    either way (so the caller can read the login page that came back), and only
    then is the jar written — never on a non-zero exit. With neither
    ``expect_*`` passed the behaviour is byte-identical to not having them,
    except that ``--fail`` no longer poisons the jar on a 4xx.
    """
    method = method.upper()

    headers, err = _parse_headers(header)
    if err:
        print(err, file=sys.stderr)
        return EXIT_USAGE
    query_pairs, err = _parse_pairs(query, "query")
    if err:
        print(err, file=sys.stderr)
        return EXIT_USAGE
    data_pairs, err = _parse_pairs(data, "data")
    if err:
        print(err, file=sys.stderr)
        return EXIT_USAGE

    if data or json_body:
        if method not in _BODY_METHODS:
            print(f"{method} does not take a body (--data/--json)", file=sys.stderr)
            return EXIT_USAGE

    content, body_headers, err = _resolve_body(method, data_pairs, json_body)
    if err:
        print(err, file=sys.stderr)
        return EXIT_USAGE

    if output not in ("body", "json", "full"):
        print(f"bad --output {output!r} (want body|json|full)", file=sys.stderr)
        return EXIT_USAGE

    status_patterns: list[str] = []
    if expect_status:
        status_patterns, err = _parse_status_spec(expect_status)
        if err:
            print(err, file=sys.stderr)
            return EXIT_USAGE

    # Implied body headers first, explicit -H on top (explicit wins).
    send_headers = {**body_headers, **headers}

    jar = httpx.Cookies()
    if session:
        cookies, err = _load_jar(session)
        if err:
            print(err, file=sys.stderr)
            return EXIT_SESSION
        for name, value in cookies.items():
            jar.set(name, value)

    try:
        with httpx.Client(
            cookies=jar,
            follow_redirects=follow,
            timeout=timeout,
        ) as client:
            resp = client.request(
                method,
                url,
                headers=send_headers or None,
                params=query_pairs or None,
                content=content,
            )
            live_cookies = client.cookies
    except (httpx.InvalidURL, httpx.UnsupportedProtocol) as exc:
        print(f"bad url: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except httpx.RequestError as exc:
        print(f"network error: {exc}", file=sys.stderr)
        return EXIT_NETWORK

    unmet = _check_expectations(resp, expect_location, status_patterns)
    if unmet:
        print(unmet, file=sys.stderr)

    # Print before deciding — a failed login's body is the diagnostic.
    text, err = _render(resp, output)
    if err:
        print(err, file=sys.stderr)
        return EXIT_USAGE
    print(text)

    http_failed = fail and resp.status_code >= 400
    if unmet or http_failed:
        # The gate: a request that did not go as declared must not replace a
        # working jar with the failure page's anonymous cookie.
        if save_session:
            print(f"not writing session jar {save_session!r}", file=sys.stderr)
        return EXIT_EXPECT if unmet else EXIT_HTTP_FAIL

    if save_session:
        err = _save_jar(save_session, live_cookies)
        if err:
            print(err, file=sys.stderr)
            return EXIT_SESSION
    return EXIT_OK
