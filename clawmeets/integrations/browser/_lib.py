# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/browser/_lib.py

Python-playwright-driven browser automation. Replaces the npm
``@playwright/mcp`` MCP server with a per-invocation CLI: each subcommand
cold-starts Chrome against a persistent profile, performs one action, and
relies on the profile dir to carry state forward.

Two-tier state model:
  - **Agent-level storage** at
    ``$CLAWMEETS_AGENT_DIR/skill-hub/state/playwright-browser/storage/<name>.json``
    — long-lived authenticated identity (logged-in accounts), a
    ``storage_state`` JSON. Created/refreshed by
    ``clawmeets browser auth --storage <name>``.
  - **Project-level session** at ``<session_dir>/profile/`` (default
    ``$PWD/.playwright-session/profile/``) — a *persistent Chrome user-data
    dir*. Cookies (incl. Cloudflare ``cf_clearance``), cache, and a stable
    browser fingerprint persist across the per-invocation cold starts, which
    is what lets the skill survive Cloudflare's managed-challenge tier. A
    sidecar ``meta.json`` carries the last-navigated URL so non-navigate
    subcommands can re-load the page on a fresh subprocess. On first use of a
    session the agent storage's cookies seed the fresh profile.

Anti-bot hardening (see ``_open_context`` / ``_build_init_script`` /
``_settle_after_load``): headed by default, automation flags stripped, a
realistic fingerprint (UA / viewport / locale / timezone / WebGL), a
``navigator.webdriver`` mask, and challenge-aware waiting that lets a
Cloudflare interstitial resolve before snapshotting.

Per-invocation cold start (~1.5–2 s) is the cost we pay for not running a
daemon process. Keeping one browser alive across actions is a follow-up.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

from clawmeets.integrations._config_resolve import resolve_skill_config_path
from clawmeets.utils.jsonc import parse_jsonc


# ---------------------------------------------------------------------------
# Playwright runtime selection
# ---------------------------------------------------------------------------

def _runtime_name() -> str:
    """``"patchright"`` when the stealth extra is installed, else ``"playwright"``.

    patchright is a drop-in patched Playwright that closes the CDP
    ``Runtime.enable`` leak which defeats Cloudflare Turnstile and similar
    bot-fight tiers — something an ``add_init_script`` cannot fix. Surfaced in
    each action's result JSON so the caller can see which runtime is active.
    """
    try:
        import patchright  # noqa: F401
    except ImportError:
        return "playwright"
    return "patchright"


_FRAME_RACE_PATCHED = False


def _patch_patchright_frame_race() -> None:
    """patchright's ``Page._on_frame_detached`` unconditionally does
    ``self._frames.remove(frame)``, raising ``ValueError`` (``list.remove(x): x
    not in list``) when a challenge/SPA iframe detaches after the frame is
    already gone. pyee logs that as a noisy ``Error occurred in event listener``
    traceback (harmless — it never reaches our code or crashes the session — but
    alarming). Wrap the handler to swallow that specific race. Idempotent;
    patchright-only (stock Playwright is covered by ``_export_storage_state``)."""
    global _FRAME_RACE_PATCHED
    if _FRAME_RACE_PATCHED:
        return
    _FRAME_RACE_PATCHED = True
    try:
        from patchright._impl._page import Page
    except ImportError:
        return
    orig = Page._on_frame_detached
    if getattr(orig, "_clawmeets_guarded", False):
        return

    def _guarded(self, frame):
        try:
            orig(self, frame)
        except ValueError:
            pass  # frame already removed — already effectively detached

    _guarded._clawmeets_guarded = True
    Page._on_frame_detached = _guarded


def _async_playwright():
    """Return an ``async_playwright()`` context manager, preferring patchright
    and falling back to stock Playwright. The two share an identical API, so
    every call site is unchanged beyond this import."""
    try:
        from patchright.async_api import async_playwright
        _patch_patchright_frame_race()
    except ImportError:
        from playwright.async_api import async_playwright
    return async_playwright()


# ---------------------------------------------------------------------------
# State paths
# ---------------------------------------------------------------------------

SKILL_NAME = "playwright-browser"
DEFAULT_SESSION_DIR_NAME = ".playwright-session"


def resolve_storage_path(name: str, agent_dir: Optional[Path] = None) -> Path:
    """Return the agent-level storage_state path for the given identity name.

    Resolution: explicit ``agent_dir`` > ``$CLAWMEETS_AGENT_DIR`` > RuntimeError.
    """
    base: Optional[Path] = agent_dir
    if base is None:
        env = os.environ.get("CLAWMEETS_AGENT_DIR")
        base = Path(env) if env else None
    if base is None:
        raise RuntimeError(
            "Cannot resolve browser storage path: pass --agent or set "
            "CLAWMEETS_AGENT_DIR."
        )
    return base / "skill-hub" / "state" / SKILL_NAME / "storage" / f"{name}.json"


def resolve_session_dir(explicit: Optional[Path]) -> Path:
    """Return the project-level session directory. Defaults to
    ``$PWD/.playwright-session`` so the LLM-shelled CLI lands its session
    inside the project sandbox naturally."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    return Path.cwd() / DEFAULT_SESSION_DIR_NAME


def _session_profile_dir(session_dir: Path) -> Path:
    """Persistent Chrome user-data dir for this session."""
    return session_dir / "profile"


# ---------------------------------------------------------------------------
# Daemon (long-lived browser) discovery + RPC client. The daemon owns one
# patchright context for the agent and one page (tab) per project; CLI
# subcommands route to it over a Unix socket when it is running, and fall back
# to the one-shot path otherwise. See browser/_daemon.py for the server.
# ---------------------------------------------------------------------------

def _agent_dir_or_env(agent_dir: Optional[Path]) -> Optional[Path]:
    if agent_dir is not None:
        return Path(agent_dir)
    env = os.environ.get("CLAWMEETS_AGENT_DIR")
    return Path(env) if env else None


def daemon_state_dir(agent_dir: Optional[Path] = None) -> Path:
    """``{agent_dir}/skill-hub/state/playwright-browser`` — holds the daemon's
    socket, pid file, and its persistent profile."""
    base = _agent_dir_or_env(agent_dir)
    if base is None:
        raise RuntimeError(
            "Cannot resolve browser daemon dir: pass --agent or set CLAWMEETS_AGENT_DIR."
        )
    return base / "skill-hub" / "state" / SKILL_NAME


def daemon_socket_path(agent_dir: Optional[Path] = None) -> Path:
    """Unix socket path for the agent's daemon. Kept in the system temp dir with
    a short, deterministic hashed name because AF_UNIX paths are capped (~104
    bytes on macOS) and the agent state dir alone overflows that. Both the daemon
    and the CLI client derive the same path from the agent's state dir."""
    import hashlib
    import tempfile

    h = hashlib.sha1(str(daemon_state_dir(agent_dir)).encode()).hexdigest()[:16]
    name = f"cmbrowser-{h}.sock"
    candidate = Path(tempfile.gettempdir()) / name
    return candidate if len(str(candidate)) <= 100 else Path("/tmp") / name


def daemon_pid_path(agent_dir: Optional[Path] = None) -> Path:
    return daemon_state_dir(agent_dir) / "daemon.pid"


def daemon_profile_dir(agent_dir: Optional[Path] = None, storage: str = "personal") -> Path:
    """The daemon runs directly on the agent identity's persistent profile (the
    same dir `auth`/`start` populate), so login is inherited with no copy/seed."""
    return _auth_profile_dir(resolve_storage_path(storage, agent_dir=agent_dir))


def project_key(session_dir: Path) -> str:
    """Stable per-project tab key. The LLM subprocess cwd is per-project (the
    sandbox), so the resolved session dir uniquely identifies the project tab."""
    return str(Path(session_dir).resolve())


def daemon_alive(agent_dir: Optional[Path] = None) -> bool:
    """True when a daemon socket is present and accepting connections. Sync (no
    event loop needed) so CLI subcommands can branch before dispatching."""
    import socket as _socket

    try:
        sock_path = daemon_socket_path(agent_dir)
    except RuntimeError:
        return False
    if not sock_path.exists():
        return False
    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect(str(sock_path))
        return True
    except OSError:
        return False
    finally:
        s.close()


async def daemon_request(
    op: str,
    *,
    key: Optional[str] = None,
    args: Optional[dict] = None,
    agent_dir: Optional[Path] = None,
    timeout: float = 1900.0,
) -> dict:
    """Send one newline-delimited JSON request to the agent's browser daemon and
    return its reply. Raises on connection/transport failure so the caller can
    fall back to the one-shot path."""
    sock_path = daemon_socket_path(agent_dir)
    reader, writer = await asyncio.open_unix_connection(str(sock_path))
    try:
        writer.write((json.dumps({"op": op, "key": key, "args": args or {}}) + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        if not line:
            raise RuntimeError("browser daemon closed the connection without replying")
        return json.loads(line.decode())
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001 - best-effort close
            pass


def _auth_profile_dir(storage_path: Path) -> Path:
    """Persistent Chrome user-data dir for the interactive auth window, kept
    next to the agent-storage JSON (``personal.json`` -> ``personal-profile/``)
    so re-auth resumes the prior login."""
    return storage_path.parent / f"{storage_path.stem}-profile"


def _session_meta_path(session_dir: Path) -> Path:
    return session_dir / "meta.json"


def _load_meta(session_dir: Path) -> dict:
    p = _session_meta_path(session_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_meta(session_dir: Path, meta: dict) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    _session_meta_path(session_dir).write_text(json.dumps(meta, indent=2))


def _profile_is_fresh(profile_dir: Path) -> bool:
    """True when the persistent profile has never been launched (so the agent
    storage's cookies should seed it). A non-existent or empty dir is fresh."""
    if not profile_dir.exists():
        return True
    return not any(profile_dir.iterdir())


# Chrome user-data subdirs that are pure regenerable cache — skipping them keeps
# the profile clone small (a few MB vs hundreds) without losing any session state.
_PROFILE_CLONE_IGNORE = shutil.ignore_patterns(
    "Singleton*", "LOCK", "lockfile", "*.lock",
    "Cache", "Code Cache", "GPUCache", "ShaderCache", "GrShaderCache",
    "DawnGraphiteCache", "DawnWebGPUCache",
    "component_crx_cache", "extensions_crx_cache",
)


def _is_session_cookie(cookie: dict) -> bool:
    """True for a session cookie — no positive expiry. ``storage_state`` emits
    ``expires: -1`` for these; Chrome keeps them only in memory, so they must be
    re-injected on every cold start (the persistent profile can't carry them)."""
    expires = cookie.get("expires", -1)
    return not isinstance(expires, (int, float)) or expires <= 0


def _seed_fresh_profile(profile_dir: Path, seed_storage_path: Path) -> bool:
    """Clone the agent auth profile into a fresh session profile for
    full-fidelity state (cookies + localStorage + IndexedDB + fingerprint +
    bot-manager sensor state).

    The cookies-only ``storage_state`` JSON drops localStorage / IndexedDB and
    the Akamai/Cloudflare sensor state bound to the profile, which leaves bot-
    protected SPAs (e.g. evaair.com) rendering logged-out even with the auth
    cookies present. Cloning the persistent auth profile carries all of it.

    Same-machine clone is safe: Chrome's cookie-encryption key lives in a shared
    per-user OS keychain entry (not per-profile), so the copied ``Cookies`` DB
    stays decryptable; localStorage/IndexedDB leveldb is unencrypted.

    Returns True if cloned; False → caller falls back to JSON cookie-replay."""
    auth_profile = _auth_profile_dir(seed_storage_path)
    if not auth_profile.is_dir():
        return False
    # Auth flow itself seeds its own profile — never clone a profile into itself.
    if auth_profile.resolve() == profile_dir.resolve():
        return False
    shutil.copytree(
        auth_profile, profile_dir, dirs_exist_ok=True, ignore=_PROFILE_CLONE_IGNORE,
    )
    return True


# ---------------------------------------------------------------------------
# Fingerprint config (optional) + resolution
# ---------------------------------------------------------------------------

# Pinned to a recent stable Chrome on macOS. Bump alongside the channel.
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
_DEFAULT_FINGERPRINT: dict[str, Any] = {
    "headless": False,  # headed by default — best Cloudflare resistance
    "user_agent": _DEFAULT_USER_AGENT,
    "locale": "en-US",
    "timezone_id": "America/New_York",
    "viewport_width": 1440,
    "viewport_height": 900,
    "webgl_vendor": "Google Inc. (Apple)",
    "webgl_renderer": (
        "ANGLE (Apple, ANGLE Metal Renderer: Apple M1, Unspecified Version)"
    ),
    # Optional proxy (IP-reputation fallback for hard "access denied" blocks).
    # All None by default — no proxy. See _proxy_kwargs.
    "proxy_server": None,      # e.g. "http://host:port" or "socks5://host:port"
    "proxy_username": None,
    "proxy_password": None,
}


def _proxy_kwargs(fp: dict) -> Optional[dict]:
    """Build Playwright's ``proxy=`` dict from the fingerprint, or None.

    Pure: returns ``None`` unless ``proxy_server`` is set; folds in
    username/password when present. Lets the user route through a residential
    proxy when a site blocks the runner's IP outright.
    """
    server = fp.get("proxy_server")
    if not server:
        return None
    proxy: dict[str, str] = {"server": str(server)}
    if fp.get("proxy_username"):
        proxy["username"] = str(fp["proxy_username"])
    if fp.get("proxy_password"):
        proxy["password"] = str(fp["proxy_password"])
    return proxy


def load_config(config_file: str = "") -> tuple[Optional[dict], Optional[str]]:
    """Read the playwright-browser fingerprint config, or self-resolve.

    Falls back to ``$CLAWMEETS_AGENT_DIR/skill-hub/configs/playwright-browser.json``
    when the caller didn't pass a path. The config is OPTIONAL — every key has
    a Cloudflare-resistant default, so a missing config is a clean noop.

    Returns ``(cfg, err)`` — mirrors ``gmail/_lib.py:load_config``:
      - ``(dict, None)`` on a parsed dict-shaped config
      - ``(None, None)`` when the path is empty / missing / empty file
      - ``(None, "...")`` when malformed JSONC or its root isn't a dict
    """
    config_file = resolve_skill_config_path(SKILL_NAME, config_file)
    if not config_file:
        return None, None
    path = Path(config_file).expanduser()
    if not path.exists():
        return None, None
    try:
        raw = path.read_text()
    except OSError as exc:
        return None, f"could not read config file {path}: {exc}"
    if not raw.strip():
        return None, None
    try:
        cfg = parse_jsonc(raw)
    except Exception as exc:
        return None, f"config file is not valid JSON: {exc}"
    if not isinstance(cfg, dict):
        return None, "config file must contain a JSON object"
    return cfg, None


def resolve_fingerprint(
    cfg: Optional[dict], headless_override: Optional[bool] = None
) -> dict:
    """Merge ``cfg`` over the CF-resistant defaults.

    Precedence: CLI flag (``headless_override``) > config value > default.
    Only keys present (and non-None) in ``cfg`` override; the returned dict
    always carries every key in ``_DEFAULT_FINGERPRINT``.
    """
    fp = dict(_DEFAULT_FINGERPRINT)
    if cfg:
        for key in fp:
            if key in cfg and cfg[key] is not None:
                fp[key] = cfg[key]
    if headless_override is not None:
        fp["headless"] = headless_override
    return fp


# ---------------------------------------------------------------------------
# Stealth init script
# ---------------------------------------------------------------------------

# Runs at document-start on every page/frame in the context, before site JS.
# Masks the cheap headless / automation tells Cloudflare fingerprints.
_STEALTH_INIT_JS = """
(() => {
  // 1. navigator.webdriver -> undefined (the canonical automation tell)
  Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

  // 2. navigator.languages (headless Chrome reports [] or a single lang)
  Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });

  // 3. navigator.plugins / length — non-empty defeats a cheap emptiness check
  const fakePlugin = (name, filename, desc) =>
    ({ name, filename, description: desc, length: 1 });
  const plugins = [
    fakePlugin('Chrome PDF Plugin', 'internal-pdf-viewer', 'Portable Document Format'),
    fakePlugin('Chrome PDF Viewer', 'mhjfbmdgcfjbbpaeojofohoefgiehjai', ''),
    fakePlugin('Native Client', 'internal-nacl-plugin', ''),
  ];
  Object.defineProperty(navigator, 'plugins', { get: () => plugins });

  // 4. WebGL vendor/renderer spoof — defeats the headless ANGLE/SwiftShader
  //    GPU signature. 37445 = UNMASKED_VENDOR_WEBGL, 37446 = UNMASKED_RENDERER.
  const patchGL = (proto) => {
    if (!proto || !proto.getParameter) return;
    const orig = proto.getParameter;
    proto.getParameter = function (p) {
      if (p === 37445) return '__WEBGL_VENDOR__';
      if (p === 37446) return '__WEBGL_RENDERER__';
      return orig.call(this, p);
    };
  };
  if (window.WebGLRenderingContext) patchGL(WebGLRenderingContext.prototype);
  if (window.WebGL2RenderingContext) patchGL(WebGL2RenderingContext.prototype);

  // 5. window.chrome stub — real Chrome exposes it; headless often doesn't
  if (!window.chrome) window.chrome = { runtime: {} };
})();
"""


def _build_init_script(fp: dict) -> str:
    """Inject the resolved WebGL strings into the static template. ``json.dumps``
    handles JS string escaping (quotes / backslashes in the renderer string)."""
    return (
        _STEALTH_INIT_JS
        .replace("'__WEBGL_VENDOR__'", json.dumps(fp["webgl_vendor"]))
        .replace("'__WEBGL_RENDERER__'", json.dumps(fp["webgl_renderer"]))
    )


# ---------------------------------------------------------------------------
# Cloudflare challenge detection + settling
# ---------------------------------------------------------------------------

_SETTLE_TIMEOUT_MS = 30000
_SETTLE_POLL_MS = 1000

# General post-load render-settle, distinct from the CF challenge settling
# below: ``goto`` returns at domcontentloaded, before a JS/SPA app fetches and
# paints, so an immediate snapshot/screenshot catches a loading splash. These
# bound a "wait for the page to actually render" step.
_RENDER_NETWORKIDLE_MS = 5000   # cap on the load + networkidle waits
_RENDER_FLOOR_MS = 400          # short stabilization floor for final paint

_CF_TITLE_MARKERS = ("just a moment", "attention required", "checking your browser")
_CF_DOM_MARKERS = (
    "challenge-running",         # #challenge-running
    "cf-browser-verification",   # legacy interstitial class
    "cf-challenge",
    "challenge-platform",        # /cdn-cgi/challenge-platform/ script src
    "_cf_chl_opt",               # turnstile / managed-challenge inline var
)


class ChallengeUnresolvedError(RuntimeError):
    """Raised when a Cloudflare challenge does not clear within the budget."""


def _is_challenge_page(
    status: Optional[int],
    headers: Optional[dict],
    title_or_html: str,
) -> bool:
    """Pure heuristic: True if the page looks like a Cloudflare interstitial.

    - ``status`` in {403, 503} AND a CF header (``cf-ray`` / ``cf-mitigated`` /
      ``server: cloudflare``), OR
    - a CF title marker, OR
    - a CF DOM / script marker substring.

    Case-insensitive on header keys/values and the html blob; ``None``-safe on
    ``status`` / ``headers`` (a same-document nav may carry no Response).
    """
    blob = (title_or_html or "").lower()
    h = {str(k).lower(): str(v).lower() for k, v in (headers or {}).items()}

    cf_header = (
        "cf-ray" in h
        or "cf-mitigated" in h
        or h.get("server", "").startswith("cloudflare")
    )
    if status in (403, 503) and cf_header:
        return True
    if any(m in blob for m in _CF_TITLE_MARKERS):
        return True
    if any(m in blob for m in _CF_DOM_MARKERS):
        return True
    return False


def _resolve_settle_ms(explicit: Optional[int]) -> int:
    """Resolve the render-settle budget: explicit flag > env > default.

    ``0`` disables the post-load render wait (the caller knows the page is
    static / already rendered)."""
    if explicit is not None:
        return explicit
    env = os.environ.get("CLAWMEETS_BROWSER_SETTLE_MS")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            pass
    return _RENDER_NETWORKIDLE_MS


async def _settle_render(page, settle_ms: Optional[int] = None) -> None:
    """Best-effort wait for a page to finish rendering after a load.

    Distinct from the Cloudflare challenge settling: ``goto`` returns at
    ``domcontentloaded``, before a JS/SPA app fetches and paints its content,
    so a snapshot/screenshot taken immediately catches a loading splash. Wait
    for the ``load`` event then a bounded ``networkidle``, plus a short floor
    for final paint. Each step is bounded and non-fatal — pages with persistent
    polling/websockets just fall through on the networkidle timeout, so this
    never blocks the action. A budget of ``0`` skips the wait entirely."""
    budget = _resolve_settle_ms(settle_ms)
    if budget <= 0:
        return
    for state in ("load", "networkidle"):
        try:
            await page.wait_for_load_state(state, timeout=budget)
        except Exception:
            # load/networkidle may never fire (persistent traffic); proceed.
            pass
    if _RENDER_FLOOR_MS:
        try:
            await page.wait_for_timeout(_RENDER_FLOOR_MS)
        except Exception:
            pass


async def _settle_after_load(
    page, response, timeout_ms: int = _SETTLE_TIMEOUT_MS,
    settle_ms: Optional[int] = None,
) -> None:
    """After a load, clear any Cloudflare interstitial, then wait for render.

    Uses the ``goto`` Response (status/headers) for the first challenge check,
    then re-reads the live DOM (title + a bounded ``outerHTML`` slice) on each
    poll — Cloudflare replaces the response after its JS reload, so later polls
    rely on DOM markers. Raises ``ChallengeUnresolvedError`` if still challenged
    at the deadline. Once not (or no longer) challenged, runs ``_settle_render``
    so JS/SPA content is painted before the caller snapshots/screenshots.
    """
    status = response.status if response is not None else None
    headers = response.headers if response is not None else None

    async def _looks_challenged() -> bool:
        try:
            title = await page.title()
        except Exception:
            title = ""
        try:
            # Bounded slice — challenge markers live in <head> / early body.
            html = await page.evaluate(
                "() => document.documentElement.outerHTML.slice(0, 4000)"
            )
        except Exception:
            html = ""
        return _is_challenge_page(status, headers, f"{title}\n{html}")

    if await _looks_challenged():
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            try:
                await page.wait_for_load_state("networkidle", timeout=_SETTLE_POLL_MS)
            except Exception:
                # networkidle may never fire during a JS challenge; keep polling.
                pass
            # After the first reload the original Response is stale.
            status = None
            headers = None
            if not await _looks_challenged():
                break
        else:
            raise ChallengeUnresolvedError(
                "Cloudflare challenge did not clear within "
                f"{timeout_ms} ms (URL: {page.url}). Try headed mode (--headed) "
                "and/or a valid --storage identity, or re-run; managed "
                "challenges can need an interactive pass."
            )

    # Not (or no longer) challenged — let JS/SPA content paint before we look.
    await _settle_render(page, settle_ms)


# ---------------------------------------------------------------------------
# Context + page lifecycle
# ---------------------------------------------------------------------------

def _build_launch_kwargs(
    runtime: str, profile_dir: Path, fp: dict
) -> tuple[dict, bool]:
    """Return ``(launch_kwargs, apply_init_script)`` for ``launch_persistent_context``.

    The two runtimes need OPPOSITE treatment:

    - **patchright**: follow its documented best practices — `channel="chrome"`,
      `no_viewport=True`, and **no** custom ``user_agent`` / ``viewport`` /
      ``args`` / ``ignore_default_args`` / init-script. patchright patches the
      automation tells itself (incl. the CDP ``Runtime.enable`` leak); overriding
      these re-introduces the very inconsistencies it removes, which is what kept
      Cloudflare Turnstile blocking us. So ``apply_init_script`` is False.
    - **stock playwright**: keep the full manual-stealth config — it's the best we
      can do without the CDP-leak patch. ``apply_init_script`` is True.

    ``channel="chrome"`` is added by the caller (so it can retry without it).
    ``headless`` and the proxy apply to both runtimes.
    """
    base: dict = {"user_data_dir": str(profile_dir), "headless": fp["headless"]}
    proxy = _proxy_kwargs(fp)
    if proxy:
        base["proxy"] = proxy

    if runtime == "patchright":
        base["no_viewport"] = True
        return base, False

    base.update(
        args=["--disable-blink-features=AutomationControlled"],
        ignore_default_args=["--enable-automation"],
        user_agent=fp["user_agent"],
        locale=fp["locale"],
        timezone_id=fp["timezone_id"],
        viewport={"width": fp["viewport_width"], "height": fp["viewport_height"]},
    )
    return base, True


async def _open_context(
    p,
    profile_dir: Path,
    fp: dict,
    *,
    seed_storage_path: Optional[Path] = None,
):
    """Launch a persistent, stealth-hardened Chrome context at ``profile_dir``.

    Shared by the action commands (profile = ``<session_dir>/profile/``) and the
    auth flow (profile = ``<storage>-profile/``). The launch config is
    runtime-aware (see ``_build_launch_kwargs``): a clean, override-free config
    under patchright; the full manual-stealth config (args + fingerprint +
    init script) under stock Playwright. An optional proxy applies to both. The
    persistent user-data dir is what makes Cloudflare ``cf_clearance`` + the
    browser fingerprint survive across the per-invocation cold starts. Falls
    back from the real-Chrome channel to bundled Chromium when Chrome is
    unavailable. On a fresh profile, seeds state from ``seed_storage_path``:
    clones the full auth profile when present (see ``_seed_fresh_profile``), else
    falls back to replaying just the JSON snapshot's cookies.

    Returns the persistent ``BrowserContext`` (it owns the browser; close it).
    """
    fresh = _profile_is_fresh(profile_dir)  # check BEFORE launch creates the dir
    # On first use, clone the full auth profile (localStorage + IndexedDB +
    # fingerprint + sensor state) — must happen before launch opens the dir.
    # Cookies are (re-)seeded from the JSON snapshot after launch (below).
    if fresh and seed_storage_path is not None:
        _seed_fresh_profile(profile_dir, seed_storage_path)
    profile_dir.mkdir(parents=True, exist_ok=True)

    launch_kwargs, apply_init = _build_launch_kwargs(
        _runtime_name(), profile_dir, fp
    )
    try:
        context = await p.chromium.launch_persistent_context(
            channel="chrome", **launch_kwargs
        )
    except Exception:
        # chrome channel unavailable (not installed / CI) -> bundled chromium.
        context = await p.chromium.launch_persistent_context(**launch_kwargs)

    # Manual stealth only on the stock-Playwright path; patchright patches these
    # itself, and an init script would re-add a detectable surface.
    if apply_init:
        await context.add_init_script(_build_init_script(fp))

    # Replay cookies from the JSON snapshot. Each subcommand is a cold start, but
    # Chrome NEVER persists session cookies (``expires:-1`` — e.g. evaair's
    # ASP.NET_SessionId / __RequestVerificationToken) to the on-disk profile, so
    # they vanish on every relaunch and the server-side login drops. We restore
    # them in-memory from ``storage_state`` on EVERY launch. Persistent cookies
    # live in the (cloned/accumulated) profile and may refresh during use, so we
    # only seed *those* on a fresh profile — never clobber fresher on-disk values
    # (e.g. a renewed cf_clearance) on later runs.
    if seed_storage_path is not None and seed_storage_path.exists():
        try:
            cookies = json.loads(seed_storage_path.read_text()).get("cookies") or []
            if not fresh:
                cookies = [c for c in cookies if _is_session_cookie(c)]
            if cookies:
                await context.add_cookies(cookies)
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    return context


async def _acquire_page(context):
    """Reuse the most-recently-opened non-closed page, or open one. Newest-first
    so that if a prior in-command step spawned a popup we land on it; neutral for
    a cold-start context (which has exactly one page)."""
    for page in reversed(context.pages):
        if not page.is_closed():
            return page
    return await context.new_page()


async def _ax_snapshot(page):
    """Snapshot the accessibility tree for the current page.

    Newer Playwright removed ``page.accessibility``; fall back to the
    ARIA snapshot (a YAML string) when the legacy API is unavailable."""
    accessibility = getattr(page, "accessibility", None)
    if accessibility is not None:
        tree = await accessibility.snapshot(interesting_only=True)
        return tree or {}
    try:
        return await page.locator("body").aria_snapshot()
    except Exception as exc:  # pragma: no cover - best-effort fallback
        return {"_snapshot_error": str(exc)}


def _persist_meta(page, session_dir: Path) -> None:
    """Persist the last-navigated URL. The profile dir auto-persists the rest."""
    try:
        last_url = page.url
    except Exception:
        last_url = None
    _save_meta(session_dir, {"last_url": last_url})


async def _navigate_to(
    page, url: Optional[str], meta: dict, settle_ms: Optional[int] = None
) -> str:
    """Navigate to ``url`` (or the session's last_url), waiting out any
    Cloudflare challenge and letting the page render. Returns the resolved URL."""
    target = url or meta.get("last_url")
    if not target:
        raise RuntimeError(
            "No URL: pass a URL on `navigate`, or run a previous subcommand "
            "in this session so meta.last_url is set."
        )
    response = await page.goto(target, wait_until="domcontentloaded")
    await _settle_after_load(page, response, settle_ms=settle_ms)
    return target


def _resolve_fp(headless: Optional[bool]) -> dict:
    """Shared action preamble: load the optional config (errors are non-fatal —
    the fingerprint is cosmetic) and resolve it against the CLI flag."""
    cfg, _err = load_config()
    return resolve_fingerprint(cfg, headless_override=headless)


# ---------------------------------------------------------------------------
# Page-driving bodies — the action logic, shared verbatim by the one-shot
# wrappers (below) and the long-lived daemon (browser/_daemon.py). Each takes a
# Page already positioned at the working URL and performs ONE action + snapshot,
# so the only difference between the two modes is how the Page is obtained (a
# fresh cold-start context vs. a persistent per-project tab).
# ---------------------------------------------------------------------------

async def _do_navigate(page, url, meta, settle_ms=None) -> dict:
    resolved = await _navigate_to(page, url, meta, settle_ms)
    snap = await _ax_snapshot(page)
    return {"url": resolved, "snapshot": snap, "runtime": _runtime_name()}


async def _do_snapshot(page) -> dict:
    snap = await _ax_snapshot(page)
    return {"url": page.url, "snapshot": snap, "runtime": _runtime_name()}


async def _do_click(page, selector, settle_ms=None) -> dict:
    prev = page.url
    await page.locator(selector).first.click()
    await page.wait_for_load_state("domcontentloaded")
    await _settle_after_load(page, None, settle_ms=settle_ms)
    snap = await _ax_snapshot(page)
    return {"url": page.url, "previous_url": prev, "snapshot": snap, "runtime": _runtime_name()}


async def _do_fill(page, selector, text) -> dict:
    await page.locator(selector).first.fill(text)
    snap = await _ax_snapshot(page)
    return {"url": page.url, "snapshot": snap, "runtime": _runtime_name()}


async def _do_press_key(page, key, settle_ms=None) -> dict:
    prev = page.url
    await page.keyboard.press(key)
    await page.wait_for_load_state("domcontentloaded")
    await _settle_after_load(page, None, settle_ms=settle_ms)
    snap = await _ax_snapshot(page)
    return {"url": page.url, "previous_url": prev, "snapshot": snap, "runtime": _runtime_name()}


async def _do_screenshot(page, out_path, full_page=False) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    await page.screenshot(path=str(out_path), full_page=full_page)
    return {"url": page.url, "path": str(out_path), "full_page": full_page, "runtime": _runtime_name()}


async def _do_wait_for(page, selector, timeout_ms=10000) -> dict:
    await page.locator(selector).first.wait_for(timeout=timeout_ms)
    snap = await _ax_snapshot(page)
    return {"url": page.url, "snapshot": snap, "runtime": _runtime_name()}


async def _surface_oneshot_popup(context, before, result, settle_ms):
    """One-shot has no next command to switch tabs in (the context is torn down
    after this call), so if the action just opened a tab, snapshot THAT tab now —
    it's the only chance to see it — and flag that persisting it needs the daemon."""
    new = [p for p in context.pages if p not in before and not p.is_closed()]
    if not new:
        return result
    popup = new[-1]
    try:
        await popup.wait_for_load_state("domcontentloaded")
    except Exception:  # noqa: BLE001 - best-effort; snapshot whatever's there
        pass
    await _settle_after_load(popup, None, settle_ms=settle_ms)
    return {
        "url": popup.url,
        "previous_url": result.get("url"),
        "snapshot": await _ax_snapshot(popup),
        "runtime": _runtime_name(),
        "note": (
            "this click opened a new tab (snapshot is of the new tab). One-shot "
            "cannot keep it for the next command — run `clawmeets browser start` "
            "for multi-tab flows."
        ),
    }


# ---------------------------------------------------------------------------
# Public action functions (async; each opens one persistent BrowserContext).
# These are the one-shot path: cold-start a context, restore the page to the
# session's last URL, run the shared _do_* body, persist meta, tear down.
# ---------------------------------------------------------------------------

async def navigate(
    url: str,
    *,
    storage_path: Optional[Path],
    session_dir: Path,
    headless: Optional[bool] = None,
    settle_ms: Optional[int] = None,
) -> dict:
    fp = _resolve_fp(headless)
    async with _async_playwright() as p:
        context = await _open_context(
            p, _session_profile_dir(session_dir), fp, seed_storage_path=storage_path
        )
        try:
            page = await _acquire_page(context)
            result = await _do_navigate(page, url, {}, settle_ms)
            _persist_meta(page, session_dir)
            return result
        finally:
            await context.close()


async def snapshot(
    *,
    storage_path: Optional[Path],
    session_dir: Path,
    headless: Optional[bool] = None,
    settle_ms: Optional[int] = None,
) -> dict:
    fp = _resolve_fp(headless)
    meta = _load_meta(session_dir)
    async with _async_playwright() as p:
        context = await _open_context(
            p, _session_profile_dir(session_dir), fp, seed_storage_path=storage_path
        )
        try:
            page = await _acquire_page(context)
            await _navigate_to(page, None, meta, settle_ms)
            result = await _do_snapshot(page)
            _persist_meta(page, session_dir)
            return result
        finally:
            await context.close()


async def click(
    selector: str,
    *,
    storage_path: Optional[Path],
    session_dir: Path,
    headless: Optional[bool] = None,
    settle_ms: Optional[int] = None,
) -> dict:
    fp = _resolve_fp(headless)
    meta = _load_meta(session_dir)
    async with _async_playwright() as p:
        context = await _open_context(
            p, _session_profile_dir(session_dir), fp, seed_storage_path=storage_path
        )
        try:
            page = await _acquire_page(context)
            await _navigate_to(page, None, meta, settle_ms)
            before = list(context.pages)
            result = await _do_click(page, selector, settle_ms)
            result = await _surface_oneshot_popup(context, before, result, settle_ms)
            _persist_meta(page, session_dir)
            return result
        finally:
            await context.close()


async def fill(
    selector: str,
    text: str,
    *,
    storage_path: Optional[Path],
    session_dir: Path,
    headless: Optional[bool] = None,
    settle_ms: Optional[int] = None,
) -> dict:
    fp = _resolve_fp(headless)
    meta = _load_meta(session_dir)
    async with _async_playwright() as p:
        context = await _open_context(
            p, _session_profile_dir(session_dir), fp, seed_storage_path=storage_path
        )
        try:
            page = await _acquire_page(context)
            await _navigate_to(page, None, meta, settle_ms)
            result = await _do_fill(page, selector, text)
            _persist_meta(page, session_dir)
            return result
        finally:
            await context.close()


async def press_key(
    key: str,
    *,
    storage_path: Optional[Path],
    session_dir: Path,
    headless: Optional[bool] = None,
    settle_ms: Optional[int] = None,
) -> dict:
    fp = _resolve_fp(headless)
    meta = _load_meta(session_dir)
    async with _async_playwright() as p:
        context = await _open_context(
            p, _session_profile_dir(session_dir), fp, seed_storage_path=storage_path
        )
        try:
            page = await _acquire_page(context)
            await _navigate_to(page, None, meta, settle_ms)
            before = list(context.pages)
            result = await _do_press_key(page, key, settle_ms)
            result = await _surface_oneshot_popup(context, before, result, settle_ms)
            _persist_meta(page, session_dir)
            return result
        finally:
            await context.close()


async def screenshot(
    out_path: Path,
    *,
    storage_path: Optional[Path],
    session_dir: Path,
    full_page: bool = False,
    headless: Optional[bool] = None,
    settle_ms: Optional[int] = None,
) -> dict:
    fp = _resolve_fp(headless)
    meta = _load_meta(session_dir)
    async with _async_playwright() as p:
        context = await _open_context(
            p, _session_profile_dir(session_dir), fp, seed_storage_path=storage_path
        )
        try:
            page = await _acquire_page(context)
            await _navigate_to(page, None, meta, settle_ms)
            result = await _do_screenshot(page, out_path, full_page)
            _persist_meta(page, session_dir)
            return result
        finally:
            await context.close()


async def wait_for(
    selector: str,
    *,
    storage_path: Optional[Path],
    session_dir: Path,
    timeout_ms: int = 10000,
    headless: Optional[bool] = None,
    settle_ms: Optional[int] = None,
) -> dict:
    fp = _resolve_fp(headless)
    meta = _load_meta(session_dir)
    async with _async_playwright() as p:
        context = await _open_context(
            p, _session_profile_dir(session_dir), fp, seed_storage_path=storage_path
        )
        try:
            page = await _acquire_page(context)
            await _navigate_to(page, None, meta, settle_ms)
            result = await _do_wait_for(page, selector, timeout_ms)
            _persist_meta(page, session_dir)
            return result
        finally:
            await context.close()


async def _export_storage_state(context, storage_path: Path) -> None:
    """Write ``storage_state`` JSON, tolerating patchright's frame-detach race.

    patchright's ``_on_frame_detached`` does an unconditional
    ``self._frames.remove(frame)``, which raises ``ValueError`` (``list.remove(x):
    x not in list``) when challenge iframes (Cloudflare/Turnstile) detach while
    ``storage_state`` iterates frames — aborting the export and leaving no seed
    file. On that specific failure, fall back to a cookies-only state captured via
    ``context.cookies()`` (a flat Network call that never touches the frame list).
    Cookies are all the session seeder consumes (see ``_open_context``)."""
    try:
        await context.storage_state(path=str(storage_path))
    except Exception as exc:  # noqa: BLE001 - narrowed by message below
        if "list.remove" not in str(exc):
            raise
        cookies = await context.cookies()
        storage_path.write_text(
            json.dumps({"cookies": cookies, "origins": []}, indent=2)
        )


async def auth_interactive(
    *,
    storage_path: Path,
    start_url: Optional[str] = None,
) -> dict:
    """Open a headed, stealth-hardened real-Chrome window for interactive login.

    Drives the same persistent + stealth context the action commands use
    (preferring patchright), instead of the old ``npx playwright open`` — which
    launched bundled Chromium with no hardening and so got blocked by
    Cloudflare/Turnstile before the user could even sign in. The human solves
    any challenge themselves in the real window; on confirmation we export the
    ``storage_state`` JSON (seeds future session profiles) while the auth
    profile dir retains the full logged-in state for re-auth.

    Blocks on an explicit terminal Enter rather than the window-close event:
    the persistent profile auto-persists, and capturing ``storage_state`` before
    close avoids the websocket-disconnect-on-quit hang of the old async path.
    """
    import asyncio

    storage_path.parent.mkdir(parents=True, exist_ok=True)
    profile_dir = _auth_profile_dir(storage_path)
    # Auth is always headed (the user needs to see + solve the challenge).
    fp = resolve_fingerprint(load_config()[0], headless_override=False)

    async with _async_playwright() as p:
        context = await _open_context(
            p, profile_dir, fp, seed_storage_path=storage_path
        )
        try:
            page = await _acquire_page(context)
            if start_url:
                await page.goto(start_url, wait_until="domcontentloaded")
            runtime = _runtime_name()
            stealth_note = (
                "stealth runtime: patchright (active)"
                if runtime == "patchright"
                else "WARNING: stock Playwright (no patchright) — Cloudflare "
                "Turnstile will likely block; run under Python <=3.13 to get "
                "the bundled patchright stealth runtime"
            )
            print(
                f"\nA Chrome window opened [{stealth_note}]. Sign in to the "
                "site(s) you need (solve any Cloudflare/Turnstile yourself), then "
                "return here and press Enter. Do NOT close the window yourself.\n",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.to_thread(input)
            await _export_storage_state(context, storage_path)
            return {"storage_path": str(storage_path), "runtime": _runtime_name()}
        finally:
            await context.close()
