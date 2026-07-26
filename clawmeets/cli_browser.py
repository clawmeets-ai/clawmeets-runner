# SPDX-License-Identifier: MIT
"""
clawmeets/cli_browser.py

``clawmeets browser <subcmd>`` — Python-playwright browser CLI.

Two-tier state model: agent-level ``--storage <name>`` (identity, written
once by ``auth``) + project-level ``--session <dir>`` (working state per
project sandbox; defaults to ``$PWD/.playwright-session``).

Subcommands:
  auth         Headed Chromium; user logs in; saves agent storage.
  navigate     Load URL; emit AX snapshot.
  snapshot     Re-load session.last_url; emit AX snapshot.
  click        CSS-selector click; emit snapshot.
  fill         Fill an input by selector.
  press-key    Press a keyboard key.
  screenshot   Save screenshot to --out.
  wait-for     Wait for a selector; emit snapshot.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer

from clawmeets.cli_lifecycle import _popen_detached_kwargs, _read_pid, _stop_pid
from clawmeets.cli_runner import DEFAULT_DATA_DIR, _resolve_agent_dir
from clawmeets.integrations.browser import _lib

app = typer.Typer(
    name="browser",
    help="Browser automation via Python playwright. Paired skill: playwright-browser.",
    no_args_is_help=True,
)


def _emit_json(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


def _storage_path(storage: str, agent_dir: Optional[Path]) -> Optional[Path]:
    return _lib.resolve_storage_path(storage, agent_dir=agent_dir) if storage else None


def _agent_dir(agent: Optional[str], data_dir: Path) -> Optional[Path]:
    """Resolve the agent dir from --agent/--data-dir, or None to fall back to
    $CLAWMEETS_AGENT_DIR (set inside the runner)."""
    return _resolve_agent_dir(data_dir, agent) if agent else None


# Shared across the action subcommands. Tri-state Optional[bool]: None lets the
# skill config decide (default headed); --headless / --headed force it.
_HEADLESS_OPTION = typer.Option(
    None, "--headless/--headed",
    help="Run headless or with a visible window. Default: headed (best "
         "Cloudflare resistance) unless the skill config sets headless=true.",
)

# Shared identity resolution so action commands work outside the runner (where
# $CLAWMEETS_AGENT_DIR isn't set). Inside the runner, omit --agent and the env
# is used.
_AGENT_OPTION = typer.Option(
    None, "--agent", "-a",
    help="Agent name (or {name}-{id} dirname) to resolve --storage outside the "
         "runner. Falls back to $CLAWMEETS_AGENT_DIR when omitted.",
)
_DATA_DIR_OPTION = typer.Option(DEFAULT_DATA_DIR, "--data-dir")

# Post-load render-settle budget (ms). None → $CLAWMEETS_BROWSER_SETTLE_MS or the
# built-in default; 0 disables the wait (page is static / already rendered).
_SETTLE_MS_OPTION = typer.Option(
    None, "--settle-ms",
    help="Max ms to wait for the page to finish rendering (load + network-idle) "
         "before snapshot/screenshot. Default ~5000; 0 to disable.",
)


def _dispatch(op: str, *, session, agent, data_dir, rpc_args: dict, one_shot) -> None:
    """Route an action to the long-lived daemon when it's running (fast path),
    else run the one-shot path. ``one_shot`` is a zero-arg coroutine factory."""
    agent_dir = _agent_dir(agent, data_dir)
    if _lib.daemon_alive(agent_dir):
        key = _lib.project_key(_lib.resolve_session_dir(session))
        _emit_json(asyncio.run(_lib.daemon_request(op, key=key, args=rpc_args, agent_dir=agent_dir)))
    else:
        _emit_json(asyncio.run(one_shot()))


def _dispatch_daemon_only(op: str, *, session, agent, data_dir, rpc_args: dict) -> None:
    """Route a daemon-only op (tab management). There's no one-shot equivalent —
    a tab can't persist across cold-start commands — so error clearly if the
    daemon isn't up."""
    agent_dir = _agent_dir(agent, data_dir)
    if not _lib.daemon_alive(agent_dir):
        _emit_json({"error": "tab management needs the daemon — run "
                             "`clawmeets browser start` first."})
        return
    key = _lib.project_key(_lib.resolve_session_dir(session))
    _emit_json(asyncio.run(_lib.daemon_request(op, key=key, args=rpc_args, agent_dir=agent_dir)))


def _start_daemon(storage: str, agent: Optional[str], data_dir: Path,
                  headless: Optional[bool]) -> bool:
    """Spawn the per-agent browser daemon detached; wait until its socket is up.
    Returns True if it's running afterward. No-op (True) if already running."""
    agent_dir = _agent_dir(agent, data_dir)
    if _lib.daemon_alive(agent_dir):
        return True
    state_dir = _lib.daemon_state_dir(agent_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "clawmeets.integrations.browser._daemon", "--storage", storage]
    if agent:
        cmd += ["--agent-dir", str(_resolve_agent_dir(data_dir, agent))]
    if headless is True:
        cmd.append("--headless")
    elif headless is False:
        cmd.append("--headed")
    logf = open(state_dir / "daemon.log", "ab")  # noqa: SIM115 - lives with the child
    subprocess.Popen(
        cmd, stdout=logf, stderr=logf, stdin=subprocess.DEVNULL,
        **_popen_detached_kwargs(),
    )
    # Chrome launch (esp. first run) can take several seconds.
    for _ in range(120):
        if _lib.daemon_alive(agent_dir):
            return True
        time.sleep(0.5)
    return _lib.daemon_alive(agent_dir)


@app.command()
def start(
    storage: str = typer.Option("personal", "--storage", help="Identity profile the daemon hosts."),
    agent: Optional[str] = _AGENT_OPTION,
    data_dir: Path = _DATA_DIR_OPTION,
    headless: Optional[bool] = _HEADLESS_OPTION,
    url: Optional[str] = typer.Option(
        None, "--url", help="Optional URL to open for sign-in once the daemon is up."),
) -> None:
    """Start the long-lived per-agent browser (fast path). One persistent Chrome
    + one tab per project; login stays live, no cold start, no state copying.
    Sign in once in the live window; leave it running."""
    agent_dir = _agent_dir(agent, data_dir)
    already = _lib.daemon_alive(agent_dir)
    if not _start_daemon(storage, agent, data_dir, headless):
        typer.echo("Error: browser daemon did not come up (see daemon.log).", err=True)
        raise typer.Exit(1)
    if already:
        typer.echo("browser daemon already running")
    if url:
        _emit_json(asyncio.run(_lib.daemon_request(
            "auth_open", key="__auth__", args={"url": url}, agent_dir=agent_dir)))
    else:
        _emit_json({"ok": True, "daemon": "running"})


@app.command()
def stop(
    agent: Optional[str] = _AGENT_OPTION,
    data_dir: Path = _DATA_DIR_OPTION,
) -> None:
    """Stop the per-agent browser daemon (frees the resident Chrome)."""
    agent_dir = _agent_dir(agent, data_dir)
    try:
        pid_path = _lib.daemon_pid_path(agent_dir)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if not _stop_pid(pid_path, "browser daemon"):
        typer.echo("no browser daemon running")
    try:
        _lib.daemon_socket_path(agent_dir).unlink()
    except (OSError, RuntimeError):
        pass


@app.command()
def auth(
    storage: str = typer.Option(
        "personal", "--storage",
        help="Storage identity name. Saved at "
             "<agent_dir>/skill-hub/state/playwright-browser/storage/<name>.json. "
             "Pick a different name (e.g. 'work') if you want a separate sign-in.",
    ),
    start_url: Optional[str] = typer.Option(
        None, "--url",
        help="Optional URL to load in the headed browser at start (e.g. an OAuth page).",
    ),
    agent: Optional[str] = typer.Option(
        None, "--agent", "-a",
        help="Agent name (or {name}-{id} dirname). Falls back to "
             "$CLAWMEETS_AGENT_DIR when omitted.",
    ),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
) -> None:
    """Sign in interactively. Fast path: ensures the long-lived browser is up and
    opens a live window — sign in there and the session stays live (no export,
    no copying). Falls back to a one-shot headed login when no daemon is available."""
    agent_dir = _agent_dir(agent, data_dir)
    if _start_daemon(storage, agent, data_dir, headless=False) and _lib.daemon_alive(agent_dir):
        res = asyncio.run(_lib.daemon_request(
            "auth_open", key="__auth__", args={"url": start_url}, agent_dir=agent_dir))
        typer.echo(
            "A live browser window is open. Sign in to the site(s) you need; the "
            "session stays live in the daemon. Leave it running, or "
            "`clawmeets browser stop` when done.", err=True)
        _emit_json(res)
        return
    # Fallback: one-shot headed interactive login (writes storage_state + profile).
    storage_path = _lib.resolve_storage_path(storage, agent_dir=agent_dir)
    try:
        result = asyncio.run(_lib.auth_interactive(
            storage_path=storage_path, start_url=start_url,
        ))
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    _emit_json(result)


@app.command()
def navigate(
    url: str = typer.Argument(..., help="URL to load."),
    storage: str = typer.Option("", "--storage", help="Storage identity to layer in."),
    session: Optional[Path] = typer.Option(
        None, "--session",
        help="Session dir (default: $PWD/.playwright-session).",
    ),
    agent: Optional[str] = _AGENT_OPTION,
    data_dir: Path = _DATA_DIR_OPTION,
    headless: Optional[bool] = _HEADLESS_OPTION,
    settle_ms: Optional[int] = _SETTLE_MS_OPTION,
) -> None:
    """Navigate to URL; emit accessibility-tree snapshot."""
    _dispatch(
        "navigate", session=session, agent=agent, data_dir=data_dir,
        rpc_args={"url": url, "settle_ms": settle_ms},
        one_shot=lambda: _lib.navigate(
            url,
            storage_path=_storage_path(storage, _agent_dir(agent, data_dir)),
            session_dir=_lib.resolve_session_dir(session),
            headless=headless,
            settle_ms=settle_ms,
        ),
    )


@app.command()
def snapshot(
    storage: str = typer.Option("", "--storage"),
    session: Optional[Path] = typer.Option(None, "--session"),
    agent: Optional[str] = _AGENT_OPTION,
    data_dir: Path = _DATA_DIR_OPTION,
    headless: Optional[bool] = _HEADLESS_OPTION,
    settle_ms: Optional[int] = _SETTLE_MS_OPTION,
) -> None:
    """Re-load the session's last URL; emit accessibility-tree snapshot."""
    _dispatch(
        "snapshot", session=session, agent=agent, data_dir=data_dir,
        rpc_args={"settle_ms": settle_ms},
        one_shot=lambda: _lib.snapshot(
            storage_path=_storage_path(storage, _agent_dir(agent, data_dir)),
            session_dir=_lib.resolve_session_dir(session),
            headless=headless,
            settle_ms=settle_ms,
        ),
    )


@app.command()
def click(
    selector: str = typer.Argument(..., help="CSS / Playwright selector."),
    storage: str = typer.Option("", "--storage"),
    session: Optional[Path] = typer.Option(None, "--session"),
    agent: Optional[str] = _AGENT_OPTION,
    data_dir: Path = _DATA_DIR_OPTION,
    headless: Optional[bool] = _HEADLESS_OPTION,
    settle_ms: Optional[int] = _SETTLE_MS_OPTION,
) -> None:
    """Click the first element matching --selector; emit snapshot."""
    _dispatch(
        "click", session=session, agent=agent, data_dir=data_dir,
        rpc_args={"selector": selector, "settle_ms": settle_ms},
        one_shot=lambda: _lib.click(
            selector,
            storage_path=_storage_path(storage, _agent_dir(agent, data_dir)),
            session_dir=_lib.resolve_session_dir(session),
            headless=headless,
            settle_ms=settle_ms,
        ),
    )


@app.command()
def fill(
    selector: str = typer.Argument(...),
    text: str = typer.Argument(...),
    storage: str = typer.Option("", "--storage"),
    session: Optional[Path] = typer.Option(None, "--session"),
    agent: Optional[str] = _AGENT_OPTION,
    data_dir: Path = _DATA_DIR_OPTION,
    headless: Optional[bool] = _HEADLESS_OPTION,
    settle_ms: Optional[int] = _SETTLE_MS_OPTION,
) -> None:
    """Fill an <input> matching --selector with --text; emit snapshot."""
    _dispatch(
        "fill", session=session, agent=agent, data_dir=data_dir,
        rpc_args={"selector": selector, "text": text},
        one_shot=lambda: _lib.fill(
            selector, text,
            storage_path=_storage_path(storage, _agent_dir(agent, data_dir)),
            session_dir=_lib.resolve_session_dir(session),
            headless=headless,
            settle_ms=settle_ms,
        ),
    )


@app.command("press-key")
def press_key(
    key: str = typer.Argument(..., help="Keyboard key name (e.g. Enter, Tab, ArrowDown)."),
    storage: str = typer.Option("", "--storage"),
    session: Optional[Path] = typer.Option(None, "--session"),
    agent: Optional[str] = _AGENT_OPTION,
    data_dir: Path = _DATA_DIR_OPTION,
    headless: Optional[bool] = _HEADLESS_OPTION,
    settle_ms: Optional[int] = _SETTLE_MS_OPTION,
) -> None:
    """Press a key on the current page; emit snapshot."""
    _dispatch(
        "press_key", session=session, agent=agent, data_dir=data_dir,
        rpc_args={"key": key, "settle_ms": settle_ms},
        one_shot=lambda: _lib.press_key(
            key,
            storage_path=_storage_path(storage, _agent_dir(agent, data_dir)),
            session_dir=_lib.resolve_session_dir(session),
            headless=headless,
            settle_ms=settle_ms,
        ),
    )


@app.command()
def screenshot(
    out: Path = typer.Option(..., "--out", help="Destination PNG path."),
    full_page: bool = typer.Option(False, "--full-page"),
    storage: str = typer.Option("", "--storage"),
    session: Optional[Path] = typer.Option(None, "--session"),
    agent: Optional[str] = _AGENT_OPTION,
    data_dir: Path = _DATA_DIR_OPTION,
    headless: Optional[bool] = _HEADLESS_OPTION,
    settle_ms: Optional[int] = _SETTLE_MS_OPTION,
) -> None:
    """Save a screenshot of the current page."""
    # Resolve to absolute so the daemon (whose cwd differs from this CLI's)
    # writes the file where the caller expects it.
    _dispatch(
        "screenshot", session=session, agent=agent, data_dir=data_dir,
        rpc_args={"out_path": str(Path(out).resolve()), "full_page": full_page},
        one_shot=lambda: _lib.screenshot(
            out,
            storage_path=_storage_path(storage, _agent_dir(agent, data_dir)),
            session_dir=_lib.resolve_session_dir(session),
            full_page=full_page,
            headless=headless,
            settle_ms=settle_ms,
        ),
    )


@app.command("wait-for")
def wait_for_cmd(
    selector: str = typer.Argument(...),
    storage: str = typer.Option("", "--storage"),
    session: Optional[Path] = typer.Option(None, "--session"),
    timeout_ms: int = typer.Option(10000, "--timeout-ms"),
    agent: Optional[str] = _AGENT_OPTION,
    data_dir: Path = _DATA_DIR_OPTION,
    headless: Optional[bool] = _HEADLESS_OPTION,
    settle_ms: Optional[int] = _SETTLE_MS_OPTION,
) -> None:
    """Wait for an element matching --selector to appear; emit snapshot."""
    _dispatch(
        "wait_for", session=session, agent=agent, data_dir=data_dir,
        rpc_args={"selector": selector, "timeout_ms": timeout_ms},
        one_shot=lambda: _lib.wait_for(
            selector,
            storage_path=_storage_path(storage, _agent_dir(agent, data_dir)),
            session_dir=_lib.resolve_session_dir(session),
            timeout_ms=timeout_ms,
            headless=headless,
            settle_ms=settle_ms,
        ),
    )


@app.command()
def tabs(
    session: Optional[Path] = typer.Option(None, "--session"),
    agent: Optional[str] = _AGENT_OPTION,
    data_dir: Path = _DATA_DIR_OPTION,
) -> None:
    """List this project's open browser tabs (daemon only)."""
    _dispatch_daemon_only(
        "tabs", session=session, agent=agent, data_dir=data_dir, rpc_args={},
    )


@app.command("switch-tab")
def switch_tab(
    index: int = typer.Argument(..., help="Tab index (from `tabs`) to make active."),
    session: Optional[Path] = typer.Option(None, "--session"),
    agent: Optional[str] = _AGENT_OPTION,
    data_dir: Path = _DATA_DIR_OPTION,
) -> None:
    """Make tab <index> the active tab; emit its snapshot (daemon only)."""
    _dispatch_daemon_only(
        "switch_tab", session=session, agent=agent, data_dir=data_dir,
        rpc_args={"index": index},
    )


@app.command("close-tab")
def close_tab(
    index: int = typer.Argument(..., help="Tab index (from `tabs`) to close."),
    session: Optional[Path] = typer.Option(None, "--session"),
    agent: Optional[str] = _AGENT_OPTION,
    data_dir: Path = _DATA_DIR_OPTION,
) -> None:
    """Close tab <index>; emit the remaining tab list (daemon only)."""
    _dispatch_daemon_only(
        "close_tab", session=session, agent=agent, data_dir=data_dir,
        rpc_args={"index": index},
    )
