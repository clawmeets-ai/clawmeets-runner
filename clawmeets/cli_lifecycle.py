# SPDX-License-Identifier: MIT
"""
clawmeets/cli_lifecycle.py

Agent lifecycle commands: start, stop, status.

Usage:
    clawmeets start          # start all agents
    clawmeets stop           # stop all agents
    clawmeets status         # show agent process status
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import typer

from clawmeets.utils.agent_processes import (
    IS_WINDOWS,
    agent_pid_file,
    agents_dir as resolve_agents_dir,
    find_agent_dir,
    list_owned_agent_short_names,
    pid_is_alive,
    popen_detached_kwargs,
    prefixed_name,
    read_pid,
    signal_kill,
    signal_terminate,
    stop_pid,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SERVER = os.environ.get("CLAWMEETS_SERVER_URL", "https://clawmeets.ai")
DEFAULT_DATA_DIR = os.environ.get("CLAWMEETS_DATA_DIR", str(Path.home() / ".clawmeets"))


# ---------------------------------------------------------------------------
# Multi-user config helpers
# ---------------------------------------------------------------------------


def get_current_user(data_dir: Path) -> str | None:
    """Read current_user file."""
    path = data_dir / "config" / "current_user"
    return path.read_text().strip() if path.exists() else None


def set_current_user(data_dir: Path, username: str) -> None:
    """Write current_user file."""
    path = data_dir / "config" / "current_user"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(username)


def get_user_config_path(data_dir: Path, username: str) -> Path:
    """Get path to a user's settings.json."""
    return data_dir / "config" / username / "settings.json"


def save_user_session(
    data_dir: Path,
    username: str,
    server_url: str,
    token: str,
    refresh_token: str | None = None,
    auth_method: str = "password",
    make_current: bool = True,
) -> Path:
    """Upsert a user's settings.json with login session info and mark them current.

    ``make_current=False`` saves the session without touching
    ``config/current_user``: the one-line installer adds an account to a
    machine that may already have a default one, and must not move it.

    Creates the file with minimal scaffolding if it does not yet exist, so
    this works both for fresh accounts and already-configured users.

    When a ``refresh_token`` is supplied, persist it (plus ``auth_method``) and
    drop any legacy plaintext ``password`` — token expiry is then renewed via
    ``POST /auth/refresh`` (see cli_runner._ensure_fresh_user_token), which is
    the only path that works for OAuth accounts and removes password-at-rest for
    password accounts. Back-compat: callers that omit ``refresh_token`` keep the
    prior token-only behavior.
    """
    path = get_user_config_path(data_dir, username)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(path.read_text()) if path.exists() else {}
    config["server_url"] = server_url
    # Agent roster written by the retired `clawmeets init` wizard; nothing
    # reads it — agents are discovered from ~/.clawmeets/agents/.
    config.pop("agents", None)
    user = config.setdefault("user", {})
    user["username"] = username
    user["token"] = token
    if refresh_token:
        user["refresh_token"] = refresh_token
        user["auth_method"] = auth_method
        user.pop("password", None)  # refresh-token renewal supersedes password-at-rest
    path.write_text(json.dumps(config, indent=2))
    if make_current:
        set_current_user(data_dir, username)
    return path


def clear_user_token(data_dir: Path, username: str) -> Path:
    """Remove the saved JWT token from a user's settings.json. No-op if absent."""
    path = get_user_config_path(data_dir, username)
    if not path.exists():
        return path
    config = json.loads(path.read_text())
    config.get("user", {}).pop("token", None)
    path.write_text(json.dumps(config, indent=2))
    return path


def load_user_config(data_dir: Path, username: str | None = None) -> tuple[dict, Path]:
    """Load a user's settings.json. Uses current_user if username not specified."""
    if username is None:
        username = get_current_user(data_dir)
    if username:
        path = get_user_config_path(data_dir, username)
        if path.exists():
            return json.loads(path.read_text()), path
    if username:
        typer.echo(
            f"Error: No config for user '{username}'. "
            f"Run `clawmeets user login {username} <password> --save` first.",
            err=True,
        )
    else:
        typer.echo(
            "Error: No user configured. "
            "Run `clawmeets user login <username> <password> --save` first.",
            err=True,
        )
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Helpers — cross-platform process management
#
# The discovery + liveness logic itself lives in
# ``clawmeets/utils/agent_processes.py``, because the connection daemon
# (``clawmeets_daemon``, shipped as its own distribution) has to answer the
# same questions about the same machine and the two must never disagree. What
# stays here is the terminal-CLI skin over it: the ``_``-prefixed aliases that
# existing imports and tests resolve, and the ``typer.echo`` reporting the
# shared module deliberately does not do.
# ---------------------------------------------------------------------------

_IS_WINDOWS = IS_WINDOWS

# Aliases, not reimplementations. `cli_server.py`, `cli_browser.py` and the
# lifecycle tests import these names; keeping them pointed at the shared
# module is what makes "one copy" true rather than aspirational.
_popen_detached_kwargs = popen_detached_kwargs
_pid_is_alive = pid_is_alive
_signal_terminate = signal_terminate
_signal_kill = signal_kill
_read_pid = read_pid


def _stop_pid(pid_file: Path, label: str) -> bool:
    """Stop a process by PID file and say so. Returns True if it was running.

    Graceful first (SIGTERM / CTRL_BREAK_EVENT), then a force kill after a
    5-second grace period — all of which lives in ``stop_pid``. This wrapper
    exists for the one thing the shared module must not do: print. The daemon
    reports the same stop over a websocket, not to a terminal.
    """
    pid = stop_pid(pid_file)
    if pid is None:
        return False
    typer.echo(f"  Stopped {label} (PID {pid})")
    return True


def _get_agents_dir() -> Path:
    """``~/.clawmeets/agents`` — resolved through the module-level
    ``DEFAULT_DATA_DIR`` so tests can repoint the whole tree by patching it."""
    return resolve_agents_dir(Path(DEFAULT_DATA_DIR))


_prefixed_name = prefixed_name
_find_agent_dir = find_agent_dir
_list_owned_agent_short_names = list_owned_agent_short_names


def _build_agent_list(agents_dir: Path, username: str) -> list[str]:
    """Build the list of agent prefixed-names by globbing ``agents_dir``.

    The assistant short-name ``assistant`` is included in the filesystem
    enumeration if its dir is present; no separate auto-append needed.
    """
    if not username:
        return []
    return [_prefixed_name(username, n) for n in _list_owned_agent_short_names(agents_dir, username)]


def _filter_requested(
    agent_names: list[str], requested: list[str] | None, username: str
) -> list[str]:
    """Keep only the owned ``agent_names`` that match the ``--agent`` filter.

    Shared by ``start`` / ``stop`` / ``status`` so the three commands target a
    name identically: each ``requested`` entry may be the short name
    ('researcher') or the prefixed form ('chengtao-researcher'); both match.
    An empty or ``None`` filter returns ``agent_names`` unchanged (the legacy
    all-agents behavior). A non-empty filter that matches nothing is a clean
    error (echo + ``Exit(1)``), never a partial action. Pure — no FS/process I/O.
    """
    cleaned = {a.strip() for a in (requested or []) if a and a.strip()}
    if not cleaned:
        return agent_names
    prefixed = {_prefixed_name(username, a) if username else a for a in cleaned}
    matched = [n for n in agent_names if n in prefixed or n in cleaned]
    if not matched:
        typer.echo(
            f"Error: --agent filter matched no agents. Requested: {sorted(cleaned)}",
            err=True,
        )
        raise typer.Exit(1)
    return matched


def _is_self(agent_dir: Path) -> bool:
    """True iff ``agent_dir`` is THIS runner's own agent dir.

    Compares against ``$CLAWMEETS_AGENT_DIR``, which is injected only inside a
    runner's LLM subprocess. Returns False when the env var is unset — a plain
    user-terminal ``clawmeets stop`` is never self-guarded and keeps its
    all-agents behavior. This is the hard CLI backstop behind the soft skill
    refusal: even if the skill's wording is bypassed, an in-runner stop will
    not kill the controlling runner.
    """
    self_dir = os.environ.get("CLAWMEETS_AGENT_DIR")
    if not self_dir:
        return False
    try:
        return agent_dir.resolve() == Path(self_dir).resolve()
    except OSError:
        return False


# ---------------------------------------------------------------------------
# start command
# ---------------------------------------------------------------------------


def start_command(
    server: Optional[str] = typer.Option(None, "--server", "-s", help="Server URL (overrides config)"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to settings.json"),
    user: Optional[str] = typer.Option(None, "--user", "-u", help="Username (overrides current_user)"),
    agent: list[str] = typer.Option(
        None, "--agent", "-a",
        help="Start only the given agent(s); repeatable. Accepts either the short "
             "name (e.g. 'sf-real-estate-analyst') or the prefixed form "
             "('chengtao-sf-real-estate-analyst'). When omitted, starts every "
             "agent the user owns under ~/.clawmeets/agents/.",
    ),
) -> None:
    """Start agents in the background.

    Reads the server URL and username from the current user's settings.json and
    starts each of their agents under ~/.clawmeets/agents/ as a background
    process. Pass ``--agent`` (repeatable) to start
    a specific subset; otherwise starts everything.

    Example:
        clawmeets start
        clawmeets start --user alice
        clawmeets start --agent sf-real-estate-analyst --agent cpa-tax
        clawmeets start --server https://my-server.com
    """
    if config_file:
        if not config_file.exists():
            typer.echo(f"Error: Config file not found: {config_file}", err=True)
            raise typer.Exit(1)
        config = json.loads(config_file.read_text())
    else:
        config, _ = load_user_config(Path(DEFAULT_DATA_DIR), user)

    server_url = server or config.get("server_url", DEFAULT_SERVER)
    agents_dir = _get_agents_dir()
    username = config.get("user", {}).get("username") or config.get("name", "")

    agent_names = _build_agent_list(agents_dir, username)

    if not agent_names:
        typer.echo(f"No agents found under {agents_dir}.")
        return

    # Same matching rule as `stop`/`status` (short or prefixed name).
    agent_names = _filter_requested(agent_names, agent, username)

    typer.echo("=== Start Agents ===\n")

    started = 0
    for name in agent_names:
        agent_dir = _find_agent_dir(agents_dir, name)
        if not agent_dir:
            typer.echo(f"  Agent '{name}' not found in {agents_dir}, skipping.")
            continue

        pid_file = agent_pid_file(agent_dir)
        existing_pid = _read_pid(pid_file)
        if existing_pid:
            typer.echo(f"  Agent '{name}' already running (PID {existing_pid})")
            continue

        # The runner reads knowledge_dir (and every other local setting) from
        # card.json itself, resolving relative paths against agent_dir.
        cmd = ["clawmeets", "agent", "run", "--server", server_url, "--agent-dir", str(agent_dir)]

        stdout_log = agent_dir / "stdout.log"
        stderr_log = agent_dir / "stderr.log"

        with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
            proc = subprocess.Popen(cmd, stdout=out, stderr=err, **_popen_detached_kwargs())

        pid_file.write_text(str(proc.pid))

        typer.echo(f"  Started '{name}' (PID {proc.pid})")
        typer.echo(f"    Logs: {stdout_log}")
        started += 1

    if started == 0:
        typer.echo("\nNo new agents started.")
    else:
        typer.echo(f"\n{started} agent(s) started.")
        typer.echo(f"\nOpen the dashboard: {server_url}/app")
        typer.echo("To stop agents: clawmeets stop")


# ---------------------------------------------------------------------------
# stop command
# ---------------------------------------------------------------------------


def stop_command(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to settings.json"),
    user: Optional[str] = typer.Option(None, "--user", "-u", help="Username (overrides current_user)"),
    agent: list[str] = typer.Option(
        None, "--agent", "-a",
        help="Stop only the given agent(s); repeatable. Accepts the short name "
             "or the prefixed form, exactly like `start --agent`. When omitted, "
             "stops every agent for the user (legacy behavior).",
    ),
) -> None:
    """Stop running agents.

    With ``--agent`` (repeatable) stops only the named subset, reusing the same
    graceful SIGTERM -> 5s -> SIGKILL escalation and stale-pidfile cleanup as
    the all-agents path (``_stop_pid``). Without it, preserves the legacy
    stop-everything behavior.

    Example:
        clawmeets stop
        clawmeets stop --user alice
        clawmeets stop --agent budget-analyst
    """
    if config_file:
        config = json.loads(config_file.read_text())
    else:
        config, _ = load_user_config(Path(DEFAULT_DATA_DIR), user)

    agents_dir = _get_agents_dir()
    username = config.get("user", {}).get("username") or config.get("name", "")
    agent_names = _build_agent_list(agents_dir, username)
    agent_names = _filter_requested(agent_names, agent, username)

    typer.echo("=== Stop Agents ===\n")

    stopped = 0
    skipped_self = False
    for name in agent_names:
        agent_dir = _find_agent_dir(agents_dir, name)
        if not agent_dir:
            continue
        # Hard self-stop backstop: never let an in-runner stop kill the
        # controlling runner (only fires when $CLAWMEETS_AGENT_DIR is set).
        if _is_self(agent_dir):
            typer.echo(
                f"  Refusing to stop '{name}' — it is the controlling runner; "
                f"an agent cannot stop itself."
            )
            skipped_self = True
            continue
        pid_file = agent_pid_file(agent_dir)
        if _stop_pid(pid_file, f"agent '{name}'"):
            stopped += 1

    if stopped == 0:
        typer.echo("  No agents stopped." if skipped_self else "  No agents were running.")
    else:
        typer.echo(f"\n{stopped} agent(s) stopped.")


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


def status_command(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to settings.json"),
    user: Optional[str] = typer.Option(None, "--user", "-u", help="Username (overrides current_user)"),
    agent: list[str] = typer.Option(
        None, "--agent", "-a",
        help="Show status for only the given agent(s); repeatable. Same name "
             "matching as `start`/`stop --agent`.",
    ),
) -> None:
    """Show status of agents.

    Each row is PID-verified via ``_read_pid`` (``_pid_is_alive``), so a crash
    that left a stale pidfile reads as ``dead (stale PID)`` rather than running.
    With ``--agent`` (repeatable) restricts the rows to the named subset.

    Example:
        clawmeets status
        clawmeets status --user alice
        clawmeets status --agent budget-analyst
    """
    if config_file:
        config = json.loads(config_file.read_text())
        config_path = config_file
    else:
        config, config_path = load_user_config(Path(DEFAULT_DATA_DIR), user)

    agents_dir = _get_agents_dir()
    username = config.get("user", {}).get("username") or config.get("name", "")
    agent_names = _build_agent_list(agents_dir, username)
    agent_names = _filter_requested(agent_names, agent, username)
    server_url = config.get("server_url", DEFAULT_SERVER)

    typer.echo("=== Agent Status ===\n")
    typer.echo(f"  Server:     {server_url}")
    typer.echo(f"  Config:     {config_path}")
    typer.echo(f"  Agents dir: {agents_dir}\n")

    for name in agent_names:
        agent_dir = _find_agent_dir(agents_dir, name)
        if not agent_dir:
            typer.echo(f"  {name:30s}  not registered")
            continue

        pid_file = agent_pid_file(agent_dir)
        pid = _read_pid(pid_file)
        if pid:
            typer.echo(f"  {name:30s}  running (PID {pid})")
        elif pid_file.exists():
            typer.echo(f"  {name:30s}  dead (stale PID)")
        else:
            typer.echo(f"  {name:30s}  stopped")
