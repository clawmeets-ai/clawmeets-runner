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
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer

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
) -> Path:
    """Upsert a user's settings.json with login session info and mark them current.

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
    user = config.setdefault("user", {})
    user["username"] = username
    user["token"] = token
    if refresh_token:
        user["refresh_token"] = refresh_token
        user["auth_method"] = auth_method
        user.pop("password", None)  # refresh-token renewal supersedes password-at-rest
    path.write_text(json.dumps(config, indent=2))
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
# ---------------------------------------------------------------------------

_IS_WINDOWS = sys.platform == "win32"


def _popen_detached_kwargs() -> dict:
    """Popen kwargs that detach the child so it outlives the parent shell.

    Windows needs DETACHED_PROCESS (no inherited console) plus
    CREATE_NEW_PROCESS_GROUP (so we can later deliver CTRL_BREAK_EVENT).
    POSIX just needs start_new_session=True.
    """
    if _IS_WINDOWS:
        flags = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
        return {"creationflags": flags}
    return {"start_new_session": True}


def _pid_is_alive(pid: int) -> bool:
    """Check whether a PID refers to a live process, without signaling it.

    On Windows, ``os.kill(pid, 0)`` actually terminates the target — so we
    must use a non-signaling query (tasklist) instead.
    """
    if _IS_WINDOWS:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, check=False,
        )
        return f'"{pid}"' in (result.stdout or "")
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _signal_terminate(pid: int) -> None:
    """Send a graceful termination request. Silently no-ops if the target is gone.

    POSIX: SIGTERM. Windows: CTRL_BREAK_EVENT to the process group (works
    because ``start_command`` spawns children with CREATE_NEW_PROCESS_GROUP).
    """
    try:
        if _IS_WINDOWS:
            os.kill(pid, getattr(signal, "CTRL_BREAK_EVENT", 15))
        else:
            os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass


def _signal_kill(pid: int) -> None:
    """Force-kill a process. Silently no-ops on failure.

    POSIX: SIGKILL. Windows: ``taskkill /F`` — TerminateProcess via the
    Win32 API equivalent, reliable even when graceful signaling didn't land.
    """
    try:
        if _IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _read_pid(pid_file: Path) -> int | None:
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
        return pid if _pid_is_alive(pid) else None
    except (ValueError, OSError):
        return None


def _stop_pid(pid_file: Path, label: str) -> bool:
    """Stop a process by PID file. Returns True if it was running.

    Graceful first (SIGTERM / CTRL_BREAK_EVENT), then a force kill after a
    5-second grace period (SIGKILL / taskkill /F).
    """
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        return False

    if not _pid_is_alive(pid):
        pid_file.unlink(missing_ok=True)
        return False

    _signal_terminate(pid)
    for _ in range(20):
        time.sleep(0.25)
        if not _pid_is_alive(pid):
            break
    else:
        _signal_kill(pid)

    pid_file.unlink(missing_ok=True)
    typer.echo(f"  Stopped {label} (PID {pid})")
    return True


def _get_agents_dir() -> Path:
    return Path(DEFAULT_DATA_DIR).expanduser() / "agents"


def _prefixed_name(username: str, agent_name: str) -> str:
    prefix = f"{username}-"
    return agent_name if agent_name.startswith(prefix) else f"{prefix}{agent_name}"


def _find_agent_dir(agents_dir: Path, prefixed_name: str) -> Path | None:
    """Find an agent's directory matching {prefixed_name}-{id}/."""
    if not agents_dir.exists():
        return None
    for d in agents_dir.iterdir():
        if d.is_dir() and d.name.startswith(f"{prefixed_name}-"):
            if (d / "credential.json").exists():
                return d
    return None


def _list_owned_agent_short_names(agents_dir: Path, username: str) -> list[str]:
    """Return owned agents' short names by globbing the filesystem.

    Pattern: ``{agents_dir}/{username}-{short}-{id}/`` with ``credential.json``
    present. Skips ``DELETED-*`` (renamed by self-destruct) and any dir
    without a ``credential.json`` (half-registered). The trailing ``-{id}``
    is stripped off the right.
    """
    if not agents_dir.exists():
        return []
    prefix = f"{username}-"
    names: list[str] = []
    for entry in sorted(agents_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("DELETED-"):
            continue
        if not entry.name.startswith(prefix):
            continue
        if not (entry / "credential.json").exists():
            continue
        rest = entry.name[len(prefix):]
        short = rest.rsplit("-", 1)[0] if "-" in rest else rest
        if short:
            names.append(short)
    return names


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
             "name as it appears in settings.json (e.g. 'sf-real-estate-analyst') "
             "or the prefixed form ('chengtao-sf-real-estate-analyst'). When "
             "omitted, starts every agent in settings.json plus the assistant.",
    ),
) -> None:
    """Start agents in the background.

    Reads agent configuration from the current user's settings.json and starts
    each agent as a background process. Pass ``--agent`` (repeatable) to start
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
        config_path = config_file
    else:
        config, config_path = load_user_config(Path(DEFAULT_DATA_DIR), user)

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

        pid_file = agent_dir / "agent.pid"
        existing_pid = _read_pid(pid_file)
        if existing_pid:
            typer.echo(f"  Agent '{name}' already running (PID {existing_pid})")
            continue

        # Read agent-specific config from card.json local_settings
        # (config.json is deprecated — local_settings in card.json is the primary source)
        knowledge_dir = ""
        card_path = agent_dir / "card.json"
        if card_path.exists():
            try:
                card_data = json.loads(card_path.read_text())
                local_settings = card_data.get("local_settings", {})
                knowledge_dir = local_settings.get("knowledge_dir", "")
            except json.JSONDecodeError:
                pass

        # Build command
        cmd = ["clawmeets", "agent", "run", "--server", server_url, "--agent-dir", str(agent_dir)]

        if knowledge_dir:
            cmd.extend(["--knowledge-dir", knowledge_dir])

        # Always pass --user-config so the runner can resolve relative
        # knowledge_dir strings against ~/.clawmeets/config/<username>/ —
        # the same base cli_init.py used when it wrote CLAUDE.md.
        cmd.extend(["--user-config", str(config_path)])

        stdout_log = agent_dir / "stdout.log"
        stderr_log = agent_dir / "stderr.log"

        with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
            proc = subprocess.Popen(cmd, stdout=out, stderr=err, **_popen_detached_kwargs())

        pid_file.write_text(str(proc.pid))

        info = f"  Started '{name}' (PID {proc.pid})"
        if knowledge_dir:
            info += f" [knowledge: {knowledge_dir}]"
        typer.echo(info)
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
        pid_file = agent_dir / "agent.pid"
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

        pid_file = agent_dir / "agent.pid"
        pid = _read_pid(pid_file)
        if pid:
            typer.echo(f"  {name:30s}  running (PID {pid})")
        elif pid_file.exists():
            typer.echo(f"  {name:30s}  dead (stale PID)")
        else:
            typer.echo(f"  {name:30s}  stopped")
