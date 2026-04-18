# SPDX-License-Identifier: MIT
"""
clawmeets/cli_lifecycle.py

Agent lifecycle commands: start, stop, status.

Replaces the corresponding project.sh commands with pure Python.

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


def save_user_session(data_dir: Path, username: str, server_url: str, token: str) -> Path:
    """Upsert a user's settings.json with login session info and mark them current.

    Creates the file with minimal scaffolding if it does not yet exist, so
    this works both for fresh accounts and already-configured users.
    """
    path = get_user_config_path(data_dir, username)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(path.read_text()) if path.exists() else {}
    config["server_url"] = server_url
    user = config.setdefault("user", {})
    user["username"] = username
    user["token"] = token
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


def add_agent_to_settings(data_dir: Path, username: str, agent_entry: dict) -> Path:
    """Append (or update) an agent entry in the user's settings.json agents[].

    Matches existing entries by name. Raises FileNotFoundError if the user
    has no settings.json yet — callers should ensure the user is logged in.
    """
    path = get_user_config_path(data_dir, username)
    if not path.exists():
        raise FileNotFoundError(
            f"No settings found for user '{username}'. Log in first with `clawmeets user login --save`."
        )
    config = json.loads(path.read_text())
    agents = config.setdefault("agents", [])
    name = agent_entry.get("name")
    for i, existing in enumerate(agents):
        if existing.get("name") == name:
            agents[i] = agent_entry
            break
    else:
        agents.append(agent_entry)
    path.write_text(json.dumps(config, indent=2))
    return path


def load_user_config(data_dir: Path, username: str | None = None) -> tuple[dict, Path]:
    """Load a user's settings.json. Uses current_user if username not specified.

    Also falls back to ./project.json in the current directory for backward
    compatibility with the project.sh workflow.
    """
    if username is None:
        username = get_current_user(data_dir)
    if username:
        path = get_user_config_path(data_dir, username)
        if path.exists():
            return json.loads(path.read_text()), path
    # Fallback: ./project.json in current directory (project.sh workflow)
    local = Path("project.json")
    if local.exists():
        return json.loads(local.read_text()), local
    if username:
        typer.echo(f"Error: No config for user '{username}'. Run `clawmeets init` first.", err=True)
    else:
        typer.echo("Error: No user configured. Run `clawmeets init` first.", err=True)
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


def _get_agents_dir(config: dict) -> Path:
    data_dir = config.get("data_dir", DEFAULT_DATA_DIR)
    return Path(data_dir).expanduser() / "agents"


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


def _build_agent_list(config: dict) -> list[str]:
    """Build the list of agent names (prefixed) from config."""
    username = config.get("user", {}).get("username") or config.get("name", "")
    agents = []
    for agent_def in config.get("agents", []):
        name = agent_def.get("name", "")
        if name and username:
            agents.append(_prefixed_name(username, name))
        elif name:
            agents.append(name)
    # Add assistant
    if username:
        agents.append(f"{username}-assistant")
    return agents


# ---------------------------------------------------------------------------
# start command
# ---------------------------------------------------------------------------


def start_command(
    server: Optional[str] = typer.Option(None, "--server", "-s", help="Server URL (overrides config)"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to settings.json"),
    user: Optional[str] = typer.Option(None, "--user", "-u", help="Username (overrides current_user)"),
) -> None:
    """Start all agents in the background.

    Reads agent configuration from the current user's settings.json and starts
    each agent as a background process.

    Example:
        clawmeets start
        clawmeets start --user alice
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
    agents_dir = _get_agents_dir(config)
    agent_names = _build_agent_list(config)

    if not agent_names:
        typer.echo("No agents found in config.")
        return

    # Read plugin config
    claude_plugin_dir = config.get("claude_plugin_dir", "")

    # Resolve relative paths from config directory
    if claude_plugin_dir and not claude_plugin_dir.startswith("/"):
        resolved = (config_path.parent / claude_plugin_dir).resolve()
        if resolved.exists():
            claude_plugin_dir = str(resolved)

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
        use_chrome = False
        card_path = agent_dir / "card.json"
        if card_path.exists():
            try:
                card_data = json.loads(card_path.read_text())
                local_settings = card_data.get("local_settings", {})
                knowledge_dir = local_settings.get("knowledge_dir", "")
                use_chrome = local_settings.get("use_chrome", False)
            except json.JSONDecodeError:
                pass

        # Build command
        cmd = ["clawmeets", "agent", "run", "--server", server_url, "--agent-dir", str(agent_dir)]

        if knowledge_dir:
            cmd.extend(["--knowledge-dir", knowledge_dir])
        if use_chrome:
            cmd.append("--chrome")
        if claude_plugin_dir:
            cmd.extend(["--claude-plugin-dir", claude_plugin_dir])

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
) -> None:
    """Stop all running agents.

    Example:
        clawmeets stop
        clawmeets stop --user alice
    """
    if config_file:
        config = json.loads(config_file.read_text())
    else:
        config, _ = load_user_config(Path(DEFAULT_DATA_DIR), user)

    agents_dir = _get_agents_dir(config)
    agent_names = _build_agent_list(config)

    typer.echo("=== Stop Agents ===\n")

    stopped = 0
    for name in agent_names:
        agent_dir = _find_agent_dir(agents_dir, name)
        if not agent_dir:
            continue
        pid_file = agent_dir / "agent.pid"
        if _stop_pid(pid_file, f"agent '{name}'"):
            stopped += 1

    if stopped == 0:
        typer.echo("  No agents were running.")
    else:
        typer.echo(f"\n{stopped} agent(s) stopped.")


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


def status_command(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to settings.json"),
    user: Optional[str] = typer.Option(None, "--user", "-u", help="Username (overrides current_user)"),
) -> None:
    """Show status of all agents.

    Example:
        clawmeets status
        clawmeets status --user alice
    """
    if config_file:
        config = json.loads(config_file.read_text())
        config_path = config_file
    else:
        config, config_path = load_user_config(Path(DEFAULT_DATA_DIR), user)

    agents_dir = _get_agents_dir(config)
    agent_names = _build_agent_list(config)
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
