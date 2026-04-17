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

CONFIG_DIR = Path.home() / ".clawmeets"
DEFAULT_SERVER = os.environ.get("CLAWMEETS_SERVER_URL", "https://clawmeets.ai")
DEFAULT_DATA_DIR = os.environ.get("CLAWMEETS_DATA", str(Path.home() / ".clawmeets_data"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config() -> tuple[dict | None, Path | None]:
    """Load project.json from ~/.clawmeets/ or current directory."""
    for candidate in [CONFIG_DIR / "project.json", Path("project.json")]:
        if candidate.exists():
            return json.loads(candidate.read_text()), candidate
    return None, None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _read_pid(pid_file: Path) -> int | None:
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
        return pid if _pid_is_alive(pid) else None
    except (ValueError, OSError):
        return None


def _stop_pid(pid_file: Path, label: str) -> bool:
    """Stop a process by PID file. Returns True if was running."""
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

    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        time.sleep(0.25)
        if not _pid_is_alive(pid):
            break
    else:
        # Force kill if still alive
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

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
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to project.json"),
) -> None:
    """Start all agents in the background.

    Reads agent configuration from ~/.clawmeets/project.json (or ./project.json)
    and starts each agent as a background process.

    Example:
        clawmeets start
        clawmeets start --server https://my-server.com
    """
    if config_file:
        if not config_file.exists():
            typer.echo(f"Error: Config file not found: {config_file}", err=True)
            raise typer.Exit(1)
        config = json.loads(config_file.read_text())
    else:
        config, config_path = _load_config()
        if config is None:
            typer.echo("Error: No project.json found. Run `clawmeets init` first.", err=True)
            raise typer.Exit(1)

    server_url = server or config.get("server_url", DEFAULT_SERVER)
    agents_dir = _get_agents_dir(config)
    agent_names = _build_agent_list(config)

    if not agent_names:
        typer.echo("No agents found in config.")
        return

    # Read git and plugin config
    git_url = config.get("git_url", "")
    git_ignored_folder = config.get("git_ignored_folder", "")
    claude_plugin_dir = config.get("claude_plugin_dir", "")

    # Resolve relative paths from config directory
    config_dir = CONFIG_DIR if (CONFIG_DIR / "project.json").exists() else Path(".")
    if git_url and not git_url.startswith("/") and not git_url.startswith("http") and not git_url.startswith("git@"):
        resolved = (config_dir / git_url).resolve()
        if resolved.exists():
            git_url = str(resolved)
    if claude_plugin_dir and not claude_plugin_dir.startswith("/"):
        resolved = (config_dir / claude_plugin_dir).resolve()
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

        # Read agent-specific config
        knowledge_dir = ""
        use_chrome = False
        agent_config_path = agent_dir / "config.json"
        if agent_config_path.exists():
            try:
                agent_config = json.loads(agent_config_path.read_text())
                knowledge_dir = agent_config.get("knowledge_dir", "")
                use_chrome = agent_config.get("use_chrome", False)
            except json.JSONDecodeError:
                pass

        # Build command
        cmd = ["clawmeets", "agent", "run", "--server", server_url, "--agent-dir", str(agent_dir)]

        if knowledge_dir:
            cmd.extend(["--knowledge-dir", knowledge_dir])
        if use_chrome:
            cmd.append("--chrome")
        if git_url:
            cmd.extend(["--git-url", git_url])
        if git_ignored_folder:
            cmd.extend(["--git-ignored-folder", git_ignored_folder])
        if claude_plugin_dir:
            cmd.extend(["--claude-plugin-dir", claude_plugin_dir])

        stdout_log = agent_dir / "stdout.log"
        stderr_log = agent_dir / "stderr.log"

        with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
            proc = subprocess.Popen(cmd, stdout=out, stderr=err, start_new_session=True)

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
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to project.json"),
) -> None:
    """Stop all running agents.

    Example:
        clawmeets stop
    """
    if config_file:
        config = json.loads(config_file.read_text())
    else:
        config, _ = _load_config()
        if config is None:
            typer.echo("Error: No project.json found. Run `clawmeets init` first.", err=True)
            raise typer.Exit(1)

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
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to project.json"),
) -> None:
    """Show status of all agents.

    Example:
        clawmeets status
    """
    if config_file:
        config = json.loads(config_file.read_text())
    else:
        config, config_path = _load_config()
        if config is None:
            typer.echo("Error: No project.json found. Run `clawmeets init` first.", err=True)
            raise typer.Exit(1)

    agents_dir = _get_agents_dir(config)
    agent_names = _build_agent_list(config)
    server_url = config.get("server_url", DEFAULT_SERVER)

    typer.echo("=== Agent Status ===\n")
    typer.echo(f"  Server:     {server_url}")
    typer.echo(f"  Config:     {config_path or 'project.json'}")
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
