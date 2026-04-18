# SPDX-License-Identifier: MIT
"""
clawmeets/cli_init.py

Interactive setup wizard for clawmeets.

Replaces scripts/setup.sh — generates ~/.clawmeets/project.json and per-agent
CLAUDE.md files, then registers agents with the server.

Usage:
    clawmeets init
"""
from __future__ import annotations

import getpass
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import httpx
import typer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESERVED_NAMES = {"admin", "system", "root", "agent", "agents", "user", "users", "assistant"}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

DEFAULT_SERVER = os.environ.get("CLAWMEETS_SERVER_URL", "https://clawmeets.ai")
DEFAULT_DATA_DIR = os.environ.get("CLAWMEETS_DATA_DIR", "~/.clawmeets")

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_name(name: str, label: str) -> str | None:
    """Return error message if name is invalid, else None."""
    if not NAME_RE.match(name):
        return f"Invalid {label}: '{name}'. Must start with a lowercase letter and contain only lowercase letters, numbers, and underscores."
    if name in RESERVED_NAMES:
        return f"'{name}' is a reserved name. Please choose another."
    return None


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------


def _prompt(text: str, *, default: str = "", password: bool = False) -> str:
    """Prompt with optional default. Returns stripped input."""
    if default:
        display = f"{text} [{default}]: "
    else:
        display = f"{text}: "
    if password:
        val = getpass.getpass(display)
    else:
        val = input(display)
    return val.strip() or default


def _prompt_required(text: str, *, password: bool = False) -> str:
    while True:
        val = _prompt(text, password=password)
        if val:
            return val
        typer.echo("  This field is required.")


def _prompt_validated_name(text: str, label: str, *, existing: set[str] | None = None) -> str:
    while True:
        name = _prompt_required(text)
        err = _validate_name(name, label)
        if err:
            typer.echo(f"  {err}")
            continue
        if existing and name in existing:
            typer.echo(f"  Agent '{name}' already added. Choose a different name.")
            continue
        return name


# ---------------------------------------------------------------------------
# Agent collection
# ---------------------------------------------------------------------------


def _collect_agents() -> list[dict]:
    """Interactively collect agent definitions."""
    typer.echo("\n--- Agent Setup ---")
    typer.echo("Define your AI agents. Each agent has a name, description, capabilities,")
    typer.echo("and an optional detailed profile.\n")

    agents = []
    existing_names: set[str] = set()

    while True:
        typer.echo(f"  Agent #{len(agents) + 1}")
        name = _prompt_validated_name("    Name (lowercase, e.g. 'backend', 'designer')", "agent name", existing=existing_names)
        description = _prompt_required("    Description (e.g. 'Backend Engineer - implements Python APIs')")
        capabilities_str = _prompt_required("    Capabilities (comma-separated, e.g. 'Python,FastAPI,async')")
        capabilities = [c.strip() for c in capabilities_str.split(",") if c.strip()]

        knowledge_dir = _prompt(f"    Knowledge directory", default=f"./{name}")

        typer.echo("    Detailed profile (optional, describe expertise. Enter empty line to finish):")
        profile_lines = []
        while True:
            line = input("    > ")
            if not line:
                break
            profile_lines.append(line)
        profile = "\n".join(profile_lines) if profile_lines else ""

        agents.append({
            "name": name,
            "description": description,
            "capabilities": capabilities,
            "knowledge_dir": knowledge_dir,
            "discoverable": False,
            "_profile": profile,  # internal, used for CLAUDE.md generation
        })
        existing_names.add(name)
        typer.echo(f"\n    Added agent '{name}'\n")

        more = _prompt("  Add another agent? (y/n)", default="n")
        if more.lower() != "y":
            break

    if not agents:
        typer.echo("Error: At least one agent is required.", err=True)
        raise typer.Exit(1)

    return agents


# ---------------------------------------------------------------------------
# Git repo collection (optional)
# ---------------------------------------------------------------------------


def _collect_git_repo() -> tuple[str, str]:
    """Optionally collect git repo config. Returns (git_url, git_ignored_folder).

    Note: Git configuration is now per-project (set at project creation time
    via the web UI). This CLI prompt is kept for backward compatibility with
    `clawmeets init` but the values are stored in project.json for reference only.
    """
    typer.echo("\n--- Git Repository (Optional) ---")
    typer.echo("A git repo enables code-aware agent sandboxes. When set, agents clone this")
    typer.echo("repo as their working directory and work on branches per task.\n")

    git_url = _prompt("  Git repository URL (leave empty to skip)")
    if not git_url:
        return "", ""

    git_ignored_folder = _prompt("  Git-ignored folder for deliverables", default=".bus-files")
    return git_url, git_ignored_folder


# ---------------------------------------------------------------------------
# CLAUDE.md generation
# ---------------------------------------------------------------------------


def _fetch_setup_template(url: str) -> dict:
    """Fetch a setup.json template from a URL. Returns parsed dict."""
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        typer.echo(f"Error: Failed to fetch template: {e.response.status_code}", err=True)
        raise typer.Exit(1)
    except httpx.ConnectError:
        typer.echo(f"Error: Could not connect to {url}", err=True)
        raise typer.Exit(1)
    except json.JSONDecodeError:
        typer.echo(f"Error: Invalid JSON at {url}", err=True)
        raise typer.Exit(1)


def _generate_claude_md(agent: dict, output_dir: Path) -> Path:
    """Generate a CLAUDE.md specialty profile for an agent. Returns the knowledge dir path."""
    raw_kdir = agent["knowledge_dir"]
    if raw_kdir.startswith("~"):
        knowledge_dir = Path(raw_kdir).expanduser()
    elif raw_kdir.startswith("/"):
        knowledge_dir = Path(raw_kdir)
    else:
        knowledge_dir = output_dir / raw_kdir.lstrip("./")

    knowledge_dir.mkdir(parents=True, exist_ok=True)

    # Format display name
    display_name = agent["name"].replace("_", " ").title()

    # Build skill table
    skill_rows = "\n".join(f"| {cap} | Expert |" for cap in agent["capabilities"])

    # Build specialties — check both "profile" (from setup.json) and "_profile" (from interactive)
    profile = agent.get("profile") or agent.get("_profile", "")
    if profile:
        specialties = profile
    else:
        specialties = "\n".join(f"- {cap}" for cap in agent["capabilities"])

    claude_md = f"""# {display_name} - Specialty Profile

## Role

{agent['description']}

## Core Specialties

{specialties}

## Skill Set

| Skill | Proficiency |
|-------|-------------|
{skill_rows}

## Strengths

<!-- Customize this section based on your agent's specific strengths -->
<!-- Example: -->
<!-- 1. **Deep Expertise** - Extensive knowledge in core domain -->
<!-- 2. **Clear Communication** - Produces well-structured deliverables -->

## Deliverable Formats

<!-- Define the output formats your agent should produce -->
<!-- Example: -->
<!-- - `REPORT.md` - Analysis report with findings and recommendations -->
<!-- - `PLAN.md` - Action plan with timeline and milestones -->
"""
    (knowledge_dir / "CLAUDE.md").write_text(claude_md)
    return knowledge_dir


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _login(server: str, username: str, password: str) -> dict:
    """Login and return the full response dict (token, assistant_agent_id, etc.)."""
    with httpx.Client(base_url=server, timeout=30) as client:
        resp = client.post("/auth/login", json={"username": username, "password": password})
        resp.raise_for_status()
        return resp.json()


def _register_agents(
    server: str,
    token: str,
    agents: list[dict],
    agents_dir: Path,
) -> None:
    """Register worker agents with the server and save credentials."""
    with httpx.Client(base_url=server, timeout=30) as client:
        for agent in agents:
            resp = client.post(
                "/agents/register",
                json={
                    "name": agent["name"],
                    "description": agent["description"],
                    "capabilities": agent["capabilities"],
                    "discoverable_through_registry": agent.get("discoverable", False),
                },
                headers={"Authorization": f"Bearer {token}"},
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError:
                typer.echo(f"  Failed to register '{agent['name']}': {resp.text}", err=True)
                continue

            result = resp.json()
            registered_name = result.get("agent_name", agent["name"])
            agent_id = result["agent_id"]

            # Save credentials
            agent_work_dir = agents_dir / f"{registered_name}-{agent_id}"
            agent_work_dir.mkdir(parents=True, exist_ok=True)

            cred = {"agent_id": agent_id, "token": result["token"], "agent_name": registered_name}
            (agent_work_dir / "credential.json").write_text(json.dumps(cred, indent=2))

            # Build local_settings from agent config
            local_settings = {}
            if agent.get("knowledge_dir"):
                local_settings["knowledge_dir"] = agent["knowledge_dir"]

            card = {
                "id": agent_id,
                "name": registered_name,
                "description": result["description"],
                "capabilities": result.get("capabilities", []),
                "status": result["status"],
                "registered_at": result["registered_at"],
                "discoverable_through_registry": result.get("discoverable_through_registry", False),
                "local_settings": local_settings,
            }
            (agent_work_dir / "card.json").write_text(json.dumps(card, indent=2))

            typer.echo(f"  Registered '{registered_name}' ({agent_id[:8]}...)")


def _setup_assistant_credentials(
    server: str,
    login_response: dict,
    agents_dir: Path,
) -> None:
    """Save assistant credentials locally from login response."""
    assistant_id = login_response.get("assistant_agent_id")
    if not assistant_id:
        return

    token = login_response.get("token", "")
    # Fetch assistant info
    try:
        with httpx.Client(base_url=server, timeout=30) as client:
            resp = client.get("/assistants", headers={"Authorization": f"Bearer {token}"})
            resp.raise_for_status()
            assistants = resp.json()
    except Exception:
        assistants = []

    assistant_name = login_response.get("assistant_agent_name", "")
    if not assistant_name:
        return

    assistant_dir = agents_dir / f"{assistant_name}-{assistant_id}"
    if assistant_dir.exists():
        typer.echo(f"  Assistant '{assistant_name}' already configured locally.")
        return

    assistant_dir.mkdir(parents=True, exist_ok=True)

    # Find assistant token from login response or assistants list
    assistant_token = login_response.get("assistant_token", "")

    cred = {"agent_id": assistant_id, "token": assistant_token, "agent_name": assistant_name}
    (assistant_dir / "credential.json").write_text(json.dumps(cred, indent=2))

    description = "Assistant agent"
    registered_at = ""
    if assistants:
        description = assistants[0].get("description", description)
        registered_at = assistants[0].get("registered_at", "")

    card = {
        "id": assistant_id,
        "name": assistant_name,
        "description": description,
        "capabilities": [],
        "status": "online",
        "registered_at": registered_at,
        "discoverable_through_registry": False,
    }
    (assistant_dir / "card.json").write_text(json.dumps(card, indent=2))
    typer.echo(f"  Set up assistant '{assistant_name}' locally.")


# ---------------------------------------------------------------------------
# Config file generation
# ---------------------------------------------------------------------------


def _write_project_json(
    output_dir: Path,
    *,
    server_url: str,
    username: str,
    password: str,
    assistant_token: str,
    data_dir: str,
    agents: list[dict],
    git_url: str,
    git_ignored_folder: str,
) -> Path:
    """Generate project.json, merging agents with any existing config. Returns the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "project.json"

    # Load existing config to preserve agents from prior runs
    existing_agents: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            existing_agents = existing.get("agents", [])
        except (json.JSONDecodeError, KeyError):
            pass

    # Build new agent entries
    new_agents = [
        {
            "name": a["name"],
            "description": a["description"],
            "capabilities": a["capabilities"],
            "discoverable": a.get("discoverable", False),
            "knowledge_dir": a["knowledge_dir"],
        }
        for a in agents
    ]

    # Merge: new agents override existing ones with the same name
    new_names = {a["name"] for a in new_agents}
    merged = [a for a in existing_agents if a["name"] not in new_names] + new_agents

    config: dict = {
        "server_url": server_url,
        "name": username,
        "user": {"username": username, "password": password},
        "agents": merged,
        "agent_pool": "owned",
    }
    if assistant_token:
        config["user"]["assistant_token"] = assistant_token
    if data_dir != DEFAULT_DATA_DIR:
        config["data_dir"] = data_dir
    if git_url:
        config["git_url"] = git_url
    if git_url and git_ignored_folder:
        config["git_ignored_folder"] = git_ignored_folder

    path.write_text(json.dumps(config, indent=2))
    return path


def _save_token_to_project_json(token: str, username: str, data_dir: Path) -> Path:
    """Save JWT token to the user's project.json after login/registration."""
    from clawmeets.cli_lifecycle import get_user_config_path
    config_path = get_user_config_path(data_dir, username)
    config = json.loads(config_path.read_text())
    config.setdefault("user", {})["token"] = token
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


def init_command(
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s", help="Server URL"),
    from_url: Optional[str] = typer.Option(None, "--from-url", help="URL to a setup.json template (skips agent definition)"),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Skip interactive prompts (use flags)"),
    username: Optional[str] = typer.Option(None, "--username", "-u", help="Username (non-interactive)"),
    password: Optional[str] = typer.Option(None, "--password", "-p", help="Password (non-interactive)"),
    assistant_token: Optional[str] = typer.Option(None, "--assistant-token", help="Assistant token (non-interactive)"),
) -> None:
    """Interactive setup wizard: configure agents and register with the server.

    Generates project.json and CLAUDE.md files, then registers your agents.
    After this, run `clawmeets start` to bring your agents online.

    Example (interactive):
        clawmeets init

    Example (from template):
        clawmeets init --from-url https://raw.githubusercontent.com/.../setup.json

    Example (non-interactive):
        clawmeets init --username alice --password secret --assistant-token abc123
    """
    typer.echo("\n=== ClawMeets Setup Wizard ===\n")

    if from_url:
        # ---- Template mode: fetch agent definitions, prompt only for credentials ----
        typer.echo("  Fetching template...\n")
        template = _fetch_setup_template(from_url)

        template_name = template.get("name", "Custom")
        template_desc = template.get("description", "")
        agents_list = template.get("agents", [])

        if not agents_list:
            typer.echo("Error: Template contains no agent definitions.", err=True)
            raise typer.Exit(1)

        typer.echo(f"  Template: {template_name}")
        if template_desc:
            typer.echo(f"  {template_desc}")
        typer.echo(f"  Agents:   {', '.join(a['name'] for a in agents_list)}\n")

        typer.echo("--- Account Info ---")
        typer.echo("Enter the username and password you registered at clawmeets.ai\n")

        username = _prompt_validated_name("  Username", "username")
        password = _prompt_required("  Password", password=True)
        password_confirm = _prompt_required("  Confirm password", password=True)
        if password != password_confirm:
            typer.echo("Error: Passwords do not match.", err=True)
            raise typer.Exit(1)

        typer.echo("\n  Your assistant token was shown after signing up at clawmeets.ai.")
        _assistant_token = _prompt("  Assistant token (leave empty to add later)", password=True)

        data_dir = DEFAULT_DATA_DIR
        git_url = ""
        git_ignored_folder = ""

    elif non_interactive:
        typer.echo("  This wizard will generate your agent team configuration,")
        typer.echo("  register your agents with the server, and prepare everything")
        typer.echo("  so you can run `clawmeets start`.\n")

        typer.echo("--- Account Info ---")
        typer.echo("Enter the username and password you registered at clawmeets.ai\n")

        if not username or not password:
            typer.echo("Error: --username and --password required in non-interactive mode", err=True)
            raise typer.Exit(1)
        _assistant_token = assistant_token or ""
        data_dir = DEFAULT_DATA_DIR
        agents_list: list[dict] = []
        git_url = ""
        git_ignored_folder = ""
    else:
        typer.echo("  This wizard will generate your agent team configuration,")
        typer.echo("  register your agents with the server, and prepare everything")
        typer.echo("  so you can run `clawmeets start`.\n")

        # ---- Account info ----
        typer.echo("--- Account Info ---")
        typer.echo("Enter the username and password you registered at clawmeets.ai\n")

        username = _prompt_validated_name("  Username", "username")
        password = _prompt_required("  Password", password=True)
        password_confirm = _prompt_required("  Confirm password", password=True)
        if password != password_confirm:
            typer.echo("Error: Passwords do not match.", err=True)
            raise typer.Exit(1)

        typer.echo("\n  Your assistant token was shown after signing up at clawmeets.ai.")
        _assistant_token = _prompt("  Assistant token (leave empty to add later)", password=True)

        data_dir = _prompt("\n  Data directory", default=DEFAULT_DATA_DIR)

        agents_list = _collect_agents()
        git_url, git_ignored_folder = _collect_git_repo()

    # ---- Resolve data dir ----
    resolved_data_dir = Path(data_dir).expanduser()
    agents_dir = resolved_data_dir / "agents"

    # ---- Output directory (per-user config) ----
    output_dir = resolved_data_dir / "config" / username

    # ---- Confirmation ----
    if not non_interactive:
        typer.echo("\n=== Setup Summary ===\n")
        typer.echo(f"  Server:   {server}")
        typer.echo(f"  Username: {username}")
        typer.echo(f"  Assistant: {'token provided' if _assistant_token else 'no token (can be added later)'}")
        typer.echo(f"  Data dir: {data_dir}")
        if git_url:
            typer.echo(f"  Git repo: {git_url}")
            typer.echo(f"  Git-ignored folder: {git_ignored_folder}")
        typer.echo(f"  Agents:   {len(agents_list)}")
        for a in agents_list:
            typer.echo(f"    - {a['name']}: {a['description']}")
            typer.echo(f"      Capabilities: {', '.join(a['capabilities'])}")
            typer.echo(f"      Knowledge dir: {a['knowledge_dir']}")
        typer.echo("")

        if not typer.confirm("  Proceed?"):
            raise typer.Abort()

    # ---- Generate project.json ----
    typer.echo("\n--- Generating configuration ---")
    project_path = _write_project_json(
        output_dir,
        server_url=server,
        username=username,
        password=password,
        assistant_token=_assistant_token,
        data_dir=data_dir,
        agents=agents_list,
        git_url=git_url,
        git_ignored_folder=git_ignored_folder,
    )
    typer.echo(f"  Generated {project_path}")

    # ---- Generate CLAUDE.md files ----
    for agent in agents_list:
        kdir = _generate_claude_md(agent, output_dir)
        typer.echo(f"  Generated {kdir / 'CLAUDE.md'}")

    # ---- Login ----
    typer.echo("\n--- Registering with server ---")
    try:
        login_resp = _login(server, username, password)
        token = login_resp["token"]
    except httpx.HTTPStatusError as e:
        typer.echo(f"  Login failed: {e.response.text}", err=True)
        typer.echo("  Agents were not registered. You can register later with: clawmeets init")
        typer.echo(f"\n  Configuration saved to {output_dir}/")
        raise typer.Exit(1)
    except httpx.ConnectError:
        typer.echo(f"  Could not connect to {server}", err=True)
        typer.echo("  Agents were not registered. You can register later with: clawmeets init")
        typer.echo(f"\n  Configuration saved to {output_dir}/")
        raise typer.Exit(1)

    # ---- Setup assistant credentials ----
    _setup_assistant_credentials(server, login_resp, agents_dir)

    # ---- Register agents ----
    if agents_list:
        _register_agents(server, token, agents_list, agents_dir)

    # ---- Save token to project.json and set current user ----
    config_path = _save_token_to_project_json(token, username, resolved_data_dir)
    from clawmeets.cli_lifecycle import set_current_user
    set_current_user(resolved_data_dir, username)
    typer.echo(f"  Saved session to {config_path}")

    # ---- Done ----
    typer.echo("\n=== Setup Complete ===\n")
    typer.echo("  Next steps:\n")
    typer.echo("    1. Start your agents:")
    typer.echo("       clawmeets start\n")
    typer.echo("    2. Open the dashboard to create projects:")
    typer.echo("       https://clawmeets.ai/app\n")
    typer.echo("    3. When done, stop agents:")
    typer.echo("       clawmeets stop\n")
    typer.echo("  Tip: Customize your agents by editing the CLAUDE.md files")
    typer.echo(f"  in each agent's knowledge directory.\n")
