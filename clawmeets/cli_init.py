# SPDX-License-Identifier: MIT
"""
clawmeets/cli_init.py

Interactive setup wizard for clawmeets. Generates
~/.clawmeets/config/{user}/settings.json and per-agent CLAUDE.md files,
then registers agents with the server.

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
VALID_LLM_PROVIDERS = ("claude", "openai", "gemini")

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

        typer.echo("    LLM backend (optional; leave empty to use 'claude')")
        llm_provider = ""
        while True:
            raw = _prompt("    LLM provider (claude / codex / gemini)", default="")
            if not raw:
                break
            if raw.lower() in VALID_LLM_PROVIDERS:
                llm_provider = raw.lower()
                break
            typer.echo(f"    Invalid provider. Choose one of: {', '.join(VALID_LLM_PROVIDERS)}")
        llm_model = _prompt("    LLM model (optional, provider-specific — e.g. 'o3' for codex, 'gemini-2.5-pro' for gemini)", default="")

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
            "llm_provider": llm_provider,
            "llm_model": llm_model,
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
    `clawmeets init` but the values are stored in settings.json for reference only.
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
    """Fetch a setup.json template. Accepts HTTP(S) URLs, file:// URLs, and
    plain filesystem paths (useful for iterating on templates locally before
    publishing them)."""
    # Resolve to a local path if the input is a file:// URL or has no scheme.
    if url.startswith("file://"):
        local_path: Optional[Path] = Path(url[len("file://"):])
    elif "://" not in url:
        local_path = Path(url).expanduser()
    else:
        local_path = None

    if local_path is not None:
        try:
            return json.loads(local_path.read_text())
        except FileNotFoundError:
            typer.echo(f"Error: Template file not found: {local_path}", err=True)
            raise typer.Exit(1)
        except json.JSONDecodeError:
            typer.echo(f"Error: Invalid JSON at {local_path}", err=True)
            raise typer.Exit(1)

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


def _resolve_knowledge_dir(raw_kdir: str, output_dir: Path) -> Path:
    """Expand a setup.json knowledge_dir path against ~, absolute, or relative-to-output."""
    if raw_kdir.startswith("~"):
        return Path(raw_kdir).expanduser()
    if raw_kdir.startswith("/"):
        return Path(raw_kdir)
    return output_dir / raw_kdir.lstrip("./")


def _generate_claude_md(agent: dict, output_dir: Path) -> tuple[Path, bool]:
    """Generate a CLAUDE.md specialty profile for an agent.

    Returns (knowledge_dir, wrote). If `{knowledge_dir}/CLAUDE.md` already
    exists it is left untouched — never clobber a user-edited profile on a
    re-run of `clawmeets init`.
    """
    knowledge_dir = _resolve_knowledge_dir(agent["knowledge_dir"], output_dir)
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    claude_md_path = knowledge_dir / "CLAUDE.md"
    if claude_md_path.exists():
        return knowledge_dir, False

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
    claude_md_path.write_text(claude_md)
    return knowledge_dir, True


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _login(server: str, username: str, password: str) -> dict:
    """Login and return the full response dict (token, assistant_agent_id, etc.)."""
    with httpx.Client(base_url=server, timeout=30) as client:
        resp = client.post("/auth/login", json={"username": username, "password": password})
        resp.raise_for_status()
        return resp.json()


def _fetch_assistant_token(server: str, jwt: str) -> str:
    """Fetch the authenticated user's assistant token from the server.

    Returns empty string on any failure so callers can fall through to
    the existing "no token yet" behavior.
    """
    try:
        with httpx.Client(base_url=server, timeout=30) as client:
            resp = client.get(
                "/auth/me/assistant-token",
                headers={"Authorization": f"Bearer {jwt}"},
            )
            resp.raise_for_status()
            return resp.json().get("assistant_token", "") or ""
    except Exception as e:
        typer.echo(f"  Warning: could not fetch assistant token automatically ({e}).", err=True)
        return ""


def _normalize_user_teams_from_setup(value) -> list[str]:
    """Read a `user_teams` field from setup.json. Accepts a list of strings
    or a comma-separated string for convenience. Returns a deduped, stripped
    list with order preserved.
    """
    if value is None:
        return []
    if isinstance(value, str):
        candidates = [t for t in value.split(",")]
    elif isinstance(value, list):
        candidates = value
    else:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for raw in candidates:
        if not isinstance(raw, str):
            continue
        stripped = raw.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        out.append(stripped)
    return out


def _register_agents(
    server: str,
    token: str,
    agents: list[dict],
    agents_dir: Path,
) -> None:
    """Register worker agents with the server and save credentials."""
    with httpx.Client(base_url=server, timeout=30) as client:
        for agent in agents:
            agent_user_teams = _normalize_user_teams_from_setup(agent.get("user_teams"))
            register_payload = {
                "name": agent["name"],
                "description": agent["description"],
                "capabilities": agent["capabilities"],
                "discoverable_through_registry": agent.get("discoverable", False),
            }
            if agent_user_teams:
                register_payload["user_teams"] = agent_user_teams
            resp = client.post(
                "/agents/register",
                json=register_payload,
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

            # Any sibling {registered_name}-<other_id>/ is a leftover from a
            # previous registration of the same name — the server already
            # dropped that agent, so the credentials inside are dead. Rename
            # them to DELETED-* so `clawmeets start`'s _find_agent_dir lands
            # on the fresh dir unambiguously (see cli_lifecycle._find_agent_dir).
            if agents_dir.exists():
                for sibling in agents_dir.iterdir():
                    if not sibling.is_dir():
                        continue
                    if sibling == agent_work_dir:
                        continue
                    if sibling.name.startswith("DELETED-"):
                        continue
                    if not sibling.name.startswith(f"{registered_name}-"):
                        continue
                    target = agents_dir / f"DELETED-{sibling.name}"
                    if target.exists():
                        continue
                    try:
                        sibling.rename(target)
                        typer.echo(f"  Archived stale {sibling.name} -> {target.name}")
                    except OSError as e:
                        typer.echo(f"  Warning: could not archive {sibling.name}: {e}", err=True)

            # Build local_settings from agent config
            local_settings = {}
            if agent.get("knowledge_dir"):
                local_settings["knowledge_dir"] = agent["knowledge_dir"]
            if agent.get("llm_provider"):
                provider = agent["llm_provider"].lower()
                if provider not in VALID_LLM_PROVIDERS:
                    typer.echo(
                        f"  Warning: skipping invalid llm_provider {agent['llm_provider']!r} for '{agent['name']}' "
                        f"(expected one of {VALID_LLM_PROVIDERS})",
                        err=True,
                    )
                else:
                    local_settings["llm_provider"] = provider
            if agent.get("llm_model"):
                local_settings["llm_model"] = agent["llm_model"]
            if "chrome" in agent:
                local_settings["use_chrome"] = bool(agent["chrome"])

            # Persist local_settings on the server card too, so the Agent
            # Settings page and server-side callers see the same value the
            # runner uses. The register endpoint does not take local_settings,
            # so we PUT it right after.
            if local_settings:
                put_resp = client.put(
                    f"/agents/{agent_id}",
                    json={"local_settings": local_settings},
                    headers={"Authorization": f"Bearer {token}"},
                )
                try:
                    put_resp.raise_for_status()
                except httpx.HTTPStatusError:
                    typer.echo(
                        f"  Warning: failed to sync local_settings for '{registered_name}': {put_resp.text}",
                        err=True,
                    )

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
            if agent_user_teams:
                card["user_teams"] = agent_user_teams
            (agent_work_dir / "card.json").write_text(json.dumps(card, indent=2))

            # Install any MCP servers listed in setup.json agents[].mcp_servers
            # via the same HTTP path the web UI uses. OAuth still requires the
            # user to run `clawmeets mcp auth <name>` on the runner.
            mcp_servers = agent.get("mcp_servers") or []
            if mcp_servers:
                mcp_resp = client.post(
                    f"/agents/{agent_id}/mcps",
                    json={"mcps": mcp_servers},
                    headers={"Authorization": f"Bearer {token}"},
                )
                if mcp_resp.status_code >= 400:
                    typer.echo(
                        f"  Warning: failed to install MCP servers "
                        f"{mcp_servers} for '{registered_name}': {mcp_resp.text}",
                        err=True,
                    )
                else:
                    added = mcp_resp.json().get("added", [])
                    if added:
                        typer.echo(f"    MCP servers installed: {', '.join(added)}")

            typer.echo(f"  Registered '{registered_name}' ({agent_id[:8]}...)")


def _build_assistant_local_settings(block: dict, label: str) -> dict:
    """Build a local_settings dict from an assistant/agent config block.

    Same validation rules as _register_agents: llm_provider must be in
    VALID_LLM_PROVIDERS, otherwise warn + skip.
    """
    local_settings: dict = {}
    if block.get("knowledge_dir"):
        local_settings["knowledge_dir"] = block["knowledge_dir"]
    if block.get("llm_provider"):
        provider = block["llm_provider"].lower()
        if provider not in VALID_LLM_PROVIDERS:
            typer.echo(
                f"  Warning: skipping invalid llm_provider {block['llm_provider']!r} for {label} "
                f"(expected one of {VALID_LLM_PROVIDERS})",
                err=True,
            )
        else:
            local_settings["llm_provider"] = provider
    if block.get("llm_model"):
        local_settings["llm_model"] = block["llm_model"]
    if "chrome" in block:
        local_settings["use_chrome"] = bool(block["chrome"])
    return local_settings


def _setup_assistant_credentials(
    server: str,
    login_response: dict,
    agents_dir: Path,
    local_settings: Optional[dict] = None,
    capabilities_override: Optional[list[str]] = None,
    description_override: Optional[str] = None,
) -> None:
    """Save assistant credentials locally from login response.

    If local_settings / capabilities_override / description_override are
    provided (from a setup.json assistant block), they are written into the
    local card.json so the runner picks them up on the next `clawmeets start`.
    """
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

    description = description_override or "Assistant agent"
    registered_at = ""
    if assistants:
        if not description_override:
            description = assistants[0].get("description", description)
        registered_at = assistants[0].get("registered_at", "")

    card: dict = {
        "id": assistant_id,
        "name": assistant_name,
        "description": description,
        "capabilities": capabilities_override if capabilities_override is not None else [],
        "status": "online",
        "registered_at": registered_at,
        "discoverable_through_registry": False,
    }
    if local_settings:
        card["local_settings"] = local_settings
    (assistant_dir / "card.json").write_text(json.dumps(card, indent=2))
    typer.echo(f"  Set up assistant '{assistant_name}' locally.")


def _apply_assistant_config(
    server: str,
    token: str,
    login_response: dict,
    assistant_block: dict,
    output_dir: Path,
) -> tuple[dict, Optional[list[str]], Optional[str]]:
    """Apply an `assistant` block from setup.json to the logged-in user's
    {username}-assistant agent on the server.

    Steps:
      1. PUT /agents/{assistant_id} with local_settings, capabilities, description.
      2. POST /agents/{assistant_id}/mcps if mcp_servers is present.
      3. Generate CLAUDE.md in the configured knowledge_dir, but only if one
         does not already exist (never clobber user-edited profiles).

    Returns (local_settings, capabilities_override, description_override) so
    the caller can mirror them into the local card.json.
    """
    assistant_id = login_response.get("assistant_agent_id")
    assistant_name = login_response.get("assistant_agent_name", "")
    if not assistant_id or not assistant_name:
        typer.echo("  Warning: login response missing assistant id/name; skipping assistant config.", err=True)
        return {}, None, None

    label = f"assistant '{assistant_name}'"
    local_settings = _build_assistant_local_settings(assistant_block, label)

    capabilities = assistant_block.get("capabilities")
    description = assistant_block.get("description")

    put_body: dict = {}
    if local_settings:
        put_body["local_settings"] = local_settings
    if capabilities is not None:
        put_body["capabilities"] = capabilities
    if description is not None:
        put_body["description"] = description

    with httpx.Client(base_url=server, timeout=30) as client:
        if put_body:
            put_resp = client.put(
                f"/agents/{assistant_id}",
                json=put_body,
                headers={"Authorization": f"Bearer {token}"},
            )
            try:
                put_resp.raise_for_status()
                typer.echo(f"  Applied assistant config for '{assistant_name}'.")
            except httpx.HTTPStatusError:
                typer.echo(
                    f"  Warning: failed to apply assistant config for '{assistant_name}': {put_resp.text}",
                    err=True,
                )

        mcp_servers = assistant_block.get("mcp_servers") or []
        if mcp_servers:
            mcp_resp = client.post(
                f"/agents/{assistant_id}/mcps",
                json={"mcps": mcp_servers},
                headers={"Authorization": f"Bearer {token}"},
            )
            if mcp_resp.status_code >= 400:
                typer.echo(
                    f"  Warning: failed to install MCP servers {mcp_servers} "
                    f"for '{assistant_name}': {mcp_resp.text}",
                    err=True,
                )
            else:
                added = mcp_resp.json().get("added", [])
                if added:
                    typer.echo(f"    MCP servers installed: {', '.join(added)}")

    # CLAUDE.md generation — _generate_claude_md skips if one already exists.
    if assistant_block.get("knowledge_dir") or assistant_block.get("profile"):
        synthetic_entry = {
            "name": assistant_name,
            "description": description or f"Assistant agent for {assistant_name}",
            "capabilities": capabilities or [],
            "knowledge_dir": assistant_block.get("knowledge_dir", f"./{assistant_name}"),
            "profile": assistant_block.get("profile", ""),
        }
        kdir, wrote = _generate_claude_md(synthetic_entry, output_dir)
        claude_md_path = kdir / "CLAUDE.md"
        if wrote:
            typer.echo(f"  Generated {claude_md_path}")
        else:
            typer.echo(f"  Skipped CLAUDE.md (already exists at {claude_md_path})")

    return local_settings, capabilities, description


# ---------------------------------------------------------------------------
# Config file generation
# ---------------------------------------------------------------------------


def _write_settings_json(
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
    assistant_block: Optional[dict] = None,
) -> Path:
    """Generate settings.json, merging agents with any existing config. Returns the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "settings.json"

    # Load existing config to preserve agents from prior runs
    existing_agents: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            existing_agents = existing.get("agents", [])
        except (json.JSONDecodeError, KeyError):
            pass

    # Build new agent entries
    new_agents = []
    for a in agents:
        entry: dict = {
            "name": a["name"],
            "description": a["description"],
            "capabilities": a["capabilities"],
            "discoverable": a.get("discoverable", False),
            "knowledge_dir": a["knowledge_dir"],
        }
        if a.get("llm_provider"):
            entry["llm_provider"] = a["llm_provider"].lower()
        if a.get("llm_model"):
            entry["llm_model"] = a["llm_model"]
        if "chrome" in a:
            entry["chrome"] = bool(a["chrome"])
        new_agents.append(entry)

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
    if assistant_block:
        config["assistant"] = assistant_block

    path.write_text(json.dumps(config, indent=2))
    return path


def _save_token_to_settings_json(token: str, username: str, data_dir: Path) -> Path:
    """Save JWT token to the user's settings.json after login/registration."""
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
    llm_provider: Optional[str] = typer.Option(
        None, "--llm-provider",
        help="Override LLM provider for every agent (and the assistant) in this setup. One of: claude, openai, gemini. Wins over per-agent llm_provider in setup.json.",
    ),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Skip interactive prompts (use flags)"),
    username: Optional[str] = typer.Option(None, "--username", "-u", help="Username (non-interactive)"),
    password: Optional[str] = typer.Option(None, "--password", "-p", help="Password (non-interactive)"),
    assistant_token: Optional[str] = typer.Option(None, "--assistant-token", help="Assistant token (non-interactive)"),
) -> None:
    """Interactive setup wizard: configure agents and register with the server.

    Generates settings.json and CLAUDE.md files, then registers your agents.
    After this, run `clawmeets start` to bring your agents online.

    Example (interactive):
        clawmeets init

    Example (from template):
        clawmeets init --from-url https://raw.githubusercontent.com/.../setup.json

    Example (non-interactive):
        clawmeets init --username alice --password secret --assistant-token abc123

    Example (from template, non-interactive — full scripted setup):
        clawmeets init --from-url /tmp/team.json --non-interactive \\
            --username alice --password secret --assistant-token abc123
    """
    typer.echo("\n=== ClawMeets Setup Wizard ===\n")
    typer.echo(f"  Before continuing, make sure you have an account at {server.rstrip('/')}/app/signup.")
    typer.echo("  If you haven't registered yet, sign up there first and verify your email.\n")

    if from_url:
        # ---- Template mode: fetch agent definitions, then collect credentials ----
        typer.echo("  Fetching template...\n")
        template = _fetch_setup_template(from_url)

        template_name = template.get("name", "Custom")
        template_desc = template.get("description", "")
        agents_list = template.get("agents", [])
        assistant_block: Optional[dict] = template.get("assistant")

        if not agents_list:
            typer.echo("Error: Template contains no agent definitions.", err=True)
            raise typer.Exit(1)

        typer.echo(f"  Template: {template_name}")
        if template_desc:
            typer.echo(f"  {template_desc}")
        typer.echo(f"  Agents:   {', '.join(a['name'] for a in agents_list)}\n")
        if assistant_block:
            typer.echo("  Assistant: config block present (will be applied to your personal assistant)\n")

        if non_interactive:
            if not username or not password:
                typer.echo("Error: --username and --password are required with --non-interactive", err=True)
                raise typer.Exit(1)
            name_err = _validate_name(username, "username")
            if name_err:
                typer.echo(f"Error: {name_err}", err=True)
                raise typer.Exit(1)
        else:
            typer.echo("--- Account Info ---")
            typer.echo(f"Enter the username and password you registered at {server.rstrip('/')}\n")

            username = _prompt_validated_name("  Username", "username")
            password = _prompt_required("  Password", password=True)
            password_confirm = _prompt_required("  Confirm password", password=True)
            if password != password_confirm:
                typer.echo("Error: Passwords do not match.", err=True)
                raise typer.Exit(1)

        _assistant_token = assistant_token or ""
        data_dir = DEFAULT_DATA_DIR
        git_url = ""
        git_ignored_folder = ""

    elif non_interactive:
        typer.echo("  This wizard will generate your agent team configuration,")
        typer.echo("  register your agents with the server, and prepare everything")
        typer.echo("  so you can run `clawmeets start`.\n")

        typer.echo("--- Account Info ---")
        typer.echo(f"Enter the username and password you registered at {server.rstrip('/')}\n")

        if not username or not password:
            typer.echo("Error: --username and --password required in non-interactive mode", err=True)
            raise typer.Exit(1)
        _assistant_token = assistant_token or ""
        data_dir = DEFAULT_DATA_DIR
        agents_list: list[dict] = []
        assistant_block = None
        git_url = ""
        git_ignored_folder = ""
    else:
        typer.echo("  This wizard will generate your agent team configuration,")
        typer.echo("  register your agents with the server, and prepare everything")
        typer.echo("  so you can run `clawmeets start`.\n")

        # ---- Account info ----
        typer.echo("--- Account Info ---")
        typer.echo(f"Enter the username and password you registered at {server.rstrip('/')}\n")

        username = _prompt_validated_name("  Username", "username")
        password = _prompt_required("  Password", password=True)
        password_confirm = _prompt_required("  Confirm password", password=True)
        if password != password_confirm:
            typer.echo("Error: Passwords do not match.", err=True)
            raise typer.Exit(1)

        _assistant_token = assistant_token or ""
        data_dir = _prompt("\n  Data directory", default=DEFAULT_DATA_DIR)

        agents_list = _collect_agents()
        assistant_block = None
        git_url, git_ignored_folder = _collect_git_repo()

    # ---- Apply --llm-provider override (wins over per-agent setup.json) ----
    if llm_provider:
        provider = llm_provider.lower()
        if provider not in VALID_LLM_PROVIDERS:
            typer.echo(
                f"Error: --llm-provider must be one of {', '.join(VALID_LLM_PROVIDERS)} "
                f"(got {llm_provider!r}).",
                err=True,
            )
            raise typer.Exit(1)
        for agent in agents_list:
            agent["llm_provider"] = provider
        if assistant_block:
            assistant_block["llm_provider"] = provider
        typer.echo(f"  LLM provider override: {provider} (applied to all agents)")

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
        if _assistant_token:
            typer.echo("  Assistant: token provided (override)")
        else:
            typer.echo("  Assistant: will be fetched from server after login")
        typer.echo(f"  Data dir: {data_dir}")
        if git_url:
            typer.echo(f"  Git repo: {git_url}")
            typer.echo(f"  Git-ignored folder: {git_ignored_folder}")
        typer.echo(f"  Agents:   {len(agents_list)}")
        for a in agents_list:
            typer.echo(f"    - {a['name']}: {a['description']}")
            typer.echo(f"      Capabilities: {', '.join(a['capabilities'])}")
            typer.echo(f"      Knowledge dir: {a['knowledge_dir']}")
            if a.get("llm_provider") or a.get("llm_model"):
                llm_bits = []
                if a.get("llm_provider"):
                    llm_bits.append(f"provider={a['llm_provider']}")
                if a.get("llm_model"):
                    llm_bits.append(f"model={a['llm_model']}")
                typer.echo(f"      LLM: {', '.join(llm_bits)}")
        typer.echo("")

        if not typer.confirm("  Proceed?", default=True):
            raise typer.Abort()

    # ---- Login ----
    typer.echo("\n--- Registering with server ---")
    try:
        login_resp = _login(server, username, password)
        token = login_resp["token"]
    except httpx.HTTPStatusError as e:
        typer.echo(f"  Login failed: {e.response.text}", err=True)
        typer.echo("  Re-run `clawmeets init` once the issue is resolved.")
        raise typer.Exit(1)
    except httpx.ConnectError:
        typer.echo(f"  Could not connect to {server}", err=True)
        typer.echo("  Re-run `clawmeets init` once the server is reachable.")
        raise typer.Exit(1)

    # ---- Fetch assistant token (unless explicitly overridden via --assistant-token) ----
    if not _assistant_token:
        _assistant_token = _fetch_assistant_token(server, token)
    login_resp["assistant_token"] = _assistant_token

    # ---- Generate settings.json ----
    typer.echo("\n--- Generating configuration ---")
    settings_path = _write_settings_json(
        output_dir,
        server_url=server,
        username=username,
        password=password,
        assistant_token=_assistant_token,
        data_dir=data_dir,
        agents=agents_list,
        git_url=git_url,
        git_ignored_folder=git_ignored_folder,
        assistant_block=assistant_block,
    )
    typer.echo(f"  Generated {settings_path}")

    # ---- Generate CLAUDE.md files (skipped per-agent if already present) ----
    for agent in agents_list:
        kdir, wrote = _generate_claude_md(agent, output_dir)
        claude_md_path = kdir / "CLAUDE.md"
        if wrote:
            typer.echo(f"  Generated {claude_md_path}")
        else:
            typer.echo(f"  Skipped CLAUDE.md (already exists at {claude_md_path})")

    # ---- Apply assistant config (template's `assistant` block) ----
    assistant_local_settings: dict = {}
    assistant_capabilities: Optional[list[str]] = None
    assistant_description: Optional[str] = None
    if assistant_block:
        (
            assistant_local_settings,
            assistant_capabilities,
            assistant_description,
        ) = _apply_assistant_config(server, token, login_resp, assistant_block, output_dir)

    # ---- Setup assistant credentials ----
    _setup_assistant_credentials(
        server,
        login_resp,
        agents_dir,
        local_settings=assistant_local_settings or None,
        capabilities_override=assistant_capabilities,
        description_override=assistant_description,
    )

    # ---- Register agents ----
    if agents_list:
        _register_agents(server, token, agents_list, agents_dir)

    # Sample requests are NOT copied from the template into the user's team
    # metadata. The TeamPage reads them live from `GET /templates/{name}` so
    # the source of truth stays in templates/*/setup.json.

    # ---- Save token to settings.json and set current user ----
    config_path = _save_token_to_settings_json(token, username, resolved_data_dir)
    from clawmeets.cli_lifecycle import set_current_user
    set_current_user(resolved_data_dir, username)
    typer.echo(f"  Saved session to {config_path}")

    # ---- Done ----
    typer.echo("\n=== Setup Complete ===\n")
    typer.echo("  Next steps:\n")
    typer.echo("    1. Start your agents:")
    typer.echo("       clawmeets start\n")
    typer.echo("    2. Open the dashboard to create projects:")
    typer.echo(f"       {server.rstrip('/')}/app\n")
    typer.echo("    3. When done, stop agents:")
    typer.echo("       clawmeets stop\n")
    typer.echo("  Tip: Customize your agents by editing the CLAUDE.md files")
    typer.echo(f"  in each agent's knowledge directory.\n")
