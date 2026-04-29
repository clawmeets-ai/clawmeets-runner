# SPDX-License-Identifier: MIT
"""
clawmeets/cli_runner.py

Runner-side CLI commands for clawmeets.

Commands
--------
  agent register            Register a new agent with the server
  agent run                 Start an agent runner (connects, listens for work)
  agent list                List all registered agents
  user login                Login and print JWT token
  user create               Create a new user with assistant agent
  user list                 List all users
  user listen               Listen for notifications from assistant
  dm send                   Send a direct message to an agent
  dm list                   List DM conversations
  dm history                Show DM history with an agent
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import httpx
import typer
import websockets

from clawmeets.api.responses import AgentRegistrationResponse
from clawmeets.api.control import ControlEnvelope, ControlMessageType
from clawmeets.api.client import ClawMeetsClient
from clawmeets.models.chat_message import ChatMessage
from clawmeets.utils.file_io import FileUtil
from clawmeets.utils.knowledge_dir import resolve_local_knowledge_dir
from clawmeets.utils.notification_center import NotificationCenter
from clawmeets.llm.base import LLMProvider
from clawmeets.llm.claude_cli import ClaudeCLI
from clawmeets.llm.codex_cli import CodexCLI
from clawmeets.llm.gemini_cli import GeminiCLI
from clawmeets.models.context import ModelContext
from clawmeets.models.agent import Agent
from clawmeets.models.user import User, NotificationConfig
from clawmeets.sync.console_subscriber import ConsoleOutputSubscriber, ConsoleConfig
from clawmeets.runner.reactive_loop import ReactiveControlLoop
from clawmeets.runner.mcp_manager import McpManager
from clawmeets.runner.personal_skill_manager import PersonalSkillManager
from clawmeets.runner.skill_manager import SkillManager

# Backward compatibility alias
AgentRegistrationResult = AgentRegistrationResponse

# Sub-command groups
agent_app = typer.Typer(help="Agent commands", no_args_is_help=True)
user_app  = typer.Typer(help="User commands",  no_args_is_help=True)
dm_app    = typer.Typer(help="Direct message commands", no_args_is_help=True)
mcp_app   = typer.Typer(help="MCP server commands (auth, list, status)", no_args_is_help=True)
team_app  = typer.Typer(help="Manage user-defined teams on your agents (the TEAMS sidebar)", no_args_is_help=True)
reflection_app = typer.Typer(help="Configure account-level reflection schedule (one cron, fans out to all your agents).", no_args_is_help=True)
bootstrap_app = typer.Typer(help="Personalize your fresh agents from your own data (one-time, opt-in).", invoke_without_command=True)


def _default_user_teams_from_env() -> list[str]:
    """Parse $CLAWMEETS_AGENT_TEAMS into a list (comma-separated). Empty list
    if unset. Used as the default for `clawmeets agent register --team` when
    no flags are passed.
    """
    raw = os.environ.get("CLAWMEETS_AGENT_TEAMS", "")
    return [t.strip() for t in raw.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Global options (env-var defaults)
# ---------------------------------------------------------------------------

DEFAULT_SERVER = os.environ.get("CLAWMEETS_SERVER_URL", "https://clawmeets.ai")
DEFAULT_DATA_DIR = os.environ.get("CLAWMEETS_DATA_DIR", str(Path.home() / ".clawmeets"))


def _server_url(server: str) -> str:
    return server.rstrip("/")


def _http(server: str) -> httpx.Client:
    return httpx.Client(base_url=_server_url(server), timeout=30)


def _ok(resp: httpx.Response) -> dict:
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        typer.echo(f"Error {resp.status_code}: {resp.text}", err=True)
        raise typer.Exit(1) from e
    return resp.json()


def _print_json(data: dict | list) -> None:
    typer.echo(json.dumps(data, indent=2, default=str))


_VALID_LLM_PROVIDERS = ("claude", "openai", "gemini")


def _build_initial_local_settings(
    llm_provider: Optional[str],
    llm_model: Optional[str],
) -> dict:
    """Build the local_settings block for a freshly generated card.json.

    Exits with code 1 if llm_provider is not one of the supported values.
    """
    settings: dict = {}
    if llm_provider:
        normalized = llm_provider.lower()
        if normalized not in _VALID_LLM_PROVIDERS:
            typer.echo(
                f"Error: --llm-provider must be one of {_VALID_LLM_PROVIDERS} "
                f"(got {llm_provider!r})",
                err=True,
            )
            raise typer.Exit(1)
        settings["llm_provider"] = normalized
    if llm_model:
        settings["llm_model"] = llm_model
    return settings


# ---------------------------------------------------------------------------
# agent register
# ---------------------------------------------------------------------------

@agent_app.command("register")
def agent_register(
    name: Optional[str] = typer.Argument(None, help="Agent name (required unless --from-card)"),
    description: Optional[str] = typer.Argument(None, help="Short description (required unless --from-card)"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="User JWT token (defaults to the saved token of --as-user or current_user)"),
    server: Optional[str] = typer.Option(None, "--server", "-s", help="Server URL (defaults to the server of --as-user or current_user, else env/default)"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Root data directory (agents at {data_dir}/agents/)"),
    save: Optional[Path] = typer.Option(None, "--save", help="Save credentials to custom path (overrides data-dir)"),
    discoverable: Optional[bool] = typer.Option(None, "--discoverable/--no-discoverable", help="Show agent in /agents list"),
    capabilities: Optional[str] = typer.Option(None, "--capabilities", "-c", help="Comma-separated list of capabilities"),
    from_card: Optional[Path] = typer.Option(None, "--from-card", help="Path to card.json to register from"),
    save_to_settings: bool = typer.Option(
        False, "--save-to-settings",
        help="Also append this agent to the logged-in user's settings.json agents[].",
    ),
    knowledge_dir: Optional[str] = typer.Option(
        None, "--knowledge-dir",
        help="Local knowledge directory for this agent (saved to settings.json; only meaningful with --save-to-settings).",
    ),
    as_user: Optional[str] = typer.Option(
        None, "--as-user",
        help="Username for --save-to-settings (defaults to current_user).",
    ),
    llm_provider: Optional[str] = typer.Option(
        None, "--llm-provider",
        help="LLM backend for this agent: 'claude' (default), 'openai', or 'gemini'. Written to card.json local_settings.",
    ),
    llm_model: Optional[str] = typer.Option(
        None, "--llm-model",
        help="Provider-specific model name (e.g. 'o3' for Codex, 'gemini-2.5-pro' for Gemini). Written to card.json local_settings.",
    ),
    team: list[str] = typer.Option(
        None, "--team",
        help="User-defined team for this agent (repeatable; appears under the TEAMS sidebar). "
             "Defaults to $CLAWMEETS_AGENT_TEAMS (comma-separated) if no --team flag is given.",
    ),
):
    """Register a new agent with the server (requires admin token).

    Can either provide name and description as arguments, or use --from-card
    to load values from a generated card.json file.

    Examples:
        # Traditional registration
        clawmeets agent register "my-agent" "My description" --token $ADMIN_TOKEN

        # From card.json (simplest)
        clawmeets agent register --from-card ./kb/card.json --token $ADMIN_TOKEN

        # Override name from card
        clawmeets agent register "custom-name" --from-card ./kb/card.json --token $ADMIN_TOKEN

        # Override capabilities from card
        clawmeets agent register --from-card ./kb/card.json --capabilities "new,caps" --token $ADMIN_TOKEN
    """
    # Auto-fill --token and --server from the logged-in user's settings.json
    # when not explicitly provided. Lets skills and scripts avoid re-parsing the
    # config file themselves.
    if not token or not server:
        from clawmeets.cli_lifecycle import get_current_user, get_user_config_path
        data_dir_p = Path(data_dir).expanduser()
        resolved_user = as_user or get_current_user(data_dir_p)
        if resolved_user:
            cfg_path = get_user_config_path(data_dir_p, resolved_user)
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text())
                token = token or cfg.get("user", {}).get("token")
                server = server or cfg.get("server_url")
    server = server or DEFAULT_SERVER
    if not token:
        typer.echo(
            "Error: --token is required. Log in with `clawmeets user login <user> <pass> --save`, "
            "or pass --token explicitly.",
            err=True,
        )
        raise typer.Exit(1)

    # Load from card.json if provided
    caps_list = []
    card_discoverable = True
    if from_card:
        if not from_card.exists():
            typer.echo(f"Error: Card file not found: {from_card}", err=True)
            raise typer.Exit(1)

        try:
            card_data = json.loads(from_card.read_text())
        except json.JSONDecodeError as e:
            typer.echo(f"Error: Invalid JSON in card file: {e}", err=True)
            raise typer.Exit(1)

        # Use card values as defaults (CLI args can override)
        name = name or card_data.get("name")
        description = description or card_data.get("description")
        caps_list = card_data.get("capabilities", [])
        card_discoverable = card_data.get("discoverable_through_registry", True)

    # Parse capabilities from comma-separated string (overrides card)
    if capabilities:
        caps_list = [c.strip() for c in capabilities.split(",") if c.strip()]

    # Determine final discoverable value
    final_discoverable = discoverable if discoverable is not None else card_discoverable

    # Validate required fields
    if not name:
        typer.echo("Error: name is required (provide as argument or via --from-card)", err=True)
        raise typer.Exit(1)
    if not description:
        typer.echo("Error: description is required (provide as argument or via --from-card)", err=True)
        raise typer.Exit(1)

    user_teams = [t.strip() for t in (team or []) if t and t.strip()]
    if not user_teams:
        user_teams = _default_user_teams_from_env()
    register_payload = {
        "name": name,
        "description": description,
        "capabilities": caps_list,
        "discoverable_through_registry": final_discoverable,
    }
    if user_teams:
        register_payload["user_teams"] = user_teams
    with _http(server) as client:
        resp = client.post(
            "/agents/register",
            json=register_payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        result = _ok(resp)
    _print_json(result)

    # Use server-returned name (may be prefixed with username)
    registered_name = result.get("agent_name", name)

    # Determine save path
    if save:
        cred_path = save
    else:
        # Save to {data_dir}/agents/{registered_name}-{agent_id}/credential.json
        agent_id = result["agent_id"]
        agents_dir = Path(data_dir).expanduser() / "agents"
        agent_work_dir = agents_dir / f"{registered_name}-{agent_id}"
        agent_work_dir.mkdir(parents=True, exist_ok=True)
        cred_path = agent_work_dir / "credential.json"

    cred_path.write_text(json.dumps(result, indent=2, default=str))
    typer.echo(f"Credentials saved to {cred_path}")

    # Create card.json with agent metadata (unless using custom save path)
    if not save:
        card = {
            "id": result["agent_id"],
            "name": registered_name,
            "description": result["description"],
            "capabilities": result.get("capabilities", []),
            "status": result["status"],
            "registered_at": result["registered_at"],
            "discoverable_through_registry": result.get("discoverable_through_registry", True),
        }
        if user_teams:
            card["user_teams"] = user_teams
        initial_local_settings = _build_initial_local_settings(llm_provider, llm_model)
        if initial_local_settings:
            card["local_settings"] = initial_local_settings
        card_path = agent_work_dir / "card.json"
        card_path.write_text(json.dumps(card, indent=2, default=str))
        typer.echo(f"Card saved to {card_path}")

    # Optionally link this agent to the logged-in user's settings.json.
    if save_to_settings:
        from clawmeets.cli_lifecycle import add_agent_to_settings, get_current_user
        data_dir_p = Path(data_dir).expanduser()
        target_user = as_user or get_current_user(data_dir_p)
        if not target_user:
            typer.echo(
                "  Warning: --save-to-settings set but no current_user and no --as-user; skipping settings.json update.",
                err=True,
            )
        else:
            # The server-returned name is typically "{username}-{name}"; strip the
            # prefix so settings.json stores the unprefixed name the runtime expects.
            prefix = f"{target_user}-"
            stored_name = registered_name[len(prefix):] if registered_name.startswith(prefix) else registered_name
            entry: dict = {
                "name": stored_name,
                "description": result["description"],
                "capabilities": result.get("capabilities", []),
                "discoverable": result.get("discoverable_through_registry", False),
            }
            if knowledge_dir:
                entry["knowledge_dir"] = knowledge_dir
            try:
                settings_path = add_agent_to_settings(data_dir_p, target_user, entry)
                typer.echo(f"  Linked to user '{target_user}' in {settings_path}.")
            except FileNotFoundError as e:
                typer.echo(f"  Warning: {e}", err=True)


# ---------------------------------------------------------------------------
# tag list / add / remove / set
# ---------------------------------------------------------------------------


def _resolve_user_session(
    data_dir: Path,
    explicit_token: Optional[str],
    explicit_server: Optional[str],
    as_user: Optional[str] = None,
) -> tuple[str, str]:
    """Return (server_url, token), filling in from the saved user session
    when not given explicitly. Mirrors what `agent register` does so the
    tag commands follow the same UX.
    """
    token = explicit_token
    server = explicit_server
    if not token or not server:
        from clawmeets.cli_lifecycle import get_current_user, get_user_config_path
        data_dir_p = Path(data_dir).expanduser()
        resolved_user = as_user or get_current_user(data_dir_p)
        if resolved_user:
            cfg_path = get_user_config_path(data_dir_p, resolved_user)
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text())
                token = token or cfg.get("user", {}).get("token")
                server = server or cfg.get("server_url")
    server = server or DEFAULT_SERVER
    if not token:
        typer.echo(
            "Error: not logged in. Run `clawmeets user login <user> <pass> --save` "
            "or pass --token explicitly.",
            err=True,
        )
        raise typer.Exit(1)
    return _server_url(server), token


def _resolve_agent_id(
    client: httpx.Client, token: str, agent_ref: str
) -> tuple[str, str]:
    """Resolve an agent reference (id or name) to (id, name)."""
    resp = client.get("/agents", headers={"Authorization": f"Bearer {token}"})
    agents = _ok(resp)
    for a in agents:
        if a["id"] == agent_ref:
            return a["id"], a["name"]
    for a in agents:
        if a["id"].startswith(agent_ref):
            return a["id"], a["name"]
    for a in agents:
        if a["name"] == agent_ref:
            return a["id"], a["name"]
    lower = agent_ref.lower()
    for a in agents:
        if a["name"].lower() == lower:
            return a["id"], a["name"]
    typer.echo(f"Error: no agent matches {agent_ref!r}.", err=True)
    raise typer.Exit(1)


def _fetch_owned_agents(client: httpx.Client, token: str) -> list[dict]:
    """Return the agents the current user owns (registered_by == self)."""
    me_resp = client.get("/auth/user/me", headers={"Authorization": f"Bearer {token}"})
    if me_resp.status_code != 200:
        typer.echo(f"Error: could not load current user ({me_resp.text})", err=True)
        raise typer.Exit(1)
    me_id = me_resp.json().get("id")
    agents_resp = client.get("/agents", headers={"Authorization": f"Bearer {token}"})
    agents = _ok(agents_resp)
    return [a for a in agents if a.get("registered_by") == me_id]


@team_app.command("list")
def team_list(
    show_agents: bool = typer.Option(False, "--agents", "-a", help="Also list each team's agents"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """List unique teams across the agents you own (derived from agent state)."""
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        owned = _fetch_owned_agents(client, token)
    by_team: dict[str, list[str]] = {}
    unassigned: list[str] = []
    for agent in owned:
        teams = agent.get("user_teams") or []
        if not teams:
            unassigned.append(agent["name"])
            continue
        for t in teams:
            by_team.setdefault(t, []).append(agent["name"])
    if not by_team and not unassigned:
        typer.echo("No agents owned. Register one with `clawmeets agent register ...`.")
        return
    for team_name in sorted(by_team):
        typer.echo(f"  {team_name}: {len(by_team[team_name])} agent(s)")
        if show_agents:
            for agent_name in sorted(by_team[team_name]):
                typer.echo(f"    - {agent_name}")
    if unassigned:
        typer.echo(f"  (no team): {len(unassigned)} agent(s)")
        if show_agents:
            for agent_name in sorted(unassigned):
                typer.echo(f"    - {agent_name}")


def _put_user_teams(
    client: httpx.Client,
    token: str,
    agent_id: str,
    user_teams: list[str],
) -> None:
    resp = client.put(
        f"/agents/{agent_id}",
        json={"user_teams": user_teams},
        headers={"Authorization": f"Bearer {token}"},
    )
    _ok(resp)


@team_app.command("add")
def team_add(
    agent: str = typer.Argument(..., help="Agent name or id"),
    team_name: str = typer.Argument(..., help="Team to add"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Add a team to an agent (no-op if already present)."""
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        agent_id, agent_name = _resolve_agent_id(client, token, agent)
        current = client.get(
            f"/agents/{agent_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        existing = _ok(current).get("user_teams") or []
        if team_name in existing:
            typer.echo(f"Agent '{agent_name}' already on team '{team_name}'.")
            return
        _put_user_teams(client, token, agent_id, [*existing, team_name])
    typer.echo(f"Added agent '{agent_name}' to team '{team_name}'.")


@team_app.command("remove")
def team_remove(
    agent: str = typer.Argument(..., help="Agent name or id"),
    team_name: str = typer.Argument(..., help="Team to remove"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Remove a team from an agent (no-op if absent)."""
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        agent_id, agent_name = _resolve_agent_id(client, token, agent)
        current = client.get(
            f"/agents/{agent_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        existing = _ok(current).get("user_teams") or []
        if team_name not in existing:
            typer.echo(f"Agent '{agent_name}' is not on team '{team_name}'.")
            return
        _put_user_teams(
            client,
            token,
            agent_id,
            [t for t in existing if t != team_name],
        )
    typer.echo(f"Removed agent '{agent_name}' from team '{team_name}'.")


@team_app.command("set")
def team_set(
    agent: str = typer.Argument(..., help="Agent name or id"),
    team: list[str] = typer.Option(
        None, "--team",
        help="Team value (repeatable). Pass with no --team flags to clear all teams.",
    ),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Replace an agent's team list with the given --team values (or clear)."""
    server_url, token = _resolve_user_session(data_dir, token, server)
    new_teams = [t.strip() for t in (team or []) if t and t.strip()]
    with _http(server_url) as client:
        agent_id, agent_name = _resolve_agent_id(client, token, agent)
        _put_user_teams(client, token, agent_id, new_teams)
    if new_teams:
        typer.echo(f"Set teams on '{agent_name}': {', '.join(new_teams)}")
    else:
        typer.echo(f"Cleared teams on '{agent_name}'.")


# ---------------------------------------------------------------------------
# team create / delete / invite / disinvite / add-sample-requests
# ---------------------------------------------------------------------------


def _extract_sample_requests(template: dict) -> list[dict]:
    """Pull the sample_requests list out of a setup.json template, validating
    shape. Returns a list of {title, request, coordinator_hint?} dicts."""
    raw = template.get("sample_requests") or []
    out: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        title = (entry.get("title") or "").strip()
        request = (entry.get("request") or "").strip()
        if not title or not request:
            continue
        item = {"title": title, "request": request}
        hint = entry.get("coordinator_hint")
        if hint:
            item["coordinator_hint"] = str(hint).strip()
        out.append(item)
    return out


@team_app.command("create")
def team_create(
    name: str = typer.Argument(..., help="Team name (matches agents' user_teams entry)"),
    from_url: Optional[str] = typer.Option(
        None, "--from-url",
        help="Import sample requests from a setup.json URL / path (same format as `init --from-url`)",
    ),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Create (or upsert) a team record on the server. If `--from-url` is
    given, the template's sample_requests are attached to the team. Existing
    sample_requests are preserved — re-running is idempotent (dedup by title).
    """
    server_url, token = _resolve_user_session(data_dir, token, server)

    sample_requests: list[dict] = []
    if from_url:
        from clawmeets.cli_init import _fetch_setup_template
        template = _fetch_setup_template(from_url)
        sample_requests = _extract_sample_requests(template)

    with _http(server_url) as client:
        payload: dict = {"name": name}
        if sample_requests:
            payload["sample_requests"] = sample_requests
        resp = client.post(
            "/teams",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        team = _ok(resp)
    typer.echo(
        f"Team '{team['name']}' ready "
        f"({len(team['sample_requests'])} sample request(s), "
        f"{team['member_count']} member(s))."
    )


@team_app.command("delete")
def team_delete(
    name: str = typer.Argument(..., help="Team name"),
    remove_from_agents: bool = typer.Option(
        False, "--remove-from-agents",
        help="Also strip the team label from every agent that carries it",
    ),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Delete a team's metadata. With `--remove-from-agents`, also strip the
    label from every owned agent that carried it.
    """
    server_url, token = _resolve_user_session(data_dir, token, server)
    params = {"remove_from_agents": "true" if remove_from_agents else "false"}
    with _http(server_url) as client:
        resp = client.delete(
            f"/teams/{name}",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        result = _ok(resp)
    extra = ""
    if remove_from_agents:
        extra = f" (removed label from {result.get('labels_removed_from_agents', 0)} agent(s))"
    typer.echo(f"Deleted team '{name}'.{extra}")


@team_app.command("invite")
def team_invite(
    team_name: str = typer.Argument(..., help="Team name"),
    agents: list[str] = typer.Argument(..., help="Agent name(s) or id(s) to invite"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Invite one or more agents to a team (adds the team label to each agent's
    user_teams). No-op for agents already on the team.
    """
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        for agent in agents:
            agent_id, agent_name = _resolve_agent_id(client, token, agent)
            resp = client.post(
                f"/teams/{team_name}/members",
                json={"agent_id": agent_id},
                headers={"Authorization": f"Bearer {token}"},
            )
            _ok(resp)
            typer.echo(f"Invited '{agent_name}' to team '{team_name}'.")


@team_app.command("disinvite")
def team_disinvite(
    team_name: str = typer.Argument(..., help="Team name"),
    agents: list[str] = typer.Argument(..., help="Agent name(s) or id(s) to disinvite"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Remove one or more agents from a team."""
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        for agent in agents:
            agent_id, agent_name = _resolve_agent_id(client, token, agent)
            resp = client.delete(
                f"/teams/{team_name}/members/{agent_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            _ok(resp)
            typer.echo(f"Removed '{agent_name}' from team '{team_name}'.")


@team_app.command("add-sample-requests")
def team_add_sample_requests(
    team_name: str = typer.Argument(..., help="Team name"),
    from_url: Optional[str] = typer.Option(
        None, "--from-url",
        help="Import sample requests from a setup.json URL / path",
    ),
    title: Optional[str] = typer.Option(
        None, "--title", help="Title for a single inline sample request"
    ),
    request: Optional[str] = typer.Option(
        None, "--request", help="Body for a single inline sample request"
    ),
    coordinator_hint: Optional[str] = typer.Option(
        None, "--coordinator-hint",
        help="Optional coordinator hint for the inline sample request",
    ),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Add sample requests to a team, either inline (`--title` + `--request`)
    or imported from a setup.json template (`--from-url`). The two modes can
    be combined. Existing samples with the same title are replaced.
    """
    if not from_url and not (title and request):
        typer.echo(
            "Error: pass --from-url, or both --title and --request "
            "(or both --from-url and an inline sample).",
            err=True,
        )
        raise typer.Exit(1)

    server_url, token = _resolve_user_session(data_dir, token, server)

    to_add: list[dict] = []
    if from_url:
        from clawmeets.cli_init import _fetch_setup_template
        template = _fetch_setup_template(from_url)
        to_add.extend(_extract_sample_requests(template))
    if title and request:
        item = {"title": title.strip(), "request": request.strip()}
        if coordinator_hint:
            item["coordinator_hint"] = coordinator_hint.strip()
        to_add.append(item)

    if not to_add:
        typer.echo(f"No sample requests to add to team '{team_name}'.")
        return

    with _http(server_url) as client:
        # Ensure the team record exists first (upsert).
        _ok(client.post(
            "/teams",
            json={"name": team_name},
            headers={"Authorization": f"Bearer {token}"},
        ))
        for sample in to_add:
            resp = client.post(
                f"/teams/{team_name}/sample-requests",
                json=sample,
                headers={"Authorization": f"Bearer {token}"},
            )
            _ok(resp)
        typer.echo(
            f"Added {len(to_add)} sample request(s) to team '{team_name}'."
        )


# ---------------------------------------------------------------------------
# agent list
# ---------------------------------------------------------------------------

@agent_app.command("list")
def agent_list(
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    full: bool = typer.Option(False, "--full", "-f", help="Show full IDs"),
):
    """List all registered agents."""
    with _http(server) as client:
        resp = client.get("/agents")
        agents = _ok(resp)
    if not agents:
        typer.echo("No agents registered.")
        return
    for a in agents:
        status = a.get("status", "?")
        aid = a['id'] if full else f"{a['id'][:8]}…"
        typer.echo(f"  [{status:7s}] {a['name']:20s}  id={aid}  {a['description']}")


# ---------------------------------------------------------------------------
# agent run  (the main runner loop)
# ---------------------------------------------------------------------------

@agent_app.command("run")
def agent_run(
    credentials: Optional[Path] = typer.Argument(None, help="JSON credentials file (optional if credential.json exists in --agent-dir)"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    agent_dir: Path = typer.Option(..., "--agent-dir", help="Agent's working directory (contains credential.json, sandbox/, projects/); e.g. ~/.clawmeets/agents/my-agent-{id}/"),
    working_dir: Optional[Path] = typer.Option(None, "--working-dir", "-w", help="Sandbox directory for Claude (default: agent-dir/sandbox)"),
    knowledge_dir: Optional[Path] = typer.Option(None, "--knowledge-dir", "-k", help="Knowledge base directory (passed as --add-dir to Claude)"),
    claude_plugin_dir: Optional[list[Path]] = typer.Option(None, "--claude-plugin-dir", help="Claude plugin directory (passed as --plugin-dir to Claude CLI, repeatable)"),
    chrome: bool = typer.Option(False, "--chrome", help="Enable Chrome browser integration via Claude Code --chrome flag"),
    user_config: Optional[Path] = typer.Option(None, "--user-config", help="Path to the owning user's settings.json; used on self-destruct when the server reports this agent was deleted."),
    settings_name: Optional[str] = typer.Option(None, "--settings-name", help="Short agent name as it appears in settings.json agents[].name; paired with --user-config for self-destruct cleanup."),
    log_level: str = typer.Option("info"),
):
    """
    Start the agent runner.
    Connects via WebSocket and dispatches all incoming envelopes via the
    control loop. Keeps running until interrupted (Ctrl-C).

    The agent_dir contains:
    - credential.json                                    (agent credentials)
    - card.json                                          (agent metadata)
    - projects/{project_name}-{project_id}/              (synced files, read-only)
    - sandbox/                                           (Claude's working directory)
    - metadata/projects/{project_name}-{project_id}/     (per-project metadata)
        - stdout.log, stderr.log                         (runner logs)
        - cli-stdout.log, cli-stderr.log                 (Claude CLI logs)
        - cost.json                                      (usage tracking)

    When --working-dir is specified, Claude runs in that directory instead of
    agent-dir/sandbox, with project data accessible via --add-dir.

    When --knowledge-dir is specified, the directory is passed as an additional
    --add-dir to Claude, enabling access to that knowledge base.

    When --claude-plugin-dir is specified, the directory is passed as --plugin-dir
    to Claude CLI, enabling access to Claude Code plugins/skills. Can be repeated
    for multiple plugin directories.
    """
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Find credentials file
    if credentials:
        creds_path = credentials
    else:
        creds_path = Path(agent_dir) / "credential.json"
        if not creds_path.exists():
            typer.echo(f"Error: No credentials file provided and {creds_path} not found.", err=True)
            raise typer.Exit(1)

    creds_data = json.loads(creds_path.read_text())
    agent_id = creds_data["agent_id"]
    token = creds_data["token"]
    agent_name = creds_data.get("agent_name") or creds_data.get("card", {}).get("name")
    if not agent_name:
        typer.echo("Error: agent_name missing from credentials", err=True)
        raise typer.Exit(1)

    typer.echo(f"Starting runner for agent '{agent_name}' ({agent_id[:8]}…)")
    typer.echo(f"Server: {server}  |  Agent dir: {agent_dir}")
    if working_dir:
        typer.echo(f"Working dir: {working_dir}")
    if knowledge_dir:
        typer.echo(f"Knowledge dir: {knowledge_dir}")
    if claude_plugin_dir:
        typer.echo(f"Claude plugin dirs: {claude_plugin_dir}")

    asyncio.run(_runner_loop(
        agent_name, agent_id, token, server, Path(agent_dir),
        working_dir, knowledge_dir, claude_plugin_dir or [],
        use_chrome=chrome,
        user_config=user_config,
        settings_name=settings_name,
    ))


async def _ws_heartbeat_task(ws, agent_id: str) -> None:
    """Send periodic heartbeats to keep connection alive."""
    while True:
        await asyncio.sleep(30)
        env = ControlEnvelope(type="heartbeat")
        await ws.send(env.model_dump_json(by_alias=True))


def _create_dispatch_callback() -> callable:
    """Create task exception handler for dispatch tasks."""
    def _handle_task_exception(task: asyncio.Task) -> None:
        try:
            exc = task.exception()
            if exc:
                logging.error(f"Dispatch task failed: {exc}", exc_info=exc)
        except asyncio.CancelledError:
            pass
    return _handle_task_exception


# ---------------------------------------------------------------------------
# user login / create / list / listen
# ---------------------------------------------------------------------------

@user_app.command("login")
def user_login(
    username: str = typer.Argument(..., help="Username"),
    password: str = typer.Argument(..., help="Password"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    save: bool = typer.Option(
        False, "--save",
        help="Persist the session: write token into settings.json and set current_user.",
    ),
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR, "--data-dir",
        help="Root data directory (only used with --save).",
    ),
):
    """Login as a user. Prints the JWT token to stdout by default.

    With --save, writes the token to ~/.clawmeets/config/{username}/settings.json
    and marks the user as current_user, so other `clawmeets` commands can find
    them without re-authenticating. Nothing is printed in --save mode beyond a
    confirmation line, so shell pipelines that capture the token should omit --save.
    """
    with _http(server) as client:
        resp = client.post("/auth/login", json={"username": username, "password": password})
        result = _ok(resp)
    token = result["token"]
    if save:
        from clawmeets.cli_lifecycle import save_user_session
        path = save_user_session(Path(data_dir).expanduser(), username, _server_url(server), token)
        typer.echo(f"Logged in as {username}. Session saved to {path}.")
    else:
        typer.echo(token)


@user_app.command("logout")
def user_logout(
    username: Optional[str] = typer.Option(
        None, "--user", "-u",
        help="Username (defaults to current_user).",
    ),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
    clear_current: bool = typer.Option(
        False, "--clear-current-user",
        help="Also clear the ~/.clawmeets/config/current_user pointer.",
    ),
):
    """Clear the saved JWT token from a user's settings.json.

    Does NOT stop agents or delete any data — running agents keep their own
    per-agent tokens and stay online until `clawmeets stop` is called.
    """
    from clawmeets.cli_lifecycle import clear_user_token, get_current_user
    data_dir_p = Path(data_dir).expanduser()
    if username is None:
        username = get_current_user(data_dir_p)
    if not username:
        typer.echo("Not logged in (no current_user).", err=True)
        raise typer.Exit(1)
    path = clear_user_token(data_dir_p, username)
    typer.echo(f"Logged out user '{username}' ({path}).")
    if clear_current:
        (data_dir_p / "config" / "current_user").unlink(missing_ok=True)
        typer.echo("Cleared current_user.")


@user_app.command("register")
def user_register(
    username: str = typer.Argument(..., help="Username"),
    password: str = typer.Argument(..., help="Password"),
    email: str = typer.Argument(..., help="Email address"),
    invitation_code: str = typer.Option(..., "--invitation-code", "-i", help="Invitation code (required)"),
    agree_tos: bool = typer.Option(False, "--agree-tos", help="Agree to Terms of Service and Privacy Policy"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Root data directory (assistant saved to {data_dir}/agents/)"),
    llm_provider: Optional[str] = typer.Option(
        None, "--llm-provider",
        help="LLM backend for this user's assistant: 'claude' (default), 'openai', or 'gemini'.",
    ),
    llm_model: Optional[str] = typer.Option(
        None, "--llm-model",
        help="Provider-specific model name (e.g. 'o3' for Codex, 'gemini-2.5-pro' for Gemini).",
    ),
):
    """Self-register a new user account (requires invitation code).

    After registration, check your email to verify your account.
    You cannot log in until your email is verified.

    Example:
        clawmeets user register alice mypassword alice@example.com --invitation-code ABC123
    """
    typer.echo("By registering, you agree to the Terms of Service (https://clawmeets.ai/tos)")
    typer.echo("and Privacy Policy (https://clawmeets.ai/privacy).")
    if not agree_tos:
        typer.confirm("Do you agree to the Terms of Service and Privacy Policy?", abort=True)

    with _http(server) as client:
        resp = client.post(
            "/auth/register",
            json={
                "username": username,
                "password": password,
                "email": email,
                "invitation_code": invitation_code,
            },
        )
        result = _ok(resp)

        typer.echo(f"Registered user: {result['username']}")
        typer.echo(f"{result['message']}")

        # Save assistant credentials locally
        agent_id = result["assistant_agent_id"]
        agent_name = result["assistant_agent_name"]
        agent_token = result["assistant_token"]
        user_id = result["user_id"]

        assistant_creds = {
            "agent_id": agent_id,
            "token": agent_token,
            "agent_name": agent_name,
        }

        agents_dir = Path(data_dir).expanduser() / "agents"
        assistant_dir = agents_dir / f"{agent_name}-{agent_id}"
        assistant_dir.mkdir(parents=True, exist_ok=True)

        cred_path = assistant_dir / "credential.json"
        cred_path.write_text(json.dumps(assistant_creds, indent=2, default=str))
        typer.echo(f"Assistant credentials saved to {cred_path}")

        # Create card.json for assistant agent
        card = {
            "id": agent_id,
            "name": agent_name,
            "description": f"Assistant agent for user {username}",
            "capabilities": [],
            "status": "online",
            "registered_at": datetime.now(UTC).isoformat(),
            "discoverable_through_registry": False,
            "registered_by": user_id,
        }
        initial_local_settings = _build_initial_local_settings(llm_provider, llm_model)
        if initial_local_settings:
            card["local_settings"] = initial_local_settings
        card_path = assistant_dir / "card.json"
        card_path.write_text(json.dumps(card, indent=2, default=str))
        typer.echo(f"Card saved to {card_path}")


@user_app.command("create")
def user_create(
    username: str = typer.Argument(..., help="Username"),
    password: str = typer.Argument(..., help="Password"),
    role: str = typer.Option("user", "--role", "-r", help="User role (admin or user)"),
    email: Optional[str] = typer.Option(None, "--email", "-e", help="User email address"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Admin JWT token (required)"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Root data directory (assistant saved to {data_dir}/agents/)"),
    llm_provider: Optional[str] = typer.Option(
        None, "--llm-provider",
        help="LLM backend for this user's assistant: 'claude' (default), 'openai', or 'gemini'.",
    ),
    llm_model: Optional[str] = typer.Option(
        None, "--llm-model",
        help="Provider-specific model name (e.g. 'o3' for Codex, 'gemini-2.5-pro' for Gemini).",
    ),
):
    """Create a new user with assistant agent (requires admin token).

    Admin-created users are pre-verified (no email verification needed).
    """
    if not token:
        typer.echo("Error: --token is required. Get admin token with: user login admin <password>", err=True)
        raise typer.Exit(1)

    with _http(server) as client:
        # Create user (response includes assistant credentials)
        payload = {"username": username, "password": password, "role": role}
        if email:
            payload["email"] = email
        resp = client.post(
            "/users",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        result = _ok(resp)
        typer.echo(f"Created user: {username}")

        # Extract assistant info from flat response
        agent_id = result["assistant_agent_id"]
        agent_name = result["assistant_agent_name"]
        agent_token = result["assistant_token"]
        user_id = result["user_id"]

        # Build credentials structure for saving
        assistant_creds = {
            "agent_id": agent_id,
            "token": agent_token,
            "agent_name": agent_name,
        }

        # Save assistant credentials to {data_dir}/agents/
        agents_dir = Path(data_dir).expanduser() / "agents"
        assistant_dir = agents_dir / f"{agent_name}-{agent_id}"
        assistant_dir.mkdir(parents=True, exist_ok=True)

        cred_path = assistant_dir / "credential.json"
        cred_path.write_text(json.dumps(assistant_creds, indent=2, default=str))
        typer.echo(f"Assistant credentials saved to {cred_path}")

        # Create card.json for assistant agent
        card = {
            "id": agent_id,
            "name": agent_name,
            "description": f"Assistant agent for user {username}",
            "capabilities": [],
            "status": "online",
            "registered_at": result["user_created_at"],
            "discoverable_through_registry": False,
            "registered_by": user_id,
        }
        initial_local_settings = _build_initial_local_settings(llm_provider, llm_model)
        if initial_local_settings:
            card["local_settings"] = initial_local_settings
        card_path = assistant_dir / "card.json"
        card_path.write_text(json.dumps(card, indent=2, default=str))
        typer.echo(f"Card saved to {card_path}")


@user_app.command("list")
def user_list(
    token: str = typer.Option(..., "--token", "-t", help="Admin JWT token"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
):
    """List all users (requires admin token)."""
    with _http(server) as client:
        resp = client.get("/users", headers={"Authorization": f"Bearer {token}"})
        users = _ok(resp)
    if not users:
        typer.echo("No users.")
        return
    for u in users:
        typer.echo(f"  [{u['role']:5s}] {u['username']:20s}  id={u['id'][:8]}…")


@user_app.command("listen")
def user_listen(
    username: str = typer.Argument(..., help="Username"),
    password: str = typer.Argument(..., help="Password"),
    script: Optional[Path] = typer.Argument(None, help="Notification script path (optional with --console)"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Root data directory (listener data at {data_dir}/users/)"),
    timeout: float = typer.Option(30.0, "--timeout", help="Script execution timeout (seconds)"),
    fail_fast: bool = typer.Option(False, "--fail-fast", help="Exit on script failure"),
    log_level: str = typer.Option("info", "--log-level"),
    console: bool = typer.Option(False, "--console", "-c", help="Enable console output mode"),
    no_colors: bool = typer.Option(False, "--no-colors", help="Disable ANSI colors in console output"),
):
    """
    Listen for notifications from the user's assistant.

    Authenticates as the user, connects via WebSocket, and either:
    - Prints changelog events to console (with --console)
    - Pipes coordinator messages to a script via stdin as JSON
    - Both (when using --console with a script)

    Examples:
        # Console output only
        clawmeets user listen alice mypassword --console

        # Script notifications only
        clawmeets user listen alice mypassword ./scripts/notify.py

        # Both console and script
        clawmeets user listen alice mypassword ./scripts/notify.py --console

        # Console without colors
        clawmeets user listen alice mypassword --console --no-colors

    The notification script receives JSON on stdin:
        {
            "event": "message",
            "project_id": "...",
            "project_name": "...",
            "chatroom_name": "...",
            "user_id": "...",
            "username": "...",
            "message": {
                "id": "...",
                "ts": "2024-03-19T10:30:00Z",
                "from_participant_id": "...",
                "from_participant_name": "...",
                "content": "..."
            }
        }
    """
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Validate: require either --console or script (or both)
    if not console and script is None:
        typer.echo("Error: Either --console or a script path is required", err=True)
        raise typer.Exit(1)

    # Validate script if provided
    if script is not None:
        if not script.exists():
            typer.echo(f"Error: Script not found: {script}", err=True)
            raise typer.Exit(1)
        if not os.access(script, os.X_OK):
            typer.echo(f"Error: Script is not executable: {script}", err=True)
            raise typer.Exit(1)

    # Authenticate user
    with _http(server) as client:
        resp = client.post("/auth/login", json={"username": username, "password": password})
        try:
            result = _ok(resp)
        except SystemExit:
            typer.echo("Error: Invalid username or password", err=True)
            raise typer.Exit(1)

    token = result["token"]
    user_id = result["user"]["id"]
    assistant_id = result.get("assistant_agent_id")

    if not assistant_id:
        typer.echo("Error: User has no linked assistant agent", err=True)
        raise typer.Exit(1)

    users_dir = Path(data_dir).expanduser() / "users"
    typer.echo(f"Authenticated as {username} (user_id={user_id[:8]}...)")
    typer.echo(f"Server: {server}  |  User dir: {users_dir}")
    if console:
        typer.echo(f"Console output: enabled (colors={'off' if no_colors else 'on'})")
    if script:
        typer.echo(f"Notification script: {script}")

    # Create extra subscribers for console output
    extra_subscribers = []
    if console:
        console_config = ConsoleConfig(colors=not no_colors)
        extra_subscribers.append(ConsoleOutputSubscriber(config=console_config))

    asyncio.run(_user_listen_loop(
        username=username,
        user_id=user_id,
        assistant_id=assistant_id,
        token=token,
        server_http=server,
        user_base_dir=users_dir,
        script=script,
        timeout=timeout,
        fail_fast=fail_fast,
        extra_subscribers=extra_subscribers,
    ))


async def _user_listen_loop(
    username: str,
    user_id: str,
    assistant_id: str,
    token: str,
    server_http: str,
    user_base_dir: Path,
    script: Optional[Path],
    timeout: float,
    fail_fast: bool,
    extra_subscribers: Optional[list] = None,
) -> None:
    """Run the reactive control loop for a user listening for notifications."""
    # Create user-specific directory under base (client-side storage)
    user_dir = user_base_dir / f"{username}-{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)

    # Create HTTP client with user auth
    http_client = httpx.AsyncClient(
        base_url=server_http,
        headers={
            "Authorization": f"Bearer {token}",
            "X-User-ID": user_id,
        },
        timeout=30.0,
    )

    # Create ClawMeetsClient wrapper
    client = ClawMeetsClient(http_client=http_client, server_url=server_http)

    # Create ModelContext for the user with client (self-contained, not shared with server)
    model_ctx = ModelContext(base_dir=user_dir, client=client, notification_center=NotificationCenter())

    # Create User participant with notification config (if script provided)
    participant = User(id=user_id, model_ctx=model_ctx)
    if script is not None:
        participant.set_notification_config(NotificationConfig(
            script_path=str(script.absolute()),
            timeout=timeout,
            fail_fast=fail_fast,
        ))

    # Build reactive control loop with extra subscribers
    loop_obj = ReactiveControlLoop(
        participant=participant,
        client=client,
        model_ctx=model_ctx,
        extra_subscribers=extra_subscribers,
    )

    # Start the loop
    await loop_obj.start()

    # Connect via WebSocket using user endpoint
    ws_url = server_http.replace("https://", "wss://").replace("http://", "ws://")
    ws_connect_url = f"{ws_url}/ws/user/{user_id}?token={token}"

    handle_exception = _create_dispatch_callback()
    reconnect_delay = 2.0

    while True:
        try:
            async with websockets.connect(ws_connect_url) as ws:
                logging.getLogger("clawmeets").info(f"WebSocket connected to {ws_url}")

                # Send auth message
                await ws.send(json.dumps({"token": token}))

                reconnect_delay = 2.0  # reset on success

                # HTTP-based catch-up on connect
                await loop_obj.catch_up()

                hb_task = asyncio.create_task(_ws_heartbeat_task(ws, user_id))

                try:
                    async for raw in ws:
                        try:
                            env = ControlEnvelope.model_validate_json(raw)
                            # Log project_id for CHANGELOG_UPDATE (typed payload guaranteed by validator)
                            proj_id = env.payload.project_id if env.type == ControlMessageType.CHANGELOG_UPDATE else None
                            logging.getLogger("clawmeets").debug(
                                f"[ws-recv] user={user_id} type={env.type} "
                                f"proj={proj_id}"
                            )
                            task = asyncio.create_task(loop_obj.dispatch(env))
                            task.add_done_callback(handle_exception)
                        except Exception as e:
                            logging.warning(f"Bad envelope: {e}")
                finally:
                    hb_task.cancel()

        except (websockets.WebSocketException, OSError) as e:
            # Covers ConnectionClosed, InvalidStatus (HTTP 4xx/5xx on WS
            # upgrade when the server is down), InvalidHandshake, and
            # transport/DNS errors. All transient — reconnect rather than
            # crash the runner.
            logging.warning(f"WebSocket disconnected: {e}. Reconnecting in {reconnect_delay}s…")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)
        except asyncio.CancelledError:
            await loop_obj.stop()
            await http_client.aclose()
            break


def _self_destruct(
    agent_dir: Path,
    user_config: Optional[Path],
    settings_name: Optional[str],
) -> None:
    """Handle the server's 'participant not found' signal: rename the local
    agent directory to DELETED-* and drop the corresponding entry from the
    owning user's settings.json (when both pieces of context were provided).

    Best-effort — logs and swallows filesystem errors so a runner shutting
    down in response to deletion never stalls on cleanup.
    """
    logger = logging.getLogger("clawmeets")
    if user_config and settings_name and user_config.exists():
        try:
            cfg = json.loads(user_config.read_text())
            before = len(cfg.get("agents", []))
            cfg["agents"] = [
                a for a in cfg.get("agents", [])
                if a.get("name") != settings_name
            ]
            if len(cfg["agents"]) != before:
                user_config.write_text(json.dumps(cfg, indent=2))
                logger.info(f"Removed '{settings_name}' from {user_config}")
        except Exception as e:
            logger.warning(f"Could not update {user_config}: {e}")
    target = agent_dir.parent / f"DELETED-{agent_dir.name}"
    if agent_dir.exists() and not target.exists():
        try:
            agent_dir.rename(target)
            logger.info(f"Renamed {agent_dir.name} -> {target.name}")
        except OSError as e:
            logger.warning(f"Could not rename {agent_dir}: {e}")


async def _runner_loop(
    agent_name: str,
    agent_id: str,
    token: str,
    server_http: str,
    agent_dir: Path,
    working_dir: Optional[Path] = None,
    knowledge_dir: Optional[Path] = None,
    claude_plugin_dirs: Optional[list[Path]] = None,
    use_chrome: bool = False,
    user_config: Optional[Path] = None,
    settings_name: Optional[str] = None,
) -> None:
    """Run the reactive control loop for an agent."""
    agent_dir.mkdir(parents=True, exist_ok=True)

    # Read card.json. Since the Agent/Assistant merge, every runner instantiates
    # as Agent regardless of discoverability — coordinator behavior is decided
    # per-project at runtime via agent.is_coordinator_for(project).
    card_path = agent_dir / "card.json"
    if not card_path.exists():
        typer.echo(f"Error: card.json not found at {card_path}", err=True)
        raise typer.Exit(1)
    card_data = json.loads(card_path.read_text())

    # Read local_settings from card.json (primary source, not synced with server).
    # CLI flags (--knowledge-dir, --chrome) serve as overrides for backward compat.
    local_settings = card_data.get("local_settings", {}) or {}

    # Migration: older cards stored llm_provider/llm_model at the top level.
    # Move them into local_settings on next save so new clients find them
    # in the expected place. Keep the top-level copy readable for this run
    # (see `card_llm_*` below) in case the save fails.
    migrated = False
    if "llm_provider" in card_data and "llm_provider" not in local_settings:
        local_settings["llm_provider"] = card_data["llm_provider"]
        migrated = True
    if "llm_model" in card_data and "llm_model" not in local_settings:
        local_settings["llm_model"] = card_data["llm_model"]
        migrated = True
    if migrated:
        card_data.pop("llm_provider", None)
        card_data.pop("llm_model", None)
        card_data["local_settings"] = local_settings
        card_path.write_text(json.dumps(card_data, indent=2, default=str))
        typer.echo("Migrated llm_provider/llm_model into local_settings in card.json")

    effective_knowledge_dir = knowledge_dir or local_settings.get("knowledge_dir", "")
    effective_use_chrome = use_chrome or local_settings.get("use_chrome", False)

    # Resolve relative knowledge_dir paths (e.g. "./owner") against the user's
    # init-time config dir (~/.clawmeets/config/<username>/), where
    # `clawmeets init` wrote CLAUDE.md. Absolute and ~-prefixed paths pass
    # through unchanged. Falls back to legacy CWD-relative behavior only when
    # --user-config is absent (shouldn't happen under cli_lifecycle).
    user_config_dir: Optional[Path] = user_config.parent if user_config else None

    # Set up skill manager (downloads skills from server on startup)
    skill_manager = SkillManager(agent_dir)

    # Set up personal-skill manager (agent-local, never synced; populated by
    # the agent itself during scheduled reflection's Promote/Correct modes).
    personal_skill_manager = PersonalSkillManager(agent_dir)

    # Set up MCP manager (downloads manifests from server on startup;
    # renders .mcp.json into each Claude invocation's cwd)
    mcp_manager = McpManager(agent_dir)

    # Pick LLM provider from card.json local_settings ("claude" default;
    # "openai" and "gemini" also supported). Skill-hub and personal-skill-hub
    # plugin dirs are appended to explicit plugin dirs for Claude; other
    # providers ignore plugin dirs. Action schema is selected per invocation
    # by the caller (Agent), based on whether this runner is coordinator of
    # the project the message is in.
    all_plugin_dirs = list(claude_plugin_dirs or []) + [
        skill_manager.plugin_dir,
        personal_skill_manager.plugin_dir,
    ]
    llm_provider_name = (local_settings.get("llm_provider") or "claude").lower()
    llm_model = local_settings.get("llm_model") or None
    cli: LLMProvider
    if llm_provider_name == "openai":
        CodexCLI.verify_cli()
        cli = CodexCLI(model=llm_model)
        typer.echo(f"LLM provider: openai (model={llm_model or 'default'})")
    elif llm_provider_name == "gemini":
        GeminiCLI.verify_cli()
        cli = GeminiCLI(model=llm_model)
        typer.echo(f"LLM provider: gemini (model={llm_model or 'default'})")
    elif llm_provider_name == "claude":
        ClaudeCLI.verify_cli()
        cli = ClaudeCLI(
            claude_plugin_dirs=all_plugin_dirs,
            use_chrome=effective_use_chrome,
            mcp_manager=mcp_manager,
        )
        typer.echo(
            f"LLM provider: claude"
            + (f" (model={llm_model})" if llm_model else "")
        )
    else:
        typer.echo(
            f"Error: unknown llm_provider '{llm_provider_name}' in "
            f"card.json local_settings (expected one of {_VALID_LLM_PROVIDERS})",
            err=True,
        )
        raise typer.Exit(1)

    # Build knowledge_dirs list (e.g., knowledge bases)
    knowledge_dirs_list: list[Path] = []
    resolved = resolve_local_knowledge_dir(str(effective_knowledge_dir), user_config_dir) if effective_knowledge_dir else None
    if resolved is not None:
        knowledge_dirs_list.append(resolved)

    # Create HTTP client with auth
    http_client = httpx.AsyncClient(
        base_url=server_http,
        headers={
            "Authorization": f"Bearer {token}",
            "X-Agent-ID": agent_id,
        },
        timeout=30.0,
    )

    # Create ClawMeetsClient wrapper
    client = ClawMeetsClient(http_client=http_client, server_url=server_http)

    # Create ModelContext for the agent with all runtime dependencies
    notification_center = NotificationCenter()
    model_ctx = ModelContext(
        base_dir=agent_dir,
        cli=cli,
        knowledge_dirs=knowledge_dirs_list,
        client=client,
        claude_plugin_dirs=all_plugin_dirs,
        notification_center=notification_center,
    )

    participant = Agent(id=agent_id, model_ctx=model_ctx)

    # Build reactive control loop
    loop_obj = ReactiveControlLoop(
        participant=participant,
        client=client,
        model_ctx=model_ctx,
        extra_subscribers=[],
        skill_manager=skill_manager,
        mcp_manager=mcp_manager,
        user_config_dir=user_config_dir,
    )

    # Start the loop
    await loop_obj.start()

    # Convert http:// → ws://
    ws_url = server_http.replace("https://", "wss://").replace("http://", "ws://")
    ws_connect_url = f"{ws_url}/ws/{agent_id}?token={token}"

    handle_exception = _create_dispatch_callback()
    reconnect_delay = 2.0

    while True:
        close_code: Optional[int] = None
        try:
            async with websockets.connect(ws_connect_url) as ws:
                logging.getLogger("clawmeets").info(f"WebSocket connected to {ws_url}")

                # Send auth message (server expects this as first message)
                await ws.send(json.dumps({"token": token}))

                reconnect_delay = 2.0  # reset on success

                # Sync installed skills from server (catch-up on connect/reconnect)
                await skill_manager.sync_from_server(client, agent_id)

                # Sync installed MCP servers from server (catch-up on connect/reconnect)
                await mcp_manager.sync_from_server(client, agent_id)

                # Kick off auto-OAuth for any MCP server that landed via
                # sync_from_server above but doesn't yet have a token (e.g.
                # user clicked Install while this runner was offline).
                loop_obj.auto_auth_pending_mcps()

                # HTTP-based catch-up on connect
                await loop_obj.catch_up()

                hb_task = asyncio.create_task(_ws_heartbeat_task(ws, agent_id))

                try:
                    async for raw in ws:
                        try:
                            env = ControlEnvelope.model_validate_json(raw)
                            proj_id = env.payload.project_id if env.type == ControlMessageType.CHANGELOG_UPDATE else None
                            logging.getLogger("clawmeets").debug(
                                f"[ws-recv] agent={agent_id} type={env.type} "
                                f"proj={proj_id}"
                            )
                            task = asyncio.create_task(loop_obj.dispatch(env))
                            task.add_done_callback(handle_exception)
                        except Exception as e:
                            logging.warning(f"Bad envelope: {e}")
                finally:
                    hb_task.cancel()
                close_code = ws.close_code

        except websockets.ConnectionClosed as e:
            close_code = e.code
            logging.warning(f"WebSocket disconnected (code={e.code}, reason={e.reason!r})")
        except (websockets.WebSocketException, OSError) as e:
            # Covers InvalidStatus (HTTP 4xx/5xx on WS upgrade — happens when
            # the server is down and ngrok/proxy returns an error),
            # InvalidHandshake, and transport/DNS errors. All transient —
            # reconnect rather than crash the runner.
            logging.warning(f"WebSocket connect/transport error: {e}. Reconnecting in {reconnect_delay}s…")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)
            continue
        except asyncio.CancelledError:
            await loop_obj.stop()
            await http_client.aclose()
            break

        # 4004 = server told us this participant no longer exists. Self-
        # destruct rather than reconnect-loop; a fresh registration of the
        # same name is a different agent with a different id.
        if close_code == 4004:
            logging.getLogger("clawmeets").warning(
                f"Agent '{agent_name}' ({agent_id[:8]}…) was deleted server-side; cleaning up local state and exiting."
            )
            _self_destruct(agent_dir, user_config, settings_name)
            await loop_obj.stop()
            await http_client.aclose()
            return

        logging.warning(f"Reconnecting in {reconnect_delay}s…")
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, 60)


# ---------------------------------------------------------------------------
# dm (direct message) commands
# ---------------------------------------------------------------------------

def _find_dm_project(client: httpx.Client, username: str) -> Optional[dict]:
    """Find the DM project for a user."""
    dm_project_name = f"DM-{username}"
    resp = client.get("/projects")
    if resp.status_code != 200:
        return None
    projects = resp.json()
    for p in projects:
        if p["name"] == dm_project_name:
            return p
    return None


def _find_or_create_dm_chatroom(
    client: httpx.Client,
    project_id: str,
    agent_name: str,
    assistant_token: str,
) -> Optional[dict]:
    """Find or create a DM chatroom for an agent."""
    chatroom_name = f"dm-{agent_name}"

    # Try to get existing chatroom
    resp = client.get(f"/projects/{project_id}/chatrooms")
    if resp.status_code == 200:
        chatrooms = resp.json()
        for room in chatrooms:
            if room["name"] == chatroom_name:
                return room

    # Chatroom doesn't exist - create it using assistant token
    # First, resolve agent name to ID
    resp = client.get("/agents")
    if resp.status_code != 200:
        return None
    agents = resp.json()
    agent = None
    for a in agents:
        if a["name"] == agent_name:
            agent = a
            break
    if not agent:
        return None

    # Create chatroom with assistant token
    resp = client.post(
        f"/projects/{project_id}/chatrooms",
        json={"name": chatroom_name, "participants": [agent["id"]]},
        headers={"Authorization": f"Bearer {assistant_token}"},
    )
    if resp.status_code == 200:
        return resp.json()
    return None


@dm_app.command("send")
def dm_send(
    agent_name: str = typer.Argument(..., help="Agent name to message"),
    message: str = typer.Argument(..., help="Message content"),
    username: str = typer.Option(..., "-u", "--username", help="Username"),
    password: str = typer.Option(..., "-p", "--password", help="Password"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Root data directory (assistant creds under {data_dir}/agents/)"),
):
    """Send a direct message to an agent.

    Creates the DM chatroom if it doesn't exist.

    Example:
        clawmeets dm send researcher "Can you help me with this research?" -u alice -p mypassword
    """
    with _http(server) as client:
        # Login as user
        resp = client.post("/auth/login", json={"username": username, "password": password})
        try:
            login_result = _ok(resp)
        except SystemExit:
            typer.echo("Error: Invalid username or password", err=True)
            raise typer.Exit(1)

        token = login_result["token"]
        assistant_id = login_result.get("assistant_agent_id")

        # Find DM project
        dm_project = _find_dm_project(client, username)
        if not dm_project:
            typer.echo(f"Error: DM project not found for user {username}", err=True)
            raise typer.Exit(1)

        # Load assistant credentials for chatroom creation
        agents_dir = Path(data_dir).expanduser() / "agents"
        assistant_cred_path = None
        if agents_dir.exists():
            for entry in agents_dir.iterdir():
                if entry.is_dir() and entry.name.endswith(f"-{assistant_id}"):
                    cred_path = entry / "credential.json"
                    if cred_path.exists():
                        assistant_cred_path = cred_path
                        break

        assistant_token = None
        if assistant_cred_path:
            creds = json.loads(assistant_cred_path.read_text())
            assistant_token = creds.get("token")

        # Find or create DM chatroom
        chatroom = _find_or_create_dm_chatroom(
            client, dm_project["id"], agent_name, assistant_token or token
        )
        if not chatroom:
            typer.echo(f"Error: Could not find or create DM chatroom for agent {agent_name}", err=True)
            raise typer.Exit(1)

        # Send message via user-message endpoint
        resp = client.post(
            f"/projects/{dm_project['id']}/chatrooms/{chatroom['name']}/user-message",
            json={"content": message},
            headers={"Authorization": f"Bearer {token}"},
        )
        result = _ok(resp)
        typer.echo(f"Message sent to @{agent_name}")


@dm_app.command("list")
def dm_list(
    username: str = typer.Option(..., "-u", "--username", help="Username"),
    password: str = typer.Option(..., "-p", "--password", help="Password"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
):
    """List all DM conversations.

    Example:
        clawmeets dm list -u alice -p mypassword
    """
    with _http(server) as client:
        # Login as user
        resp = client.post("/auth/login", json={"username": username, "password": password})
        try:
            login_result = _ok(resp)
        except SystemExit:
            typer.echo("Error: Invalid username or password", err=True)
            raise typer.Exit(1)

        # Find DM project
        dm_project = _find_dm_project(client, username)
        if not dm_project:
            typer.echo(f"No DM project found for user {username}")
            return

        # List chatrooms that are DM chatrooms
        resp = client.get(f"/projects/{dm_project['id']}/chatrooms")
        chatrooms = _ok(resp)

        dm_rooms = [r for r in chatrooms if r["name"].startswith("dm-")]
        if not dm_rooms:
            typer.echo("No DM conversations yet.")
            return

        typer.echo("DM Conversations:")
        for room in dm_rooms:
            agent_name = room["name"][3:]  # Remove "dm-" prefix
            # Get message count
            resp = client.get(f"/projects/{dm_project['id']}/chatrooms/{room['name']}/messages")
            if resp.status_code == 200:
                messages = resp.json()
                msg_count = len(messages)
                last_msg = messages[-1] if messages else None
                last_preview = ""
                if last_msg:
                    content = last_msg.get("content", "")[:50]
                    from_name = last_msg.get("from_participant_name", "?")
                    last_preview = f" | Last: {from_name}: {content}..."
                typer.echo(f"  @{agent_name:20s}  ({msg_count} messages){last_preview}")
            else:
                typer.echo(f"  @{agent_name}")


@dm_app.command("history")
def dm_history(
    agent_name: str = typer.Argument(..., help="Agent name"),
    username: str = typer.Option(..., "-u", "--username", help="Username"),
    password: str = typer.Option(..., "-p", "--password", help="Password"),
    limit: int = typer.Option(20, "-n", "--limit", help="Number of messages to show"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
):
    """Show DM history with an agent.

    Example:
        clawmeets dm history researcher -u alice -p mypassword -n 50
    """
    with _http(server) as client:
        # Login as user
        resp = client.post("/auth/login", json={"username": username, "password": password})
        try:
            login_result = _ok(resp)
        except SystemExit:
            typer.echo("Error: Invalid username or password", err=True)
            raise typer.Exit(1)

        # Find DM project
        dm_project = _find_dm_project(client, username)
        if not dm_project:
            typer.echo(f"Error: DM project not found for user {username}", err=True)
            raise typer.Exit(1)

        chatroom_name = f"dm-{agent_name}"

        # Get messages
        resp = client.get(f"/projects/{dm_project['id']}/chatrooms/{chatroom_name}/messages")
        if resp.status_code == 404:
            typer.echo(f"No conversation with @{agent_name} yet.")
            return
        messages = _ok(resp)

        if not messages:
            typer.echo(f"No messages with @{agent_name} yet.")
            return

        # Show last N messages
        messages = messages[-limit:]
        typer.echo(f"DM History with @{agent_name} (last {len(messages)} messages):")
        typer.echo("-" * 60)
        for m in messages:
            ts = m.get("ts", "")[:19]
            from_name = m.get("from_participant_name", "?")
            content = m.get("content", "")
            typer.echo(f"[{ts}] {from_name}:")
            typer.echo(f"  {content}")
            typer.echo("")


# ---------------------------------------------------------------------------
# dm schedule / schedules / unschedule
# ---------------------------------------------------------------------------

@dm_app.command("schedule")
def dm_schedule(
    agent_name: str = typer.Argument(..., help="Agent name to schedule messages to"),
    message: str = typer.Argument(..., help="Message content"),
    cron: str = typer.Option(..., "--cron", "-c", help="Cron expression (e.g. '@daily', '0 9 * * *')"),
    end_at: Optional[str] = typer.Option(None, "--end-at", help="Expiration time (ISO 8601)"),
    username: str = typer.Option(..., "-u", "--username", help="Username"),
    password: str = typer.Option(..., "-p", "--password", help="Password"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Root data directory (assistant creds under {data_dir}/agents/)"),
):
    """Schedule a recurring DM to an agent.

    Creates the DM chatroom if it doesn't exist, then schedules the message.

    Examples:
        clawmeets dm schedule researcher "Check for new findings" --cron "@daily" -u alice -p mypass
        clawmeets dm schedule analyst "Run weekly report" --cron "0 9 * * 1" -u alice -p mypass
    """
    with _http(server) as client:
        # Login
        resp = client.post("/auth/login", json={"username": username, "password": password})
        try:
            login_result = _ok(resp)
        except SystemExit:
            typer.echo("Error: Invalid username or password", err=True)
            raise typer.Exit(1)

        token = login_result["token"]
        assistant_id = login_result.get("assistant_agent_id")

        # Find DM project
        dm_project = _find_dm_project(client, username)
        if not dm_project:
            typer.echo(f"Error: DM project not found for user {username}", err=True)
            raise typer.Exit(1)

        # Load assistant credentials for chatroom creation
        agents_dir = Path(data_dir).expanduser() / "agents"
        assistant_token = None
        if agents_dir.exists():
            for entry in agents_dir.iterdir():
                if entry.is_dir() and entry.name.endswith(f"-{assistant_id}"):
                    cred_path = entry / "credential.json"
                    if cred_path.exists():
                        creds = json.loads(cred_path.read_text())
                        assistant_token = creds.get("token")
                        break

        # Find or create DM chatroom
        chatroom = _find_or_create_dm_chatroom(
            client, dm_project["id"], agent_name, assistant_token or token
        )
        if not chatroom:
            typer.echo(f"Error: Could not find or create DM chatroom for agent {agent_name}", err=True)
            raise typer.Exit(1)

        # Create scheduled message
        payload: dict = {
            "project_id": dm_project["id"],
            "chatroom_name": chatroom["name"],
            "content": message,
            "cron_expression": cron,
        }
        if end_at:
            payload["end_at"] = end_at

        resp = client.post(
            "/scheduled-messages",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        result = _ok(resp)
        typer.echo(
            f"Scheduled message to @{agent_name}: cron={cron!r} "
            f"next fire: {result['next_fire_at']}"
        )


@dm_app.command("schedules")
def dm_schedules(
    username: str = typer.Option(..., "-u", "--username", help="Username"),
    password: str = typer.Option(..., "-p", "--password", help="Password"),
    all_: bool = typer.Option(False, "--all", "-a", help="Include inactive schedules"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
):
    """List your scheduled DM messages.

    Example:
        clawmeets dm schedules -u alice -p mypassword
    """
    with _http(server) as client:
        # Login
        resp = client.post("/auth/login", json={"username": username, "password": password})
        try:
            login_result = _ok(resp)
        except SystemExit:
            typer.echo("Error: Invalid username or password", err=True)
            raise typer.Exit(1)

        token = login_result["token"]

        params = {"active_only": "false"} if all_ else {}
        resp = client.get(
            "/scheduled-messages",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        schedules = _ok(resp)

        # Filter to DM chatrooms only
        dm_schedules = [s for s in schedules if s["chatroom_name"].startswith("dm-")]

        if not dm_schedules:
            typer.echo("No scheduled DM messages.")
            return

        for s in dm_schedules:
            agent_name = s["chatroom_name"].removeprefix("dm-")
            status = "active" if s["is_active"] else "inactive"
            typer.echo(
                f"  [{status}] {s['id'][:8]}... "
                f"@{agent_name} cron={s['cron_expression']!r} "
                f"next={s['next_fire_at']} "
                f"content={s['content'][:60]!r}"
            )


@dm_app.command("unschedule")
def dm_unschedule(
    schedule_id: str = typer.Argument(..., help="Scheduled message ID to cancel"),
    username: str = typer.Option(..., "-u", "--username", help="Username"),
    password: str = typer.Option(..., "-p", "--password", help="Password"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
):
    """Cancel a scheduled DM message.

    Example:
        clawmeets dm unschedule abc12345-... -u alice -p mypassword
    """
    with _http(server) as client:
        # Login
        resp = client.post("/auth/login", json={"username": username, "password": password})
        try:
            login_result = _ok(resp)
        except SystemExit:
            typer.echo("Error: Invalid username or password", err=True)
            raise typer.Exit(1)

        token = login_result["token"]

        resp = client.delete(
            f"/scheduled-messages/{schedule_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        _ok(resp)
        typer.echo(f"Scheduled message {schedule_id[:8]}... cancelled.")


# ---------------------------------------------------------------------------
# MCP commands (runner-local; operate on agent directories on this machine)
# ---------------------------------------------------------------------------

def _resolve_agent_dir(data_dir: Path, agent: str) -> Path:
    """Find an agent's working directory under {data_dir}/agents/ by name or id.

    Matches on either the full directory name ({name}-{id}) or a prefix match
    where the prefix equals the agent name.
    """
    agents_dir = Path(data_dir).expanduser() / "agents"
    if not agents_dir.exists():
        typer.echo(f"Error: no agents directory at {agents_dir}", err=True)
        raise typer.Exit(1)
    matches = [
        d for d in agents_dir.iterdir()
        if d.is_dir() and (d.name == agent or d.name.startswith(f"{agent}-"))
    ]
    if not matches:
        typer.echo(
            f"Error: no agent matching {agent!r} under {agents_dir}. "
            f"Available: {[d.name for d in agents_dir.iterdir() if d.is_dir()]}",
            err=True,
        )
        raise typer.Exit(1)
    if len(matches) > 1:
        typer.echo(
            f"Error: multiple agents match {agent!r}: {[d.name for d in matches]}. "
            f"Pass the full directory name.",
            err=True,
        )
        raise typer.Exit(1)
    return matches[0]


@mcp_app.command("auth")
def mcp_auth(
    mcp_name: str = typer.Argument(..., help="MCP server name (e.g. 'gmail', 'google-calendar')"),
    agent: str = typer.Option(..., "--agent", "-a", help="Agent name (or {name}-{id} dirname)"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
    credentials: Optional[Path] = typer.Option(
        None, "--credentials",
        help="Path to the Google OAuth installed-app client secrets JSON. Overrides "
             "CLAWMEETS_GOOGLE_OAUTH_CREDENTIALS and ~/.clawmeets/google_oauth_client.json.",
    ),
):
    """Authenticate an MCP server for an agent (one-time OAuth setup).

    Opens the default browser, completes the installed-app OAuth flow against the
    provider (e.g. Google), and writes the resulting token to
    {agent_dir}/mcp-hub/servers/{mcp_name}/token.json (mode 0600). Tokens never
    transit the ClawMeets server.
    """
    from clawmeets.runner.mcp_manager import McpManager

    agent_dir = _resolve_agent_dir(data_dir, agent)
    manager = McpManager(agent_dir)
    manifest = manager.get_manifest(mcp_name)
    if manifest is None:
        typer.echo(
            f"Error: MCP {mcp_name!r} is not installed for agent {agent!r}. "
            f"Install it via the web UI or the /agents/{{id}}/mcps endpoint first.",
            err=True,
        )
        raise typer.Exit(1)

    auth = manifest.get("auth") or {}
    method = auth.get("method")
    if not method:
        typer.echo(f"MCP {mcp_name!r} does not require authentication.")
        raise typer.Exit(0)

    if method == "google_oauth_installed":
        from clawmeets.mcp.auth.google_oauth import GoogleOAuthError, run_installed_flow
        scopes = auth.get("scopes") or []
        if not scopes:
            typer.echo(f"Error: no scopes defined in {mcp_name!r} manifest", err=True)
            raise typer.Exit(1)
        token_path = manager.token_path(mcp_name)
        typer.echo(f"Starting Google OAuth for {mcp_name} (agent={agent_dir.name})")
        typer.echo(f"  scopes: {scopes}")
        typer.echo(f"  token:  {token_path}")
        try:
            run_installed_flow(scopes=scopes, token_path=token_path, client_secrets=credentials)
        except GoogleOAuthError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
        typer.echo(f"OK. {mcp_name} is now authenticated for {agent_dir.name}.")
        return

    typer.echo(f"Error: unsupported auth method {method!r} for {mcp_name!r}", err=True)
    raise typer.Exit(1)


@mcp_app.command("list")
def mcp_list(
    agent: str = typer.Option(..., "--agent", "-a", help="Agent name"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """List installed MCP servers for an agent, showing auth status."""
    from clawmeets.runner.mcp_manager import McpManager

    agent_dir = _resolve_agent_dir(data_dir, agent)
    manager = McpManager(agent_dir)
    installed = manager.installed_mcps()
    if not installed:
        typer.echo(f"No MCP servers installed for {agent_dir.name}.")
        return
    for name in installed:
        status = "needs-auth" if manager.needs_auth(name) else "ready"
        typer.echo(f"  {name:24s}  {status}")


@mcp_app.command("status")
def mcp_status(
    mcp_name: str = typer.Argument(..., help="MCP server name"),
    agent: str = typer.Option(..., "--agent", "-a", help="Agent name"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Show the authentication status of one MCP server for an agent."""
    from clawmeets.runner.mcp_manager import McpManager

    agent_dir = _resolve_agent_dir(data_dir, agent)
    manager = McpManager(agent_dir)
    manifest = manager.get_manifest(mcp_name)
    if manifest is None:
        typer.echo(f"{mcp_name}: not installed for {agent_dir.name}")
        raise typer.Exit(1)
    auth = manifest.get("auth") or {}
    if not auth.get("method"):
        typer.echo(f"{mcp_name}: ready (no auth required)")
        return
    if manager.needs_auth(mcp_name):
        typer.echo(
            f"{mcp_name}: needs-auth — run `clawmeets mcp auth {mcp_name} "
            f"--agent {agent}`"
        )
        raise typer.Exit(2)
    typer.echo(f"{mcp_name}: ready (token at {manager.token_path(mcp_name)})")



# ---------------------------------------------------------------------------
# reflection commands (account-level)
# ---------------------------------------------------------------------------

@reflection_app.command("set")
def reflection_set(
    cron: str = typer.Option(..., "--cron", help="Reflect cron (e.g. '0 9 * * *' for daily 9am)."),
    lint_cron: Optional[str] = typer.Option(
        None, "--lint-cron",
        help="Optional lint cron (e.g. '0 9 * * 1' for weekly Mon 9am). "
        "Lint mode audits existing memory; reflect mode distills new lessons.",
    ),
    no_lint: bool = typer.Option(
        False, "--no-lint",
        help="Clear the lint cadence (disables structural lint pass). "
        "Mutually exclusive with --lint-cron.",
    ),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Create or update the account-level reflection schedule.

    Reflect cadence (--cron) is required. Lint cadence is optional: pass
    --lint-cron to enable, --no-lint to clear, or omit both to leave the
    server-side lint setting unchanged.
    """
    if lint_cron and no_lint:
        typer.echo("Error: --lint-cron and --no-lint are mutually exclusive.", err=True)
        raise typer.Exit(1)

    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        # Look up the current lint cron so an unspecified --lint-cron preserves
        # whatever the user already had.
        existing_lint: Optional[str] = None
        try:
            current = client.get(
                "/account/reflection-schedule",
                headers={"Authorization": f"Bearer {token}"},
            )
            if current.status_code == 200 and current.content:
                payload = current.json()
                if payload:
                    existing_lint = payload.get("lint_cron_expression")
        except Exception:
            pass

        if no_lint:
            new_lint = None
        elif lint_cron:
            new_lint = lint_cron
        else:
            new_lint = existing_lint

        resp = client.put(
            "/account/reflection-schedule",
            json={
                "cron_expression": cron,
                "is_active": True,
                "lint_cron_expression": new_lint,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        result = _ok(resp)

    typer.echo(
        f"Reflect cron: {result['cron_expression']!r}  next: {result['next_fire_at']}"
    )
    if result.get("lint_cron_expression"):
        typer.echo(
            f"Lint cron:    {result['lint_cron_expression']!r}  next: {result.get('next_lint_fire_at')}"
        )
    else:
        typer.echo("Lint cron:    (off)")


@reflection_app.command("off")
def reflection_off(
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Deactivate the account-level reflection schedule."""
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        resp = client.delete(
            "/account/reflection-schedule",
            headers={"Authorization": f"Bearer {token}"},
        )
        _ok(resp)
    typer.echo("Reflection schedule deactivated.")


@reflection_app.command("show")
def reflection_show(
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Show the current account-level reflection schedule."""
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        resp = client.get(
            "/account/reflection-schedule",
            headers={"Authorization": f"Bearer {token}"},
        )
        result = _ok(resp) if resp.status_code != 200 or resp.content else None
        # _ok() raises on non-2xx; if 200 with null body, result is None.
        if resp.status_code == 200:
            try:
                result = resp.json()
            except Exception:
                result = None
    if result is None:
        typer.echo("No reflection schedule configured. Run `clawmeets reflection set --cron \"0 9 * * *\"` to enable.")
        return
    typer.echo(f"Active:           {result['is_active']}")
    typer.echo(f"Reflect cron:     {result['cron_expression']}")
    typer.echo(f"  last fired:     {result.get('last_fired_at') or 'never'}")
    typer.echo(f"  next fire:      {result['next_fire_at']}")
    lint_cron = result.get("lint_cron_expression")
    if lint_cron:
        typer.echo(f"Lint cron:        {lint_cron}")
        typer.echo(f"  last fired:     {result.get('last_lint_fired_at') or 'never'}")
        typer.echo(f"  next fire:      {result.get('next_lint_fire_at') or '—'}")
    else:
        typer.echo("Lint cron:        (off — pass --lint-cron to enable)")


# ---------------------------------------------------------------------------
# bootstrap (two-phase personalized first-fill)
# ---------------------------------------------------------------------------
#
# `clawmeets bootstrap` is a one-shot orchestrator that personalizes a freshly
# installed team from the user's own data:
#
#   Phase 1 — gather a profile dump from the user's Gmail + Calendar (or fall
#             back to a 3-question prompt), DM it to the assistant; the
#             assistant's reflect skill writes USER.md.
#   Phase 2 — for each worker agent, do a deep-research pass on the agent's
#             domain decorated by USER.md, DM it to the agent; the agent's
#             reflect skill writes learnings/.
#
# All transport rides existing rails (DM POST + reflect-trigger marker). The
# only new piece on the agent side is the Bootstrap mode added to reflect's
# SKILL.md.

_BOOTSTRAP_MARKER = "<!-- clawmeets:bootstrap-trigger -->"
_PHASE1_TIMEOUT_DEFAULT = 300   # 5 min for assistant to write USER.md
_PHASE2_TIMEOUT_DEFAULT = 600   # 10 min per worker (web research is slow)
_GATHER_TIMEOUT = 600           # 10 min cap on a single `claude` gather call


def _ensure_fresh_user_token(server_url: str, data_dir: Path, username: str, current_token: str) -> str:
    """Verify the saved JWT still works; auto-refresh from saved password if expired.

    `clawmeets init` saves both the JWT and the password into settings.json.
    JWTs eventually expire; rather than telling users to manually re-login, we
    silently re-issue using the saved password and persist the new token.
    """
    from clawmeets.cli_lifecycle import get_user_config_path

    with httpx.Client(base_url=server_url, timeout=30) as c:
        ping = c.get("/auth/user/me", headers={"Authorization": f"Bearer {current_token}"})
        if ping.status_code == 200:
            return current_token

    cfg_path = get_user_config_path(Path(data_dir).expanduser(), username)
    if not cfg_path.exists():
        typer.echo(
            f"Error: session expired and no saved config at {cfg_path}.\n"
            f"Run `clawmeets user login {username} <password> --save` and retry.",
            err=True,
        )
        raise typer.Exit(1)
    try:
        cfg = json.loads(cfg_path.read_text())
    except (json.JSONDecodeError, OSError):
        typer.echo(f"Error: settings.json at {cfg_path} is corrupt; can't refresh.", err=True)
        raise typer.Exit(1)
    password = (cfg.get("user") or {}).get("password")
    if not password:
        typer.echo(
            f"Error: session expired and no saved password to refresh with.\n"
            f"Run `clawmeets user login {username} <password> --save` and retry.",
            err=True,
        )
        raise typer.Exit(1)

    with httpx.Client(base_url=server_url, timeout=30) as c:
        resp = c.post("/auth/login", json={"username": username, "password": password})
    if resp.status_code != 200:
        typer.echo(f"Error: token refresh failed ({resp.text}).", err=True)
        raise typer.Exit(1)
    new_token = resp.json().get("token")
    if not new_token:
        typer.echo("Error: refresh succeeded but server returned no token.", err=True)
        raise typer.Exit(1)

    cfg.setdefault("user", {})["token"] = new_token
    cfg_path.write_text(json.dumps(cfg, indent=2))
    typer.echo("  (refreshed expired session)")
    return new_token


def _find_agent_dir_by_id(agents_dir: Path, agent_id: str) -> Optional[Path]:
    """Locate `{agents_dir}/{name}-{agent_id}/`. Skips DELETED-* archives."""
    if not agents_dir.is_dir():
        return None
    for entry in agents_dir.iterdir():
        if not entry.is_dir() or entry.name.startswith("DELETED-"):
            continue
        if entry.name.endswith(f"-{agent_id}"):
            return entry
    return None


def _resolve_agent_knowledge_dir(
    server_card: Optional[dict],
    agent_dir: Optional[Path],
    user_config_dir: Path,
) -> Optional[Path]:
    """Read `local_settings.knowledge_dir`, preferring the server's card
    (just-saved value) and falling back to the local card.json on disk
    if the server doesn't expose it.

    The runner mirrors AGENT_SETTINGS_CHANGE into local card.json, but the
    web-UI save → broadcast → write cycle has a delay window. Reading the
    server first means we pick up changes the user just made even when the
    local card hasn't caught up yet.
    """
    raw = ""
    if server_card:
        raw = (server_card.get("local_settings") or {}).get("knowledge_dir") or ""
    if not raw and agent_dir is not None:
        card_path = agent_dir / "card.json"
        if card_path.exists():
            try:
                card = json.loads(card_path.read_text())
                raw = (card.get("local_settings") or {}).get("knowledge_dir") or ""
            except (json.JSONDecodeError, OSError):
                pass
    if not raw:
        return None
    return resolve_local_knowledge_dir(str(raw), user_config_dir)


def _check_gmail_calendar_ready(assistant_dir: Path) -> tuple[bool, list[str]]:
    """Both Gmail + Calendar MCPs must be installed AND OAuthed. Returns
    (ready, missing_reasons) so the caller can print actionable guidance."""
    mcp_mgr = McpManager(assistant_dir)
    installed = set(mcp_mgr.installed_mcps())
    reasons: list[str] = []
    for name in ("gmail", "google-calendar"):
        if name not in installed:
            reasons.append(f"{name} not installed (run `clawmeets mcp install {name}`)")
        elif not mcp_mgr.has_token(name):
            reasons.append(f"{name} not authed (run `clawmeets mcp auth {name}`)")
    return (not reasons), reasons


def _gather_user_profile_rich(assistant_dir: Path) -> str:
    """Spawn a one-shot `claude` with the assistant's MCP stack and capture a
    profile dump distilled from Gmail + Calendar.

    Renders `.mcp.json` into a fresh tmp working dir (so we don't trample the
    agent's running sandbox), seeds it with the assistant's MCP manifests +
    tokens, and runs `claude --print` non-interactively.
    """
    import shutil as _shutil
    import subprocess
    import tempfile

    if _shutil.which("claude") is None:
        typer.echo("Error: `claude` CLI not on PATH. Bootstrap requires Claude Code installed on the runner.", err=True)
        raise typer.Exit(1)

    mcp_mgr = McpManager(assistant_dir)
    with tempfile.TemporaryDirectory(prefix="clawmeets-bootstrap-") as td:
        cwd = Path(td)
        mcp_mgr.render_mcp_json(cwd)  # writes cwd/.mcp.json with gmail + gcal

        prompt = (
            "Use your Gmail and Calendar tools to gather signal on the user "
            "(last ~90 days of sent mail, calendar events, recurring meetings, "
            "frequent contacts). Output a single Markdown profile dump covering: "
            "role + industry, geography, recurring contacts (who they are, what "
            "they do), current priorities, voice/tone, any 'do not' preferences. "
            "Do not include raw email bodies or PII beyond names — distill into "
            "prose. 1500–3000 words. Output the dump only, no preamble."
        )
        cmd = [
            "claude",
            "--print",
            "--permission-mode", "bypassPermissions",
            prompt,
        ]
        typer.echo("  [phase 1] gathering Gmail + Calendar signal via claude (this can take a few minutes)…")
        try:
            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=_GATHER_TIMEOUT)
        except subprocess.TimeoutExpired:
            typer.echo(f"Error: gather call timed out after {_GATHER_TIMEOUT}s.", err=True)
            raise typer.Exit(1)
        if proc.returncode != 0:
            typer.echo(f"Error: gather call failed (exit {proc.returncode}).", err=True)
            if proc.stderr.strip():
                typer.echo(proc.stderr.rstrip(), err=True)
            raise typer.Exit(1)
        return proc.stdout.strip() + "\n"


def _gather_user_profile_degraded(non_interactive: bool, missing_reasons: list[str]) -> str:
    """Interactive 3-question fallback when Gmail/Calendar aren't ready."""
    if non_interactive:
        typer.echo(
            "Error: cannot run bootstrap without Gmail/Calendar in non-interactive mode.",
            err=True,
        )
        for r in missing_reasons:
            typer.echo(f"  - {r}", err=True)
        typer.echo("Either install/auth those MCPs or drop --non-interactive.", err=True)
        raise typer.Exit(1)

    typer.echo(
        "Gmail/Calendar isn't fully set up for your assistant, so I can't "
        "auto-derive your profile. Three quick questions to bootstrap with —"
    )
    for r in missing_reasons:
        typer.echo(f"  · {r}")
    typer.echo("(answers stay local in USER.md)\n")
    role = _prompt_required("  1. Role + industry (e.g. 'Senior PM at an AI infra startup')")
    location = _prompt_required("  2. Where you're based (city / region / country)")
    priority = _prompt_required("  3. Single biggest priority for the next 1–3 months")
    return (
        "# User profile (degraded bootstrap — user-provided)\n"
        "\n"
        f"- Role + industry: {role}\n"
        f"- Location: {location}\n"
        f"- Current priority: {priority}\n"
    )


def _prompt_required(text: str) -> str:
    while True:
        val = input(f"{text}: ").strip()
        if val:
            return val
        typer.echo("    (required)")


def _gather_agent_research(agent_name: str, description: str, capabilities: list[str], user_profile: str) -> str:
    """Spawn a one-shot `claude` with web tools enabled, capture a deep-research
    dump for one worker agent decorated by the user profile."""
    import shutil as _shutil
    import subprocess
    import tempfile

    if _shutil.which("claude") is None:
        typer.echo("Error: `claude` CLI not on PATH.", err=True)
        raise typer.Exit(1)

    cap_line = ", ".join(capabilities) if capabilities else "(none listed)"
    prompt = (
        f"You are doing a deep-research dump for the `{agent_name}` agent.\n"
        f"Agent description: {description}\n"
        f"Capabilities: {cap_line}\n"
        f"\n"
        f"Decorate your research with this user profile — weight resources, "
        f"comp data, regulations, and examples toward the user's segment:\n"
        f"\n"
        f"----- USER PROFILE -----\n"
        f"{user_profile}\n"
        f"----- END USER PROFILE -----\n"
        f"\n"
        f"Cover the 4–8 things the agent will actually be asked to do, "
        f"frameworks/rules of thumb/tactics for each, common failure modes, "
        f"and current public sources (cite URLs inline as [source: <url>]). "
        f"Use web search aggressively for current data. 2000–4000 words. "
        f"Output the dump only, no preamble."
    )
    with tempfile.TemporaryDirectory(prefix=f"clawmeets-bootstrap-{agent_name}-") as td:
        cwd = Path(td)
        cmd = [
            "claude",
            "--print",
            "--permission-mode", "bypassPermissions",
            prompt,
        ]
        try:
            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=_GATHER_TIMEOUT)
        except subprocess.TimeoutExpired:
            typer.echo(f"  [{agent_name}] gather timed out after {_GATHER_TIMEOUT}s — skipping.", err=True)
            return ""
        if proc.returncode != 0:
            typer.echo(f"  [{agent_name}] gather failed (exit {proc.returncode}) — skipping.", err=True)
            if proc.stderr.strip():
                typer.echo(proc.stderr.rstrip(), err=True)
            return ""
        return proc.stdout.strip() + "\n"


def _post_bootstrap_dm(
    client: httpx.Client,
    user_jwt: str,
    assistant_token: Optional[str],
    dm_project_id: str,
    target_agent_name: str,
    body: str,
) -> bool:
    """Find or create dm-{name} chatroom, post a user message with the
    bootstrap-trigger marker. Returns True on 2xx."""
    chatroom = _find_or_create_dm_chatroom(client, dm_project_id, target_agent_name, assistant_token or user_jwt)
    if not chatroom:
        typer.echo(f"Error: could not find or create dm-{target_agent_name}", err=True)
        return False
    resp = client.post(
        f"/projects/{dm_project_id}/chatrooms/{chatroom['name']}/user-message",
        json={"content": body},
        headers={"Authorization": f"Bearer {user_jwt}"},
    )
    if resp.status_code >= 400:
        typer.echo(f"Error posting to dm-{target_agent_name}: {resp.text}", err=True)
        return False
    return True


def _poll_for_file(path: Path, timeout_sec: int, label: str) -> bool:
    """Poll until `path` exists with size > 0, or timeout. Prints a tasteful
    one-line progress that updates every 10s."""
    import time
    start = time.time()
    while True:
        if path.exists() and path.stat().st_size > 0:
            elapsed = int(time.time() - start)
            typer.echo(f"  [{label}] done in {elapsed}s — {path}")
            return True
        elapsed = time.time() - start
        if elapsed > timeout_sec:
            typer.echo(f"  [{label}] TIMED OUT after {int(elapsed)}s waiting for {path}", err=True)
            return False
        # Sleep but show heartbeat every 10s
        sleep_chunk = min(10.0, timeout_sec - elapsed)
        if sleep_chunk <= 0:
            continue
        time.sleep(sleep_chunk)
        if int(elapsed) % 30 < 10:  # ~every 30s
            typer.echo(f"  [{label}] still waiting… ({int(elapsed)}s/{timeout_sec}s)")


@bootstrap_app.callback()
def bootstrap_command(
    phase: str = typer.Option("all", "--phase", help="1, 2, or all (default)"),
    agent: Optional[str] = typer.Option(None, "--agent", help="Limit Phase 2 to a single worker agent name"),
    force: bool = typer.Option(False, "--force", help="Re-trigger even if files already exist (skill still gates the actual overwrite)"),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Fail in degraded path instead of prompting"),
    allow_missing_user_profile: bool = typer.Option(
        False, "--allow-missing-user-profile",
        help="Run Phase 2 even when USER.md is missing (research will be generic)",
    ),
    timeout: int = typer.Option(_PHASE1_TIMEOUT_DEFAULT, "--timeout", help="Per-agent poll timeout in seconds"),
    username: Optional[str] = typer.Option(None, "-u", "--username", help="Username (default: current saved session)"),
    password: Optional[str] = typer.Option(None, "-p", "--password", help="Password (default: from saved session)"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Personalize a freshly installed agent team using your own data.

    Phase 1: assistant USER.md from your Gmail + Calendar (or 3-question fallback).
    Phase 2: each worker agent's learnings/ from a deep-research pass shaped by USER.md.

    Idempotent — re-running skips agents whose files are already populated. Use --force
    to re-trigger; the reflect skill itself decides whether to overwrite (it doesn't,
    by default, so delete USER.md or learnings/INDEX.md first if you really want a redo).

    Run after `clawmeets init --from-url` and `clawmeets agent start`.
    """
    if phase not in ("1", "2", "all"):
        typer.echo("Error: --phase must be one of: 1, 2, all", err=True)
        raise typer.Exit(1)

    # ----- Resolve session -----
    server_url, user_jwt = _resolve_user_session(data_dir, None, server, as_user=username)

    # We need both username (for DM-{username} project lookup) and the saved
    # password is NOT required — a JWT is enough for the user-message POST.
    # But _find_dm_project still needs `username`; pull it from the saved session.
    if username is None:
        from clawmeets.cli_lifecycle import get_current_user
        username = get_current_user(Path(data_dir).expanduser())
    if not username:
        typer.echo("Error: no username known. Pass -u or run `clawmeets user login --save` first.", err=True)
        raise typer.Exit(1)

    data_dir_p = Path(data_dir).expanduser()
    agents_dir = data_dir_p / "agents"
    user_config_dir = data_dir_p / "config" / username

    # Refresh the saved JWT if it's expired — `clawmeets init` saved the
    # password too, so we can re-login transparently.
    user_jwt = _ensure_fresh_user_token(server_url, data_dir_p, username, user_jwt)

    with _http(server_url) as client:
        # Inject the user JWT for `_find_dm_project` and friends (they all
        # call `client.get/post` against `server_url`; we pass the bearer per
        # call where needed).
        client.headers.update({"Authorization": f"Bearer {user_jwt}"})

        # ----- Resolve assistant -----
        me_resp = client.get("/auth/user/me")
        if me_resp.status_code != 200:
            typer.echo(f"Error: could not load /auth/user/me ({me_resp.text})", err=True)
            raise typer.Exit(1)
        me = me_resp.json()
        assistant_id = me.get("assistant_agent_id")
        assistant_name = me.get("assistant_agent_name") or f"{username}-assistant"
        if not assistant_id:
            typer.echo("Error: user has no assistant agent on the server.", err=True)
            raise typer.Exit(1)

        assistant_dir = _find_agent_dir_by_id(agents_dir, assistant_id)
        if assistant_dir is None:
            typer.echo(
                f"Error: no local agent dir for assistant id {assistant_id[:8]}…. "
                f"Run `clawmeets init --from-url …` first to set up the assistant locally.",
                err=True,
            )
            raise typer.Exit(1)

        # Fetch the assistant's server-side card — that's where the just-saved
        # web-UI value lives. Local card.json is the fallback.
        asst_resp = client.get(f"/agents/{assistant_id}")
        assistant_server_card = asst_resp.json() if asst_resp.status_code == 200 else None

        assistant_kdir = _resolve_agent_knowledge_dir(assistant_server_card, assistant_dir, user_config_dir)
        if assistant_kdir is None:
            typer.echo(
                f"Error: assistant has no knowledge_dir configured. "
                f"Open the web UI agent settings page for '{assistant_name}', "
                f"set the Knowledge Directory field (e.g. ./knowledge), click "
                f"Save Changes, then re-run `clawmeets bootstrap`.",
                err=True,
            )
            raise typer.Exit(1)

        # ----- DM project -----
        dm_project = _find_dm_project(client, username)
        if not dm_project:
            typer.echo(f"Error: DM project DM-{username} not found on server.", err=True)
            raise typer.Exit(1)

        # Assistant's own token (for chatroom creation only)
        assistant_token: Optional[str] = None
        cred_path = assistant_dir / "credential.json"
        if cred_path.exists():
            try:
                assistant_token = json.loads(cred_path.read_text()).get("token")
            except (json.JSONDecodeError, OSError):
                pass

        # ============================================================
        # Phase 1
        # ============================================================
        if phase in ("1", "all"):
            user_md_path = assistant_kdir / "USER.md"
            if user_md_path.exists() and not force:
                typer.echo(f"[phase 1] skipped — USER.md already exists at {user_md_path}")
                typer.echo("           pass --force to re-trigger (skill still won't overwrite without `rm USER.md`)")
            else:
                typer.echo(f"[phase 1] personalizing assistant '{assistant_name}'")
                ready, reasons = _check_gmail_calendar_ready(assistant_dir)
                if ready:
                    dump = _gather_user_profile_rich(assistant_dir)
                else:
                    dump = _gather_user_profile_degraded(non_interactive, reasons)

                body = (
                    f"{_BOOTSTRAP_MARKER}\n\n"
                    "You're being bootstrapped. Treat the dump below as authoritative — "
                    "distill into USER.md per the reflect skill's Bootstrap mode.\n\n"
                    "== USER PROFILE DUMP ==\n"
                    f"{dump}"
                )
                if not _post_bootstrap_dm(client, user_jwt, assistant_token, dm_project["id"], assistant_name, body):
                    raise typer.Exit(1)
                typer.echo(f"  [phase 1] dispatched to dm-{assistant_name}; waiting for USER.md…")
                if not _poll_for_file(user_md_path, timeout, "phase 1"):
                    typer.echo(
                        "[phase 1] FAILED — USER.md was not produced. Make sure "
                        "`clawmeets agent start` is running and the assistant is online.",
                        err=True,
                    )
                    raise typer.Exit(1)

        # ============================================================
        # Phase 2
        # ============================================================
        if phase in ("2", "all"):
            user_md_path = assistant_kdir / "USER.md"
            if user_md_path.exists():
                user_profile = user_md_path.read_text()
            elif allow_missing_user_profile:
                typer.echo("[phase 2] WARNING: USER.md missing, proceeding without personalization (--allow-missing-user-profile)")
                user_profile = "(no user profile available; do generic research)"
            else:
                typer.echo(
                    "Error: USER.md missing — run `clawmeets bootstrap --phase 1` first, "
                    "or pass --allow-missing-user-profile to run unpersonalized.",
                    err=True,
                )
                raise typer.Exit(1)

            workers = _fetch_owned_agents(client, user_jwt)
            workers = [a for a in workers if a.get("id") != assistant_id]
            if agent:
                workers = [a for a in workers if a.get("name") == agent]
                if not workers:
                    typer.echo(f"Error: no owned worker agent named {agent!r}.", err=True)
                    raise typer.Exit(1)

            if not workers:
                typer.echo("[phase 2] no worker agents to bootstrap. Done.")
                return

            seeded = 0
            skipped = 0
            failed = 0
            for w in workers:
                w_id = w["id"]
                w_name = w["name"]
                w_dir = _find_agent_dir_by_id(agents_dir, w_id)
                if w_dir is None:
                    typer.echo(f"  [{w_name}] no local dir — skipping (was this agent registered locally?)")
                    skipped += 1
                    continue
                # `w` came from /agents (server-side), so its local_settings reflects
                # the latest UI save. Local card.json is the disk fallback.
                w_kdir = _resolve_agent_knowledge_dir(w, w_dir, user_config_dir)
                if w_kdir is None:
                    typer.echo(f"  [{w_name}] no knowledge_dir set (web UI agent settings) — skipping")
                    skipped += 1
                    continue
                index_path = w_kdir / "learnings" / "INDEX.md"
                if index_path.exists() and not force:
                    typer.echo(f"  [{w_name}] skipped — learnings/INDEX.md already exists")
                    skipped += 1
                    continue

                typer.echo(f"[phase 2] {w_name}: gathering web research…")
                dump = _gather_agent_research(
                    w_name,
                    w.get("description") or "",
                    w.get("capabilities") or [],
                    user_profile,
                )
                if not dump:
                    failed += 1
                    continue

                body = (
                    f"{_BOOTSTRAP_MARKER}\n\n"
                    "You're being bootstrapped. Treat the dump below as authoritative — "
                    "structure into learnings/ per the reflect skill's Bootstrap mode.\n\n"
                    "== USER PROFILE (read-only context) ==\n"
                    f"{user_profile}\n"
                    f"== DEEP-RESEARCH DUMP ({w.get('description') or w_name}) ==\n"
                    f"{dump}"
                )
                if not _post_bootstrap_dm(client, user_jwt, assistant_token, dm_project["id"], w_name, body):
                    failed += 1
                    continue
                typer.echo(f"  [{w_name}] dispatched; waiting for learnings/INDEX.md…")
                # Phase 2 timeout is longer than Phase 1 by default; respect --timeout for both.
                phase2_timeout = max(timeout, _PHASE2_TIMEOUT_DEFAULT) if timeout == _PHASE1_TIMEOUT_DEFAULT else timeout
                if _poll_for_file(index_path, phase2_timeout, w_name):
                    seeded += 1
                else:
                    failed += 1

            typer.echo(f"\n[phase 2] summary: bootstrapped={seeded} skipped={skipped} failed/timed-out={failed}")
