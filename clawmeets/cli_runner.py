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
from typing import List, Optional

import httpx
import typer
import websockets

from clawmeets.api.responses import AgentRegistrationResponse
from clawmeets.api.control import ControlEnvelope, ControlMessageType
from clawmeets.api.client import ClawMeetsClient
from clawmeets.models.chat_message import ChatMessage
from clawmeets.utils.file_io import FileUtil
from clawmeets.utils.knowledge_dir import resolve_local_knowledge_dir, resolve_local_dwh_dir
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
from clawmeets.runner.knowledge_pack_manager import KnowledgePackManager
from clawmeets.runner.personal_skill_manager import PersonalSkillManager
from clawmeets.runner.skill_manager import SkillManager

# Backward compatibility alias
AgentRegistrationResult = AgentRegistrationResponse

# Sub-command groups
agent_app = typer.Typer(help="Agent commands", no_args_is_help=True)
user_app  = typer.Typer(help="User commands",  no_args_is_help=True)
dm_app    = typer.Typer(help="Direct message commands", no_args_is_help=True)
front_desk_app = typer.Typer(
    help=(
        "Front Desk commands. A Front Desk project is a long-lived, DM-shaped "
        "channel whose coordinator is the targeted agent — it can spawn worker "
        "rooms with other agents (per the agent's invite allowlist) instead of "
        "running solo. Internal (your own assistant) and external (someone else's "
        "discoverable agent) channels share the same shape."
    ),
    no_args_is_help=True,
)
mcp_app   = typer.Typer(help="MCP server commands (auth, list, status)", no_args_is_help=True)
team_app  = typer.Typer(help="Manage user-defined teams on your agents (the TEAMS sidebar)", no_args_is_help=True)
knowledge_pack_app = typer.Typer(
    help=(
        "Manage your knowledge packs — named, user-curated markdown bundles you "
        "can install on any of your agents. Once installed, they render in the "
        "agent's AUTHORITATIVE memory layer."
    ),
    no_args_is_help=True,
)
reflection_app = typer.Typer(help="Configure account-level reflection schedule (one cron, fans out to all your agents).", no_args_is_help=True)
bootstrap_app = typer.Typer(
    help=(
        "Personalize your fresh agents from your own data (one-time, opt-in). "
        "Two subcommands: `assistant` (Phase 1 — writes USER.md from your Gmail + Calendar) "
        "then `agent` (Phase 2 — each worker writes learnings/ from a deep-research pass shaped by USER.md). "
        "Run `assistant` first, watch the chat for completion, then run `agent`."
    ),
    no_args_is_help=True,
)


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


def _build_llm_provider(
    provider: str,
    model: Optional[str],
    *,
    plugin_dirs: list[Path],
    mcp_manager: Optional["McpManager"],
    agent_env: dict[str, str],
) -> LLMProvider:
    """Construct a fresh CLI for the given provider+model.

    Shared by the startup path and the AGENT_SETTINGS_CHANGE hot-swap path
    so both build identical instances. ``verify_cli()`` raises
    ``LLMNotFoundError`` if the binary isn't on PATH; an unknown provider
    name raises ``LLMNotFoundError`` here too, so the reactive loop can
    surface both failure modes uniformly.
    """
    from clawmeets.llm.base import LLMNotFoundError

    normalized = (provider or "claude").lower()
    if normalized == "openai":
        CodexCLI.verify_cli()
        return CodexCLI(model=model, agent_env=agent_env)
    if normalized == "gemini":
        GeminiCLI.verify_cli()
        return GeminiCLI(model=model, agent_env=agent_env)
    if normalized == "claude":
        ClaudeCLI.verify_cli()
        return ClaudeCLI(
            claude_plugin_dirs=plugin_dirs,
            mcp_manager=mcp_manager,
            agent_env=agent_env,
            model=model,
        )
    raise LLMNotFoundError(
        f"unknown llm_provider {provider!r} "
        f"(expected one of {_VALID_LLM_PROVIDERS})"
    )


def _build_initial_local_settings(
    llm_provider: Optional[str],
    llm_model: Optional[str],
    dwh_dir: Optional[str] = None,
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
    if dwh_dir:
        settings["dwh_dir"] = dwh_dir
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
    dwh_dir: Optional[str] = typer.Option(
        None, "--dwh-dir",
        help="Personal data-warehouse root for this agent (typically a network shared file system mount, e.g. /mnt/dwh). "
             "Written to card.json local_settings; rendered into the agent prompt.",
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

    # Re-register returns token=None so a server-side rotate doesn't silently
    # invalidate any running runner's in-memory credential. First-time
    # register always returns a token; write the file then. Card-only updates
    # (description/teams/capabilities) below still happen either way.
    if result.get("token"):
        cred_path.write_text(json.dumps(result, indent=2, default=str))
        typer.echo(f"Credentials saved to {cred_path}")
    else:
        typer.echo(f"Re-registered; existing credentials at {cred_path} preserved.")

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
        initial_local_settings = _build_initial_local_settings(llm_provider, llm_model, dwh_dir)
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
# knowledge-pack list / create / edit / delete / install / uninstall
# ---------------------------------------------------------------------------


def _read_file_content_arg(content_file: Optional[Path]) -> str:
    if content_file is None:
        return ""
    if not content_file.exists():
        typer.echo(f"Error: --content-file {content_file} does not exist", err=True)
        raise typer.Exit(1)
    return content_file.read_text()


def _read_file_b64_arg(content_file: Optional[Path]) -> str:
    """Read a file as raw bytes and return base64-encoded ASCII. Used for
    knowledge-pack uploads, which now treat every file as opaque bytes."""
    import base64 as _b64
    if content_file is None:
        return ""
    if not content_file.exists():
        typer.echo(f"Error: --content-file {content_file} does not exist", err=True)
        raise typer.Exit(1)
    return _b64.b64encode(content_file.read_bytes()).decode("ascii")


@knowledge_pack_app.command("list")
def knowledge_pack_list(
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """List every knowledge pack you own."""
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        resp = client.get(
            "/knowledge-packs",
            headers={"Authorization": f"Bearer {token}"},
        )
        packs = _ok(resp)
    if not packs:
        typer.echo("No knowledge packs. Create one with `clawmeets knowledge-pack create ...`.")
        return
    for pack in packs:
        files = pack.get("files") or {}
        total_bytes = 0
        for entry in files.values():
            b64 = (entry or {}).get("content_b64") if isinstance(entry, dict) else ""
            b64 = b64 or ""
            pad = 2 if b64.endswith("==") else (1 if b64.endswith("=") else 0)
            total_bytes += max(0, (len(b64) * 3 // 4) - pad)
        typer.echo(
            f"  {pack['slug']} — {pack.get('name', pack['slug'])} "
            f"({len(files)} file(s), {total_bytes}B)"
        )
        if pack.get("description"):
            typer.echo(f"      {pack['description']}")
        for fname in sorted(files.keys()):
            typer.echo(f"      - {fname} ({len(files[fname] or '')}B)")


@knowledge_pack_app.command("create")
def knowledge_pack_create(
    slug: str = typer.Argument(..., help="Slug (lowercase, hyphens/underscores allowed)"),
    name: str = typer.Option(..., "--name", help="Human-readable pack name"),
    description: str = typer.Option("", "--description", help="One-line trigger hint shown in the agent's KNOWLEDGE_PACKS.md index"),
    file: Optional[list[str]] = typer.Option(
        None, "--file",
        help="Seed a file in the new pack (repeatable). Format: '<path-in-pack>:<local-path>'. The path-in-pack may contain '/' for nested layout; the local file is read as bytes (text or binary).",
    ),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Create a new knowledge pack in your registry."""
    files: dict[str, str] = {}
    for spec in file or []:
        if ":" not in spec:
            typer.echo(f"Error: --file must be '<path-in-pack>:<local-path>', got {spec!r}", err=True)
            raise typer.Exit(1)
        fname, fpath = spec.split(":", 1)
        files[fname.strip()] = _read_file_b64_arg(Path(fpath.strip()))

    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        resp = client.post(
            "/knowledge-packs",
            json={
                "slug": slug,
                "name": name,
                "description": description,
                "files": files,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        pack = _ok(resp)
    typer.echo(f"Created knowledge pack '{pack['slug']}' ({len(pack.get('files') or {})} file(s)).")


@knowledge_pack_app.command("edit")
def knowledge_pack_edit(
    slug: str = typer.Argument(..., help="Slug to edit"),
    name: Optional[str] = typer.Option(None, "--name", help="New human-readable name"),
    description: Optional[str] = typer.Option(None, "--description", help="New one-line trigger hint"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Patch an existing pack's metadata (name / description). For file-level
    edits use ``knowledge-pack add-file`` / ``remove-file``.
    """
    payload: dict = {}
    if name is not None:
        payload["name"] = name
    if description is not None:
        payload["description"] = description
    if not payload:
        typer.echo("Error: pass at least one of --name, --description", err=True)
        raise typer.Exit(1)

    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        resp = client.put(
            f"/knowledge-packs/{slug}",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        pack = _ok(resp)
    typer.echo(f"Updated knowledge pack '{pack['slug']}' (updated_at={pack.get('updated_at')}).")


@knowledge_pack_app.command("add-file")
def knowledge_pack_add_file(
    slug: str = typer.Argument(..., help="Pack slug"),
    filepath: str = typer.Argument(..., help="Path inside the pack (e.g. tactics.md or notes/intro.md)"),
    content_file: Path = typer.Option(..., "--content-file", help="Local file to upload (text or binary)"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Add (or replace) a file inside a pack. Re-broadcasts to installed agents."""
    content_b64 = _read_file_b64_arg(content_file)
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        resp = client.put(
            f"/knowledge-packs/{slug}/files/{filepath}",
            json={"content_b64": content_b64},
            headers={"Authorization": f"Bearer {token}"},
        )
        pack = _ok(resp)
    typer.echo(
        f"Saved file '{filepath}' in pack '{pack['slug']}' "
        f"(pack now has {len(pack.get('files') or {})} file(s))."
    )


@knowledge_pack_app.command("remove-file")
def knowledge_pack_remove_file(
    slug: str = typer.Argument(..., help="Pack slug"),
    filepath: str = typer.Argument(..., help="Path to remove (e.g. tactics.md or notes/intro.md)"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Remove a file from a pack."""
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        resp = client.delete(
            f"/knowledge-packs/{slug}/files/{filepath}",
            headers={"Authorization": f"Bearer {token}"},
        )
        pack = _ok(resp)
    typer.echo(
        f"Removed file '{filepath}' from pack '{pack['slug']}' "
        f"({len(pack.get('files') or {})} file(s) remain)."
    )


@knowledge_pack_app.command("delete")
def knowledge_pack_delete(
    slug: str = typer.Argument(..., help="Slug to delete"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Delete a pack and uninstall it from every agent that has it."""
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        resp = client.delete(
            f"/knowledge-packs/{slug}",
            headers={"Authorization": f"Bearer {token}"},
        )
        result = _ok(resp)
    n = len(result.get("uninstalled_from") or [])
    typer.echo(f"Deleted '{slug}' (uninstalled from {n} agent(s)).")


@knowledge_pack_app.command("install")
def knowledge_pack_install(
    agent: str = typer.Argument(..., help="Agent name or id"),
    slugs: list[str] = typer.Argument(..., help="One or more pack slugs to install"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Install one or more knowledge packs on an agent."""
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        agent_id, agent_name = _resolve_agent_id(client, token, agent)
        resp = client.post(
            f"/agents/{agent_id}/knowledge-packs",
            json={"packs": slugs},
            headers={"Authorization": f"Bearer {token}"},
        )
        result = _ok(resp)
    added = result.get("added") or []
    if added:
        typer.echo(f"Installed on '{agent_name}': {', '.join(added)}")
    else:
        typer.echo(f"No new packs installed on '{agent_name}' (already present).")


@knowledge_pack_app.command("uninstall")
def knowledge_pack_uninstall(
    agent: str = typer.Argument(..., help="Agent name or id"),
    slug: str = typer.Argument(..., help="Pack slug to uninstall"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Uninstall a knowledge pack from an agent."""
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        agent_id, agent_name = _resolve_agent_id(client, token, agent)
        resp = client.delete(
            f"/agents/{agent_id}/knowledge-packs/{slug}",
            headers={"Authorization": f"Bearer {token}"},
        )
        _ok(resp)
    typer.echo(f"Uninstalled '{slug}' from '{agent_name}'.")


# ---------------------------------------------------------------------------
# agent list
# ---------------------------------------------------------------------------

@agent_app.command("set-dwh-dir")
def agent_set_dwh_dir(
    agent: str = typer.Argument(..., help="Agent name or id"),
    dwh_dir: str = typer.Argument(..., help="Personal data-warehouse root (e.g. /mnt/dwh). Pass empty string to clear."),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Set the dwh_dir on an agent's local_settings (in-place merge).

    Reads the current local_settings via GET /agents/{id}, sets/clears
    ``dwh_dir``, and PUTs back. Triggers AGENT_SETTINGS_CHANGE so a running
    runner picks it up on the next LLM invocation.
    """
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        agent_id, agent_name = _resolve_agent_id(client, token, agent)
        current = _ok(client.get(
            f"/agents/{agent_id}",
            headers={"Authorization": f"Bearer {token}"},
        ))
        local_settings = dict(current.get("local_settings") or {})
        if dwh_dir:
            local_settings["dwh_dir"] = dwh_dir
        else:
            local_settings.pop("dwh_dir", None)
        put_resp = client.put(
            f"/agents/{agent_id}",
            json={"local_settings": local_settings},
            headers={"Authorization": f"Bearer {token}"},
        )
        _ok(put_resp)
    if dwh_dir:
        typer.echo(f"Set dwh_dir={dwh_dir!r} on agent '{agent_name}'.")
    else:
        typer.echo(f"Cleared dwh_dir on agent '{agent_name}'.")


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
    dwh_dir: Optional[str] = typer.Option(
        None, "--dwh-dir",
        help="Personal data-warehouse root for the assistant (typically a network shared file system mount, e.g. /mnt/dwh).",
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
        initial_local_settings = _build_initial_local_settings(llm_provider, llm_model, dwh_dir)
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
    dwh_dir: Optional[str] = typer.Option(
        None, "--dwh-dir",
        help="Personal data-warehouse root for the assistant (typically a network shared file system mount, e.g. /mnt/dwh).",
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
        initial_local_settings = _build_initial_local_settings(llm_provider, llm_model, dwh_dir)
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
    # CLI flags (--knowledge-dir) serve as overrides for backward compat.
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
    # Identity exposed to every LLM subprocess so Bash-shelled `clawmeets ...`
    # commands inside the agent can authenticate as this agent without
    # re-resolving credentials. Read by project-skill invocations like
    # `clawmeets project create ... --token $CLAWMEETS_AGENT_TOKEN`.
    # CLAWMEETS_AGENT_DIR is the absolute path of this agent's root directory
    # (~/.clawmeets/agents/<name>-<id>/), used to anchor personal-skill-hub
    # writes — the LLM's cwd is the per-project sandbox, and knowledge_dir
    # can be configured anywhere, so neither is a reliable anchor.
    agent_env = {
        "CLAWMEETS_AGENT_ID": agent_id,
        "CLAWMEETS_AGENT_TOKEN": token,
        "CLAWMEETS_SERVER_URL": server_http,
        "CLAWMEETS_AGENT_DIR": str(agent_dir),
    }

    llm_provider_name = (local_settings.get("llm_provider") or "claude").lower()
    llm_model = local_settings.get("llm_model") or None

    # Factory closes over runner-scoped args (plugin dirs, MCP manager,
    # agent env). Shared by the startup path here and the reactive
    # loop's hot-swap path so both build identical CLI instances.
    def cli_factory(provider: str, model: Optional[str]) -> LLMProvider:
        return _build_llm_provider(
            provider,
            model,
            plugin_dirs=all_plugin_dirs,
            mcp_manager=mcp_manager,
            agent_env=agent_env,
        )

    try:
        cli = cli_factory(llm_provider_name, llm_model)
    except Exception as e:
        typer.echo(
            f"Error: failed to construct LLM provider "
            f"({llm_provider_name!r}, model={llm_model!r}): {e}",
            err=True,
        )
        raise typer.Exit(1)
    typer.echo(
        f"LLM provider: {llm_provider_name}"
        + (f" (model={llm_model})" if llm_model else "")
    )

    # Build knowledge_dirs list (e.g., knowledge bases)
    knowledge_dirs_list: list[Path] = []
    resolved = resolve_local_knowledge_dir(str(effective_knowledge_dir), user_config_dir) if effective_knowledge_dir else None
    if resolved is not None:
        knowledge_dirs_list.append(resolved)

    # Resolve dwh_dir (personal data warehouse root, typically network-shared).
    # None when unset — the prompt block is omitted in that case.
    raw_dwh_dir = local_settings.get("dwh_dir") or ""
    resolved_dwh_dir = resolve_local_dwh_dir(str(raw_dwh_dir), user_config_dir) if raw_dwh_dir else None

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
        dwh_dir=resolved_dwh_dir,
    )

    participant = Agent(id=agent_id, model_ctx=model_ctx)

    # Set up knowledge-pack manager (writes user-curated packs to
    # {agent_dir}/knowledge_packs/<slug>/; index lives at
    # {agent_dir}/memory/KNOWLEDGE_PACKS.md alongside the other
    # AUTHORITATIVE indexes)
    knowledge_pack_manager = KnowledgePackManager(model_ctx)

    # Build reactive control loop
    loop_obj = ReactiveControlLoop(
        participant=participant,
        client=client,
        model_ctx=model_ctx,
        extra_subscribers=[],
        skill_manager=skill_manager,
        mcp_manager=mcp_manager,
        knowledge_pack_manager=knowledge_pack_manager,
        user_config_dir=user_config_dir,
        cli_factory=cli_factory,
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

                # Sync installed knowledge packs from server (catch-up on connect/reconnect)
                await knowledge_pack_manager.sync_from_server(client, agent_id)

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
# Front Desk commands
# ---------------------------------------------------------------------------

@front_desk_app.command("ensure")
def front_desk_ensure(
    agent_full_name: str = typer.Argument(..., help="Agent's full registry name, e.g. chuswine-customer_support"),
    username: str = typer.Option(..., "-u", "--username", help="Your username (the requester)"),
    password: str = typer.Option(..., "-p", "--password", help="Your password"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
):
    """Ensure a Front Desk project exists for the named agent.

    Idempotent. Returns the existing project if one already exists; otherwise
    creates a new one named ``{your_username}-fd-{agent_short_name}`` on the
    agent owner's side. Prints the project name + id.

    Example:
        clawmeets front-desk ensure chuswine-customer_support -u chengtao -p ****
    """
    with _http(server) as client:
        resp = client.post("/auth/login", json={"username": username, "password": password})
        try:
            login_result = _ok(resp)
        except SystemExit:
            typer.echo("Error: Invalid username or password", err=True)
            raise typer.Exit(1)
        token = login_result["token"]
        resp = client.post(
            f"/me/front-desk/{agent_full_name}/ensure",
            headers={"Authorization": f"Bearer {token}"},
        )
        project = _ok(resp)
        typer.echo(f"{project['name']} (id={project['id']})")


@front_desk_app.command("send")
def front_desk_send(
    agent_full_name: str = typer.Argument(..., help="Agent's full registry name, e.g. chuswine-customer_support"),
    message: str = typer.Argument(..., help="Message content"),
    username: str = typer.Option(..., "-u", "--username", help="Your username (the requester)"),
    password: str = typer.Option(..., "-p", "--password", help="Your password"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
):
    """Send a message to a Front Desk channel's user-communication chatroom.

    Ensures the Front Desk project exists, then posts to user-communication.

    Example:
        clawmeets front-desk send chuswine-customer_support "Hi, can you help?" -u chengtao -p ****
    """
    with _http(server) as client:
        resp = client.post("/auth/login", json={"username": username, "password": password})
        try:
            login_result = _ok(resp)
        except SystemExit:
            typer.echo("Error: Invalid username or password", err=True)
            raise typer.Exit(1)
        token = login_result["token"]

        resp = client.post(
            f"/me/front-desk/{agent_full_name}/ensure",
            headers={"Authorization": f"Bearer {token}"},
        )
        project = _ok(resp)

        resp = client.post(
            f"/projects/{project['id']}/chatrooms/user-communication/user-message",
            json={"content": message},
            headers={"Authorization": f"Bearer {token}"},
        )
        _ok(resp)
        typer.echo(f"Sent to {project['name']}")


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


@mcp_app.command("install")
def mcp_install(
    agent: str = typer.Argument(..., help="Agent name or id"),
    mcps: List[str] = typer.Argument(..., help="One or more MCP names to install"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Install one or more MCP servers on an agent."""
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        agent_id, agent_name = _resolve_agent_id(client, token, agent)
        resp = client.post(
            f"/agents/{agent_id}/mcps",
            json={"mcps": mcps},
            headers={"Authorization": f"Bearer {token}"},
        )
        result = _ok(resp)
    added = result.get("added") or []
    if added:
        typer.echo(f"Installed on '{agent_name}': {', '.join(added)}")
    else:
        typer.echo(f"No new MCPs installed on '{agent_name}' (already present).")


@mcp_app.command("set-config")
def mcp_set_config(
    agent: str = typer.Argument(..., help="Agent name or id"),
    mcp_name: str = typer.Argument(..., help="MCP server name"),
    config_file: Path = typer.Argument(..., help="JSON file with the config payload"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Set the per-agent config for an MCP server (uploads JSON from a file).

    Persists to card.json.local_settings.mcp_configs[mcp_name] and broadcasts
    AGENT_SETTINGS_CHANGE so the runner writes through to
    {agent_dir}/mcp-hub/configs/<mcp_name>.json.
    """
    if not config_file.is_file():
        typer.echo(f"Error: config file not found: {config_file}", err=True)
        raise typer.Exit(1)
    try:
        payload = json.loads(config_file.read_text())
    except json.JSONDecodeError as e:
        typer.echo(f"Error: {config_file} is not valid JSON: {e}", err=True)
        raise typer.Exit(1)
    if not isinstance(payload, dict):
        typer.echo(
            f"Error: config root must be a JSON object, got {type(payload).__name__}",
            err=True,
        )
        raise typer.Exit(1)

    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        agent_id, agent_name = _resolve_agent_id(client, token, agent)
        resp = client.put(
            f"/agents/{agent_id}/mcps/{mcp_name}/config",
            json={"config": payload},
            headers={"Authorization": f"Bearer {token}"},
        )
        _ok(resp)
    typer.echo(f"Set config for {mcp_name!r} on '{agent_name}'.")



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

_REFERENCES_MARKER = "<!-- clawmeets:references-trigger -->"


def _list_reference_files(kdir: Path) -> list[Path]:
    """List user-pre-seeded reference files under a knowledge_dir.

    Returns sorted relative paths (relative to kdir), excluding:
      - USER.md, REFERENCES.md (agent-authored memory / index)
      - CLAUDE.md, README.md (Claude Code instructions / human-facing readme)
      - learnings/ (subtree, agent-authored)
      - skills/ (subtree, Claude Code plugin slash-commands)
      - config/ (subtree, runner config)
      - dotfiles and __pycache__

    Follows symlinked directories so shared knowledge trees (e.g.
    `Knowledge/Core -> ../../shared/Knowledge/Core`) are walked. Tracks
    realpath of visited directories to avoid cycles.
    """
    if not kdir.exists() or not kdir.is_dir():
        return []
    excluded_top_dirs = {"learnings", "skills", "config"}
    excluded_files = {"USER.md", "REFERENCES.md", "CLAUDE.md", "README.md"}
    out: list[Path] = []
    seen_real_dirs: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(kdir, followlinks=True):
        # Cycle protection: if a symlink loops back to an already-visited
        # real directory, prune and continue.
        real = os.path.realpath(dirpath)
        if real in seen_real_dirs:
            dirnames[:] = []
            continue
        seen_real_dirs.add(real)

        rel_dir = Path(os.path.relpath(dirpath, kdir))

        # Prune subdirectories in-place: dotdirs, __pycache__, and
        # excluded top-level dirs.
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".")
            and d != "__pycache__"
            and not (rel_dir == Path(".") and d in excluded_top_dirs)
        ]

        for fn in filenames:
            if fn in excluded_files:
                continue
            if fn.startswith("."):
                continue
            rel_file = rel_dir / fn if rel_dir != Path(".") else Path(fn)
            out.append(rel_file)

    return sorted(out)


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


def _bootstrap_resolve(
    client: httpx.Client,
    user_jwt: str,
    username: str,
    data_dir_p: Path,
) -> tuple[str, str, Path, Path, dict, Optional[str]]:
    """Resolve the bits both bootstrap subcommands need from the server.

    Returns (assistant_id, assistant_name, assistant_dir, assistant_kdir,
    dm_project, assistant_token).

    Exits via typer.Exit on any failure. Caller has already injected the
    user JWT into client.headers and refreshed the token if needed.
    """
    agents_dir = data_dir_p / "agents"
    user_config_dir = data_dir_p / "config" / username

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

    dm_project = _find_dm_project(client, username)
    if not dm_project:
        typer.echo(f"Error: DM project DM-{username} not found on server.", err=True)
        raise typer.Exit(1)

    assistant_token: Optional[str] = None
    cred_path = assistant_dir / "credential.json"
    if cred_path.exists():
        try:
            assistant_token = json.loads(cred_path.read_text()).get("token")
        except (json.JSONDecodeError, OSError):
            pass

    return assistant_id, assistant_name, assistant_dir, assistant_kdir, dm_project, assistant_token


def _bootstrap_session_setup(
    data_dir: Path,
    server: Optional[str],
    username: Optional[str],
) -> tuple[str, str, str, Path]:
    """Resolve server URL, user JWT, username, and data_dir_p. Exits on failure."""
    server_url, user_jwt = _resolve_user_session(data_dir, None, server, as_user=username)
    if username is None:
        from clawmeets.cli_lifecycle import get_current_user
        username = get_current_user(Path(data_dir).expanduser())
    if not username:
        typer.echo("Error: no username known. Pass -u or run `clawmeets user login --save` first.", err=True)
        raise typer.Exit(1)
    data_dir_p = Path(data_dir).expanduser()
    user_jwt = _ensure_fresh_user_token(server_url, data_dir_p, username, user_jwt)
    return server_url, user_jwt, username, data_dir_p


@bootstrap_app.command("references")
def bootstrap_references(
    agents: List[str] = typer.Option(
        [], "--agent",
        help="Agent name (full or short, e.g. 'marketer' or 'chengtao-marketer'). Repeatable; omit to index all owned agents incl. assistant.",
    ),
    force: bool = typer.Option(False, "--force", help="Re-trigger agents whose REFERENCES.md already exists (skill still gates the actual overwrite)"),
    username: Optional[str] = typer.Option(None, "-u", "--username", help="Username (default: current saved session)"),
    password: Optional[str] = typer.Option(None, "-p", "--password", help="Password (default: from saved session)"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Build REFERENCES.md from user-pre-seeded files in each agent's knowledge_dir.

    For each owned agent (or just the named ones), lists files under the agent's
    knowledge_dir (excluding USER.md, REFERENCES.md, learnings/, dotfiles) and
    posts a references-trigger DM with the file paths. The agent's
    /clawmeets:references skill reads each file and writes a one-line
    "when to invoke" entry per file into knowledge_dir/REFERENCES.md.

    Idempotent: skips agents whose REFERENCES.md already exists. Use --force
    to re-trigger. Skips agents with no reference files (no LLM call).

    Re-run after adding or removing reference files to refresh the index.
    """
    server_url, user_jwt, username, data_dir_p = _bootstrap_session_setup(data_dir, server, username)
    agents_dir = data_dir_p / "agents"
    user_config_dir = data_dir_p / "config" / username

    with _http(server_url) as client:
        client.headers.update({"Authorization": f"Bearer {user_jwt}"})
        (
            _assistant_id,
            _assistant_name,
            _assistant_dir,
            _assistant_kdir,
            dm_project,
            assistant_token,
        ) = _bootstrap_resolve(client, user_jwt, username, data_dir_p)

        all_agents = _fetch_owned_agents(client, user_jwt)

        if agents:
            from clawmeets.utils.agent_namespace import short_name

            def _matches(card: dict, needle: str) -> bool:
                name = card.get("name") or ""
                short = short_name(name, username)
                lower_needle = needle.lower()
                return (
                    name == needle
                    or name.lower() == lower_needle
                    or short == needle
                    or short.lower() == lower_needle
                )

            matched: list[dict] = []
            unmatched: list[str] = []
            for needle in agents:
                hits = [a for a in all_agents if _matches(a, needle)]
                if not hits:
                    unmatched.append(needle)
                else:
                    matched.extend(hits)
            if unmatched:
                typer.echo(
                    f"Error: no owned agent matches: {', '.join(repr(u) for u in unmatched)}. "
                    f"Tried full name and short name (with '{username}-' prefix stripped).",
                    err=True,
                )
                raise typer.Exit(1)
            seen: set[str] = set()
            targets = [a for a in matched if not (a["id"] in seen or seen.add(a["id"]))]
        else:
            targets = all_agents

        if not targets:
            typer.echo("[bootstrap references] no agents to index. Done.")
            return

        dispatched = 0
        skipped = 0
        failed = 0
        for a in targets:
            a_id = a["id"]
            a_name = a["name"]
            a_dir = _find_agent_dir_by_id(agents_dir, a_id)
            if a_dir is None:
                typer.echo(f"  [{a_name}] no local dir — skipping (was this agent registered locally?)")
                skipped += 1
                continue
            a_kdir = _resolve_agent_knowledge_dir(a, a_dir, user_config_dir)
            if a_kdir is None:
                typer.echo(f"  [{a_name}] no knowledge_dir set (web UI agent settings) — skipping")
                skipped += 1
                continue
            references_path = a_dir / "memory" / "REFERENCES.md"
            if references_path.exists() and not force:
                typer.echo(f"  [{a_name}] skipped — REFERENCES.md already exists")
                skipped += 1
                continue
            files = _list_reference_files(a_kdir)
            if not files:
                typer.echo(f"  [{a_name}] no reference files to index — skipping")
                skipped += 1
                continue

            # Inline ABSOLUTE paths in the trigger so the agent can pass them
            # verbatim to Read and embed them in REFERENCES.md (its Read tool
            # resolves absolute paths regardless of working directory).
            file_lines = "\n".join(f"- {a_kdir / p}" for p in files)
            body = (
                f"{_REFERENCES_MARKER}\n\n"
                "You're being bootstrapped to index reference files. Build "
                "REFERENCES.md per the /clawmeets:references skill.\n\n"
                "== REFERENCE FILES (absolute paths) ==\n"
                f"{file_lines}\n"
            )
            if not _post_bootstrap_dm(client, user_jwt, assistant_token, dm_project["id"], a_name, body):
                failed += 1
                continue
            typer.echo(f"  [{a_name}] dispatched ({len(files)} files).")
            dispatched += 1

        typer.echo(
            f"\n[bootstrap references] summary: dispatched={dispatched} skipped={skipped} failed={failed}."
        )
        if dispatched:
            typer.echo(
                "  Each agent takes ~1–2 minutes to write REFERENCES.md. "
                "Watch the chat UI for the agent's reply."
            )


@bootstrap_app.command("browser")
def bootstrap_browser(
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Install Chromium for the playwright-browser skill (one-time per machine).

    Verifies Node.js >= 20 (prints a platform-specific install hint and exits 1
    if missing), then runs `npx playwright install chromium` (~150 MB download;
    Playwright skips browsers that are already cached). On Linux also runs
    `npx playwright install-deps chromium`. Touches a marker file at
    {data_dir}/.playwright_bootstrapped so the skill's preflight can answer
    instantly.

    Idempotent: safe to re-run any time. Browsers are global state on the
    runner machine; auth state stays per-agent under each agent's
    personal-skill-hub/_playwright/storage/.
    """
    import platform
    import shutil
    import subprocess
    import sys

    typer.echo("[bootstrap browser] checking Node.js…")
    node_bin = shutil.which("node")
    if node_bin is None:
        typer.echo("Error: Node.js not found on PATH.", err=True)
        system = platform.system()
        if system == "Darwin":
            typer.echo("  Install: brew install node", err=True)
        elif system == "Linux":
            typer.echo(
                "  Install: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && "
                "sudo apt install -y nodejs",
                err=True,
            )
        else:
            typer.echo("  Install: https://nodejs.org/en/download/", err=True)
        raise typer.Exit(1)

    try:
        version_proc = subprocess.run(
            [node_bin, "--version"], capture_output=True, text=True, timeout=10
        )
    except subprocess.TimeoutExpired:
        typer.echo("Error: `node --version` timed out after 10s.", err=True)
        raise typer.Exit(1)
    raw_version = (version_proc.stdout or "").strip().lstrip("v")
    try:
        major = int(raw_version.split(".", 1)[0])
    except ValueError:
        typer.echo(
            f"Error: could not parse Node version from {version_proc.stdout!r}.",
            err=True,
        )
        raise typer.Exit(1)
    if major < 20:
        typer.echo(
            f"Error: Node.js >= 20 required (found v{raw_version}). "
            "Upgrade and re-run.",
            err=True,
        )
        raise typer.Exit(1)
    typer.echo(f"  Node.js v{raw_version} OK")

    npx_bin = shutil.which("npx")
    if npx_bin is None:
        typer.echo("Error: `npx` not on PATH (ships with Node.js).", err=True)
        raise typer.Exit(1)

    typer.echo("[bootstrap browser] installing Chromium via npx playwright install chromium…")
    typer.echo("  (~150 MB download on first run; subsequent runs are instant)")
    install_proc = subprocess.run(
        [npx_bin, "--yes", "playwright", "install", "chromium"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if install_proc.returncode != 0:
        typer.echo("Error: `playwright install chromium` failed.", err=True)
        raise typer.Exit(install_proc.returncode)

    if platform.system() == "Linux":
        typer.echo("[bootstrap browser] installing system libs (Linux only)…")
        deps_proc = subprocess.run(
            [npx_bin, "--yes", "playwright", "install-deps", "chromium"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if deps_proc.returncode != 0:
            typer.echo(
                "Warning: `playwright install-deps chromium` failed; "
                "headed browser may not start. Re-run with sudo if needed.",
                err=True,
            )

    data_dir_p = Path(data_dir).expanduser()
    data_dir_p.mkdir(parents=True, exist_ok=True)
    marker = data_dir_p / ".playwright_bootstrapped"
    marker.touch()
    typer.echo(f"[bootstrap browser] done. Marker: {marker}")
    typer.echo(
        "  Install the playwright-browser skill on any agent that needs browser "
        "automation (Agent settings → Skills → playwright-browser)."
    )
