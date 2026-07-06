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
import base64
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
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
from clawmeets.cli_lifecycle import (
    clear_user_token,
    get_current_user,
    get_user_config_path,
    save_user_session,
)
from clawmeets.models.chat_message import ChatMessage
from clawmeets.utils.file_io import FileUtil
from clawmeets.utils.notification_center import NotificationCenter
from clawmeets.llm.base import LLMNotFoundError, LLMProvider
from clawmeets.llm.claude_cli import ClaudeCLI
from clawmeets.llm.codex_cli import CodexCLI
from clawmeets.llm.gemini_cli import GeminiCLI
from clawmeets.llm.opencode_cli import OpenCodeCLI
from clawmeets.models.context import ModelContext
from clawmeets.models.agent import Agent
from clawmeets.models.user import User, NotificationConfig
from clawmeets.sync.console_subscriber import ConsoleOutputSubscriber, ConsoleConfig
from clawmeets.runner.reactive_loop import ReactiveControlLoop
from clawmeets.runner.mcp_manager import McpManager
from clawmeets.runner.knowledge_pack_manager import KnowledgePackManager
from clawmeets.runner.references_index import build_references_index
from clawmeets.runner.personal_skill_manager import PersonalSkillManager
from clawmeets.runner.skill_manager import SkillManager
from clawmeets.runner.system_skill_manager import SystemSkillManager

logger = logging.getLogger("clawmeets")

# Backward compatibility alias
AgentRegistrationResult = AgentRegistrationResponse

# Sub-command groups
agent_app = typer.Typer(help="Agent commands", no_args_is_help=True)
user_app  = typer.Typer(help="User commands",  no_args_is_help=True)
dm_app    = typer.Typer(help="Direct message commands", no_args_is_help=True)
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
        "One-time machine-level setup tasks. Currently: `browser` installs "
        "Chromium for the playwright-browser skill."
    ),
    no_args_is_help=True,
)
assistant_app = typer.Typer(
    help="Create and manage your personal assistant agent ({username}-assistant).",
    no_args_is_help=True,
)
agent_team_app = typer.Typer(
    help="Bulk-register a team of worker agents from a setup.json template.",
    no_args_is_help=True,
)


def _default_user_teams_from_env() -> list[str]:
    """Parse $CLAWMEETS_AGENT_TEAMS into a list (comma-separated). Empty list
    if unset. Used as the default for `clawmeets agent register --team` when
    no flags are passed.
    """
    raw = os.environ.get("CLAWMEETS_AGENT_TEAMS", "")
    return [t.strip() for t in raw.split(",") if t.strip()]


def _detect_clawmeets_plugin_dir() -> str:
    """Locate the bundled `plugins/clawmeets/` directory.

    Two cases, checked in order:

    1. **Pip-installed (also uv tool / pipx)** — the runner ships the
       plugin at ``clawmeets/_clawmeets_plugin/`` (see
       ``scripts/build-runner-package.sh`` + the ``package-data`` entry
       in ``packaging/runner/pyproject.toml``).
    2. **Source-repo dev** — the package lives in a checkout that also
       has ``plugins/clawmeets/`` as a sibling of the ``clawmeets/``
       module dir. Walk up from ``__file__`` looking for the
       ``.claude-plugin/plugin.json`` marker.

    Returns "" if neither matches.
    """
    here = Path(__file__).resolve()
    bundled = here.parent / "_clawmeets_plugin" / ".claude-plugin" / "plugin.json"
    if bundled.exists():
        return str(bundled.parent.parent)
    for parent in here.parents:
        candidate = parent / "plugins" / "clawmeets" / ".claude-plugin" / "plugin.json"
        if candidate.exists():
            return str(candidate.parent.parent)
    return ""


def _fetch_setup_template(url: str) -> dict:
    """Fetch a setup.json template. Accepts HTTP(S) URLs, file:// URLs, and
    plain filesystem paths (useful for iterating on templates locally before
    publishing them)."""
    import urllib.parse  # local import; rarely hot

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


def _host_iana_timezone() -> str:
    """Return the host machine's IANA timezone name, e.g. 'America/Los_Angeles'.

    Falls back to 'UTC' if tzlocal can't resolve it.
    """
    try:
        import tzlocal  # type: ignore[import-not-found]
        return str(tzlocal.get_localzone_name() or "UTC")
    except Exception:
        return "UTC"


def _daily_at_to_cron(daily_at: str) -> str:
    """Convert an `HH:MM` string into a 5-field cron expression `M H * * *`.

    Exits with code 1 on invalid input (we want the CLI to fail loudly, not
    silently produce a schedule that fires at midnight).
    """
    raw = daily_at.strip()
    parts = raw.split(":")
    if len(parts) != 2:
        typer.echo(f"Error: --reflect-daily-at expects 'HH:MM' (got {daily_at!r})", err=True)
        raise typer.Exit(1)
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        typer.echo(f"Error: --reflect-daily-at expects 'HH:MM' (got {daily_at!r})", err=True)
        raise typer.Exit(1)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        typer.echo(
            f"Error: --reflect-daily-at hour 0..23 and minute 0..59 (got {daily_at!r})",
            err=True,
        )
        raise typer.Exit(1)
    return f"{minute} {hour} * * *"


# ---------------------------------------------------------------------------
# Global options (env-var defaults)
# ---------------------------------------------------------------------------

DEFAULT_SERVER = os.environ.get("CLAWMEETS_SERVER_URL", "https://clawmeets.ai")
DEFAULT_DATA_DIR = os.environ.get("CLAWMEETS_DATA_DIR", str(Path.home() / ".clawmeets"))


def _server_url(server: str) -> str:
    return server.rstrip("/")


def _env_identity_headers() -> dict[str, str]:
    """Per-process agent identity injected by the runner, if present.

    Mirrors the runner's own httpx client (``Authorization: Bearer`` +
    ``X-Agent-ID``) so any ``clawmeets`` command shelled inside an agent
    authenticates as that agent's owner — unambiguous per process even when
    multiple runners share a host. Empty when not running inside a runner
    (interactive human use), where the saved session applies instead.
    """
    token = os.environ.get("CLAWMEETS_AGENT_TOKEN")
    agent_id = os.environ.get("CLAWMEETS_AGENT_ID")
    if token and agent_id:
        return {"Authorization": f"Bearer {token}", "X-Agent-ID": agent_id}
    return {}


def _http(server: str) -> httpx.Client:
    # Default headers carry the per-process agent identity (if any). Commands
    # that set their own Authorization per request override it; the server's
    # resolve_viewer only consults X-Agent-ID as a post-JWT fallback, so the
    # extra default header is harmless on every route.
    return httpx.Client(
        base_url=_server_url(server), timeout=30, headers=_env_identity_headers()
    )


def _ok(resp: httpx.Response) -> dict:
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        typer.echo(f"Error {resp.status_code}: {resp.text}", err=True)
        raise typer.Exit(1) from e
    return resp.json()


def _print_json(data: dict | list) -> None:
    typer.echo(json.dumps(data, indent=2, default=str))


# Provider values. Bare names shell the Code CLI binary (max-fidelity local);
# the ``-api`` suffix selects the in-process BYO-key provider (no binary,
# runner-portable). One field encodes both the model family and the execution
# model.
_VALID_LLM_PROVIDERS = (
    "claude", "openai", "gemini", "opencode",
    "claude-api", "openai-api", "gemini-api", "openrouter-api",
)

# Standard env vars a BYO-key API provider falls back to when card.json
# carries no literal llm_api_key — convenient for hosted runners that inject
# the key as an environment variable instead of persisting it on the card.
_API_KEY_ENV: dict[str, tuple[str, ...]] = {
    "claude": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def _llm_base_provider(provider: Optional[str]) -> str:
    """Strip the ``-api`` suffix to get the bare model family (claude/openai/gemini)."""
    p = (provider or "claude").lower()
    return p[: -len("-api")] if p.endswith("-api") else p


def _is_api_provider(provider: Optional[str]) -> bool:
    return (provider or "").lower().endswith("-api")


def _resolve_api_key(provider: str, settings: dict) -> Optional[str]:
    """Resolve the BYO key for a ``*-api`` provider.

    Priority: explicit ``local_settings.llm_api_key`` (set via ``--llm-api-key``
    or the settings UI) → provider-standard env var (``ANTHROPIC_API_KEY`` /
    ``OPENAI_API_KEY`` / ``GEMINI_API_KEY`` | ``GOOGLE_API_KEY``). The card
    value always wins over the env var. ``provider`` may carry the ``-api``
    suffix (stripped for the env lookup).
    """
    key = settings.get("llm_api_key")
    if key:
        return str(key)
    for env_name in _API_KEY_ENV.get(_llm_base_provider(provider), ()):
        val = os.environ.get(env_name)
        if val:
            return val
    return None



# Per-turn caps an agent's card may carry in ``local_settings`` to tune the
# in-process ``-api`` harness (the eval "profiles" — web_max_uses / max_requests
# / max_tokens, plus the coordinator-only web clamp). Only forwarded to
# ``ApiLLMProvider``; CLI providers ignore them (their caps are internal to the
# binary). Absent keys ⇒ provider defaults.
_API_CAP_KEYS = (
    "max_requests", "web_max_uses", "coordinator_web_max_uses", "web_fetch_max_uses",
    "max_tokens", "enable_web", "max_total_tokens", "reasoning_effort",
)


def _api_caps_from_settings(settings: dict) -> dict:
    """Extract the optional ``-api`` cap overrides from local_settings.

    Returns only the keys that are present and non-None, so missing caps fall
    through to ``ApiLLMProvider``'s own defaults rather than overriding them.
    """
    return {
        k: settings[k]
        for k in _API_CAP_KEYS
        if settings.get(k) is not None
    }


def _build_llm_provider(
    provider: str,
    model: Optional[str],
    *,
    plugin_dirs: list[Path],
    skill_dirs: list[Path],
    agent_env: dict[str, str],
    api_key: Optional[str] = None,
    api_caps: Optional[dict] = None,
    base_url: Optional[str] = None,
) -> LLMProvider:
    """Construct a fresh CLI for the given provider+model.

    Shared by the startup path and the AGENT_SETTINGS_CHANGE hot-swap path
    so both build identical instances. ``verify_cli()`` raises
    ``LLMNotFoundError`` if the binary isn't on PATH; an unknown provider
    name raises ``LLMNotFoundError`` here too, so the reactive loop can
    surface both failure modes uniformly.

    ``plugin_dirs`` is Claude-specific (drives ``--plugin-dir`` and the
    slash-command surface that backs ``/clawmeets:*`` — bundled at
    ``plugins/clawmeets/``). ``skill_dirs`` is the set of skill content
    roots (skill-hub + personal-skill-hub) discoverable via the INDEX.md
    files the prompt builder advertises — Gemini sandboxes file reads so
    it needs them in ``--include-directories``. Codex inherits the agent
    dir at the cwd level and reads INDEX/SKILL paths directly.

    MCP servers are surfaced through ``model_ctx.mcp_dist_dir`` at
    ``invoke()`` time, not through the constructor — providers no longer
    hold a reference to McpManager.
    """
    normalized = (provider or "claude").lower()

    # ``*-api`` provider: in-process, BYO-key, no CLI binary (runner-portable).
    # One Pydantic-AI-backed provider drives all three model families; the bare
    # name maps to its API SDK identity.
    if _is_api_provider(normalized):
        from clawmeets.llm.api_provider import ApiLLMProvider

        api_name = {
            "claude": "anthropic",
            "openai": "openai",
            "gemini": "google",
            "openrouter": "openrouter",
        }.get(_llm_base_provider(normalized))
        if api_name is None:
            raise LLMNotFoundError(
                f"unknown llm_provider {provider!r} "
                f"(expected one of {_VALID_LLM_PROVIDERS})"
            )
        ApiLLMProvider.verify_cli()
        return ApiLLMProvider(
            provider=api_name,
            api_key=api_key or "",
            agent_env=agent_env,
            model=model,
            # Custom OpenAI-compatible endpoint (ollama /v1, vLLM, …) — only the
            # openai family consults it; None ⇒ the hosted SDK default.
            base_url=base_url,
            # Optional per-turn caps (eval "profiles"); None → provider defaults.
            # CLI providers below ignore these (their caps are internal).
            **(api_caps or {}),
        )

    if normalized == "openai":
        CodexCLI.verify_cli()
        return CodexCLI(model=model, agent_env=agent_env)
    if normalized == "gemini":
        GeminiCLI.verify_cli()
        return GeminiCLI(model=model, agent_env=agent_env, skill_dirs=skill_dirs)
    if normalized == "opencode":
        OpenCodeCLI.verify_cli()
        return OpenCodeCLI(model=model, agent_env=agent_env, skill_dirs=skill_dirs)
    if normalized == "claude":
        ClaudeCLI.verify_cli()
        return ClaudeCLI(
            claude_plugin_dirs=plugin_dirs,
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
    llm_api_key: Optional[str] = None,
    git_url: Optional[str] = None,
    git_base_branch: Optional[str] = None,
    llm_base_url: Optional[str] = None,
) -> dict:
    """Build the local_settings block for a freshly generated card.json.

    Exits with code 1 if llm_provider is not a supported value. The ``-api``
    provider variants (e.g. ``claude-api``) select the in-process BYO-key
    provider; ``llm_api_key`` is the key for those (wins over env vars).

    ``git_url`` binds this agent to a git repo. It is surfaced to the LLM as
    ``$CLAWMEETS_AGENT_GIT_URL`` and drives the git-workflow skill; ``git_base_branch``
    (optional) overrides the branch new work is cut from (default: repo default).
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
    if llm_api_key:
        settings["llm_api_key"] = llm_api_key
    if llm_base_url:
        settings["llm_base_url"] = llm_base_url
    if dwh_dir:
        settings["dwh_dir"] = dwh_dir
    if git_url:
        settings["git_url"] = git_url
    if git_base_branch:
        settings["git_base_branch"] = git_base_branch
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
    discoverable: Optional[bool] = typer.Option(None, "--discoverable/--no-discoverable", help="Publish agent in the public /agents registry (default: private)"),
    capabilities: Optional[str] = typer.Option(None, "--capabilities", "-c", help="Comma-separated list of capabilities"),
    from_card: Optional[Path] = typer.Option(None, "--from-card", help="Path to card.json to register from"),
    as_user: Optional[str] = typer.Option(
        None, "--as-user",
        help="Username whose saved JWT to use (defaults to current_user).",
    ),
    llm_provider: Optional[str] = typer.Option(
        None, "--llm-provider",
        help="LLM backend for this agent. Bare names shell the Code CLI: "
             "'claude' (default), 'openai', 'gemini', 'opencode' (opencode.ai — "
             "Zen gateway incl. free models; set --llm-model to a provider/model "
             "slug like 'opencode/deepseek-v4-flash-free'). The '-api' variants "
             "run in-process with a BYO key (no binary): 'claude-api', "
             "'openai-api', 'gemini-api', 'openrouter-api' (OpenAI-compatible "
             "gateway — set --llm-model to an OpenRouter slug). Written to "
             "card.json local_settings.",
    ),
    llm_model: Optional[str] = typer.Option(
        None, "--llm-model",
        help="Provider-specific model name (e.g. 'o3' for Codex, 'gemini-2.5-pro' for Gemini). Written to card.json local_settings.",
    ),
    llm_api_key: Optional[str] = typer.Option(
        None, "--llm-api-key", envvar="CLAWMEETS_LLM_API_KEY",
        help="API key for a '-api' provider (BYO-key). Persisted to card.json "
             "local_settings and takes priority over the provider's standard env "
             "var (ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY), which is "
             "the fallback when this is omitted.",
    ),
    llm_base_url: Optional[str] = typer.Option(
        None, "--llm-base-url",
        help="Custom OpenAI-compatible endpoint for the 'openai-api' provider, "
             "e.g. a local ollama server at 'http://localhost:11434/v1' (also "
             "vLLM / LM Studio). Routes a LOCAL model through the in-process "
             "provider with schema-enforced output; --llm-api-key is optional "
             "(ollama ignores it). Written to card.json local_settings.",
    ),
    dwh_dir: Optional[str] = typer.Option(
        None, "--dwh-dir",
        help="Personal data-warehouse root for this agent (typically a network shared file system mount, e.g. /mnt/dwh). "
             "Written to card.json local_settings; rendered into the agent prompt.",
    ),
    git_url: Optional[str] = typer.Option(
        None, "--git-url", envvar="CLAWMEETS_AGENT_GIT_URL",
        help="Bind this agent to a git repo (URL or path). Surfaced to the LLM as "
             "$CLAWMEETS_AGENT_GIT_URL; the git-workflow skill clones it into the "
             "sandbox, branches per request, commits and pushes. Written to card.json "
             "local_settings.",
    ),
    git_base_branch: Optional[str] = typer.Option(
        None, "--git-base-branch",
        help="Branch new work branches are cut from (default: the repo's default branch). "
             "Surfaced as $CLAWMEETS_AGENT_GIT_BASE_BRANCH. Written to card.json local_settings.",
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
    # Auto-fill --token and --server: --token explicit > env
    # ($CLAWMEETS_ASSISTANT_TOKEN / $CLAWMEETS_USER_TOKEN — set when the
    # user's assistant agent shells this command) > saved user session.
    server, token = _resolve_user_session(data_dir, token, server, as_user=as_user)

    # Load from card.json if provided
    caps_list = []
    # Private by default — publishing to the registry is an explicit opt-in
    # (pass --discoverable, or set discoverable_through_registry in the card).
    card_discoverable = False
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
        card_discoverable = card_data.get("discoverable_through_registry", False)

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
        initial_local_settings = _build_initial_local_settings(
            llm_provider, llm_model, dwh_dir, llm_api_key, git_url, git_base_branch,
            llm_base_url=llm_base_url,
        )
        if initial_local_settings:
            card["local_settings"] = initial_local_settings
        card_path = agent_work_dir / "card.json"
        card_path.write_text(json.dumps(card, indent=2, default=str))
        typer.echo(f"Card saved to {card_path}")


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

    Resolution order for the token:
      1. --token (explicit_token)
      2. $CLAWMEETS_ASSISTANT_TOKEN env var (set by the runner when the
         user's `{username}-assistant` agent shells these admin commands)
      3. $CLAWMEETS_USER_TOKEN env var (escape hatch for scripted callers)
      4. Saved user config (created by `clawmeets user login --save`)
    """
    token = explicit_token
    server = explicit_server
    if not token:
        token = os.environ.get("CLAWMEETS_ASSISTANT_TOKEN") or os.environ.get(
            "CLAWMEETS_USER_TOKEN"
        )
    if not server:
        server = os.environ.get("CLAWMEETS_SERVER_URL") or None
    if not token or not server:
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


def _resolve_session_for_config(
    data_dir: Path,
    explicit_token: Optional[str],
    explicit_server: Optional[str],
    agent_ref: str,
) -> tuple[str, str, str, str]:
    """Resolve (server_url, token, agent_id, agent_name) for *-set-config commands.

    The set-config endpoints (PUT /agents/{id}/{skills,mcps}/{name}/config)
    accept the agent's own self-token in addition to owner/assistant
    credentials, so a worker that's been asked to mutate its own config can
    route the change through the canonical CLI rather than editing
    `{agent_dir}/skill-hub/configs/<name>.json` directly.

    When `agent_ref == "self"`, use the agent-env vars injected by the
    runner (`CLAWMEETS_AGENT_ID` / `CLAWMEETS_AGENT_TOKEN` /
    `CLAWMEETS_SERVER_URL`, see `cli_runner.py` agent_env block). The
    agent's name isn't injected, so we echo the id back in success
    messages — the runner-side write-through is keyed on id anyway.

    Otherwise fall back to `_resolve_user_session` + `_resolve_agent_id`
    (the user/assistant session path used by every other admin command).
    """
    if agent_ref == "self":
        agent_id = os.environ.get("CLAWMEETS_AGENT_ID")
        agent_token = explicit_token or os.environ.get("CLAWMEETS_AGENT_TOKEN")
        server = explicit_server or os.environ.get("CLAWMEETS_SERVER_URL")
        if not (agent_id and agent_token and server):
            typer.echo(
                "Error: 'self' requires CLAWMEETS_AGENT_ID, CLAWMEETS_AGENT_TOKEN, "
                "and CLAWMEETS_SERVER_URL in the environment (set by the runner when "
                "spawning the agent subprocess). Pass an explicit <agent> name instead.",
                err=True,
            )
            raise typer.Exit(1)
        return _server_url(server), agent_token, agent_id, agent_id

    server_url, token = _resolve_user_session(data_dir, explicit_token, explicit_server)
    with _http(server_url) as client:
        agent_id, agent_name = _resolve_agent_id(client, token, agent_ref)
    return server_url, token, agent_id, agent_name


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


def _resolve_dm_session(
    data_dir: Path,
    username: Optional[str],
    password: Optional[str],
    explicit_token: Optional[str],
    explicit_server: Optional[str],
) -> tuple[str, str]:
    """Return (server_url, token) for the dm commands.

    Precedence (flag > env > default): explicit ``-u``/``-p`` login, then
    ``--token``, then the `_resolve_user_session` chain
    ($CLAWMEETS_ASSISTANT_TOKEN → $CLAWMEETS_USER_TOKEN → saved session).
    """
    if (username is None) != (password is None):
        typer.echo(
            "Error: -u/--username and -p/--password must be given together.",
            err=True,
        )
        raise typer.Exit(1)
    if username is not None and explicit_token is not None:
        typer.echo("Error: pass either -u/-p or --token, not both.", err=True)
        raise typer.Exit(1)
    if username is not None:
        server = _server_url(
            explicit_server or os.environ.get("CLAWMEETS_SERVER_URL") or DEFAULT_SERVER
        )
        with _http(server) as client:
            resp = client.post(
                "/auth/login", json={"username": username, "password": password}
            )
            if resp.status_code != 200:
                typer.echo("Error: Invalid username or password", err=True)
                raise typer.Exit(1)
            return server, resp.json()["access_token"]
    return _resolve_user_session(data_dir, explicit_token, explicit_server)


def _fetch_username(client: httpx.Client, token: str) -> str:
    """Resolve the acting user's username from a user JWT or assistant token."""
    resp = client.get("/auth/user/me", headers={"Authorization": f"Bearer {token}"})
    if resp.status_code != 200:
        typer.echo(f"Error: could not resolve current user ({resp.status_code})", err=True)
        raise typer.Exit(1)
    return resp.json()["username"]


def _resolve_project_ref(
    client: httpx.Client,
    token: Optional[str],
    ref: str,
    agent_id: Optional[str] = None,
) -> str:
    """Resolve a project reference (exact id or exact name) to a project id.

    Name matches are disambiguated by the caller: an agent (``agent_id``)
    keeps projects it participates in or coordinates; a user/assistant token
    keeps projects they created. Permissions are still enforced server-side
    at the target endpoint — this only picks *which* project was meant.
    """
    resp = client.get(f"/projects/{ref}")
    if resp.status_code == 200:
        return resp.json()["id"]
    resp = client.get("/projects")
    if resp.status_code != 200:
        typer.echo(f"Error: could not list projects ({resp.status_code})", err=True)
        raise typer.Exit(1)
    candidates = [p for p in resp.json() if p["name"] == ref]
    if len(candidates) > 1:
        if agent_id:
            scoped = [
                p
                for p in candidates
                if agent_id in (p.get("participating_agents") or [])
                or p.get("coordinator_id") == agent_id
            ]
        elif token:
            me = client.get(
                "/auth/user/me", headers={"Authorization": f"Bearer {token}"}
            )
            user_id = me.json().get("id") if me.status_code == 200 else None
            scoped = [p for p in candidates if p.get("created_by") == user_id]
        else:
            scoped = []
        if scoped:
            candidates = scoped
    if not candidates:
        typer.echo(f"Error: no project matches {ref!r}.", err=True)
        raise typer.Exit(1)
    if len(candidates) > 1:
        typer.echo(f"Error: project name {ref!r} is ambiguous:", err=True)
        for p in candidates:
            typer.echo(f"  {p['name']}  id={p['id']}", err=True)
        typer.echo("Pass the project id instead.", err=True)
        raise typer.Exit(1)
    return candidates[0]["id"]


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


def _put_user_teams_op(
    client: httpx.Client,
    token: str,
    agent_id: str,
    *,
    user_teams: Optional[list[str]] = None,
    add: Optional[str] = None,
    remove: Optional[str] = None,
) -> dict:
    """POST one atomic op to PUT /agents/{id}/user_teams.

    Accepts agent self-token, owner JWT, assistant token, or admin —
    see ``Auth.authorize_agent_self_or_credential``. Exactly one of
    ``user_teams`` / ``add`` / ``remove`` must be provided.
    """
    body: dict[str, object] = {}
    if user_teams is not None:
        body["user_teams"] = user_teams
    if add is not None:
        body["add"] = add
    if remove is not None:
        body["remove"] = remove
    resp = client.put(
        f"/agents/{agent_id}/user_teams",
        json=body,
        headers={"Authorization": f"Bearer {token}"},
    )
    return _ok(resp)


@team_app.command("add")
def team_add(
    agent: str = typer.Argument(..., help="Agent name or id, or 'self' (agent self-tags via runner-injected env)"),
    team_name: str = typer.Argument(..., help="Team to add"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Add a team to an agent (no-op if already present)."""
    server_url, token, agent_id, agent_name = _resolve_session_for_config(
        data_dir, token, server, agent
    )
    with _http(server_url) as client:
        _put_user_teams_op(client, token, agent_id, add=team_name)
    typer.echo(f"Added agent '{agent_name}' to team '{team_name}'.")


@team_app.command("remove")
def team_remove(
    agent: str = typer.Argument(..., help="Agent name or id, or 'self' (agent self-tags via runner-injected env)"),
    team_name: str = typer.Argument(..., help="Team to remove"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Remove a team from an agent (no-op if absent)."""
    server_url, token, agent_id, agent_name = _resolve_session_for_config(
        data_dir, token, server, agent
    )
    with _http(server_url) as client:
        _put_user_teams_op(client, token, agent_id, remove=team_name)
    typer.echo(f"Removed agent '{agent_name}' from team '{team_name}'.")


@team_app.command("set")
def team_set(
    agent: str = typer.Argument(..., help="Agent name or id, or 'self' (agent self-tags via runner-injected env)"),
    team: list[str] = typer.Option(
        None, "--team",
        help="Team value (repeatable). Pass with no --team flags to clear all teams.",
    ),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Replace an agent's team list with the given --team values (or clear)."""
    server_url, token, agent_id, agent_name = _resolve_session_for_config(
        data_dir, token, server, agent
    )
    new_teams = [t.strip() for t in (team or []) if t and t.strip()]
    with _http(server_url) as client:
        _put_user_teams_op(client, token, agent_id, user_teams=new_teams)
    if new_teams:
        typer.echo(f"Set teams on '{agent_name}': {', '.join(new_teams)}")
    else:
        typer.echo(f"Cleared teams on '{agent_name}'.")


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
    if content_file is None:
        return ""
    if not content_file.exists():
        typer.echo(f"Error: --content-file {content_file} does not exist", err=True)
        raise typer.Exit(1)
    return base64.b64encode(content_file.read_bytes()).decode("ascii")


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
    description: Optional[str] = typer.Option(
        None, "--description", "-d",
        help="One-line 'consult when ...' hint shown per-file in the installed-pack index.",
    ),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Add (or replace) a file inside a pack. Re-broadcasts to installed agents."""
    content_b64 = _read_file_b64_arg(content_file)
    body: dict = {"content_b64": content_b64}
    if description is not None:
        body["description"] = description
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        resp = client.put(
            f"/knowledge-packs/{slug}/files/{filepath}",
            json=body,
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


@agent_app.command("reconfigure")
def agent_reconfigure(
    agent: str = typer.Argument(..., help="Agent name or id, or 'self' (agent self-reconfigures via runner-injected env)"),
    git_url: Optional[str] = typer.Option(None, "--git-url", help="Bind to this git repo (empty string clears)."),
    git_base_branch: Optional[str] = typer.Option(None, "--git-base-branch", help="Base branch new work is cut from (empty string clears)."),
    knowledge_dir: Optional[str] = typer.Option(None, "--knowledge-dir", help="Proprietary-knowledge dir (empty string clears)."),
    dwh_dir: Optional[str] = typer.Option(None, "--dwh-dir", help="Personal data-warehouse root (empty string clears)."),
    llm_provider: Optional[str] = typer.Option(None, "--llm-provider", help=f"LLM backend, one of {_VALID_LLM_PROVIDERS} (empty string clears)."),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", help="Provider-specific model (empty string clears)."),
    llm_api_key: Optional[str] = typer.Option(None, "--llm-api-key", help="BYO key for a '-api' provider (empty string clears). NOTE: prefer the web UI — keys typed into chat sync to the server."),
    llm_base_url: Optional[str] = typer.Option(None, "--llm-base-url", help="Custom OpenAI-compatible endpoint for 'openai-api' (e.g. local ollama 'http://localhost:11434/v1'; empty string clears)."),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Change an agent's local_settings (partial merge) via PATCH /agents/{id}/local-settings.

    Supersedes the single-key ``set-dwh-dir``: sets/clears any of git_url,
    git_base_branch, knowledge_dir, dwh_dir, llm_provider, llm_model,
    llm_api_key, llm_base_url in one call. Only flags you pass are touched; pass an empty
    string to clear a key. Triggers AGENT_SETTINGS_CHANGE so a running runner
    picks it up on the next LLM invocation.

    Use ``self`` to target the calling agent (self-reconfigure from inside a
    runner subprocess, using the runner-injected CLAWMEETS_AGENT_* env). Any
    other value resolves a name/id via the user/assistant session — the
    server scopes the self-token to the calling agent only.
    """
    # Only flags actually supplied (None = untouched). Empty string is kept —
    # the server treats it as a clear.
    fields: dict[str, Optional[str]] = {}
    for key, value in (
        ("git_url", git_url),
        ("git_base_branch", git_base_branch),
        ("knowledge_dir", knowledge_dir),
        ("dwh_dir", dwh_dir),
        ("llm_provider", llm_provider),
        ("llm_model", llm_model),
        ("llm_api_key", llm_api_key),
        ("llm_base_url", llm_base_url),
    ):
        if value is not None:
            fields[key] = value
    if not fields:
        typer.echo("Error: pass at least one setting flag (e.g. --git-url).", err=True)
        raise typer.Exit(1)
    if llm_provider:
        normalized = llm_provider.lower()
        if normalized not in _VALID_LLM_PROVIDERS:
            typer.echo(
                f"Error: --llm-provider must be one of {_VALID_LLM_PROVIDERS} "
                f"(got {llm_provider!r})",
                err=True,
            )
            raise typer.Exit(1)
        fields["llm_provider"] = normalized

    server_url, token, agent_id, agent_name = _resolve_session_for_config(
        data_dir, token, server, agent,
    )
    with _http(server_url) as client:
        resp = client.patch(
            f"/agents/{agent_id}/local-settings",
            json={"local_settings": fields},
            headers={"Authorization": f"Bearer {token}"},
        )
        _ok(resp)
    changed = ", ".join(
        f"{k}={'(cleared)' if v == '' else v!r}" for k, v in fields.items()
    )
    typer.echo(f"Reconfigured agent '{agent_name}': {changed}")


@agent_app.command("list")
def agent_list(
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="User JWT to scope the listing to that account"),
    as_user: Optional[str] = typer.Option(None, "--as-user", help="Username whose saved session to use (defaults to current_user)"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Root data directory"),
    full: bool = typer.Option(False, "--full", "-f", help="Show full IDs"),
):
    """List agents visible to the calling identity (own + discoverable).

    Identity is resolved per process so this is unambiguous when several
    runners share a host: explicit --token, then the injected agent env
    (CLAWMEETS_AGENT_TOKEN + CLAWMEETS_AGENT_ID, attached by _http), then the
    saved session (--as-user / current_user). With no identity, only public
    (discoverable) agents are returned.
    """
    server_url = _server_url(server or os.environ.get("CLAWMEETS_SERVER_URL") or DEFAULT_SERVER)

    # Header precedence: explicit --token > injected agent env (handled by
    # _http's default Bearer+X-Agent-ID) > saved user session > anonymous.
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif not _env_identity_headers():
        # No per-process agent identity — fall back to the saved session
        # (non-fatal: unauthenticated listing still returns public agents).
        data_dir_p = Path(data_dir).expanduser()
        resolved_user = as_user or get_current_user(data_dir_p)
        session_token: Optional[str] = (
            os.environ.get("CLAWMEETS_ASSISTANT_TOKEN")
            or os.environ.get("CLAWMEETS_USER_TOKEN")
        )
        if not session_token and resolved_user:
            cfg_path = get_user_config_path(data_dir_p, resolved_user)
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text())
                session_token = cfg.get("user", {}).get("token")
        if session_token:
            headers["Authorization"] = f"Bearer {session_token}"
        else:
            typer.echo(
                "Note: no identity resolved (not in a runner and not logged in); "
                "showing public agents only. Use --as-user or --token to scope.",
                err=True,
            )

    with _http(server_url) as client:
        resp = client.get("/agents", headers=headers)
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
    user_config: Optional[Path] = typer.Option(None, "--user-config", help="Path to the owning user's settings.json; used as the base for resolving relative knowledge_dir/dwh_dir strings in card.json local_settings."),
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
    # Resolve plugin dir: flag > $CLAWMEETS_CLAUDE_PLUGIN_DIR > bundled `_clawmeets_plugin/`.
    if not claude_plugin_dir:
        env_path = os.environ.get("CLAWMEETS_CLAUDE_PLUGIN_DIR")
        detected = env_path or _detect_clawmeets_plugin_dir()
        claude_plugin_dir = [Path(detected)] if detected else []

    if working_dir:
        typer.echo(f"Working dir: {working_dir}")
    if knowledge_dir:
        typer.echo(f"Knowledge dir: {knowledge_dir}")
    if claude_plugin_dir:
        typer.echo(f"Claude plugin dirs: {claude_plugin_dir}")

    asyncio.run(_runner_loop(
        agent_name, agent_id, token, server, Path(agent_dir),
        working_dir, knowledge_dir, claude_plugin_dir,
        user_config=user_config,
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
    token = result["access_token"]
    if save:
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
):
    """Self-register a new user account (requires invitation code).

    After registration, check your email to verify your account.
    Once verified, run `clawmeets assistant register` to create your personal
    assistant agent locally.

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
    typer.echo("Registration successful. Check your email to verify your account.")
    typer.echo("Once verified, run `clawmeets assistant register` to create your assistant.")


@user_app.command("create")
def user_create(
    username: str = typer.Argument(..., help="Username"),
    password: str = typer.Argument(..., help="Password"),
    role: str = typer.Option("user", "--role", "-r", help="User role (admin or user)"),
    email: Optional[str] = typer.Option(None, "--email", "-e", help="User email address"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Admin JWT token (required)"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
):
    """Create a new user (requires admin token).

    Admin-created users are pre-verified (no email verification needed).
    The assistant agent is created later with `clawmeets assistant register`.
    """
    if not token:
        typer.echo("Error: --token is required. Get admin token with: user login admin <password>", err=True)
        raise typer.Exit(1)

    with _http(server) as client:
        payload = {"username": username, "password": password, "role": role}
        if email:
            payload["email"] = email
        resp = client.post(
            "/users",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        _ok(resp)
    typer.echo(f"Created user: {username}")
    typer.echo("Run `clawmeets assistant register` (as that user) to create the assistant agent.")


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

    token = result["access_token"]
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


def _self_destruct(agent_dir: Path) -> None:
    """Handle the server's 'participant not found' signal: rename the local
    agent directory to ``DELETED-*``.

    The filesystem rename is the authoritative cleanup — the agent list
    in ``clawmeets start`` is filesystem-derived and skips ``DELETED-*``
    dirs. Best-effort: logs and swallows errors so a runner shutting
    down in response to deletion never stalls on cleanup.
    """
    logger = logging.getLogger("clawmeets")
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
    # config dir (~/.clawmeets/config/<username>/) — the anchor for any
    # CLAUDE.md the user keeps alongside their settings. Absolute and
    # ~-prefixed paths pass through unchanged. Falls back to legacy
    # CWD-relative behavior only when --user-config is absent (shouldn't
    # happen under cli_lifecycle).
    user_config_dir: Optional[Path] = user_config.parent if user_config else None

    # Set up skill manager (downloads skills from server on startup)
    skill_manager = SkillManager(agent_dir)

    # Set up personal-skill manager (agent-local, never synced; populated by
    # the agent itself during scheduled reflection's Promote/Correct modes).
    personal_skill_manager = PersonalSkillManager(agent_dir)

    # Set up system-skill manager (bundled with the runner; materializes
    # per-audience symlink trees so the per-invocation skill_source_dirs
    # picks the right subset for this turn's role).
    system_skill_manager = SystemSkillManager(agent_dir)
    system_skill_manager.materialize_audiences()

    # Set up MCP manager (downloads manifests from server on startup;
    # renders .mcp.json into each Claude invocation's cwd)
    mcp_manager = McpManager(agent_dir)

    # Pick LLM provider from card.json local_settings ("claude" default;
    # "openai" and "gemini" also supported). Both skill-hub and
    # personal-skill-hub content are discovered uniformly across providers
    # via the INDEX.md files the prompt builder advertises (see
    # prompt_builder._build_runtime_context). Neither is a Claude plugin
    # anymore — references like ``/personal:<slug>`` in shipped SKILL.md
    # bodies read as instructions for the next-turn LLM to load that
    # SKILL.md (via the INDEX), interpretable by every provider. The
    # bundled ``plugins/clawmeets/`` (``/clawmeets:reflect`` etc.) stays a
    # Claude plugin because those slash commands have no cross-provider
    # equivalent today. Action schema is selected per invocation by the
    # caller (Agent) based on whether this runner is coordinator.
    all_plugin_dirs = list(claude_plugin_dirs or [])
    # Skill content roots — passed to Gemini so its sandboxed Read resolves
    # the absolute SKILL.md paths the prompt advertises. Claude reaches them
    # via the agent dir; Codex reads them off the prompt directly.
    # The bundled ``system_skills/`` source is added because Gemini's
    # ``--include-directories`` must cover the symlink targets in
    # ``{agent_dir}/system-skill-hub/skills-<role>/<name>``.
    all_skill_dirs = [
        SystemSkillManager.source_dir(),
        skill_manager.skill_hub_dir / "skills",
        personal_skill_manager.hub_dir / "skills",
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

    # Git binding (optional). When set, the git-workflow skill reads these to
    # clone the repo into the sandbox and branch per request. Empty/unset means
    # the agent does no git at all. Kept on the shared agent_env dict so a
    # settings hot-swap (reactive_loop._apply_local_settings) can update them
    # for the next invocation.
    _git_url = local_settings.get("git_url") or ""
    _git_base_branch = local_settings.get("git_base_branch") or ""
    if _git_url:
        agent_env["CLAWMEETS_AGENT_GIT_URL"] = _git_url
    if _git_base_branch:
        agent_env["CLAWMEETS_AGENT_GIT_BASE_BRANCH"] = _git_base_branch

    # The user's assistant agent (name ends with "-assistant") gets its own
    # bearer token exposed as $CLAWMEETS_ASSISTANT_TOKEN so admin system
    # skills (install-skill, register-agent, etc.) can shell `clawmeets <cmd>`
    # with user-level authority without re-prompting for credentials. The
    # assistant's own token IS the owner's assistant_token — they're the
    # same secret.
    if agent_name.endswith("-assistant"):
        agent_env["CLAWMEETS_ASSISTANT_TOKEN"] = token

    llm_provider_name = (local_settings.get("llm_provider") or "claude").lower()
    llm_model = local_settings.get("llm_model") or None

    # Factory closes over runner-scoped args (plugin dirs, skill dirs,
    # agent env). Shared by the startup path here and the reactive loop's
    # hot-swap path; takes the live local_settings so provider (incl. the
    # ``-api`` variant) / model / BYO-key all resolve from one source.
    def cli_factory(settings: dict) -> LLMProvider:
        provider = (settings.get("llm_provider") or "claude").lower()
        model = settings.get("llm_model") or None
        api_key = _resolve_api_key(provider, settings) if _is_api_provider(provider) else None
        base_url = settings.get("llm_base_url") or None
        return _build_llm_provider(
            provider,
            model,
            plugin_dirs=all_plugin_dirs,
            skill_dirs=all_skill_dirs,
            agent_env=agent_env,
            api_key=api_key,
            api_caps=_api_caps_from_settings(settings),
            base_url=base_url,
        )

    try:
        cli = cli_factory(local_settings)
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
    resolved = FileUtil.resolve_local_dir(str(effective_knowledge_dir), user_config_dir) if effective_knowledge_dir else None
    if resolved is not None:
        knowledge_dirs_list.append(resolved)

    # Resolve dwh_dir (personal data warehouse root, typically network-shared).
    # None when unset — the prompt block is omitted in that case.
    raw_dwh_dir = local_settings.get("dwh_dir") or ""
    resolved_dwh_dir = FileUtil.resolve_local_dir(str(raw_dwh_dir), user_config_dir) if raw_dwh_dir else None

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
        git_url=_git_url or None,
    )

    participant = Agent(id=agent_id, model_ctx=model_ctx)

    # Set up knowledge-pack manager (writes user-curated packs to
    # {agent_dir}/knowledge_packs/<slug>/; index lives at
    # {agent_dir}/memory/KNOWLEDGE_PACKS.md alongside the other
    # AUTHORITATIVE indexes)
    knowledge_pack_manager = KnowledgePackManager(model_ctx)

    # Build the proprietary-knowledge index (memory/REFERENCES.md) once at
    # startup — deterministic, from knowledge_dir file content. Hot-rebuilt on
    # knowledge_dir settings changes by the reactive loop.
    build_references_index(model_ctx.memory_dir, model_ctx.knowledge_dirs)

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
        # Shared agent_env dict so a git_url/git_base_branch settings change can
        # update CLAWMEETS_AGENT_GIT_URL for the next invocation (the CLI is
        # rebuilt from this same dict on change).
        agent_env=agent_env,
        # Owner username from the startup credential (agent_name == "{owner}-{suffix}";
        # usernames have no hyphens). Lets catch_up build a populated AGENTS.md roster
        # on the first cold-start sync, before any self peer-card exists.
        owner_username=agent_name.split("-", 1)[0] if "-" in agent_name else agent_name,
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
        except httpx.HTTPError as e:
            # The on-connect HTTP catch-up (skill/mcp/knowledge-pack sync +
            # loop_obj.catch_up()) runs right after the socket connects. A
            # transient tunnel error there — ngrok serving 404/503 or a slow
            # ReadTimeout during a restart/rate-limit window — must back off and
            # reconnect, NOT escape the loop and kill the runner. Same policy as
            # the WS-transport arm above (the WS upgrade succeeding then the
            # immediate HTTP burst failing is exactly the gap this closes).
            logging.warning(f"HTTP catch-up error: {e}. Reconnecting in {reconnect_delay}s…")
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
            _self_destruct(agent_dir)
            await loop_obj.stop()
            await http_client.aclose()
            return

        logging.warning(f"Reconnecting in {reconnect_delay}s…")
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, 60)


# ---------------------------------------------------------------------------
# dm (direct message) commands
# ---------------------------------------------------------------------------

DM_CHATROOM_NAME = "user-communication"


def _ensure_dm_target(
    client: httpx.Client,
    token: str,
    agent_full_name: str,
) -> Optional[dict]:
    """Idempotently resolve the per-agent DM/FD project for ``agent_full_name``.

    Returns the project dict (with ``id``, ``name``) or ``None`` if no agent
    by that name exists. Tries the own-agent DM endpoint first; on 403
    (foreign agent) falls back to the FD endpoint.
    """
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.post(
        f"/me/dms/{agent_full_name}/ensure", headers=headers,
    )
    if resp.status_code == 403:
        resp = client.post(
            f"/me/fd/{agent_full_name}/ensure", headers=headers,
        )
    if resp.status_code != 200:
        return None
    return resp.json()


@dm_app.command("send")
def dm_send(
    agent_name: str = typer.Argument(..., help="Full agent name to message (e.g. 'alice-researcher')"),
    message: str = typer.Argument(..., help="Message content"),
    username: Optional[str] = typer.Option(None, "-u", "--username", help="Username (with -p; optional — defaults to token/session auth)"),
    password: Optional[str] = typer.Option(None, "-p", "--password", help="Password (with -u)"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="User JWT or assistant token (defaults to $CLAWMEETS_ASSISTANT_TOKEN / $CLAWMEETS_USER_TOKEN / saved session)"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Send a direct message to an agent.

    Resolves the per-agent DM project (own agent → ``{username}-dm-{short}``,
    foreign agent → ``{username}-fd-{short}``) and posts to its
    ``user-communication`` chatroom. The project is created lazily on first
    use. The message appears as the user — including when the user's
    assistant shells this command with its own bearer token.

    Examples:
        clawmeets dm send alice-researcher "Can you help me?"
        clawmeets dm send alice-researcher "Can you help me?" -u alice -p mypassword
    """
    server_url, token = _resolve_dm_session(data_dir, username, password, token, server)
    with _http(server_url) as client:
        dm_project = _ensure_dm_target(client, token, agent_name)
        if not dm_project:
            typer.echo(f"Error: Could not resolve DM project for agent {agent_name}", err=True)
            raise typer.Exit(1)

        # Send message via user-message endpoint
        resp = client.post(
            f"/projects/{dm_project['id']}/chatrooms/{DM_CHATROOM_NAME}/user-message",
            json={"content": message},
            headers={"Authorization": f"Bearer {token}"},
        )
        _ok(resp)
        typer.echo(f"Message sent to @{agent_name}")


@dm_app.command("list")
def dm_list(
    username: Optional[str] = typer.Option(None, "-u", "--username", help="Username (with -p; optional — defaults to token/session auth)"),
    password: Optional[str] = typer.Option(None, "-p", "--password", help="Password (with -u)"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="User JWT or assistant token (defaults to $CLAWMEETS_ASSISTANT_TOKEN / $CLAWMEETS_USER_TOKEN / saved session)"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """List all DM conversations.

    Enumerates the caller's per-agent DM projects (``{username}-dm-*``)
    and per-agent FD projects (``{username}-fd-*``); each one is exactly
    one DM thread on its ``user-communication`` chatroom.

    Example:
        clawmeets dm list
    """
    server_url, token = _resolve_dm_session(data_dir, username, password, token, server)
    with _http(server_url) as client:
        if username is None:
            username = _fetch_username(client, token)
        resp = client.get("/projects", headers={"Authorization": f"Bearer {token}"})
        projects = _ok(resp)

        dm_prefix = f"{username}-dm-"
        fd_prefix = f"{username}-fd-"
        dm_projects = [
            p for p in projects
            if p["name"].startswith(dm_prefix) or p["name"].startswith(fd_prefix)
        ]
        if not dm_projects:
            typer.echo("No DM conversations yet.")
            return

        typer.echo("DM Conversations:")
        for proj in dm_projects:
            if proj["name"].startswith(dm_prefix):
                agent_name = proj["name"][len(dm_prefix):]
            else:
                agent_name = proj["name"][len(fd_prefix):]
            resp = client.get(
                f"/projects/{proj['id']}/chatrooms/{DM_CHATROOM_NAME}/messages",
                headers={"Authorization": f"Bearer {token}"},
            )
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
    agent_name: str = typer.Argument(..., help="Full agent name"),
    username: Optional[str] = typer.Option(None, "-u", "--username", help="Username (with -p; optional — defaults to token/session auth)"),
    password: Optional[str] = typer.Option(None, "-p", "--password", help="Password (with -u)"),
    limit: int = typer.Option(20, "-n", "--limit", help="Number of messages to show"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="User JWT or assistant token (defaults to $CLAWMEETS_ASSISTANT_TOKEN / $CLAWMEETS_USER_TOKEN / saved session)"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Show DM history with an agent.

    Example:
        clawmeets dm history alice-researcher -n 50
    """
    server_url, token = _resolve_dm_session(data_dir, username, password, token, server)
    with _http(server_url) as client:
        dm_project = _ensure_dm_target(client, token, agent_name)
        if not dm_project:
            typer.echo(f"Error: Could not resolve DM project for agent {agent_name}", err=True)
            raise typer.Exit(1)

        resp = client.get(
            f"/projects/{dm_project['id']}/chatrooms/{DM_CHATROOM_NAME}/messages",
            headers={"Authorization": f"Bearer {token}"},
        )
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
    agent_name: str = typer.Argument(..., help="Full agent name to schedule messages to"),
    message: str = typer.Argument(..., help="Message content"),
    cron: str = typer.Option(..., "--cron", "-c", help="Cron expression (e.g. '@daily', '0 9 * * *')"),
    end_at: Optional[str] = typer.Option(None, "--end-at", help="Expiration time (ISO 8601)"),
    username: Optional[str] = typer.Option(None, "-u", "--username", help="Username (with -p; optional — defaults to token/session auth)"),
    password: Optional[str] = typer.Option(None, "-p", "--password", help="Password (with -u)"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="User JWT or assistant token (defaults to $CLAWMEETS_ASSISTANT_TOKEN / $CLAWMEETS_USER_TOKEN / saved session)"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Schedule a recurring DM to an agent.

    Resolves the per-agent DM project (lazy-creating it on first use), then
    schedules the message into its ``user-communication`` chatroom. The cron
    expression is evaluated in UTC.

    Examples:
        clawmeets dm schedule alice-researcher "Check for new findings" --cron "@daily"
        clawmeets dm schedule alice-analyst "Run weekly report" --cron "0 9 * * 1"
    """
    server_url, token = _resolve_dm_session(data_dir, username, password, token, server)
    with _http(server_url) as client:
        dm_project = _ensure_dm_target(client, token, agent_name)
        if not dm_project:
            typer.echo(f"Error: Could not resolve DM project for agent {agent_name}", err=True)
            raise typer.Exit(1)

        # Create scheduled message
        payload: dict = {
            "project_id": dm_project["id"],
            "chatroom_name": DM_CHATROOM_NAME,
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
    username: Optional[str] = typer.Option(None, "-u", "--username", help="Username (with -p; optional — defaults to token/session auth)"),
    password: Optional[str] = typer.Option(None, "-p", "--password", help="Password (with -u)"),
    all_: bool = typer.Option(False, "--all", "-a", help="Include inactive schedules"),
    full: bool = typer.Option(False, "--full", help="Print full message content (no truncation)"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="User JWT or assistant token (defaults to $CLAWMEETS_ASSISTANT_TOKEN / $CLAWMEETS_USER_TOKEN / saved session)"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """List your scheduled DM messages.

    Example:
        clawmeets dm schedules
    """
    server_url, token = _resolve_dm_session(data_dir, username, password, token, server)
    with _http(server_url) as client:
        if username is None:
            username = _fetch_username(client, token)
        # Build a map of per-agent DM/FD project ID → agent short name so we
        # can filter scheduled messages to just those targeting DM threads.
        resp = client.get("/projects", headers={"Authorization": f"Bearer {token}"})
        projects = _ok(resp)
        dm_prefix = f"{username}-dm-"
        fd_prefix = f"{username}-fd-"
        dm_project_agent: dict[str, str] = {}
        for p in projects:
            if p["name"].startswith(dm_prefix):
                dm_project_agent[p["id"]] = p["name"][len(dm_prefix):]
            elif p["name"].startswith(fd_prefix):
                dm_project_agent[p["id"]] = p["name"][len(fd_prefix):]

        params = {"active_only": "false"} if all_ else {}
        resp = client.get(
            "/scheduled-messages",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        schedules = _ok(resp)

        dm_schedules = [
            s for s in schedules
            if s["chatroom_name"] == DM_CHATROOM_NAME
            and s["project_id"] in dm_project_agent
        ]

        if not dm_schedules:
            typer.echo("No scheduled DM messages.")
            return

        for s in dm_schedules:
            agent_name = dm_project_agent[s["project_id"]]
            status = "active" if s["is_active"] else "inactive"
            content = s["content"] if full else s["content"][:60]
            typer.echo(
                f"  [{status}] {s['id'][:8]}... "
                f"@{agent_name} cron={s['cron_expression']!r} "
                f"next={s['next_fire_at']} "
                f"content={content!r}"
            )


@dm_app.command("unschedule")
def dm_unschedule(
    schedule_id: str = typer.Argument(..., help="Scheduled message ID to cancel"),
    username: Optional[str] = typer.Option(None, "-u", "--username", help="Username (with -p; optional — defaults to token/session auth)"),
    password: Optional[str] = typer.Option(None, "-p", "--password", help="Password (with -u)"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="User JWT or assistant token (defaults to $CLAWMEETS_ASSISTANT_TOKEN / $CLAWMEETS_USER_TOKEN / saved session)"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Cancel a scheduled DM message.

    Example:
        clawmeets dm unschedule abc12345-...
    """
    server_url, token = _resolve_dm_session(data_dir, username, password, token, server)
    with _http(server_url) as client:
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
        from clawmeets.integrations.auth.google_oauth import GoogleOAuthError, run_installed_flow
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
    agent: str = typer.Argument(..., help="Agent name or id, or 'self' (worker self-installs via runner-injected env)"),
    mcps: List[str] = typer.Argument(..., help="One or more MCP names to install"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Install one or more MCP servers on an agent."""
    server_url, token, agent_id, agent_name = _resolve_session_for_config(
        data_dir, token, server, agent,
    )
    with _http(server_url) as client:
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


@mcp_app.command("uninstall")
def mcp_uninstall(
    agent: str = typer.Argument(..., help="Agent name or id, or 'self' (worker self-uninstalls via runner-injected env)"),
    mcp_name: str = typer.Argument(..., help="MCP server name to uninstall"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Uninstall an MCP server from an agent."""
    server_url, token, agent_id, agent_name = _resolve_session_for_config(
        data_dir, token, server, agent,
    )
    with _http(server_url) as client:
        resp = client.delete(
            f"/agents/{agent_id}/mcps/{mcp_name}",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = _ok(resp)
    typer.echo(f"Uninstalled {data.get('removed', mcp_name)!r} from '{agent_name}'.")


@mcp_app.command("set-config")
def mcp_set_config(
    agent: str = typer.Argument(
        ...,
        help="Agent name/id, or 'self' to target the calling agent "
             "(via env-injected CLAWMEETS_AGENT_{ID,TOKEN,SERVER_URL}).",
    ),
    mcp_name: str = typer.Argument(..., help="MCP server name"),
    config_file: Path = typer.Argument(..., help="JSON file with the config payload"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Set the per-agent config for an MCP server (uploads JSON from a file).

    Persists to card.json.local_settings.mcp_configs[mcp_name] and broadcasts
    AGENT_SETTINGS_CHANGE so the runner writes through to
    {agent_dir}/mcp-hub/configs/<mcp_name>.json. Use 'self' as the agent
    argument from inside an agent subprocess to update that agent's own
    config without a user session.
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

    server_url, token, agent_id, agent_name = _resolve_session_for_config(
        data_dir, token, server, agent,
    )
    with _http(server_url) as client:
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
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Create or update the account-level reflection schedule.

    The reflect skill distills new lessons from recent activity AND audits the
    existing wiki for contradictions / staleness in one pass per cycle.
    """
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        resp = client.put(
            "/account/reflection-schedule",
            json={
                "cron_expression": cron,
                "is_active": True,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        result = _ok(resp)

    typer.echo(
        f"Reflect cron: {result['cron_expression']!r}  next: {result['next_fire_at']}"
    )


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

def _ensure_fresh_user_token(server_url: str, data_dir: Path, username: str, current_token: str) -> str:
    """Verify the saved JWT still works; auto-refresh from saved password if expired.

    `clawmeets user login --save` saves both the JWT and the password into
    settings.json. JWTs eventually expire; rather than telling users to manually
    re-login, we silently re-issue using the saved password and persist the new
    token.
    """
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
    new_token = resp.json().get("access_token")
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
    return FileUtil.resolve_local_dir(str(raw), user_config_dir)


def _bootstrap_session_setup(
    data_dir: Path,
    server: Optional[str],
    username: Optional[str],
) -> tuple[str, str, str, Path]:
    """Resolve server URL, user JWT, username, and data_dir_p. Exits on failure."""
    server_url, user_jwt = _resolve_user_session(data_dir, None, server, as_user=username)
    if username is None:
        username = get_current_user(Path(data_dir).expanduser())
    if not username:
        typer.echo("Error: no username known. Pass -u or run `clawmeets user login --save` first.", err=True)
        raise typer.Exit(1)
    data_dir_p = Path(data_dir).expanduser()
    user_jwt = _ensure_fresh_user_token(server_url, data_dir_p, username, user_jwt)
    return server_url, user_jwt, username, data_dir_p


@bootstrap_app.command("browser")
def bootstrap_browser(
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Install Chromium for the playwright-browser skill (one-time per machine).

    Verifies the Python ``playwright`` package is importable, then runs
    ``python -m playwright install chromium`` (~150 MB download; Playwright
    skips browsers already cached). On Linux also runs
    ``python -m playwright install-deps chromium``. patchright (the stealth
    runtime) ships by default on Python ≤3.13, so this also runs
    ``python -m patchright install chrome`` to ready the Cloudflare-resistant
    path; on Python 3.14 patchright has no build and the skill falls back to
    stock Playwright. Touches
    ``{data_dir}/.playwright_bootstrapped`` so the skill's preflight can
    answer instantly.

    Idempotent: safe to re-run any time. Browsers are global state on the
    runner machine; agent-level identity storage stays per-agent under
    ``{agent_dir}/skill-hub/state/playwright-browser/storage/``, and
    project-scoped sessions land under each project's sandbox as a persistent
    Chrome profile at ``.playwright-session/profile/``.
    """
    typer.echo("[bootstrap browser] checking Python playwright package…")
    try:
        import playwright  # noqa: F401
    except ImportError:
        typer.echo(
            "Error: `playwright` is not installed. The clawmeets runner should "
            "bundle it as a dep. Try: pip install --upgrade clawmeets",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo("[bootstrap browser] installing Chromium via python -m playwright install chromium…")
    typer.echo("  (~150 MB download on first run; subsequent runs are instant)")
    install_proc = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if install_proc.returncode != 0:
        typer.echo("Error: `playwright install chromium` failed.", err=True)
        raise typer.Exit(install_proc.returncode)

    if platform.system() == "Linux":
        typer.echo("[bootstrap browser] installing system libs (Linux only)…")
        deps_proc = subprocess.run(
            [sys.executable, "-m", "playwright", "install-deps", "chromium"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if deps_proc.returncode != 0:
            typer.echo(
                "Warning: `playwright install-deps chromium` failed; "
                "headed browser may not start. Re-run with sudo if needed.",
                err=True,
            )

    # Stealth runtime (optional extra): patchright is a patched Playwright that
    # closes the CDP Runtime.enable leak defeating Cloudflare Turnstile. It ships
    # its own browser fetch and recommends real Chrome (channel="chrome").
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    try:
        import patchright  # noqa: F401
        has_patchright = True
    except ImportError:
        has_patchright = False
    if has_patchright:
        typer.echo("[bootstrap browser] stealth runtime detected; installing patchright Chrome…")
        pr_proc = subprocess.run(
            [sys.executable, "-m", "patchright", "install", "chrome"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if pr_proc.returncode != 0:
            typer.echo(
                "Warning: `patchright install chrome` failed; install Google "
                "Chrome on this machine (channel=chrome) for best results.",
                err=True,
            )
    else:
        typer.echo(
            f"WARNING: stealth runtime INACTIVE — patchright is not importable "
            f"under Python {py_ver} (patchright supports 3.9–3.13; 3.14 has no "
            f"build, so it ships by default only on ≤3.13). The browser skill "
            f"will run stock Playwright, which Cloudflare Turnstile / airline "
            f"sites will likely block. To fix: run clawmeets under Python ≤3.13 "
            f"and reinstall, then re-run this command.",
            err=True,
        )

    data_dir_p = Path(data_dir).expanduser()
    data_dir_p.mkdir(parents=True, exist_ok=True)
    marker = data_dir_p / ".playwright_bootstrapped"
    marker.touch()
    typer.echo(f"[bootstrap browser] done. Marker: {marker}")
    typer.echo(
        f"  Python {py_ver} · stealth runtime: "
        f"{'patchright (active)' if has_patchright else 'playwright (stock — Cloudflare-prone)'}"
    )
    typer.echo(
        "  Install the playwright-browser skill on any agent that needs browser "
        "automation (Agent settings → Skills → playwright-browser). Then run "
        "`clawmeets browser auth --storage <name>` to log in once and save an "
        "agent-level identity."
    )


# ---------------------------------------------------------------------------
# assistant register / agent-team register
# ---------------------------------------------------------------------------

_PERSONALIZE_TRIGGER_MARKER = "<!-- clawmeets:personalize-trigger -->"


def _login_for_jwt(server: str, username: str, password: str) -> str:
    """Login and return the JWT token. Exits on failure."""
    with _http(server) as client:
        resp = client.post(
            "/auth/login",
            json={"username": username, "password": password},
        )
        try:
            result = _ok(resp)
        except SystemExit:
            typer.echo("Error: Invalid username or password.", err=True)
            raise typer.Exit(1)
    return result["access_token"]


def _write_agent_card(
    *,
    agent_dir: Path,
    agent_id: str,
    agent_name: str,
    description: str,
    capabilities: list[str],
    status: str,
    registered_at: str,
    discoverable: bool,
    registered_by: Optional[str] = None,
    user_teams: Optional[list[str]] = None,
    local_settings: Optional[dict] = None,
) -> Path:
    """Write a card.json for an agent. Returns the path written."""
    card: dict = {
        "id": agent_id,
        "name": agent_name,
        "description": description,
        "capabilities": list(capabilities or []),
        "status": status,
        "registered_at": registered_at,
        "discoverable_through_registry": discoverable,
    }
    if registered_by:
        card["registered_by"] = registered_by
    if user_teams:
        card["user_teams"] = list(user_teams)
    if local_settings:
        card["local_settings"] = dict(local_settings)
    card_path = agent_dir / "card.json"
    card_path.write_text(json.dumps(card, indent=2, default=str))
    return card_path


def _write_agent_credential(
    *, agent_dir: Path, agent_id: str, agent_name: str, token: str,
) -> Path:
    cred = {"agent_id": agent_id, "token": token, "agent_name": agent_name}
    cred_path = agent_dir / "credential.json"
    cred_path.write_text(json.dumps(cred, indent=2, default=str))
    return cred_path


def _archive_stale_agent_dirs(agents_dir: Path, registered_name: str, keep: Path) -> None:
    """Archive sibling dirs `{registered_name}-<other_id>/` left over from prior
    registrations under the same name so `_find_agent_dir` lands on `keep`.
    """
    if not agents_dir.exists():
        return
    for sibling in agents_dir.iterdir():
        if not sibling.is_dir() or sibling == keep:
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


def _register_one_assistant(
    client: httpx.Client,
    token: str,
    *,
    username: str,
    description: str,
    local_settings: dict,
) -> dict:
    """POST /agents/register for `{username}-assistant` and return the response.

    Discoverable=false, capabilities standardized for coordinator role. The
    server auto-links the assistant to the user + creates the DM project.
    """
    payload = {
        "name": "assistant",  # server prefixes with `{username}-`
        "description": description,
        "capabilities": ["coordination", "planning", "delegation", "user communication"],
        "discoverable_through_registry": False,
    }
    resp = client.post(
        "/agents/register",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    return _ok(resp)


def _post_dm_marker(
    client: httpx.Client,
    token: str,
    agent_full_name: str,
    marker_body: str,
) -> bool:
    """Send `marker_body` as a user message into agent's DM. Returns success."""
    dm_project = _ensure_dm_target(client, token, agent_full_name)
    if not dm_project:
        typer.echo(
            f"  Warning: could not resolve DM project for '{agent_full_name}'.",
            err=True,
        )
        return False
    resp = client.post(
        f"/projects/{dm_project['id']}/chatrooms/{DM_CHATROOM_NAME}/user-message",
        json={"content": marker_body},
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status_code >= 400:
        typer.echo(
            f"  Warning: failed to post bootstrap DM to '{agent_full_name}': {resp.text}",
            err=True,
        )
        return False
    return True


def _put_reflection_schedule(
    client: httpx.Client,
    token: str,
    *,
    cron_expression: str,
    timezone: str,
) -> dict:
    resp = client.put(
        "/account/reflection-schedule",
        json={
            "cron_expression": cron_expression,
            "is_active": True,
            "timezone": timezone,
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    return _ok(resp)


@assistant_app.command("register")
def assistant_register(
    username: Optional[str] = typer.Option(
        None, "--username", "-u",
        help="Your ClawMeets username (defaults to current_user from a saved session).",
    ),
    password: Optional[str] = typer.Option(
        None, "--password", "-p",
        help="Your ClawMeets password (only needed when no saved session exists).",
    ),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR, "--data-dir",
        help="Root data directory (assistant saved to {data_dir}/agents/)",
    ),
    llm_provider: Optional[str] = typer.Option(
        None, "--llm-provider",
        help="LLM backend for the assistant. Bare names shell the Code CLI: "
             "'claude' (default), 'openai', 'gemini', 'opencode' (Zen gateway; "
             "set --llm-model to a provider/model slug). The '-api' variants run "
             "in-process with a BYO key: 'claude-api', 'openai-api', 'gemini-api', "
             "'openrouter-api' (OpenAI-compatible gateway — set --llm-model to a slug).",
    ),
    llm_model: Optional[str] = typer.Option(
        None, "--llm-model",
        help="Provider-specific model name (e.g. 'o3' for Codex, 'gemini-2.5-pro' for Gemini).",
    ),
    llm_api_key: Optional[str] = typer.Option(
        None, "--llm-api-key", envvar="CLAWMEETS_LLM_API_KEY",
        help="API key for a '-api' provider (BYO-key). Persisted to card.json "
             "local_settings and takes priority over the provider's standard env "
             "var (ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY).",
    ),
    llm_base_url: Optional[str] = typer.Option(
        None, "--llm-base-url",
        help="Custom OpenAI-compatible endpoint for 'openai-api' (e.g. local "
             "ollama 'http://localhost:11434/v1'). Written to card.json local_settings.",
    ),
    no_personalize: bool = typer.Option(
        False, "--no-personalize",
        help="Skip posting the personalize-trigger DM after registering.",
    ),
    reflect_daily_at: str = typer.Option(
        "09:00", "--reflect-daily-at",
        help="Local time of day (HH:MM) to fire the daily reflection.",
    ),
    reflect_timezone: Optional[str] = typer.Option(
        None, "--reflect-timezone",
        help="IANA timezone for the reflection schedule (default: host machine's timezone).",
    ),
):
    """Create your personal assistant agent (`{username}-assistant`).

    Steps in order:
      1. Log in.
      2. Register the assistant agent on the server (non-discoverable, owned
         by you). The server auto-links it to your user record and creates
         the DM project.
      3. Write `{data_dir}/agents/{name}-assistant-{id}/credential.json`
         and `card.json` locally.
      4. Upsert the account-level reflection schedule
         (cron derived from `--reflect-daily-at` + `--reflect-timezone`).
      5. Unless `--no-personalize`: post `<!-- clawmeets:personalize-trigger -->`
         as a user message into your assistant's DM so the
         `/clawmeets:personalize` skill (assistant variant) kicks off USER.md.

    Example:
        clawmeets assistant register --llm-provider=claude --reflect-daily-at=03:00
        clawmeets assistant register -u alice -p secret --no-personalize \\
            --reflect-daily-at 14:00 --reflect-timezone America/Los_Angeles \\
            --llm-provider gemini
    """
    tz = (reflect_timezone or "").strip() or _host_iana_timezone()
    cron_expression = _daily_at_to_cron(reflect_daily_at)
    local_settings = _build_initial_local_settings(
        llm_provider, llm_model, dwh_dir=None, llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
    )

    if username and password:
        token = _login_for_jwt(server, username, password)
    else:
        _, token = _resolve_user_session(data_dir, None, server, as_user=username)
        if not username:
            username = get_current_user(Path(data_dir).expanduser())
        if not username:
            typer.echo(
                "Error: not logged in. Pass -u/-p, or run "
                "`clawmeets user login <user> <pass> --save` first.",
                err=True,
            )
            raise typer.Exit(1)

    typer.echo(f"  Server: {server}")
    typer.echo(f"  Reflection: cron={cron_expression!r}  tz={tz!r}")

    with _http(server) as client:
        result = _register_one_assistant(
            client,
            token,
            username=username,
            description=f"Personal assistant for {username}",
            local_settings=local_settings,
        )
        agent_id = result["agent_id"]
        agent_name = result.get("agent_name") or f"{username}-assistant"

        # local_settings is set via a follow-up PUT because /agents/register
        # doesn't accept it. Skip when no settings to send.
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
                    f"  Warning: failed to sync local_settings for assistant: {put_resp.text}",
                    err=True,
                )

        agents_dir = Path(data_dir).expanduser() / "agents"
        assistant_dir = agents_dir / f"{agent_name}-{agent_id}"
        assistant_dir.mkdir(parents=True, exist_ok=True)
        new_token = result.get("token")
        if new_token:
            cred_path = _write_agent_credential(
                agent_dir=assistant_dir,
                agent_id=agent_id,
                agent_name=agent_name,
                token=new_token,
            )
            typer.echo(f"  Credentials: {cred_path}")
        else:
            typer.echo(
                "  Re-registered; existing credentials preserved (server kept the old token)."
            )
        _archive_stale_agent_dirs(agents_dir, agent_name, keep=assistant_dir)

        card_path = _write_agent_card(
            agent_dir=assistant_dir,
            agent_id=agent_id,
            agent_name=agent_name,
            description=result.get("description") or f"Personal assistant for {username}",
            capabilities=result.get("capabilities", []),
            status=result.get("status", "offline"),
            registered_at=str(result.get("registered_at") or datetime.now(UTC).isoformat()),
            discoverable=False,
            local_settings=local_settings,
        )
        typer.echo(f"  Card: {card_path}")

        schedule = _put_reflection_schedule(
            client, token, cron_expression=cron_expression, timezone=tz,
        )
        typer.echo(
            f"  Reflection schedule set (next fire: {schedule.get('next_fire_at')})"
        )

        if not no_personalize:
            if _post_dm_marker(client, token, agent_name, _PERSONALIZE_TRIGGER_MARKER + "\n"):
                typer.echo(
                    f"  Posted personalize-trigger DM to '{agent_name}'."
                )

    typer.echo("\nAssistant ready. Next steps:")
    typer.echo("  - `clawmeets start` to bring your agent online")
    typer.echo("  - open the dashboard to chat with your assistant")


def _register_one_worker(
    client: httpx.Client,
    token: str,
    agent: dict,
    agents_dir: Path,
    *,
    llm_provider_override: Optional[str],
    llm_base_url_override: Optional[str] = None,
) -> Optional[dict]:
    """Register one worker from a setup.json `agents[]` entry. Returns the
    server response on success, None on failure.
    """
    name = agent.get("name")
    description = agent.get("description", "")
    capabilities = agent.get("capabilities") or []
    discoverable = bool(agent.get("discoverable", False))
    agent_user_teams = _normalize_user_teams_from_setup(agent.get("user_teams"))

    register_payload = {
        "name": name,
        "description": description,
        "capabilities": capabilities,
        "discoverable_through_registry": discoverable,
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
        typer.echo(f"  Failed to register '{name}': {resp.text}", err=True)
        return None

    result = resp.json()
    registered_name = result.get("agent_name", name)
    agent_id = result["agent_id"]

    agent_work_dir = agents_dir / f"{registered_name}-{agent_id}"
    agent_work_dir.mkdir(parents=True, exist_ok=True)

    new_token = result.get("token")
    if new_token:
        _write_agent_credential(
            agent_dir=agent_work_dir,
            agent_id=agent_id,
            agent_name=registered_name,
            token=new_token,
        )
    _archive_stale_agent_dirs(agents_dir, registered_name, keep=agent_work_dir)

    # Build local_settings (mirrors cli_init._register_agents logic).
    local_settings: dict = {}
    if agent.get("knowledge_dir"):
        local_settings["knowledge_dir"] = agent["knowledge_dir"]
    provider_for_agent = llm_provider_override or agent.get("llm_provider")
    if provider_for_agent:
        provider = provider_for_agent.lower()
        if provider not in _VALID_LLM_PROVIDERS:
            typer.echo(
                f"  Warning: skipping invalid llm_provider {provider_for_agent!r} for '{name}' "
                f"(expected one of {_VALID_LLM_PROVIDERS})",
                err=True,
            )
        else:
            local_settings["llm_provider"] = provider
    if agent.get("llm_model"):
        local_settings["llm_model"] = agent["llm_model"]
    base_url_for_agent = llm_base_url_override or agent.get("llm_base_url")
    if base_url_for_agent:
        local_settings["llm_base_url"] = base_url_for_agent
    agent_dwh_dir = agent.get("dwh_dir") or os.environ.get("CLAWMEETS_DWH_DIR", "")
    if agent_dwh_dir:
        local_settings["dwh_dir"] = agent_dwh_dir

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

    final_teams = result.get("user_teams") or agent_user_teams
    _write_agent_card(
        agent_dir=agent_work_dir,
        agent_id=agent_id,
        agent_name=registered_name,
        description=result.get("description", description),
        capabilities=result.get("capabilities", capabilities),
        status=result.get("status", "offline"),
        registered_at=str(result.get("registered_at") or datetime.now(UTC).isoformat()),
        discoverable=result.get("discoverable_through_registry", discoverable),
        user_teams=final_teams or None,
        local_settings=local_settings or None,
    )

    # MCPs + skills declared in setup.json: same HTTP path the web UI uses.
    for kind, endpoint, items in (
        ("MCP servers", f"/agents/{agent_id}/mcps", agent.get("mcp_servers") or []),
        ("Skills", f"/agents/{agent_id}/skills", agent.get("skills") or []),
    ):
        if not items:
            continue
        body_key = "mcps" if kind == "MCP servers" else "skills"
        sub_resp = client.post(
            endpoint,
            json={body_key: items},
            headers={"Authorization": f"Bearer {token}"},
        )
        if sub_resp.status_code >= 400:
            typer.echo(
                f"  Warning: failed to install {kind} {items} for '{registered_name}': {sub_resp.text}",
                err=True,
            )
        else:
            added = sub_resp.json().get("added", [])
            if added:
                typer.echo(f"    {kind} installed: {', '.join(added)}")

    typer.echo(f"  Registered '{registered_name}' ({agent_id[:8]}...)")
    return result


@agent_team_app.command("register")
def agent_team_register(
    url: str = typer.Argument(..., help="URL or path to setup.json template"),
    username: Optional[str] = typer.Option(
        None, "--username", "-u",
        help="Your ClawMeets username (defaults to current_user from a saved session).",
    ),
    password: Optional[str] = typer.Option(
        None, "--password", "-p",
        help="Your ClawMeets password (only needed when no saved session exists).",
    ),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR, "--data-dir",
        help="Root data directory (agents saved to {data_dir}/agents/)",
    ),
    agent: List[str] = typer.Option(
        None, "--agent",
        help=(
            "Register only these agents from the template (repeatable; matches "
            "setup.json agent `name`). Default: register every agent in the template."
        ),
    ),
    llm_provider: Optional[str] = typer.Option(
        None, "--llm-provider",
        help=(
            "Override LLM provider for every worker registered in this run. "
            "One of: claude, openai, gemini, opencode (CLI) or claude-api, "
            "openai-api, gemini-api, openrouter-api (in-process BYO-key). "
            "Wins over per-agent llm_provider in setup.json."
        ),
    ),
    llm_base_url: Optional[str] = typer.Option(
        None, "--llm-base-url",
        help=(
            "Override the OpenAI-compatible endpoint (openai-api) for every "
            "worker in this run, e.g. local ollama 'http://localhost:11434/v1'. "
            "Wins over per-agent llm_base_url in setup.json."
        ),
    ),
    no_personalize: bool = typer.Option(
        False, "--no-personalize",
        help="Skip the personalize-trigger DM fan-out after registering.",
    ),
):
    """Bulk-register a team of worker agents from a setup.json template.

    For each registered worker, by default a personalize-trigger DM is posted
    into the worker's per-agent DM project (`user-communication`) so the
    worker self-personalizes on its own runner. Pass `--no-personalize` to
    skip the fan-out.

    Example:
        clawmeets agent-team register https://example.com/team.json
        clawmeets agent-team register ./team.json -u alice -p secret --no-personalize
    """
    if llm_provider is not None and llm_provider.lower() not in _VALID_LLM_PROVIDERS:
        typer.echo(
            f"Error: --llm-provider must be one of {_VALID_LLM_PROVIDERS} (got {llm_provider!r}).",
            err=True,
        )
        raise typer.Exit(1)

    template = _fetch_setup_template(url)
    agents_list = template.get("agents") or []
    if not agents_list:
        typer.echo("Error: Template contains no agent definitions.", err=True)
        raise typer.Exit(1)

    if agent:
        requested = list(dict.fromkeys(agent))
        available = {a["name"] for a in agents_list}
        unknown = [n for n in requested if n not in available]
        if unknown:
            typer.echo(
                f"Error: --agent name(s) not in template: {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(available))}.",
                err=True,
            )
            raise typer.Exit(1)
        keep = set(requested)
        total = len(agents_list)
        agents_list = [a for a in agents_list if a["name"] in keep]
        typer.echo(
            f"  --agent filter: registering {len(agents_list)} of {total} "
            f"({', '.join(a['name'] for a in agents_list)})"
        )
    else:
        typer.echo(
            f"  Registering {len(agents_list)} agent(s): "
            f"{', '.join(a['name'] for a in agents_list)}"
        )

    if username and password:
        token = _login_for_jwt(server, username, password)
    else:
        _, token = _resolve_user_session(data_dir, None, server, as_user=username)
        if not username:
            username = get_current_user(Path(data_dir).expanduser())
        if not username:
            typer.echo(
                "Error: not logged in. Pass -u/-p, or run "
                "`clawmeets user login <user> <pass> --save` first.",
                err=True,
            )
            raise typer.Exit(1)

    agents_dir = Path(data_dir).expanduser() / "agents"

    registered_names: list[str] = []
    with _http(server) as client:
        for entry in agents_list:
            result = _register_one_worker(
                client, token, entry, agents_dir,
                llm_provider_override=llm_provider,
                llm_base_url_override=llm_base_url,
            )
            if result is not None:
                registered_names.append(result.get("agent_name") or entry["name"])

        if not no_personalize and registered_names:
            typer.echo("\n  Posting personalize-trigger DMs...")
            for worker_name in registered_names:
                if _post_dm_marker(
                    client, token, worker_name, _PERSONALIZE_TRIGGER_MARKER + "\n",
                ):
                    typer.echo(f"    -> {worker_name}")

    typer.echo("\nDone. Run `clawmeets start` to bring your agents online.")
