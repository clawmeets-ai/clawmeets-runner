# SPDX-License-Identifier: MIT
"""
clawmeets/cli_skill.py — client-side skill-hub + scheduled-message CLIs.

Homes the ``clawmeets skill`` and ``clawmeets schedule`` command groups. Both
are pure HTTP clients: they import only ``_http``/``_ok``/``_print_json`` and
the session/project resolvers from ``cli_runner`` — **no** ``uvicorn`` or
``clawmeets.server`` imports. That keeps this module importable (and therefore
shippable) inside the runner wheel, unlike ``cli_server`` where these two
groups used to live. ``cli_server`` re-exports ``skill_app``/``schedule_app``
from here so the full CLI and any other importer keep working unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from clawmeets.cli_runner import (
    _http, _ok, _print_json,
    _resolve_user_session, _resolve_project_ref, _resolve_session_for_config,
    DEFAULT_SERVER, DEFAULT_DATA_DIR,
)

# Sub-command groups
schedule_app = typer.Typer(help="Scheduled message commands", no_args_is_help=True)
skill_app    = typer.Typer(help="Skill hub commands", no_args_is_help=True)


# ---------------------------------------------------------------------------
# schedule create / list / cancel
# ---------------------------------------------------------------------------

@schedule_app.command("create")
def schedule_create(
    project: str = typer.Argument(..., help="Target project name or ID"),
    chatroom_name: str = typer.Argument(..., help="Target chatroom name"),
    content: Optional[str] = typer.Argument(None, help="Inline message content (omit when --file is used)"),
    cron: str = typer.Option(..., "--cron", "-c", help="Cron expression (e.g. '0 9 * * *', '@daily', '@hourly'); evaluated in UTC"),
    start_at: Optional[str] = typer.Option(None, "--start-at", help="First fire time (ISO 8601, e.g. '2026-04-05T09:00:00Z')"),
    end_at: Optional[str] = typer.Option(None, "--end-at", help="Expiration time (ISO 8601)"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Read message content from file (mutually exclusive with the content positional)"),
    idle_only: bool = typer.Option(False, "--idle-only", help="Skip cron ticks while the target project has open batches"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="User JWT or assistant token (defaults to $CLAWMEETS_ASSISTANT_TOKEN / $CLAWMEETS_USER_TOKEN / saved session)"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Create a scheduled message with a cron expression."""
    if content is None and file is None:
        raise typer.BadParameter("Provide either the content positional argument or --file PATH")
    if content is not None and file is not None:
        raise typer.BadParameter("Cannot use both the content positional argument and --file")

    if file is not None:
        if not file.exists():
            raise typer.BadParameter(f"File not found: {file}")
        content = file.read_text()

    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        project_id = _resolve_project_ref(client, token, project)
        payload: dict = {
            "project_id": project_id,
            "chatroom_name": chatroom_name,
            "content": content,
            "cron_expression": cron,
            "idle_only": idle_only,
        }
        if start_at:
            payload["start_at"] = start_at
        if end_at:
            payload["end_at"] = end_at

        resp = client.post(
            "/scheduled-messages",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        result = _ok(resp)
    _print_json(result)
    typer.echo(f"Scheduled message created: {result['id'][:8]}... next fire: {result['next_fire_at']}")


@schedule_app.command("list")
def schedule_list(
    token: Optional[str] = typer.Option(None, "--token", "-t", help="User JWT or assistant token (defaults to $CLAWMEETS_ASSISTANT_TOKEN / $CLAWMEETS_USER_TOKEN / saved session)"),
    all_: bool = typer.Option(False, "--all", "-a", help="Include inactive schedules"),
    full: bool = typer.Option(False, "--full", help="Print full message content (no truncation)"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """List your scheduled messages."""
    server_url, token = _resolve_user_session(data_dir, token, server)
    params = {"active_only": "false"} if all_ else {}
    with _http(server_url) as client:
        resp = client.get(
            "/scheduled-messages",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        schedules = _ok(resp)

    if not schedules:
        typer.echo("No scheduled messages.")
        return

    for s in schedules:
        status = "active" if s["is_active"] else "inactive"
        content = s["content"] if full else s["content"][:60]
        typer.echo(
            f"  [{status}] {s['id'][:8]}... "
            f"cron={s['cron_expression']!r} "
            f"room={s['chatroom_name']} "
            f"next={s['next_fire_at']} "
            f"content={content!r}"
        )


@schedule_app.command("cancel")
def schedule_cancel(
    schedule_id: str = typer.Argument(..., help="Scheduled message ID"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="User JWT or assistant token (defaults to $CLAWMEETS_ASSISTANT_TOKEN / $CLAWMEETS_USER_TOKEN / saved session)"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Cancel a scheduled message."""
    server_url, token = _resolve_user_session(data_dir, token, server)
    with _http(server_url) as client:
        resp = client.delete(
            f"/scheduled-messages/{schedule_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        _ok(resp)
    typer.echo(f"Scheduled message {schedule_id[:8]}... cancelled.")


# ---------------------------------------------------------------------------
# skill commands
# ---------------------------------------------------------------------------

@skill_app.command("list")
def skill_list(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Search query"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
):
    """List available skills in the hub."""
    with _http(server) as client:
        params = {}
        if query:
            params["q"] = query
        resp = client.get("/skills", params=params)
        skills = _ok(resp)
    if not skills:
        typer.echo("No skills found.")
        return
    for s in skills:
        tags = ", ".join(s.get("tags", []))
        typer.echo(f"  {s['name']:<20} {s['description']}")
        if tags:
            typer.echo(f"  {'':20} tags: {tags}")


@skill_app.command("show")
def skill_show(
    name: str = typer.Argument(..., help="Skill name"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
):
    """Show skill detail and SKILL.md preview."""
    with _http(server) as client:
        resp = client.get(f"/skills/{name}")
        data = _ok(resp)
    typer.echo(f"Name:        {data['name']}")
    typer.echo(f"Description: {data['description']}")
    typer.echo(f"Version:     {data.get('version', 'N/A')}")
    tags = ", ".join(data.get("tags", []))
    if tags:
        typer.echo(f"Tags:        {tags}")
    attribution = data.get("attribution")
    if attribution:
        typer.echo(f"Attribution: {attribution}")
    content = data.get("content")
    if content:
        typer.echo(f"\n--- SKILL.md preview (first 20 lines) ---")
        for line in content.splitlines()[:20]:
            typer.echo(line)
        typer.echo("---")


@skill_app.command("install")
def skill_install(
    agent: str = typer.Argument(..., help="Agent name or id, or 'self' (worker self-installs via runner-injected env)"),
    skills: List[str] = typer.Argument(..., help="One or more skill names to install"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Install one or more skills on an agent."""
    server_url, token, agent_id, agent_name = _resolve_session_for_config(
        data_dir, token, server, agent,
    )
    with _http(server_url) as client:
        resp = client.post(
            f"/agents/{agent_id}/skills",
            json={"skills": skills},
            headers={"Authorization": f"Bearer {token}"},
        )
        result = _ok(resp)
    added = result.get("added") or []
    if added:
        typer.echo(f"Installed on '{agent_name}': {', '.join(added)}")
    else:
        typer.echo(f"No new skills installed on '{agent_name}' (already present).")


@skill_app.command("uninstall")
def skill_uninstall(
    agent: str = typer.Argument(..., help="Agent name or id, or 'self' (worker self-uninstalls via runner-injected env)"),
    skill_name: str = typer.Argument(..., help="Skill name to uninstall"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Uninstall a skill from an agent."""
    server_url, token, agent_id, agent_name = _resolve_session_for_config(
        data_dir, token, server, agent,
    )
    with _http(server_url) as client:
        resp = client.delete(
            f"/agents/{agent_id}/skills/{skill_name}",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = _ok(resp)
    typer.echo(f"Uninstalled {data.get('removed', skill_name)!r} from '{agent_name}'.")


@skill_app.command("installed")
def skill_installed(
    agent: str = typer.Argument(..., help="Agent name or id, or 'self' (worker self-reads via runner-injected env)"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """List skills installed for an agent."""
    server_url, token, agent_id, agent_name = _resolve_session_for_config(
        data_dir, token, server, agent,
    )
    with _http(server_url) as client:
        resp = client.get(
            f"/agents/{agent_id}/skills",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = _ok(resp)
    skills = data.get("installed_skills", [])
    if not skills:
        typer.echo(f"No skills installed on '{agent_name}'.")
    else:
        typer.echo(f"Skills installed on '{agent_name}':")
        for s in skills:
            typer.echo(f"  {s}")


@skill_app.command("set-config")
def skill_set_config(
    agent: str = typer.Argument(
        ...,
        help="Agent name/id, or 'self' to target the calling agent "
             "(via env-injected CLAWMEETS_AGENT_{ID,TOKEN,SERVER_URL}).",
    ),
    skill_name: str = typer.Argument(..., help="Skill name"),
    config_file: Path = typer.Argument(..., help="JSON file with the config payload"),
    token: Optional[str] = typer.Option(None, "--token", "-t"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Set the per-agent config for a skill (uploads JSON from a file).

    Persists to card.json.local_settings.skill_configs[skill_name] and
    broadcasts AGENT_SETTINGS_CHANGE so the runner writes through to
    {agent_dir}/skill-hub/configs/<skill_name>.json. Use 'self' as the
    agent argument from inside an agent subprocess to update that agent's
    own config without a user session.
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
            f"/agents/{agent_id}/skills/{skill_name}/config",
            json={"config": payload},
            headers={"Authorization": f"Bearer {token}"},
        )
        _ok(resp)
    typer.echo(f"Set config for {skill_name!r} on '{agent_name}'.")


@skill_app.command("reload")
def skill_reload(
    token: str = typer.Option(..., "--token", "-t", help="Admin auth token"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
):
    """Reload skill registry from disk (admin only). Run after git pull to pick up new skills."""
    with _http(server) as client:
        resp = client.post(
            "/skills/reload",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = _ok(resp)
    typer.echo(f"Reloaded {data['reloaded']} skills: {', '.join(data['skills'])}")
