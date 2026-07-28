# SPDX-License-Identifier: MIT
"""
clawmeets/cli_client.py

Client-side resource CLI commands for clawmeets — pure HTTP against a running
server, no server-side imports.

Extracted out of cli_server.py so they ship in the runner wheel: cli_server
imports uvicorn + clawmeets.server.app at module top, which the wheel does not
carry, but these 16 commands are shelled by bundled system skills
(propose-project, manage-project-roster, post-chat-message,
project-completion-report, coding-project-completion-report, schedule-message)
that DO ship. Same move as cli_skill / cli_env / cli_consult before it.

Commands
--------
  project create/list/get/complete/allowlist/upsert-report/delete-report/cancel/delete
  chatroom list/create
  message send/clear/list
  file upload/list
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer

# Reuse helpers from cli_runner (MIT, already in the wheel).
from clawmeets.cli_runner import (
    _http, _ok, _print_json,
    _resolve_user_session, _resolve_project_ref,
    DEFAULT_SERVER, DEFAULT_DATA_DIR,
)

proj_app = typer.Typer(help="Project commands", no_args_is_help=True)
room_app = typer.Typer(help="Chatroom commands", no_args_is_help=True)
msg_app  = typer.Typer(help="Message commands", no_args_is_help=True)
file_app = typer.Typer(help="File commands",    no_args_is_help=True)


# ---------------------------------------------------------------------------
# project create / list / get / complete / delete
# ---------------------------------------------------------------------------

@proj_app.command("create")
def project_create(
    name: str = typer.Argument(...),
    coordinator_id: str = typer.Argument(..., help="Agent ID of the coordinator"),
    request: str = typer.Argument(..., help="User request / task description"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    created_by: str = typer.Option(None, "--created-by", "-u", help="User ID of creator"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Auth token (coordinator agent token or user JWT)"),
    agent_pool: str = typer.Option("verified", "--agent-pool", help="Agent pool: owned, verified (default), or all"),
    team: list[str] = typer.Option(
        None, "--team",
        help="Restrict the candidate agent pool to agents on this user_team (repeatable). "
             "Composes via OR with --agent. Hard-enforced at chatroom create. "
             "Omit both --team and --agent for no filter.",
    ),
    agent: list[str] = typer.Option(
        None, "--agent",
        help="Restrict the candidate agent pool to a specific agent (repeatable). "
             "Accepts the agent's id, full registry name, or owner-relative short name. "
             "Composes via OR with --team. Hard-enforced at chatroom create.",
    ),
    post_initial_message: bool = typer.Option(
        True, "--post-initial-message/--no-post-initial-message",
        help="Post the request as the opening user-communication message to wake the "
             "coordinator (default: on). Use --no-post-initial-message to create quietly.",
    ),
):
    """Create a new project."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with _http(server) as client:
        payload = {
            "name": name,
            "coordinator_id": coordinator_id,
            "request": request,
            "agent_pool": agent_pool,
            "post_initial_message": post_initial_message,
        }
        if created_by:
            payload["created_by"] = created_by
        if team:
            payload["agent_teams"] = team
        if agent:
            payload["agent_names"] = agent
        resp = client.post("/projects", json=payload, headers=headers)
        _print_json(_ok(resp))


@proj_app.command("list")
def project_list(
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    full: bool = typer.Option(False, "--full", "-f", help="Show full IDs"),
    token: Optional[str] = typer.Option(
        None, "--token", "-t",
        help="Auth token (user JWT or agent token). Omit inside an agent runner, "
             "where the per-process agent identity is sent automatically.",
    ),
):
    """List the projects you can see.

    ``GET /projects`` is authenticated and caller-scoped: a user gets their own
    projects (owned + Front Desk requester ends + shared-to-them), an admin gets
    all of them. Shelled inside a runner the agent identity in ``_http``'s
    default headers authenticates as the agent's owner, so ``--token`` is only
    needed for interactive use against a server with no saved session.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with _http(server) as client:
        projects = _ok(client.get("/projects", headers=headers))
    if not projects:
        typer.echo("No projects.")
        return
    for p in projects:
        pid = p['id'] if full else f"{p['id'][:8]}…"
        label = p.get('display_name') or p['name']
        typer.echo(f"  [{p['status']:6s}] {label:28s}  name={p['name']}  id={pid}  {p['request'][:50]}")


@proj_app.command("get")
def project_get(
    project_id: str = typer.Argument(..., help="Project ID"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
):
    """Get a project by ID."""
    with _http(server) as client:
        resp = client.get(f"/projects/{project_id}")
        _print_json(_ok(resp))


@proj_app.command("complete")
def project_complete(
    project_id: str = typer.Argument(...),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
):
    """Mark a project as completed and broadcast PROJECT_COMPLETED to all participants."""
    with _http(server) as client:
        _ok(client.post(f"/projects/{project_id}/complete"))
    typer.echo(f"Project {project_id[:8]}… marked as completed.")


@proj_app.command("allowlist")
def project_allowlist(
    project_id: str = typer.Argument(...),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Auth token (coordinator agent token or user JWT)"),
    agent: list[str] = typer.Option(
        None, "--agent",
        help="Agent to ADD to the invitable allowlist (repeatable). "
             "Accepts id, full registry name, or owner-relative short name.",
    ),
    team: list[str] = typer.Option(
        None, "--team", help="user_team to ADD to the invitable allowlist (repeatable).",
    ),
    remove_agent: list[str] = typer.Option(
        None, "--remove-agent", help="Agent to REMOVE from the allowlist (repeatable).",
    ),
    remove_team: list[str] = typer.Option(
        None, "--remove-team", help="user_team to REMOVE from the allowlist (repeatable).",
    ),
    replace: bool = typer.Option(
        False, "--replace",
        help="Replace the whole allowlist with --agent/--team instead of merging.",
    ),
):
    """Add/remove agents or teams on a project's invitable allowlist.

    Default is MERGE (union with --agent/--team, then drop --remove-*).
    Pass --replace to SET the allowlist to exactly --agent/--team.

    Auth: with --token, that token + --server are used. Without --token, falls
    back to the agent-runtime env injection (CLAWMEETS_SERVER_URL/AGENT_ID/
    AGENT_TOKEN) so a coordinator can edit its own project's roster in-runtime
    with no token juggling — same pattern as ``project upsert-report``. The
    server gates on ``project.coordinator_id`` (or owner/admin JWT).
    """
    payload = {
        "add_agents": agent or [],
        "add_teams": team or [],
        "remove_agents": remove_agent or [],
        "remove_teams": remove_team or [],
        "replace": replace,
    }
    if token:
        headers = {"Authorization": f"Bearer {token}"}
        client = _http(server)
    else:
        client, headers = _agent_http()
    with client:
        proj = _ok(client.put(f"/projects/{project_id}/allowlist", json=payload, headers=headers))
    typer.echo(
        f"Allowlist for {proj['name']} updated — "
        f"agent_names={proj['agent_names']} agent_teams={proj['agent_teams']}"
    )


def _agent_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        typer.echo(
            f"Error: ${name} is not set. ``clawmeets project upsert-report`` "
            f"runs inside the coordinator's agent runtime, which injects "
            f"CLAWMEETS_SERVER_URL/AGENT_ID/AGENT_TOKEN.",
            err=True,
        )
        raise typer.Exit(1)
    return val


def _agent_http():
    import httpx
    server = _agent_env("CLAWMEETS_SERVER_URL").rstrip("/")
    headers = {
        "Authorization": f"Bearer {_agent_env('CLAWMEETS_AGENT_TOKEN')}",
        "X-Agent-ID": _agent_env("CLAWMEETS_AGENT_ID"),
    }
    return httpx.Client(base_url=server, timeout=30), headers


def _read_json(path: Path) -> dict | list:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        typer.echo(f"Error reading {path}: {e}", err=True)
        raise typer.Exit(1) from e
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        typer.echo(f"Error: {path} is not valid JSON: {e}", err=True)
        raise typer.Exit(1) from e


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        typer.echo(f"Error reading {path}: {e}", err=True)
        raise typer.Exit(1) from e


@proj_app.command("upsert-report")
def project_upsert_report(
    project_id: str = typer.Argument(..., help="Project ID (the coordinator's own project)."),
    title: str = typer.Option(
        "", "--title",
        help="Report title shown at the top of the rendered tab.",
    ),
    data: Path = typer.Option(
        ..., "--data",
        exists=True, file_okay=True, dir_okay=False, readable=True,
        help="Path to data.json (any JSON your render code understands; ≤ 64 KB).",
    ),
    render_code: Path = typer.Option(
        ..., "--render-code",
        exists=True, file_okay=True, dir_okay=False, readable=True,
        help="Path to render.js — BODY of function(mount, data, lib); ≤ 64 KB.",
    ),
) -> None:
    """Publish (or replace) the interactive completion report for a project.

    Coordinator-only — server checks the agent token against
    ``project.coordinator_id``. Auth resolved from the standard agent-runtime
    env injection (CLAWMEETS_SERVER_URL/AGENT_ID/AGENT_TOKEN), same pattern
    as ``clawmeets brief upsert-tab``.

    Re-running overwrites; the frontend re-renders in place via
    PROJECT_REPORT_SYNC.
    """
    body = {
        "title": title,
        "data": _read_json(data),
        "render_code_js": _read_text(render_code),
    }
    client, headers = _agent_http()
    with client:
        resp = client.put(
            f"/projects/{project_id}/report", json=body, headers=headers
        )
    if resp.status_code >= 400:
        typer.echo(f"Error {resp.status_code}: {resp.text}", err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(resp.json(), indent=2, ensure_ascii=False))


@proj_app.command("delete-report")
def project_delete_report(
    project_id: str = typer.Argument(..., help="Project ID."),
) -> None:
    """Delete a project's report. Coordinator-only (env-injected token)."""
    client, headers = _agent_http()
    with client:
        resp = client.delete(
            f"/projects/{project_id}/report", headers=headers
        )
    if resp.status_code >= 400:
        typer.echo(f"Error {resp.status_code}: {resp.text}", err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(resp.json(), indent=2, ensure_ascii=False))


@proj_app.command("cancel")
def project_cancel(
    project_id: str = typer.Argument(..., help="Project ID"),
    chatroom_name: str = typer.Argument(..., help="Chatroom name"),
    agent_id: str = typer.Argument(..., help="Agent ID whose in-flight LLM invocation to cancel"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    token: str = typer.Option(..., "--token", "-t", help="User JWT token (must be the project owner)"),
):
    """Cancel an agent's in-flight LLM invocation in a chatroom.

    Tells the agent's runner to kill the running CLI subprocess and posts an
    ack message ("Message canceled.") to the chatroom.
    """
    with _http(server) as client:
        result = _ok(client.post(
            f"/projects/{project_id}/chatrooms/{chatroom_name}/cancel",
            headers={"Authorization": f"Bearer {token}"},
            json={"agent_id": agent_id},
        ))
    if result.get("delivered"):
        typer.echo(f"Cancel sent to runner for agent {agent_id[:8]}…; ack message {result.get('message_id', '?')[:8]}…")
    else:
        typer.echo(f"Agent {agent_id[:8]}… is not connected (no runner to cancel); ack message posted: {result.get('message_id', '?')[:8]}…")


@proj_app.command("delete")
def project_delete(
    project_id: str = typer.Argument(..., help="Project ID to delete"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    token: str = typer.Option(..., "--token", "-t", help="User JWT token"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
):
    """Delete a project and all its data. Requires user JWT token."""
    if not force:
        confirm = typer.confirm(f"Delete project {project_id[:8]}…? This cannot be undone")
        if not confirm:
            raise typer.Abort()

    with _http(server) as client:
        _ok(client.delete(
            f"/projects/{project_id}",
            headers={"Authorization": f"Bearer {token}"},
        ))
    typer.echo(f"Project {project_id[:8]}… deleted.")


# ---------------------------------------------------------------------------
# chatroom list / create
# ---------------------------------------------------------------------------

@room_app.command("list")
def chatroom_list(
    project_id: str = typer.Argument(...),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    full: bool = typer.Option(False, "--full", "-f", help="Show full IDs"),
):
    """List chatrooms in a project."""
    with _http(server) as client:
        rooms = _ok(client.get(f"/projects/{project_id}/chatrooms"))
    for r in rooms:
        participants = ", ".join(r.get("participants", []))
        rid = r['id'] if full else f"{r['id'][:8]}…"
        typer.echo(f"  {r['name']:20s}  id={rid}  participants=[{participants[:60]}]")


@room_app.command("create")
def chatroom_create(
    project_id: str = typer.Argument(...),
    name: str = typer.Argument(...),
    participants: str = typer.Argument(..., help="Comma-separated list of agent IDs or names"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
):
    """Create a new chatroom in a project with participants."""
    participant_list = [p.strip() for p in participants.split(",") if p.strip()]
    with _http(server) as client:
        _print_json(_ok(client.post(
            f"/projects/{project_id}/chatrooms",
            json={"name": name, "participants": participant_list}
        )))


# ---------------------------------------------------------------------------
# message send / list
# ---------------------------------------------------------------------------


def _resolve_message_session(
    data_dir: Path,
    token: Optional[str],
    agent_id: Optional[str],
    server: Optional[str],
    as_user: bool,
) -> tuple[str, str, Optional[str]]:
    """Resolve (server_url, token, agent_id_or_none) for the message commands.

    Agent path (returned ``agent_id`` is not None — caller sends
    ``X-Agent-ID``): explicit ``--agent-id``, or the runner-injected
    ``$CLAWMEETS_AGENT_ID`` / ``$CLAWMEETS_AGENT_TOKEN`` pair present in
    every agent subprocess. ``--as-user`` skips it so an assistant (whose
    env carries both kinds of token) can act as the user instead.

    User path (``agent_id`` is None): user JWT or assistant token via
    ``_resolve_user_session`` (--token → $CLAWMEETS_ASSISTANT_TOKEN →
    $CLAWMEETS_USER_TOKEN → saved session).
    """
    env_agent_id = os.environ.get("CLAWMEETS_AGENT_ID")
    env_agent_token = os.environ.get("CLAWMEETS_AGENT_TOKEN")
    if not as_user and (agent_id or (env_agent_id and env_agent_token)):
        resolved_id = agent_id or env_agent_id
        resolved_token = token or env_agent_token
        if not resolved_token:
            typer.echo(
                "Error: --agent-id requires --token (or run inside an agent "
                "subprocess where $CLAWMEETS_AGENT_TOKEN is set).",
                err=True,
            )
            raise typer.Exit(1)
        server_url = (
            server or os.environ.get("CLAWMEETS_SERVER_URL") or DEFAULT_SERVER
        ).rstrip("/")
        return server_url, resolved_token, resolved_id
    server_url, user_token = _resolve_user_session(data_dir, token, server)
    return server_url, user_token, None


@msg_app.command("send")
def message_send(
    project: str = typer.Argument(..., help="Project name or ID"),
    chatroom_name: str = typer.Argument(..., help="Chatroom name"),
    content: str = typer.Argument(...),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Agent token (with --agent-id) or user JWT / assistant token"),
    agent_id: Optional[str] = typer.Option(None, "--agent-id", help="Post as this agent (default: $CLAWMEETS_AGENT_ID when set)"),
    as_user: bool = typer.Option(False, "--as-user", help="Post as the user even when agent env credentials are present"),
    source_version: Optional[int] = typer.Option(None, "--source-version", help="Changelog version this message reacts to (agent path only)"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """Post a text message to a project chatroom.

    The sender derives from credentials: with agent credentials (explicit
    ``--agent-id``/``--token`` or the runner-injected agent env) the message
    posts as that agent, which must be a participant of the chatroom
    (server-enforced). Otherwise it posts as the user, resolved from a user
    JWT or assistant token.
    """
    server_url, resolved_token, resolved_agent_id = _resolve_message_session(
        data_dir, token, agent_id, server, as_user
    )
    with _http(server_url) as client:
        if resolved_agent_id:
            project_id = _resolve_project_ref(
                client, None, project, agent_id=resolved_agent_id
            )
            result = _ok(client.post(
                f"/projects/{project_id}/chatrooms/{chatroom_name}/messages",
                json={"content": content, "source_version": source_version},
                headers={
                    "Authorization": f"Bearer {resolved_token}",
                    "X-Agent-ID": resolved_agent_id,
                },
            ))
        else:
            project_id = _resolve_project_ref(client, resolved_token, project)
            result = _ok(client.post(
                f"/projects/{project_id}/chatrooms/{chatroom_name}/user-message",
                json={"content": content},
                headers={"Authorization": f"Bearer {resolved_token}"},
            ))
    _print_json(result)


@msg_app.command("clear")
def message_clear(
    project_id: str = typer.Argument(...),
    chatroom_name: str = typer.Argument(..., help="Chatroom name"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="User JWT (defaults to saved session)"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation prompt"),
):
    """Wipe a chatroom's message history.

    The room itself remains; files under files/ are untouched. Prior messages
    are archived to a CHATS.ndjson.cleared-<ts>.bak sibling on every node
    (server + runners). Requires a user JWT — the caller must own the project
    (or be the external Front Desk requester, limited to user-communication).
    """
    server_url, token = _resolve_user_session(data_dir, token, server)
    if not force:
        confirm = typer.confirm(
            f"Clear all messages in {chatroom_name!r} (project {project_id[:8]}…)? "
            f"This cannot be undone"
        )
        if not confirm:
            raise typer.Abort()
    with _http(server_url) as client:
        resp = client.delete(
            f"/projects/{project_id}/chatrooms/{chatroom_name}/messages",
            headers={"Authorization": f"Bearer {token}"},
        )
        try:
            resp.raise_for_status()
        except Exception:
            typer.echo(f"Error {resp.status_code}: {resp.text}", err=True)
            raise typer.Exit(1)
    typer.echo(f"Cleared history of {chatroom_name!r} in project {project_id[:8]}…")


@msg_app.command("list")
def message_list(
    project: str = typer.Argument(..., help="Project name or ID"),
    chatroom_name: str = typer.Argument(..., help="Chatroom name"),
    since: Optional[str] = typer.Option(None, "--since", help="Show messages after this message ID"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Agent token (with --agent-id) or user JWT / assistant token"),
    agent_id: Optional[str] = typer.Option(None, "--agent-id", help="Read as this agent (default: $CLAWMEETS_AGENT_ID when set)"),
    as_user: bool = typer.Option(False, "--as-user", help="Read as the user even when agent env credentials are present"),
    server: Optional[str] = typer.Option(None, "--server", "-s"),
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir"),
):
    """List messages in a chatroom."""
    server_url, resolved_token, resolved_agent_id = _resolve_message_session(
        data_dir, token, agent_id, server, as_user
    )
    params = {"since": since} if since else {}
    headers = {"Authorization": f"Bearer {resolved_token}"}
    if resolved_agent_id:
        headers["X-Agent-ID"] = resolved_agent_id
    with _http(server_url) as client:
        project_id = _resolve_project_ref(
            client,
            None if resolved_agent_id else resolved_token,
            project,
            agent_id=resolved_agent_id,
        )
        messages = _ok(client.get(
            f"/projects/{project_id}/chatrooms/{chatroom_name}/messages",
            params=params,
            headers=headers,
        ))
    for m in messages:
        ts = m.get("ts", "")[:19]
        frm = m.get("from_participant_name") or m.get("from_participant_id", "?")
        body = m.get("content") or f"[file: {m.get('filename')}]"
        typer.echo(f"  {ts}  {frm:20s}  {body[:80]}")


# ---------------------------------------------------------------------------
# file upload / list
# ---------------------------------------------------------------------------

@file_app.command("upload")
def file_upload(
    project_id: str = typer.Argument(...),
    chatroom_name: str = typer.Argument(..., help="Chatroom name"),
    filepath: Path = typer.Argument(..., help="Local file to upload"),
    remote_name: Optional[str] = typer.Option(None, "--name", help="Remote filename (default: same as local)"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="User JWT or agent token"),
    agent_id: Optional[str] = typer.Option(None, "--agent-id", help="Agent ID (sent as X-Agent-ID; required when --token is an agent token)"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
):
    """Upload a file to a chatroom.

    With ``--token <user-jwt>`` and no ``--agent-id``, the upload is recorded
    as coming from the project coordinator (used to seed shared-context files
    like PRD.md / AUDIT_PROCEDURE.md). With ``--token <agent-token> --agent-id
    <id>``, the legacy agent path is used.
    """
    if not filepath.exists():
        typer.echo(f"File not found: {filepath}", err=True)
        raise typer.Exit(1)
    fname = remote_name or filepath.name
    content = filepath.read_bytes()
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if agent_id:
        headers["X-Agent-ID"] = agent_id
    with _http(server) as client:
        resp = client.put(
            f"/projects/{project_id}/chatrooms/{chatroom_name}/files/{fname}",
            content=content,
            headers=headers,
        )
        _ok(resp)
    typer.echo(f"Uploaded {fname} ({len(content):,} bytes) → chatroom {chatroom_name}")


@file_app.command("list")
def file_list(
    project_id: str = typer.Argument(...),
    chatroom_name: str = typer.Argument(..., help="Chatroom name"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
):
    """List files in a chatroom."""
    with _http(server) as client:
        files = _ok(client.get(
            f"/projects/{project_id}/chatrooms/{chatroom_name}/files"
        ))
    if not files:
        typer.echo("(empty)")
    for f in files:
        typer.echo(f"  {f}")
