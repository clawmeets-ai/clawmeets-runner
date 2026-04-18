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
from clawmeets.utils.notification_center import NotificationCenter
from clawmeets.llm.claude_cli import ClaudeCLI
from clawmeets.api.actions import WORKER_ACTION_SCHEMA, COORDINATOR_ACTION_SCHEMA
from clawmeets.models.context import ModelContext
from clawmeets.models.agent import Agent
from clawmeets.models.assistant import Assistant
from clawmeets.models.user import User, NotificationConfig
from clawmeets.sync.console_subscriber import ConsoleOutputSubscriber, ConsoleConfig
from clawmeets.runner.reactive_loop import ReactiveControlLoop
from clawmeets.runner.skill_manager import SkillManager

# Backward compatibility alias
AgentRegistrationResult = AgentRegistrationResponse

# Sub-command groups
agent_app = typer.Typer(help="Agent commands", no_args_is_help=True)
user_app  = typer.Typer(help="User commands",  no_args_is_help=True)
dm_app    = typer.Typer(help="Direct message commands", no_args_is_help=True)


# ---------------------------------------------------------------------------
# Global options (env-var defaults)
# ---------------------------------------------------------------------------

DEFAULT_SERVER = os.environ.get("CLAWMEETS_SERVER_URL", "https://clawmeets.ai")
DEFAULT_DATA_DIR = os.environ.get("CLAWMEETS_DATA_DIR", str(Path.home() / ".clawmeets"))
DEFAULT_AGENTS_DIR = str(Path(DEFAULT_DATA_DIR) / "agents")
DEFAULT_USERS_DIR = str(Path(DEFAULT_DATA_DIR) / "users")
DEFAULT_SERVER_DIR = str(Path(DEFAULT_DATA_DIR) / "server")


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


# ---------------------------------------------------------------------------
# agent register
# ---------------------------------------------------------------------------

@agent_app.command("register")
def agent_register(
    name: Optional[str] = typer.Argument(None, help="Agent name (required unless --from-card)"),
    description: Optional[str] = typer.Argument(None, help="Short description (required unless --from-card)"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Admin JWT token (required)"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    agent_dir: Path = typer.Option(DEFAULT_AGENTS_DIR, "--agent-dir", help="Base directory for agents"),
    save: Optional[Path] = typer.Option(None, "--save", help="Save credentials to custom path (overrides agent-dir)"),
    discoverable: Optional[bool] = typer.Option(None, "--discoverable/--no-discoverable", help="Show agent in /agents list"),
    capabilities: Optional[str] = typer.Option(None, "--capabilities", "-c", help="Comma-separated list of capabilities"),
    from_card: Optional[Path] = typer.Option(None, "--from-card", help="Path to card.json to register from"),
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
    if not token:
        typer.echo("Error: --token is required. Get admin token with: user login admin <password>", err=True)
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

    with _http(server) as client:
        resp = client.post(
            "/agents/register",
            json={
                "name": name,
                "description": description,
                "capabilities": caps_list,
                "discoverable_through_registry": final_discoverable,
            },
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
        # Save to {agent-dir}/{registered_name}-{agent_id}/credential.json
        agent_id = result["agent_id"]
        agent_work_dir = Path(agent_dir) / f"{registered_name}-{agent_id}"
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
        card_path = agent_work_dir / "card.json"
        card_path.write_text(json.dumps(card, indent=2, default=str))
        typer.echo(f"Card saved to {card_path}")


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
    credentials: Optional[Path] = typer.Argument(None, help="JSON credentials file (optional if credential.json exists in agent-dir)"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    agent_dir: Path = typer.Option(DEFAULT_AGENTS_DIR, "--agent-dir"),
    working_dir: Optional[Path] = typer.Option(None, "--working-dir", "-w", help="Sandbox directory for Claude (default: agent-dir/sandbox)"),
    knowledge_dir: Optional[Path] = typer.Option(None, "--knowledge-dir", "-k", help="Knowledge base directory (passed as --add-dir to Claude)"),
    claude_plugin_dir: Optional[list[Path]] = typer.Option(None, "--claude-plugin-dir", help="Claude plugin directory (passed as --plugin-dir to Claude CLI, repeatable)"),
    chrome: bool = typer.Option(False, "--chrome", help="Enable Chrome browser integration via Claude Code --chrome flag"),
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
    --add-dir to Claude, enabling access to a knowledge base generated by
    'clawmeets generate crawler'.

    When --claude-plugin-dir is specified, the directory is passed as --plugin-dir
    to Claude CLI, enabling access to Claude Code plugins/skills (e.g.,
    save-to-knowledge). Can be repeated for multiple plugin directories.
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

    asyncio.run(_runner_loop(agent_name, agent_id, token, server, Path(agent_dir), working_dir, knowledge_dir, claude_plugin_dir or [], use_chrome=chrome))


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
):
    """Login as a user and print the JWT token."""
    with _http(server) as client:
        resp = client.post("/auth/login", json={"username": username, "password": password})
        result = _ok(resp)
    typer.echo(result["token"])


@user_app.command("register")
def user_register(
    username: str = typer.Argument(..., help="Username"),
    password: str = typer.Argument(..., help="Password"),
    email: str = typer.Argument(..., help="Email address"),
    invitation_code: str = typer.Option(..., "--invitation-code", "-i", help="Invitation code (required)"),
    agree_tos: bool = typer.Option(False, "--agree-tos", help="Agree to Terms of Service and Privacy Policy"),
    server: str = typer.Option(DEFAULT_SERVER, "--server", "-s"),
    agent_dir: Path = typer.Option(DEFAULT_AGENTS_DIR, "--agent-dir", help="Base directory for agents"),
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

        assistant_creds = {
            "agent_id": agent_id,
            "token": agent_token,
            "agent_name": agent_name,
        }

        assistant_dir = Path(agent_dir) / f"{agent_name}-{agent_id}"
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
        }
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
    agent_dir: Path = typer.Option(DEFAULT_AGENTS_DIR, "--agent-dir", help="Base directory for agents"),
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

        # Build credentials structure for saving
        assistant_creds = {
            "agent_id": agent_id,
            "token": agent_token,
            "agent_name": agent_name,
        }

        # Save assistant credentials to agent directory
        assistant_dir = Path(agent_dir) / f"{agent_name}-{agent_id}"
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
        }
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
    user_dir: Path = typer.Option(DEFAULT_USERS_DIR, "--user-dir", help="Base directory for user listener data"),
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

    typer.echo(f"Authenticated as {username} (user_id={user_id[:8]}...)")
    typer.echo(f"Server: {server}  |  User dir: {user_dir}")
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
        user_base_dir=Path(user_dir),
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

        except (websockets.ConnectionClosed, OSError) as e:
            logging.warning(f"WebSocket disconnected: {e}. Reconnecting in {reconnect_delay}s…")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)
        except asyncio.CancelledError:
            await loop_obj.stop()
            await http_client.aclose()
            break


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
) -> None:
    """Run the reactive control loop for an agent."""
    agent_dir.mkdir(parents=True, exist_ok=True)

    # Check card.json to determine if this is an assistant or worker agent
    card_path = agent_dir / "card.json"
    if not card_path.exists():
        typer.echo(f"Error: card.json not found at {card_path}", err=True)
        raise typer.Exit(1)
    card_data = json.loads(card_path.read_text())
    # Assistants have discoverable_through_registry=False
    is_assistant = not card_data.get("discoverable_through_registry", True)

    # Read local_settings from card.json (primary source, synced with server)
    # CLI flags (--knowledge-dir, --chrome) serve as overrides for backward compat
    local_settings = card_data.get("local_settings", {})
    effective_knowledge_dir = knowledge_dir or local_settings.get("knowledge_dir", "")
    effective_use_chrome = use_chrome or local_settings.get("use_chrome", False)

    # Determine action schema based on role
    if is_assistant:
        action_schema = COORDINATOR_ACTION_SCHEMA
    else:
        action_schema = WORKER_ACTION_SCHEMA

    # Set up skill manager (downloads skills from server on startup)
    skill_manager = SkillManager(agent_dir)

    # Set up Claude CLI with role-appropriate schema (fail early if CLI not available)
    # Include skill-hub plugin dir alongside any explicit plugin dirs
    all_plugin_dirs = list(claude_plugin_dirs or []) + [skill_manager.plugin_dir]
    ClaudeCLI.verify_cli()
    cli = ClaudeCLI(action_schema=action_schema, claude_plugin_dirs=all_plugin_dirs, use_chrome=effective_use_chrome)

    # Build knowledge_dirs list (e.g., knowledge bases)
    knowledge_dirs_list: list[Path] = []
    if effective_knowledge_dir:
        knowledge_dirs_list.append(Path(effective_knowledge_dir))

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

    # Create appropriate participant type
    if is_assistant:
        participant = Assistant(id=agent_id, model_ctx=model_ctx)
    else:
        participant = Agent(id=agent_id, model_ctx=model_ctx)

    # Build reactive control loop
    loop_obj = ReactiveControlLoop(
        participant=participant,
        client=client,
        model_ctx=model_ctx,
        extra_subscribers=[],
        skill_manager=skill_manager,
    )

    # Start the loop
    await loop_obj.start()

    # Convert http:// → ws://
    ws_url = server_http.replace("https://", "wss://").replace("http://", "ws://")
    ws_connect_url = f"{ws_url}/ws/{agent_id}?token={token}"

    handle_exception = _create_dispatch_callback()
    reconnect_delay = 2.0

    while True:
        try:
            async with websockets.connect(ws_connect_url) as ws:
                logging.getLogger("clawmeets").info(f"WebSocket connected to {ws_url}")

                # Send auth message (server expects this as first message)
                await ws.send(json.dumps({"token": token}))

                reconnect_delay = 2.0  # reset on success

                # Sync installed skills from server (catch-up on connect/reconnect)
                await skill_manager.sync_from_server(client, agent_id)

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

        except (websockets.ConnectionClosed, OSError) as e:
            logging.warning(f"WebSocket disconnected: {e}. Reconnecting in {reconnect_delay}s…")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)
        except asyncio.CancelledError:
            await loop_obj.stop()
            await http_client.aclose()
            break


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
    agent_dir: Path = typer.Option(DEFAULT_AGENTS_DIR, "--agent-dir", help="Base directory for agents"),
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
        assistant_cred_path = None
        for entry in Path(agent_dir).iterdir():
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
    agent_dir: Path = typer.Option(DEFAULT_AGENTS_DIR, "--agent-dir", help="Base directory for agents"),
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
        assistant_token = None
        for entry in Path(agent_dir).iterdir():
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
