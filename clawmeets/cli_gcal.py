# SPDX-License-Identifier: MIT
"""
clawmeets/cli_gcal.py — Google Calendar CLI.

Subcommands: calendars, list-events, get, create, update, delete, sync, auth.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from clawmeets.integrations._config_resolve import resolve_skill_token_path
from clawmeets.integrations.gcal import _lib

app = typer.Typer(
    name="gcal",
    help="Google Calendar (list / get / create / update / delete / sync). Paired skill: google-calendar.",
    no_args_is_help=True,
)


def _svc(token: str):
    return _lib.build_service(resolve_skill_token_path("google-calendar", explicit=token))


def _emit_json(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command()
def calendars(token: str = typer.Option("", "--token")) -> None:
    """List the user's available calendars."""
    _emit_json(_lib.list_calendars(_svc(token)))


@app.command("list-events")
def list_events_cmd(
    calendar_id: str = typer.Option("primary", "--calendar"),
    time_min: Optional[str] = typer.Option(None, "--time-min"),
    time_max: Optional[str] = typer.Option(None, "--time-max"),
    max_results: int = typer.Option(50, "--max"),
    token: str = typer.Option("", "--token"),
) -> None:
    """List events in a time window."""
    _emit_json(_lib.list_events(
        _svc(token), calendar_id=calendar_id,
        time_min=time_min, time_max=time_max, max_results=max_results,
    ))


@app.command()
def get(
    event_id: str = typer.Argument(...),
    calendar_id: str = typer.Option("primary", "--calendar"),
    token: str = typer.Option("", "--token"),
) -> None:
    """Fetch a single event."""
    _emit_json(_lib.get_event(_svc(token), event_id, calendar_id=calendar_id))


@app.command()
def create(
    summary: str = typer.Option(..., "--summary"),
    start: str = typer.Option(..., "--start", help="RFC3339 datetime."),
    end: str = typer.Option(..., "--end", help="RFC3339 datetime."),
    calendar_id: str = typer.Option("primary", "--calendar"),
    description: Optional[str] = typer.Option(None, "--description"),
    attendees: Optional[str] = typer.Option(
        None, "--attendees",
        help="Comma-separated emails.",
    ),
    token: str = typer.Option("", "--token"),
) -> None:
    """Create a timed event."""
    att_list = [e.strip() for e in attendees.split(",") if e.strip()] if attendees else None
    _emit_json(_lib.create_event(
        _svc(token), summary, start, end,
        calendar_id=calendar_id, description=description, attendees=att_list,
    ))


@app.command()
def update(
    event_id: str = typer.Argument(...),
    fields_json: str = typer.Option(
        "", "--fields",
        help="JSON object of fields to patch. Pass '-' to read from stdin.",
    ),
    calendar_id: str = typer.Option("primary", "--calendar"),
    token: str = typer.Option("", "--token"),
) -> None:
    """Patch an existing event."""
    if fields_json == "-":
        fields_json = sys.stdin.read()
    try:
        fields = json.loads(fields_json)
    except json.JSONDecodeError as exc:
        typer.echo(f"Error: --fields is not valid JSON: {exc}", err=True)
        raise typer.Exit(2) from exc
    _emit_json(_lib.update_event(
        _svc(token), event_id, fields, calendar_id=calendar_id,
    ))


@app.command()
def delete(
    event_id: str = typer.Argument(...),
    calendar_id: str = typer.Option("primary", "--calendar"),
    token: str = typer.Option("", "--token"),
) -> None:
    """Delete an event."""
    _emit_json({"deleted": _lib.delete_event(_svc(token), event_id, calendar_id=calendar_id)})


@app.command()
def sync(
    dwh: str = typer.Option(..., "--dwh"),
    config: str = typer.Option("", "--config"),
    max_runtime: int = typer.Option(1500, "--max-runtime"),
    token: str = typer.Option("", "--token"),
) -> None:
    """Sync calendars into the warehouse per --config."""
    _emit_json(_lib.sync_to_warehouse(
        _svc(token), dwh,
        config_file=config, max_runtime_seconds=max_runtime,
    ))


@app.command()
def auth(
    credentials: Optional[Path] = typer.Option(None, "--credentials"),
    token: str = typer.Option("", "--token"),
) -> None:
    """Run Google OAuth for the google-calendar skill."""
    from clawmeets.integrations.auth.google_oauth import (
        GoogleOAuthError, run_installed_flow,
    )
    token_path = resolve_skill_token_path("google-calendar", explicit=token)
    try:
        run_installed_flow(_lib.SCOPES, token_path, credentials)
    except GoogleOAuthError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Authenticated. Token at {token_path} (0600).")


@app.command()
def revoke(
    token: str = typer.Option("", "--token", help="Override token path."),
) -> None:
    """Revoke the google-calendar skill's Google grant and delete its token.

    Headless counterpart to the web Disconnect: POSTs the token to Google's
    revoke endpoint, then removes token.json. Idempotent.
    """
    from clawmeets.integrations.auth.google_oauth import revoke_token

    token_path = resolve_skill_token_path("google-calendar", explicit=token)
    if revoke_token(token_path):
        typer.echo(f"Disconnected. Revoked Google grant and removed {token_path}.")
    else:
        typer.echo(f"Nothing to disconnect — no token at {token_path}.")
