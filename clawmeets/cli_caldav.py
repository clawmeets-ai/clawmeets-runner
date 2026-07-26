# SPDX-License-Identifier: MIT
"""
clawmeets/cli_caldav.py — CalDAV calendar CLI.

Subcommands: list, list-events, get, create, update, delete, sync.
"""
from __future__ import annotations

import json
from typing import Optional

import typer

from clawmeets.integrations.caldav import _lib

app = typer.Typer(
    name="caldav",
    help="CalDAV calendar (iCloud / Fastmail / Nextcloud / etc.). Paired skill: calendar.",
    no_args_is_help=True,
)


def _emit_json(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command("list")
def list_cmd(config: str = typer.Option("", "--config")) -> None:
    """List calendars on the configured CalDAV account."""
    try:
        _emit_json(_lib.list_calendars(config))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("list-events")
def list_events_cmd(
    time_min: str = typer.Option(..., "--time-min", help="ISO-8601 UTC."),
    time_max: str = typer.Option(..., "--time-max", help="ISO-8601 UTC."),
    calendar_url: Optional[str] = typer.Option(None, "--calendar-url"),
    config: str = typer.Option("", "--config"),
) -> None:
    """List events overlapping a time window."""
    try:
        _emit_json(_lib.list_events(config, time_min, time_max, calendar_url=calendar_url))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def get(
    uid: str = typer.Argument(...),
    calendar_url: Optional[str] = typer.Option(None, "--calendar-url"),
    config: str = typer.Option("", "--config"),
) -> None:
    """Fetch one event by iCalendar UID."""
    try:
        _emit_json(_lib.get_event(config, uid, calendar_url=calendar_url))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def create(
    summary: str = typer.Option(..., "--summary"),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    description: Optional[str] = typer.Option(None, "--description"),
    location: Optional[str] = typer.Option(None, "--location"),
    attendees: Optional[str] = typer.Option(None, "--attendees",
                                             help="Comma-separated emails."),
    calendar_url: Optional[str] = typer.Option(None, "--calendar-url"),
    config: str = typer.Option("", "--config"),
) -> None:
    """Create a timed event."""
    att_list = [a.strip() for a in attendees.split(",") if a.strip()] if attendees else None
    try:
        _emit_json(_lib.create_event(
            config, summary, start, end,
            calendar_url=calendar_url, description=description,
            location=location, attendees=att_list,
        ))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def update(
    uid: str = typer.Argument(...),
    summary: Optional[str] = typer.Option(None, "--summary"),
    start: Optional[str] = typer.Option(None, "--start"),
    end: Optional[str] = typer.Option(None, "--end"),
    description: Optional[str] = typer.Option(None, "--description"),
    location: Optional[str] = typer.Option(None, "--location"),
    calendar_url: Optional[str] = typer.Option(None, "--calendar-url"),
    config: str = typer.Option("", "--config"),
) -> None:
    """Patch fields on an existing event."""
    try:
        _emit_json(_lib.update_event(
            config, uid, calendar_url=calendar_url,
            summary=summary, start=start, end=end,
            description=description, location=location,
        ))
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def delete(
    uid: str = typer.Argument(...),
    calendar_url: Optional[str] = typer.Option(None, "--calendar-url"),
    config: str = typer.Option("", "--config"),
) -> None:
    """Delete an event."""
    try:
        _emit_json({"deleted": _lib.delete_event(config, uid, calendar_url=calendar_url)})
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def sync(
    dwh: str = typer.Option(..., "--dwh"),
    config: str = typer.Option("", "--config"),
    max_runtime: int = typer.Option(1500, "--max-runtime"),
) -> None:
    """Sync CalDAV calendars into the warehouse."""
    _emit_json(_lib.sync_to_warehouse(
        dwh, config_file=config, max_runtime_seconds=max_runtime,
    ))
