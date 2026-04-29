# SPDX-License-Identifier: MIT
"""
clawmeets/mcp/servers/gcal_server.py

Google Calendar MCP server. Exposes list/get/create/update/delete events as
MCP tools, backed by google-api-python-client. Runs as a stdio subprocess of
Claude Code.

Reads the OAuth token from the path in CLAWMEETS_GCAL_TOKEN_FILE.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

SCOPES = ["https://www.googleapis.com/auth/calendar"]


def _token_path() -> Path:
    p = os.environ.get("CLAWMEETS_GCAL_TOKEN_FILE")
    if not p:
        raise RuntimeError(
            "CLAWMEETS_GCAL_TOKEN_FILE is not set. The Google Calendar MCP "
            "server is expected to be launched by the clawmeets runner, which "
            "sets this via the mcps/google-calendar/mcp.json launch spec."
        )
    return Path(p)


def _service():
    from googleapiclient.discovery import build
    from clawmeets.mcp.auth.google_oauth import load_credentials

    creds = load_credentials(_token_path(), SCOPES)
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The `mcp` package is required but missing — the clawmeets runner "
            "should bundle it by default. Try: pip install --upgrade clawmeets"
        ) from exc

    mcp = FastMCP("clawmeets-gcal")

    @mcp.tool()
    def list_events(
        calendar_id: str = "primary",
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        max_results: int = 50,
    ) -> list[dict]:
        """List events in a time window. Times are RFC3339 strings (e.g. 2026-04-20T00:00:00Z)."""
        svc = _service()
        resp = svc.events().list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        return resp.get("items", [])

    @mcp.tool()
    def get_event(event_id: str, calendar_id: str = "primary") -> dict:
        """Fetch a single event by id."""
        svc = _service()
        return svc.events().get(calendarId=calendar_id, eventId=event_id).execute()

    @mcp.tool()
    def create_event(
        summary: str,
        start: str,
        end: str,
        calendar_id: str = "primary",
        description: Optional[str] = None,
        attendees: Optional[list[str]] = None,
    ) -> dict:
        """Create a timed event. `start` and `end` are RFC3339 strings."""
        svc = _service()
        body: dict = {
            "summary": summary,
            "start": {"dateTime": start},
            "end": {"dateTime": end},
        }
        if description:
            body["description"] = description
        if attendees:
            body["attendees"] = [{"email": e} for e in attendees]
        return svc.events().insert(calendarId=calendar_id, body=body).execute()

    @mcp.tool()
    def update_event(
        event_id: str,
        fields: dict,
        calendar_id: str = "primary",
    ) -> dict:
        """Patch an existing event with the given fields (partial update)."""
        svc = _service()
        return svc.events().patch(
            calendarId=calendar_id, eventId=event_id, body=fields,
        ).execute()

    @mcp.tool()
    def delete_event(event_id: str, calendar_id: str = "primary") -> str:
        """Delete an event. Returns the deleted event id on success."""
        svc = _service()
        svc.events().delete(calendarId=calendar_id, eventId=event_id).execute()
        return event_id

    mcp.run()


if __name__ == "__main__":
    main()
