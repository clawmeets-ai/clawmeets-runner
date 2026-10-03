# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/gcal/_lib.py

Pure-Python Google Calendar integration. Drives ``clawmeets gcal <subcmd>``;
paired skill ``skills/google-calendar/SKILL.md``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

SCOPES = ["https://www.googleapis.com/auth/calendar"]


def build_service(token_path: Path):
    from googleapiclient.discovery import build
    from clawmeets.integrations.auth.google_oauth import load_credentials

    creds = load_credentials(token_path, SCOPES)
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------------------
# Tool bodies
# ---------------------------------------------------------------------------


def list_calendars(svc) -> list[dict]:
    """List the user's available calendars."""
    resp = svc.calendarList().list().execute()
    return resp.get("items", [])


def list_events(
    svc,
    calendar_id: str = "primary",
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    max_results: int = 50,
) -> list[dict]:
    """List events in a time window. Times are RFC3339 strings."""
    resp = svc.events().list(
        calendarId=calendar_id,
        timeMin=time_min, timeMax=time_max,
        maxResults=max_results,
        singleEvents=True, orderBy="startTime",
    ).execute()
    return resp.get("items", [])


def get_event(svc, event_id: str, calendar_id: str = "primary") -> dict:
    """Fetch a single event by id."""
    return svc.events().get(calendarId=calendar_id, eventId=event_id).execute()


def create_event(
    svc,
    summary: str, start: str, end: str,
    calendar_id: str = "primary",
    description: Optional[str] = None,
    attendees: Optional[list[str]] = None,
) -> dict:
    """Create a timed event. `start` / `end` are RFC3339 datetimes."""
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


def update_event(svc, event_id: str, fields: dict, calendar_id: str = "primary") -> dict:
    """Patch an existing event with the given fields."""
    return svc.events().patch(
        calendarId=calendar_id, eventId=event_id, body=fields,
    ).execute()


def delete_event(svc, event_id: str, calendar_id: str = "primary") -> str:
    """Delete an event; return the deleted id."""
    svc.events().delete(calendarId=calendar_id, eventId=event_id).execute()
    return event_id
