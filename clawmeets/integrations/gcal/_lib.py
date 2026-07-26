# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/gcal/_lib.py

Pure-Python Google Calendar integration. Drives ``clawmeets gcal <subcmd>``;
paired skill ``skills/google-calendar/SKILL.md``.

Named-slice sync model: each ``calendars_to_sync`` entry binds one calendar
(``calendarId``) to its own warehouse dataset under
``{dwh_dir}/sources/google-calendar/<name>/``.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from clawmeets.integrations._config_resolve import resolve_skill_config_path
from clawmeets.integrations._sync_warehouse import (
    SyncBudget,
    run_slice_sync,
    run_slices,
    utcnow_iso,
    write_howto,
)
from clawmeets.utils.file_io import FileUtil
from clawmeets.utils.jsonc import parse_jsonc

SCOPES = ["https://www.googleapis.com/auth/calendar"]


def build_service(token_path: Path):
    from googleapiclient.discovery import build
    from clawmeets.integrations.auth.google_oauth import load_credentials

    creds = load_credentials(token_path, SCOPES)
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def load_config(config_file: str) -> tuple[Optional[dict], Optional[str]]:
    config_file = resolve_skill_config_path("google-calendar", config_file)
    if not config_file:
        return None, None
    path = Path(config_file).expanduser()
    if not path.exists():
        return None, None
    try:
        raw = path.read_text()
    except OSError as exc:
        return None, f"could not read config file {path}: {exc}"
    if not raw.strip():
        return None, None
    try:
        cfg = parse_jsonc(raw)
    except Exception as exc:
        return None, f"config file is not valid JSON: {exc}"
    if not isinstance(cfg, dict):
        return None, "config file must contain a JSON object"
    return cfg, None


_SAFE_EVENT_ID_RE = re.compile(r"[^A-Za-z0-9_-]")


def _safe_event_id(event_id: str) -> str:
    return _SAFE_EVENT_ID_RE.sub("_", event_id)


def _build_envelope(event: dict, slice_name: str, calendar_id: str) -> tuple[Optional[dict], Optional[str]]:
    updated = event.get("updated", "")
    if not updated:
        return None, None
    envelope = {
        "ts": updated,
        "id": event.get("id"),
        "ical_uid": event.get("iCalUID"),
        "calendar_id": calendar_id,
        "updated": updated,
        "summary": event.get("summary", ""),
        "start": event.get("start"),
        "end": event.get("end"),
        "attendees": event.get("attendees", []),
        "location": event.get("location"),
        "status": event.get("status"),
        "raw": event,
        "slice": slice_name,
    }
    # Native tombstone: a cancelled event (surfaced by showDeleted) is a delete.
    if event.get("status") == "cancelled":
        envelope["deleted"] = True
    return envelope, updated


def _sync_one_slice(
    *,
    cal_svc,
    slice_cfg: dict,
    dwh_dir: str,
    budget: SyncBudget,
) -> dict:
    raw_name = slice_cfg.get("name") if isinstance(slice_cfg, dict) else None
    if not isinstance(raw_name, str) or not raw_name.strip():
        return {
            "name": "<unnamed>", "rows_written": 0, "watermarks": None,
            "has_more": False, "error": "slice config missing required 'name' field",
        }
    try:
        name = FileUtil.validate_fs_name(raw_name)
    except ValueError as exc:
        return {
            "name": raw_name, "rows_written": 0, "watermarks": None,
            "has_more": False, "error": f"invalid slice name {raw_name!r}: {exc}",
        }

    calendar_id = slice_cfg.get("calendar_id")
    if not isinstance(calendar_id, str) or not calendar_id.strip():
        return {
            "name": name, "rows_written": 0, "watermarks": None,
            "has_more": False,
            "error": "slice config missing required 'calendar_id' field "
                     "(e.g. 'primary' or '<id>@group.calendar.google.com')",
        }

    dwh_root = Path(dwh_dir).expanduser().resolve()
    source = f"google-calendar/{name}"
    base = dwh_root / "raw" / source

    howto_err = write_howto(slice_cfg.get("howto"), snapshot_dir=base)
    if howto_err:
        return {"name": name, "rows_written": 0, "watermarks": None,
                "has_more": False, "error": howto_err}

    def fetch(window_start: str, _window_end: str, bud: SyncBudget, emit) -> bool:
        # gcal events.list bounds only the lower end (updatedMin); the driver's
        # upsert dedups any overlap a backfill re-pull produces. showDeleted=True
        # surfaces cancellations, which _build_envelope marks as tombstones.
        page_token: Optional[str] = None
        while True:
            if bud.should_stop():
                return True
            resp = cal_svc.events().list(
                calendarId=calendar_id, singleEvents=True, showDeleted=True,
                orderBy="updated", maxResults=250, pageToken=page_token,
                updatedMin=window_start,
            ).execute()
            for event in resp.get("items", []):
                if bud.should_stop():
                    return True
                envelope, _updated = _build_envelope(event, name, calendar_id)
                if envelope is None or not envelope.get("id"):
                    continue
                bud.rows_written += 1
                emit(envelope)
            page_token = resp.get("nextPageToken")
            if not page_token:
                return False

    return run_slice_sync(
        source=source, dwh_dir=dwh_dir, budget=budget, fetch=fetch,
        id_field="id", ts_field="ts", start_at=slice_cfg.get("start_at"),
        snapshot_fmt="ndjson",
    )


# ---------------------------------------------------------------------------
# Interactive tool bodies
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


def sync_to_warehouse(
    svc,
    dwh_dir: str,
    config_file: str = "",
    max_runtime_seconds: int = 1500,
) -> dict:
    """Sync new / updated Calendar events into the data warehouse.

    Triggered by ``<!-- clawmeets:google-calendar-sync-trigger -->``."""
    window_end = utcnow_iso()
    cfg, err = load_config(config_file)
    if err:
        return {
            "status": "error", "source": "google-calendar", "rows_written": 0,
            "window": [window_end, window_end], "watermarks": None,
            "has_more": False, "error": err, "per_slice": {},
        }
    if cfg is None:
        return {
            "status": "noop", "source": "google-calendar", "rows_written": 0,
            "window": [window_end, window_end], "watermarks": None,
            "has_more": False, "error": None, "per_slice": {},
        }

    slices = cfg.get("calendars_to_sync")
    if not isinstance(slices, list) or len(slices) == 0:
        return {
            "status": "noop", "source": "google-calendar", "rows_written": 0,
            "window": [window_end, window_end], "watermarks": None,
            "has_more": False, "error": None, "per_slice": {},
        }

    budget = SyncBudget(max_runtime_seconds)
    return run_slices(
        source_family="google-calendar", slices=slices, budget=budget,
        dwh_dir=dwh_dir,
        run_one=lambda sc: _sync_one_slice(
            cal_svc=svc, slice_cfg=sc, dwh_dir=dwh_dir, budget=budget),
    )
