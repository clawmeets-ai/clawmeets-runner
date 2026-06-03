# SPDX-License-Identifier: MIT
"""
clawmeets/mcp/servers/gcal_server.py

Google Calendar MCP server. Exposes list/get/create/update/delete events
and named-slice ``sync_to_warehouse`` as MCP tools, backed by
google-api-python-client. Runs as a stdio subprocess of Claude Code.

Reads the OAuth token from the path in CLAWMEETS_GCAL_TOKEN_FILE.

Named-slice sync model: each entry in ``calendars_to_sync`` (in the per-MCP
config at ``{agent_dir}/mcp-hub/configs/google-calendar.json``) binds one
Google Calendar (by ``calendarId``) to its own warehouse dataset under
``{dwh_dir}/sources/google-calendar/<name>/`` and
``{dwh_dir}/merged/google-calendar/<name>.json``.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from clawmeets.mcp.servers._sync_warehouse import (
    SyncBudget,
    _read_state,
    atomic_write_json,
    gc_old_timestamps,
    merge_json_envelopes,
    new_timestamp_dir,
    utcnow_iso,
    validate_merge_policy,
    write_howto,
)
from clawmeets.utils.jsonc import parse_jsonc
from clawmeets.utils.validation import validate_name

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


def _load_config(config_file: str) -> tuple[Optional[dict], Optional[str]]:
    """Read this MCP's config from the file path supplied by the agent.

    Returns ``(cfg, err)`` with the same noop-on-missing semantics as the
    other configurable sync MCPs.
    """
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
    """Sanitize a Google Calendar event id into a filesystem-safe segment.
    Google ids are usually URL-safe but defensive sanitization is cheap."""
    return _SAFE_EVENT_ID_RE.sub("_", event_id)


def _build_envelope(event: dict, slice_name: str, calendar_id: str) -> tuple[Optional[dict], Optional[str]]:
    """Convert a Google Calendar event dict into the warehouse envelope.

    Returns ``(envelope, updated_iso)`` or ``(None, None)`` if the event
    lacks ``updated`` (skip).
    """
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
    return envelope, updated


def _sync_one_slice(
    *,
    cal_svc,
    slice_cfg: dict,
    dwh_dir: str,
    budget: SyncBudget,
    window_end: str,
) -> dict:
    """Sync a single named google-calendar slice; return its per-slice summary.

    Mirrors ``caldav_server._sync_one_slice``: owns its own
    ``sync-state.json`` under ``{dwh_dir}/sources/google-calendar/<name>/``
    and advances its watermark independently of sibling slices. The shared
    ``cal_svc`` and ``budget`` are passed in so all slices in one call share
    one OAuth handshake and one wall-clock budget.
    """
    raw_name = slice_cfg.get("name") if isinstance(slice_cfg, dict) else None
    if not isinstance(raw_name, str) or not raw_name.strip():
        return {
            "name": "<unnamed>",
            "rows_written": 0, "watermarks": None,
            "has_more": False,
            "error": "slice config missing required 'name' field",
        }
    try:
        name = validate_name(raw_name)
    except ValueError as exc:
        return {
            "name": raw_name,
            "rows_written": 0, "watermarks": None,
            "has_more": False,
            "error": f"invalid slice name {raw_name!r}: {exc}",
        }

    calendar_id = slice_cfg.get("calendar_id")
    if not isinstance(calendar_id, str) or not calendar_id.strip():
        return {
            "name": name,
            "rows_written": 0, "watermarks": None,
            "has_more": False,
            "error": "slice config missing required 'calendar_id' field "
                     "(e.g. 'primary' or '<id>@group.calendar.google.com')",
        }

    merge_policy, upsert_id_column, merge_err = validate_merge_policy(slice_cfg)
    if merge_err:
        return {
            "name": name,
            "rows_written": 0, "watermarks": None,
            "has_more": False,
            "error": merge_err,
        }

    dwh_root = Path(dwh_dir).expanduser().resolve()
    source = f"google-calendar/{name}"
    source_dir = dwh_root / "sources" / source
    state_path = source_dir / "sync-state.json"
    merged_path = dwh_root / "merged" / "google-calendar" / f"{name}.json"

    howto_err = write_howto(
        slice_cfg.get("howto"),
        source_dir=source_dir,
        merged_path=merged_path,
    )
    if howto_err:
        return {
            "name": name,
            "rows_written": 0, "watermarks": None,
            "has_more": False,
            "error": howto_err,
        }

    state = _read_state(state_path, source)

    if merge_policy != "replace" and state.get("last_sync_at") is None:
        start_at_raw = slice_cfg.get("start_at")
        if isinstance(start_at_raw, str) and start_at_raw.strip():
            try:
                dt = datetime.fromisoformat(start_at_raw.strip().replace("Z", "+00:00"))
            except ValueError as exc:
                return {
                    "name": name,
                    "rows_written": 0, "watermarks": None,
                    "has_more": False,
                    "error": (
                        f"invalid start_at {start_at_raw!r} for slice {name!r}: "
                        f"{exc} (must be ISO-8601, e.g. '2024-01-01' or "
                        f"'2024-01-01T00:00:00Z')"
                    ),
                }
            # Google's events.list `updatedMin` requires a full RFC3339 datetime
            # with timezone — a bare date like "2024-01-01" parses here but
            # returns HTTP 400. Normalize to midnight UTC before persisting.
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            start_at = dt.isoformat()
            state["low_watermark"] = start_at
            state["high_watermark"] = start_at

    low = state.get("low_watermark") or utcnow_iso()
    high = state.get("high_watermark") or utcnow_iso()

    if merge_policy == "replace":
        # Replace mode: list every event in the calendar each run; merge step
        # rewrites the consolidated JSON. Drop updatedMin so the API returns
        # everything regardless of update time.
        updated_min: Optional[str] = None
        window_start = "1970-01-01T00:00:00+00:00"
    else:
        window_start = max(low, high)
        if window_start >= window_end:
            return {
                "name": name,
                "rows_written": 0,
                "watermarks": {"low": low, "high": high},
                "has_more": False, "error": None,
            }
        updated_min = window_start

    timestamp_dir = new_timestamp_dir(source_dir)
    rows_written_start = budget.rows_written
    latest_seen = window_start
    has_more = False

    try:
        page_token: Optional[str] = None
        while True:
            if budget.should_stop():
                has_more = True
                break
            list_kwargs = {
                "calendarId": calendar_id,
                "singleEvents": True,
                "showDeleted": False,
                "orderBy": "updated",
                "maxResults": 250,
                "pageToken": page_token,
            }
            if updated_min is not None:
                list_kwargs["updatedMin"] = updated_min
            resp = cal_svc.events().list(**list_kwargs).execute()
            items = resp.get("items", [])
            for event in items:
                if budget.should_stop():
                    has_more = True
                    break
                envelope, updated = _build_envelope(event, name, calendar_id)
                if envelope is None:
                    continue
                # Defensive: skip rows updated at/after window_end (upsert only).
                if merge_policy == "upsert" and updated >= window_end:
                    continue
                event_id = envelope["id"]
                if not event_id:
                    continue
                atomic_write_json(
                    timestamp_dir / f"{_safe_event_id(event_id)}.json",
                    envelope,
                )
                if updated > latest_seen:
                    latest_seen = updated
                budget.rows_written += 1
            if has_more:
                break
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        _rmdir_if_empty(timestamp_dir)
        state["last_sync_at"] = utcnow_iso()
        state["last_sync_count"] = budget.rows_written - rows_written_start
        state["last_error"] = err
        atomic_write_json(state_path, state)
        return {
            "name": name,
            "rows_written": budget.rows_written - rows_written_start,
            "watermarks": {"low": low, "high": high},
            "has_more": False, "error": err,
        }

    _rmdir_if_empty(timestamp_dir)

    if timestamp_dir.exists():
        merge_err_msg = merge_json_envelopes(
            timestamp_dir, merged_path,
            policy=merge_policy, id_column=upsert_id_column,
        )
        if merge_err_msg:
            state["last_sync_at"] = utcnow_iso()
            state["last_sync_count"] = budget.rows_written - rows_written_start
            state["last_error"] = merge_err_msg
            atomic_write_json(state_path, state)
            return {
                "name": name,
                "rows_written": budget.rows_written - rows_written_start,
                "watermarks": {"low": low, "high": high},
                "has_more": False, "error": merge_err_msg,
            }
        gc_old_timestamps(source_dir)

    if merge_policy == "replace":
        state["last_sync_at"] = utcnow_iso()
        state["last_sync_count"] = budget.rows_written - rows_written_start
        state["last_error"] = None
        atomic_write_json(state_path, state)
        return {
            "name": name,
            "rows_written": budget.rows_written - rows_written_start,
            "watermarks": {"low": low, "high": high},
            "has_more": has_more, "error": None,
        }

    new_high = window_end if not has_more else max(latest_seen, high)
    state["high_watermark"] = new_high
    state["last_sync_at"] = utcnow_iso()
    state["last_sync_count"] = budget.rows_written - rows_written_start
    state["last_error"] = None
    atomic_write_json(state_path, state)

    return {
        "name": name,
        "rows_written": budget.rows_written - rows_written_start,
        "watermarks": {"low": low, "high": new_high},
        "has_more": has_more, "error": None,
    }


def _rmdir_if_empty(path: Path) -> None:
    """Remove ``path`` if it's an empty directory; ignore otherwise."""
    try:
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    except OSError:
        pass


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

    @mcp.tool()
    def sync_to_warehouse(
        dwh_dir: str,
        config_file: str = "",
        max_runtime_seconds: int = 1500,
    ) -> dict:
        """Sync new / updated Calendar events into the personal data warehouse.

        Call this exactly once when you receive a DM whose body starts with
        ``<!-- clawmeets:google-calendar-sync-trigger -->``. Read ``dwh_dir``
        from your prompt's ``== DATA WAREHOUSE ==`` block and ``config_file``
        from the ``== MCP CONFIG FILES ==`` block (the path next to
        ``google-calendar``).

        Note the trigger marker is ``google-calendar-sync-trigger`` (not
        ``calendar-sync-trigger``) so installing this MCP side-by-side with
        the CalDAV ``calendar`` MCP doesn't double-fire on the same DM.

        Named-slice model: the config carries a ``calendars_to_sync`` list,
        each entry a ``{name, calendar_id, merge_policy?,
        merge_policy_upsert_id_column?, start_at?, howto?}`` dict. Each
        slice gets its own output directory and watermark at
        ``{dwh_dir}/sources/google-calendar/<name>/``; per-run envelopes
        land in ``<TIMESTAMP>/<event_id_safe>.json`` siblings of
        ``sync-state.json``, and the consolidated dataset rebuilds at
        ``{dwh_dir}/merged/google-calendar/<name>.json`` (JSON array sorted
        by ``ts``) per the slice's ``merge_policy`` (default ``upsert``
        keyed on per-occurrence event ``id``). Slices advance independently.

        Watermark semantics: in ``upsert`` mode, the per-slice filter is
        ``updatedMin=<window_start>`` — i.e. events whose ``updated`` field
        falls in ``(window_start, window_end)``. This catches both
        newly-created events AND mutations of existing events (a meeting
        moved to a different time, attendees changing RSVP, etc.) —
        including events scheduled for past dates that were edited
        recently. Recurring events are expanded (``singleEvents=true``)
        so each occurrence has a distinct ``id`` and a distinct envelope;
        switch ``merge_policy_upsert_id_column`` to ``"ical_uid"`` to
        dedupe occurrences back to the master event instead.

        Empty/missing config or empty ``calendars_to_sync`` list ⇒
        ``status: "noop"`` (no API calls, no directories created).

        Each row is the envelope built by ``_build_envelope`` plus
        ``ts`` (= ``updated``), ``slice`` (= the slice's slug), and the
        full source event under ``raw``.

        Args:
            dwh_dir: Personal data warehouse root.
            config_file: Path to this agent's per-MCP config file.
                Empty / missing file ⇒ noop.
            max_runtime_seconds: Wall-clock budget shared across all slices.
                Default 1500 (25 min). When elapsed mid-run, returns
                ``has_more=true`` so the next scheduled trigger resumes.

        Returns the standard sync envelope plus a ``per_slice`` map:
        ``{status, source, rows_written, window, watermarks, has_more,
        error, per_slice}``.
        """
        window_end = utcnow_iso()
        cfg, err = _load_config(config_file)
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

        # One Calendar service shared across slices — same auth handshake.
        cal_svc = _service()

        budget = SyncBudget(max_runtime_seconds)
        per_slice: dict[str, dict] = {}
        any_error = False
        any_has_more = False
        agg_low: Optional[str] = None
        agg_high: Optional[str] = None
        first_error: Optional[str] = None

        for slice_cfg in slices:
            s_name = slice_cfg.get("name") if isinstance(slice_cfg, dict) else None
            display_name = s_name if isinstance(s_name, str) and s_name else "<unnamed>"
            if budget.should_stop():
                any_has_more = True
                per_slice[display_name] = {
                    "name": display_name, "rows_written": 0,
                    "watermarks": None, "has_more": True, "error": None,
                }
                continue
            summary = _sync_one_slice(
                cal_svc=cal_svc,
                slice_cfg=slice_cfg if isinstance(slice_cfg, dict) else {},
                dwh_dir=dwh_dir,
                budget=budget,
                window_end=window_end,
            )
            per_slice[summary["name"]] = summary
            if summary.get("error"):
                any_error = True
                if first_error is None:
                    first_error = summary["error"]
            if summary.get("has_more"):
                any_has_more = True
            wms = summary.get("watermarks") or {}
            if wms.get("low"):
                agg_low = wms["low"] if agg_low is None else min(agg_low, wms["low"])
            if wms.get("high"):
                agg_high = wms["high"] if agg_high is None else max(agg_high, wms["high"])

        if any_error and budget.rows_written == 0:
            status = "error"
        elif any_has_more:
            status = "partial"
        elif budget.rows_written == 0:
            status = "noop"
        else:
            status = "ok"

        return {
            "status": status,
            "source": "google-calendar",
            "rows_written": budget.rows_written,
            "window": [agg_low or window_end, window_end],
            "watermarks": (
                {"low": agg_low, "high": agg_high}
                if (agg_low or agg_high) else None
            ),
            "has_more": any_has_more,
            "error": first_error,
            "per_slice": per_slice,
        }

    mcp.run()


if __name__ == "__main__":
    main()
