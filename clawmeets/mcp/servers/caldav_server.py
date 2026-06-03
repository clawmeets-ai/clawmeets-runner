# SPDX-License-Identifier: MIT
"""
clawmeets/mcp/servers/caldav_server.py

CalDAV MCP server. Provider-agnostic calendar access — works with any
server that speaks CalDAV (iCloud, Fastmail, Nextcloud, Radicale,
SOGo, mailcow, etc.).

Reads its config from the per-MCP file at
``{agent_dir}/mcp-hub/configs/calendar.json``, supplied to each tool call
as the ``config_file`` argument (the agent reads the path from its prompt's
``== MCP CONFIG FILES ==`` block). ``${VAR}`` placeholders inside the file
resolve from ``os.environ`` — same pattern as the http-api, database, and
mailbox MCPs. No OAuth, no token files; users export credentials as env
vars on the runner before ``clawmeets start``.

Config schema (the file at the path the agent passes):

    {
      "caldav": {
        "url": "https://caldav.fastmail.com/dav/calendars/user/me@example.com/",
        "username": "${CALENDAR_USERNAME}",
        "password": "${CALENDAR_PASSWORD}"
      },
      "sync_lookback_days": 90,
      "sync_lookahead_days": 365,
      "calendars_to_sync": [
        {
          "name": "personal",
          "calendar": "Personal",
          "merge_policy": "upsert",
          "merge_policy_upsert_id_column": "uid",
          "howto": "..."
        }
      ]
    }

``calendars_to_sync`` is a list of named slices; each slice maps one CalDAV
calendar (resolved by display name, case-insensitive) onto an independent
warehouse dataset at ``{dwh_dir}/sources/calendar/<name>/`` and
``{dwh_dir}/merged/calendar/<name>.json``. Empty list / missing field ⇒
sync_to_warehouse is a no-op.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from clawmeets.mcp.servers._sync_warehouse import (
    SyncBudget,
    _read_state,
    atomic_write_json,
    expand_env,
    gc_old_timestamps,
    merge_json_envelopes,
    new_timestamp_dir,
    utcnow_iso,
    validate_merge_policy,
    write_howto,
)
from clawmeets.utils.jsonc import parse_jsonc
from clawmeets.utils.validation import validate_name


def _load_config(config_file: str) -> tuple[Optional[dict], Optional[str]]:
    """Read this MCP's config from the file path supplied by the agent.

    Returns ``(cfg, err)``:
      - ``(dict, None)`` on a successfully-parsed dict-shaped config
      - ``(None, None)`` when the file path is empty, missing, or empty
        (caller treats as a clean noop — fresh installs don't error)
      - ``(None, "...")`` when the file is malformed JSONC, its root isn't
        a dict, or the required ``caldav`` block is missing/wrong-typed
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
    if not isinstance(cfg.get("caldav"), dict):
        return None, "config file must contain a `caldav` object with url/username/password"
    return cfg, None


def _require_config(config_file: str) -> dict:
    """Interactive tools want a hard error on missing/bad config, not a noop.

    Wraps ``_load_config`` and raises ``RuntimeError`` with an actionable
    message for either failure mode (empty path, missing file, malformed,
    missing ``caldav`` block).
    """
    cfg, err = _load_config(config_file)
    if cfg is not None:
        return cfg
    if err is None:
        raise RuntimeError(
            "calendar config_file is required — pass the path from your "
            "`== MCP CONFIG FILES ==` prompt block (next to `calendar`); "
            "save the config via the Configure modal in Agent Settings "
            "(see mcps/calendar/README.md)"
        )
    raise RuntimeError(err)


def _resolve(cfg: dict, scope: dict[str, str] | None = None) -> tuple[dict, list[str]]:
    missing: list[str] = []
    expanded = expand_env(cfg, scope or {}, missing)
    return expanded, missing


def _client(caldav_cfg: dict):
    try:
        import caldav  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "the `caldav` package is missing — install it on the runner: "
            "pip install caldav vobject"
        ) from exc
    url = caldav_cfg.get("url")
    if not url:
        raise RuntimeError("caldav.url is required in config.json")
    user = caldav_cfg.get("username") or ""
    password = caldav_cfg.get("password") or ""
    if not user or not password:
        raise RuntimeError(
            "caldav.username and caldav.password are required (resolve env vars before running)"
        )
    return caldav.DAVClient(url=url, username=user, password=password)


def _select_calendars(client, names: list[str] | None) -> list:
    """Return the list of calendars to sync. None/empty ⇒ all calendars."""
    principal = client.principal()
    calendars = list(principal.calendars())
    if not names:
        return calendars
    wanted = {n.strip().lower() for n in names if n}
    out: list = []
    for cal in calendars:
        try:
            disp = (cal.get_display_name() or "").strip().lower()
        except Exception:
            disp = ""
        if disp in wanted:
            out.append(cal)
    return out


def _select_one_calendar(client, name: str):
    """Return the single CalDAV calendar matching ``name`` (case-insensitive),
    or ``None`` if no calendar with that display name exists on the account.
    Used by ``_sync_one_slice``; ambiguous names resolve to the first match."""
    cals = _select_calendars(client, [name])
    return cals[0] if cals else None


def _slice_calendar_names(slices: Any) -> Optional[list[str]]:
    """Extract the ``calendar`` field from a ``calendars_to_sync`` slice list.

    Used by the interactive tools (``list_events`` / ``get_event`` / etc.)
    to scope queries to the same calendars the sync targets. Returns
    ``None`` when no slices are configured (caller should treat as "all
    calendars"). Skips entries that aren't dicts or lack a ``calendar``.
    """
    if not isinstance(slices, list) or not slices:
        return None
    out = [
        s["calendar"] for s in slices
        if isinstance(s, dict) and isinstance(s.get("calendar"), str) and s["calendar"]
    ]
    return out or None


def _ical_to_event(ical_obj, calendar_url: str, etag: str | None) -> Optional[dict]:
    """Turn one icalendar.Event into the provider-agnostic envelope."""
    try:
        import vobject  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "the `vobject` package is missing — install it on the runner: "
            "pip install caldav vobject"
        ) from exc

    raw = ical_obj.data if hasattr(ical_obj, "data") else str(ical_obj)
    try:
        cal = vobject.readOne(raw)
    except Exception:
        return None
    vevent = getattr(cal, "vevent", None)
    if vevent is None:
        return None

    def _get(name: str, default=None):
        v = getattr(vevent, name, None)
        return v.value if v is not None else default

    def _iso(dt) -> Optional[str]:
        if dt is None:
            return None
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        return str(dt)  # date-only

    def _addr_list(name: str) -> list[dict]:
        out: list[dict] = []
        for item in getattr(vevent, name + "_list", []) or []:
            value = (item.value or "").replace("mailto:", "").replace("MAILTO:", "")
            params = {k: v for k, v in (getattr(item, "params", {}) or {}).items()}
            out.append({
                "email": value,
                "rsvp": (params.get("PARTSTAT") or [None])[0] if params.get("PARTSTAT") else None,
                "role": (params.get("ROLE") or [None])[0] if params.get("ROLE") else None,
                "name": (params.get("CN") or [None])[0] if params.get("CN") else None,
            })
        return out

    organizer = None
    org = getattr(vevent, "organizer", None)
    if org is not None:
        organizer = {
            "email": (org.value or "").replace("mailto:", "").replace("MAILTO:", ""),
            "name": ((org.params or {}).get("CN") or [None])[0]
                if (org.params or {}).get("CN") else None,
        }

    return {
        "uid": _get("uid", ""),
        "etag": etag,
        "calendar_url": calendar_url,
        "summary": _get("summary", ""),
        "description": _get("description"),
        "location": _get("location"),
        "start": _iso(_get("dtstart")),
        "end":   _iso(_get("dtend")),
        "rrule": str(_get("rrule")) if _get("rrule") is not None else None,
        "attendees": _addr_list("attendee"),
        "organizer": organizer,
        "status": _get("status"),
        "created": _iso(_get("created")),
        "last_modified": _iso(_get("last_modified")) or _iso(_get("dtstamp")),
        "raw_ical": raw,
    }


def _safe_uid(uid: str) -> str:
    """Sanitize an iCal UID into a filesystem-safe filename segment."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in uid)


def _sync_one_slice(
    *,
    client,
    slice_cfg: dict,
    dwh_dir: str,
    budget: SyncBudget,
    window_end: str,
    lookback_days: int,
    lookahead_days: int,
) -> dict:
    """Sync a single named calendar slice; return its per-slice summary.

    Mirrors ``gdrive_server._sync_one_slice``: owns its own
    ``sync-state.json`` under ``{dwh_dir}/sources/calendar/<name>/`` and
    advances its watermark independently of sibling slices. The shared
    ``client`` and ``budget`` are passed in so all slices in one call share
    one CalDAV auth handshake and one wall-clock budget.
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

    calendar_name = slice_cfg.get("calendar")
    if not isinstance(calendar_name, str) or not calendar_name.strip():
        return {
            "name": name,
            "rows_written": 0, "watermarks": None,
            "has_more": False,
            "error": "slice config missing required 'calendar' field (CalDAV display name)",
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
    source = f"calendar/{name}"
    source_dir = dwh_root / "sources" / source
    state_path = source_dir / "sync-state.json"
    merged_path = dwh_root / "merged" / "calendar" / f"{name}.json"

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

    # Optional first-time watermark override; same semantics as gdrive.
    if merge_policy != "replace" and state.get("last_sync_at") is None:
        start_at_raw = slice_cfg.get("start_at")
        if isinstance(start_at_raw, str) and start_at_raw.strip():
            start_at = start_at_raw.strip()
            try:
                datetime.fromisoformat(start_at.replace("Z", "+00:00"))
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
            state["low_watermark"] = start_at
            state["high_watermark"] = start_at

    low = state.get("low_watermark") or utcnow_iso()
    high = state.get("high_watermark") or utcnow_iso()

    if merge_policy == "replace":
        # Replace mode lists every event in the lookback/lookahead window
        # each run; the merge step rewrites the consolidated JSON array.
        window_start: Optional[str] = None
    else:
        window_start = max(low, high)
        if window_start >= window_end:
            return {
                "name": name,
                "rows_written": 0,
                "watermarks": {"low": low, "high": high},
                "has_more": False, "error": None,
            }

    cal = _select_one_calendar(client, calendar_name)
    if cal is None:
        return {
            "name": name,
            "rows_written": 0,
            "watermarks": {"low": low, "high": high},
            "has_more": False,
            "error": f"calendar {calendar_name!r} not found on the configured CalDAV account",
        }

    timestamp_dir = new_timestamp_dir(source_dir)
    rows_written_start = budget.rows_written
    latest_seen = window_start or low
    has_more = False

    try:
        now = datetime.now(timezone.utc)
        range_start = now - timedelta(days=lookback_days)
        range_end = now + timedelta(days=lookahead_days)
        try:
            events = cal.search(start=range_start, end=range_end, expand=False)
        except Exception as exc:
            raise RuntimeError(f"calendar.search failed: {exc}") from exc

        for ev in events:
            if budget.should_stop():
                has_more = True
                break
            etag = getattr(ev, "etag", None)
            envelope = _ical_to_event(ev, str(cal.url), etag)
            if envelope is None:
                continue
            last_mod = envelope.get("last_modified")
            # Upsert mode: filter by watermark window. Replace mode: take
            # everything the search returned in the lookback/lookahead band.
            if merge_policy == "upsert":
                if not last_mod or last_mod <= window_start or last_mod >= window_end:
                    continue
            uid = envelope.get("uid") or ""
            if not uid:
                continue
            envelope["ts"] = last_mod or window_end
            envelope["slice"] = name
            atomic_write_json(timestamp_dir / f"{_safe_uid(uid)}.json", envelope)
            if last_mod and last_mod > latest_seen:
                latest_seen = last_mod
            budget.rows_written += 1
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

    mcp = FastMCP("clawmeets-calendar")

    @mcp.tool()
    def list_calendars(config_file: str) -> list[dict]:
        """Discover all calendars on the configured CalDAV account.

        Returns ``[{name, url}]``. Use the names as the ``calendar`` field
        on entries in ``calendars_to_sync`` to bind a sync slice to a
        specific calendar.
        """
        cfg = _require_config(config_file)
        resolved, missing = _resolve(cfg)
        if missing:
            raise RuntimeError(f"unset env vars: {sorted(set(missing))}")
        client = _client(resolved.get("caldav") or {})
        principal = client.principal()
        out: list[dict] = []
        for cal in principal.calendars():
            try:
                name = cal.get_display_name() or ""
            except Exception:
                name = ""
            out.append({"name": name, "url": str(cal.url)})
        return out

    @mcp.tool()
    def list_events(
        config_file: str,
        time_min: str,
        time_max: str,
        calendar_url: Optional[str] = None,
    ) -> list[dict]:
        """List events overlapping ``[time_min, time_max]``. ISO-8601 strings
        in UTC (``2026-05-09T00:00:00Z``). If ``calendar_url`` is omitted,
        searches every calendar in ``calendars_to_sync`` (or all calendars
        if unset)."""
        cfg = _require_config(config_file)
        resolved, missing = _resolve(cfg)
        if missing:
            raise RuntimeError(f"unset env vars: {sorted(set(missing))}")
        client = _client(resolved.get("caldav") or {})
        start = datetime.fromisoformat(time_min.replace("Z", "+00:00"))
        end = datetime.fromisoformat(time_max.replace("Z", "+00:00"))
        if calendar_url:
            cals = [client.calendar(url=calendar_url)]
        else:
            cals = _select_calendars(client, _slice_calendar_names(resolved.get("calendars_to_sync")))
        out: list[dict] = []
        for cal in cals:
            for ev in cal.search(start=start, end=end, expand=False):
                etag = getattr(ev, "etag", None)
                env = _ical_to_event(ev, str(cal.url), etag)
                if env is not None:
                    out.append(env)
        return out

    @mcp.tool()
    def get_event(
        config_file: str,
        uid: str,
        calendar_url: Optional[str] = None,
    ) -> dict:
        """Fetch one event by iCalendar UID. If ``calendar_url`` is omitted,
        searches every configured calendar."""
        cfg = _require_config(config_file)
        resolved, missing = _resolve(cfg)
        if missing:
            raise RuntimeError(f"unset env vars: {sorted(set(missing))}")
        client = _client(resolved.get("caldav") or {})
        if calendar_url:
            cals = [client.calendar(url=calendar_url)]
        else:
            cals = _select_calendars(client, _slice_calendar_names(resolved.get("calendars_to_sync")))
        for cal in cals:
            try:
                ev = cal.event_by_uid(uid)
            except Exception:
                continue
            etag = getattr(ev, "etag", None)
            env = _ical_to_event(ev, str(cal.url), etag)
            if env is not None:
                return env
        raise RuntimeError(f"event UID {uid!r} not found")

    @mcp.tool()
    def create_event(
        config_file: str,
        summary: str,
        start: str,
        end: str,
        calendar_url: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[list[str]] = None,
    ) -> dict:
        """Create a timed event on the chosen calendar (default: first
        calendar). ``start`` and ``end`` are ISO-8601 UTC."""
        try:
            from icalendar import Calendar, Event  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "the `icalendar` package is missing — install it on the runner: "
                "pip install icalendar"
            ) from exc
        cfg = _require_config(config_file)
        resolved, missing = _resolve(cfg)
        if missing:
            raise RuntimeError(f"unset env vars: {sorted(set(missing))}")
        client = _client(resolved.get("caldav") or {})
        if calendar_url:
            cal = client.calendar(url=calendar_url)
        else:
            cals = _select_calendars(client, _slice_calendar_names(resolved.get("calendars_to_sync")))
            if not cals:
                raise RuntimeError("no calendars available")
            cal = cals[0]

        ical = Calendar()
        ical.add("prodid", "-//clawmeets//caldav-mcp//EN")
        ical.add("version", "2.0")
        ev = Event()
        import uuid as _uuid
        ev.add("uid", str(_uuid.uuid4()))
        ev.add("summary", summary)
        ev.add("dtstart", datetime.fromisoformat(start.replace("Z", "+00:00")))
        ev.add("dtend",   datetime.fromisoformat(end.replace("Z", "+00:00")))
        ev.add("dtstamp", datetime.now(timezone.utc))
        if description:
            ev.add("description", description)
        if location:
            ev.add("location", location)
        for addr in attendees or []:
            ev.add("attendee", f"mailto:{addr}")
        ical.add_component(ev)
        saved = cal.save_event(ical.to_ical().decode("utf-8"))
        return {"uid": str(ev.get("uid")), "url": str(saved.url)}

    @mcp.tool()
    def update_event(
        config_file: str,
        uid: str,
        calendar_url: Optional[str] = None,
        summary: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
    ) -> dict:
        """Patch fields on an existing event. Omitted fields stay unchanged."""
        cfg = _require_config(config_file)
        resolved, missing = _resolve(cfg)
        if missing:
            raise RuntimeError(f"unset env vars: {sorted(set(missing))}")
        client = _client(resolved.get("caldav") or {})
        if calendar_url:
            cals = [client.calendar(url=calendar_url)]
        else:
            cals = _select_calendars(client, _slice_calendar_names(resolved.get("calendars_to_sync")))
        for cal in cals:
            try:
                ev = cal.event_by_uid(uid)
            except Exception:
                continue
            try:
                import vobject  # type: ignore
                vobj = vobject.readOne(ev.data)
                vevent = vobj.vevent
                if summary is not None:
                    vevent.summary.value = summary
                if description is not None:
                    if hasattr(vevent, "description"):
                        vevent.description.value = description
                    else:
                        vevent.add("description").value = description
                if location is not None:
                    if hasattr(vevent, "location"):
                        vevent.location.value = location
                    else:
                        vevent.add("location").value = location
                if start is not None:
                    vevent.dtstart.value = datetime.fromisoformat(start.replace("Z", "+00:00"))
                if end is not None:
                    vevent.dtend.value = datetime.fromisoformat(end.replace("Z", "+00:00"))
                ev.data = vobj.serialize()
                ev.save()
                return {"uid": uid, "url": str(ev.url)}
            except Exception as exc:
                raise RuntimeError(f"update failed: {exc}") from exc
        raise RuntimeError(f"event UID {uid!r} not found")

    @mcp.tool()
    def delete_event(
        config_file: str,
        uid: str,
        calendar_url: Optional[str] = None,
    ) -> str:
        """Delete an event by UID. Returns the deleted UID on success."""
        cfg = _require_config(config_file)
        resolved, missing = _resolve(cfg)
        if missing:
            raise RuntimeError(f"unset env vars: {sorted(set(missing))}")
        client = _client(resolved.get("caldav") or {})
        if calendar_url:
            cals = [client.calendar(url=calendar_url)]
        else:
            cals = _select_calendars(client, _slice_calendar_names(resolved.get("calendars_to_sync")))
        for cal in cals:
            try:
                ev = cal.event_by_uid(uid)
            except Exception:
                continue
            ev.delete()
            return uid
        raise RuntimeError(f"event UID {uid!r} not found")

    @mcp.tool()
    def sync_to_warehouse(
        dwh_dir: str,
        config_file: str,
        max_runtime_seconds: int = 1500,
    ) -> dict:
        """Sync new / updated events into the personal data warehouse.

        Call this exactly once when you receive a DM whose body starts with
        ``<!-- clawmeets:calendar-sync-trigger -->``. Read ``dwh_dir`` from
        your prompt's ``== DATA WAREHOUSE ==`` block and ``config_file``
        from your ``== MCP CONFIG FILES ==`` block (the path next to
        ``calendar``).

        Named-slice model: the config carries a ``calendars_to_sync`` list,
        each entry a ``{name, calendar, merge_policy?,
        merge_policy_upsert_id_column?, start_at?, howto?}`` dict. Each slice
        gets its own output directory and watermark at
        ``{dwh_dir}/sources/calendar/<name>/``; per-run envelopes land in
        ``<TIMESTAMP>/<event_uid_safe>.json`` siblings of ``sync-state.json``,
        and the consolidated dataset rebuilds at
        ``{dwh_dir}/merged/calendar/<name>.json`` (JSON array of envelopes
        sorted by ``ts``) per the slice's ``merge_policy`` (default
        ``upsert``; ``upsert`` requires ``merge_policy_upsert_id_column`` —
        typically ``"uid"``). Up to ``KEEP_RECENT_DUMPS`` timestamp folders
        are retained per slice. Slices advance independently — a failure on
        one does not roll back another's watermark.

        Watermark semantics: in ``upsert`` mode, the per-slice filter is
        ``last_modified`` (falling back to ``DTSTAMP``) ``in
        (window_start, window_end)``. Each cycle queries the slice's
        calendar in the time window
        ``[now - sync_lookback_days, now + sync_lookahead_days]`` and writes
        events whose ``last_modified > window_start``. Same ``uid``
        overwrites in the merged JSON — events mutate; warehouse stores
        current state, not history. In ``replace`` mode the watermark is
        ignored entirely.

        Empty/missing config or empty ``calendars_to_sync`` list ⇒
        ``status: "noop"`` (no directories created).

        Each row is the iCalendar-derived envelope from ``_ical_to_event``
        plus ``ts`` (= ``last_modified``) and ``slice`` (= the slice's
        slug).

        Args:
            dwh_dir: Personal data warehouse root.
            config_file: Path to this agent's per-MCP config file.
                Empty / missing file ⇒ noop.
            max_runtime_seconds: Wall-clock budget shared across all
                slices. Default 1500.

        Returns the standard sync envelope plus a ``per_slice`` map:
        ``{status, source, rows_written, window, watermarks, has_more,
        error, per_slice}``.
        """
        window_end = utcnow_iso()
        cfg, err = _load_config(config_file)
        if err:
            return {
                "status": "error", "source": "calendar", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False, "error": err, "per_slice": {},
            }
        if cfg is None:
            # Empty path / missing file / blank file ⇒ noop.
            return {
                "status": "noop", "source": "calendar", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False, "error": None, "per_slice": {},
            }
        resolved, missing = _resolve(cfg)
        if missing:
            return {
                "status": "error", "source": "calendar", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False,
                "error": f"unset env vars: {sorted(set(missing))}",
                "per_slice": {},
            }

        slices = resolved.get("calendars_to_sync")
        if not isinstance(slices, list) or len(slices) == 0:
            return {
                "status": "noop", "source": "calendar", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False, "error": None, "per_slice": {},
            }

        caldav_cfg = resolved.get("caldav") or {}
        lookback = int(resolved.get("sync_lookback_days") or 90)
        lookahead = int(resolved.get("sync_lookahead_days") or 365)

        # One CalDAV client shared across slices — same auth / connection.
        client = _client(caldav_cfg)

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
                client=client,
                slice_cfg=slice_cfg if isinstance(slice_cfg, dict) else {},
                dwh_dir=dwh_dir,
                budget=budget,
                window_end=window_end,
                lookback_days=lookback,
                lookahead_days=lookahead,
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
            "source": "calendar",
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
