# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/caldav/_lib.py

CalDAV calendar integration. Provider-agnostic (iCloud / Fastmail /
Nextcloud / Radicale / SOGo / mailcow). Credentials from ``${VAR}`` in
``$CLAWMEETS_AGENT_DIR/skill-hub/configs/calendar.json``.

Skill name remains ``calendar`` (matches the existing trigger
``<!-- clawmeets:calendar-sync-trigger -->``).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from clawmeets.integrations._config_resolve import resolve_skill_config_path
from clawmeets.integrations._sync_warehouse import (
    SyncBudget,
    expand_env,
    run_slice_sync,
    run_slices,
    utcnow_iso,
    write_howto,
)
from clawmeets.utils.file_io import FileUtil
from clawmeets.utils.jsonc import parse_jsonc


def load_config(config_file: str) -> tuple[Optional[dict], Optional[str]]:
    config_file = resolve_skill_config_path("calendar", config_file)
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
    cfg, err = load_config(config_file)
    if cfg is not None:
        return cfg
    if err is None:
        raise RuntimeError(
            "calendar config not found. Set up Agent Settings → Skills → "
            "calendar → Configure first."
        )
    raise RuntimeError(err)


def _resolve(cfg: dict, scope: Optional[dict[str, str]] = None) -> tuple[dict, list[str]]:
    missing: list[str] = []
    expanded = expand_env(cfg, scope or {}, missing)
    return expanded, missing


def _client(caldav_cfg: dict):
    try:
        import caldav  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "the `caldav` package is missing — install it: pip install caldav vobject"
        ) from exc
    url = caldav_cfg.get("url")
    if not url:
        raise RuntimeError("caldav.url is required")
    user = caldav_cfg.get("username") or ""
    password = caldav_cfg.get("password") or ""
    if not user or not password:
        raise RuntimeError(
            "caldav.username and caldav.password are required (resolve env vars first)"
        )
    return caldav.DAVClient(url=url, username=user, password=password)


def _select_calendars(client, names: Optional[list[str]]) -> list:
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
    cals = _select_calendars(client, [name])
    return cals[0] if cals else None


def _slice_calendar_names(slices: Any) -> Optional[list[str]]:
    if not isinstance(slices, list) or not slices:
        return None
    out = [
        s["calendar"] for s in slices
        if isinstance(s, dict) and isinstance(s.get("calendar"), str) and s["calendar"]
    ]
    return out or None


def _ical_to_event(ical_obj, calendar_url: str, etag: Optional[str]) -> Optional[dict]:
    try:
        import vobject  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "the `vobject` package is missing — install it: pip install caldav vobject"
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
        return str(dt)

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
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in uid)


def _sync_one_slice(*, client, slice_cfg, dwh_dir, budget,
                    lookback_days, lookahead_days):
    raw_name = slice_cfg.get("name") if isinstance(slice_cfg, dict) else None
    if not isinstance(raw_name, str) or not raw_name.strip():
        return {"name": "<unnamed>", "rows_written": 0, "watermarks": None,
                "has_more": False, "error": "slice config missing 'name' field"}
    try:
        name = FileUtil.validate_fs_name(raw_name)
    except ValueError as exc:
        return {"name": raw_name, "rows_written": 0, "watermarks": None,
                "has_more": False, "error": f"invalid slice name {raw_name!r}: {exc}"}

    calendar_name = slice_cfg.get("calendar")
    if not isinstance(calendar_name, str) or not calendar_name.strip():
        return {"name": name, "rows_written": 0, "watermarks": None,
                "has_more": False,
                "error": "slice config missing 'calendar' field (CalDAV display name)"}

    dwh_root = Path(dwh_dir).expanduser().resolve()
    source = f"calendar/{name}"
    base = dwh_root / "raw" / source

    howto_err = write_howto(slice_cfg.get("howto"), snapshot_dir=base)
    if howto_err:
        return {"name": name, "rows_written": 0, "watermarks": None,
                "has_more": False, "error": howto_err}

    cal = _select_one_calendar(client, calendar_name)
    if cal is None:
        return {"name": name, "rows_written": 0, "watermarks": None,
                "has_more": False,
                "error": f"calendar {calendar_name!r} not found on the configured CalDAV account"}

    now = datetime.now(timezone.utc)
    range_start = now - timedelta(days=lookback_days)
    range_end = now + timedelta(days=lookahead_days)
    range_start_iso = range_start.isoformat()
    range_end_iso = range_end.isoformat()

    def fetch(_window_start: str, window_end: str, bud: SyncBudget, emit) -> bool:
        # Full re-read of the [now-lookback, now+lookahead] window; the driver
        # diffs against the snapshot to find changes + deletes.
        try:
            events = cal.search(start=range_start, end=range_end, expand=False)
        except Exception as exc:
            raise RuntimeError(f"calendar.search failed: {exc}") from exc
        for ev in events:
            if bud.should_stop():
                return True
            etag = getattr(ev, "etag", None)
            envelope = _ical_to_event(ev, str(cal.url), etag)
            if envelope is None:
                continue
            uid = envelope.get("uid") or ""
            if not uid:
                continue
            envelope["ts"] = envelope.get("last_modified") or window_end
            envelope["slice"] = name
            bud.rows_written += 1
            emit(envelope)
        return False

    def _in_scope(prior: dict) -> bool:
        start = prior.get("start") or ""
        return range_start_iso <= start <= range_end_iso

    return run_slice_sync(
        source=source, dwh_dir=dwh_dir, budget=budget, fetch=fetch,
        id_field="uid", ts_field="ts", full_scan=True, snapshot_fmt="ndjson",
        in_scope=_in_scope, volatile_fields={"etag"},
    )


# ---------------------------------------------------------------------------
# Public tool bodies
# ---------------------------------------------------------------------------


def list_calendars(config_file: str) -> list[dict]:
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


def list_events(config_file, time_min, time_max, calendar_url=None) -> list[dict]:
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


def get_event(config_file, uid, calendar_url=None) -> dict:
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


def create_event(config_file, summary, start, end, calendar_url=None,
                 description=None, location=None, attendees=None) -> dict:
    try:
        from icalendar import Calendar, Event  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "the `icalendar` package is missing — install: pip install icalendar"
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
    ical.add("prodid", "-//clawmeets//caldav//EN")
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


def update_event(config_file, uid, calendar_url=None,
                 summary=None, start=None, end=None,
                 description=None, location=None) -> dict:
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


def delete_event(config_file, uid, calendar_url=None) -> str:
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


def sync_to_warehouse(dwh_dir, config_file="", max_runtime_seconds=1500) -> dict:
    """Sync CalDAV calendars into the warehouse.

    Triggered by ``<!-- clawmeets:calendar-sync-trigger -->``.
    """
    window_end = utcnow_iso()
    cfg, err = load_config(config_file)
    if err:
        return {"status": "error", "source": "calendar", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False, "error": err, "per_slice": {}}
    if cfg is None:
        return {"status": "noop", "source": "calendar", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False, "error": None, "per_slice": {}}
    resolved, missing = _resolve(cfg)
    if missing:
        return {"status": "error", "source": "calendar", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False,
                "error": f"unset env vars: {sorted(set(missing))}",
                "per_slice": {}}

    slices = resolved.get("calendars_to_sync")
    if not isinstance(slices, list) or len(slices) == 0:
        return {"status": "noop", "source": "calendar", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False, "error": None, "per_slice": {}}

    caldav_cfg = resolved.get("caldav") or {}
    lookback = int(resolved.get("sync_lookback_days") or 90)
    lookahead = int(resolved.get("sync_lookahead_days") or 365)
    client = _client(caldav_cfg)

    budget = SyncBudget(max_runtime_seconds)
    return run_slices(
        source_family="calendar", slices=slices, budget=budget,
        dwh_dir=dwh_dir,
        run_one=lambda sc: _sync_one_slice(
            client=client, slice_cfg=sc, dwh_dir=dwh_dir, budget=budget,
            lookback_days=lookback, lookahead_days=lookahead),
    )
