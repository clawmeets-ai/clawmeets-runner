# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/database/_lib.py

Generic SQL-database sync. Reads ``connection_string`` + a list of named
``queries`` from the per-agent config; each query is a SQL template with
``${VAR}`` placeholders (``WATERMARK_ISO`` / ``CURSOR`` / ``OFFSET`` /
``PAGE_SIZE`` injected by the runtime; everything else falls through to
``os.environ``).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from clawmeets.integrations._config_resolve import resolve_skill_config_path
from clawmeets.integrations._sync_warehouse import (
    EPOCH_ISO,
    SyncBudget,
    expand_env,
    run_slice_sync,
    run_slices,
    utcnow_iso,
    write_howto,
)
from clawmeets.utils.jsonc import parse_jsonc


def _is_incremental(sql_template: str) -> bool:
    """A query is incremental iff it references a watermark/cursor token —
    otherwise it's a full-table fetch the driver diffs for changes + deletes."""
    return ("${WATERMARK" in sql_template) or ("${CURSOR" in sql_template)


def _coerce_ts(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, (int, float)):
        return f"{int(value):020d}"
    return str(value)


def load_config(config_file: str) -> tuple[Optional[dict], Optional[str]]:
    config_file = resolve_skill_config_path("database", config_file)
    if not config_file:
        return None, (
            "config_file is required (set up Agent Settings → Skills → "
            "database → Configure first)"
        )
    path = Path(config_file).expanduser()
    if not path.exists():
        return None, (
            f"config file not found at {path} — save the database config via "
            "the Configure modal in Agent Settings"
        )
    try:
        cfg = parse_jsonc(path.read_text())
    except Exception as exc:
        return None, f"config file is not valid JSON: {exc}"
    if not isinstance(cfg, dict):
        return None, "config file must contain a JSON object"
    if not cfg.get("connection_string"):
        return None, "config file missing required field: connection_string"
    queries = cfg.get("queries")
    if not isinstance(queries, list) or not queries:
        return None, "config file must list at least one query under `queries`"
    return cfg, None


def _build_static_scope(window_start: str, window_end: str, page_size: int) -> dict[str, str]:
    def to_epoch(iso: str) -> str:
        try:
            return str(int(datetime.fromisoformat(iso).timestamp()))
        except Exception:
            return ""
    return {
        "WATERMARK_ISO": window_start,
        "WATERMARK_EPOCH": to_epoch(window_start),
        "WATERMARK_END_ISO": window_end,
        "WATERMARK_END_EPOCH": to_epoch(window_end),
        "PAGE_SIZE": str(page_size),
        "OFFSET": "0",
    }


def _sync_one_query(
    *,
    engine,
    query_cfg: dict,
    dwh_dir: str,
    budget: SyncBudget,
) -> dict:
    from sqlalchemy import text

    name = query_cfg.get("name")
    sql_template = query_cfg.get("sql")
    if not (name and sql_template):
        return {"name": name or "<unnamed>", "rows_written": 0, "watermarks": None,
                "has_more": False, "error": "query config missing one of: name, sql"}

    id_field = query_cfg.get("id_field")
    if not isinstance(id_field, str) or not id_field:
        return {"name": name, "rows_written": 0, "watermarks": None,
                "has_more": False,
                "error": "query config missing required `id_field` (the primary-key "
                         "column — the integration can't infer it for an arbitrary query)"}
    ts_field_raw = query_cfg.get("ts_field")
    ts_field = ts_field_raw if isinstance(ts_field_raw, str) and ts_field_raw else None

    pagination = query_cfg.get("pagination") or {"type": "single"}
    pag_type = pagination.get("type") or "single"
    page_size = int(pagination.get("page_size") or 500)
    cursor_field = pagination.get("cursor_field") or id_field
    if pag_type == "cursor" and not cursor_field:
        return {"name": name, "rows_written": 0, "watermarks": None,
                "has_more": False,
                "error": "cursor pagination requires `pagination.cursor_field` or `id_field`"}

    incremental = _is_incremental(sql_template)
    dwh_root = Path(dwh_dir).expanduser().resolve()
    source = f"database/{name}"
    base = dwh_root / "raw" / source

    howto_err = write_howto(query_cfg.get("howto"), snapshot_dir=base)
    if howto_err:
        return {"name": name, "rows_written": 0, "watermarks": None,
                "has_more": False, "error": howto_err}

    def _normalize(row_dict: dict) -> dict:
        if ts_field is not None:
            row_dict[ts_field] = _coerce_ts(row_dict.get(ts_field)) or ""
        return row_dict

    def fetch(window_start: str, window_end: str, bud: SyncBudget, emit) -> bool:
        static_scope = _build_static_scope(window_start, window_end, page_size)
        with engine.connect() as conn:
            if pag_type == "cursor":
                cursor_value: Optional[str] = None
                while True:
                    if bud.should_stop():
                        return True
                    scope = dict(static_scope)
                    if cursor_value is not None:
                        scope["CURSOR"] = cursor_value
                    missing: list[str] = []
                    sql = expand_env(sql_template, scope, missing)
                    if missing:
                        raise RuntimeError(f"unset env vars: {sorted(set(missing))}")
                    rows = conn.execute(text(sql)).mappings().all()
                    if not rows:
                        return False
                    last_row: Optional[dict] = None
                    for row in rows:
                        if bud.should_stop():
                            return True
                        last_row = dict(row)
                        bud.rows_written += 1
                        emit(_normalize(dict(last_row)))
                    if last_row is None:
                        return False
                    next_cursor = last_row.get(cursor_field)
                    if next_cursor is None or str(next_cursor) == cursor_value:
                        return False
                    cursor_value = str(next_cursor)
            elif pag_type == "offset":
                offset = 0
                while True:
                    if bud.should_stop():
                        return True
                    scope = dict(static_scope)
                    scope["OFFSET"] = str(offset)
                    missing = []
                    sql = expand_env(sql_template, scope, missing)
                    if missing:
                        raise RuntimeError(f"unset env vars: {sorted(set(missing))}")
                    rows = conn.execute(text(sql)).mappings().all()
                    if not rows:
                        return False
                    for row in rows:
                        if bud.should_stop():
                            return True
                        bud.rows_written += 1
                        emit(_normalize(dict(row)))
                    if len(rows) < page_size:
                        return False
                    offset += len(rows)
            else:  # single
                scope = dict(static_scope)
                missing = []
                sql = expand_env(sql_template, scope, missing)
                if missing:
                    raise RuntimeError(f"unset env vars: {sorted(set(missing))}")
                rows = conn.execute(text(sql)).mappings().all()
                for row in rows:
                    if bud.should_stop():
                        return True
                    bud.rows_written += 1
                    emit(_normalize(dict(row)))
                return False

    return run_slice_sync(
        source=source, dwh_dir=dwh_dir, budget=budget, fetch=fetch,
        id_field=id_field, ts_field=ts_field or "ts",
        start_at=query_cfg.get("start_at"),
        full_scan=not incremental, snapshot_fmt="tsv",
    )


def sync_to_warehouse(
    dwh_dir: str,
    config_file: str = "",
    max_runtime_seconds: int = 1500,
) -> dict:
    """Sync configured SQL queries into the data warehouse.

    Triggered by ``<!-- clawmeets:db-sync-trigger -->``.
    """
    try:
        from sqlalchemy import create_engine
    except ImportError:
        return {
            "status": "error", "source": "database", "rows_written": 0,
            "window": None, "watermarks": None, "has_more": False,
            "error": (
                "sqlalchemy is not installed. On the runner, install it with "
                "the matching driver: `pip install sqlalchemy psycopg[binary]` "
                "(Postgres), `pip install sqlalchemy pymysql` (MySQL), etc."
            ),
            "per_query": {},
        }

    cfg, err = load_config(config_file)
    window_end = utcnow_iso()
    if err or cfg is None:
        return {
            "status": "error", "source": "database", "rows_written": 0,
            "window": [window_end, window_end], "watermarks": None,
            "has_more": False, "error": err, "per_query": {},
        }

    missing: list[str] = []
    connection_string = expand_env(cfg["connection_string"], {}, missing)
    if missing:
        return {
            "status": "error", "source": "database", "rows_written": 0,
            "window": [window_end, window_end], "watermarks": None,
            "has_more": False,
            "error": f"unset env vars in connection_string: {sorted(set(missing))}",
            "per_query": {},
        }

    try:
        engine = create_engine(connection_string)
    except Exception as exc:
        return {
            "status": "error", "source": "database", "rows_written": 0,
            "window": [window_end, window_end], "watermarks": None,
            "has_more": False,
            "error": f"could not create engine: {type(exc).__name__}: {exc}",
            "per_query": {},
        }

    budget = SyncBudget(max_runtime_seconds)
    try:
        return run_slices(
            source_family="database", slices=cfg["queries"], budget=budget,
            dwh_dir=dwh_dir,
            run_one=lambda qc: _sync_one_query(
                engine=engine, query_cfg=qc, dwh_dir=dwh_dir, budget=budget),
        )
    finally:
        engine.dispose()
