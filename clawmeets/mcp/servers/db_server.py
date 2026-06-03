# SPDX-License-Identifier: MIT
"""
clawmeets/mcp/servers/db_server.py

Generic SQL-database MCP server. Reads a per-agent config from the file
at ``{agent_dir}/mcp-hub/configs/database.json``, supplied to each tool
call as the ``config_file`` argument (the agent reads the path from its
prompt's ``== MCP CONFIG FILES ==`` block). The config lists one or more
**raw SQL queries** with ``${VAR}`` placeholders, then incrementally
executes each into the personal data warehouse on a scheduled trigger.

Lazy-imports SQLAlchemy 2.x — the runner package does not pull it in by
default. If the user installs the database MCP without ``sqlalchemy`` and
the matching driver, ``sync_to_warehouse`` returns ``status="error"`` with
a clear install hint.

Each query is a free-form SQL template with ``${VAR}`` placeholders. The
MCP injects reserved runtime tokens (``${WATERMARK_ISO}``, ``${CURSOR}``,
``${OFFSET}``, ``${PAGE_SIZE}``, …) per request and resolves user-set names
from ``os.environ`` (think ``${PG_PWD}`` in the connection string).

Substitution is **naive string replacement**, uniform with the http-api MCP.
The user is responsible for quoting string-typed reserved tokens
(``'${WATERMARK_ISO}'``) in their SQL and for not putting SQL-special
characters in their own env vars referenced from SQL. Reserved tokens are
MCP-generated and contain no SQL-special characters by construction.

Config schema (the file at the path the agent passes):

    {
      "connection_string": "postgresql+psycopg://${PG_USER}:${PG_PWD}@host/db",
      "queries": [
        {
          "name": "orders",
          "sql": "SELECT id, customer_id, total_cents, updated_at FROM orders WHERE updated_at > '${WATERMARK_ISO}' AND updated_at < '${WATERMARK_END_ISO}' ORDER BY updated_at ASC LIMIT ${PAGE_SIZE} OFFSET ${OFFSET}",
          "id_field": "id",
          "ts_field": "updated_at",
          "pagination": {"type": "offset", "page_size": 500}
        },
        {
          "name": "events_keyset",
          "sql": "SELECT id, kind, payload, ts FROM events WHERE id > ${CURSOR} ORDER BY id ASC LIMIT ${PAGE_SIZE}",
          "id_field": "id",
          "ts_field": "id",
          "pagination": {"type": "cursor", "cursor_field": "id", "page_size": 500}
        }
      ]
    }

``ts_field`` should be a strictly-increasing column on the result rows. The
WHERE clause in your SQL is the contract — pick a column with sub-second
resolution if writes are bursty so same-watermark ties don't cross page
boundaries.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from clawmeets.utils.jsonc import parse_jsonc
from clawmeets.mcp.servers._sync_warehouse import (
    SyncBudget,
    TsvSliceWriter,
    atomic_write_json,
    expand_env,
    gc_old_timestamps,
    merge_tsv,
    new_timestamp_dir,
    utcnow_iso,
    validate_merge_policy,
    write_howto,
)


EPOCH_ISO = "1970-01-01T00:00:00+00:00"
"""Replace-mode watermark sentinel.

In replace mode the user is expected to omit ``${WATERMARK_*}`` from their
SQL/request template — the merge step (not the watermark) provides
full-refresh semantics. But a stray reference shouldn't make the query
fail: substituting epoch gives ``WHERE updated_at > '1970-01-01...'``,
which matches everything. ``EPOCH_ISO`` is not persisted in
``sync-state.json`` — replace mode never reads or writes the slice's
watermarks."""


def _read_state(state_path: Path, source: str) -> dict:
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    now = utcnow_iso()
    return {
        "source": source,
        "low_watermark": now,
        "high_watermark": now,
        "last_sync_at": None,
        "last_sync_count": 0,
        "last_error": None,
    }


def _coerce_ts(value: Any) -> Optional[str]:
    """Render a row ts_field value as an ISO-8601 string for watermark tracking."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, (int, float)):
        return f"{int(value):020d}"
    return str(value)


def _load_config(config_file: str) -> tuple[Optional[dict], Optional[str]]:
    if not config_file:
        return None, (
            "config_file is required (pass the path from your "
            "`== MCP CONFIG FILES ==` prompt block, next to `database`)"
        )
    path = Path(config_file).expanduser()
    if not path.exists():
        return None, (
            f"config file not found at {path} — save the database config via "
            "the Configure modal in Agent Settings (see mcps/database/README.md)"
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
    """``OFFSET`` defaults to 0 so single-mode and cursor-mode SQL templates
    can reference ``${OFFSET}`` (a common SQL idiom: ``LIMIT N OFFSET M``)
    without breaking. Offset-paginated queries overwrite this per page.
    """
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
    window_end: str,
) -> dict:
    """Sync a single query; return its per-query summary.

    Durable shape: one ``<TIMESTAMP>/data.tsv`` per run plus a merged
    ``{dwh_dir}/merged/database/<name>.tsv`` rebuilt per ``merge_policy``
    after each successful run. Rows are buffered in memory by
    ``TsvSliceWriter`` during pagination and flushed once at the success /
    partial exit; the merge step runs immediately after that flush. On
    exception OR merge error the watermark is **not** advanced, so the
    next sync retries the same window.
    """
    from sqlalchemy import text

    name = query_cfg.get("name")
    sql_template = query_cfg.get("sql")
    if not (name and sql_template):
        return {
            "name": name or "<unnamed>",
            "rows_written": 0,
            "watermarks": None,
            "has_more": False,
            "error": "query config missing one of: name, sql",
        }

    merge_policy, upsert_id_column, merge_err = validate_merge_policy(query_cfg)
    if merge_err:
        return {
            "name": name,
            "rows_written": 0,
            "watermarks": None,
            "has_more": False,
            "error": merge_err,
        }

    # ``id_field`` and ``ts_field`` are OPTIONAL:
    #   - When ``id_field`` is set, rows missing that column are skipped
    #     (per-row identity contract for downstream dedup). Unset/empty =
    #     every well-shaped row gets written.
    #   - When ``ts_field`` is set, the column drives the watermark and
    #     rows with ``ts >= window_end`` are held back for the next sync
    #     (log-structured contract). Unset/empty = every row's synthesized
    #     ts is ``window_start``, latest_seen stays at ``window_start``
    #     and the success-path watermark advances to ``window_end`` —
    #     suitable for non-log-structured "state table" syncs where the
    #     user owns the delta filter via ``${VAR}`` placeholders in the
    #     SQL itself.
    id_field_raw = query_cfg.get("id_field")
    ts_field_raw = query_cfg.get("ts_field")
    id_field: Optional[str] = id_field_raw if isinstance(id_field_raw, str) and id_field_raw else None
    ts_field: Optional[str] = ts_field_raw if isinstance(ts_field_raw, str) and ts_field_raw else None

    pagination = query_cfg.get("pagination") or {"type": "single"}
    pag_type = pagination.get("type") or "single"
    page_size = int(pagination.get("page_size") or 500)
    cursor_field = pagination.get("cursor_field") or id_field

    # Cursor pagination needs SOME column to advance through. Without
    # cursor_field (explicitly OR derived from id_field), the loop would
    # silently terminate after one page (last_row.get(None) is None). Fail
    # loudly instead — the user can pick offset/single or wire a column.
    if pag_type == "cursor" and not cursor_field:
        return {
            "name": name,
            "rows_written": 0,
            "watermarks": None,
            "has_more": False,
            "error": (
                "cursor pagination requires either `pagination.cursor_field` "
                "or `id_field` to be set"
            ),
        }

    dwh_root = Path(dwh_dir).expanduser().resolve()
    source = f"database/{name}"
    source_dir = dwh_root / "sources" / source
    state_path = source_dir / "sync-state.json"
    merged_path = dwh_root / "merged" / "database" / f"{name}.tsv"

    # Mirror howto to both layers before fetch — the howto describes the
    # slice's contract and stays valid even if the fetch errors out below.
    howto_err = write_howto(
        query_cfg.get("howto"),
        source_dir=source_dir,
        merged_path=merged_path,
    )
    if howto_err:
        return {
            "name": name,
            "rows_written": 0,
            "watermarks": None,
            "has_more": False,
            "error": howto_err,
        }

    state = _read_state(state_path, source)
    if merge_policy == "replace":
        # Replace mode: watermark state is informational only. The user's SQL
        # is expected to return the full source per run (no ${WATERMARK_*}
        # clause); we substitute EPOCH_ISO defensively so a stray reference
        # still yields a full refresh rather than an empty result.
        high = state.get("high_watermark") or utcnow_iso()
        low = state.get("low_watermark") or high
        window_start = EPOCH_ISO
    else:
        high = state.get("high_watermark") or utcnow_iso()
        low = state.get("low_watermark") or high
        window_start = max(low, high)

    timestamp_dir = new_timestamp_dir(source_dir)
    writer = TsvSliceWriter(timestamp_dir)

    static_scope = _build_static_scope(window_start, window_end, page_size)
    rows_written_start = budget.rows_written
    latest_seen = window_start

    has_more = False
    schema_err: Optional[str] = None  # set when add_row rejects a row

    def _handle_row(row_dict: dict) -> bool:
        """Buffer one row. Returns False to break out of pagination
        (schema mismatch — caller distinguishes via ``schema_err``).
        Updates ``latest_seen`` + ``budget.rows_written`` in-place via
        the enclosing scope. ``id_field`` / ``ts_field`` filters apply
        only when configured."""
        nonlocal latest_seen, schema_err
        if id_field is not None and row_dict.get(id_field) is None:
            return True  # skip rows missing the identity field (CDC contract)
        if ts_field is not None:
            ts_value = row_dict.get(ts_field)
            ts_iso = _coerce_ts(ts_value)
            if ts_iso is None:
                # ts_field configured but null on this row — keep the row,
                # synth ts so it stays inside the window and the watermark
                # logic below stays well-defined.
                ts_iso = window_start
            elif ts_iso >= window_end:
                return True  # log-structured contract: hold back for next sync
        else:
            # State-table mode: no per-row ts. Synth from window_start so
            # latest_seen stays put; success-path watermark advances to
            # window_end regardless.
            ts_iso = window_start
        if not writer.add_row(row_dict):
            schema_err = writer.error
            return False
        if ts_iso > latest_seen:
            latest_seen = ts_iso
        budget.rows_written += 1
        return True

    try:
        with engine.connect() as conn:
            if pag_type == "cursor":
                cursor_value: Optional[str] = None
                while True:
                    if budget.should_stop():
                        has_more = True
                        break
                    scope = dict(static_scope)
                    if cursor_value is not None:
                        scope["CURSOR"] = cursor_value
                    missing: list[str] = []
                    sql = expand_env(sql_template, scope, missing)
                    if missing:
                        return {
                            "name": name,
                            "rows_written": budget.rows_written - rows_written_start,
                            "watermarks": {"low": low, "high": high},
                            "has_more": False,
                            "error": f"unset env vars: {sorted(set(missing))}",
                        }
                    rows = conn.execute(text(sql)).mappings().all()
                    if not rows:
                        break
                    last_row: Optional[dict] = None
                    page_aborted = False
                    for row in rows:
                        if budget.should_stop():
                            has_more = True
                            break
                        last_row = dict(row)
                        if not _handle_row(last_row):
                            page_aborted = True
                            break
                    if page_aborted or schema_err is not None:
                        break
                    if has_more:
                        break
                    if last_row is None:
                        break
                    next_cursor = last_row.get(cursor_field)
                    if next_cursor is None or str(next_cursor) == cursor_value:
                        break
                    cursor_value = str(next_cursor)
            elif pag_type == "offset":
                offset = 0
                while True:
                    if budget.should_stop():
                        has_more = True
                        break
                    scope = dict(static_scope)
                    scope["OFFSET"] = str(offset)
                    missing = []
                    sql = expand_env(sql_template, scope, missing)
                    if missing:
                        return {
                            "name": name,
                            "rows_written": budget.rows_written - rows_written_start,
                            "watermarks": {"low": low, "high": high},
                            "has_more": False,
                            "error": f"unset env vars: {sorted(set(missing))}",
                        }
                    rows = conn.execute(text(sql)).mappings().all()
                    if not rows:
                        break
                    page_aborted = False
                    for row in rows:
                        if budget.should_stop():
                            has_more = True
                            break
                        if not _handle_row(dict(row)):
                            page_aborted = True
                            break
                    if page_aborted or schema_err is not None:
                        break
                    if has_more:
                        break
                    if len(rows) < page_size:
                        break
                    offset += len(rows)
            else:  # single
                scope = dict(static_scope)
                missing = []
                sql = expand_env(sql_template, scope, missing)
                if missing:
                    return {
                        "name": name,
                        "rows_written": budget.rows_written - rows_written_start,
                        "watermarks": {"low": low, "high": high},
                        "has_more": False,
                        "error": f"unset env vars: {sorted(set(missing))}",
                    }
                rows = conn.execute(text(sql)).mappings().all()
                for row in rows:
                    if budget.should_stop():
                        has_more = True
                        break
                    if not _handle_row(dict(row)):
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
            "has_more": False,
            "error": err,
        }

    if schema_err is not None:
        # Two rows within this run had different shapes — buffer is discarded,
        # nothing lands on disk, watermark NOT advanced. Surface to the user
        # so they fix the query before retrying.
        _rmdir_if_empty(timestamp_dir)
        state["last_sync_at"] = utcnow_iso()
        state["last_sync_count"] = 0
        state["last_error"] = schema_err
        atomic_write_json(state_path, state)
        return {
            "name": name,
            "rows_written": 0,
            "watermarks": {"low": low, "high": high},
            "has_more": False,
            "error": schema_err,
        }

    # Success / partial: flush the per-run buffer, then merge into the
    # consolidated dataset and GC old timestamp folders.
    try:
        writer.flush()
    except OSError as exc:
        err = f"OSError flushing data.tsv: {exc}"
        _rmdir_if_empty(timestamp_dir)
        state["last_sync_at"] = utcnow_iso()
        state["last_sync_count"] = budget.rows_written - rows_written_start
        state["last_error"] = err
        atomic_write_json(state_path, state)
        return {
            "name": name,
            "rows_written": budget.rows_written - rows_written_start,
            "watermarks": {"low": low, "high": high},
            "has_more": False,
            "error": err,
        }

    # Empty fetch: no data.tsv was written. Clean up the empty timestamp dir
    # so it doesn't crowd the slice's parent. Watermark still advances for
    # upsert mode (the window was processed cleanly, just yielded no rows).
    _rmdir_if_empty(timestamp_dir)

    if timestamp_dir.exists():
        merge_err_msg = merge_tsv(
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
                "has_more": False,
                "error": merge_err_msg,
            }
        gc_old_timestamps(source_dir)

    if merge_policy == "replace":
        # Watermarks aren't authoritative in replace mode — leave them
        # untouched so a later switch back to upsert resumes from the prior
        # high. Track last_sync_at + last_sync_count + last_error only.
        state["last_sync_at"] = utcnow_iso()
        state["last_sync_count"] = budget.rows_written - rows_written_start
        state["last_error"] = None
        atomic_write_json(state_path, state)
        return {
            "name": name,
            "rows_written": budget.rows_written - rows_written_start,
            "watermarks": {"low": low, "high": high},
            "has_more": has_more,
            "error": None,
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
        "has_more": has_more,
        "error": None,
    }


def _rmdir_if_empty(path: Path) -> None:
    """Remove ``path`` if it's an empty directory; ignore otherwise.

    Used after a sync run to clean up a freshly-created timestamp folder
    that ended up with nothing written into it (empty window, exception
    before any flush, etc.) — keeps the slice's parent free of empty stubs.
    """
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

    mcp = FastMCP("clawmeets-db")

    @mcp.tool()
    def sync_to_warehouse(
        dwh_dir: str,
        config_file: str,
        max_runtime_seconds: int = 1500,
    ) -> dict:
        """Sync configured SQL queries into the personal data warehouse.

        Call this exactly once when you receive a DM whose body starts with
        ``<!-- clawmeets:db-sync-trigger -->``. Pass ``dwh_dir`` from your
        ``== DATA WAREHOUSE ==`` prompt block and ``config_file`` from your
        ``== MCP CONFIG FILES ==`` prompt block (the path next to ``database``).

        Reads ``config_file`` for the connection string and
        the list of queries to run. Each query is a SQL template with
        ``${VAR}`` placeholders; the MCP injects reserved runtime tokens
        (``WATERMARK_ISO``, ``WATERMARK_EPOCH``, ``WATERMARK_END_ISO``,
        ``WATERMARK_END_EPOCH``, ``CURSOR``, ``OFFSET``, ``PAGE_SIZE``) and
        falls through to ``os.environ`` for everything else. State per query
        at ``{dwh_dir}/sources/database/<name>/sync-state.json``; the run's
        rows land at ``{dwh_dir}/sources/database/<name>/<TIMESTAMP>/data.tsv``
        and the consolidated dataset rebuilds at
        ``{dwh_dir}/merged/database/<name>.tsv`` per the query's
        ``merge_policy`` (default ``replace``; ``upsert`` requires
        ``merge_policy_upsert_id_column``). Up to ``KEEP_RECENT_DUMPS``
        timestamp folders are retained per query for audit/recovery.

        On any merge-time error (``upsert`` with a missing id column, TSV
        header mismatch between the dump and the existing merged file, …)
        the per-query summary carries the error and ``merged/`` is left
        untouched. The dump in ``<TIMESTAMP>/`` stays on disk so the next
        run can fold it in once the user fixes the config.

        Returns the standard sync envelope plus a ``per_query`` map.
        ``status`` is ``error`` if config is missing/invalid, SQLAlchemy
        isn't installed, or any user-set env var is unset; ``partial`` if
        any query hit ``has_more=true``; ``noop`` if no rows written and no
        errors; ``ok`` otherwise.
        """
        try:
            from sqlalchemy import create_engine
        except ImportError:
            return {
                "status": "error",
                "source": "database",
                "rows_written": 0,
                "window": None,
                "watermarks": None,
                "has_more": False,
                "error": (
                    "sqlalchemy is not installed. On the runner, install it "
                    "with the driver for your DB: e.g. "
                    "`pip install sqlalchemy psycopg[binary]` (Postgres), "
                    "`pip install sqlalchemy pymysql` (MySQL), or just "
                    "`pip install sqlalchemy` (SQLite)."
                ),
                "per_query": {},
            }

        cfg, err = _load_config(config_file)
        window_end = utcnow_iso()
        if err or cfg is None:
            return {
                "status": "error",
                "source": "database",
                "rows_written": 0,
                "window": [window_end, window_end],
                "watermarks": None,
                "has_more": False,
                "error": err,
                "per_query": {},
            }

        # Connection-string ${VAR} substitution (env vars only — no reserved
        # tokens are meaningful here). Connection strings are URL-grammar so
        # naive replacement is safe; it's the standard libpq env-indirection
        # pattern dressed up in our token syntax.
        missing: list[str] = []
        connection_string = expand_env(cfg["connection_string"], {}, missing)
        if missing:
            return {
                "status": "error",
                "source": "database",
                "rows_written": 0,
                "window": [window_end, window_end],
                "watermarks": None,
                "has_more": False,
                "error": f"unset env vars in connection_string: {sorted(set(missing))}",
                "per_query": {},
            }

        try:
            engine = create_engine(connection_string)
        except Exception as exc:
            return {
                "status": "error",
                "source": "database",
                "rows_written": 0,
                "window": [window_end, window_end],
                "watermarks": None,
                "has_more": False,
                "error": f"could not create engine: {type(exc).__name__}: {exc}",
                "per_query": {},
            }

        budget = SyncBudget(max_runtime_seconds)
        per_query: dict[str, dict] = {}
        any_error = False
        any_has_more = False
        agg_low: Optional[str] = None
        agg_high: Optional[str] = None
        first_error: Optional[str] = None

        for query_cfg in cfg["queries"]:
            q_name = query_cfg.get("name") or "<unnamed>"
            if budget.should_stop():
                # Wall-clock budget elapsed before this query started — not
                # an error, just unfinished. has_more=True signals "resume
                # next trigger"; error stays None so the top-level status
                # compute classifies the run as 'partial' rather than 'error'.
                any_has_more = True
                per_query[q_name] = {
                    "name": q_name,
                    "rows_written": 0,
                    "watermarks": None,
                    "has_more": True,
                    "error": None,
                }
                continue
            summary = _sync_one_query(
                engine=engine,
                query_cfg=query_cfg,
                dwh_dir=dwh_dir,
                budget=budget,
                window_end=window_end,
            )
            per_query[summary["name"]] = summary
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

        engine.dispose()

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
            "source": "database",
            "rows_written": budget.rows_written,
            "window": [agg_low or window_end, window_end],
            "watermarks": {"low": agg_low, "high": agg_high} if (agg_low or agg_high) else None,
            "has_more": any_has_more,
            "error": first_error,
            "per_query": per_query,
        }

    mcp.run()


if __name__ == "__main__":
    main()
