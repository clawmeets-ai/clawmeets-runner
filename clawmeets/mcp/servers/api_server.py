# SPDX-License-Identifier: MIT
"""
clawmeets/mcp/servers/api_server.py

Generic HTTP-API MCP server. Reads a per-agent config from the file at
``{agent_dir}/mcp-hub/configs/http-api.json``, supplied to each tool call
as the ``config_file`` argument (the agent reads the path from its prompt's
``== MCP CONFIG FILES ==`` block). The config lists one or more REST
endpoints, then incrementally pulls rows from each into the personal data
warehouse on a scheduled trigger.

Each endpoint is a free-form HTTP request template (``method``, ``url``,
``headers``, ``query_params``, ``body``) with ``${VAR}`` placeholders. The
MCP injects reserved runtime tokens (``${WATERMARK_ISO}``, ``${CURSOR}``,
``${OFFSET}``, ``${PAGE_SIZE}``, …) per request and resolves user-set names
from ``os.environ``. Auth is just a value in ``headers`` — nothing
auth-specific in the schema.

Config schema (the file at the path the agent passes):

    {
      "endpoints": [
        {
          "name": "stripe-charges",
          "url": "https://api.stripe.com/v1/charges",
          "method": "GET",
          "headers": {"Authorization": "Bearer ${STRIPE_KEY}"},
          "query_params": {
            "created[gte]": "${WATERMARK_EPOCH}",
            "starting_after": "${CURSOR}",
            "limit": "${PAGE_SIZE}"
          },
          "body": null,
          "pagination": {"type": "cursor", "cursor_field": "id", "page_size": 100},
          "row_path": "data",
          "id_field": "id",
          "ts_field": "created"
        }
      ]
    }

``ts_field`` should be a strictly-increasing field on the row. Same-watermark
rows that cross a page boundary may be skipped — pick a field with sub-second
resolution if writes are bursty.
"""
from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

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
"""Replace-mode watermark sentinel; see db_server.py for rationale."""


def _rmdir_if_empty(path: Path) -> None:
    try:
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    except OSError:
        pass


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


def _load_config(config_file: str) -> tuple[Optional[dict], Optional[str]]:
    if not config_file:
        return None, (
            "config_file is required (pass the path from your "
            "`== MCP CONFIG FILES ==` prompt block, next to `http-api`)"
        )
    path = Path(config_file).expanduser()
    if not path.exists():
        return None, (
            f"config file not found at {path} — save the http-api config via "
            "the Configure modal in Agent Settings (see mcps/http-api/README.md)"
        )
    try:
        cfg = parse_jsonc(path.read_text())
    except Exception as exc:
        return None, f"config file is not valid JSON: {exc}"
    if not isinstance(cfg, dict):
        return None, "config file must contain a JSON object"
    endpoints = cfg.get("endpoints")
    if not isinstance(endpoints, list) or not endpoints:
        return None, "config file must list at least one endpoint under `endpoints`"
    return cfg, None


def _coerce_ts(value: Any) -> Optional[str]:
    """Render a row ts_field value as an ISO-8601 string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()
        except Exception:
            return f"{int(value):020d}"
    return str(value)


def _walk(obj: Any, path: Optional[str]) -> list:
    """Walk a dotted JSON path; return the resolved value as a list."""
    if not path:
        cur = obj
    else:
        cur = obj
        for part in path.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                return []
    if cur is None:
        return []
    if isinstance(cur, list):
        return cur
    if isinstance(cur, dict):
        return [cur]
    return []


# Per-endpoint ``response_format`` selects how the HTTP body is parsed into
# the row dicts that flow into ``data.tsv``. Default is ``"json"`` so existing
# configs keep working unchanged.
SUPPORTED_RESPONSE_FORMATS = ("json", "csv", "tsv")


def _parse_response(
    resp: "httpx.Response",
    response_format: str,
    row_path: Optional[str],
) -> tuple[list, Optional[str]]:
    """Decode the HTTP response body into a list of row dicts.

    Returns ``(rows, err)``:
      - ``("json", ...)``: ``resp.json()`` then ``_walk(payload, row_path)``.
      - ``("csv", ...)`` / ``("tsv", ...)``: parse ``resp.text`` with
        ``csv.DictReader`` (first row = header). ``row_path`` is ignored.
        Empty body ⇒ ``[]``.
      - Any other format ⇒ ``([], "unsupported response_format: ...")``.

    Decode errors return ``([], "<ExcType>: ...")``; the caller bubbles
    that up as a per-endpoint error envelope without touching ``data.tsv``.
    """
    if response_format == "json":
        try:
            payload = resp.json()
        except Exception as exc:
            return [], f"failed to parse JSON response: {type(exc).__name__}: {exc}"
        return _walk(payload, row_path), None
    if response_format in ("csv", "tsv"):
        delim = "," if response_format == "csv" else "\t"
        text = resp.text
        # Strip a UTF-8 BOM if present — common in Excel-exported CSVs and
        # makes the first column name look like ``"﻿id"`` otherwise.
        if text.startswith("﻿"):
            text = text[1:]
        if not text.strip():
            return [], None
        try:
            reader = csv.DictReader(io.StringIO(text), delimiter=delim)
            rows = [dict(r) for r in reader]
        except csv.Error as exc:
            return [], f"failed to parse {response_format.upper()} response: {exc}"
        return rows, None
    return [], (
        f"unsupported response_format: {response_format!r} — "
        f"must be one of {SUPPORTED_RESPONSE_FORMATS}"
    )


def _build_static_scope(window_start: str, window_end: str, page_size: int) -> dict[str, str]:
    """Tokens that don't change across pages within a single endpoint sync.

    ``OFFSET`` is included with a default of 0 so single-mode and cursor-mode
    requests can reference ``${OFFSET}`` without falling through to env or
    substituting empty string. Offset-paginated requests overwrite this in
    the per-request scope before each page.
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


def _expand_request(
    endpoint: dict,
    scope: dict[str, str],
) -> tuple[Optional[dict], list[str]]:
    """Expand ${VAR} tokens across the request-shaped fields. Returns (expanded, missing)."""
    template = {
        "method": endpoint.get("method") or "GET",
        "url": endpoint.get("url") or "",
        "headers": endpoint.get("headers") or {},
        "query_params": endpoint.get("query_params") or {},
        "body": endpoint.get("body"),
    }
    missing: list[str] = []
    expanded = expand_env(template, scope, missing)
    return expanded, missing


def _sync_one_endpoint(
    *,
    endpoint: dict,
    dwh_dir: str,
    budget: SyncBudget,
    window_end: str,
) -> dict:
    """Sync a single endpoint; return its per-endpoint summary.

    Durable shape: one ``raw/data.tsv`` per endpoint (append-only, header
    on creation; first row is the union of the first page's row keys,
    subsequent rows must match exactly). Rows are buffered in memory by
    ``TsvSliceWriter`` and flushed atomically once on the success/partial
    exit. On exception OR schema mismatch the buffer is discarded and the
    watermark is not advanced.
    """
    name = endpoint.get("name") or "<unnamed>"
    if not endpoint.get("url"):
        return {
            "name": name,
            "rows_written": 0,
            "watermarks": None,
            "has_more": False,
            "error": "endpoint missing required field: url",
        }

    merge_policy, upsert_id_column, merge_err = validate_merge_policy(endpoint)
    if merge_err:
        return {
            "name": name,
            "rows_written": 0,
            "watermarks": None,
            "has_more": False,
            "error": merge_err,
        }

    row_path = endpoint.get("row_path")
    # `id_field` is OPTIONAL. When set, we skip rows missing that field
    # (it's the row's identity contract for downstream dedup). When unset,
    # every well-shaped row gets written — useful for CSV dumps and other
    # snapshot-style sources that don't carry a stable primary key.
    id_field_raw = endpoint.get("id_field")
    id_field: Optional[str] = id_field_raw if isinstance(id_field_raw, str) and id_field_raw else None
    ts_field_raw = endpoint.get("ts_field")
    ts_field: Optional[str] = ts_field_raw if isinstance(ts_field_raw, str) and ts_field_raw else None
    pagination = endpoint.get("pagination") or {"type": "single"}
    pag_type = pagination.get("type") or "single"
    page_size = int(pagination.get("page_size") or 100)
    cursor_field = pagination.get("cursor_field") or id_field
    response_format = (endpoint.get("response_format") or "json").lower()

    dwh_root = Path(dwh_dir).expanduser().resolve()
    source = f"api/{name}"
    source_dir = dwh_root / "sources" / source
    state_path = source_dir / "sync-state.json"
    merged_path = dwh_root / "merged" / "api" / f"{name}.tsv"

    state = _read_state(state_path, source)
    high = state.get("high_watermark") or utcnow_iso()
    low = state.get("low_watermark") or high

    # Mirror howto to both layers before fetch — the howto describes the
    # endpoint's contract and stays valid even if the fetch errors out below.
    howto_err = write_howto(
        endpoint.get("howto"),
        source_dir=source_dir,
        merged_path=merged_path,
    )
    if howto_err:
        return {
            "name": name,
            "rows_written": 0,
            "watermarks": {"low": low, "high": high},
            "has_more": False,
            "error": howto_err,
        }

    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        return {
            "name": name,
            "rows_written": 0,
            "watermarks": {"low": low, "high": high},
            "has_more": False,
            "error": (
                f"unsupported response_format: {response_format!r} — "
                f"must be one of {SUPPORTED_RESPONSE_FORMATS}"
            ),
        }
    if pag_type == "cursor" and not cursor_field:
        return {
            "name": name,
            "rows_written": 0,
            "watermarks": {"low": low, "high": high},
            "has_more": False,
            "error": (
                "cursor pagination requires either `pagination.cursor_field` "
                "or `id_field` to be set"
            ),
        }

    if merge_policy == "replace":
        window_start = EPOCH_ISO
    else:
        window_start = max(low, high)

    timestamp_dir = new_timestamp_dir(source_dir)
    writer = TsvSliceWriter(timestamp_dir)

    rows_written_start = budget.rows_written
    latest_seen = window_start

    static_scope = _build_static_scope(window_start, window_end, page_size)

    has_more = False
    abort_err: Optional[str] = None  # set by schema-mismatch OR parse-error → discard buffer, no advance

    def _handle_row(row_obj: Any) -> bool:
        """Buffer one row. Returns False to break out of pagination
        (schema mismatch — distinguish via ``abort_err``). ``id_field``
        and ``ts_field`` filters apply only when configured (see the
        docstring on ``_sync_one_endpoint`` / db_server's
        ``_sync_one_query`` for the state-table-mode semantics).
        """
        nonlocal latest_seen, abort_err
        if not isinstance(row_obj, dict):
            return True
        if id_field is not None and row_obj.get(id_field) is None:
            return True  # CDC contract: drop rows missing the identity field
        if ts_field is not None:
            ts_value = row_obj.get(ts_field)
            ts_iso = _coerce_ts(ts_value)
            if ts_iso is None:
                # ts_field configured but null on this row — keep the row
                # with a synthesized ts inside the window, so the success-
                # path watermark math stays well-defined.
                ts_iso = window_start
            elif ts_iso >= window_end:
                return True  # log-structured contract: hold back for next sync
        else:
            # State-table mode: no per-row ts. Synth from window_start so
            # latest_seen stays put; the success-path watermark still
            # advances to window_end. User is expected to encode any
            # delta-filter in the request template via ${VAR} placeholders.
            ts_iso = window_start
        if not writer.add_row(row_obj):
            abort_err = writer.error
            return False
        if ts_iso > latest_seen:
            latest_seen = ts_iso
        budget.rows_written += 1
        return True

    try:
        with httpx.Client(timeout=60.0) as client:
            if pag_type == "cursor":
                cursor_value: Optional[str] = None  # omitted from scope on first request
                while True:
                    if budget.should_stop():
                        has_more = True
                        break
                    scope = dict(static_scope)
                    if cursor_value is not None:
                        scope["CURSOR"] = cursor_value
                    expanded, missing = _expand_request(endpoint, scope)
                    if missing:
                        _rmdir_if_empty(timestamp_dir)
                        return {
                            "name": name,
                            "rows_written": budget.rows_written - rows_written_start,
                            "watermarks": {"low": low, "high": high},
                            "has_more": False,
                            "error": f"unset env vars: {sorted(set(missing))}",
                        }
                    body = expanded["body"]
                    resp = client.request(
                        expanded["method"],
                        expanded["url"],
                        headers=expanded["headers"],
                        params=expanded["query_params"],
                        json=body if isinstance(body, dict) else None,
                        content=body if isinstance(body, str) else None,
                    )
                    resp.raise_for_status()
                    rows, parse_err = _parse_response(resp, response_format, row_path)
                    if parse_err:
                        abort_err = parse_err
                        break
                    if not rows:
                        break
                    last_row: Optional[dict] = None
                    page_aborted = False
                    for row in rows:
                        if budget.should_stop():
                            has_more = True
                            break
                        last_row = row if isinstance(row, dict) else None
                        if not _handle_row(row):
                            page_aborted = True
                            break
                    if page_aborted or abort_err is not None:
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
                    expanded, missing = _expand_request(endpoint, scope)
                    if missing:
                        _rmdir_if_empty(timestamp_dir)
                        return {
                            "name": name,
                            "rows_written": budget.rows_written - rows_written_start,
                            "watermarks": {"low": low, "high": high},
                            "has_more": False,
                            "error": f"unset env vars: {sorted(set(missing))}",
                        }
                    body = expanded["body"]
                    resp = client.request(
                        expanded["method"],
                        expanded["url"],
                        headers=expanded["headers"],
                        params=expanded["query_params"],
                        json=body if isinstance(body, dict) else None,
                        content=body if isinstance(body, str) else None,
                    )
                    resp.raise_for_status()
                    rows, parse_err = _parse_response(resp, response_format, row_path)
                    if parse_err:
                        abort_err = parse_err
                        break
                    if not rows:
                        break
                    page_aborted = False
                    for row in rows:
                        if budget.should_stop():
                            has_more = True
                            break
                        if not _handle_row(row):
                            page_aborted = True
                            break
                    if page_aborted or abort_err is not None:
                        break
                    if has_more:
                        break
                    if len(rows) < page_size:
                        break
                    offset += len(rows)
            else:  # single
                scope = dict(static_scope)
                expanded, missing = _expand_request(endpoint, scope)
                if missing:
                    _rmdir_if_empty(timestamp_dir)
                    return {
                        "name": name,
                        "rows_written": budget.rows_written - rows_written_start,
                        "watermarks": {"low": low, "high": high},
                        "has_more": False,
                        "error": f"unset env vars: {sorted(set(missing))}",
                    }
                body = expanded["body"]
                resp = client.request(
                    expanded["method"],
                    expanded["url"],
                    headers=expanded["headers"],
                    params=expanded["query_params"],
                    json=body if isinstance(body, dict) else None,
                    content=body if isinstance(body, str) else None,
                )
                resp.raise_for_status()
                rows, parse_err = _parse_response(resp, response_format, row_path)
                if parse_err:
                    abort_err = parse_err
                    rows = []
                for row in rows:
                    if budget.should_stop():
                        has_more = True
                        break
                    if not _handle_row(row):
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

    if abort_err is not None:
        _rmdir_if_empty(timestamp_dir)
        state["last_sync_at"] = utcnow_iso()
        state["last_sync_count"] = 0
        state["last_error"] = abort_err
        atomic_write_json(state_path, state)
        return {
            "name": name,
            "rows_written": 0,
            "watermarks": {"low": low, "high": high},
            "has_more": False,
            "error": abort_err,
        }

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


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The `mcp` package is required but missing — the clawmeets runner "
            "should bundle it by default. Try: pip install --upgrade clawmeets"
        ) from exc

    mcp = FastMCP("clawmeets-api")

    @mcp.tool()
    def sync_to_warehouse(
        dwh_dir: str,
        config_file: str,
        max_runtime_seconds: int = 1500,
    ) -> dict:
        """Sync configured HTTP endpoints into the personal data warehouse.

        Call this exactly once when you receive a DM whose body starts with
        ``<!-- clawmeets:api-sync-trigger -->``. Pass ``dwh_dir`` from your
        ``== DATA WAREHOUSE ==`` prompt block and ``config_file`` from your
        ``== MCP CONFIG FILES ==`` prompt block (the path next to
        ``http-api``).

        Reads ``config_file`` for the list of endpoints. Each
        endpoint is a request template with ``${VAR}`` placeholders; the MCP
        injects reserved runtime tokens (``WATERMARK_ISO``, ``WATERMARK_EPOCH``,
        ``WATERMARK_END_ISO``, ``WATERMARK_END_EPOCH``, ``CURSOR``, ``OFFSET``,
        ``PAGE_SIZE``) and falls through to ``os.environ`` for everything
        else. State per endpoint at
        ``{dwh_dir}/sources/api/<endpoint>/sync-state.json``; the run's rows
        land at ``{dwh_dir}/sources/api/<endpoint>/<TIMESTAMP>/data.tsv``
        and the consolidated dataset rebuilds at
        ``{dwh_dir}/merged/api/<endpoint>.tsv`` per the endpoint's
        ``merge_policy`` (default ``replace``; ``upsert`` requires
        ``merge_policy_upsert_id_column``). Up to ``KEEP_RECENT_DUMPS``
        timestamp folders are retained per endpoint for audit/recovery.

        On any merge-time error (``upsert`` with a missing id column, TSV
        header mismatch between the dump and the existing merged file, …)
        the per-endpoint summary carries the error and ``merged/`` is left
        untouched. The dump in ``<TIMESTAMP>/`` stays on disk so the next
        run can fold it in once the user fixes the config.

        Returns the standard sync envelope plus a ``per_endpoint`` map.
        ``status`` is ``error`` if config is missing/invalid or any user-set
        env var is unset; ``partial`` if any endpoint hit ``has_more=true``;
        ``noop`` if no rows written and no errors; ``ok`` otherwise.
        """
        cfg, err = _load_config(config_file)
        window_end = utcnow_iso()
        if err or cfg is None:
            return {
                "status": "error",
                "source": "api",
                "rows_written": 0,
                "window": [window_end, window_end],
                "watermarks": None,
                "has_more": False,
                "error": err,
                "per_endpoint": {},
            }

        budget = SyncBudget(max_runtime_seconds)
        per_endpoint: dict[str, dict] = {}
        any_error = False
        any_has_more = False
        agg_low: Optional[str] = None
        agg_high: Optional[str] = None
        first_error: Optional[str] = None

        for endpoint in cfg["endpoints"]:
            ep_name = endpoint.get("name") or "<unnamed>"
            if budget.should_stop():
                # Wall-clock budget elapsed before this endpoint started —
                # not an error, just unfinished. has_more=True signals
                # "resume next trigger"; error stays None.
                any_has_more = True
                per_endpoint[ep_name] = {
                    "name": ep_name,
                    "rows_written": 0,
                    "watermarks": None,
                    "has_more": True,
                    "error": None,
                }
                continue
            summary = _sync_one_endpoint(
                endpoint=endpoint,
                dwh_dir=dwh_dir,
                budget=budget,
                window_end=window_end,
            )
            per_endpoint[summary["name"]] = summary
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
            "source": "api",
            "rows_written": budget.rows_written,
            "window": [agg_low or window_end, window_end],
            "watermarks": {"low": agg_low, "high": agg_high} if (agg_low or agg_high) else None,
            "has_more": any_has_more,
            "error": first_error,
            "per_endpoint": per_endpoint,
        }

    mcp.run()


if __name__ == "__main__":
    main()
