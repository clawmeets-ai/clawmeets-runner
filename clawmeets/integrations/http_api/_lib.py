# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/http_api/_lib.py

Generic HTTP-API sync. Each endpoint is a free-form HTTP request template
(``method``, ``url``, ``headers``, ``query_params``, ``body``) with
``${VAR}`` placeholders. Auth is just a value in ``headers``.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

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

SUPPORTED_RESPONSE_FORMATS = ("json", "csv", "tsv")


def _is_incremental(endpoint: dict) -> bool:
    """Incremental iff the request template references a watermark/cursor token
    (in url / params / headers / body) — else a full fetch the driver diffs."""
    blob = json.dumps(endpoint, default=str)
    return ("${WATERMARK" in blob) or ("${CURSOR" in blob)


def load_config(config_file: str) -> tuple[Optional[dict], Optional[str]]:
    config_file = resolve_skill_config_path("http-api", config_file)
    if not config_file:
        return None, (
            "config_file is required (set up Agent Settings → Skills → "
            "http-api → Configure first)"
        )
    path = Path(config_file).expanduser()
    if not path.exists():
        return None, f"config file not found at {path}"
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


def _parse_response(
    resp: httpx.Response, response_format: str, row_path: Optional[str],
) -> tuple[list, Optional[str]]:
    if response_format == "json":
        try:
            payload = resp.json()
        except Exception as exc:
            return [], f"failed to parse JSON response: {type(exc).__name__}: {exc}"
        return _walk(payload, row_path), None
    if response_format in ("csv", "tsv"):
        delim = "," if response_format == "csv" else "\t"
        text = resp.text
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


def _expand_request(endpoint: dict, scope: dict[str, str]) -> tuple[Optional[dict], list[str]]:
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
) -> dict:
    name = endpoint.get("name") or "<unnamed>"
    if not endpoint.get("url"):
        return {"name": name, "rows_written": 0, "watermarks": None,
                "has_more": False, "error": "endpoint missing required field: url"}

    id_field = endpoint.get("id_field")
    if not isinstance(id_field, str) or not id_field:
        return {"name": name, "rows_written": 0, "watermarks": None,
                "has_more": False,
                "error": "endpoint missing required `id_field` (the row's stable key — "
                         "the integration can't infer it for an arbitrary API response)"}
    row_path = endpoint.get("row_path")
    ts_field_raw = endpoint.get("ts_field")
    ts_field = ts_field_raw if isinstance(ts_field_raw, str) and ts_field_raw else None
    pagination = endpoint.get("pagination") or {"type": "single"}
    pag_type = pagination.get("type") or "single"
    page_size = int(pagination.get("page_size") or 100)
    cursor_field = pagination.get("cursor_field") or id_field
    response_format = (endpoint.get("response_format") or "json").lower()

    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        return {"name": name, "rows_written": 0, "watermarks": None, "has_more": False,
                "error": (f"unsupported response_format: {response_format!r} — "
                          f"must be one of {SUPPORTED_RESPONSE_FORMATS}")}
    if pag_type == "cursor" and not cursor_field:
        return {"name": name, "rows_written": 0, "watermarks": None, "has_more": False,
                "error": "cursor pagination requires `pagination.cursor_field` or `id_field`"}

    incremental = _is_incremental(endpoint)
    dwh_root = Path(dwh_dir).expanduser().resolve()
    source = f"api/{name}"
    base = dwh_root / "raw" / source

    howto_err = write_howto(endpoint.get("howto"), snapshot_dir=base)
    if howto_err:
        return {"name": name, "rows_written": 0, "watermarks": None,
                "has_more": False, "error": howto_err}

    def _normalize(row_obj: dict) -> dict:
        if ts_field is not None:
            row_obj[ts_field] = _coerce_ts(row_obj.get(ts_field)) or ""
        return row_obj

    def _request(client, scope):
        expanded, missing = _expand_request(endpoint, scope)
        if missing:
            raise RuntimeError(f"unset env vars: {sorted(set(missing))}")
        body = expanded["body"]
        resp = client.request(
            expanded["method"], expanded["url"],
            headers=expanded["headers"], params=expanded["query_params"],
            json=body if isinstance(body, dict) else None,
            content=body if isinstance(body, str) else None,
        )
        resp.raise_for_status()
        rows, parse_err = _parse_response(resp, response_format, row_path)
        if parse_err:
            raise RuntimeError(parse_err)
        return rows

    def fetch(window_start: str, window_end: str, bud: SyncBudget, emit) -> bool:
        static_scope = _build_static_scope(window_start, window_end, page_size)
        with httpx.Client(timeout=60.0) as client:
            if pag_type == "cursor":
                cursor_value: Optional[str] = None
                while True:
                    if bud.should_stop():
                        return True
                    scope = dict(static_scope)
                    if cursor_value is not None:
                        scope["CURSOR"] = cursor_value
                    rows = _request(client, scope)
                    if not rows:
                        return False
                    last_row: Optional[dict] = None
                    for row in rows:
                        if bud.should_stop():
                            return True
                        if not isinstance(row, dict) or row.get(id_field) is None:
                            continue
                        last_row = row
                        bud.rows_written += 1
                        emit(_normalize(dict(row)))
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
                    rows = _request(client, scope)
                    if not rows:
                        return False
                    for row in rows:
                        if bud.should_stop():
                            return True
                        if not isinstance(row, dict) or row.get(id_field) is None:
                            continue
                        bud.rows_written += 1
                        emit(_normalize(dict(row)))
                    if len(rows) < page_size:
                        return False
                    offset += len(rows)
            else:  # single
                rows = _request(client, dict(static_scope))
                for row in rows:
                    if bud.should_stop():
                        return True
                    if not isinstance(row, dict) or row.get(id_field) is None:
                        continue
                    bud.rows_written += 1
                    emit(_normalize(dict(row)))
                return False

    return run_slice_sync(
        source=source, dwh_dir=dwh_dir, budget=budget, fetch=fetch,
        id_field=id_field, ts_field=ts_field or "ts",
        start_at=endpoint.get("start_at"),
        full_scan=not incremental, snapshot_fmt="tsv",
    )


def sync_to_warehouse(
    dwh_dir: str,
    config_file: str = "",
    max_runtime_seconds: int = 1500,
) -> dict:
    """Sync configured HTTP endpoints into the warehouse.

    Triggered by ``<!-- clawmeets:api-sync-trigger -->``.
    """
    cfg, err = load_config(config_file)
    window_end = utcnow_iso()
    if err or cfg is None:
        return {
            "status": "error", "source": "api", "rows_written": 0,
            "window": [window_end, window_end], "watermarks": None,
            "has_more": False, "error": err, "per_endpoint": {},
        }

    budget = SyncBudget(max_runtime_seconds)
    return run_slices(
        source_family="api", slices=cfg["endpoints"], budget=budget,
        dwh_dir=dwh_dir,
        run_one=lambda ep: _sync_one_endpoint(
            endpoint=ep, dwh_dir=dwh_dir, budget=budget),
    )
