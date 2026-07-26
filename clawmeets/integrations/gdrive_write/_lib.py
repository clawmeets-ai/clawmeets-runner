# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/gdrive_write/_lib.py

Google Sheets read+write — per-target configuration via the per-agent config
under ``sheet_targets``. Paired skill: ``google-drive-write``.

Separate from the read-only ``google-drive`` integration so the OAuth write
scope is opt-in. Tokens at
``$CLAWMEETS_AGENT_DIR/skill-hub/state/google-drive-write/token.json``.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from clawmeets.integrations._config_resolve import resolve_skill_config_path

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

_COL_LETTERS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
]


def build_service(token_path: Path):
    from googleapiclient.discovery import build
    from clawmeets.integrations.auth.google_oauth import load_credentials

    creds = load_credentials(token_path, SCOPES)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def _resolve_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if "${" not in value:
        return value
    out = value
    for key, val in os.environ.items():
        out = out.replace(f"${{{key}}}", val)
    return out


def load_targets(config_file: str) -> dict[str, dict]:
    """Read the per-agent config; return resolved sheet targets keyed by name.

    Raises ``RuntimeError`` if no valid targets remain or names collide.
    """
    config_file = resolve_skill_config_path("google-drive-write", config_file)
    if not config_file:
        raise RuntimeError(
            "google-drive-write config not found. Open Agent Settings → "
            "Skills → google-drive-write → Configure to set up sheet_targets."
        )
    p = Path(config_file).expanduser()
    if not p.exists():
        raise RuntimeError(f"google-drive-write config file not found: {p}")
    raw = p.read_text()
    try:
        from clawmeets.utils.jsonc import parse_jsonc
        cfg = parse_jsonc(raw)
    except Exception:
        cfg = json.loads(raw)
    if not isinstance(cfg, dict):
        raise RuntimeError(
            f"google-drive-write config must be a JSON object, got {type(cfg).__name__}"
        )

    raw_targets = cfg.get("sheet_targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise RuntimeError(
            "google-drive-write config: `sheet_targets` must be a non-empty list."
        )

    resolved: dict[str, dict] = {}
    for entry in raw_targets:
        if not isinstance(entry, dict):
            logger.warning("google-drive-write: ignoring non-dict entry %r", entry)
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            logger.warning("google-drive-write: entry missing `name`: %r", entry)
            continue
        spreadsheet_id = _resolve_env(entry.get("spreadsheet_id"))
        sheet_name = _resolve_env(entry.get("sheet_name"))
        row_id_column = entry.get("row_id_column") or "id"
        if not isinstance(spreadsheet_id, str) or not spreadsheet_id or spreadsheet_id.startswith("${"):
            logger.warning(
                "google-drive-write: entry %r missing/unresolved spreadsheet_id", name,
            )
            continue
        if not isinstance(sheet_name, str) or not sheet_name:
            logger.warning("google-drive-write: entry %r missing sheet_name", name)
            continue
        if name in resolved:
            raise RuntimeError(
                f"google-drive-write config: duplicate sheet_targets name {name!r}."
            )
        resolved[name] = {
            "spreadsheet_id": spreadsheet_id,
            "sheet_name": sheet_name,
            "row_id_column": row_id_column,
        }

    if not resolved:
        raise RuntimeError(
            "google-drive-write config: no valid entries in `sheet_targets`."
        )
    return resolved


def _resolve_target(targets: dict[str, dict], target_name: str) -> dict:
    if not isinstance(target_name, str) or not target_name:
        raise RuntimeError(
            f"`target_name` is required. Valid names: {sorted(targets.keys())}"
        )
    if target_name not in targets:
        raise RuntimeError(
            f"unknown target_name {target_name!r}. Valid names: {sorted(targets.keys())}"
        )
    return targets[target_name]


def _read_grid(svc, spreadsheet_id: str, sheet_name: str) -> list[list[str]]:
    resp = svc.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=sheet_name,
        valueRenderOption="FORMATTED_VALUE",
    ).execute()
    rows = resp.get("values", []) or []
    return [[("" if c is None else str(c)) for c in r] for r in rows]


def _grid_to_records(grid: list[list[str]]) -> tuple[list[str], list[dict]]:
    if not grid:
        return [], []
    header = grid[0]
    records: list[dict] = []
    for raw_row in grid[1:]:
        row = list(raw_row) + [""] * max(0, len(header) - len(raw_row))
        records.append({header[i]: row[i] for i in range(len(header))})
    return header, records


def _col_letter(idx: int) -> str:
    if idx < 0 or idx >= len(_COL_LETTERS):
        raise RuntimeError(
            f"column index {idx} out of range (sheet has > {len(_COL_LETTERS)} cols)"
        )
    return _COL_LETTERS[idx]


def read_sheet_rows(svc, config_file: str, target_name: str) -> dict:
    """Read every row of the named target as a list of dicts keyed by the header."""
    targets = load_targets(config_file)
    tgt = _resolve_target(targets, target_name)
    grid = _read_grid(svc, tgt["spreadsheet_id"], tgt["sheet_name"])
    header, records = _grid_to_records(grid)
    return {"header": header, "rows": records}


def append_sheet_rows(svc, config_file: str, target_name: str, rows: list[dict]) -> dict:
    """Append rows to the named target, ordering cells by the existing header."""
    if not isinstance(rows, list):
        raise RuntimeError("`rows` must be a list of dicts")
    targets = load_targets(config_file)
    tgt = _resolve_target(targets, target_name)
    grid = _read_grid(svc, tgt["spreadsheet_id"], tgt["sheet_name"])
    if not grid:
        raise RuntimeError(
            f"sheet {tgt['sheet_name']!r} is empty — add a header row first."
        )
    header = grid[0]
    values: list[list[str]] = []
    for r in rows:
        if not isinstance(r, dict):
            raise RuntimeError("each row must be a dict keyed by column name")
        values.append([str(r.get(col, "")) for col in header])
    if not values:
        return {"appended_count": 0, "updated_range": None}
    resp = svc.spreadsheets().values().append(
        spreadsheetId=tgt["spreadsheet_id"],
        range=tgt["sheet_name"],
        valueInputOption="RAW", insertDataOption="INSERT_ROWS",
        body={"values": values},
    ).execute()
    return {
        "appended_count": len(values),
        "updated_range": (resp.get("updates") or {}).get("updatedRange"),
    }


def update_sheet_cell(
    svc,
    config_file: str, target_name: str,
    row_id: str, column: str, value: str,
) -> dict:
    """Update one cell by id-column lookup."""
    targets = load_targets(config_file)
    tgt = _resolve_target(targets, target_name)
    grid = _read_grid(svc, tgt["spreadsheet_id"], tgt["sheet_name"])
    if not grid:
        raise RuntimeError(f"sheet {tgt['sheet_name']!r} is empty")
    header = grid[0]
    try:
        id_idx = header.index(tgt["row_id_column"])
    except ValueError:
        raise RuntimeError(
            f"id column {tgt['row_id_column']!r} not found in header: {header}"
        )
    try:
        col_idx = header.index(column)
    except ValueError:
        raise RuntimeError(
            f"column {column!r} not found in header: {header}. Add it to the sheet first."
        )
    target_row: Optional[int] = None
    for i in range(1, len(grid)):
        row = grid[i]
        if id_idx < len(row) and row[id_idx] == row_id:
            target_row = i + 1
            break
    if target_row is None:
        raise RuntimeError(f"no row found where {tgt['row_id_column']}={row_id!r}")
    col_letter = _col_letter(col_idx)
    cell_range = f"{tgt['sheet_name']}!{col_letter}{target_row}"
    svc.spreadsheets().values().update(
        spreadsheetId=tgt["spreadsheet_id"],
        range=cell_range,
        valueInputOption="RAW",
        body={"values": [[value]]},
    ).execute()
    return {
        "updated_range": cell_range,
        "row_number": target_row,
        "column_letter": col_letter,
    }
