# SPDX-License-Identifier: MIT
"""
clawmeets/mcp/servers/gdrive_write_server.py

Google Sheets write MCP server. Read AND write N configured Google Sheet
targets — append rows, update cells by primary-key lookup, read all rows.
Each tool takes a ``target_name`` selecting which configured sheet to
operate on; targets live in the per-agent config under ``sheet_targets``.

Separate from the read-only `google-drive` MCP so the user can opt into
Sheets-write OAuth scope independently (and so a single agent can hold
both an OAuth token for read-only Drive sync AND one for Sheets write,
without scope collisions).

Reads the OAuth token from the path in CLAWMEETS_GDRIVE_WRITE_TOKEN_FILE.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# A1-notation column letters. 26 cols is more than the marketing schema needs
# and keeps the lookup simple; widen by extending this if a real workflow
# accretes more columns.
_COL_LETTERS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
]


def _token_path() -> Path:
    p = os.environ.get("CLAWMEETS_GDRIVE_WRITE_TOKEN_FILE")
    if not p:
        raise RuntimeError(
            "CLAWMEETS_GDRIVE_WRITE_TOKEN_FILE is not set. The google-drive-write "
            "MCP server is expected to be launched by the clawmeets runner, which "
            "sets this via the mcps/google-drive-write/mcp.json launch spec."
        )
    return Path(p)


def _sheets_service():
    from googleapiclient.discovery import build
    from clawmeets.mcp.auth.google_oauth import load_credentials

    creds = load_credentials(_token_path(), SCOPES)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def _resolve_env(value: Any) -> Any:
    """Resolve `${VAR}` placeholders in a string against runner env. Non-strings
    pass through. Missing env vars are left as the literal placeholder so the
    caller can surface a clearer error than KeyError."""
    if not isinstance(value, str):
        return value
    if "${" not in value:
        return value
    out = value
    for key, val in os.environ.items():
        out = out.replace(f"${{{key}}}", val)
    return out


def _load_targets(config_file: str) -> dict[str, dict]:
    """Read the per-agent config and return a dict of resolved sheet targets
    keyed by target ``name``.

    Expected shape:
        {
          "sheet_targets": [
            {
              "name": "ideas",                    // logical key the LLM uses
              "spreadsheet_id": "...",            // literal id or "${VAR}"
              "sheet_name": "ideas",              // tab name (NOT file name)
              "row_id_column": "id"               // optional; defaults to "id"
            },
            ...
          ]
        }

    Each resolved target is a dict ``{spreadsheet_id, sheet_name, row_id_column}``
    with ${VAR} placeholders resolved against the runner env. Invalid entries
    (missing fields, unresolved placeholders, non-dicts) are logged and
    skipped, mirroring the read-side gdrive_server `_extract_sheet_tabs`
    posture so one typo doesn't take the whole config down. Raises if no
    valid targets remain or names collide.
    """
    if not config_file:
        raise RuntimeError(
            "config_file is required. Pass the path from your prompt's "
            "`== MCP CONFIG FILES ==` block (next to `google-drive-write`)."
        )
    p = Path(config_file).expanduser()
    if not p.exists():
        raise RuntimeError(f"google-drive-write config file not found: {p}")
    raw = p.read_text()
    # Tolerate JSONC-style comments + trailing commas via clawmeets.utils.jsonc.
    try:
        from clawmeets.utils.jsonc import parse_jsonc
        cfg = parse_jsonc(raw)
    except Exception:
        cfg = json.loads(raw)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"google-drive-write config must be a JSON object, got {type(cfg).__name__}")

    raw_targets = cfg.get("sheet_targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise RuntimeError(
            "google-drive-write config: `sheet_targets` must be a non-empty "
            "list. See mcps/google-drive-write/starter_config.jsonc for the "
            "expected shape."
        )

    resolved: dict[str, dict] = {}
    for entry in raw_targets:
        if not isinstance(entry, dict):
            logger.warning("google-drive-write: ignoring non-dict sheet_targets entry %r", entry)
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            logger.warning("google-drive-write: sheet_targets entry missing `name`: %r", entry)
            continue
        spreadsheet_id = _resolve_env(entry.get("spreadsheet_id"))
        sheet_name = _resolve_env(entry.get("sheet_name"))
        row_id_column = entry.get("row_id_column") or "id"
        if not isinstance(spreadsheet_id, str) or not spreadsheet_id or spreadsheet_id.startswith("${"):
            logger.warning(
                "google-drive-write: sheet_targets entry %r has missing/unresolved spreadsheet_id",
                name,
            )
            continue
        if not isinstance(sheet_name, str) or not sheet_name:
            logger.warning("google-drive-write: sheet_targets entry %r has missing sheet_name", name)
            continue
        if name in resolved:
            raise RuntimeError(
                f"google-drive-write config: duplicate sheet_targets name {name!r}. "
                f"Each target's `name` must be unique."
            )
        resolved[name] = {
            "spreadsheet_id": spreadsheet_id,
            "sheet_name": sheet_name,
            "row_id_column": row_id_column,
        }

    if not resolved:
        raise RuntimeError(
            "google-drive-write config: no valid entries in `sheet_targets`. "
            "Each entry needs `name`, `spreadsheet_id` (resolved), and `sheet_name`."
        )
    return resolved


def _resolve_target(targets: dict[str, dict], target_name: str) -> dict:
    """Pick one target out of the resolved dict; raise with the valid names
    listed inline so the LLM can self-correct on the next call."""
    if not isinstance(target_name, str) or not target_name:
        raise RuntimeError(
            f"`target_name` is required. Valid names from this config: "
            f"{sorted(targets.keys())}"
        )
    if target_name not in targets:
        raise RuntimeError(
            f"unknown target_name {target_name!r}. Valid names from this config: "
            f"{sorted(targets.keys())}"
        )
    return targets[target_name]


def _read_grid(svc, spreadsheet_id: str, sheet_name: str) -> list[list[str]]:
    """Pull every row of the named sheet as a 2D list of strings."""
    resp = svc.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=sheet_name,
        valueRenderOption="FORMATTED_VALUE",
    ).execute()
    rows = resp.get("values", []) or []
    return [[("" if c is None else str(c)) for c in r] for r in rows]


def _grid_to_records(grid: list[list[str]]) -> tuple[list[str], list[dict]]:
    """Split a grid into (header, records). Empty grid → ([], [])."""
    if not grid:
        return [], []
    header = grid[0]
    records: list[dict] = []
    for raw_row in grid[1:]:
        # Pad short rows with empty strings so every record carries every key.
        row = list(raw_row) + [""] * max(0, len(header) - len(raw_row))
        records.append({header[i]: row[i] for i in range(len(header))})
    return header, records


def _col_letter(idx: int) -> str:
    """0-indexed column → A1-notation letter. Caps at len(_COL_LETTERS)-1."""
    if idx < 0 or idx >= len(_COL_LETTERS):
        raise RuntimeError(
            f"column index {idx} out of range (sheet has > {len(_COL_LETTERS)} columns; "
            f"widen _COL_LETTERS in gdrive_write_server.py if needed)"
        )
    return _COL_LETTERS[idx]


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The `mcp` package is required but missing — the clawmeets runner "
            "should bundle it by default. Try: pip install --upgrade clawmeets"
        ) from exc

    mcp = FastMCP("clawmeets-gdrive-write")

    @mcp.tool()
    def read_sheet_rows(config_file: str, target_name: str) -> dict:
        """Read every row of the named sheet target as a list of dicts keyed
        by the header row.

        Pass the config path from your prompt's `== MCP CONFIG FILES ==` block
        (next to `google-drive-write`) and the ``target_name`` matching one of
        the entries in the config's ``sheet_targets`` list.

        Returns ``{header, rows}``. ``rows`` is empty if the sheet has only a
        header (or is blank). Cell values are always strings (formatted value).
        """
        targets = _load_targets(config_file)
        tgt = _resolve_target(targets, target_name)
        svc = _sheets_service()
        grid = _read_grid(svc, tgt["spreadsheet_id"], tgt["sheet_name"])
        header, records = _grid_to_records(grid)
        return {"header": header, "rows": records}

    @mcp.tool()
    def append_sheet_rows(config_file: str, target_name: str, rows: list[dict]) -> dict:
        """Append rows to the named sheet target, ordering each row's cells by
        the existing header.

        Unknown keys in ``rows[i]`` are silently dropped — extend the header in
        the Sheet first if you need a new column. Missing keys are written as
        empty strings.

        Returns ``{appended_count, updated_range}``.
        """
        if not isinstance(rows, list):
            raise RuntimeError("`rows` must be a list of dicts")
        targets = _load_targets(config_file)
        tgt = _resolve_target(targets, target_name)
        svc = _sheets_service()
        grid = _read_grid(svc, tgt["spreadsheet_id"], tgt["sheet_name"])
        if not grid:
            raise RuntimeError(
                f"sheet {tgt['sheet_name']!r} is empty — add a header row "
                f"before appending. See the README for the marketing schema."
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
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": values},
        ).execute()
        return {
            "appended_count": len(values),
            "updated_range": (resp.get("updates") or {}).get("updatedRange"),
        }

    @mcp.tool()
    def update_sheet_cell(
        config_file: str,
        target_name: str,
        row_id: str,
        column: str,
        value: str,
    ) -> dict:
        """Update a single cell on the named sheet target by looking up the
        row whose id-column value matches ``row_id`` and writing ``value``
        into the named ``column``.

        The id-column defaults to ``id`` (configurable per target via
        ``row_id_column`` in the per-agent config).

        Returns ``{updated_range, row_number, column_letter}``. Raises if the
        row or column can't be found.
        """
        targets = _load_targets(config_file)
        tgt = _resolve_target(targets, target_name)
        svc = _sheets_service()
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
                f"column {column!r} not found in header: {header}. Add it to "
                f"the sheet first (or fix the agent's prompt to use an existing column)."
            )
        # 1-indexed: row 1 is the header, so the first data row is row 2.
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

    mcp.run()


if __name__ == "__main__":
    main()
