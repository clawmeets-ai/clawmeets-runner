# SPDX-License-Identifier: MIT
"""
clawmeets/mcp/servers/gdrive_server.py

Google Drive MCP server. Read-only access to a user's Drive — list/search
files and sync incremental changes into the personal data warehouse. Runs as
a stdio subprocess of Claude Code.

Reads the OAuth token from the path in CLAWMEETS_GDRIVE_TOKEN_FILE.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from clawmeets.mcp.servers._sync_warehouse import atomic_write_json
from clawmeets.utils.jsonc import parse_jsonc
from clawmeets.utils.validation import validate_name

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


# Drive file ids are URL-safe base64-ish; require ≥10 chars to filter typos.
_FOLDER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,}$")

# Drive's exportable Google-native types map to plain text equivalents.
# Spreadsheets export as TSV — agents downstream parse with csv.reader(delimiter='\t').
GDOC_EXPORT_MIME = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/tab-separated-values",
    "application/vnd.google-apps.presentation": "text/plain",
}

# Sidecar file extension per Google-native mime — written next to the JSON
# envelope in the run's timestamp folder so downstream tools can open the
# export directly without parsing the warehouse envelope.
SIDECAR_EXT = {
    "application/vnd.google-apps.document": ".txt",
    "application/vnd.google-apps.spreadsheet": ".tsv",
    "application/vnd.google-apps.presentation": ".txt",
}

# Plain-text mime types whose body is small enough to embed verbatim.
TEXT_MIME_PREFIXES = ("text/",)
TEXT_MIME_EXACT = {"application/json", "application/xml", "application/x-yaml"}

# Hard ceiling on inlined body bytes — avoids dumping a 50 MB markdown export
# into the warehouse JSON. Larger bodies are skipped (path + metadata only).
MAX_INLINE_BYTES = 256 * 1024


def _token_path() -> Path:
    p = os.environ.get("CLAWMEETS_GDRIVE_TOKEN_FILE")
    if not p:
        raise RuntimeError(
            "CLAWMEETS_GDRIVE_TOKEN_FILE is not set. The Google Drive MCP "
            "server is expected to be launched by the clawmeets runner, which "
            "sets this via the mcps/google-drive/mcp.json launch spec."
        )
    return Path(p)


def _service():
    from googleapiclient.discovery import build
    from clawmeets.mcp.auth.google_oauth import load_credentials

    creds = load_credentials(_token_path(), SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _sheets_service():
    """Build a Sheets API v4 client on the same credentials. Required to
    enumerate per-tab data for multi-tab spreadsheets (Drive's `files().export`
    only exports the first tab). `drive.readonly` already grants read access
    via Sheets API."""
    from googleapiclient.discovery import build
    from clawmeets.mcp.auth.google_oauth import load_credentials

    creds = load_credentials(_token_path(), SCOPES)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


_FILENAME_BAD_CHARS = re.compile(r"[\x00-\x1f\x7f/\\:*?<>|\"]+")
_FILENAME_WS = re.compile(r"\s+")


def _safe_filename_segment(s: str, *, fallback: str = "file", max_len: int = 80) -> str:
    """Sanitize a Drive name or sheet tab title into a filename segment that's
    safe on any common filesystem. Replaces path separators and control / shell
    metacharacters with a dash, normalizes unicode, collapses whitespace, trims
    surrounding dots/spaces, caps length. Empty after cleanup → ``fallback``.
    """
    if not s:
        return fallback
    s = unicodedata.normalize("NFC", s)
    s = _FILENAME_BAD_CHARS.sub("-", s)
    s = _FILENAME_WS.sub(" ", s).strip()
    s = s.strip(".-_ ")
    if len(s) > max_len:
        s = s[:max_len].rstrip(".-_ ")
    return s or fallback


def _rows_to_tsv(rows: list[list[Any]]) -> str:
    """Serialize a 2D Sheets API ``values`` payload as TSV. Uses csv.writer so
    cells containing tabs/newlines/quotes get RFC-4180 quoting; the agent's
    `csv.reader(delimiter='\\t')` round-trips cleanly. Missing trailing cells
    on short rows are preserved as empty strings up to the widest row width."""
    if not rows:
        return ""
    width = max((len(r) for r in rows), default=0)
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    for r in rows:
        padded = ["" if v is None else str(v) for v in r]
        if len(padded) < width:
            padded.extend([""] * (width - len(padded)))
        w.writerow(padded)
    return buf.getvalue()


def _list_spreadsheet_tabs(sheets_svc, file_id: str) -> Optional[list[dict]]:
    """Return ``[{title, index}, ...]`` for each tab in a spreadsheet, sorted
    by Drive's natural index. Returns None on API error so the caller can fall
    back to Drive export (first tab only)."""
    try:
        meta = sheets_svc.spreadsheets().get(
            spreadsheetId=file_id,
            fields="sheets(properties(sheetId,title,index))",
        ).execute()
    except Exception as e:
        logger.warning("gdrive: failed to list tabs for %s: %s", file_id, e)
        return None
    props = []
    for s in meta.get("sheets", []) or []:
        p = s.get("properties") or {}
        props.append({"title": p.get("title") or "", "index": p.get("index", 0)})
    props.sort(key=lambda p: p["index"])
    return props


def _fetch_tab_tsv(
    sheets_svc, file_id: str, tab_title: str
) -> tuple[Optional[str], Optional[str]]:
    """Fetch a single tab as TSV text.

    Returns ``(tsv, reason)``:
      - ``(tsv, None)``: success. ``tsv`` may be ``""`` for a blank tab.
      - ``(tsv, "oversize: <N>")``: fetch succeeded but the rendered TSV
        exceeds ``MAX_INLINE_BYTES``. ``tsv`` is the full text — the caller
        decides whether to persist it. The ``file_ids`` path skips oversize
        tabs (a runaway tab would balloon the merged JSON); the
        ``sheet_tabs`` path writes the sidecar regardless and only gates
        inline ``content``.
      - ``(None, "api_error: <Exception>")``: Sheets API call failed.
    """
    try:
        resp = sheets_svc.spreadsheets().values().get(
            spreadsheetId=file_id,
            range=tab_title,
            valueRenderOption="FORMATTED_VALUE",
        ).execute()
    except Exception as e:
        logger.warning(
            "gdrive: failed to fetch tab %r in %s: %s", tab_title, file_id, e,
        )
        return None, f"api_error: {type(e).__name__}: {e}"
    rows = resp.get("values", []) or []
    tsv = _rows_to_tsv(rows)
    size = len(tsv.encode("utf-8"))
    if size > MAX_INLINE_BYTES:
        logger.info(
            "gdrive: tab %r in %s exceeds %d bytes (got %d)",
            tab_title, file_id, MAX_INLINE_BYTES, size,
        )
        return tsv, f"oversize: {size}"
    return tsv, None


def _build_spreadsheet_sidecars(
    sheets_svc, file_id: str, name: str,
) -> tuple[Optional[str], list[tuple[str, str]]]:
    """Resolve the on-disk sidecar set for a Google Sheets file.

    Returns ``(first_tab_tsv, [(sidecar_filename, tsv_text), ...])``:
      - ``first_tab_tsv`` is the TSV body of the first tab (used to populate
        the JSON envelope's ``content`` field for downstream backwards-compat;
        ``None`` if listing/fetching the first tab failed or it's too large).
      - The list pairs each tab's sidecar filename with its TSV body. Naming
        rule: single-tab → ``{name}.tsv``, multi-tab → ``{name}-{tab}.tsv``.
        Tabs that fail to fetch or exceed the size cap are silently skipped
        (the others still get written).
    """
    tabs = _list_spreadsheet_tabs(sheets_svc, file_id)
    if not tabs:
        return None, []
    safe_name = _safe_filename_segment(name, fallback=file_id)
    sidecars: list[tuple[str, str]] = []
    first_tab_tsv: Optional[str] = None
    multi = len(tabs) > 1
    for i, tab in enumerate(tabs):
        title = tab["title"]
        body, reason = _fetch_tab_tsv(sheets_svc, file_id, title)
        # Preserve file_ids-path semantics: oversize first tab leaves
        # `content` null (the merged JSON would otherwise inline a runaway
        # tab — this is a whole-workbook fetch, not a surgical one).
        if i == 0:
            first_tab_tsv = body if reason is None else None
        if body is None or reason is not None:
            continue  # error or oversize — skip just this tab
        if multi:
            safe_tab = _safe_filename_segment(title, fallback=f"sheet-{i + 1}")
            fname = f"{safe_name}-{safe_tab}.tsv"
        else:
            fname = f"{safe_name}.tsv"
        sidecars.append((fname, body))
    return first_tab_tsv, sidecars


def _write_sidecars(
    raw_root: Path,
    file_id: str,
    name: str,
    mime: str,
    body: Optional[str],
    sheets_svc_factory: Callable[[], Any],
    previous_sidecars: list[str],
) -> tuple[Optional[str], list[str]]:
    """Materialize sidecar files for one synced item and clean up stale
    sidecars from a prior name (rename support).

    Returns ``(content_override, sidecar_filenames)``:
      - ``content_override``: a replacement value for the envelope's
        ``content`` field. For spreadsheets, this is the first tab's TSV
        (sourced from Sheets API, possibly different from Drive's
        first-tab export). For other types, the original ``body`` is used —
        callers should ignore this when it is ``None``.
      - ``sidecar_filenames``: relative filenames (not paths) of every sidecar
        written under ``raw_root``. Persisted in the envelope so the next
        sync can clean up renames.
    """
    # First, evict any sidecars this file_id wrote previously. The envelope's
    # `sidecars` list is the source of truth — if the file was renamed,
    # previous_sidecars points at the old-name siblings.
    for prev in previous_sidecars or []:
        if not isinstance(prev, str) or "/" in prev or "\\" in prev or ".." in prev:
            continue  # defensive — never delete outside raw_root
        old = raw_root / prev
        if old.is_file():
            try:
                old.unlink()
            except OSError as e:
                logger.warning("gdrive: failed to delete stale sidecar %s: %s", old, e)

    ext = SIDECAR_EXT.get(mime)
    if ext is None:
        return body, []

    raw_root.mkdir(parents=True, exist_ok=True)

    if mime == "application/vnd.google-apps.spreadsheet":
        # Spreadsheets get per-tab sidecars; first tab also drives `content`.
        first_tab_tsv, sidecars = _build_spreadsheet_sidecars(
            sheets_svc_factory(), file_id, name,
        )
        if sidecars:
            written: list[str] = []
            for fname, tsv in sidecars:
                (raw_root / fname).write_text(tsv, encoding="utf-8")
                written.append(fname)
            # Envelope `content` follows the first tab's TSV when we have it.
            return (first_tab_tsv if first_tab_tsv is not None else body), written
        # Sheets API gave us nothing useful (transient error, restricted
        # workbook, etc.). Fall back to a single sidecar from Drive's
        # first-tab export so the user still has a `.tsv` file on disk.
        if body is not None:
            safe_name = _safe_filename_segment(name, fallback=file_id)
            fname = f"{safe_name}.tsv"
            (raw_root / fname).write_text(body, encoding="utf-8")
            return body, [fname]
        return body, []

    # Docs / Slides → single sidecar with the plain-text body, if we got one.
    if body is None:
        return body, []
    safe_name = _safe_filename_segment(name, fallback=file_id)
    fname = f"{safe_name}{ext}"
    (raw_root / fname).write_text(body, encoding="utf-8")
    return body, [fname]


def _read_previous_envelope(path: Path) -> Optional[dict]:
    """Return the prior envelope for a given file_id, if any. Used to look up
    the previous ``sidecars`` list so we can evict orphans on rename."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _is_text_mime(mime: str) -> bool:
    return mime.startswith(TEXT_MIME_PREFIXES) or mime in TEXT_MIME_EXACT


def _load_config(config_file: str) -> tuple[Optional[dict], Optional[str]]:
    """Read this MCP's config from the file path supplied by the agent
    (sourced from the prompt's ``== MCP CONFIG FILES ==`` block).

    Returns ``(cfg, err)``:
      - ``(dict, None)`` on a successfully-parsed dict-shaped config
      - ``(None, None)`` when the file path is empty, missing, or empty (the
        caller treats this as a clean noop — fresh installs don't error)
      - ``(None, "...")`` when the file is malformed JSONC or its root isn't
        a dict (the caller surfaces this as a structured error envelope so
        the user knows to fix the config rather than silently no-op'ing)
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
        return None, f"malformed JSONC in {path}: {exc}"
    if not isinstance(cfg, dict):
        return None, f"config root must be a JSON object, got {type(cfg).__name__}"
    return cfg, None


def _validated_folder_ids(slice_cfg: dict) -> list[str]:
    """Validate + return the slice's ``folder_ids`` list.

    Same shape as ``_extract_file_ids`` but for the list-path scope.
    Invalid ids are logged at WARNING and skipped so a typo doesn't take
    the whole sync down. Read once per slice and reused by both
    ``_build_query`` (which consumes the validated list) and the
    ``_sync_one_slice`` no-scope gate (which only needs to know whether
    the list is non-empty).
    """
    raw = slice_cfg.get("folder_ids") or []
    if not isinstance(raw, list):
        return []
    valid: list[str] = []
    for fid in raw:
        if isinstance(fid, str) and _FOLDER_ID_RE.match(fid):
            valid.append(fid)
        else:
            logger.warning("gdrive: ignoring invalid folder_id %r", fid)
    return valid


def _build_query(window_start: Optional[str], slice_cfg: dict) -> str:
    """Compose the Drive ``q`` for one slice in sync_to_warehouse.

    Always: ``trashed = false``.
    When ``window_start`` is non-empty, also: ``modifiedTime > '<window_start>'``
    (the incremental upsert path). Replace-mode passes ``None`` to list every
    matching file each run regardless of when it was last modified.

    Optional groups (each only appended if non-empty after validation):
      - folder_ids → ``('id1' in parents OR 'id2' in parents OR ...)``
      - query      → free-form Drive query language clause, AND'd verbatim
                     (use this for ``mimeType = '…'`` clauses too — the
                     dedicated ``mime_types`` field was removed since the
                     same expressiveness already lives in ``query``)

    Caller is responsible for NOT calling this when both ``folder_ids``
    and ``query`` are empty — ``_sync_one_slice`` skips the whole list
    path in that case (see the no-scope gate). If called anyway, the
    return is just ``trashed = false`` (+ optional watermark), which
    would scan the user's entire Drive.
    """
    clauses = ["trashed = false"]
    if window_start:
        clauses.insert(0, f"modifiedTime > '{window_start}'")
    valid_folders = _validated_folder_ids(slice_cfg)
    if valid_folders:
        clauses.append(
            "(" + " or ".join(f"'{fid}' in parents" for fid in valid_folders) + ")"
        )

    extra = slice_cfg.get("query")
    if isinstance(extra, str) and extra.strip():
        clauses.append(f"({extra.strip()})")

    return " and ".join(clauses)


def _extract_file_ids(slice_cfg: dict) -> list[str]:
    """Validate + return the slice's ``file_ids`` list.

    File ids share the same shape as folder ids (URL-safe-base64-ish,
    ≥10 chars per ``_FOLDER_ID_RE``). Invalid ids are logged and skipped
    — same defensive behavior as ``_build_query`` applies to ``folder_ids``,
    so one typo doesn't take the whole slice down.

    Drive's ``q=`` query language has no ``id`` field, so file ids can't
    ride along on the ``files.list`` query. The caller fetches each via
    ``drive_svc.files().get(fileId=...)`` instead.
    """
    raw = slice_cfg.get("file_ids") or []
    if not isinstance(raw, list):
        return []
    valid: list[str] = []
    for fid in raw:
        if isinstance(fid, str) and _FOLDER_ID_RE.match(fid):
            valid.append(fid)
        else:
            logger.warning("gdrive: ignoring invalid file_id %r", fid)
    return valid


def _extract_sheet_tabs(slice_cfg: dict) -> list[dict]:
    """Validate + return the slice's ``sheet_tabs`` list.

    Each entry must be a dict shaped ``{"file_id": str, "sheet_name": str}``
    — file_id matches ``_FOLDER_ID_RE`` (URL-safe-base64-ish, ≥10 chars),
    sheet_name is non-empty after strip. Invalid entries are logged at
    WARNING and skipped (same defensive posture as ``_extract_file_ids``).

    Distinct from ``file_ids``: where ``file_ids`` pulls a whole Sheet
    (every tab as a sidecar via the existing files.export path),
    ``sheet_tabs`` pulls EXACTLY one tab via the Sheets API per entry.
    Lets users surgically slice multi-tab workbooks (e.g. one Ledger
    workbook with 3 tabs → only sync 2 of them).
    """
    raw = slice_cfg.get("sheet_tabs") or []
    if not isinstance(raw, list):
        return []
    valid: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            logger.warning("gdrive: ignoring non-dict sheet_tabs entry %r", entry)
            continue
        fid = entry.get("file_id")
        sheet_name = entry.get("sheet_name")
        if not (isinstance(fid, str) and _FOLDER_ID_RE.match(fid)):
            logger.warning("gdrive: sheet_tabs entry has invalid file_id %r", entry)
            continue
        if not (isinstance(sheet_name, str) and sheet_name.strip()):
            logger.warning("gdrive: sheet_tabs entry has invalid sheet_name %r", entry)
            continue
        valid.append({"file_id": fid, "sheet_name": sheet_name})
    return valid


def _process_one_file(
    *,
    file_meta: dict,
    drive_svc,
    sheets_svc_lazy: Callable[[], Any],
    raw_root: Path,
    slice_name: str,
    slice_cfg: dict,
    dwh_root: Path,
    seen: set[str],
) -> Optional[str]:
    """Build + write the warehouse envelope for one Drive file.

    Used by both fetch paths inside ``_sync_one_slice``: the paginated
    ``files.list`` loop and the per-id ``files.get`` loop. Centralizes
    the body fetch / sidecar writes / atomic envelope write so the two
    paths can't drift.

    Returns the file's ``modifiedTime`` (so the caller can advance
    ``latest_seen`` and bump ``budget.rows_written``), or ``None`` if the
    file was skipped (no ``id``, already in ``seen``). The caller is
    responsible for the watermark / window-end filter — this helper writes
    whatever ``file_meta`` it's handed.

    Mutates ``seen`` to include the written file_id so the sibling
    ``files.get`` loop can dedup against the list-path output.

    When ``slice_cfg.download_files`` is true and the file is a binary blob
    (``content is None and not sidecars``) within ``max_file_size_mb``
    (default 25), downloads the raw bytes to
    ``{dwh_root}/merged/google-drive/<slice>.files/<file_id>-<safe_name>``
    and populates envelope.file_path (warehouse-relative) + downloaded_at.
    Idempotent: skips re-fetch when target exists at the same size.
    """
    file_id = file_meta.get("id")
    if not file_id:
        return None
    if file_id in seen:
        return None
    modified = file_meta.get("modifiedTime", "")
    if not modified:
        return None
    mime = file_meta.get("mimeType", "")
    fname = file_meta.get("name", "")
    body = _fetch_body(drive_svc, file_id, mime)
    path = raw_root / f"{file_id}.json"
    prior = _read_previous_envelope(path) or {}
    prev_sidecars = prior.get("sidecars") if isinstance(prior, dict) else None
    content, sidecars = _write_sidecars(
        raw_root=raw_root,
        file_id=file_id,
        name=fname,
        mime=mime,
        body=body,
        sheets_svc_factory=sheets_svc_lazy,
        previous_sidecars=prev_sidecars or [],
    )
    size_int = int(file_meta["size"]) if file_meta.get("size") else None
    file_path_rel, downloaded_at = _maybe_download_binary(
        drive_svc=drive_svc,
        file_id=file_id,
        fname=fname,
        size_int=size_int,
        content=content,
        sidecars=sidecars,
        slice_cfg=slice_cfg,
        slice_name=slice_name,
        dwh_root=dwh_root,
        prior=prior,
    )
    envelope = {
        "ts": modified,
        "file_id": file_id,
        "name": fname,
        "mime_type": mime,
        "size": size_int,
        "parents": file_meta.get("parents", []),
        "web_view_link": file_meta.get("webViewLink"),
        "owners": [
            o.get("emailAddress")
            for o in (file_meta.get("owners") or [])
        ],
        "content": content,
        "sidecars": sidecars,
        "file_path": file_path_rel,
        "downloaded_at": downloaded_at,
        "raw": file_meta,
        "slice": slice_name,
    }
    atomic_write_json(path, envelope)
    seen.add(file_id)
    return modified


def _maybe_download_binary(
    *,
    drive_svc,
    file_id: str,
    fname: str,
    size_int: Optional[int],
    content: Optional[str],
    sidecars: list[str],
    slice_cfg: dict,
    slice_name: str,
    dwh_root: Path,
    prior: dict,
) -> tuple[Optional[str], Optional[str]]:
    """Persist binary blob bytes when slice opts in via ``download_files``.

    Only fires for binary mimes — i.e. cases where ``_fetch_body`` returned
    None AND ``_write_sidecars`` produced no sheet sidecars. Returns the
    warehouse-relative ``file_path`` + ``downloaded_at`` for the envelope,
    or (None, None) when gated off / oversize / errored.
    """
    if not slice_cfg.get("download_files"):
        return None, None
    if content is not None or sidecars:
        return None, None
    if not size_int or size_int <= 0:
        return None, None
    max_mb = slice_cfg.get("max_file_size_mb")
    if not isinstance(max_mb, (int, float)) or max_mb <= 0:
        max_mb = 25
    max_bytes = int(max_mb * 1024 * 1024)
    if size_int > max_bytes:
        return None, None

    safe = re.sub(r"[^A-Za-z0-9._-]", "_", fname)[:120] or "file"
    files_dir = dwh_root / "merged" / "google-drive" / f"{slice_name}.files"
    target = files_dir / f"{file_id}-{safe}"
    rel_path = str(target.relative_to(dwh_root))

    try:
        if target.exists() and target.stat().st_size == size_int:
            prior_dl = prior.get("downloaded_at") if isinstance(prior, dict) else None
            if isinstance(prior_dl, str) and prior_dl:
                return rel_path, prior_dl
            mtime = target.stat().st_mtime
            return rel_path, datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except OSError as exc:
        logger.warning(
            "gdrive: download_files stat() failed for %s/%s: %s",
            slice_name, file_id, exc,
        )

    try:
        data = drive_svc.files().get_media(fileId=file_id).execute()
    except Exception as exc:  # noqa: BLE001 — per-file resilience
        logger.warning(
            "gdrive: download_files get_media failed for %s/%s: %s",
            slice_name, file_id, exc,
        )
        return None, None
    if not isinstance(data, bytes) or not data:
        return None, None
    try:
        files_dir.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(target)
    except OSError as exc:
        logger.warning(
            "gdrive: download_files write failed for %s/%s: %s",
            slice_name, file_id, exc,
        )
        return None, None
    return rel_path, datetime.now(timezone.utc).isoformat()


def _process_one_tab(
    *,
    file_meta: dict,
    sheet_name: str,
    sheets_svc_lazy: Callable[[], Any],
    raw_root: Path,
    slice_name: str,
    seen: set[str],
) -> tuple[Optional[str], Optional[str]]:
    """Build + write the warehouse envelope for one Google Sheet tab.

    Sibling of ``_process_one_file`` for the per-tab fetch path. Where
    ``_process_one_file`` writes ``<file_id>.json`` and sidecars ALL tabs
    of a Sheet, this writes ``<file_id>__<safe-sheet>.json`` with EXACTLY
    one tab as the sidecar — surgical slicing for multi-tab workbooks.

    Returns ``(modified_ts, skip_reason)``:
      - ``(ts, None)``: sidecar + envelope written. ``ts`` is the parent
        file's ``modifiedTime``. The envelope's ``content`` field carries
        the inlined TSV when ≤ ``MAX_INLINE_BYTES``, else ``None`` — but
        the sidecar TSV is ALWAYS on disk when fetch succeeds (the user
        named exactly this tab; throwing it away would defeat the surgical
        intent of ``sheet_tabs``). On oversize, the envelope's new
        ``content_skipped_reason`` field carries ``"oversize: <N>"`` so
        downstream readers can route through the sidecar.
      - ``(None, reason)``: tab was skipped. ``reason`` is one of
        ``"wrong_mime: <mime>"``, ``"missing_modified_time"``,
        ``"api_error: <…>"``, or ``None`` for already-deduped tabs.

    Mutates ``seen`` to add the per-tab dedup key
    ``f"{file_id}__{safe_sheet}"`` — distinct from the file_ids dedup
    key (``file_id``) so the two paths can coexist in one slice without
    cross-collision.
    """
    file_id = file_meta.get("id")
    if not file_id:
        return None, "missing_file_id"
    mime = file_meta.get("mimeType")
    if mime != "application/vnd.google-apps.spreadsheet":
        logger.warning(
            "gdrive: sheet_tabs entry %r/%r is not a spreadsheet (mime=%r); skipping",
            file_id, sheet_name, mime,
        )
        return None, f"wrong_mime: {mime}"
    safe_sheet = _safe_filename_segment(sheet_name, fallback="sheet")
    tab_id = f"{file_id}__{safe_sheet}"
    if tab_id in seen:
        return None, None
    modified = file_meta.get("modifiedTime", "")
    if not modified:
        return None, "missing_modified_time"
    tsv, fetch_reason = _fetch_tab_tsv(sheets_svc_lazy(), file_id, sheet_name)
    if tsv is None:
        # API error — nothing to persist. The reason carries the cause.
        return None, fetch_reason
    raw_root.mkdir(parents=True, exist_ok=True)
    sidecar_name = f"{safe_sheet}.tsv"
    path = raw_root / f"{tab_id}.json"
    prior = _read_previous_envelope(path) or {}
    prev_sidecars = prior.get("sidecars") if isinstance(prior, dict) else None
    # Evict stale sidecars from a prior name (rename support — same defensive
    # bounds checks as _write_sidecars).
    for prev in prev_sidecars or []:
        if not isinstance(prev, str) or "/" in prev or "\\" in prev or ".." in prev:
            continue
        if prev == sidecar_name:
            continue  # we're about to overwrite it anyway
        old = raw_root / prev
        if old.is_file():
            try:
                old.unlink()
            except OSError as e:
                logger.warning("gdrive: failed to delete stale sidecar %s: %s", old, e)
    # Always write the sidecar — even when oversize, the user asked for THIS
    # tab specifically. Only the inline `content` in the merged JSON is gated
    # by MAX_INLINE_BYTES so the merged file stays manageable.
    (raw_root / sidecar_name).write_text(tsv, encoding="utf-8")
    oversize = fetch_reason and fetch_reason.startswith("oversize:")
    envelope = {
        "ts": modified,
        "file_id": file_id,
        "sheet_name": sheet_name,
        "tab_id": tab_id,
        "name": file_meta.get("name", ""),
        "mime_type": file_meta.get("mimeType", ""),
        "size": int(file_meta["size"]) if file_meta.get("size") else None,
        "parents": file_meta.get("parents", []),
        "web_view_link": file_meta.get("webViewLink"),
        "owners": [
            o.get("emailAddress")
            for o in (file_meta.get("owners") or [])
        ],
        "content": None if oversize else tsv,
        "content_skipped_reason": fetch_reason if oversize else None,
        "sidecars": [sidecar_name],
        "raw": file_meta,
        "slice": slice_name,
    }
    atomic_write_json(path, envelope)
    seen.add(tab_id)
    return modified, None


# Same fields list both fetch paths request, so list-path envelopes and
# get-path envelopes have identical shape.
_FILE_FIELDS = (
    "id, name, mimeType, modifiedTime, size, "
    "webViewLink, parents, owners(emailAddress)"
)


def _sync_one_slice(
    *,
    drive_svc,
    sheets_svc_lazy: Callable[[], Any],
    slice_cfg: dict,
    dwh_dir: str,
    budget,  # SyncBudget; imported lazily inside sync_to_warehouse
    window_end: str,
) -> dict:
    """Sync a single named slice; return its per-slice summary.

    Mirrors ``db_server._sync_one_query``: owns its own ``sync-state.json``
    under ``{dwh_dir}/sources/google-drive/<name>/`` and advances its
    watermark independently of sibling slices. The shared ``budget`` (and
    the Drive + Sheets API clients) are passed in so all slices in one call
    share one auth refresh.
    """
    from clawmeets.mcp.servers._sync_warehouse import (
        _read_state,
        atomic_write_json,
        gc_old_timestamps,
        merge_json_envelopes,
        new_timestamp_dir,
        utcnow_iso,
        validate_merge_policy,
        write_howto,
    )

    raw_name = slice_cfg.get("name") if isinstance(slice_cfg, dict) else None
    if not isinstance(raw_name, str) or not raw_name.strip():
        return {
            "name": "<unnamed>",
            "rows_written": 0,
            "watermarks": None,
            "has_more": False,
            "error": "slice config missing required 'name' field",
        }
    try:
        name = validate_name(raw_name)
    except ValueError as exc:
        return {
            "name": raw_name,
            "rows_written": 0,
            "watermarks": None,
            "has_more": False,
            "error": f"invalid slice name {raw_name!r}: {exc}",
        }

    merge_policy, upsert_id_column, merge_err = validate_merge_policy(slice_cfg)
    if merge_err:
        return {
            "name": name,
            "rows_written": 0,
            "watermarks": None,
            "has_more": False,
            "error": merge_err,
        }

    dwh_root = Path(dwh_dir).expanduser().resolve()
    source = f"google-drive/{name}"
    source_dir = dwh_root / "sources" / source
    state_path = source_dir / "sync-state.json"
    merged_path = dwh_root / "merged" / "google-drive" / f"{name}.json"

    # Mirror howto to both layers before fetch — the howto describes the
    # slice's contract and stays valid even if the fetch errors out below.
    howto_err = write_howto(
        slice_cfg.get("howto"),
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

    # Optional first-time watermark override. `start_at` is consulted only
    # when no successful sync has run yet (state file absent or freshly
    # initialized after corruption). Once a sync has stamped `last_sync_at`,
    # the persisted watermark wins and `start_at` is ignored — re-apply by
    # `rm`-ing the slice's `sync-state.json`. Ignored entirely for replace
    # mode (watermarks aren't authoritative).
    if merge_policy != "replace" and state.get("last_sync_at") is None:
        start_at_raw = slice_cfg.get("start_at")
        if isinstance(start_at_raw, str) and start_at_raw.strip():
            start_at = start_at_raw.strip()
            try:
                # Python's fromisoformat accepts the trailing 'Z' since 3.11.
                # Normalize defensively so older interpreters wouldn't break.
                datetime.fromisoformat(start_at.replace("Z", "+00:00"))
            except ValueError as exc:
                return {
                    "name": name,
                    "rows_written": 0,
                    "watermarks": None,
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
        # Replace mode lists EVERY matching file each run; the merge step
        # rewrites the consolidated JSON array as a whole.
        window_start: Optional[str] = None
    else:
        window_start = max(low, high)
        if window_start >= window_end:
            return {
                "name": name,
                "rows_written": 0,
                "watermarks": {"low": low, "high": high},
                "has_more": False,
                "error": None,
            }

    # Resolve scope-filter shape ONCE so both the gate below and the
    # downstream loops read from one place. Order matches the four
    # scope filters in the slice schema.
    list_folders = _validated_folder_ids(slice_cfg)
    list_query = (slice_cfg.get("query") or "").strip() if isinstance(slice_cfg.get("query"), str) else ""
    has_list_scope = bool(list_folders) or bool(list_query)
    file_ids = _extract_file_ids(slice_cfg)
    sheet_tabs = _extract_sheet_tabs(slice_cfg)

    # No-scope gate: a slice MUST configure at least one scope filter.
    # Without one the list path's q would be just `trashed = false [+ watermark]`
    # — matching every non-trashed file in the entire Drive — and (per the
    # field reports) accumulate thousands of irrelevant sidecars before
    # ever reaching sibling slices. Surface this as a per-slice config
    # error WITHOUT touching state or creating a slice directory.
    if not (has_list_scope or file_ids or sheet_tabs):
        return {
            "name": name,
            "rows_written": 0,
            "watermarks": {"low": low, "high": high},
            "has_more": False,
            "error": (
                "slice has no scope filters configured "
                "(folder_ids / query / file_ids / sheet_tabs)"
            ),
        }

    timestamp_dir = new_timestamp_dir(source_dir)
    raw_root = timestamp_dir  # the per-run dump dir takes raw_root's place

    rows_written_start = budget.rows_written
    latest_seen = window_start or low
    has_more = False

    seen: set[str] = set()
    # Per-entry skip diagnostics for the sheet_tabs path. Surfaced in the
    # slice summary as `info.skipped_tabs` so the LLM caller (and the user
    # reading its reply) can diagnose silent zero-row runs without tailing
    # the runner stderr. Empty list ⇒ no diagnostic entry.
    skipped_tabs: list[dict] = []

    try:
        # List path: only runs when there's a list-path scope (folder_ids
        # OR query). If both are empty, file_ids + sheet_tabs are the
        # only fetch paths for this slice — skipping the list pass
        # entirely avoids an unbounded scan of the user's whole Drive.
        if has_list_scope:
            q = _build_query(window_start, slice_cfg)
            page_token: Optional[str] = None
            while True:
                if budget.should_stop():
                    has_more = True
                    break
                resp = drive_svc.files().list(
                    q=q,
                    pageSize=200,
                    pageToken=page_token,
                    orderBy="modifiedTime",  # ascending
                    fields=f"nextPageToken, files({_FILE_FIELDS})",
                ).execute()
                items = resp.get("files", [])
                for f in items:
                    if budget.should_stop():
                        has_more = True
                        break
                    modified = f.get("modifiedTime", "")
                    # Hold back files whose modifiedTime is at or past sync-start —
                    # likely still being edited. Next run picks them up cleanly.
                    if not modified or modified >= window_end:
                        continue
                    ts = _process_one_file(
                        file_meta=f,
                        drive_svc=drive_svc,
                        sheets_svc_lazy=sheets_svc_lazy,
                        raw_root=raw_root,
                        slice_name=name,
                        slice_cfg=slice_cfg,
                        dwh_root=dwh_root,
                        seen=seen,
                    )
                    if ts is None:
                        continue
                    if ts > latest_seen:
                        latest_seen = ts
                    budget.rows_written += 1
                if has_more:
                    break
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break

        # Per-id fetch path: for each file_id in the slice config, fetch
        # individually via files.get (Drive's q= has no `id` field). One
        # bad id (404 / permission error) is logged + skipped — does NOT
        # fail the whole slice. Dedup against the list-path output via
        # ``seen`` so a file matching both paths is fetched once.
        for file_id in file_ids:
            if budget.should_stop():
                has_more = True
                break
            if file_id in seen:
                continue
            try:
                file_meta = drive_svc.files().get(
                    fileId=file_id, fields=_FILE_FIELDS,
                ).execute()
            except Exception as exc:  # noqa: BLE001 — per-id resilience
                logger.warning("gdrive: file_id %r fetch failed: %s", file_id, exc)
                continue
            modified = file_meta.get("modifiedTime", "")
            if not modified or modified >= window_end:
                continue
            # Upsert mode: skip files at/before high_watermark (already
            # synced). Replace mode: window_start is None — pull every run.
            if window_start is not None and modified <= window_start:
                continue
            ts = _process_one_file(
                file_meta=file_meta,
                drive_svc=drive_svc,
                sheets_svc_lazy=sheets_svc_lazy,
                raw_root=raw_root,
                slice_name=name,
                slice_cfg=slice_cfg,
                dwh_root=dwh_root,
                seen=seen,
            )
            if ts is None:
                continue
            if ts > latest_seen:
                latest_seen = ts
            budget.rows_written += 1

        # Per-tab fetch path: for each {file_id, sheet_name} entry, fetch
        # parent metadata via files.get and the tab body via the Sheets
        # API. Distinct from file_ids: file_ids pulls the WHOLE Sheet
        # (every tab as a sidecar); sheet_tabs surgically pulls one tab.
        # Per-entry resilience: 404 / non-spreadsheet / missing tab is
        # logged + skipped, slice continues. Dedup key is
        # ``f"{file_id}__{safe_sheet}"`` (distinct from file_id) so the
        # two scope filters can coexist in one slice.
        for entry in sheet_tabs:
            if budget.should_stop():
                has_more = True
                break
            file_id = entry["file_id"]
            sheet_name = entry["sheet_name"]
            tab_id = f"{file_id}__{_safe_filename_segment(sheet_name, fallback='sheet')}"
            if tab_id in seen:
                continue
            try:
                file_meta = drive_svc.files().get(
                    fileId=file_id, fields=_FILE_FIELDS,
                ).execute()
            except Exception as exc:  # noqa: BLE001 — per-entry resilience
                logger.warning(
                    "gdrive: tab %r in %r metadata fetch failed: %s",
                    sheet_name, file_id, exc,
                )
                skipped_tabs.append({
                    "file_id": file_id,
                    "sheet_name": sheet_name,
                    "reason": f"metadata_api_error: {type(exc).__name__}: {exc}",
                })
                continue
            modified = file_meta.get("modifiedTime", "")
            if not modified or modified >= window_end:
                # Hold-back: file's modifiedTime is missing OR at-or-after
                # the sync-start instant (still being edited, or clock skew
                # vs Drive). Next run picks it up.
                skipped_tabs.append({
                    "file_id": file_id,
                    "sheet_name": sheet_name,
                    "reason": (
                        "missing_modified_time"
                        if not modified
                        else f"modified_at_or_after_window_end (modified={modified}, window_end={window_end})"
                    ),
                })
                continue
            if window_start is not None and modified <= window_start:
                # Upsert mode incremental skip — already synced. Not a
                # diagnostic concern, don't surface.
                continue
            ts, skip_reason = _process_one_tab(
                file_meta=file_meta,
                sheet_name=sheet_name,
                sheets_svc_lazy=sheets_svc_lazy,
                raw_root=raw_root,
                slice_name=name,
                seen=seen,
            )
            if ts is None:
                if skip_reason is not None:
                    skipped_tabs.append({
                        "file_id": file_id,
                        "sheet_name": sheet_name,
                        "reason": skip_reason,
                    })
                continue
            if ts > latest_seen:
                latest_seen = ts
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
            "has_more": False,
            "error": err,
            "info": {"skipped_tabs": skipped_tabs} if skipped_tabs else None,
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
                "has_more": False,
                "error": merge_err_msg,
                "info": {"skipped_tabs": skipped_tabs} if skipped_tabs else None,
            }
        # Promote sheet_tab sidecars to the merged folder as flat TSVs — the
        # canonical artifact for surgical single-tab pulls. The JSON envelope
        # was just wrapping. See `_promote_tab_sidecars_to_merged` for naming.
        _promote_tab_sidecars_to_merged(
            timestamp_dir=timestamp_dir,
            merged_path=merged_path,
            slice_name=name,
            merge_policy=merge_policy,
        )
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
            "info": {"skipped_tabs": skipped_tabs} if skipped_tabs else None,
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
        "info": {"skipped_tabs": skipped_tabs} if skipped_tabs else None,
    }


def _rmdir_if_empty(path: Path) -> None:
    """Remove ``path`` if it's an empty directory; ignore otherwise.

    Used after a sync run to clean up a freshly-created timestamp folder
    that ended up with nothing written into it (no matching files this
    window, exception before any envelope write, etc.)."""
    try:
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    except OSError:
        pass


def _promote_tab_sidecars_to_merged(
    *,
    timestamp_dir: Path,
    merged_path: Path,
    slice_name: str,
    merge_policy: str,
) -> None:
    """Promote sheet_tab sidecar TSVs from the timestamp dir to the merged
    folder as flat artifacts.

    The merged JSON envelope is overhead for a sheet_tab pull — the TSV IS
    the data. After ``merge_json_envelopes`` succeeds, this helper:

      1. Scans ``timestamp_dir`` for envelopes with ``tab_id`` (sheet_tab
         outputs) vs other envelopes (file_ids / folder_ids outputs).
      2. Copies each tab's sidecar TSV to ``merged/google-drive/<slice>.tsv``
         (single tab in slice) or ``<slice>-<safe-sheet>.tsv`` (multi-tab,
         flat siblings — mirrors ``_build_spreadsheet_sidecars`` naming).
      3. For a pure-sheet_tabs slice (no non-tab envelopes), deletes the
         merged JSON — the TSVs are now the canonical artifact.
      4. In replace mode, wipes any pre-existing ``<slice>.tsv`` /
         ``<slice>-*.tsv`` files first so a tab dropped from config doesn't
         leak. In upsert mode, prior TSVs are preserved (incremental
         semantics).

    Errors on a single tab (missing sidecar, copy failure) are logged and
    skipped — does not fail the slice.
    """
    merged_dir = merged_path.parent
    if not merged_dir.exists():
        merged_dir.mkdir(parents=True, exist_ok=True)

    tab_envelopes: list[dict] = []
    other_envelopes: list[dict] = []
    for child in sorted(timestamp_dir.glob("*.json")):
        if not child.is_file():
            continue
        try:
            env = json.loads(child.read_text())
        except Exception:
            continue
        if not isinstance(env, dict):
            continue
        if env.get("tab_id"):
            tab_envelopes.append(env)
        else:
            other_envelopes.append(env)

    if not tab_envelopes:
        return

    # In replace mode, drop any prior TSVs for this slice so a removed tab
    # doesn't leak. Match both the single-tab name and the multi-tab pattern.
    if merge_policy == "replace":
        for stale in list(merged_dir.glob(f"{slice_name}.tsv")) + list(
            merged_dir.glob(f"{slice_name}-*.tsv")
        ):
            try:
                stale.unlink()
            except OSError as e:
                logger.warning("gdrive: failed to delete stale merged tsv %s: %s", stale, e)

    multi = len(tab_envelopes) > 1
    for env in tab_envelopes:
        sidecars = env.get("sidecars") or []
        if not sidecars:
            continue
        source = timestamp_dir / sidecars[0]
        if not source.is_file():
            logger.warning(
                "gdrive: merged-promote skipped — sidecar %s missing for tab_id %r",
                source, env.get("tab_id"),
            )
            continue
        if multi:
            safe_sheet = _safe_filename_segment(
                env.get("sheet_name") or "sheet", fallback="sheet",
            )
            dest = merged_dir / f"{slice_name}-{safe_sheet}.tsv"
        else:
            dest = merged_dir / f"{slice_name}.tsv"
        try:
            dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError as e:
            logger.warning("gdrive: failed to write merged tsv %s: %s", dest, e)

    # Pure sheet_tabs slice: drop the merged JSON — TSVs are canonical.
    if not other_envelopes and merged_path.exists():
        try:
            merged_path.unlink()
        except OSError as e:
            logger.warning(
                "gdrive: failed to remove merged json %s: %s", merged_path, e,
            )


def _fetch_body(svc, file_id: str, mime_type: str) -> Optional[str]:
    """Return UTF-8 body text for small text-or-Google-native files; else None.

    For Google-native docs (Docs/Sheets/Slides), exports to plain text/CSV.
    For text/* and a few JSON/XML/YAML mimes, downloads the raw bytes. Skips
    binary blobs (PDFs, images, archives) and anything over MAX_INLINE_BYTES.
    """
    try:
        if mime_type in GDOC_EXPORT_MIME:
            export_mime = GDOC_EXPORT_MIME[mime_type]
            data = svc.files().export(
                fileId=file_id, mimeType=export_mime,
            ).execute()
        elif _is_text_mime(mime_type):
            data = svc.files().get_media(fileId=file_id).execute()
        else:
            return None
    except Exception:
        return None  # surface as path-only; sync should not fail per file

    if not data:
        return ""
    if isinstance(data, bytes):
        if len(data) > MAX_INLINE_BYTES:
            return None
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return None
    return str(data)[:MAX_INLINE_BYTES]


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The `mcp` package is required but missing — the clawmeets runner "
            "should bundle it by default. Try: pip install --upgrade clawmeets"
        ) from exc

    mcp = FastMCP("clawmeets-gdrive")

    @mcp.tool()
    def search_files(query: str, max_results: int = 25) -> list[dict]:
        """Search Drive using the standard Drive query syntax.

        See https://developers.google.com/drive/api/guides/search-files for
        the full query language. Common patterns:
          - `name contains 'foo'`
          - `mimeType = 'application/vnd.google-apps.document'`
          - `'<folder_id>' in parents`
          - `modifiedTime > '2025-01-01T00:00:00Z'`

        Returns a list of {id, name, mime_type, modified_time, size,
        web_view_link}.
        """
        svc = _service()
        resp = svc.files().list(
            q=query,
            pageSize=max_results,
            fields=(
                "files(id, name, mimeType, modifiedTime, size, "
                "webViewLink, parents)"
            ),
        ).execute()
        out: list[dict] = []
        for f in resp.get("files", []):
            out.append({
                "id": f.get("id"),
                "name": f.get("name", ""),
                "mime_type": f.get("mimeType", ""),
                "modified_time": f.get("modifiedTime"),
                "size": int(f["size"]) if f.get("size") else None,
                "web_view_link": f.get("webViewLink"),
                "parents": f.get("parents", []),
            })
        return out

    @mcp.tool()
    def get_file_content(file_id: str) -> dict:
        """Fetch the text body of a single Drive file by id.

        For Google Docs / Sheets / Slides, exports as plain text or TSV (first
        tab only — use `sync_to_warehouse` for per-tab sidecars). For text/*
        files, downloads the raw bytes. Binary types and bodies over
        256 KB return ``content: null``.

        Returns ``{id, name, mime_type, modified_time, content}``.
        """
        svc = _service()
        meta = svc.files().get(
            fileId=file_id,
            fields="id, name, mimeType, modifiedTime",
        ).execute()
        mime = meta.get("mimeType", "")
        return {
            "id": meta.get("id"),
            "name": meta.get("name", ""),
            "mime_type": mime,
            "modified_time": meta.get("modifiedTime"),
            "content": _fetch_body(svc, meta["id"], mime),
        }

    @mcp.tool()
    def sync_to_warehouse(
        dwh_dir: str,
        config_file: str = "",
        max_runtime_seconds: int = 1500,
    ) -> dict:
        """Sync new / updated Drive files into the personal data warehouse.

        Call this exactly once when you receive a DM whose body starts with
        ``<!-- clawmeets:gdrive-sync-trigger -->``. Read ``dwh_dir`` from your
        prompt's ``== DATA WAREHOUSE ==`` block and ``config_file`` from the
        ``== MCP CONFIG FILES ==`` block (the path next to ``google-drive``).

        Named-slice model: the config carries a ``slices`` list, each entry a
        ``{name, folder_ids?, query?, file_ids?, sheet_tabs?, merge_policy?, merge_policy_upsert_id_column?}``
        dict. Each slice gets its own output directory and watermark at
        ``{dwh_dir}/sources/google-drive/<name>/``; per-run envelopes land in
        ``<TIMESTAMP>/<file_id>.json`` siblings of ``sync-state.json``, and
        the consolidated dataset rebuilds per the slice's ``merge_policy``
        (default ``replace``; ``upsert`` requires
        ``merge_policy_upsert_id_column`` — typically ``"file_id"``).

        Merged artifact shape:
          - ``sheet_tabs`` outputs are promoted as flat TSVs at
            ``{dwh_dir}/merged/google-drive/<name>.tsv`` (single tab in
            slice) or ``<name>-<safe-sheet>.tsv`` siblings (multi-tab). For
            pure-sheet_tabs slices, the JSON envelope wrapper is dropped —
            the TSV IS the data, no need to parse JSON to unwrap it.
          - ``file_ids`` / ``folder_ids`` outputs rebuild as
            ``<name>.json`` (a JSON array of envelopes sorted by ``ts``).
          - Mixed slices: both shapes coexist (TSVs for the tab envelopes,
            JSON for the file/folder envelopes).

        Up to ``KEEP_RECENT_DUMPS`` timestamp folders are retained per slice
        for audit/recovery. Slices advance independently — a failure on one
        does not roll back another's watermark.

        Watermark semantics: in ``upsert`` mode, the per-slice filter is
        ``modifiedTime > <window_start>`` — i.e. files whose ``modifiedTime``
        falls in ``(high_watermark, now]``. This catches both new files AND
        edits to existing files; same ``file_id`` overwrites in the merged
        JSON. In ``replace`` mode the watermark is ignored entirely — every
        matching file is listed each run and the merged JSON is rewritten
        from the run's snapshot.

        Scope filter: ``trashed = false`` plus optional ``folder_ids``
        (direct-parents only — Drive's ``in parents`` does not recurse) and
        a free-form ``query`` clause, plus ``modifiedTime > <watermark>`` in
        upsert mode. Filter changes are forward-looking under upsert —
        widening a slice's scope does NOT backfill files modified before
        that slice's previous ``high_watermark``. To re-scan under a wider
        filter, delete the slice's ``sync-state.json``. (Replace mode
        backfills naturally; every run is a full scan.)

        Per-id scope (``file_ids``): an OR-additive third filter, fetched
        per id via ``files.get`` (Drive's ``q=`` language has no ``id``
        field). Useful for hand-picked files anywhere in Drive — including
        files shared with you that aren't in any of your folders, and files
        that don't share a common parent. Each id gets the same watermark
        filter as the list path; one bad id (404 / permission error) is
        logged and skipped without failing the slice. Files matching both
        the list path (folder_ids / query) and ``file_ids`` are fetched
        once. Same id format as ``folder_ids`` (URL-safe-base64-ish, ≥10
        chars); invalid ids are logged and skipped.

        Per-tab scope (``sheet_tabs``): an OR-additive fourth filter for
        Google Sheets, fetched per ``{file_id, sheet_name}`` entry via
        the Sheets API. Distinct from ``file_ids``, which pulls the WHOLE
        Sheet (every tab as a sidecar via Drive's export); ``sheet_tabs``
        surgically pulls EXACTLY the named tab. Useful for slicing one tab
        out of a multi-tab workbook (e.g. a Ledger workbook with `Stock
        Movements` + `System Orders` + `Manual Orders` where you only
        want two of the three). Each entry's parent file is fetched via
        ``files.get`` for metadata + watermark, then the tab's TSV via
        ``spreadsheets.values.get``. Per-entry resilience: bad file_id
        / non-spreadsheet mime / missing tab is logged and skipped without
        failing the slice. Per-tab envelope adds two fields beyond the
        full-file shape: ``sheet_name`` (the tab title) and ``tab_id``
        (``f"{file_id}__{safe_sheet_name}"`` — also the on-disk envelope
        filename, so two tabs from the same file in one slice don't
        collide on ``<file_id>.json``). Watermark uses the parent file's
        ``modifiedTime`` (Sheets API has no per-tab modtime — editing tab
        A causes redundant re-sync of tab B in the same workbook).
        Natural upsert dedup column for per-tab slices is ``"tab_id"``
        (set ``merge_policy_upsert_id_column`` accordingly).

        Empty/missing config or empty ``slices`` list ⇒ ``status: "noop"`` (no
        directories created). This makes a fresh install harmless on the
        first scheduled trigger before the user has configured any slices.

        Each row is ``{ts, file_id, name, mime_type, size, parents,
        web_view_link, content, sidecars, raw, slice}`` — ``content`` is the
        inlined text body for small text/Google-native files (cap 256 KB; for
        spreadsheets, the first tab's TSV), else ``null``. ``sidecars`` lists
        on-disk siblings written alongside the envelope: Google Sheets get one
        TSV per tab (``{name}.tsv`` for single-tab, ``{name}-{tab}.tsv`` per
        tab when N > 1); Docs and Slides get ``{name}.txt``. Sidecar filenames
        sanitize ``/\\:*?<>|"`` and control characters out of the source name.
        Renames clean up old sidecars on the next sync; same-name files in
        Drive will collide on disk and is a known limitation. The agent /
        downstream skill calls ``get_file_content`` on demand for any body the
        warehouse opted not to inline.

        Args:
            dwh_dir: Personal data warehouse root.
            config_file: Path to this agent's per-MCP config file (see the
                prompt's ``== MCP CONFIG FILES ==`` block). Empty / missing
                file ⇒ noop.
            max_runtime_seconds: Wall-clock budget shared across all
                slices. Default 1500 (25 min). When this elapses mid-run,
                later slices are pre-start-skipped with `has_more=true`.

        Returns the standard sync envelope plus a ``per_slice`` map:
        ``{status, source, rows_written, window, watermarks, has_more,
        error, per_slice}``. ``status`` is ``error`` if the config is
        malformed; ``partial`` if any slice hit ``has_more=true``; ``noop``
        if no rows written and no errors; ``ok`` otherwise. If
        ``has_more=true``, the next scheduled trigger resumes — do NOT loop.
        """
        from clawmeets.mcp.servers._sync_warehouse import SyncBudget, utcnow_iso

        cfg, err = _load_config(config_file)
        window_end = utcnow_iso()
        if err:
            return {
                "status": "error",
                "source": "google-drive",
                "rows_written": 0,
                "window": [window_end, window_end],
                "watermarks": None,
                "has_more": False,
                "error": err,
                "per_slice": {},
                "info": None,
            }

        slices = (cfg or {}).get("slices") if isinstance(cfg, dict) else None
        if not isinstance(slices, list) or len(slices) == 0:
            return {
                "status": "noop",
                "source": "google-drive",
                "rows_written": 0,
                "window": [window_end, window_end],
                "watermarks": None,
                "has_more": False,
                "error": None,
                "per_slice": {},
                "info": None,
            }

        drive_svc = _service()
        sheets_svc_cache: list = []  # lazy single-shot, shared across slices

        def _sheets_lazy():
            if not sheets_svc_cache:
                sheets_svc_cache.append(_sheets_service())
            return sheets_svc_cache[0]

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
                # Wall-clock budget elapsed before this slice started — not
                # an error, just unfinished. has_more=True signals "resume
                # next trigger"; error stays None.
                any_has_more = True
                per_slice[display_name] = {
                    "name": display_name,
                    "rows_written": 0,
                    "watermarks": None,
                    "has_more": True,
                    "error": None,
                }
                continue
            summary = _sync_one_slice(
                drive_svc=drive_svc,
                sheets_svc_lazy=_sheets_lazy,
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

        # Roll up per-slice `info.skipped_tabs` into a one-line top-level
        # summary so the LLM caller's reply shows it without parsing the
        # per_slice tree. Empty/missing collapses to no info field.
        total_skipped = 0
        slices_with_skips = 0
        for s in per_slice.values():
            tabs = ((s.get("info") or {}).get("skipped_tabs") or [])
            if tabs:
                total_skipped += len(tabs)
                slices_with_skips += 1
        info: Optional[dict] = None
        if total_skipped:
            info = {
                "skipped_tabs_total": total_skipped,
                "slices_with_skips": slices_with_skips,
                "message": (
                    f"{total_skipped} sheet_tabs entries skipped across "
                    f"{slices_with_skips} slice(s) — see per_slice[*].info.skipped_tabs[*].reason"
                ),
            }

        return {
            "status": status,
            "source": "google-drive",
            "rows_written": budget.rows_written,
            "window": [agg_low or window_end, window_end],
            "watermarks": (
                {"low": agg_low, "high": agg_high}
                if (agg_low or agg_high) else None
            ),
            "has_more": any_has_more,
            "error": first_error,
            "per_slice": per_slice,
            "info": info,
        }

    mcp.run()


if __name__ == "__main__":
    main()
