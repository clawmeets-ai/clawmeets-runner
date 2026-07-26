# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/gdrive/_lib.py

Google Drive (read-only). Drives ``clawmeets gdrive <subcmd>``; paired skill:
``google-drive``.

Named-slice sync supports four scope filters: ``folder_ids`` /
``query`` (list-path) + ``file_ids`` / ``sheet_tabs`` (per-id paths).
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from clawmeets.integrations._config_resolve import resolve_skill_config_path
from clawmeets.integrations._sync_warehouse import (
    SyncBudget,
    diff_snapshot,
    load_snapshot,
    prune_deltas,
    read_sync_state,
    run_slice_sync,
    run_slices,
    utcnow_iso,
    write_delta,
    write_howto,
    write_snapshot,
    write_sync_state,
)
from clawmeets.utils.file_io import FileUtil
from clawmeets.utils.jsonc import parse_jsonc

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

_FOLDER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,}$")

GDOC_EXPORT_MIME = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/tab-separated-values",
    "application/vnd.google-apps.presentation": "text/plain",
}

SIDECAR_EXT = {
    "application/vnd.google-apps.document": ".txt",
    "application/vnd.google-apps.spreadsheet": ".tsv",
    "application/vnd.google-apps.presentation": ".txt",
}

TEXT_MIME_PREFIXES = ("text/",)
TEXT_MIME_EXACT = {"application/json", "application/xml", "application/x-yaml"}

MAX_INLINE_BYTES = 256 * 1024


def build_service(token_path: Path):
    from googleapiclient.discovery import build
    from clawmeets.integrations.auth.google_oauth import load_credentials

    creds = load_credentials(token_path, SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def build_sheets_service(token_path: Path):
    """Sheets API client on the same drive.readonly token."""
    from googleapiclient.discovery import build
    from clawmeets.integrations.auth.google_oauth import load_credentials

    creds = load_credentials(token_path, SCOPES)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


_FILENAME_BAD_CHARS = re.compile(r"[\x00-\x1f\x7f/\\:*?<>|\"]+")
_FILENAME_WS = re.compile(r"\s+")


def _safe_filename_segment(s: str, *, fallback: str = "file", max_len: int = 80) -> str:
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


def _fetch_tab_tsv(sheets_svc, file_id: str, tab_title: str) -> tuple[Optional[str], Optional[str]]:
    try:
        resp = sheets_svc.spreadsheets().values().get(
            spreadsheetId=file_id, range=tab_title,
            valueRenderOption="FORMATTED_VALUE",
        ).execute()
    except Exception as e:
        logger.warning("gdrive: failed to fetch tab %r in %s: %s", tab_title, file_id, e)
        return None, f"api_error: {type(e).__name__}: {e}"
    rows = resp.get("values", []) or []
    tsv = _rows_to_tsv(rows)
    size = len(tsv.encode("utf-8"))
    if size > MAX_INLINE_BYTES:
        return tsv, f"oversize: {size}"
    return tsv, None


def _is_text_mime(mime: str) -> bool:
    return mime.startswith(TEXT_MIME_PREFIXES) or mime in TEXT_MIME_EXACT


def load_config(config_file: str) -> tuple[Optional[dict], Optional[str]]:
    config_file = resolve_skill_config_path("google-drive", config_file)
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


def _build_query(window_start: Optional[str], window_end: Optional[str], slice_cfg: dict) -> str:
    clauses = ["trashed = false"]
    if window_end:
        clauses.insert(0, f"modifiedTime <= '{window_end}'")
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


_FILE_FIELDS = (
    "id, name, mimeType, modifiedTime, size, "
    "webViewLink, parents, owners(emailAddress)"
)


def _build_file_envelope(*, file_meta, drive_svc, slice_cfg, files_root,
                         dwh_root, prior_by_id) -> Optional[dict]:
    """Build one file-scope envelope (inline content + optional binary download)."""
    file_id = file_meta.get("id")
    modified = file_meta.get("modifiedTime", "")
    if not file_id or not modified:
        return None
    mime = file_meta.get("mimeType", "")
    fname = file_meta.get("name", "")
    content = _fetch_body(drive_svc, file_id, mime)
    size_int = int(file_meta["size"]) if file_meta.get("size") else None
    file_path_rel, downloaded_at = _maybe_download_binary(
        drive_svc=drive_svc, file_id=file_id, fname=fname, size_int=size_int,
        content=content, slice_cfg=slice_cfg, files_root=files_root,
        dwh_root=dwh_root, prior=prior_by_id.get(file_id) or {},
    )
    return {
        "ts": modified,
        "file_id": file_id, "name": fname,
        "mime_type": mime, "size": size_int,
        "parents": file_meta.get("parents", []),
        "web_view_link": file_meta.get("webViewLink"),
        "owners": [o.get("emailAddress") for o in (file_meta.get("owners") or [])],
        "content": content,
        "file_path": file_path_rel,
        "downloaded_at": downloaded_at,
        "raw": file_meta,
        "slice": file_meta.get("_slice"),
    }


def _maybe_download_binary(*, drive_svc, file_id, fname, size_int, content,
                           slice_cfg, files_root, dwh_root, prior):
    if not slice_cfg.get("download_files"):
        return None, None
    if content is not None:
        return None, None
    if not size_int or size_int <= 0:
        return None, None
    max_mb = slice_cfg.get("max_file_size_mb")
    if not isinstance(max_mb, (int, float)) or max_mb <= 0:
        max_mb = 25
    if size_int > int(max_mb * 1024 * 1024):
        return None, None

    safe = re.sub(r"[^A-Za-z0-9._-]", "_", fname)[:120] or "file"
    target = files_root / f"{file_id}-{safe}"
    rel_path = str(target.relative_to(dwh_root))

    try:
        if target.exists() and target.stat().st_size == size_int:
            prior_dl = prior.get("downloaded_at") if isinstance(prior, dict) else None
            if isinstance(prior_dl, str) and prior_dl:
                return rel_path, prior_dl
            mtime = target.stat().st_mtime
            return rel_path, datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except OSError as exc:
        logger.warning("gdrive: download_files stat() failed for %s: %s", file_id, exc)

    try:
        data = drive_svc.files().get_media(fileId=file_id).execute()
    except Exception as exc:
        logger.warning("gdrive: download_files get_media failed for %s: %s", file_id, exc)
        return None, None
    if not isinstance(data, bytes) or not data:
        return None, None
    try:
        files_root.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(target)
    except OSError as exc:
        logger.warning("gdrive: download_files write failed for %s: %s", file_id, exc)
        return None, None
    return rel_path, datetime.now(timezone.utc).isoformat()


def _fetch_body(svc, file_id: str, mime_type: str) -> Optional[str]:
    try:
        if mime_type in GDOC_EXPORT_MIME:
            export_mime = GDOC_EXPORT_MIME[mime_type]
            data = svc.files().export(fileId=file_id, mimeType=export_mime).execute()
        elif _is_text_mime(mime_type):
            data = svc.files().get_media(fileId=file_id).execute()
        else:
            return None
    except Exception:
        return None

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


def _file_scope_source(*, drive_svc, slice_cfg, name, dwh_dir, budget,
                       file_ids, has_list_scope) -> dict:
    """File-scope sub-source (folder_ids / query / file_ids) → file envelopes."""
    dwh_root = Path(dwh_dir).expanduser().resolve()
    source = f"google-drive/{name}"
    base = dwh_root / "raw" / source
    files_root = base / "files"

    howto_err = write_howto(slice_cfg.get("howto"), snapshot_dir=base)
    if howto_err:
        return {"name": name, "rows_written": 0, "watermarks": None,
                "has_more": False, "error": howto_err}

    prior_by_id = {str(r.get("file_id")): r for r in load_snapshot(base, "ndjson")
                   if r.get("file_id")}

    def fetch(window_start: str, window_end: str, bud, emit) -> bool:
        if has_list_scope:
            q = _build_query(window_start, window_end, slice_cfg)
            page_token = None
            while True:
                if bud.should_stop():
                    return True
                resp = drive_svc.files().list(
                    q=q, pageSize=200, pageToken=page_token, orderBy="modifiedTime",
                    fields=f"nextPageToken, files({_FILE_FIELDS})",
                ).execute()
                for f in resp.get("files", []):
                    if bud.should_stop():
                        return True
                    f["_slice"] = name
                    env = _build_file_envelope(
                        file_meta=f, drive_svc=drive_svc, slice_cfg=slice_cfg,
                        files_root=files_root, dwh_root=dwh_root, prior_by_id=prior_by_id,
                    )
                    if env is None:
                        continue
                    bud.rows_written += 1
                    emit(env)
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
        for file_id in file_ids:
            if bud.should_stop():
                return True
            try:
                file_meta = drive_svc.files().get(fileId=file_id, fields=_FILE_FIELDS).execute()
            except Exception as exc:
                logger.warning("gdrive: file_id %r fetch failed: %s", file_id, exc)
                continue
            modified = file_meta.get("modifiedTime", "")
            if not modified or modified <= window_start or modified > window_end:
                continue
            file_meta["_slice"] = name
            env = _build_file_envelope(
                file_meta=file_meta, drive_svc=drive_svc, slice_cfg=slice_cfg,
                files_root=files_root, dwh_root=dwh_root, prior_by_id=prior_by_id,
            )
            if env is None:
                continue
            bud.rows_written += 1
            emit(env)
        return False

    return run_slice_sync(
        source=source, dwh_dir=dwh_dir, budget=budget, fetch=fetch,
        id_field="file_id", ts_field="ts", start_at=slice_cfg.get("start_at"),
        snapshot_fmt="ndjson",
    )


def _sync_tab_source(*, drive_svc, sheets_svc_lazy, source, dwh_dir, budget,
                     file_id, sheet_name, howto) -> dict:
    """One sheet tab → a tabular source. Gated on the spreadsheet's modifiedTime;
    on change, re-read the tab and diff rows by stable row index (`_row`)."""
    name = source.rsplit("/", 1)[-1]

    def _err(msg):
        return {"name": name, "rows_written": 0, "watermarks": None,
                "has_more": False, "error": msg}

    dwh_root = Path(dwh_dir).expanduser().resolve()
    base = dwh_root / "raw" / source
    write_howto(howto, snapshot_dir=base)
    state_path = base / "sync-state.json"
    state = read_sync_state(state_path, source)
    now = utcnow_iso()

    try:
        file_meta = drive_svc.files().get(fileId=file_id, fields=_FILE_FIELDS).execute()
    except Exception as exc:
        return _err(f"metadata_api_error: {type(exc).__name__}: {exc}")
    if file_meta.get("mimeType") != "application/vnd.google-apps.spreadsheet":
        return _err(f"not_a_spreadsheet: mime={file_meta.get('mimeType')!r}")
    modified = file_meta.get("modifiedTime", "")
    if not modified:
        return _err("missing_modified_time")

    cursor = state.get("cursor")
    if cursor and modified <= cursor:
        return {"name": name, "rows_written": 0,
                "watermarks": {"cursor": cursor, "floor": state.get("floor")},
                "has_more": False, "error": None}

    tsv, reason = _fetch_tab_tsv(sheets_svc_lazy(), file_id, sheet_name)
    if tsv is None:
        return _err(f"fetch_failed: {reason}")

    parsed = list(csv.reader(io.StringIO(tsv), delimiter="\t"))
    header = parsed[0] if parsed else []
    rows: list[dict] = []
    for i, dr in enumerate(parsed[1:]):
        width = max(len(header), len(dr))
        row = {
            (header[j] if j < len(header) and header[j] else f"col{j}"):
                (dr[j] if j < len(dr) else "")
            for j in range(width)
        }
        row["_row"] = str(i)
        row["ts"] = modified
        rows.append(row)

    prior = load_snapshot(base, "tsv")
    prior_by_id = {str(r.get("_row")): r for r in prior if r.get("_row") not in (None, "")}
    changed, tombstones, snapshot = diff_snapshot(
        prior_by_id, rows, id_field="_row", ts_field="ts", volatile_fields={"ts"})

    write_delta(base / "deltas", changed + tombstones)
    write_snapshot(base, snapshot, "tsv", id_field="_row", ts_field="ts")
    state["source"] = source
    state["cursor"] = modified
    state["floor"] = state.get("floor") or modified
    state["last_run_at"] = now
    state["last_run_count"] = len(changed)
    state["last_error"] = None
    write_sync_state(state_path, state)
    prune_deltas(base / "deltas")
    budget.rows_written += len(changed)
    return {"name": name, "rows_written": len(changed),
            "watermarks": {"cursor": modified, "floor": state["floor"]},
            "has_more": False, "error": None}


def _sync_one_slice(*, drive_svc, sheets_svc_lazy, slice_cfg, dwh_dir, budget) -> dict:
    raw_name = slice_cfg.get("name") if isinstance(slice_cfg, dict) else None
    if not isinstance(raw_name, str) or not raw_name.strip():
        return {"name": "<unnamed>", "rows_written": 0, "watermarks": None,
                "has_more": False, "error": "slice config missing required 'name' field"}
    try:
        name = FileUtil.validate_fs_name(raw_name)
    except ValueError as exc:
        return {"name": raw_name, "rows_written": 0, "watermarks": None,
                "has_more": False, "error": f"invalid slice name {raw_name!r}: {exc}"}

    query = slice_cfg.get("query")
    has_list_scope = bool(_validated_folder_ids(slice_cfg)) or bool(
        isinstance(query, str) and query.strip())
    file_ids = _extract_file_ids(slice_cfg)
    sheet_tabs = _extract_sheet_tabs(slice_cfg)

    if not (has_list_scope or file_ids or sheet_tabs):
        return {"name": name, "rows_written": 0, "watermarks": None,
                "has_more": False,
                "error": "slice has no scope filters configured "
                         "(folder_ids / query / file_ids / sheet_tabs)"}

    results: list[dict] = []
    has_file_scope = has_list_scope or bool(file_ids)
    if has_file_scope:
        results.append(_file_scope_source(
            drive_svc=drive_svc, slice_cfg=slice_cfg, name=name, dwh_dir=dwh_dir,
            budget=budget, file_ids=file_ids, has_list_scope=has_list_scope))

    # Multiple tab outputs (or file-scope + tabs) get per-tab source suffixes;
    # a lone tab keeps the bare slice name (matches the single-table convention).
    multi_out = len(sheet_tabs) > 1 or (sheet_tabs and has_file_scope)
    for entry in sheet_tabs:
        if budget.should_stop():
            results.append({"name": name, "rows_written": 0, "watermarks": None,
                            "has_more": True, "error": None})
            break
        suffix = _safe_filename_segment(entry["sheet_name"], fallback="sheet")
        sub = f"{name}-{suffix}" if multi_out else name
        results.append(_sync_tab_source(
            drive_svc=drive_svc, sheets_svc_lazy=sheets_svc_lazy,
            source=f"google-drive/{sub}", dwh_dir=dwh_dir, budget=budget,
            file_id=entry["file_id"], sheet_name=entry["sheet_name"],
            howto=slice_cfg.get("howto")))

    # Roll the sub-sources up into one per-slice summary for run_slices.
    rows = sum(r.get("rows_written", 0) for r in results)
    err = next((r["error"] for r in results if r.get("error")), None)
    has_more = any(r.get("has_more") for r in results)
    floor = cursor = None
    for r in results:
        wm = r.get("watermarks") or {}
        if wm.get("floor"):
            floor = wm["floor"] if floor is None else min(floor, wm["floor"])
        if wm.get("cursor"):
            cursor = wm["cursor"] if cursor is None else max(cursor, wm["cursor"])
    return {"name": name, "rows_written": rows,
            "watermarks": {"floor": floor, "cursor": cursor} if (floor or cursor) else None,
            "has_more": has_more, "error": err}


# ---------------------------------------------------------------------------
# Public interactive tools
# ---------------------------------------------------------------------------


def search_files(svc, query: str, max_results: int = 25) -> list[dict]:
    """Search Drive with the standard Drive query syntax."""
    resp = svc.files().list(
        q=query, pageSize=max_results,
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


def get_file_content(svc, file_id: str) -> dict:
    """Fetch the text body of a single Drive file by id."""
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


def sync_to_warehouse(*, drive_svc, sheets_svc_factory, dwh_dir,
                      config_file="", max_runtime_seconds=1500) -> dict:
    """Sync Drive files into the warehouse per --config.

    Triggered by ``<!-- clawmeets:gdrive-sync-trigger -->``.
    """
    cfg, err = load_config(config_file)
    window_end = utcnow_iso()
    if err:
        return {"status": "error", "source": "google-drive", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False, "error": err, "per_slice": {}, "info": None}

    slices = (cfg or {}).get("slices") if isinstance(cfg, dict) else None
    if not isinstance(slices, list) or len(slices) == 0:
        return {"status": "noop", "source": "google-drive", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False, "error": None, "per_slice": {}, "info": None}

    sheets_svc_cache: list = []

    def _sheets_lazy():
        if not sheets_svc_cache:
            sheets_svc_cache.append(sheets_svc_factory())
        return sheets_svc_cache[0]

    budget = SyncBudget(max_runtime_seconds)
    return run_slices(
        source_family="google-drive", slices=slices, budget=budget,
        dwh_dir=dwh_dir,
        run_one=lambda sc: _sync_one_slice(
            drive_svc=drive_svc, sheets_svc_lazy=_sheets_lazy,
            slice_cfg=sc, dwh_dir=dwh_dir, budget=budget),
    )
