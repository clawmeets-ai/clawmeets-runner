# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/osxphotos/_lib.py

Read-only macOS Photos library access (Photos.app + iCloud Photo Library) via
the ``osxphotos`` package, plus single-dataset sync into the warehouse and
JPEG transcoding for vision-friendly Read consumption.
"""
from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from clawmeets.integrations._config_resolve import resolve_skill_config_path
from clawmeets.integrations._sync_warehouse import (
    EPOCH_ISO,
    SyncBudget,
    normalize_iso,
    run_slices,
    run_slice_sync,
    utcnow_iso,
    write_howto,
)
from clawmeets.utils.jsonc import parse_jsonc


def check_platform() -> None:
    if platform.system() != "Darwin":
        raise RuntimeError(
            "osxphotos requires macOS — the Photos library is macOS-only. "
            f"Current platform: {platform.system()}."
        )


def _import_osxphotos():
    try:
        import osxphotos  # type: ignore
        return osxphotos
    except ImportError as exc:
        raise RuntimeError(
            "The `osxphotos` package is required but missing. Install it on "
            "the runner: pip install osxphotos"
        ) from exc


def load_config(config_file: str) -> tuple[Optional[dict], Optional[str]]:
    config_file = resolve_skill_config_path("osxphotos", config_file)
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
    return cfg, None


def _transcode_to_jpeg(src: str, dst: Path, max_dim: int, quality: int) -> tuple[bool, Optional[str]]:
    proc = subprocess.run(
        ["sips", "-s", "format", "jpeg",
         "-Z", str(max_dim),
         "-s", "formatOptions", str(quality),
         src, "--out", str(dst)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not dst.exists():
        msg = proc.stderr.strip() or proc.stdout.strip() or "no output"
        return False, f"sips failed (rc={proc.returncode}): {msg}"
    return True, None


def _populate_scaled_path(row: dict, *, scaled_dir: Path, dwh_root: Path,
                          max_dim: int, quality: int) -> None:
    src = row.get("path")
    if not src:
        row["scaled_path"] = None
        row["scaled_at"] = None
        row["scaled_error"] = "iCloud-only (path null)"
        return
    out_path = scaled_dir / f"{row['uuid']}.jpg"
    try:
        src_mtime = Path(src).stat().st_mtime
    except OSError as exc:
        row["scaled_path"] = None
        row["scaled_at"] = None
        row["scaled_error"] = f"stat src failed: {type(exc).__name__}: {exc}"
        return
    if not (out_path.exists() and out_path.stat().st_mtime >= src_mtime):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok, err = _transcode_to_jpeg(src, out_path, max_dim, quality)
        if not ok:
            row["scaled_path"] = None
            row["scaled_at"] = None
            row["scaled_error"] = err
            return
    row["scaled_path"] = str(out_path.relative_to(dwh_root))
    row["scaled_at"] = utcnow_iso()


def _photo_to_dict(p) -> dict:
    loc = None
    if p.location and p.location[0] is not None and p.location[1] is not None:
        loc = {"lat": p.location[0], "lon": p.location[1]}
    return {
        "uuid": p.uuid,
        "filename": p.original_filename,
        "date": p.date.isoformat() if p.date else None,
        "date_added": p.date_added.isoformat() if getattr(p, "date_added", None) else None,
        "location": loc,
        "persons": list(p.persons) if p.persons else [],
        "favorite": bool(p.favorite),
        "hidden": bool(p.hidden),
        "path": p.path,
    }


def list_albums() -> list[dict]:
    """List every album with photo count + date range."""
    check_platform()
    osxphotos = _import_osxphotos()
    db = osxphotos.PhotosDB()
    out = []
    for ai in db.album_info:
        photos = ai.photos
        if not photos:
            out.append({"name": ai.title, "photo_count": 0,
                        "date_min": None, "date_max": None})
            continue
        dates = [p.date for p in photos if p.date]
        out.append({
            "name": ai.title,
            "photo_count": len(photos),
            "date_min": min(dates).isoformat() if dates else None,
            "date_max": max(dates).isoformat() if dates else None,
        })
    return out


def list_photos(
    album: Optional[str] = None,
    year: Optional[int] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """List photos (metadata + paths, no bytes)."""
    check_platform()
    osxphotos = _import_osxphotos()
    db = osxphotos.PhotosDB()
    if album:
        photos = db.photos(albums=[album])
    else:
        photos = db.photos()
    if year is not None:
        photos = [p for p in photos if p.date and p.date.year == year]
    photos = sorted(photos, key=lambda p: p.date or p.date_added)
    if limit is not None and limit > 0:
        photos = photos[:limit]
    return [_photo_to_dict(p) for p in photos]


def export_photo(uuid: str, dest_dir: str) -> dict:
    """Force-download an iCloud-optimized photo."""
    check_platform()
    osxphotos = _import_osxphotos()
    dest = Path(dest_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    db = osxphotos.PhotosDB()
    results = db.photos(uuid=[uuid])
    if not results:
        return {"ok": False, "exported_path": None,
                "error": f"No photo with uuid {uuid!r}"}
    photo = results[0]
    try:
        paths = photo.export(str(dest), use_photos_export=True)
    except Exception as exc:
        return {"ok": False, "exported_path": None, "error": str(exc)}
    if not paths:
        return {"ok": False, "exported_path": None,
                "error": "Export returned no path (photo may be missing)."}
    return {"ok": True, "exported_path": paths[0], "error": None}


def export_photo_as_jpeg(uuid: str, dest_dir: str, max_dim: int = 1200, quality: int = 65) -> dict:
    """Transcode to a JPEG sized to fit Claude Code's 256 KB Read cap."""
    check_platform()
    osxphotos = _import_osxphotos()
    results = osxphotos.PhotosDB().photos(uuid=[uuid])
    if not results:
        return {"ok": False, "exported_path": None, "byte_size": None,
                "error": f"No photo with uuid {uuid!r}"}
    photo = results[0]
    src = photo.path
    if not src:
        return {"ok": False, "exported_path": None, "byte_size": None,
                "error": "Photo not cached locally (iCloud-only). Export first."}
    dest = Path(dest_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / f"{uuid}.jpg"
    ok, err = _transcode_to_jpeg(src, out_path, max_dim, quality)
    if not ok:
        return {"ok": False, "exported_path": None, "byte_size": None, "error": err}
    return {"ok": True, "exported_path": str(out_path),
            "byte_size": out_path.stat().st_size, "error": None}


def _sync_photos(osxphotos, cfg: dict, dwh_dir: str, budget: SyncBudget) -> dict:
    dwh_root = Path(dwh_dir).expanduser().resolve()
    source = "osxphotos"
    base = dwh_root / "raw" / source
    files_root = base / "files"   # scaled-jpeg sidecars

    howto_err = write_howto(cfg.get("howto"), snapshot_dir=base)
    if howto_err:
        return {"name": source, "rows_written": 0, "watermarks": None,
                "has_more": False, "error": howto_err}

    start_at = cfg.get("start_at")
    try:
        floor = normalize_iso(start_at) or EPOCH_ISO
    except ValueError as exc:
        return {"name": source, "rows_written": 0, "watermarks": None,
                "has_more": False, "error": f"invalid start_at {start_at!r}: {exc}"}

    scale_to_jpeg = bool(cfg.get("scale_to_jpeg"))
    scale_max_dim = int(cfg.get("scale_max_dim") or 1200)
    scale_quality = int(cfg.get("scale_quality") or 65)

    def fetch(window_start: str, _window_end: str, bud: SyncBudget, emit) -> bool:
        # Full library re-scan, filtered client-side to date_added >= floor.
        db = osxphotos.PhotosDB()
        for p in db.photos():
            if bud.should_stop():
                return True
            added = getattr(p, "date_added", None)
            if not added:
                continue
            added_iso = added.isoformat()
            if added_iso < window_start:
                continue
            row = _photo_to_dict(p)
            row["ts"] = added_iso
            if scale_to_jpeg:
                _populate_scaled_path(
                    row, scaled_dir=files_root, dwh_root=dwh_root,
                    max_dim=scale_max_dim, quality=scale_quality,
                )
            bud.rows_written += 1
            emit(row)
        return False

    return run_slice_sync(
        source=source, dwh_dir=dwh_dir, budget=budget, fetch=fetch,
        id_field="uuid", ts_field="ts", start_at=start_at, full_scan=True,
        snapshot_fmt="ndjson",
        in_scope=lambda r: (r.get("ts") or "") >= floor,
        volatile_fields={"scaled_at"},
    )


def sync_to_warehouse(
    dwh_dir: str,
    config_file: str = "",
    max_runtime_seconds: int = 1500,
) -> dict:
    """Sync the macOS Photos library into the warehouse (full-scan + diff).

    Triggered by ``<!-- clawmeets:photo-sync-trigger -->``. Re-reads the whole
    library each run filtered to ``date_added >= start_at`` and diffs against the
    prior snapshot, so edits (favorites, persons, location) and deletes are
    caught and lowering ``start_at`` re-includes older photos. Optional
    ``scale_to_jpeg: true`` pre-bakes ≤256KB JPEGs for vision-Read.
    """
    check_platform()
    osxphotos = _import_osxphotos()

    now = utcnow_iso()
    cfg, cfg_err = load_config(config_file)
    if cfg_err:
        return {
            "status": "error", "source": "osxphotos", "rows_written": 0,
            "window": [now, now], "watermarks": None,
            "has_more": False, "error": cfg_err,
        }
    cfg = cfg or {}

    budget = SyncBudget(max_runtime_seconds)
    return run_slices(
        source_family="osxphotos", slices=[{}], budget=budget,
        dwh_dir=dwh_dir,
        run_one=lambda _sc: _sync_photos(osxphotos, cfg, dwh_dir, budget),
    )
