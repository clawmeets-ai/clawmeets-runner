# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/osxphotos/_lib.py

Read-only macOS Photos library access (Photos.app + iCloud Photo Library) via
the ``osxphotos`` package, plus JPEG transcoding for vision-friendly Read
consumption.
"""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Optional


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
