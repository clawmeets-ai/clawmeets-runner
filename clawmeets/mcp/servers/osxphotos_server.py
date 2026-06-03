# SPDX-License-Identifier: MIT
"""
clawmeets/mcp/servers/osxphotos_server.py

osxphotos MCP server. Read-only access to the macOS Photos library
(Photos.app + iCloud Photo Library) via the osxphotos library. Tools:
  - list_albums: enumerate every album in the library with photo counts
  - list_photos: list photos in an album (optionally filtered by year),
                 returning metadata + filesystem paths but NOT image bytes
  - export_photo: force-download an iCloud-optimized photo to a destination
  - export_photo_as_jpeg: transcode to a small JPEG so Claude Code's Read
                          tool (256 KB image cap) can ingest it for vision

The agent reads image bytes via Claude's native Read tool from the paths
returned by list_photos. Keeping bytes out of MCP responses avoids dumping
multi-MB image blobs into the JSON-RPC channel.

Requires macOS and `osxphotos` installed on the runner.
"""
from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from clawmeets.mcp.servers._sync_warehouse import (
    SyncBudget,
    _read_state,
    atomic_write_json,
    gc_old_timestamps,
    merge_json_envelopes,
    new_timestamp_dir,
    utcnow_iso,
    write_howto,
)
from clawmeets.utils.jsonc import parse_jsonc


def _load_config(config_file: str) -> tuple[Optional[dict], Optional[str]]:
    """Read this MCP's optional config from the file path supplied by the
    agent. The osxphotos MCP doesn't take a slice list — the only field
    today is ``howto`` (mirrored to the warehouse on each sync).

    Returns ``(cfg, err)`` with the same noop-on-missing semantics as the
    other personal_data MCPs.
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
        return None, f"config file is not valid JSON: {exc}"
    if not isinstance(cfg, dict):
        return None, "config file must contain a JSON object"
    return cfg, None


def _rmdir_if_empty(path: Path) -> None:
    """Remove ``path`` if it's an empty directory; ignore otherwise."""
    try:
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    except OSError:
        pass


def _check_platform() -> None:
    if platform.system() != "Darwin":
        raise RuntimeError(
            "osxphotos MCP requires macOS — the Photos library is macOS-only. "
            f"Current platform: {platform.system()}."
        )


def _transcode_to_jpeg(
    src: str, dst: Path, max_dim: int, quality: int,
) -> tuple[bool, Optional[str]]:
    """Shell sips to transcode `src` into a scaled JPEG at `dst`. Returns
    (ok, error_msg). Used by both the on-demand export_photo_as_jpeg tool
    and the sync_to_warehouse `scale_to_jpeg` pre-bake path."""
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


def _populate_scaled_path(
    row: dict, *, scaled_dir: Path, dwh_root: Path,
    max_dim: int, quality: int,
) -> None:
    """Decorate the envelope `row` with `scaled_path` + `scaled_at` (or
    `scaled_error` on failure). Mirrors mailbox `_persist_attachments`:
    per-photo errors never raise.

    Idempotent — if the destination JPEG already exists with mtime >= source
    mtime, skip the sips invocation and reuse the existing file.
    """
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


def _import_osxphotos():
    try:
        import osxphotos  # type: ignore
        return osxphotos
    except ImportError as exc:
        raise RuntimeError(
            "The `osxphotos` package is required but missing. Install it on "
            "the runner: pip install osxphotos"
        ) from exc


def _photo_to_dict(p) -> dict:
    """Flatten an osxphotos PhotoInfo into a JSON-friendly dict.

    `path` is the on-disk location of the original; None for iCloud-optimized
    photos that haven't been downloaded. Use export_photo to force-download.

    Both `date` (when the photo was taken, from EXIF) and `date_added` (when
    it was added to the Photos library) are surfaced — sync workflows want
    `date_added` (so an old EXIF photo imported today shows up in today's
    sync), while curation workflows want `date`.
    """
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


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The `mcp` package is required but missing — the clawmeets runner "
            "should bundle it by default. Try: pip install --upgrade clawmeets"
        ) from exc

    _check_platform()
    osxphotos = _import_osxphotos()

    mcp = FastMCP("clawmeets-osxphotos")

    @mcp.tool()
    def list_albums() -> list[dict]:
        """List every album in the user's Photos library.

        Returns one dict per album: {name, photo_count, date_min, date_max}.
        `date_min` and `date_max` are ISO timestamps of the earliest/latest
        photo in the album, useful for picking the album that matches a year.

        Call this once at the start of a session and pick the album whose
        name matches the user's request (e.g. "2025", "Year 2025", "Trips").
        """
        db = osxphotos.PhotosDB()
        out = []
        for ai in db.album_info:
            photos = ai.photos
            if not photos:
                out.append({
                    "name": ai.title,
                    "photo_count": 0,
                    "date_min": None,
                    "date_max": None,
                })
                continue
            dates = [p.date for p in photos if p.date]
            out.append({
                "name": ai.title,
                "photo_count": len(photos),
                "date_min": min(dates).isoformat() if dates else None,
                "date_max": max(dates).isoformat() if dates else None,
            })
        return out

    @mcp.tool()
    def list_photos(
        album: Optional[str] = None,
        year: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """List photos in the library or a specific album, returning metadata
        and filesystem paths but NOT image bytes.

        Args:
            album: album name (use list_albums first to discover names). If
                None, returns all photos in the library — combine with `year`
                to keep the result manageable.
            year: optional 4-digit year filter (e.g. 2025). Applied on photo
                creation date.
            limit: optional max number of photos to return. Useful for
                sanity-checks before pulling the full list.

        Each returned dict has: {uuid, filename, date, location, persons,
        favorite, hidden, path}. `path` may be None for iCloud-optimized
        photos that aren't downloaded locally — call export_photo to force
        a download.

        Recommended workflow for large albums:
          1. Call with `limit=50` to inspect a sample.
          2. Cluster the full list by (date-day, location) to identify events.
          3. Read images via Read tool only for the candidate keepers — this
             tool returns paths, not bytes, on purpose.
        """
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

    @mcp.tool()
    def sync_to_warehouse(
        dwh_dir: str,
        config_file: str = "",
        max_runtime_seconds: int = 1500,
    ) -> dict:
        """Sync newly-added photos into the personal data warehouse.

        Call this exactly once when you receive a DM whose body starts with
        ``<!-- clawmeets:photo-sync-trigger -->``. Read ``dwh_dir`` from
        your prompt's ``== DATA WAREHOUSE ==`` block and ``config_file``
        from your ``== MCP CONFIG FILES ==`` block (the path next to
        ``osxphotos``). The config file is optional — its only field today
        is ``howto`` (mirrored to ``howto.md`` in both warehouse layers).

        Single-dataset model (no slices — there's only one Photos library
        on the host). Per-run envelopes land in
        ``{dwh_dir}/sources/osxphotos/<TIMESTAMP>/<uuid>.json`` siblings
        of ``sync-state.json``, and the consolidated dataset rebuilds at
        ``{dwh_dir}/merged/osxphotos.json`` (a JSON array sorted by ``ts``,
        deduped by ``uuid`` via merge_policy=upsert). Up to
        ``KEEP_RECENT_DUMPS`` timestamp folders are retained.

        Watermark semantics: filter is ``p.date_added`` (when the photo
        was added to the Photos library), NOT ``p.date`` (when it was
        taken). An old 2018-EXIF photo imported today shows up in today's
        sync. Each run pulls photos with ``date_added`` in
        ``(high_watermark, now]``.

        First-run seed: when ``sync-state.json`` is missing, both
        watermarks default to ``window_end`` (forward-only — the existing
        library is NOT backfilled). To backfill instead, set
        ``start_at: "1970-01-01T00:00:00Z"`` (or any earlier ISO-8601
        timestamp) in this MCP's config; that value seeds both watermarks
        on the first sync only (``state.last_sync_at == None``).

        Bytes are NOT copied by default — each envelope stores filesystem
        ``path`` (or ``null`` for iCloud-only photos). Downstream skills
        call ``export_photo(uuid, dest_dir=...)`` for the bytes when they
        actually need them.

        Opt-in pre-bake (``scale_to_jpeg: true`` in config): for each photo
        whose ``path`` is non-null, sync also writes a scaled JPEG to
        ``{dwh_dir}/merged/osxphotos.scaled/<uuid>.jpg`` and adds
        ``scaled_path`` (warehouse-relative) + ``scaled_at`` to the
        envelope. Defaults ``scale_max_dim=1200``, ``scale_quality=65``
        produce ~120 KB per photo — under Claude Code's 256 KB Read cap so
        LLM-vision consumers can ingest directly. Idempotent: a re-sync
        skips photos whose scaled JPEG already exists with mtime ≥ source.
        iCloud-only photos get ``scaled_error: "iCloud-only (path null)"``.

        Args:
            dwh_dir: Personal data warehouse root.
            config_file: Optional path to this agent's per-MCP config
                file. Empty / missing file ⇒ proceed without howto.
            max_runtime_seconds: Wall-clock budget. Default 1500.

        Returns: ``{status, source, rows_written, window, watermarks,
        has_more, error}``.
        """
        window_end = utcnow_iso()

        cfg, cfg_err = _load_config(config_file)
        if cfg_err:
            return {
                "status": "error", "source": "osxphotos", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False, "error": cfg_err,
            }

        dwh_root = Path(dwh_dir).expanduser().resolve()
        source = "osxphotos"
        source_dir = dwh_root / "sources" / source
        state_path = source_dir / "sync-state.json"
        merged_path = dwh_root / "merged" / "osxphotos.json"

        howto_err = write_howto(
            (cfg or {}).get("howto"),
            source_dir=source_dir,
            merged_path=merged_path,
        )
        if howto_err:
            return {
                "status": "error", "source": "osxphotos", "rows_written": 0,
                "window": [window_end, window_end], "watermarks": None,
                "has_more": False, "error": howto_err,
            }

        state = _read_state(state_path, source)

        # First sync: seed both watermarks from optional config.start_at,
        # defaulting to window_end (= "forward only from now"). Without
        # this, the fresh-state low/high default to a utcnow_iso() called
        # AFTER window_end was captured, so window_start ≥ window_end and
        # the noop branch fires on every run — sync never bootstraps.
        if state.get("last_sync_at") is None:
            start_at = window_end
            start_at_raw = (cfg or {}).get("start_at")
            if isinstance(start_at_raw, str) and start_at_raw.strip():
                candidate = start_at_raw.strip()
                try:
                    datetime.fromisoformat(candidate.replace("Z", "+00:00"))
                except ValueError as exc:
                    return {
                        "status": "error", "source": "osxphotos", "rows_written": 0,
                        "window": [window_end, window_end], "watermarks": None,
                        "has_more": False,
                        "error": (
                            f"invalid start_at {start_at_raw!r}: {exc} "
                            "(must be ISO-8601, e.g. '2024-01-01' or "
                            "'2024-01-01T00:00:00Z')"
                        ),
                    }
                start_at = candidate
            state["low_watermark"] = start_at
            state["high_watermark"] = start_at

        low = state.get("low_watermark") or utcnow_iso()
        high = state.get("high_watermark") or utcnow_iso()
        window_start = max(low, high)

        if window_start >= window_end:
            # Persist state so the seeded watermarks (and last_sync_at)
            # survive — otherwise the next run re-enters from a fresh
            # _read_state and noops identically forever.
            state["last_sync_at"] = utcnow_iso()
            state["last_sync_count"] = 0
            state["last_error"] = None
            atomic_write_json(state_path, state)
            return {
                "status": "noop", "source": "osxphotos", "rows_written": 0,
                "window": [window_start, window_end],
                "watermarks": {"low": low, "high": high},
                "has_more": False, "error": None,
            }

        budget = SyncBudget(max_runtime_seconds)
        timestamp_dir = new_timestamp_dir(source_dir)
        latest_seen = window_start
        has_more = False

        # Optional pre-bake: scale each photo's bytes to a sub-256 KB JPEG so
        # downstream LLM-vision consumers can Read it without per-photo
        # export_photo_as_jpeg calls. Mirrors mailbox/gmail download_attachments.
        scale_to_jpeg = bool((cfg or {}).get("scale_to_jpeg"))
        scale_max_dim = int((cfg or {}).get("scale_max_dim") or 1200)
        scale_quality = int((cfg or {}).get("scale_quality") or 65)
        scaled_dir = dwh_root / "merged" / "osxphotos.scaled"

        try:
            db = osxphotos.PhotosDB()
            # Filter by date_added in window, sort ascending so watermark
            # advancement is monotonic.
            candidates: list[tuple[str, object]] = []
            for p in db.photos():
                added = getattr(p, "date_added", None)
                if not added:
                    continue
                added_iso = added.isoformat()
                if added_iso <= window_start or added_iso >= window_end:
                    continue
                candidates.append((added_iso, p))
            candidates.sort(key=lambda x: x[0])

            for added_iso, p in candidates:
                if budget.should_stop():
                    has_more = True
                    break
                row = _photo_to_dict(p)
                row["ts"] = added_iso
                if scale_to_jpeg:
                    _populate_scaled_path(
                        row, scaled_dir=scaled_dir, dwh_root=dwh_root,
                        max_dim=scale_max_dim, quality=scale_quality,
                    )
                atomic_write_json(timestamp_dir / f"{p.uuid}.json", row)
                if added_iso > latest_seen:
                    latest_seen = added_iso
                budget.rows_written += 1
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            _rmdir_if_empty(timestamp_dir)
            state["last_sync_at"] = utcnow_iso()
            state["last_sync_count"] = budget.rows_written
            state["last_error"] = err
            atomic_write_json(state_path, state)
            return {
                "status": "error", "source": "osxphotos",
                "rows_written": budget.rows_written,
                "window": [window_start, window_end],
                "watermarks": {"low": low, "high": high},
                "has_more": False, "error": err,
            }

        _rmdir_if_empty(timestamp_dir)

        if timestamp_dir.exists():
            merge_err = merge_json_envelopes(
                timestamp_dir, merged_path,
                policy="upsert", id_column="uuid",
            )
            if merge_err:
                state["last_sync_at"] = utcnow_iso()
                state["last_sync_count"] = budget.rows_written
                state["last_error"] = merge_err
                atomic_write_json(state_path, state)
                return {
                    "status": "error", "source": "osxphotos",
                    "rows_written": budget.rows_written,
                    "window": [window_start, window_end],
                    "watermarks": {"low": low, "high": high},
                    "has_more": False, "error": merge_err,
                }
            gc_old_timestamps(source_dir)

        new_high = window_end if not has_more else max(latest_seen, high)
        state["high_watermark"] = new_high
        state["last_sync_at"] = utcnow_iso()
        state["last_sync_count"] = budget.rows_written
        state["last_error"] = None
        atomic_write_json(state_path, state)

        return {
            "status": "partial" if has_more else (
                "ok" if budget.rows_written > 0 else "noop"
            ),
            "source": "osxphotos",
            "rows_written": budget.rows_written,
            "window": [window_start, window_end],
            "watermarks": {"low": low, "high": new_high},
            "has_more": has_more,
            "error": None,
        }

    @mcp.tool()
    def export_photo(uuid: str, dest_dir: str) -> dict:
        """Force-download an iCloud-optimized photo and write it to dest_dir.

        Use this when list_photos returned `path: null` for a photo you
        actually need to read — that means the photo is in iCloud but not
        cached locally, so neither Read nor any path-based tool can see it.

        Args:
            uuid: the photo's UUID from list_photos.
            dest_dir: absolute directory path to write the exported file
                into. Created if missing.

        Returns: {ok, exported_path, error}. On success exported_path is
        absolute and points at the original-quality copy.
        """
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

    @mcp.tool()
    def export_photo_as_jpeg(
        uuid: str,
        dest_dir: str,
        max_dim: int = 1200,
        quality: int = 65,
    ) -> dict:
        """Transcode a photo to a JPEG sized to fit Claude Code's 256 KB Read cap.

        Use this when `Read envelope.path` returned "File content exceeds
        maximum allowed size (256KB)" — typical for iPhone HEIC originals
        (1–7 MB). Default `max_dim=1200, quality=65` produces ~120 KB for a
        typical 12 MP receipt photo, comfortably under the cap with margin
        for OCR detail. Never shell `sips` by hand — this tool is the only
        correct path.

        Args:
            uuid: the photo's UUID from list_photos.
            dest_dir: absolute directory path to write the JPEG into. Created
                if missing. Output filename is `<uuid>.jpg`.
            max_dim: longest-edge pixel cap. Default 1200.
            quality: JPEG quality 0-100. Default 65.

        Returns: {ok, exported_path, byte_size, error}. On success
        exported_path is absolute and points at the transcoded JPEG. On
        iCloud-only photos (path is null), returns ok=False so the caller
        can skip-and-continue.
        """
        results = osxphotos.PhotosDB().photos(uuid=[uuid])
        if not results:
            return {"ok": False, "exported_path": None, "byte_size": None,
                    "error": f"No photo with uuid {uuid!r}"}
        photo = results[0]
        src = photo.path
        if not src:
            return {"ok": False, "exported_path": None, "byte_size": None,
                    "error": "Photo not cached locally (iCloud-only). Call "
                             "export_photo first to force-download, then retry."}

        dest = Path(dest_dir).expanduser().resolve()
        dest.mkdir(parents=True, exist_ok=True)
        out_path = dest / f"{uuid}.jpg"

        ok, err = _transcode_to_jpeg(src, out_path, max_dim, quality)
        if not ok:
            return {"ok": False, "exported_path": None, "byte_size": None,
                    "error": err}

        return {"ok": True, "exported_path": str(out_path),
                "byte_size": out_path.stat().st_size, "error": None}

    mcp.run()


if __name__ == "__main__":
    main()
