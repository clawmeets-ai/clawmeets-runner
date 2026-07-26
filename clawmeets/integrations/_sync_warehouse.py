# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/_sync_warehouse.py

Shared bookkeeping for the per-source ``sync_to_warehouse`` CLI subcommands
(gmail / gcal / gdrive / mailbox / caldav / database / http-api / osxphotos).

**Warehouse v2 — universal delta + snapshot.** Two layers:

    {dwh}/raw/<source>/
        snapshot.ndjson | snapshot.tsv   # current state, upsert-by-id, ts-sorted
        deltas/<runstamp>.ndjson         # append-only change log (incl. tombstones)
        files/...                        # sidecars (attachments, bodies, scaled jpegs)
        sync-state.json                  # {cursor, floor, backfill_target, last_run_*}
        howto.md
    {dwh}/derived/<rule>.tsv (+ deltas/) # etl output (see integrations/etl/_lib.py)

Every source exposes one ``sync_to_warehouse(...)`` entry point that, per slice,
hands a ``fetch`` callback to :func:`run_slice_sync`. The driver owns ALL the
bookkeeping (state, forward + backfill window math, snapshot upsert, delta-log
emission, full-scan diff + tombstones, retention) so the integrations are just
their fetch loops.

**Fetch models** (a per-integration property, not user config):
  - *incremental*: ``fetch(window_start, window_end, budget, emit)`` paginates an
    upstream query bounded by the window and ``emit(row)``s each envelope. The
    driver runs it twice — forward ``(cursor, now]`` then, when ``backfill_target
    < floor``, backfill ``[backfill_target, floor)`` — so lowering ``start_at``
    extends the window BACKWARD. Native deletes arrive as ``emit({..., deleted: True})``.
  - *full-scan* (``full_scan=True``): the integration re-reads the whole source
    each run (filtered client-side) and ``emit``s every current row; the driver
    diffs against the prior snapshot to compute changed rows + tombstones, with an
    ``in_scope`` predicate deciding which absent prior rows count as deletions
    (vs. merely out-of-window). A partial scan (budget fired) degrades to
    upsert-only so a truncated read never tombstones the tail.

**ID contract**: every emitted row MUST carry a stable ``id_field`` value (+ a
``ts_field`` timestamp, recommended for ordering). Intrinsic-id sources derive it
themselves; database/http-api require a user-configured ``id_field`` since the
primary key of an arbitrary query isn't knowable.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional


ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

RESERVED_TOKENS = frozenset({
    "WATERMARK_ISO",
    "WATERMARK_EPOCH",
    "WATERMARK_END_ISO",
    "WATERMARK_END_EPOCH",
    "CURSOR",
    "OFFSET",
    "PAGE_SIZE",
})

EPOCH_ISO = "1970-01-01T00:00:00+00:00"

DELTA_RETENTION_DAYS = 30
"""Delta files older than this are pruned. A consumer whose checkpoint falls
off the tail re-bootstraps from the snapshot (see etl/_lib.py), so pruning is
safe — it only costs a re-scan, never data."""


def expand_env(value, scope: dict[str, str], missing: list[str]):
    """Substitute ``${VAR}`` tokens in any string value (recursing into dicts and lists).

    Resolution order: ``scope`` first (per-request runtime values), then
    ``os.environ`` (user-set defaults). Reserved tokens (``RESERVED_TOKENS``)
    substitute ``""`` when unset; user tokens append to ``missing`` so the
    caller can short-circuit with a clear error.
    """
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            name = m.group(1)
            v = scope.get(name) if name in scope else os.environ.get(name)
            if v is None:
                if name not in RESERVED_TOKENS:
                    missing.append(name)
                return ""
            return str(v)
        return ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: expand_env(v, scope, missing) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, scope, missing) for v in value]
    return value


def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp with microsecond precision."""
    return datetime.now(timezone.utc).isoformat()


def normalize_iso(value: Any) -> Optional[str]:
    """Validate + return an ISO-8601 string (accepting a trailing ``Z``), or None.

    Raises ``ValueError`` if non-empty but unparseable, so config typos surface.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    datetime.fromisoformat(s.replace("Z", "+00:00"))  # raises on garbage
    return s


def atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically — write to .tmp, then rename. Creates parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def atomic_write_json(path: Path, data) -> None:
    """Write a JSON document atomically. ``default=str`` handles datetime/Path."""
    atomic_write_text(path, json.dumps(data, indent=2, default=str))


def write_howto(howto: Any, *, snapshot_dir: Path) -> Optional[str]:
    """Mirror a slice's ``howto`` string to ``{snapshot_dir}/howto.md``.

    Returns ``None`` on success or documented no-op (``howto`` is None /
    blank — a prior howto.md is left untouched), or a one-line error if
    ``howto`` is set to a non-string.
    """
    if howto is None:
        return None
    if not isinstance(howto, str):
        return (
            f"howto must be a string if set; got {type(howto).__name__} "
            f"(remove the field or wrap the value in quotes)"
        )
    if not howto.strip():
        return None
    atomic_write_text(snapshot_dir / "howto.md", howto)
    return None


# ---------------------------------------------------------------------------
# Warehouse catalog (aggregate index over raw snapshots + derived tables)
# ---------------------------------------------------------------------------


def _count_rows(path: Path) -> Optional[int]:
    """Row count for a snapshot/derived table. ``.tsv`` → lines minus the
    header; ``.ndjson`` → non-empty lines. None if unreadable."""
    try:
        n = 0
        with path.open("r", encoding="utf-8") as f:
            if path.suffix == ".tsv":
                first = False
                for line in f:
                    first = True
                    n += 1
                return max(0, n - 1) if first else 0
            for line in f:
                if line.strip():
                    n += 1
        return n
    except OSError:
        return None


def _howto_first_line(howto_path: Path) -> str:
    """First substantive line of a sibling ``howto.md`` — the table's 'consult
    when' discriminator. Prefers the first non-heading prose line (so a howto
    that opens with ``# Title`` surfaces its description, not its title), and
    falls back to the title text if the file is heading-only. '' if absent."""
    fallback = ""
    try:
        for raw in howto_path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if not fallback:
                    fallback = stripped.lstrip("#").strip()
                continue
            return stripped[:140]
    except OSError:
        pass
    return fallback[:140]


def _last_run(state_path: Path) -> str:
    try:
        data = json.loads(state_path.read_text())
        if isinstance(data, dict):
            return str(data.get("last_run_at") or "")
    except (OSError, ValueError):
        pass
    return ""


def refresh_catalog(dwh_dir) -> Optional[str]:
    """(Re)build ``{dwh}/CATALOG.md`` — the aggregate index the agent reads to
    discover what's in the warehouse without walking it blind.

    Rolls up every ``raw/<source>/`` snapshot and every ``derived/<rule>.tsv``
    into the shared knowledge-index shape (``clawmeets.utils.knowledge_index``):
    one bullet per table carrying row count + last-sync time as ``(meta)`` and
    the first line of the sibling ``howto.md`` as the 'consult when'. Paths are
    absolute so the agent's Read tool resolves them from any cwd.

    Deterministic and idempotent — called at the tail of every sync run
    (``run_slices``) and ETL ``merge``, or on demand via ``clawmeets dwh
    catalog``. Returns None on success / no-op (dwh missing), or a one-line
    error string. Never raises.
    """
    # Imported lazily so the heavy sync module stays import-light for callers
    # that only need the fetch helpers.
    from clawmeets.utils.knowledge_index import (
        INDEX_PREAMBLE,
        render_freshness_header,
        render_index_entry,
    )

    try:
        dwh = Path(dwh_dir).expanduser().resolve()
    except (TypeError, ValueError) as exc:
        return f"invalid dwh_dir: {exc}"
    if not dwh.is_dir():
        return None

    raw_entries: list[str] = []
    raw_root = dwh / "raw"
    if raw_root.is_dir():
        seen: set[Path] = set()
        for snap in sorted(list(raw_root.rglob("snapshot.tsv"))
                           + list(raw_root.rglob("snapshot.ndjson"))):
            slice_dir = snap.parent
            if slice_dir in seen:
                continue
            seen.add(slice_dir)
            source = slice_dir.relative_to(raw_root).as_posix()
            n = _count_rows(snap)
            meta = f"{n} rows" if n is not None else ""
            last = _last_run(slice_dir / "sync-state.json")
            if last:
                meta = f"{meta}, synced {last}" if meta else f"synced {last}"
            raw_entries.append(render_index_entry(
                f"raw/{source}", snap.as_posix(),
                when=_howto_first_line(slice_dir / "howto.md"), meta=meta))

    derived_entries: list[str] = []
    derived_root = dwh / "derived"
    if derived_root.is_dir():
        for tsv in sorted(derived_root.glob("*.tsv")):
            if tsv.name.startswith("."):
                continue
            rule = tsv.stem
            n = _count_rows(tsv)
            meta = f"{n} rows" if n is not None else ""
            last = _last_run(derived_root / f"{rule}.sync-state.json")
            if last:
                meta = f"{meta}, built {last}" if meta else f"built {last}"
            derived_entries.append(render_index_entry(
                f"derived/{tsv.name}", tsv.as_posix(),
                when=_howto_first_line(derived_root / f"{rule}.howto.md"),
                meta=meta))

    parts = [
        render_freshness_header({"last_built": utcnow_iso()}),
        "",
        "# Data warehouse catalog",
        "",
        INDEX_PREAMBLE,
    ]
    if raw_entries:
        parts += ["", "## Raw sources", "", *raw_entries]
    if derived_entries:
        parts += ["", "## Derived tables", "", *derived_entries]
    if not raw_entries and not derived_entries:
        parts += ["", "_No tables yet — run a sync or ETL rule to populate the warehouse._"]

    try:
        atomic_write_text(dwh / "CATALOG.md", "\n".join(parts) + "\n")
    except OSError as exc:
        return f"catalog write failed: {exc}"
    return None


# ---------------------------------------------------------------------------
# Soft budget
# ---------------------------------------------------------------------------


class SyncBudget:
    """Tracks elapsed wall-clock time; signals when over budget.

    The fetch callback checks ``should_stop()`` between rows and breaks out
    when True. ``rows_written`` is bumped for the informational rollup but does
    NOT gate ``should_stop`` — only wall-clock time does.
    """

    def __init__(self, max_runtime_seconds: int) -> None:
        self.max_runtime_seconds = max(1, int(max_runtime_seconds))
        self.start = time.monotonic()
        self.rows_written = 0

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def should_stop(self) -> bool:
        return self.elapsed() > self.max_runtime_seconds


# ---------------------------------------------------------------------------
# TSV cell rendering + atomic read/write (snapshots for tabular sources)
# ---------------------------------------------------------------------------


def _cell(v: Any) -> str:
    """Render a row value as one TSV cell.

      - ``None`` → ``""``
      - ``datetime`` → ISO-8601 UTC (naive stamped UTC first)
      - ``bytes`` → base64
      - everything else → ``str(v)``
    """
    if v is None:
        return ""
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc).isoformat()
    if isinstance(v, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(v)).decode()
    return str(v)


def _read_tsv(path: Path) -> tuple[Optional[list[str]], list[list[str]]]:
    """Read a TSV into ``(header, rows)``; ``header`` is None when missing/empty."""
    if not path.exists():
        return None, []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            return None, []
        rows = [row for row in reader]
    return header, rows


def _write_tsv_atomic(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Atomic TSV write (``QUOTE_MINIMAL`` + ``\\n``)."""
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    w.writerow(header)
    for row in rows:
        w.writerow(row)
    atomic_write_text(path, buf.getvalue())


# ---------------------------------------------------------------------------
# Sync state (per source)
# ---------------------------------------------------------------------------


def read_sync_state(state_path: Path, source: str) -> dict:
    """Read sync-state.json; initialize a fresh dict if missing/corrupt.

    Schema::

        {source, cursor, floor, backfill_target, last_run_at, last_run_count, last_error}

    ``cursor`` = forward high watermark (newest synced); ``floor`` = backfill low
    watermark (oldest synced back to); ``backfill_target`` = the config ``start_at``
    echo that pulls ``floor`` down. On a fresh state all three are None.
    """
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {
        "source": source,
        "cursor": None,
        "floor": None,
        "backfill_target": None,
        "last_run_at": None,
        "last_run_count": 0,
        "last_error": None,
    }


def write_sync_state(state_path: Path, state: dict) -> None:
    atomic_write_json(state_path, state)


# ---------------------------------------------------------------------------
# Snapshot I/O (ndjson for envelope sources, tsv for tabular)
# ---------------------------------------------------------------------------


def snapshot_path(base: Path, fmt: str) -> Path:
    return base / ("snapshot.ndjson" if fmt == "ndjson" else "snapshot.tsv")


def load_snapshot(base: Path, fmt: str) -> list[dict]:
    """Load the current snapshot as a list of row dicts ([] if absent)."""
    path = snapshot_path(base, fmt)
    if not path.exists():
        return []
    if fmt == "ndjson":
        out: list[dict] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out
    header, rows = _read_tsv(path)
    if header is None:
        return []
    return [dict(zip(header, r)) for r in rows]


def _sort_rows(rows: list[dict], id_field: str, ts_field: str) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (str(r.get(ts_field) or ""), str(r.get(id_field) or "")),
    )


def write_snapshot(
    base: Path,
    rows: list[dict],
    fmt: str,
    *,
    id_field: str,
    ts_field: str,
    tsv_columns: Optional[list[str]] = None,
) -> None:
    """Atomic-write the snapshot, ts-sorted (then id) for tail-recency consumers."""
    ordered = _sort_rows(rows, id_field, ts_field)
    path = snapshot_path(base, fmt)
    if fmt == "ndjson":
        body = "".join(json.dumps(r, default=str) + "\n" for r in ordered)
        atomic_write_text(path, body)
        return
    # tsv
    columns = tsv_columns
    if columns is None:
        columns = list(ordered[0].keys()) if ordered else None
    if columns is None:
        # Nothing to write and no schema known — leave any prior file as-is.
        return
    body_rows = [[_cell(r.get(c)) for c in columns] for r in ordered]
    _write_tsv_atomic(path, columns, body_rows)


# ---------------------------------------------------------------------------
# Delta log
# ---------------------------------------------------------------------------


def _runstamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def write_delta(deltas_dir: Path, rows: list[dict]) -> Optional[Path]:
    """Append one delta file (ndjson) for this run. No-op (returns None) on empty."""
    if not rows:
        return None
    deltas_dir.mkdir(parents=True, exist_ok=True)
    name = _runstamp()
    path = deltas_dir / f"{name}.ndjson"
    # Guard the pathological same-microsecond collision.
    suffix = 0
    while path.exists():
        suffix += 1
        path = deltas_dir / f"{name}-{suffix}.ndjson"
    body = "".join(json.dumps(r, default=str) + "\n" for r in rows)
    atomic_write_text(path, body)
    return path


def prune_deltas(deltas_dir: Path, *, retention_days: int = DELTA_RETENTION_DAYS) -> None:
    """Delete delta files whose runstamp is older than the retention window."""
    if not deltas_dir.is_dir():
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    for child in deltas_dir.iterdir():
        if not child.is_file() or not child.name.endswith(".ndjson"):
            continue
        stamp = child.name[:15]  # YYYYMMDDTHHMMSS
        try:
            when = datetime.strptime(stamp, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if when < cutoff:
            try:
                child.unlink()
            except OSError:
                pass


def list_deltas(deltas_dir: Path) -> list[str]:
    """Sorted (chronological) delta filenames under ``deltas_dir``."""
    if not deltas_dir.is_dir():
        return []
    return sorted(
        p.name for p in deltas_dir.iterdir()
        if p.is_file() and p.name.endswith(".ndjson")
    )


# ---------------------------------------------------------------------------
# Snapshot mutation: upsert (incremental) and diff (full-scan)
# ---------------------------------------------------------------------------


def _content_eq(a: dict, b: dict, volatile: set[str]) -> bool:
    """Compare two rows ignoring ``_``-prefixed transient keys + ``volatile`` keys.

    Values are normalized through ``_cell`` so a row that round-tripped through
    a TSV snapshot (all strings) compares equal to the same row freshly fetched
    with native types (``40`` vs ``"40"``) — otherwise every full-scan re-run
    would spuriously flag every row as changed.
    """
    def proj(d: dict) -> dict:
        return {
            k: _cell(v) for k, v in d.items()
            if not k.startswith("_") and k not in volatile
        }
    return proj(a) == proj(b)


def apply_upsert(
    prior_by_id: dict[str, dict],
    rows: list[dict],
    *,
    id_field: str,
) -> list[dict]:
    """Overlay ``rows`` onto the prior snapshot by id; ``deleted: True`` rows remove.

    Returns the new snapshot row list (unsorted — caller sorts on write).
    """
    by_id = dict(prior_by_id)
    for r in rows:
        rid = str(r.get(id_field))
        if r.get("deleted"):
            by_id.pop(rid, None)
        else:
            by_id[rid] = r
    return list(by_id.values())


def diff_snapshot(
    prior_by_id: dict[str, dict],
    current_rows: list[dict],
    *,
    id_field: str,
    ts_field: str,
    in_scope: Optional[Callable[[dict], bool]] = None,
    volatile_fields: Optional[set[str]] = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Diff a full re-scan against the prior snapshot.

    Returns ``(changed_rows, tombstones, new_snapshot_rows)``:
      - ``changed_rows``: rows new or whose content changed (→ delta log).
      - ``tombstones``: ``{id, ts, deleted: True}`` for prior rows absent from
        the scan AND ``in_scope`` (genuine deletes). Out-of-scope absent rows are
        preserved untouched — widening re-includes them, narrowing never deletes.
      - ``new_snapshot_rows``: out-of-scope preserved rows + all current rows.
    """
    volatile = volatile_fields or set()
    in_scope = in_scope or (lambda _r: True)
    current_by_id = {str(r.get(id_field)): r for r in current_rows
                     if r.get(id_field) not in (None, "")}

    changed: list[dict] = []
    for cid, crow in current_by_id.items():
        prior = prior_by_id.get(cid)
        if prior is None or not _content_eq(prior, crow, volatile):
            changed.append(crow)

    tombstones: list[dict] = []
    preserved: list[dict] = []
    for pid, prow in prior_by_id.items():
        if pid in current_by_id:
            continue
        if in_scope(prow):
            tombstones.append({
                id_field: prow.get(id_field, pid),
                ts_field: prow.get(ts_field, ""),
                "deleted": True,
            })
        else:
            preserved.append(prow)

    new_snapshot = preserved + list(current_by_id.values())
    return changed, tombstones, new_snapshot


# ---------------------------------------------------------------------------
# The shared driver
# ---------------------------------------------------------------------------


EmitFn = Callable[[dict], None]
FetchFn = Callable[[str, str, "SyncBudget", EmitFn], bool]
"""``fetch(window_start_iso, window_end_iso, budget, emit) -> has_more``.

Paginate the upstream within the window, ``emit(row)`` each envelope (a dict
carrying ``id_field`` + ``ts_field``; set ``deleted: True`` for native
tombstones). Check ``budget.should_stop()`` between rows; return True if you
bailed before the window was exhausted. For ``full_scan`` sources the window is
``(filter_floor, now]`` and you re-read the whole source, filtering client-side.
"""


def _collect(fetch: FetchFn, w_start: str, w_end: str, budget: SyncBudget,
             id_field: str, ts_field: str) -> tuple[list[dict], Optional[str], Optional[str], bool]:
    """Run one fetch pass; return ``(rows, min_ts, max_ts, has_more)``.

    Raises ``ValueError`` immediately if an emitted row lacks ``id_field``
    (an integration bug — fail fast rather than write an unkeyed row).
    """
    rows: list[dict] = []
    mn: Optional[str] = None
    mx: Optional[str] = None

    def emit(row: dict) -> None:
        nonlocal mn, mx
        if row.get(id_field) in (None, ""):
            raise ValueError(
                f"emitted row missing required id field {id_field!r}: keys={sorted(row)}"
            )
        rows.append(row)
        ts = row.get(ts_field)
        if isinstance(ts, str) and ts:
            if mn is None or ts < mn:
                mn = ts
            if mx is None or ts > mx:
                mx = ts

    has_more = bool(fetch(w_start, w_end, budget, emit))
    return rows, mn, mx, has_more


def run_slice_sync(
    *,
    source: str,
    dwh_dir: str,
    budget: SyncBudget,
    fetch: FetchFn,
    id_field: str = "id",
    ts_field: str = "ts",
    start_at: Any = None,
    full_scan: bool = False,
    snapshot_fmt: str = "ndjson",
    tsv_columns: Optional[list[str]] = None,
    in_scope: Optional[Callable[[dict], bool]] = None,
    volatile_fields: Optional[set[str]] = None,
) -> dict:
    """Run one slice's sync cycle: fetch → snapshot upsert/diff → delta → state.

    Returns ``{name, rows_written, watermarks, has_more, error}`` (``name`` is the
    last path segment of ``source``). ``watermarks`` is ``{cursor, floor}``.
    """
    name = source.rsplit("/", 1)[-1]

    def _result(rows_written, st, has_more, error):
        return {
            "name": name,
            "rows_written": rows_written,
            "watermarks": {"cursor": st.get("cursor"), "floor": st.get("floor")},
            "has_more": has_more,
            "error": error,
        }

    dwh = Path(dwh_dir).expanduser().resolve()
    base = dwh / "raw" / source
    state_path = base / "sync-state.json"
    deltas_dir = base / "deltas"
    state = read_sync_state(state_path, source)
    now = utcnow_iso()

    try:
        target = normalize_iso(start_at)
    except ValueError as exc:
        state["last_run_at"] = now
        state["last_error"] = f"invalid start_at {start_at!r}: {exc}"
        write_sync_state(state_path, state)
        return _result(0, state, False, state["last_error"])

    prior_rows = load_snapshot(base, snapshot_fmt)
    prior_by_id = {str(r[id_field]): r for r in prior_rows
                   if r.get(id_field) not in (None, "")}

    delta_rows: list[dict] = []
    rows_written = 0
    has_more = False

    try:
        if full_scan:
            floor = target or EPOCH_ISO
            collected, _mn, _mx, scan_more = _collect(
                fetch, floor, now, budget, id_field, ts_field)
            rows_written = len(collected)
            if scan_more:
                # Partial scan: never tombstone the unseen tail. Upsert only.
                snapshot_rows = apply_upsert(prior_by_id, collected, id_field=id_field)
                delta_rows = collected
                has_more = True
            else:
                changed, tombstones, snapshot_rows = diff_snapshot(
                    prior_by_id, collected, id_field=id_field, ts_field=ts_field,
                    in_scope=in_scope, volatile_fields=volatile_fields)
                delta_rows = changed + tombstones
            state["cursor"] = now
            state["floor"] = floor
            state["backfill_target"] = target
        else:
            cursor = state.get("cursor") or now
            floor = state.get("floor") or cursor

            # Forward pass: (cursor, now]
            fwd, _fmin, fmax, fwd_more = _collect(
                fetch, cursor, now, budget, id_field, ts_field)
            new_cursor = now if not fwd_more else max(fmax or cursor, cursor)

            # Backfill pass: [backfill_target, floor)
            bf: list[dict] = []
            bf_more = False
            new_floor = floor
            if target and target < floor and not budget.should_stop():
                bf, bfmin, _bfmax, bf_more = _collect(
                    fetch, target, floor, budget, id_field, ts_field)
                new_floor = target if not bf_more else min(bfmin or floor, floor)

            all_rows = fwd + bf
            rows_written = len(all_rows)
            snapshot_rows = apply_upsert(prior_by_id, all_rows, id_field=id_field)
            delta_rows = all_rows
            has_more = fwd_more or bf_more
            state["cursor"] = new_cursor
            state["floor"] = new_floor
            state["backfill_target"] = target
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        state["last_run_at"] = now
        state["last_error"] = err
        write_sync_state(state_path, state)
        return _result(0, state, False, err)

    # Durable writes: delta → snapshot → state (crash-safe; deltas at-least-once).
    write_delta(deltas_dir, delta_rows)
    write_snapshot(base, snapshot_rows, snapshot_fmt,
                   id_field=id_field, ts_field=ts_field, tsv_columns=tsv_columns)
    state["source"] = source
    state["last_run_at"] = now
    state["last_run_count"] = rows_written
    state["last_error"] = None
    write_sync_state(state_path, state)
    prune_deltas(deltas_dir)

    return _result(rows_written, state, has_more, None)


# ---------------------------------------------------------------------------
# Multi-slice aggregation (the sync_to_warehouse outer loop, factored once)
# ---------------------------------------------------------------------------


def run_slices(
    *,
    source_family: str,
    slices: list,
    budget: SyncBudget,
    run_one: Callable[[dict], dict],
    dwh_dir=None,
) -> dict:
    """Loop slices through ``run_one``, aggregating into the uniform result dict.

    ``run_one(slice_cfg) -> per-slice summary`` (the dict :func:`run_slice_sync`
    returns, or an error dict with the same keys). Returns
    ``{status, source, rows_written, window, watermarks, has_more, error, per_slice}``.

    When ``dwh_dir`` is provided, the warehouse ``CATALOG.md`` index is
    refreshed once after all slices run so the agent's discovery index reflects
    this sync (see :func:`refresh_catalog`).
    """
    now = utcnow_iso()
    per_slice: dict[str, dict] = {}
    any_error = False
    any_has_more = False
    agg_floor: Optional[str] = None
    agg_cursor: Optional[str] = None
    first_error: Optional[str] = None

    for slice_cfg in slices:
        s_name = slice_cfg.get("name") if isinstance(slice_cfg, dict) else None
        display = s_name if isinstance(s_name, str) and s_name else "<unnamed>"
        if budget.should_stop():
            any_has_more = True
            per_slice[display] = {
                "name": display, "rows_written": 0,
                "watermarks": None, "has_more": True, "error": None,
            }
            continue
        summary = run_one(slice_cfg if isinstance(slice_cfg, dict) else {})
        per_slice[summary["name"]] = summary
        if summary.get("error"):
            any_error = True
            if first_error is None:
                first_error = summary["error"]
        if summary.get("has_more"):
            any_has_more = True
        wms = summary.get("watermarks") or {}
        if wms.get("floor"):
            agg_floor = wms["floor"] if agg_floor is None else min(agg_floor, wms["floor"])
        if wms.get("cursor"):
            agg_cursor = wms["cursor"] if agg_cursor is None else max(agg_cursor, wms["cursor"])

    if any_error and budget.rows_written == 0:
        status = "error"
    elif any_has_more:
        status = "partial"
    elif budget.rows_written == 0:
        status = "noop"
    else:
        status = "ok"

    if dwh_dir is not None:
        refresh_catalog(dwh_dir)

    return {
        "status": status,
        "source": source_family,
        "rows_written": budget.rows_written,
        "window": [agg_floor or now, now],
        "watermarks": (
            {"floor": agg_floor, "cursor": agg_cursor}
            if (agg_floor or agg_cursor) else None
        ),
        "has_more": any_has_more,
        "error": first_error,
        "per_slice": per_slice,
    }
