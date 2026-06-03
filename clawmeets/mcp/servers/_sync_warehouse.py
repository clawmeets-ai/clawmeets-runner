# SPDX-License-Identifier: MIT
"""
clawmeets/mcp/servers/_sync_warehouse.py

Shared bookkeeping for the per-source ``sync_to_warehouse`` MCP tools.

Each source MCP (gmail, gcal, photos) exposes one ``sync_to_warehouse(dwh_dir,
max_runtime_seconds)`` tool. The source-specific code is just the
paginate callback — everything else (state file I/O, window math, soft
budget, atomic writes, watermark advancement, structured return dict) lives
here so the three implementations stay aligned and the bookkeeping is
deterministic Python rather than three near-identical copies.

The agent-facing trigger contract:
  - DM body starts with ``<!-- clawmeets:<source>-sync-trigger -->``
  - Server (`routes/messages.py`) extends batch timeout to 1h on match
  - Agent calls the matching ``sync_to_warehouse(dwh_dir=…)`` tool exactly once
  - Tool returns ``{status, source, rows_written, window, watermarks, has_more, error}``
  - Reply triggers ``last_synced_at`` bump via ``reflection_completion.py``
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
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


def expand_env(value, scope: dict[str, str], missing: list[str]):
    """Substitute ``${VAR}`` tokens in any string value (recursing into dicts and lists).

    Resolution order: ``scope`` first (MCP-injected per-request runtime values),
    then ``os.environ`` (user-set defaults). Net effect: env vars act as
    *defaults* that the MCP can override at runtime.

    Unset-token policy:
      - **Reserved** tokens (``RESERVED_TOKENS``): substitute ``""`` and continue.
        The MCP doesn't always have a runtime value (e.g. ``CURSOR`` on the
        first request); empty-string fall-through lets opaque-cursor APIs work
        without forcing the user to set a sentinel, while SQL users with
        broken WHERE clauses get an informative DB error from the driver.
      - **User** tokens: append the name to ``missing``; substitute ``""``.
        Caller is expected to inspect ``missing`` and short-circuit with a
        clear error envelope so config typos surface immediately.

    Numbers, bools, ``None`` pass through untouched.
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


def atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically — write to .tmp, then rename. Creates parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def atomic_write_json(path: Path, data) -> None:
    """Write a JSON document atomically. ``default=str`` handles datetime/Path."""
    atomic_write_text(path, json.dumps(data, indent=2, default=str))


def write_howto(
    howto: Any,
    *,
    source_dir: Path,
    merged_path: Path,
) -> Optional[str]:
    """Mirror the slice's ``howto`` string to both layers of the warehouse.

    Writes two files atomically when ``howto`` is a non-empty string:

      - ``{source_dir}/howto.md``                 (alongside ``sync-state.json``)
      - ``{merged_path.parent}/{merged_path.stem}.howto.md``
        (sibling of the consolidated dataset)

    Returns:
      - ``None`` on success or on a documented no-op:
          • ``howto is None`` (field absent)
          • ``howto == ""`` or whitespace-only (user blanked it)
        In the no-op case, **no files are written or deleted** — a previously
        written howto.md stays on disk untouched. Users who want it gone
        ``rm`` it themselves.
      - A one-line error message when ``howto`` is set to a non-string
        (list / dict / number / bool). Caller surfaces this as a per-entry
        ``error`` envelope and skips the fetch.

    The string is written verbatim (no trailing newline added, no Markdown
    rendering). Treating the content as Markdown is a convention for human
    readers; the MCP doesn't parse it.
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

    sources_howto = source_dir / "howto.md"
    merged_howto = merged_path.parent / f"{merged_path.stem}.howto.md"
    atomic_write_text(sources_howto, howto)
    atomic_write_text(merged_howto, howto)
    return None


def _read_state(state_path: Path, source: str) -> dict:
    """Read sync-state.json; initialize a fresh state dict if missing/corrupt."""
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass  # fall through to fresh init
    now = utcnow_iso()
    return {
        "source": source,
        "low_watermark": now,
        "high_watermark": now,
        "last_sync_at": None,
        "last_sync_count": 0,
        "last_error": None,
    }


class SyncBudget:
    """Tracks elapsed time; signals when over budget.

    The paginate callback checks ``should_stop()`` between rows and breaks
    out when True. ``rows_written`` is mutated by callbacks for the
    informational top-level rollup, but does NOT gate ``should_stop`` —
    only wall-clock time does. Synced rows go straight to disk via
    ``TsvSliceWriter.flush()`` (or per-row writers in the simpler MCPs),
    not through the tool's return value; the LLM only ever sees a
    constant-size per-slice summary, so a row-count cap was never
    defending against a token-cost concern. Runner-RAM is bounded
    per-slice by the writer's own buffering (sequential outer loop).
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
# TsvSliceWriter — append-only TSV per slice (one data.tsv per query/endpoint)
# ---------------------------------------------------------------------------


TSV_FILENAME = "data.tsv"


def _cell(v: Any) -> str:
    """Render a row value as one TSV cell.

    Contract (mirrors db_server's old ``_jsonable`` shape for scalar cells):
      - ``None`` → ``""`` (TSV's "no value" convention; ``csv.reader`` returns
        ``""`` here, so the round-trip preserves emptiness without ambiguity)
      - ``datetime`` → ISO-8601 UTC (naive datetimes are stamped UTC first)
      - ``bytes``/``bytearray``/``memoryview`` → base64
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


class TsvSliceWriter:
    """In-memory row buffer + on-flush schema-validated append to ``data.tsv``.

    One instance per slice (per query in db_server / per endpoint in
    api_server). Lifecycle::

        w = TsvSliceWriter(raw_root)
        if w.error: …                      # preload failed (file unreadable)
        for row in rows:
            if not w.add_row(row): …       # schema mismatch — bail out of slice
        w.flush()                          # success/partial exit only

    On schema mismatch ``add_row`` sets ``self.error`` and returns ``False``;
    subsequent ``flush()`` is a no-op (don't corrupt the file). On exception
    inside the sync loop, callers MUST NOT call ``flush()`` — the buffer is
    discarded so the next sync re-emits the same window cleanly.

    Header row is captured from the first ``add_row`` call when the file
    does not yet exist; otherwise it's preloaded from the existing file's
    first line. Subsequent rows must produce the same insertion-ordered
    key list (SQLAlchemy ``mappings()`` and httpx-decoded JSON both
    preserve dict insertion order).
    """

    def __init__(self, raw_root: Path) -> None:
        self._raw_root = raw_root
        self._path = raw_root / TSV_FILENAME
        self._buffer: list[list[str]] = []
        self._expected_columns: Optional[list[str]] = None
        self._error: Optional[str] = None
        self._needs_header_write = False
        if self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8", newline="") as f:
                    first = f.readline()
                if first:
                    # csv.reader handles QUOTE_MINIMAL escaping (embedded \t in
                    # column names would be rare but legal).
                    parsed = next(
                        csv.reader([first.rstrip("\n")], delimiter="\t"),
                        None,
                    )
                    if parsed is not None:
                        self._expected_columns = list(parsed)
            except OSError as exc:
                self._error = f"could not read existing {self._path}: {exc}"

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def expected_columns(self) -> Optional[list[str]]:
        return self._expected_columns

    def add_row(self, row: dict) -> bool:
        """Schema-validate and buffer one row. Returns False on mismatch."""
        incoming = list(row.keys())
        if self._expected_columns is None:
            self._expected_columns = incoming
            self._needs_header_write = True
        elif incoming != self._expected_columns:
            self._error = (
                f"TSV header mismatch: existing columns {self._expected_columns} "
                f"vs incoming {incoming} — rm {self._path} to re-emit "
                f"under the new schema"
            )
            return False
        self._buffer.append([_cell(row[c]) for c in self._expected_columns])
        return True

    def flush(self) -> None:
        """Atomic append-and-fsync. No-op if the buffer is empty."""
        if not self._buffer:
            return
        if self._error is not None:
            # Defensive: never write after a schema error was raised.
            return
        self._raw_root.mkdir(parents=True, exist_ok=True)
        # Build the entire block in memory so the actual write is one syscall
        # (POSIX guarantees a single small write to an O_APPEND fd is
        # serialized w.r.t. other appenders; not strictly required here since
        # only one sync runs at a time, but it keeps crash surface minimal).
        buf = io.StringIO()
        w = csv.writer(
            buf,
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        if self._needs_header_write:
            assert self._expected_columns is not None  # always set by add_row
            w.writerow(self._expected_columns)
        for row in self._buffer:
            w.writerow(row)
        block = buf.getvalue()
        # Open in append mode (creates the file if missing). One large write
        # plus fsync; the page-cache flush before close is what makes the
        # rows durable.
        with self._path.open("a", encoding="utf-8", newline="") as f:
            f.write(block)
            f.flush()
            os.fsync(f.fileno())
        self._buffer.clear()
        self._needs_header_write = False


# ---------------------------------------------------------------------------
# Timestamp dumps, retention, and merge_policy bookkeeping
# ---------------------------------------------------------------------------
#
# Per-run dumps land under ``{dwh_dir}/sources/<source>/<name>/<TIMESTAMP>/``;
# the per-name consolidated dataset lives at
# ``{dwh_dir}/merged/<source>/<name>.<ext>`` and is rebuilt from the latest
# dump after every successful sync. Retention keeps the most-recent
# ``KEEP_RECENT_DUMPS`` timestamp folders; older ones are GC'd (their content
# is already folded into the merged file).


KEEP_RECENT_DUMPS = 5
"""Per-slice timestamp folders kept under ``sources/<source>/<name>/``.

The merged file is the source of truth; older dumps stay around for audit/
recovery only. Hardcoded — bump if you find yourself rolling back more than
five runs into the past."""


TIMESTAMP_RE = re.compile(r"^\d{8}T\d{6}Z$")
"""``YYYYMMDDTHHMMSSZ`` — the on-disk format ``new_timestamp_dir`` emits.

Used by ``gc_old_timestamps`` to filter siblings of ``sync-state.json``
without accidentally deleting unrelated files (legacy ``raw/`` from older
checkouts, an editor's swapfile, etc.). Lexicographic sort matches
chronological sort because every field is fixed-width zero-padded UTC."""


def new_timestamp_dir(source_dir: Path) -> Path:
    """Create and return a fresh ``<source_dir>/<YYYYMMDDTHHMMSSZ>/`` folder.

    Run-start UTC, second-resolution. Two runs starting within the same
    second on the same slice would collide — pathological but possible if a
    user manually retriggers immediately after a fast noop. Cheap defense:
    spin in a tight retry loop appending ``-1``, ``-2``, … if the directory
    already exists. Single-process sync, so the loop bound is small.
    """
    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = source_dir / base
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = source_dir / f"{base}-{suffix}"
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def gc_old_timestamps(source_dir: Path, *, keep: int = KEEP_RECENT_DUMPS) -> None:
    """Delete all but the ``keep`` most-recent timestamp folders.

    No-op when ``source_dir`` doesn't exist or has ``≤ keep`` matching
    children. Only directories whose names match ``TIMESTAMP_RE`` are
    candidates — anything else (the slice's ``sync-state.json``, a legacy
    ``raw/`` directory from a pre-refactor checkout, an editor backup) is
    left alone.
    """
    if not source_dir.is_dir():
        return
    candidates = sorted(
        (p for p in source_dir.iterdir() if p.is_dir() and TIMESTAMP_RE.match(p.name)),
        key=lambda p: p.name,
    )
    for stale in candidates[:-keep] if keep > 0 else candidates:
        shutil.rmtree(stale, ignore_errors=True)


def validate_merge_policy(cfg: dict) -> tuple[str, Optional[str], Optional[str]]:
    """Resolve and validate a slice's ``merge_policy`` / ``merge_policy_upsert_id_column``.

    Returns ``(policy, id_column, error)``.
      - ``policy``: ``"replace"`` (default) or ``"upsert"``.
      - ``id_column``: the upsert key when ``policy == "upsert"``, else ``None``.
      - ``error``: a one-line per-slice error message, or ``None`` on success.

    Failure modes:
      - ``merge_policy`` set to anything other than ``"replace"`` / ``"upsert"``.
      - ``merge_policy == "upsert"`` with an empty/missing ``merge_policy_upsert_id_column``.
    """
    raw_policy = cfg.get("merge_policy")
    if raw_policy is None or raw_policy == "":
        policy = "replace"
    elif raw_policy in ("replace", "upsert"):
        policy = raw_policy
    else:
        return ("replace", None, (
            f"invalid merge_policy {raw_policy!r}: must be one of "
            "{'replace', 'upsert'} (default: 'replace')"
        ))

    if policy == "upsert":
        raw_id = cfg.get("merge_policy_upsert_id_column")
        if not isinstance(raw_id, str) or not raw_id.strip():
            return (policy, None, (
                "merge_policy='upsert' requires a non-empty "
                "`merge_policy_upsert_id_column`"
            ))
        return (policy, raw_id.strip(), None)

    return (policy, None, None)


def _read_tsv(path: Path) -> tuple[Optional[list[str]], list[list[str]]]:
    """Read a TSV into ``(header, rows)``.

    ``header`` is ``None`` only when the file is missing or empty. ``rows``
    omits the header line. The caller decides what to do with mismatched or
    short rows — this helper is dumb on purpose.
    """
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
    """Atomic write: build the body in memory, write to ``.tmp``, ``os.replace``.

    Mirrors ``TsvSliceWriter.flush`` quoting (``QUOTE_MINIMAL`` + ``\\n``
    line terminator) so the merged file round-trips through
    ``csv.reader(delimiter='\\t')`` identically to per-run dumps.
    """
    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    w.writerow(header)
    for row in rows:
        w.writerow(row)
    atomic_write_text(path, buf.getvalue())


def merge_tsv(
    timestamp_dir: Path,
    merged_path: Path,
    *,
    policy: str,
    id_column: Optional[str],
) -> Optional[str]:
    """Fold ``<timestamp_dir>/data.tsv`` into ``merged_path`` per policy.

    Returns an error message on failure, or ``None`` on success / noop.

    Contract:
      - ``policy == "replace"``: ``merged_path`` becomes a copy of the
        just-written dump. No header validation against any prior merged
        file — the dump's schema wins.
      - ``policy == "upsert"``: read prior merged + new dump; both must
        share the same header; ``id_column`` must be in the header; later
        rows (within the new dump, or across the dump→merged boundary)
        win for any given id. Rows missing or empty in ``id_column`` are
        dropped (they have no identity to merge on).

    A missing ``<timestamp_dir>/data.tsv`` is a noop (the fetch produced
    zero rows; merge has nothing to do).
    """
    dump_path = timestamp_dir / TSV_FILENAME
    dump_header, dump_rows = _read_tsv(dump_path)
    if dump_header is None:
        return None  # empty fetch — merged stays as-is

    if policy == "replace":
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        _write_tsv_atomic(merged_path, dump_header, dump_rows)
        return None

    # upsert
    if not id_column:
        return "merge_policy='upsert' requires a non-empty merge_policy_upsert_id_column"
    if id_column not in dump_header:
        return (
            f"TSV upsert: id column {id_column!r} not in dump header "
            f"{dump_header} — fix the query/endpoint to project that column"
        )

    merged_header, merged_rows = _read_tsv(merged_path)
    if merged_header is None:
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        # First merge for this slice — dedup within the dump itself by id_column,
        # then write. Later rows win.
        idx = dump_header.index(id_column)
        by_id: dict[str, list[str]] = {}
        for row in dump_rows:
            if idx >= len(row):
                continue
            key = row[idx]
            if not key:
                continue
            by_id[key] = row
        _write_tsv_atomic(merged_path, dump_header, list(by_id.values()))
        return None

    if merged_header != dump_header:
        return (
            f"TSV header mismatch: merged {merged_header} vs dump {dump_header} "
            f"— rm {merged_path} to re-emit under the new schema"
        )

    idx = merged_header.index(id_column)
    by_id = {}
    for row in merged_rows:
        if idx >= len(row):
            continue
        key = row[idx]
        if not key:
            continue
        by_id[key] = row
    for row in dump_rows:
        if idx >= len(row):
            continue
        key = row[idx]
        if not key:
            continue
        by_id[key] = row  # dump wins
    _write_tsv_atomic(merged_path, merged_header, list(by_id.values()))
    return None


def _envelope_sort_key(env: dict) -> str:
    """Stable sort key for merged JSON envelopes — falls back gracefully when
    ``ts`` is missing/non-string so a malformed envelope can't crash the merge.
    """
    ts = env.get("ts")
    return ts if isinstance(ts, str) else ""


def merge_json_envelopes(
    timestamp_dir: Path,
    merged_path: Path,
    *,
    policy: str,
    id_column: Optional[str],
) -> Optional[str]:
    """Fold ``<timestamp_dir>/*.json`` envelopes into ``merged_path`` (JSON array).

    Returns an error message on failure, or ``None`` on success / noop.

    Glob matches only top-level ``*.json`` files in ``timestamp_dir`` — the
    gdrive sidecars are ``.tsv`` / ``.txt`` and excluded.

    Contract:
      - ``policy == "replace"``: ``merged_path`` becomes ``sorted(dump_envelopes)``.
      - ``policy == "upsert"``: prior merged envelopes (loaded as a list)
        are indexed by ``id_column``; dump envelopes overlay; result is
        sorted by ``ts`` and written. Envelopes missing ``id_column`` are
        an error (they have no identity to merge on).

    A timestamp folder containing zero ``*.json`` files is a noop.
    """
    if not timestamp_dir.is_dir():
        return None
    dump_envelopes: list[dict] = []
    for child in sorted(timestamp_dir.glob("*.json")):
        if not child.is_file():
            continue
        try:
            obj = json.loads(child.read_text())
        except Exception as exc:
            return f"could not parse {child}: {exc}"
        if isinstance(obj, dict):
            dump_envelopes.append(obj)
    if not dump_envelopes:
        return None

    if policy == "replace":
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            merged_path,
            sorted(dump_envelopes, key=_envelope_sort_key),
        )
        return None

    # upsert
    if not id_column:
        return "merge_policy='upsert' requires a non-empty merge_policy_upsert_id_column"
    for env in dump_envelopes:
        if env.get(id_column) in (None, ""):
            return (
                f"JSON upsert: envelope missing id column {id_column!r}: "
                f"keys={sorted(env.keys())}"
            )

    by_id: dict[str, dict] = {}
    if merged_path.exists():
        try:
            prior = json.loads(merged_path.read_text())
        except Exception as exc:
            return f"could not parse existing {merged_path}: {exc}"
        if isinstance(prior, list):
            for env in prior:
                if isinstance(env, dict) and env.get(id_column) not in (None, ""):
                    by_id[str(env[id_column])] = env
    for env in dump_envelopes:
        by_id[str(env[id_column])] = env

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        merged_path,
        sorted(by_id.values(), key=_envelope_sort_key),
    )
    return None


PaginateFn = Callable[[str, str, SyncBudget, Callable[[str], None], Path], bool]
"""Source-specific paginator.

Args (in order):
- window_start: ISO-8601 lower bound (exclusive in practice)
- window_end:   ISO-8601 upper bound (exclusive)
- budget:       check ``should_stop()`` between rows; bump ``rows_written``
                exactly once per durable row write
- advance_watermark(ts_iso): call after each successful row write with the
                row's authoritative timestamp (creation/update/added);
                used to set ``high_watermark`` on partial runs
- source_dir:   ``{dwh_dir}/sources/<source>/`` — write rows under ``raw/``

Returns True if the budget fired before the source was exhausted (the next
trigger will resume from the advanced ``high_watermark``).
"""


def run_sync(
    *,
    source: str,
    dwh_dir: str,
    max_runtime_seconds: int,
    paginate: PaginateFn,
) -> dict:
    """Run one sync cycle.

    Steps:
      1. Read (or initialize) ``{dwh_dir}/sources/<source>/sync-state.json``.
      2. Compute window = (max(low, high), now]; noop if empty.
      3. Hand off to ``paginate``; track latest_ts_written via the callback.
      4. On success: advance high_watermark = window_end (or latest_ts_written
         on partial run); persist state atomically.
      5. On exception: persist last_error, do NOT advance high, return error.

    Returns the uniform dict:
        {status, source, rows_written, window, watermarks, has_more, error}
    where status ∈ {"noop", "ok", "partial", "error"}.
    """
    dwh = Path(dwh_dir).expanduser().resolve()
    source_dir = dwh / "sources" / source
    state_path = source_dir / "sync-state.json"

    state = _read_state(state_path, source)
    low = state.get("low_watermark") or utcnow_iso()
    high = state.get("high_watermark") or utcnow_iso()
    window_start = max(low, high)
    window_end = utcnow_iso()

    if window_start >= window_end:
        return {
            "status": "noop",
            "source": source,
            "rows_written": 0,
            "window": [window_start, window_end],
            "watermarks": {"low": low, "high": high},
            "has_more": False,
            "error": None,
        }

    budget = SyncBudget(max_runtime_seconds)
    latest_ts_written = [window_start]  # mutable ref so callback can update

    def advance_watermark(ts_iso: str) -> None:
        if ts_iso and ts_iso > latest_ts_written[0]:
            latest_ts_written[0] = ts_iso

    try:
        has_more = paginate(
            window_start, window_end, budget, advance_watermark, source_dir
        )
    except Exception as exc:  # noqa: BLE001 — surface any failure to the caller
        err_msg = f"{type(exc).__name__}: {exc}"
        state["last_sync_at"] = utcnow_iso()
        state["last_sync_count"] = budget.rows_written
        state["last_error"] = err_msg
        # Do NOT advance high_watermark on error — next trigger retries
        atomic_write_json(state_path, state)
        return {
            "status": "error",
            "source": source,
            "rows_written": budget.rows_written,
            "window": [window_start, window_end],
            "watermarks": {"low": low, "high": high},
            "has_more": False,
            "error": err_msg,
        }

    # Advance high_watermark: full window if exhausted, else last row's ts
    if has_more:
        new_high = max(latest_ts_written[0], high)
    else:
        new_high = window_end

    state["high_watermark"] = new_high
    state["last_sync_at"] = utcnow_iso()
    state["last_sync_count"] = budget.rows_written
    state["last_error"] = None
    atomic_write_json(state_path, state)

    return {
        "status": "partial" if has_more else "ok",
        "source": source,
        "rows_written": budget.rows_written,
        "window": [window_start, window_end],
        "watermarks": {"low": low, "high": new_high},
        "has_more": has_more,
        "error": None,
    }
