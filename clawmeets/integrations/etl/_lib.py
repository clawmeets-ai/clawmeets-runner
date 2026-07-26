# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/etl/_lib.py

Deterministic half of the ``etl`` skill: per-rule candidate selection over the
warehouse's delta+snapshot sources, and batch → derived-TSV merge. The LLM half
(applying ``rule.prompt`` to each candidate and emitting batch rows) stays in
the skill.

**Consumption model (warehouse v2).** Each source is a `raw/<path>/` dir holding
`snapshot.{ndjson,tsv}` + `deltas/<ts>.ndjson`. A rule consumes each source by a
log-positioned checkpoint stored in `derived/<rule>.sync-state.json`:

  - **bootstrap**: stream the current snapshot rows (paged by ``max_per_run`` via
    a row offset); record the latest delta filename present at bootstrap start as
    the boundary, since the snapshot already reflects every delta up to it.
  - **tail**: once the snapshot is consumed, read delta files whose name sorts
    after the boundary — new changes + tombstones (``{..., deleted: true}``).

A pruned checkpoint (the boundary delta aged out, see DELTA_RETENTION_DAYS) makes
the source re-bootstrap from the snapshot — correct, just a re-scan; flagged in
the result so the caller can surface it.

Source paths resolve under ``{dwh_dir}/raw/`` (e.g. ``gmail/inbox``); a
``derived/<rule>``-prefixed path reads a prior derived table (its own
snapshot/deltas), so rules can chain. The rule's own ``merge_policy`` /
``on_source_delete`` govern how candidates fold into the derived TSV.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from clawmeets.integrations._config_resolve import resolve_skill_config_path
from clawmeets.integrations._sync_warehouse import (
    atomic_write_json,
    atomic_write_text,
    list_deltas,
    load_snapshot,
    refresh_catalog,
    utcnow_iso,
    write_delta,
)
from clawmeets.utils.jsonc import parse_jsonc


DEFAULT_MAX_PER_RUN = 100
RULE_NAME_RE = re.compile(r"^[a-z0-9_-]+$")


# ───────────────────────── config + rule resolution ─────────────────────────

def load_config(config_file: str) -> tuple[Optional[dict], Optional[str]]:
    config_file = resolve_skill_config_path("etl", config_file)
    if not config_file:
        return None, (
            "No etl config yet. Open Agent Settings → Skills → etl → "
            "Configure to define rules."
        )
    path = Path(config_file).expanduser()
    if not path.exists():
        return None, (
            f"config file not found at {path} — save the etl config via "
            "the Configure modal in Agent Settings"
        )
    try:
        cfg = parse_jsonc(path.read_text())
    except Exception as exc:
        return None, f"config file is not valid JSON: {exc}"
    if not isinstance(cfg, dict):
        return None, "config file must contain a JSON object"
    rules = cfg.get("rules")
    if not isinstance(rules, list) or not rules:
        return None, "config file must list at least one rule under `rules`"
    return cfg, None


def resolve_rule(cfg: dict, rule_name: str) -> tuple[Optional[dict], Optional[str]]:
    """Look up + validate one rule; returns (rule-with-defaults, error)."""
    rules = {r.get("name"): r for r in cfg.get("rules", []) if isinstance(r, dict)}
    if rule_name not in rules:
        return None, (
            f"rule '{rule_name}' not found. available: {sorted(k for k in rules if k)}"
        )
    r = dict(rules[rule_name])

    errs: list[str] = []
    if not RULE_NAME_RE.match(r.get("name") or ""):
        errs.append("name must match [a-z0-9_-]+")
    sources = r.get("sources")
    if not isinstance(sources, list) or not sources:
        errs.append("sources empty")
    else:
        for s in sources:
            if not isinstance(s, dict) or not s.get("path"):
                errs.append("every source needs a `path`")
                break
    if not str(r.get("output") or "").endswith(".tsv"):
        errs.append("output must end in .tsv")
    if not r.get("columns"):
        errs.append("columns empty")
    if r.get("merge_policy") not in ("upsert", "replace"):
        errs.append("merge_policy must be upsert|replace")
    if r.get("merge_policy") == "upsert":
        if not r.get("key"):
            errs.append("upsert requires `key`")
        elif r["key"] not in (r.get("columns") or []):
            errs.append(f"key '{r['key']}' not in columns")
    on_del = r.get("on_source_delete")
    if on_del not in (None, "", "ignore") and on_del not in (r.get("columns") or []):
        errs.append(f"on_source_delete '{on_del}' not in columns")
    if errs:
        return None, "; ".join(errs)

    max_per_run = r.get("max_per_run")
    if not isinstance(max_per_run, int) or max_per_run <= 0:
        r["max_per_run"] = DEFAULT_MAX_PER_RUN
    return r, None


# ───────────────────────────── path helpers ─────────────────────────────────

def _candidates_path(dwh: Path, rule_name: str) -> Path:
    return dwh / "derived" / f".{rule_name}.candidates.json"


def _batch_path(dwh: Path, rule_name: str) -> Path:
    return dwh / "derived" / f".{rule_name}.batch.ndjson"


def _state_path(dwh: Path, rule_name: str) -> Path:
    return dwh / "derived" / f"{rule_name}.sync-state.json"


def _source_base(dwh: Path, path: str) -> Path:
    """A source path resolves under ``raw/`` (or ``raw/derived/<rule>`` is wrong —
    derived tables live at ``derived/<rule>`` with their own snapshot/deltas)."""
    if path.startswith("derived/"):
        return dwh / path
    return dwh / "raw" / path


def _snapshot_fmt(base: Path) -> str:
    return "tsv" if (base / "snapshot.tsv").exists() else "ndjson"


# ─────────────────────── per-source delta-log reader ────────────────────────

def _read_state(dwh: Path, rule_name: str) -> dict:
    path = _state_path(dwh, rule_name)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"rule": rule_name, "checkpoints": {}, "last_run_at": None,
            "last_run_count": 0, "last_run_error": None}


def _delta_rows(base: Path, fname: str) -> list[dict]:
    out: list[dict] = []
    for line in (base / "deltas" / fname).read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _pull_source(base: Path, checkpoint: dict, budget: int) -> tuple[list[dict], dict, list[str]]:
    """Pull up to ``budget`` candidate rows from one source past its checkpoint.

    Returns ``(rows, new_checkpoint, warnings)``. ``rows`` are dicts (tombstones
    carry ``deleted: True``). Checkpoint shape:
      - bootstrap: ``{phase: "bootstrap", offset, boundary}`` — offset into the
        snapshot; boundary = latest delta filename captured at bootstrap start.
      - tail: ``{phase: "tail", boundary, cursor_file, offset}`` — deltas named
        ``<= boundary`` are already reflected in the snapshot; ``cursor_file`` +
        ``offset`` mark the last delta row consumed past the boundary.
    """
    warnings: list[str] = []
    deltas = list_deltas(base / "deltas")
    phase = checkpoint.get("phase")

    # Pruned-past checkpoint → re-bootstrap (the snapshot is always current).
    if phase == "tail":
        boundary = checkpoint.get("boundary") or ""
        if boundary and deltas and boundary < deltas[0] and boundary not in deltas:
            warnings.append(
                f"checkpoint boundary {boundary!r} pruned past oldest delta "
                f"{deltas[0]!r} — re-bootstrapping from snapshot")
            phase = None

    if phase != "tail":
        # Bootstrap: stream snapshot rows from offset; capture the boundary once.
        snap = load_snapshot(base, _snapshot_fmt(base))
        if phase == "bootstrap":
            offset = int(checkpoint.get("offset") or 0)
            boundary = checkpoint.get("boundary") or ""
        else:
            offset = 0
            boundary = deltas[-1] if deltas else ""
        chunk = snap[offset:offset + budget]
        new_offset = offset + len(chunk)
        if new_offset >= len(snap):
            return chunk, {"phase": "tail", "boundary": boundary,
                           "cursor_file": "", "offset": 0}, warnings
        return chunk, {"phase": "bootstrap", "offset": new_offset,
                       "boundary": boundary}, warnings

    # Tail: consume delta files sorted after the boundary, resuming mid-file.
    boundary = checkpoint.get("boundary") or ""
    cursor_file = checkpoint.get("cursor_file") or ""
    offset = int(checkpoint.get("offset") or 0)
    rows: list[dict] = []
    remaining = budget
    new_ck = {"phase": "tail", "boundary": boundary,
              "cursor_file": cursor_file, "offset": offset}
    for fname in [d for d in deltas if d > boundary]:
        if fname < cursor_file:
            continue  # already fully consumed
        start = offset if fname == cursor_file else 0
        drows = _delta_rows(base, fname)
        avail = drows[start:start + remaining]
        rows.extend(avail)
        remaining -= len(avail)
        consumed = start + len(avail)
        if consumed >= len(drows):
            new_ck = {"phase": "tail", "boundary": fname,
                      "cursor_file": "", "offset": 0}
        else:
            new_ck = {"phase": "tail", "boundary": boundary,
                      "cursor_file": fname, "offset": consumed}
            break
        if remaining <= 0:
            break
    return rows, new_ck, warnings


# ───────────────────────────── load-candidates ──────────────────────────────

def load_candidates(dwh_dir: str, rule_name: str, config_file: str = "") -> dict:
    """Build the candidates file for one rule from its sources' delta logs."""
    def _err(msg: str) -> dict:
        return {"status": "error", "rule": rule_name, "error": msg}

    cfg, err = load_config(config_file)
    if err or cfg is None:
        return _err(err or "config unavailable")
    rule, err = resolve_rule(cfg, rule_name)
    if err or rule is None:
        return _err(err or "rule unavailable")

    dwh = Path(dwh_dir).expanduser().resolve()
    if not dwh.is_dir():
        return _err(f"dwh dir not found: {dwh}")

    state = _read_state(dwh, rule_name)
    checkpoints: dict = dict(state.get("checkpoints") or {})
    max_per_run = rule["max_per_run"]

    candidates: list[dict] = []
    warnings: list[str] = []
    next_checkpoints = dict(checkpoints)
    has_more = False

    for s in rule["sources"]:
        spath = s["path"]
        base = _source_base(dwh, spath)
        if not base.is_dir():
            warnings.append(f"missing source: {base}")
            continue
        remaining = max_per_run - len(candidates)
        if remaining <= 0:
            has_more = True
            break
        rows, new_ck, warns = _pull_source(base, checkpoints.get(spath) or {}, remaining)
        warnings.extend(f"{spath}: {w}" for w in warns)
        for row in rows:
            candidates.append({"source": spath, "row": row,
                               "deleted": bool(row.get("deleted"))})
        next_checkpoints[spath] = new_ck
        # If this source still has data past the new checkpoint, flag has_more.
        if _source_has_more(base, new_ck):
            has_more = True

    if not candidates:
        return {
            "status": "no_candidates", "rule": rule_name,
            "candidate_count": 0, "has_more": has_more,
            "warnings": warnings, "error": None,
        }

    (dwh / "derived").mkdir(parents=True, exist_ok=True)
    manifest = {
        "rule_name": rule["name"],
        "columns": rule["columns"],
        "merge_policy": rule["merge_policy"],
        "key": rule.get("key"),
        "on_source_delete": rule.get("on_source_delete") or "ignore",
        "candidates": candidates,
        "next_checkpoints": next_checkpoints,
        "has_more": has_more,
        "warnings": warnings,
    }
    atomic_write_json(_candidates_path(dwh, rule_name), manifest)
    _batch_path(dwh, rule_name).write_text("")  # truncate/create for appends

    return {
        "status": "ok", "rule": rule_name,
        "candidate_count": len(candidates), "has_more": has_more,
        "warnings": warnings,
        "candidates_path": str(_candidates_path(dwh, rule_name)),
        "batch_path": str(_batch_path(dwh, rule_name)),
        "columns": rule["columns"], "key": rule.get("key"),
        "on_source_delete": rule.get("on_source_delete") or "ignore",
        "prompt": rule.get("prompt") or "",
        "error": None,
    }


def _source_has_more(base: Path, ck: dict) -> bool:
    if ck.get("phase") == "bootstrap":
        return True
    boundary = ck.get("boundary") or ""
    cursor_file = ck.get("cursor_file") or ""
    offset = int(ck.get("offset") or 0)
    if cursor_file and offset < len(_delta_rows(base, cursor_file)):
        return True
    return any(d > boundary and d != cursor_file for d in list_deltas(base / "deltas"))


# ──────────────────────────────── merge ─────────────────────────────────────

def _validate_batch(batch_file: Path, columns: list[str]) -> tuple[Optional[list[dict]], Optional[str]]:
    new_rows: list[dict] = []
    if not batch_file.exists():
        return new_rows, None
    with open(batch_file) as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                return None, f"batch line {n}: bad json: {exc}"
            if not isinstance(row, dict):
                return None, f"batch line {n}: not a JSON object"
            extra = set(row) - set(columns)
            missing = set(columns) - set(row)
            if extra:
                return None, f"batch line {n}: extra keys {sorted(extra)}"
            if missing:
                return None, f"batch line {n}: missing keys {sorted(missing)}"
            clean = {}
            for c in columns:
                v = row[c]
                clean[c] = "" if v is None else (
                    str(v).replace("\t", " ").replace("\n", " ").replace("\r", " "))
            new_rows.append(clean)
    return new_rows, None


def _write_output_tsv(output: Path, columns: list[str], rows: list[dict]) -> None:
    import os
    import tempfile

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(output.parent), prefix=f".{output.name}.")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns, delimiter="\t",
                               quoting=csv.QUOTE_NONE, escapechar="\\")
            w.writeheader()
            for row in rows:
                w.writerow({c: row.get(c, "") for c in columns})
        os.replace(tmp, output)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def merge(dwh_dir: str, rule_name: str, config_file: str = "") -> dict:
    """Merge the batch into the derived TSV, apply source-deletes, advance
    checkpoints, and emit a derived delta. Any failure leaves state untouched."""
    def _err(msg: str) -> dict:
        return {"status": "error", "rule": rule_name, "error": msg}

    cfg, err = load_config(config_file)
    if err or cfg is None:
        return _err(err or "config unavailable")
    rule, err = resolve_rule(cfg, rule_name)
    if err or rule is None:
        return _err(err or "rule unavailable")

    dwh = Path(dwh_dir).expanduser().resolve()
    manifest_path = _candidates_path(dwh, rule_name)
    if not manifest_path.exists():
        return _err(
            f"no candidates manifest at {manifest_path} — run "
            f"`clawmeets etl load-candidates {rule_name}` first")
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        return _err(f"unreadable candidates manifest: {exc}")

    columns: list[str] = rule["columns"]
    policy: str = rule["merge_policy"]
    key: Optional[str] = rule.get("key")
    on_source_delete: str = manifest.get("on_source_delete") or "ignore"
    output = dwh / rule["output"]
    batch_file = _batch_path(dwh, rule_name)

    new_rows, err = _validate_batch(batch_file, columns)
    if err or new_rows is None:
        return _err(f"{err} — checkpoints NOT advanced; batch retained at {batch_file}")

    # Source-deletes: tombstoned candidates whose source id maps to derived rows.
    deleted_keys: set[str] = set()
    if on_source_delete != "ignore":
        for c in manifest.get("candidates", []):
            if c.get("deleted"):
                rid = c.get("row", {}).get("id")
                if rid is not None:
                    deleted_keys.add(str(rid))

    # Build the derived output.
    if policy == "replace":
        final = list(new_rows)
    else:
        existing: dict[str, dict] = {}
        if output.exists():
            with open(output, newline="") as f:
                rdr = csv.DictReader(f, delimiter="\t")
                if rdr.fieldnames != columns:
                    return _err(
                        f"existing TSV header {rdr.fieldnames} does not match "
                        f"configured columns {columns} — rm {output} to start fresh")
                for row in rdr:
                    existing[row[key]] = row
        # Apply source-deletes against the on_source_delete column.
        removed = 0
        if on_source_delete != "ignore" and deleted_keys:
            for k in list(existing):
                if str(existing[k].get(on_source_delete)) in deleted_keys:
                    del existing[k]
                    removed += 1
        for r in new_rows:
            existing[r[key]] = r
        final = list(existing.values())

    try:
        _write_output_tsv(output, columns, final)
    except OSError as exc:
        return _err(f"TSV write failed: {exc} — checkpoints NOT advanced")

    # Howto mirror (hash-gated).
    state = _read_state(dwh, rule_name)
    prior_hash = state.get("howto_hash")
    howto_text = rule.get("howto") or ""
    howto_written = False
    howto_hash: Optional[str] = None
    if howto_text:
        howto_hash = hashlib.sha256(howto_text.encode("utf-8")).hexdigest()
        howto_path = dwh / "derived" / f"{rule_name}.howto.md"
        if howto_hash != prior_hash or not howto_path.exists():
            atomic_write_text(howto_path, howto_text)
            howto_written = True

    # Emit a derived delta so downstream rules can consume this table as a source.
    delta_rows = list(new_rows)
    if on_source_delete != "ignore" and deleted_keys:
        for k in deleted_keys:
            delta_rows.append({"id": k, "deleted": True})
    write_delta(output.parent / f"{output.stem}.deltas", delta_rows)

    # Advance checkpoints + state.
    new_state = {
        "rule": rule_name,
        "checkpoints": manifest.get("next_checkpoints") or {},
        "last_run_at": utcnow_iso(),
        "last_run_count": len(new_rows),
        "last_run_error": None,
        "howto_hash": howto_hash,
    }
    try:
        atomic_write_json(_state_path(dwh, rule_name), new_state)
    except OSError as exc:
        return _err(f"state write failed after writing {len(new_rows)} rows to {output}: {exc}")

    batch_file.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)

    # Register the freshly-written derived table in the warehouse catalog so the
    # agent's discovery index reflects it on the next turn.
    refresh_catalog(dwh)

    has_more = bool(manifest.get("has_more"))
    summary = (
        f"Rule `{rule_name}`: {len(manifest.get('candidates', []))} candidates → "
        f"{len(new_rows)} output rows (final {len(final)}). "
        f"has_more={'true' if has_more else 'false'}")
    return {
        "status": "ok", "rule": rule_name,
        "appended_count": len(new_rows), "final_count": len(final),
        "has_more": has_more, "howto_written": howto_written,
        "output": str(output), "summary": summary, "error": None,
    }
