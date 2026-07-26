# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/website_monitor/_lib.py

Deterministic half of the ``website-monitor`` skill: config validation, URL
canonicalization, content-hash dedup, first_seen_at preservation, and the
batch → snapshot+delta merge. The LLM half (the smart crawl: navigate from the
entry URL, judge items against ``content_of_interest``, emit
``{title, summary, source_url}`` rows) stays in the skill.

**Warehouse-v2 fit.** A crawl is bounded (``max_pages`` / ``max_per_run``), so
absence of an item from a run does NOT mean it was removed upstream — this
source is **incremental-upsert with NO tombstones**. There is no time-cursor
(the LLM crawls the web, not a windowed API), so it doesn't use the
``run_slice_sync`` driver; instead it folds a crawl batch into
``{dwh}/raw/website/<rule>/`` directly via the shared primitives, exactly like
gdrive's sheet-tab source.

Row id is ``content_hash = sha256(normalized_url)[:24]``. The snapshot is a
6-column TSV (``content_hash, title, summary, source_url, first_seen_at,
crawled_at``); each run's new / content-changed rows are appended to the delta
log so downstream etl rules can consume ``raw/website/<rule>`` like any source.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from clawmeets.integrations._config_resolve import resolve_skill_config_path
from clawmeets.integrations._sync_warehouse import (
    diff_snapshot,
    load_snapshot,
    prune_deltas,
    read_sync_state,
    utcnow_iso,
    write_delta,
    write_howto,
    write_snapshot,
    write_sync_state,
)
from clawmeets.utils.jsonc import parse_jsonc


RULE_NAME_RE = re.compile(r"^[a-z0-9_-]+$")
COLUMNS = ["content_hash", "title", "summary", "source_url", "first_seen_at", "crawled_at"]
EMIT_KEYS = {"title", "summary", "source_url"}
_TRACKING = re.compile(r"^(utm_|ref$|fbclid$|gclid$|mc_eid$|mc_cid$)")
DEFAULT_MAX_PER_RUN = 50
DEFAULT_MAX_PAGES = 20


# ───────────────────────── config + rule resolution ─────────────────────────

def load_config(config_file: str) -> tuple[Optional[dict], Optional[str]]:
    config_file = resolve_skill_config_path("website-monitor", config_file)
    if not config_file:
        return None, (
            "No website-monitor config yet. Open Agent Settings → Skills → "
            "website-monitor → Configure to define rules."
        )
    path = Path(config_file).expanduser()
    if not path.exists():
        return None, (
            f"config file not found at {path} — save the website-monitor config "
            "via the Configure modal in Agent Settings"
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
    website = r.get("website")
    if not isinstance(website, str) or not re.match(r"^https?://", website.strip()):
        errs.append("website must be a non-empty http(s):// URL")
    if not isinstance(r.get("content_of_interest"), str) or not r["content_of_interest"].strip():
        errs.append("content_of_interest must be a non-empty string")
    if errs:
        return None, "; ".join(errs)

    mpr = r.get("max_per_run")
    r["max_per_run"] = mpr if isinstance(mpr, int) and mpr > 0 else DEFAULT_MAX_PER_RUN
    mp = r.get("max_pages")
    r["max_pages"] = mp if isinstance(mp, int) and mp > 0 else DEFAULT_MAX_PAGES
    return r, None


def _normalize_url(u: str) -> str:
    """Strip tracking params + lowercase host so the same item hashes stably."""
    p = urlsplit(u.strip())
    if not p.scheme or not p.netloc:
        return u
    host = p.netloc.lower()
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
         if not _TRACKING.match(k)]
    return urlunsplit((p.scheme, host, p.path, urlencode(q), p.fragment))


# ───────────────────────────── path helpers ─────────────────────────────────

def _base(dwh: Path, rule_name: str) -> Path:
    return dwh / "raw" / "website" / rule_name


def _batch_path(dwh: Path, rule_name: str) -> Path:
    return _base(dwh, rule_name) / ".batch.ndjson"


# ──────────────────────────────── begin ─────────────────────────────────────

def begin(dwh_dir: str, rule_name: str, config_file: str = "") -> dict:
    """Validate the rule, mirror its howto, and open a fresh batch file for the
    crawl to append ``{title, summary, source_url}`` lines to."""
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
    base = _base(dwh, rule_name)
    base.mkdir(parents=True, exist_ok=True)
    write_howto(rule.get("howto"), snapshot_dir=base)
    _batch_path(dwh, rule_name).write_text("")  # truncate/create

    return {
        "status": "ok", "rule": rule_name,
        "batch_path": str(_batch_path(dwh, rule_name)),
        "entry_url": rule["website"],
        "content_of_interest": rule["content_of_interest"],
        "max_per_run": rule["max_per_run"],
        "max_pages": rule["max_pages"],
        "error": None,
    }


# ──────────────────────────────── merge ─────────────────────────────────────

def _read_batch(batch_file: Path) -> tuple[Optional[list[dict]], int, Optional[str]]:
    """Parse + validate the crawl batch. Returns (rows, input_count, error)."""
    rows: list[dict] = []
    input_count = 0
    if not batch_file.exists():
        return rows, 0, None
    for n, line in enumerate(batch_file.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        input_count += 1
        try:
            row = json.loads(line)
        except Exception as exc:
            return None, input_count, f"batch line {n}: bad json: {exc}"
        if not isinstance(row, dict):
            return None, input_count, f"batch line {n}: not a JSON object"
        extra = set(row) - EMIT_KEYS
        missing = EMIT_KEYS - set(row)
        if extra:
            return None, input_count, f"batch line {n}: extra keys {sorted(extra)}"
        if missing:
            return None, input_count, f"batch line {n}: missing keys {sorted(missing)}"
        rows.append(row)
    return rows, input_count, None


def merge(dwh_dir: str, rule_name: str, config_file: str = "") -> dict:
    """Fold the crawl batch into the snapshot + delta log. first_seen_at is
    preserved across runs; nothing is ever tombstoned (a crawl is partial)."""
    def _err(msg: str) -> dict:
        return {"status": "error", "rule": rule_name, "error": msg}

    cfg, err = load_config(config_file)
    if err or cfg is None:
        return _err(err or "config unavailable")
    rule, err = resolve_rule(cfg, rule_name)
    if err or rule is None:
        return _err(err or "rule unavailable")

    dwh = Path(dwh_dir).expanduser().resolve()
    base = _base(dwh, rule_name)
    batch_file = _batch_path(dwh, rule_name)

    batch_rows, input_count, err = _read_batch(batch_file)
    if err or batch_rows is None:
        return _err(f"{err} — state NOT advanced; batch retained at {batch_file}")

    now = utcnow_iso()
    prior = load_snapshot(base, "tsv")
    prior_by_id = {str(r["content_hash"]): r for r in prior if r.get("content_hash")}
    first_seen_by_hash = {h: r.get("first_seen_at") or now for h, r in prior_by_id.items()}

    # Build full rows; dedup within-run by content_hash (last wins).
    by_hash: dict[str, dict] = {}
    for row in batch_rows:
        url = _normalize_url(str(row["source_url"]))
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        by_hash[h] = {
            "content_hash":  h,
            "title":         _clean(row["title"]),
            "summary":       _clean(row["summary"]),
            "source_url":    url,
            "first_seen_at": first_seen_by_hash.get(h, now),
            "crawled_at":    now,
        }
    rows = list(by_hash.values())

    # No tombstones (in_scope=False ⇒ un-recrawled prior rows are preserved);
    # crawled_at is volatile so an unchanged re-crawl isn't a spurious delta.
    changed, _tombstones, snapshot = diff_snapshot(
        prior_by_id, rows, id_field="content_hash", ts_field="first_seen_at",
        in_scope=lambda _r: False, volatile_fields={"crawled_at"})
    new_count = sum(1 for r in rows if r["content_hash"] not in prior_by_id)

    try:
        write_delta(base / "deltas", changed)
        write_snapshot(base, snapshot, "tsv",
                       id_field="content_hash", ts_field="first_seen_at",
                       tsv_columns=COLUMNS)
    except OSError as exc:
        return _err(f"snapshot/delta write failed: {exc} — state NOT advanced")

    write_sync_state(base / "sync-state.json", {
        "rule": rule_name,
        "last_run_at": now,
        "last_run_input_count": input_count,
        "last_run_count": len(rows),
        "last_run_new_count": new_count,
        "last_run_error": None,
    })
    prune_deltas(base / "deltas")
    batch_file.unlink(missing_ok=True)

    summary = (
        f"Rule `{rule_name}`: {input_count} crawled → {len(rows)} kept "
        f"({new_count} new). Snapshot {len(snapshot)} rows total.")
    return {
        "status": "ok", "rule": rule_name,
        "kept_count": len(rows), "new_count": new_count,
        "final_count": len(snapshot), "changed_count": len(changed),
        "summary": summary, "error": None,
    }


def _clean(v) -> str:
    return str(v).replace("\t", " ").replace("\n", " ").replace("\r", " ")
