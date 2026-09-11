# SPDX-License-Identifier: MIT
"""
clawmeets/models/brief_tab.py

Brief-tab registry — owner-scoped briefing artifacts that the My Desk
"Scheduled briefings" section composes into a tab strip.

A brief tab is the output of one agent's run of the ``brief`` skill: the
agent writes ONE complete ``<!doctype html>`` document and shells
``clawmeets brief upsert-tab <slug> --html briefing.html`` to upload it.
The server stores the document keyed by ``(owner_user_id, slug)`` and
pushes a ``BRIEF_TAB_SYNC`` cursor so the owner's browser refetches.

The document is stored and returned **byte-verbatim**. The server never
parses, sanitizes, re-serializes or minifies it, and never injects
anything into it — the desk's height bootstrap is injected client-side,
immediately before the frame is mounted, and is neither stored here nor
counted against :data:`MAX_BRIEF_HTML_BYTES`.

Storage — the artifact is a PAIR::

    {data_dir}/brief-tabs/
      <owner_user_id>/
        <slug>.json          # BriefTabMeta — metadata sidecar
        <slug>.html          # the document, verbatim bytes
        ...

Split so that listing never opens a document: the desk loads a metadata
strip and fetches one body only when the user opens a briefing.

One pair per tab so individual upserts/deletes don't fight for a single
JSON document. Slug is validated identically to knowledge packs.
"""
from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from clawmeets.utils.file_io import FileUtil

_lock = asyncio.Lock()

TABS_DIR = "brief-tabs"

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# The cap. ONE number, imported by the route AND by cli_brief.py so a
# publishing agent fails locally, before the upload, with the same byte
# count the server's 413 would have named.
#
# Measured on ``len(html.encode("utf-8"))`` — the author's own bytes.
# Never the JSON envelope the document rides inside, and never the
# ~2 KB client-side height bootstrap (which the server neither injects
# nor stores, so the byte-verbatim guarantee stays absolute and the
# bootstrap costs an author nothing).
MAX_BRIEF_HTML_BYTES = 262_144  # 256 KB


class BriefTab(BaseModel):
    """A single brief tab — metadata plus the document.

    ``html`` is one complete ``<!doctype html>`` document, UTF-8, stored
    and returned byte-verbatim. It replaced the old ``data`` +
    ``render_code_js`` pair: a briefing now brings its own typography,
    colour and layout and the desk imposes none of its own.

    ``title`` is a separate stored field and the document's own
    ``<title>`` is ignored — one string, one source of truth.
    """

    slug: str
    title: str = ""
    owner_user_id: str
    owner_agent_id: str
    owner_agent_name: str
    html: str
    generated_at: str


class BriefTabMeta(BaseModel):
    """The ``<slug>.json`` sidecar — everything except the document.

    ``html_bytes`` is REQUIRED, and that is load-bearing: it is the
    legacy discriminator. A pre-standalone-HTML ``<slug>.json`` carries
    slug, title, the three owner fields and ``generated_at`` — every
    field a metadata model needs — plus ``data`` and ``render_code_js``,
    which Pydantic ignores as unknown keys. Such a record would validate
    cleanly against a model without ``html_bytes``, list, and then 404
    the moment the user clicked it. Requiring a key no legacy record
    carries is what actually makes "skip them, delete nothing" work.

    The value itself is advisory only — ``<slug>.html`` is the truth
    about the body — and it never reaches the wire: :meth:`to_wire`
    strips it, so the list response is exactly the six agreed fields.
    """

    slug: str
    title: str = ""
    owner_user_id: str
    owner_agent_id: str
    owner_agent_name: str
    generated_at: str
    html_bytes: int

    def to_wire(self) -> dict:
        """The six fields the list route puts on the wire. No ``html``
        (that is what the per-slug route is for), no ``html_bytes``."""
        return self.model_dump(exclude={"html_bytes"})


def validate_slug(slug: str) -> str:
    """Normalize and validate a tab slug. Raises ValueError if invalid."""
    cleaned = (slug or "").strip().lower()
    if not cleaned:
        raise ValueError("Tab slug cannot be empty")
    if len(cleaned) > 80:
        raise ValueError("Tab slug must be 80 characters or fewer")
    if not _SLUG_RE.match(cleaned):
        raise ValueError(
            "Tab slug must start with [a-z0-9] and contain only lowercase "
            "letters, digits, hyphens, and underscores"
        )
    return cleaned


def _user_dir(data_dir: Path, owner_user_id: str) -> Path:
    return Path(data_dir) / TABS_DIR / owner_user_id


def _meta_path(data_dir: Path, owner_user_id: str, slug: str) -> Path:
    return _user_dir(data_dir, owner_user_id) / f"{validate_slug(slug)}.json"


def _html_path(data_dir: Path, owner_user_id: str, slug: str) -> Path:
    return _user_dir(data_dir, owner_user_id) / f"{validate_slug(slug)}.html"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_meta(meta_path: Path) -> BriefTabMeta | None:
    """Read one sidecar. Returns None — never raises, never 500s the
    caller — when the record is not a live standalone-HTML briefing.

    Two ways that happens, and both mean *absent*, not *error*:

    1. ``model_validate`` fails. ``html_bytes`` is required and no
       legacy record has it, so every pre-migration ``<slug>.json`` is
       skipped right here. Nothing is deleted and nothing is migrated.
    2. The ``<slug>.html`` sibling is missing. Metadata pointing at a
       body that isn't there is not a briefing; skipping it guarantees
       every row the list returns can actually be opened.
    """
    data = FileUtil.read(meta_path, "json")
    if not isinstance(data, dict):
        return None
    try:
        meta = BriefTabMeta.model_validate(data)
    except Exception:
        return None
    if not meta_path.with_suffix(".html").is_file():
        return None
    return meta


def get_meta(data_dir: Path, owner_user_id: str, slug: str) -> BriefTabMeta | None:
    """Sidecar only — for callers that need ownership or the ETag and
    not the body: the DELETE ownership probe, and the If-None-Match arm
    of the per-slug GET. Neither should pull 256 KB off disk."""
    return _load_meta(_meta_path(data_dir, owner_user_id, slug))


def get_tab(data_dir: Path, owner_user_id: str, slug: str) -> BriefTab | None:
    """Sidecar + document.

    Reads the body as BYTES and decodes UTF-8 explicitly. NOT
    ``FileUtil.read(..., "text")``: ``Path.read_text`` opens in
    universal-newline mode, which rewrites ``\\r\\n`` and lone ``\\r``
    to ``\\n``. That would silently break the byte-verbatim guarantee —
    green tests, wrong bytes — and is exactly what the CRLF round-trip
    test asserts against.

    Returns None if either half is absent or the body is not valid UTF-8.
    """
    meta = get_meta(data_dir, owner_user_id, slug)
    if meta is None:
        return None
    raw = FileUtil.read(_html_path(data_dir, owner_user_id, slug), "bytes")
    if raw is None:
        return None
    try:
        html = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return BriefTab(
        slug=meta.slug,
        title=meta.title,
        owner_user_id=meta.owner_user_id,
        owner_agent_id=meta.owner_agent_id,
        owner_agent_name=meta.owner_agent_name,
        html=html,
        generated_at=meta.generated_at,
    )


def list_metas(data_dir: Path, owner_user_id: str) -> list[BriefTabMeta]:
    """Every live briefing for one user, newest first (by
    ``generated_at`` descending).

    Reads only the small sidecars — never opens a ``.html``. Iterates
    ``suffix == ".json"``, so bodies and any stale ``.tmp`` are
    invisible, and a legacy artifact is absent from the result rather
    than an error: ``_load_meta`` swallows it and the caller never
    learns it existed.
    """
    user_dir = _user_dir(data_dir, owner_user_id)
    if not user_dir.is_dir():
        return []
    metas: list[BriefTabMeta] = []
    for entry in user_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".json":
            continue
        meta = _load_meta(entry)
        if meta is not None:
            metas.append(meta)
    metas.sort(key=lambda m: m.generated_at, reverse=True)
    return metas


async def upsert_tab(
    data_dir: Path,
    owner_user_id: str,
    owner_agent_id: str,
    owner_agent_name: str,
    slug: str,
    title: str,
    html: str,
) -> BriefTab:
    """Create or replace a tab. Always succeeds (overwrite is the
    contract — the publishing agent owns the slug and re-runs on every
    scheduled refresh; ``generated_at`` is restamped on every write,
    which is what keeps the tile's last-refreshed-at honest).

    Write order is load-bearing: ``.html`` FIRST and ``.json`` SECOND,
    both through FileUtil's atomic temp-and-rename, both inside the
    module lock. A crash between the two leaves a body with no sidecar,
    which ``_load_meta`` reads as absent. The reverse order would leave
    a sidecar pointing at a body that isn't there — a tile that 404s
    when clicked.

    Stores ``html.encode("utf-8")`` as raw bytes. Does NOT check the
    cap: the cap is a request-admission concern and lives at the route,
    the only layer that can answer it with a status code.
    """
    slug = validate_slug(slug)
    body = html.encode("utf-8")
    async with _lock:
        generated_at = _now()
        FileUtil.write(
            _html_path(data_dir, owner_user_id, slug),
            body,
            "bytes",
        )
        meta = BriefTabMeta(
            slug=slug,
            title=(title or "").strip() or slug,
            owner_user_id=owner_user_id,
            owner_agent_id=owner_agent_id,
            owner_agent_name=owner_agent_name,
            generated_at=generated_at,
            html_bytes=len(body),
        )
        FileUtil.write(
            _meta_path(data_dir, owner_user_id, slug),
            meta.model_dump(),
            "json",
        )
        return BriefTab(
            slug=meta.slug,
            title=meta.title,
            owner_user_id=meta.owner_user_id,
            owner_agent_id=meta.owner_agent_id,
            owner_agent_name=meta.owner_agent_name,
            html=html,
            generated_at=meta.generated_at,
        )


async def delete_tab(data_dir: Path, owner_user_id: str, slug: str) -> bool:
    """Delete both halves of the artifact. Returns True if the sidecar
    existed.

    Unlink order mirrors the write: ``.json`` FIRST, ``.html`` SECOND,
    so a crash mid-delete leaves an orphan body — invisible to every
    reader — rather than a sidecar with no body.
    """
    slug = validate_slug(slug)
    async with _lock:
        meta_path = _meta_path(data_dir, owner_user_id, slug)
        html_path = _html_path(data_dir, owner_user_id, slug)
        existed = meta_path.is_file()
        if existed:
            meta_path.unlink()
        if html_path.is_file():
            html_path.unlink()
        return existed
