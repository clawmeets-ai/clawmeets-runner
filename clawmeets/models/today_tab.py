# SPDX-License-Identifier: MIT
"""
clawmeets/models/today_tab.py

Today-tab registry — owner-scoped, per-tab JSON artifacts that the
``/today`` frontend composes into a tab strip.

A today tab is the output of one agent's run of the ``today`` skill:
the agent gathers data, writes a short render JS body, and shells
``clawmeets today upsert-tab <slug>`` to upload both. The server stores
the bundle keyed by ``(owner_user_id, slug)`` and pushes a
``TODAY_TAB_SYNC`` envelope so the owner's browser refetches.

Storage::

    {data_dir}/today-tabs/
      <owner_user_id>/
        <slug>.json          # one tab artifact
        ...

One file per tab so individual upserts/deletes don't fight for a single
JSON document. Slug is validated identically to knowledge packs.
"""
from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from clawmeets.utils.file_io import FileUtil

_lock = asyncio.Lock()

TABS_DIR = "today-tabs"

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class TodayTab(BaseModel):
    """A single today tab.

    ``data`` is opaque JSON the render body consumes. ``render_code_js``
    is the body of ``function(mount, data, lib)`` — no signature, no
    surrounding braces. Both are populated by the publishing agent on
    every refresh; the frontend never edits either.
    """

    slug: str
    title: str = ""
    owner_user_id: str
    owner_agent_id: str
    owner_agent_name: str
    data: dict | list = Field(default_factory=dict)
    render_code_js: str = ""
    generated_at: str


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


def _tab_path(data_dir: Path, owner_user_id: str, slug: str) -> Path:
    return _user_dir(data_dir, owner_user_id) / f"{validate_slug(slug)}.json"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_tab(path: Path) -> TodayTab | None:
    data = FileUtil.read(path, "json")
    if not isinstance(data, dict):
        return None
    try:
        return TodayTab.model_validate(data)
    except Exception:
        return None


def get_tab(data_dir: Path, owner_user_id: str, slug: str) -> TodayTab | None:
    return _load_tab(_tab_path(data_dir, owner_user_id, slug))


def list_tabs(data_dir: Path, owner_user_id: str) -> list[TodayTab]:
    """Return every persisted tab for the given user. Newest first
    (sorted by ``generated_at`` descending)."""
    user_dir = _user_dir(data_dir, owner_user_id)
    if not user_dir.is_dir():
        return []
    tabs: list[TodayTab] = []
    for entry in user_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".json":
            continue
        tab = _load_tab(entry)
        if tab is not None:
            tabs.append(tab)
    tabs.sort(key=lambda t: t.generated_at, reverse=True)
    return tabs


async def upsert_tab(
    data_dir: Path,
    owner_user_id: str,
    owner_agent_id: str,
    owner_agent_name: str,
    slug: str,
    title: str,
    data: dict | list,
    render_code_js: str,
) -> TodayTab:
    """Create or replace a tab. Always succeeds (overwrite is the
    contract — the publishing agent owns the slug and re-runs on
    every refresh)."""
    slug = validate_slug(slug)
    async with _lock:
        tab = TodayTab(
            slug=slug,
            title=(title or "").strip() or slug,
            owner_user_id=owner_user_id,
            owner_agent_id=owner_agent_id,
            owner_agent_name=owner_agent_name,
            data=data,
            render_code_js=render_code_js,
            generated_at=_now(),
        )
        FileUtil.write(
            _tab_path(data_dir, owner_user_id, slug),
            tab.model_dump(),
            "json",
        )
        return tab


async def delete_tab(data_dir: Path, owner_user_id: str, slug: str) -> bool:
    """Delete a tab. Returns True if it existed."""
    slug = validate_slug(slug)
    async with _lock:
        path = _tab_path(data_dir, owner_user_id, slug)
        if not path.is_file():
            return False
        path.unlink()
        return True
