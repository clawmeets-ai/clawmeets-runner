# SPDX-License-Identifier: MIT
"""
clawmeets/models/desk_sop.py

Desk SOP store — the My Desk right-rail "SOP" library: stored, reusable
prompts the manager hands to an agent over and over.

An SOP body carries typed blanks written ``{{Label|kind:config}}`` (see
``web/frontend/src/components/composer/placeholders.ts`` for the syntax).
Clicking one in the rail loads it into the desk composer as fillable chips,
already addressed to ``agent_id``.

Unlike the to-do plate this list is *searched*, not dragged — so there is no
reorder verb and no ordering contract beyond "newest first". It is still one
ordered JSON document per user, because that keeps a write a single
whole-file rewrite under one lock.

Storage::

    {data_dir}/desk-sops/
      <owner_user_id>.json     # list[DeskSop], newest first

Mutations are broadcast to every live session the owner holds via
``DESK_SOP_SYNC`` (see ``server/routes/desk_sops.py``), so a library edited
in one browser tab updates in all the others without a reload.
"""
from __future__ import annotations

import asyncio
import secrets
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from clawmeets.utils.file_io import FileUtil

_lock = asyncio.Lock()

SOPS_DIR = "desk-sops"

# Longest body we will store. Generous for a prompt template, small enough
# that a runaway paste can't bloat the owner's single JSON document.
MAX_BODY_CHARS = 16_384

# Distinguishes "key absent from a PATCH body" (leave the stored value
# untouched) from an explicit JSON ``null`` (clear the field). Used only for
# the ``agent_*`` pair: a user with no agents saves an SOP with no recipient,
# and later assigning/unassigning one needs absent != null. The route layer
# detects key presence off the raw request body (see
# ``server/routes/desk_sops.py``) since a non-serializable sentinel can't be
# a FastAPI ``Body`` default. Mirrors ``desk_todo._UNSET``.
_UNSET: object = object()


class DeskSop(BaseModel):
    """One stored prompt in the owner's SOP library."""

    id: str
    owner_user_id: str
    title: str
    desc: str = ""
    # Default recipient. Both id and name are persisted so the rail still
    # labels the card correctly after a rename or removal; the frontend
    # re-resolves against the live roster (``utils/sopRecipient.ts``) and
    # falls back to the owner's assistant. Same precedent as
    # ``DeskTodo.draft_recipient_{id,name}``.
    agent_id: str | None = None
    agent_name: str | None = None
    # The prompt itself, with ``{{blanks}}`` left un-substituted.
    body: str
    created_at: str
    updated_at: str


def gen_id() -> str:
    return "sop-" + secrets.token_hex(6)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _path(data_dir: Path, owner_user_id: str) -> Path:
    return Path(data_dir) / SOPS_DIR / f"{owner_user_id}.json"


def _load(data_dir: Path, owner_user_id: str) -> list[DeskSop]:
    raw = FileUtil.read(_path(data_dir, owner_user_id), "json")
    if not isinstance(raw, list):
        return []
    out: list[DeskSop] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            out.append(DeskSop.model_validate(row))
        except Exception:
            continue
    return out


def _save(data_dir: Path, owner_user_id: str, sops: list[DeskSop]) -> None:
    FileUtil.write(
        _path(data_dir, owner_user_id),
        [s.model_dump() for s in sops],
        "json",
    )


def list_sops(data_dir: Path, owner_user_id: str) -> list[DeskSop]:
    """Return the owner's library, newest first."""
    return _load(data_dir, owner_user_id)


def get_sop(data_dir: Path, owner_user_id: str, sop_id: str) -> DeskSop | None:
    for s in _load(data_dir, owner_user_id):
        if s.id == sop_id:
            return s
    return None


async def create_sop(
    data_dir: Path,
    owner_user_id: str,
    *,
    title: str,
    body: str,
    desc: str = "",
    agent_id: str | None = None,
    agent_name: str | None = None,
) -> DeskSop:
    """Store a new SOP; prepended so the newest is first."""
    title = (title or "").strip()
    if not title:
        raise ValueError("SOP title cannot be empty")
    body = (body or "").strip()
    if not body:
        raise ValueError("SOP body cannot be empty")
    if len(body) > MAX_BODY_CHARS:
        raise ValueError(f"SOP body exceeds {MAX_BODY_CHARS} characters")
    async with _lock:
        sops = _load(data_dir, owner_user_id)
        now = _now()
        sop = DeskSop(
            id=gen_id(),
            owner_user_id=owner_user_id,
            title=title,
            desc=(desc or "").strip(),
            agent_id=agent_id,
            agent_name=agent_name,
            body=body,
            created_at=now,
            updated_at=now,
        )
        sops.insert(0, sop)
        _save(data_dir, owner_user_id, sops)
        return sop


async def patch_sop(
    data_dir: Path,
    owner_user_id: str,
    sop_id: str,
    *,
    title: str | None = None,
    desc: str | None = None,
    body: str | None = None,
    agent_id: str | None = _UNSET,  # _UNSET → leave untouched
    agent_name: str | None = _UNSET,  # None → clear, str → set
) -> DeskSop | None:
    """Patch an SOP in place. Returns the updated SOP, or None if missing.

    ``title`` / ``body`` never clear: a blank value is ignored, because both
    are required and the dialog already blocks Save on either being empty.
    ``desc`` DOES clear on an empty string — it is optional, so "the user
    deleted the description" is a real intent. ``agent_id`` / ``agent_name``
    are three-way (see ``_UNSET``)."""
    if body is not None and len(body) > MAX_BODY_CHARS:
        raise ValueError(f"SOP body exceeds {MAX_BODY_CHARS} characters")
    async with _lock:
        sops = _load(data_dir, owner_user_id)
        target: DeskSop | None = None
        for s in sops:
            if s.id == sop_id:
                target = s
                break
        if target is None:
            return None
        if title is not None:
            title = title.strip()
            if title:
                target.title = title
        if body is not None:
            body = body.strip()
            if body:
                target.body = body
        if desc is not None:
            target.desc = desc.strip()
        if agent_id is not _UNSET:
            target.agent_id = agent_id
        if agent_name is not _UNSET:
            target.agent_name = agent_name
        target.updated_at = _now()
        _save(data_dir, owner_user_id, sops)
        return target


async def delete_sop(data_dir: Path, owner_user_id: str, sop_id: str) -> bool:
    """Remove an SOP. Returns True if it existed."""
    async with _lock:
        sops = _load(data_dir, owner_user_id)
        kept = [s for s in sops if s.id != sop_id]
        if len(kept) == len(sops):
            return False
        _save(data_dir, owner_user_id, kept)
        return True
