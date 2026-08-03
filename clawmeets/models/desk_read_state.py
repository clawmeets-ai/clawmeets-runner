# SPDX-License-Identifier: MIT
"""
clawmeets/models/desk_read_state.py

Desk read-state store — the server-side memory of the My Desk **Sign Off**
action, so a card signed off on one device is signed off on every device.

Sign Off is the ONLY writer: ``dismiss()`` in ``UpdateCard.tsx`` is the single
caller of the client's ``markRead``. Send Back deliberately does not write here
(it leaves the card on the desk, re-laned to *Working*), so every row in this
store is a sign-off by construction — that is what lets the read side
(``models/desk_sign_off.py``) call itself "Signed off today" honestly.

Granularity is one watermark per desk card = per ``(project_id,
chatroom_name)`` (today ``chatroom_name == "user-communication"``). The value
stored is a *last-seen message-id watermark*; the sentinel
``"__dismissed__"`` marks an empty room (no real message to point at yet).
The row key is ``(owner_user_id, project_id, chatroom_name)`` — the owner
always comes from the JWT, never the request body, so it can't be spoofed.

Storage (mirrors ``desk-todos`` / ``brief-tabs`` — file-based JSON per user,
NOT SQL)::

    {data_dir}/desk-read-state/
      <owner_user_id>.json     # dict keyed "{project_id}|{chatroom_name}" -> row

Message ids are random ``uuid4()`` (see ``server/routes/messages.py``) and are
therefore NOT sortable, so the server cannot compute a "max" watermark. The
merge key is instead last-writer-wins on a *server-assigned* ``updated_at``
timestamp. ``updated_at`` is stamped INSIDE the per-user lock and forced to be
strictly increasing per user (``_monotonic_now``), so LWW is a total,
unambiguous order even for two writes that land in the same wall-clock
microsecond. The identical string is echoed in the PUT 200 body and in the
``DESK_READ_STATE_SYNC`` WS payload for the same write.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel

from clawmeets.utils.file_io import FileUtil

_lock = asyncio.Lock()

# Last ``updated_at`` stamped per owner, so consecutive serialized writes get a
# strictly increasing clock even inside the same microsecond. Bounded by the
# number of users on the server; survives a restart harmlessly (real wall-clock
# time has advanced far past any prior in-memory value).
_last_stamp: dict[str, str] = {}

READ_STATE_DIR = "desk-read-state"

# Watermark for a card whose room has no real message to mark against yet.
# Stored verbatim, never validated against real message ids — when a real
# message later arrives its id != the sentinel, so the client correctly
# re-surfaces the card (parity with today's localStorage behavior).
DISMISSED_SENTINEL = "__dismissed__"


class DeskReadState(BaseModel):
    """One desk card's read watermark for one owner."""

    owner_user_id: str          # from JWT; the row owner
    project_id: str             # UUID of the project the desk card belongs to
    chatroom_name: str          # e.g. "user-communication"
    last_seen_message_id: str   # watermark; DISMISSED_SENTINEL allowed
    updated_at: str             # ISO-8601 UTC, server-assigned (LWW clock)


class DeskReadStateUpsert(BaseModel):
    """PUT request body. No ``owner_user_id`` — it comes from the JWT."""

    project_id: str
    chatroom_name: str
    last_seen_message_id: str


def _key(project_id: str, chatroom_name: str) -> str:
    return f"{project_id}|{chatroom_name}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _monotonic_now(owner_user_id: str) -> str:
    """A per-user, strictly increasing ISO-8601 UTC timestamp.

    Must be called under ``_lock`` (writes serialize there, so the read /
    compare / store of ``_last_stamp`` is race-free). All timestamps share the
    same fixed ``+00:00`` UTC format, so lexicographic string comparison equals
    chronological comparison.
    """
    now = _now()
    last = _last_stamp.get(owner_user_id)
    if last is not None and now <= last:
        now = (datetime.fromisoformat(last) + timedelta(microseconds=1)).isoformat()
    _last_stamp[owner_user_id] = now
    return now


def _path(data_dir: Path, owner_user_id: str) -> Path:
    return Path(data_dir) / READ_STATE_DIR / f"{owner_user_id}.json"


def _load(data_dir: Path, owner_user_id: str) -> dict[str, DeskReadState]:
    raw = FileUtil.read(_path(data_dir, owner_user_id), "json")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, DeskReadState] = {}
    for key, row in raw.items():
        if not isinstance(row, dict):
            continue
        try:
            out[key] = DeskReadState.model_validate(row)
        except Exception:
            continue
    return out


def _save(data_dir: Path, owner_user_id: str, rows: dict[str, DeskReadState]) -> None:
    FileUtil.write(
        _path(data_dir, owner_user_id),
        {key: row.model_dump() for key, row in rows.items()},
        "json",
    )


async def list_read_state(data_dir: Path, owner_user_id: str) -> list[DeskReadState]:
    """Return all of the owner's watermarks (hydration GET). Loaded under the
    lock for a consistent snapshot against concurrent upserts."""
    async with _lock:
        return list(_load(data_dir, owner_user_id).values())


class SignOffWindow(BaseModel):
    """The answer to one sign-off query.

    Both fields are computed from a SINGLE snapshot, so the window and the
    all-time high can never disagree with each other.
    """

    rows: list[DeskReadState]        # inside the window, newest-first
    last_signed_off_at: str | None = None   # newest updated_at across ALL rows


def _parsed(row: DeskReadState) -> datetime:
    """``row.updated_at`` as an aware datetime; unparseable rows sort oldest.

    Every stamp this module writes is ``_monotonic_now``'s fixed ``+00:00``
    format, but a hand-edited / legacy file could hold anything, and a raise
    here would take the whole feed down for one bad row.
    """
    try:
        parsed = datetime.fromisoformat(row.updated_at)
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=UTC)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


async def list_sign_offs(
    data_dir: Path,
    owner_user_id: str,
    since: datetime | None = None,
) -> SignOffWindow:
    """All of the owner's sign-offs at or after ``since``, newest first.

    Also returns the newest ``updated_at`` across EVERY row (ignoring
    ``since``), so a caller can say "nothing today; last one was Tuesday"
    without a second query.

    ``since`` is an aware datetime and each row's ISO ``updated_at`` is PARSED
    and compared as a datetime. A string compare would be wrong the moment the
    caller's boundary is expressed in a non-UTC offset or with a ``Z`` suffix —
    lexicographic order only equals chronological order while every value
    shares this module's fixed ``+00:00`` format, and the boundary comes from
    the client. ``since=None`` means no window.

    No limit, no clamp, no skip — the caller windows. Read-only: it reuses
    ``_load()`` and never touches the monotonic clock. The snapshot is taken
    under ``_lock`` for consistency against a concurrent upsert.
    """
    async with _lock:
        rows = list(_load(data_dir, owner_user_id).values())
    rows.sort(key=_parsed, reverse=True)
    last = rows[0].updated_at if rows else None
    if since is not None:
        rows = [r for r in rows if _parsed(r) >= since]
    return SignOffWindow(rows=rows, last_signed_off_at=last)


async def upsert_read_state(
    data_dir: Path,
    owner_user_id: str,
    project_id: str,
    chatroom_name: str,
    last_seen_message_id: str,
) -> DeskReadState:
    """Idempotent upsert of ONE card's watermark.

    Re-PUTting the same ``(project_id, chatroom_name)`` overwrites the same
    dict key (no duplicate row) with a refreshed, strictly-later
    ``updated_at``. Returns the stored row — the exact object echoed to the PUT
    caller and carried in the ``DESK_READ_STATE_SYNC`` WS payload.
    """
    async with _lock:
        rows = _load(data_dir, owner_user_id)
        row = DeskReadState(
            owner_user_id=owner_user_id,
            project_id=project_id,
            chatroom_name=chatroom_name,
            last_seen_message_id=last_seen_message_id,
            updated_at=_monotonic_now(owner_user_id),
        )
        rows[_key(project_id, chatroom_name)] = row
        _save(data_dir, owner_user_id, rows)
        return row
