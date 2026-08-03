# SPDX-License-Identifier: MIT
"""
clawmeets/models/desk_sign_off.py

Card-assembly (hydration) service for the My Desk "Signed off today" section.

"Signed off today" is a **read-only** view over the existing ``DeskReadState``
store (``models/desk_read_state.py``): the cards the user hit **Sign Off** on
since a caller-supplied boundary, most-recently-signed-off first. There is no
new table, model, or migration — ``list_sign_offs`` reads the same per-user
watermark file the Sign Off write path already fills, and every row in it is a
sign-off by construction (Send Back no longer writes one).

Each watermark row carries only ``(project_id, chatroom_name,
last_seen_message_id, updated_at, ...)``. This module hydrates a row into a
renderable ``SignOffCard`` by loading the ``Project`` and reading the message
the watermark points at + its sibling files, mirroring the frontend's
client-side ``DeskUpdate`` assembly (``web/frontend/src/hooks/useDeskUpdates.ts``
— ``filesForMessage``). The backend carries the sign-off ordering +
message-content snapshot the frontend cannot cheaply derive here; the frontend
still resolves the live ``agent`` object / ``navTo`` / ``lane`` / ``roomLabel``
from ``coordinator_id`` exactly as it does for every other lane.

The quote is the **watermarked** message, not the room's current newest one: a
card shows what the user actually signed off, even when someone posted to the
room afterwards.

Skipped rows (the returned list may be shorter than the window — no backfill):
  * the project was deleted (``Project.get`` raises ``ValueError``), or
  * the caller is not the project owner (viewer-only / no-longer-participates —
    My Desk shows the caller's OWN DMs + projects; shared/viewer projects and
    foreign rows are excluded via ``project.created_by``).
With no limit there is no short list for a skip to contradict, which is why the
skip is invisible now in a way it was not under "Last 5".

A row whose watermark can't resolve (the ``__dismissed__`` sentinel, a deleted
message, an empty room) falls back to the room's newest message, and then to a
title-only card with an empty quote and no ``quoted_message_id``.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

from clawmeets.models.chat_message import ChatFileEvent, ChatLogEntry, ChatMessage
from clawmeets.models.desk_read_state import list_sign_offs
from clawmeets.models.project import Project

if TYPE_CHECKING:
    from clawmeets.server.context import ServerContext


class SignOffFile(BaseModel):
    """One file riding the quoted message of a signed-off card (snake_case wire
    shape; the frontend renames ``entry_type`` -> ``entryType``)."""

    filename: str
    entry_type: str  # "file_created" | "file_updated"


class SignOffCard(BaseModel):
    """A hydrated "Signed off today" card. Field-for-field the payload the
    frontend's existing ``UpdateCard`` (``DeskUpdate``) consumes, minus the bits
    it derives client-side (``agent`` / ``agentName`` / ``navTo`` /
    ``roomLabel`` / ``lane``)."""

    project_id: str
    chatroom_name: str
    kind: str                         # "dm" | "project"
    coordinator_id: str               # frontend resolves live `agent` from its cache
    agent_name: str                   # fallback label when agent not in cache
    source_name: str                  # DM title or project name -> sourceName / title
    title: str
    quote: str                        # watermarked message content, trimmed ("" if none)
    quoted_message_id: Optional[str]  # the watermarked message, NOT the room's newest
    files: list[SignOffFile]          # [] when the quoted message carried none
    signed_off_at: str                # DeskReadState.updated_at — ordering key AND the "when"


class SignOffFeed(BaseModel):
    """One "Signed off today" response.

    ``last_signed_off_at`` is the newest sign-off EVER (``None`` when the user
    has never signed anything off). It is what lets the desk tell "new user"
    apart from "quiet morning" — the first-run zero state must not come back for
    someone who signed off yesterday — and it powers the empty state's "last
    sign-off Nh ago" subtitle. It is NOT a card timestamp.
    """

    cards: list[SignOffCard]
    last_signed_off_at: Optional[str] = None


def _message_by_id(entries: list[ChatLogEntry], mid: Optional[str]) -> Optional[ChatMessage]:
    """The message ``mid`` points at, or ``None`` when it can't resolve.

    ``None`` covers the ``__dismissed__`` sentinel (never a real id), a message
    since deleted, and a room with no messages. Acks are skipped for parity with
    ``_newest_message`` — the client never watermarks one.
    """
    if not mid:
        return None
    for e in entries:
        if not isinstance(e, ChatMessage) or e.is_ack:
            continue
        if e.id == mid:
            return e
    return None


def _newest_message(entries: list[ChatLogEntry]) -> Optional[ChatMessage]:
    """The newest non-ack chat message in ``entries``, or ``None``.

    Mirrors the client's ``newestMessage``: skip file/batch rows and ``is_ack``
    markers, take the max by ``ts`` (ties resolve to the later file position via
    ``>=``, matching the frontend's fold order). Kept as the fallback for when
    the watermark can't resolve.
    """
    best: Optional[ChatMessage] = None
    for e in entries:
        if not isinstance(e, ChatMessage) or e.is_ack:
            continue
        if best is None or e.ts >= best.ts:
            best = e
    return best


def _files_for_message(
    entries: list[ChatLogEntry],
    msg: Optional[ChatMessage],
) -> list[SignOffFile]:
    """The file events that rode in with ``msg`` (``[]`` when ``msg`` is None).

    Mirrors the client's ``filesForMessage``: a ``ChatFileEvent`` with a
    non-null ``source_version``, the same sender as ``msg``, and whose
    ``source_version`` points back at ``msg`` — either ``msg.source_version``
    (an agent reply + its ``update_file`` share one source_version) or
    ``msg.version`` (a root/user message the file was appended to). Deduped by
    filename, latest wins.
    """
    if msg is None:
        return []
    by_name: dict[str, SignOffFile] = {}
    for e in entries:
        if not isinstance(e, ChatFileEvent) or e.source_version is None:
            continue
        if e.from_participant_id != msg.from_participant_id:
            continue
        matches = (
            (msg.source_version is not None and msg.source_version == e.source_version)
            or (msg.version is not None and msg.version == e.source_version)
        )
        if not matches:
            continue
        by_name[e.filename] = SignOffFile(filename=e.filename, entry_type=e.entry_type)
    return list(by_name.values())


async def build_sign_off_feed(
    ctx: "ServerContext",
    owner_user_id: str,
    since: Optional[datetime] = None,
) -> SignOffFeed:
    """Hydrate the owner's sign-offs in ``since`` into renderable cards.

    Rows come back newest-first and already windowed from ``list_sign_offs``;
    each is hydrated in order and skipped — never backfilled — when its project
    is gone or not owned by the caller.

    Quote source: the WATERMARKED message via ``_message_by_id(entries,
    row.last_seen_message_id)``, so the card shows what the user actually signed
    off rather than whatever landed in the room afterwards. Falls back to
    ``_newest_message`` when the watermark can't resolve, and to a title-only
    card (quote ``""``, ``quoted_message_id`` ``None``) when neither does.

    ``last_signed_off_at`` is passed through from the store UNFILTERED by
    ownership: it only answers "has this user ever signed anything off", which
    gates the desk zero-state. Do not use it as a card timestamp.

    Read-only throughout.
    """
    model_ctx = ctx.model_ctx
    window = await list_sign_offs(model_ctx.participants_dir, owner_user_id, since)

    cards: list[SignOffCard] = []
    for row in window.rows:
        try:
            project = Project.get(row.project_id, model_ctx)
        except ValueError:
            continue  # project deleted since the watermark was written
        # My Desk shows the caller's OWN cards; a viewer/shared row (or a foreign
        # frontdesk row) has a different owner and is skipped.
        if project.created_by != owner_user_id:
            continue

        kind = "dm" if project.is_dm_project else "project"
        room = project.get_chatroom(row.chatroom_name)
        entries = room.get_log_entries() if room is not None else []
        quoted = _message_by_id(entries, row.last_seen_message_id)
        if quoted is None:
            quoted = _newest_message(entries)
        files = _files_for_message(entries, quoted)

        # Human label: model-set display_name, falling back to the slug on legacy
        # rows (same as the frontend's `display_name ?? name`).
        title = project.display_name or project.name

        cards.append(
            SignOffCard(
                project_id=project.id,
                chatroom_name=row.chatroom_name,
                kind=kind,
                coordinator_id=project.coordinator_id,
                agent_name=project.coordinator_name,
                source_name=title,
                title=title,
                quote=quoted.content.strip() if quoted is not None else "",
                quoted_message_id=quoted.id if quoted is not None else None,
                files=files,
                signed_off_at=row.updated_at,
            )
        )
    return SignOffFeed(cards=cards, last_signed_off_at=window.last_signed_off_at)
