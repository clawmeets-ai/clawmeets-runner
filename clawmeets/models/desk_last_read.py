# SPDX-License-Identifier: MIT
"""
clawmeets/models/desk_last_read.py

Card-assembly (hydration) service for the My Desk "Last 5 Read" section.

"Last 5 Read" is a **read-only** view over the existing ``DeskReadState`` store
(``models/desk_read_state.py``): the N cards the user most recently hit "Mark
Read" on, most-recently-marked-read first. There is no new table, model, or
migration — ``list_last_read`` reads the same per-user watermark file the
"Mark Read" write path already fills.

Each watermark row carries only ``(project_id, chatroom_name, updated_at, ...)``.
This module hydrates a row into a renderable ``LastReadCard`` by loading the
``Project`` and reading the room's newest ``user-communication`` message +
sibling files, mirroring the frontend's client-side ``DeskUpdate`` assembly
(``web/frontend/src/hooks/useDeskUpdates.ts`` — ``newestMessage`` /
``filesForMessage``). The backend carries the read ordering + message-content
snapshot the frontend cannot cheaply derive here; the frontend still resolves
the live ``agent`` object / ``navTo`` / ``lane`` from ``coordinator_id`` exactly
as it does for every other lane.

Skipped rows (the returned list may be < ``limit`` even when ``limit`` rows
exist — no backfill):
  * the project was deleted (``Project.get`` raises ``ValueError``), or
  * the caller is not the project owner (viewer-only / no-longer-participates —
    My Desk shows the caller's OWN DMs + projects; shared/viewer projects and
    foreign rows are excluded via ``project.created_by``).

A row whose room has no real message yet (e.g. marked read via the
``__dismissed__`` sentinel) is still INCLUDED — it renders title-only with an
empty quote and no ``latest_message_id``, because ``_newest_message`` returns
``None`` for an empty room. The content snapshot always reflects the room's
CURRENT newest message (not the marked-read watermark), matching what the
existing card renders for the same room.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

from clawmeets.models.chat_message import ChatFileEvent, ChatLogEntry, ChatMessage
from clawmeets.models.desk_read_state import list_last_read
from clawmeets.models.project import Project

if TYPE_CHECKING:
    from clawmeets.server.context import ServerContext


class LastReadFile(BaseModel):
    """One file riding the newest message of a read card (snake_case wire shape;
    the frontend renames ``entry_type`` -> ``entryType``)."""

    filename: str
    entry_type: str  # "file_created" | "file_updated"


class LastReadCard(BaseModel):
    """A hydrated "Last 5 Read" card. Field-for-field the payload the frontend's
    existing ``UpdateCard`` (``DeskUpdate``) consumes, minus the bits it derives
    client-side (``agent`` / ``agentName`` / ``navTo`` / ``lane``)."""

    project_id: str
    chatroom_name: str
    kind: str                       # "dm" | "project"
    coordinator_id: str             # frontend resolves live `agent` from its cache
    agent_name: str                 # fallback label when agent not in cache
    source_name: str                # DM title or project name -> sourceName / title
    room_label: str                 # "direct message" | "# user-communication"
    title: str
    quote: str                      # newest message content, trimmed ("" if none)
    ts: str                         # ISO: newest message ts, else project last_modified
    latest_message_id: Optional[str]
    files: list[LastReadFile]       # [] when the newest message carried none
    marked_read_at: str             # DeskReadState.updated_at — the ordering key


def _newest_message(entries: list[ChatLogEntry]) -> Optional[ChatMessage]:
    """The newest non-ack chat message in ``entries``, or ``None``.

    Mirrors the client's ``newestMessage``: skip file/batch rows and ``is_ack``
    markers, take the max by ``ts`` (ties resolve to the later file position via
    ``>=``, matching the frontend's fold order).
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
) -> list[LastReadFile]:
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
    by_name: dict[str, LastReadFile] = {}
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
        by_name[e.filename] = LastReadFile(filename=e.filename, entry_type=e.entry_type)
    return list(by_name.values())


async def build_last_read_cards(
    ctx: "ServerContext",
    owner_user_id: str,
    limit: int = 5,
) -> list[LastReadCard]:
    """Hydrate the owner's top-``limit`` marked-read rows into renderable cards.

    Rows come back newest-first from ``list_last_read`` (already sliced to
    ``limit``); each is hydrated in order and skipped — never backfilled — when
    its project is gone or not owned by the caller, so the result may be shorter
    than ``limit``. ``limit`` is clamped to ``[0, 5]``. Read-only throughout.
    """
    limit = max(0, min(limit, 5))
    model_ctx = ctx.model_ctx
    rows = await list_last_read(model_ctx.participants_dir, owner_user_id, limit)

    cards: list[LastReadCard] = []
    for row in rows:
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
        latest = _newest_message(entries)
        files = _files_for_message(entries, latest)

        # Human label: model-set display_name, falling back to the slug on legacy
        # rows (same as the frontend's `display_name ?? name`).
        title = project.display_name or project.name
        room_label = "direct message" if kind == "dm" else "# user-communication"

        cards.append(
            LastReadCard(
                project_id=project.id,
                chatroom_name=row.chatroom_name,
                kind=kind,
                coordinator_id=project.coordinator_id,
                agent_name=project.coordinator_name,
                source_name=title,
                room_label=room_label,
                title=title,
                quote=latest.content.strip() if latest is not None else "",
                ts=(latest.ts if latest is not None else project.last_modified).isoformat(),
                latest_message_id=latest.id if latest is not None else None,
                files=files,
                marked_read_at=row.updated_at,
            )
        )
    return cards
