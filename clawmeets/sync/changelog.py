# SPDX-License-Identifier: MIT
"""
clawmeets/sync/changelog.py
Changelog entry types and payloads for the unified changelog.

This module is part of Layer 0 (pure - no domain model dependencies).
It defines the changelog types that are used for event sourcing.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Union

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NDJSON line safety
# ---------------------------------------------------------------------------

# NDJSON is \n-delimited, but ``str.splitlines()`` (and some external tooling)
# also breaks on U+2028/U+2029/U+0085. Pydantic's ``model_dump_json`` escapes
# control chars < 0x20 but emits those three raw inside JSON strings, so a
# pasted message body carrying one would split a single entry across "lines"
# and wedge every reader of the changelog. Escaping on write is lossless: a raw
# separator can only occur inside a JSON string literal, where the \uXXXX form
# parses back to the identical character.
#
# Keyed by ordinal so this source stays free of invisible characters —
# str.translate accepts a {ordinal: replacement} mapping directly.
_NDJSON_UNSAFE = {
    0x2028: "\\u2028",  # LINE SEPARATOR
    0x2029: "\\u2029",  # PARAGRAPH SEPARATOR
    0x0085: "\\u0085",  # NEXT LINE
}


def ndjson_safe(line: str) -> str:
    """Escape separator chars that survive JSON serialization un-escaped."""
    return line.translate(_NDJSON_UNSAFE)


# ---------------------------------------------------------------------------
# Enums (moved from domain/enums.py to keep sync/ self-contained)
# ---------------------------------------------------------------------------

class ChangelogEntryType(str, Enum):
    """Types of entries in the unified changelog."""
    PROJECT_CREATED = "project_created"  # Project created
    MESSAGE = "message"              # Chat message
    FILE_CREATED = "file_created"    # File created/uploaded
    FILE_UPDATED = "file_updated"    # File modified
    ROOM_CREATED = "room_created"    # Chatroom created
    ROOM_DELETED = "room_deleted"    # Chatroom deleted (FD teardown, DM cleanup)
    PROJECT_COMPLETED = "project_completed"  # Project completed
    PROJECT_REACTIVATED = "project_reactivated"  # Completed/failed task resumed by user message
    BATCH_COMPLETE = "batch_complete"  # All expected agents responded
    BATCH_TIMEOUT = "batch_timeout"    # Some agents didn't respond in time
    PARTICIPANT_ADDED = "participant_added"  # Participant added to existing room
    CHATROOM_CLEARED = "chatroom_cleared"    # All messages in a chatroom wiped by user
    PROJECT_ALLOWLIST_UPDATED = "project_allowlist_updated"  # Snapshot of agent_names/agent_teams refreshed
    DISPLAY_NAME_CHANGED = "display_name_changed"  # Project-scoped rename (dm thread auto-title)
    PROJECT_PLAN_STATE = "project_plan_state"  # Plan lifecycle projection (phase inputs + open-note count) fanned to runners
    AGENT_OFFLINE = "agent_offline"  # Targets had no live runner at dispatch time — we did not wait, nobody was there


class ProjectStatus(str, Enum):
    """Project lifecycle status."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Changelog Payloads
# ---------------------------------------------------------------------------

class ChatroomPayload(BaseModel):
    """Base class for payloads scoped to a specific chatroom.

    Chatroom-scoped entries (MESSAGE, FILE_*, ROOM_CREATED, BATCH_COMPLETE,
    BATCH_TIMEOUT) inherit from this class and include chatroom_name in
    the payload.

    Project-level entries (PROJECT_CREATED, PROJECT_COMPLETED,
    PROJECT_REACTIVATED) do NOT inherit from this class and have no
    chatroom_name.
    """
    chatroom_name: str


class MessagePayload(ChatroomPayload):
    """Payload for MESSAGE entries in unified changelog.

    This is a flat structure with all message fields directly on the payload.
    Layer 0 (pure) - no dependencies on Layer 1 models.

    To convert to ChatMessage for callbacks, use:
        from clawmeets.models.chat_message import ChatMessage
        chat_message = ChatMessage.from_message_payload(payload)
    """
    # protected_namespaces=() so the ``model_config_name`` field (per-request
    # model override) doesn't collide with Pydantic's ``model_`` namespace.
    model_config = {"protected_namespaces": ()}

    # chatroom_name inherited from ChatroomPayload
    id: str
    ts: datetime
    from_participant_id: str
    from_participant_name: str  # Required - authenticated participant must have a name
    content: str
    expects_response_from: list[str] = Field(default_factory=list)
    is_ack: bool = Field(default=False)
    # Per-request model override (spec #3): names a config on the responding
    # agent. null/absent ⇒ use the agent's default config; overrides that one
    # turn only. Resolved runner-side against the agent's stored configs;
    # unknown ⇒ silent fallback to the default (never errors).
    model_config_name: Optional[str] = None


class FilePayload(ChatroomPayload):
    """Payload for FILE_CREATED and FILE_UPDATED entries in unified changelog.

    from_participant_id / from_participant_name attribute the file touch to
    its uploader so CHATS.ndjson can log who touched the file (used by the
    web UI's inline file pills). Older payloads without these fields default
    to empty strings for backward compat.
    """
    filename: str
    content_b64: str  # Base64-encoded file content (required)
    sha256: str       # SHA256 hash of the content (required)
    from_participant_id: str = ""  # Uploader's participant ID (empty = unknown / legacy)
    from_participant_name: str = ""  # Uploader's display name


class RoomCreatedParticipant(BaseModel):
    """Participant info for room creation."""
    id: str
    name: str
    external: bool = False  # True ⇒ foreign agent; their runner's notifier skips processing this membership (the FD tunnel handles delivery)


class RoomCreatedPayload(ChatroomPayload):
    """Payload for ROOM_CREATED entries in unified changelog."""
    # chatroom_name inherited from ChatroomPayload
    participants: list[RoomCreatedParticipant] = Field(default_factory=list)


class RoomDeletedPayload(ChatroomPayload):
    """Payload for ROOM_DELETED entries in unified changelog.

    Emitted when a work room is torn down (Front Desk project deletion, the
    legacy-DM cleanup scripts). Subscribers must be idempotent: deleting an
    already-absent dir / branch is a no-op. `shared-context` and
    `user-communication` are never deletable through this entry — callers
    are expected to enforce that, and subscribers do the same defensively.
    """
    # chatroom_name inherited from ChatroomPayload
    pass


class ProjectCompletedPayload(BaseModel):
    """Payload for PROJECT_COMPLETED entries in unified changelog.

    Project-level entry - no chatroom_name field.
    """
    pass  # No fields needed - the entry type itself indicates completion


class ProjectReactivatedPayload(BaseModel):
    """Payload for PROJECT_REACTIVATED entries — project-level, no chatroom_name.

    Emitted when a user posts a message into a completed/failed task's
    user-communication chatroom; flips status back to ACTIVE so the row
    moves out of COMPLETED TASKS in the sidebar.
    """
    pass


class BatchCompletePayload(ChatroomPayload):
    """Payload for BATCH_COMPLETE entries in unified changelog."""
    message_id: str
    coordinator_id: str
    responded_participants: list[str]


class BatchTimeoutPayload(ChatroomPayload):
    """Payload for BATCH_TIMEOUT entries in unified changelog.

    ``offline_participants`` is the subset of ``timed_out_participants`` whose
    runner had no live hub connection **at the moment the timeout fired** — it
    is re-checked there, never carried forward from dispatch time, because the
    case it exists for is precisely an agent that WAS live when the batch
    opened and died mid-work.

    It is defaulted so every ``batch_timeout`` row already on disk still
    parses, exactly as every other list field on this payload is.

    It is what keeps "ignored you" distinguishable from "was never there" on a
    row that already timed out. The sibling distinction — "never received the
    message at all" — is a separate :class:`AgentOfflinePayload` row, not a
    flavour of this one, because only that case is fixed by resending.
    """
    message_id: str
    coordinator_id: str
    responded_participants: list[str]
    timed_out_participants: list[str]
    offline_participants: list[str] = Field(default_factory=list)


class AgentOfflinePayload(ChatroomPayload):
    """Payload for AGENT_OFFLINE entries in unified changelog.

    Sibling of :class:`BatchTimeoutPayload`, deliberately NOT a field on it.
    The two rows mean different things and have different producers and
    different clocks: BATCH_TIMEOUT means "we waited ``timeout_seconds`` and
    nobody came", this means "we did not wait, nobody was there". Fusing them
    would fire three side effects that are all wrong at second zero — the
    "did not respond within 30 min" system message, ``CANCEL_LLM`` to an agent
    that is not there, and a coordinator wake for a batch that never started.

    Both lists carry AGENT IDS, never names: the client indexes recipient
    status by id, so a name silently misses every lookup.

    ``message_id`` is REQUIRED. It keys the offline state to the ``@mention``
    that would have opened the batch, exactly as ``batch_timeout`` does. The
    client's per-recipient index is ``(message_id, agent_id)``; without it an
    agent addressed twice — once while offline, once after ``clawmeets start``
    — would read OFFLINE on both rows forever, and the once-per-message
    explainer would have no grouping key.

    ``dispatched_participants`` is derivable from expects minus offline, but is
    carried anyway to match the ``batch_timeout`` precedent
    (``responded_participants`` + ``timed_out_participants``) and keep the row
    self-describing. It is ``[]`` in the all-offline case.
    """
    message_id: str
    coordinator_id: str
    offline_participants: list[str]      # Agent ids with no live hub connection at dispatch
    dispatched_participants: list[str]   # Agent ids the batch actually opened for


class ParticipantAddedPayload(ChatroomPayload):
    """Payload for PARTICIPANT_ADDED entries in unified changelog.

    Used when adding a participant to an existing chatroom (e.g., auto-adding
    agents to shared-context when they join a project via a work room).
    """
    participant_id: str
    participant_name: str
    external: bool = False  # True ⇒ foreign agent; their runner's notifier skips processing this membership


class ChatroomClearedPayload(ChatroomPayload):
    """Payload for CHATROOM_CLEARED entries in unified changelog.

    Carries the metadata each subscriber needs to rewrite its local
    CHATS.ndjson identically: the same archive filename appears on every
    runner so a support engineer can diff one server-side backup against
    its runner-side twin.
    """
    cleared_by_participant_id: str   # user id that initiated the clear
    cleared_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    cleared_through_version: int     # changelog version this clear supersedes
    archive_filename: str | None = None  # None when room was already empty


class ProjectAllowlistUpdatedPayload(BaseModel):
    """Payload for PROJECT_ALLOWLIST_UPDATED entries.

    Project-level entry — refreshes ``Project.agent_names`` / ``Project.agent_teams``.
    Fanned out by ``PUT /agents/{id}`` when the coordinator's Front Desk
    invitable list changes so existing FD projects pick up the edit.
    """
    agent_names: list[str] = Field(default_factory=list)
    agent_teams: list[str] = Field(default_factory=list)


class ProjectPlanStatePayload(BaseModel):
    """Payload for PROJECT_PLAN_STATE entries — the plan's lifecycle projection.

    Project-level entry (no ``chatroom_name``), same scope as
    ``ProjectCompletedPayload``. It carries the six facts ``Project.phase``, the
    execution gate (§7.4) and the coordinator's spec-lock line are computed
    from, so all three are answerable from a runner's own synced ``meta.json``.

    **Why this exists at all, since §7.2 says the fields are "denormalized like
    ``report_published_at``".** That precedent is a *server-only* field: nothing
    replays it, and a runner never learns it. The gate and the coordinator's
    steady-state prompt block both run **on the runner**, off
    ``Project.get(pid, model_ctx)`` and a sidecar that lives only in the
    server's ``metadata/`` tree. A direct ``meta.json`` write reaches neither,
    so a gate built on one is inert in production and green in any test whose
    server and runner share a context. **The changelog is the one channel that
    carries server state to a runner**, which is why the projection rides it.

    The sidecar remains the source of truth; this is its projection, written by
    one publisher (``models/project_plan.publish_plan_state``) and reconciled
    against its source by test.
    """
    project_id: str
    #: ``init_plan_sidecar`` and nothing else (§3.5 property 3). A pre-feature
    #: project never gets one, so it reads ``executing`` with no backfill.
    plan_seeded_at: str | None = None
    plan_accepted_at: str | None = None
    plan_accepted_revision: int = 0
    #: ``spec_digest`` as it read at acceptance. The runner compares the synced
    #: ``PLAN.md`` against it for §7.3's "the spec moved" line — a pure function
    #: of a synced file and a stored string, so no sidecar read.
    plan_accepted_spec_digest: str = ""
    #: The projection of ``project_plan.open_notes_for_you`` — §3.3's one number
    #: carried to the one process that cannot open the sidecar.
    plan_open_notes: int = 0
    #: The projection of ``ProjectPlan.first_user_review_at`` — when the OWNER
    #: first opened a review round, which is the spec lock's start line before
    #: acceptance (``project_plan._spec_is_locked``). Carried for the
    #: coordinator's steady-state prompt block, which must state the constraint
    #: BEFORE the model forms the intent rather than after the API refuses it.
    #: Optional with a ``None`` default, so every entry already on a changelog
    #: replays unchanged.
    plan_user_reviewed_at: str | None = None
    #: The projection of ``project_plan._plan_was_offered`` as an instant — the
    #: first round's ``opened_at``, i.e. when the coordinator first put this plan
    #: in front of the user. The OTHER half of the spec lock
    #: (``project_plan._spec_is_locked``): a plan can be accepted without ever
    #: having been offered, because the confirm row is dismissible from the desk
    #: card while it still reads ``GO_NOTE_DRAFTING``, and the keeper still owns
    #: the document until it has been handed over. Carried for the same reason
    #: its neighbour is — the prompt must state the constraint before the model
    #: forms the intent, and this process cannot open the sidecar. Same
    #: ``None``-default replay guarantee.
    plan_offered_at: str | None = None


class DisplayNameChangedPayload(BaseModel):
    """Payload for DISPLAY_NAME_CHANGED entries in unified changelog.

    Project-level entry (no chatroom_name) — mirrors ``ProjectCompletedPayload``
    in scope. Carries the new model-generated title that replaces the seeded
    ``"New chat"`` placeholder on a DM thread after its first exchange. Applied
    by ``ModelContextChangelogSubscriber._handle_display_name_changed`` via the
    already-shipped ``ProjectState.set_thread_title`` (slug/dir untouched).
    """
    project_id: str
    display_name: str  # the new model-generated title (replaces "New chat")


class ProjectCreatedPayload(BaseModel):
    """Payload for PROJECT_CREATED entries in unified changelog.

    Project-level entry - no chatroom_name field.
    """
    project_id: str
    project_name: str
    coordinator_id: str
    coordinator_name: str  # Name of coordinator (required - avoids lookup on workers)
    request: str
    created_by: str  # user_id of creator (required - derived from auth)
    agent_pool: str = "verified"  # "self", "owned", "verified", or "all". "self" = coordinator only (used by own-DM).
    agent_teams: list[str] = Field(default_factory=list)  # Hard allowlist by user_team; pairs with agent_names. Empty teams + empty names = no filter.
    agent_names: list[str] = Field(default_factory=list)  # Hard allowlist by agent display name (id, full name, or owner-relative short name); pairs with agent_teams. OR semantics across both lists.
    surface: str = "regular"  # "regular" | "dm" — explicit project shape
    display_name: Optional[str] = None  # model-set label; "New chat" placeholder for a fresh dm thread. Optional/default for wire compat.


# Union type for changelog payloads
ChangelogPayload = Union[
    ProjectCreatedPayload,
    MessagePayload,
    FilePayload,
    RoomCreatedPayload,
    RoomDeletedPayload,
    ProjectCompletedPayload,
    ProjectReactivatedPayload,
    BatchCompletePayload,
    BatchTimeoutPayload,
    AgentOfflinePayload,
    ParticipantAddedPayload,
    ChatroomClearedPayload,
    ProjectAllowlistUpdatedPayload,
    DisplayNameChangedPayload,
    ProjectPlanStatePayload,
]


class MirroredFromRef(BaseModel):
    """Pointer to the source entry that a cross-project mirror reflects.

    Written by ``TunnelSubscriber`` when mirroring an entry from one project's
    room into a bound room in another project. Subscribers (including the
    TunnelSubscriber itself) use this annotation as a loop-guard: never act on
    an entry whose ``mirrored_from`` is set.
    """
    model_config = {"frozen": True}

    project_id: str
    version: int


# ---------------------------------------------------------------------------
# Changelog Entry
# ---------------------------------------------------------------------------

class ChangelogEntry(BaseModel):
    """Single entry in the unified changelog.

    The unified changelog provides monotonic versioning across all events
    (messages, file changes, invites, etc.) in a project.

    Chatroom-scoped entries have chatroom_name in their payload (via ChatroomPayload).
    Project-level entries (PROJECT_CREATED, PROJECT_COMPLETED, PROJECT_REACTIVATED) have no chatroom_name.
    Access chatroom_name via: payload.chatroom_name (for typed payloads) or
    getattr(entry.payload, 'chatroom_name', None) (for mixed entry types).

    This is a pure model without Active Record methods.
    For persistence methods, use clawmeets.models.ChangelogEntry.
    """
    model_config = {"frozen": True}

    version: int
    entry_type: ChangelogEntryType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: ChangelogPayload
    source_version: int | None = None  # Version of the entry that triggered this one (reply-to link)
    mirrored_from: MirroredFromRef | None = None  # Set by TunnelSubscriber when this entry is a cross-project mirror; loop-guard.

    @model_validator(mode="before")
    @classmethod
    def coerce_payload_type(cls, data: dict) -> dict:
        """Coerce payload dict to correct type before Pydantic validation.

        Pydantic v2's Union discrimination doesn't always correctly infer
        the payload type from JSON. This validator explicitly coerces the
        payload based on entry_type.
        """
        if not isinstance(data, dict):
            return data

        entry_type = data.get("entry_type")
        payload = data.get("payload")

        if entry_type and isinstance(payload, dict):
            payload_types = {
                "project_created": ProjectCreatedPayload,
                "message": MessagePayload,
                "file_created": FilePayload,
                "file_updated": FilePayload,
                "room_created": RoomCreatedPayload,
                "room_deleted": RoomDeletedPayload,
                "project_completed": ProjectCompletedPayload,
                "project_reactivated": ProjectReactivatedPayload,
                "batch_complete": BatchCompletePayload,
                "batch_timeout": BatchTimeoutPayload,
                "agent_offline": AgentOfflinePayload,
                "participant_added": ParticipantAddedPayload,
                "chatroom_cleared": ChatroomClearedPayload,
                "project_allowlist_updated": ProjectAllowlistUpdatedPayload,
                "display_name_changed": DisplayNameChangedPayload,
            }

            payload_cls = payload_types.get(entry_type)
            if payload_cls:
                data = dict(data)  # Make a copy to avoid mutating input
                data["payload"] = payload_cls.model_validate(payload)

        return data

    @model_validator(mode="after")
    def validate_payload_type(self) -> "ChangelogEntry":
        """Ensure payload type matches entry_type."""
        expected_types = {
            ChangelogEntryType.PROJECT_CREATED: ProjectCreatedPayload,
            ChangelogEntryType.MESSAGE: MessagePayload,
            ChangelogEntryType.FILE_CREATED: FilePayload,
            ChangelogEntryType.FILE_UPDATED: FilePayload,
            ChangelogEntryType.ROOM_CREATED: RoomCreatedPayload,
            ChangelogEntryType.ROOM_DELETED: RoomDeletedPayload,
            ChangelogEntryType.PROJECT_COMPLETED: ProjectCompletedPayload,
            ChangelogEntryType.PROJECT_REACTIVATED: ProjectReactivatedPayload,
            ChangelogEntryType.BATCH_COMPLETE: BatchCompletePayload,
            ChangelogEntryType.BATCH_TIMEOUT: BatchTimeoutPayload,
            ChangelogEntryType.AGENT_OFFLINE: AgentOfflinePayload,
            ChangelogEntryType.PARTICIPANT_ADDED: ParticipantAddedPayload,
            ChangelogEntryType.CHATROOM_CLEARED: ChatroomClearedPayload,
            ChangelogEntryType.PROJECT_ALLOWLIST_UPDATED: ProjectAllowlistUpdatedPayload,
            ChangelogEntryType.DISPLAY_NAME_CHANGED: DisplayNameChangedPayload,
            ChangelogEntryType.PROJECT_PLAN_STATE: ProjectPlanStatePayload,
        }
        expected = expected_types[self.entry_type]
        if not isinstance(self.payload, expected):
            raise ValueError(
                f"entry_type {self.entry_type} requires {expected.__name__} payload, "
                f"got {type(self.payload).__name__}"
            )
        return self

    def to_log_line(self) -> str:
        """Serialize to NDJSON line."""
        return ndjson_safe(self.model_dump_json())

    @classmethod
    def from_log_line(cls, line: str) -> "ChangelogEntry":
        """Deserialize from NDJSON line."""
        return cls.model_validate_json(line)




# ---------------------------------------------------------------------------
# Reading changelog.ndjson without reading all of it
#
# **The file is an append-only log that also stores file bodies.** `FilePayload`
# above carries `content_b64`, so one entry can be 18 MB and one project's
# changelog 1.2 GB in 4,700 lines. Every reader here used to be
# `read_text().split("\n")`, which on the server — single-process, single event
# loop — meant a `GET /participants/{id}/projects` over 1001 projects read
# ~2.2 GB synchronously and no WebSocket handshake could be serviced while it
# did. 147 of 228 runners could not connect, each retry calling that same
# endpoint, so it degraded instead of settling.
#
# Two primitives replace that, and both rest on ONE invariant: **the changelog
# is append-only and ascending by version.** Nothing here may assume it is
# DENSE — the runner's copy is fetched with `participant_id` filtering and has
# gaps.
#
# **Deliberately NOT shared with `_ChatLogCache`** (`models/chatroom.py`), which
# solves the same-shaped problem for CHATS.ndjson. That class caches the parsed
# entry list; doing so here would pin ~1.2 GB of Python objects for one project,
# and a list request touches a thousand. Its `_parse_from` also slurps
# `f.read()` to EOF, which is right for a 2.4 MB chat log and is exactly the
# 9.3 s bug for a changelog. What IS borrowed is the discipline: validate with
# (inode, mtime_ns, size), consume only complete lines, split on b"\n" alone.
# Unify only if a third caller appears.
# ---------------------------------------------------------------------------

#: Backwards/forwards scan granularity. Also the read size for `iter_entries`.
_TAIL_CHUNK = 65536

#: How much of a line to inspect for its version before falling back to a full
#: parse. `version` is `ChangelogEntry`'s FIRST field and pydantic preserves
#: declaration order, so a written line always begins `{"version":N,`.
_HEAD_WINDOW = 4096

#: **Anchored at the line start, and that is load-bearing.** An unanchored
#: `"version"\s*:` also matches the nested key inside `mirrored_from`
#: (:class:`MirroredFromRef`), so a mirrored entry as the last line would
#: silently report *a foreign project's* version as this project's tip — and the
#: tip is what `append` adds 1 to.
_HEAD_VERSION_RE = re.compile(rb'^\{\s*"version"\s*:\s*(\d+)')


def _read_exactly(fd: int, offset: int, length: int) -> bytes:
    """`length` bytes from `offset`, tolerating short reads."""
    out = bytearray()
    os.lseek(fd, offset, os.SEEK_SET)
    while len(out) < length:
        chunk = os.read(fd, length - len(out))
        if not chunk:
            break
        out += chunk
    return bytes(out)


def _last_line_bounds(fd: int, size: int) -> tuple[int, int] | None:
    """`(start, end)` of the last non-blank line, or None if there is none.

    Scans backwards a chunk at a time and **keeps at most one chunk in memory**
    — an 18 MB final line costs ~280 reads and 64 KB of RSS, not 18 MB. A chunk
    containing no newline is the ordinary case here, not an error.
    """
    # Trailing "\n" run first, so a file ending in a newline, a file ending
    # without one, and a file with blank lines at the end all agree.
    end = size
    while end > 0:
        step = min(_TAIL_CHUNK, end)
        stripped = _read_exactly(fd, end - step, step).rstrip(b"\n")
        if stripped:
            end = end - step + len(stripped)
            break
        end -= step
    if end == 0:
        return None

    # Then the newline that precedes it. `end - 1` is a content byte, so any
    # match is strictly inside the previous line's terminator.
    pos = end
    while pos > 0:
        step = min(_TAIL_CHUNK, pos)
        nl = _read_exactly(fd, pos - step, step).rfind(b"\n")
        if nl >= 0:
            return pos - step + nl + 1, end
        pos -= step
    return 0, end  # the whole file is one line


def read_tip_version_and_stat(path: Path) -> tuple[int, os.stat_result] | None:
    """`(version of the last entry, stat)`, or None when the file is absent.

    **The stat comes from `fstat` on the same descriptor the read used**, and
    callers that cache on it depend on that: a separate `os.stat` before or
    after could pair "size S" with a version read at size S' > S, and a cache
    keyed on the smaller size would then hand `append` a stale tip and duplicate
    a version.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except (FileNotFoundError, NotADirectoryError):
        return None
    try:
        st = os.fstat(fd)
        bounds = _last_line_bounds(fd, st.st_size)
        if bounds is None:
            return 0, st
        start, end = bounds
        head = _read_exactly(fd, start, min(_HEAD_WINDOW, end - start))
        m = _HEAD_VERSION_RE.match(head)
        if m:
            return int(m.group(1)), st
        # A line whose keys were not written in declaration order — a fixture
        # using FileUtil.write(..., "ndjson"), say. Correctness over speed.
        try:
            return int(json.loads(_read_exactly(fd, start, end - start))["version"]), st
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Unreadable last changelog line in {path}: {e}")
            return 0, st
    finally:
        os.close(fd)


def read_tip_version(path: Path) -> int:
    """The last entry's version — `0` when the file is absent or has no entries.

    O(size of the last entry), not O(file). For the caller that also needs the
    stat to cache on, use :func:`read_tip_version_and_stat`.
    """
    result = read_tip_version_and_stat(path)
    return 0 if result is None else result[0]


def iter_entries(path: Path) -> Iterator[ChangelogEntry]:
    """Every entry, in file order, **retaining nothing**.

    Replaces `read_text(...).split("\\n")`, which held the file, a decoded copy
    and a list of lines at once — `.strip()` on a 1.2 GB string alone cost 4.3 s
    of the 13.6 s that `get_current_version` used to take.

    Yields the trailing line even when the file does not end in `\\n`, which is
    what the `split("\\n")` it replaces did; a genuinely half-written append
    therefore raises here exactly as it did before.
    """
    try:
        f = open(path, "rb")
    except (FileNotFoundError, NotADirectoryError):
        return
    with f:
        buf = bytearray()
        while True:
            chunk = f.read(_TAIL_CHUNK)
            if not chunk:
                break
            # bytearray `+=` amortises; `del buf[:start]` below is one memmove
            # per chunk. Slicing a growing `bytes` instead would make an 18 MB
            # line quadratic.
            buf += chunk
            start = 0
            while (nl := buf.find(b"\n", start)) >= 0:
                line = bytes(buf[start:nl]).strip()
                start = nl + 1
                if line:
                    yield ChangelogEntry.model_validate_json(line)
            if start:
                del buf[:start]
        tail = bytes(buf).strip()
        if tail:
            yield ChangelogEntry.model_validate_json(tail)
