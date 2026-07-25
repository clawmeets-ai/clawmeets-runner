# SPDX-License-Identifier: MIT
"""
clawmeets/models/chatroom.py
Chatroom model with Active Record persistence methods.

## Changelog-First Architecture

Chatroom is a **frozen** Pydantic model (`model_config = {"frozen": True}`).
All mutations flow through the changelog (acting as a redo log), ensuring:
1. Atomic recording on the server before any local writes
2. Eventual consistency across all runners via sync
3. Idempotent replay for crash recovery

Direct mutation is prevented by the frozen config. Use `chatroom.state()`
to access ChatroomState for changelog-driven writes.

## Read/Write Separation

- **Chatroom** (frozen): Read-only data representation, path properties,
  association methods for loading related objects
- **ChatroomState**: Handles all filesystem writes triggered by changelog
  processing (create, append_message, write_file, delete_file, invite)

This separation ensures changes are visible to other agents only after
they flow through the distributed changelog system.
"""
from __future__ import annotations

import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field, PrivateAttr, computed_field

from ..utils.file_io import FileUtil
from .chat_message import (
    ChatBatchTimeoutEvent,
    ChatFileEvent,
    ChatLogEntry,
    ChatMessage,
    parse_log_line,
)
from .participant import Participant

if TYPE_CHECKING:
    from .agent import Agent
    from .context import ModelContext


# A mention is @<handle> at a word boundary. The handle charset mirrors
# FileUtil.validate_fs_name (used at user/agent registration): [a-zA-Z0-9_-],
# never a dot. So the trailing negative lookahead `(?!\.[a-zA-Z])` — which
# skips a domain/TLD suffix like `@clawmeets.ai` — can never clip a real
# handle. The name class is possessive (`*+`, 3.11+) so the engine can't
# backtrack into a truncated match (`clawmeet` out of `@clawmeets.ai`).
_MENTION_RE = re.compile(r'(?:^|(?<=[^a-zA-Z0-9_]))@([a-zA-Z][a-zA-Z0-9_-]*+)(?!\.[a-zA-Z])')


class Chatroom(BaseModel):
    """Chatroom metadata with persistence and association methods.

    A chatroom is a communication space within a project where
    participants can exchange messages and share files.

    The chatroom name is unique within a project and serves as the
    primary identifier for lookups and file paths.
    """
    model_config = {"frozen": True}

    id: str  # Legacy field, kept for internal reference
    name: str  # Primary identifier, unique within project
    project_id: str
    project_name: str  # Required for path construction
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Private runtime state (not serialized)
    _ctx: Optional["ModelContext"] = PrivateAttr(default=None)

    @property
    def ctx(self) -> "ModelContext":
        """Get the model context (required for I/O operations).

        Raises:
            RuntimeError: If context is not set (use Chatroom.get() to load with context)
        """
        if self._ctx is None:
            raise RuntimeError(
                "Chatroom requires ModelContext. "
                "Use Chatroom.get() to load with context."
            )
        return self._ctx

    # -------------------------------------------------------------------------
    # Path Properties (derived from ctx + project_id + project_name + name)
    # -------------------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        """Path to chatroom data directory (projects/{name}-{id}/chatrooms/{chatroom_name}/)."""
        return (
            self.ctx.projects_dir
            / f"{self.project_name}-{self.project_id}"
            / "chatrooms"
            / self.name
        )

    @property
    def meta_dir(self) -> Path:
        """Path to chatroom metadata directory."""
        return (
            self.ctx.metadata_dir
            / f"{self.project_name}-{self.project_id}"
            / "chatrooms"
            / self.name
        )

    @property
    def meta_path(self) -> Path:
        """Path to chatroom meta.json file."""
        return self.meta_dir / "meta.json"

    @property
    def chats_path(self) -> Path:
        """Path to CHATS.ndjson file."""
        return self.data_dir / "CHATS.ndjson"

    @property
    def files_dir(self) -> Path:
        """Path to files/ directory."""
        return self.data_dir / "files"

    @property
    def participants_path(self) -> Path:
        """Path to PARTICIPANTS.ndjson file."""
        return self.data_dir / "PARTICIPANTS.ndjson"

    @computed_field
    @property
    def participants(self) -> list[str]:
        """List participant IDs by reading PARTICIPANTS.ndjson.

        Derived from filesystem: reads entries from PARTICIPANTS.ndjson
        and extracts unique agent_ids (preserving insertion order).

        Returns:
            List of participant IDs (deduplicated, order preserved)
        """
        if not self.participants_path.exists():
            return []
        entries = FileUtil.read(self.participants_path, "ndjson")
        return [entry["agent_id"] for entry in entries]

    def is_external_member(self, agent_id: str) -> bool:
        """True if ``agent_id`` is in PARTICIPANTS.ndjson with ``external=true``.

        Used by the runner-side notifier to skip LLM invocation for foreign
        memberships (the FD tunnel delivers the message via FD instead).
        Returns False for legacy rows missing the ``external`` field.
        """
        if not self.participants_path.exists():
            return False
        entries = FileUtil.read(self.participants_path, "ndjson")
        for entry in entries:
            if entry.get("agent_id") == agent_id:
                return bool(entry.get("external", False))
        return False

    @property
    def is_shared_context_room(self) -> bool:
        """Check if this is the shared-context room for project-wide knowledge."""
        return self.name.startswith("shared-context")

    @property
    def is_user_communication_room(self) -> bool:
        """Check if this is the user-communication room for user<->assistant chat."""
        return self.name.startswith("user-communication")

    @property
    def last_cleared_at(self) -> Optional[datetime]:
        """When the chatroom history was last cleared, if ever.

        Read from meta.json; written by ChatroomState.clear_history(). None
        means the room has never been cleared.
        """
        data = FileUtil.read(self.meta_path, "json", default=None)
        if not data:
            return None
        raw = data.get("last_cleared_at")
        if not raw:
            return None
        return datetime.fromisoformat(raw)

    @property
    def last_cleared_through_version(self) -> Optional[int]:
        """Changelog version that the last clear superseded, if ever."""
        data = FileUtil.read(self.meta_path, "json", default=None)
        if not data:
            return None
        return data.get("last_cleared_through_version")

    # -------------------------------------------------------------------------
    # Association Methods (lookup-based, no caching)
    # -------------------------------------------------------------------------

    def project(self):
        """Load the parent project.

        Returns:
            Project or None if not found
        """
        from .project import Project
        return Project.get(self.project_id, self.ctx)

    def parse_mentions(self, content: Optional[str]) -> list[str]:
        """Extract @agent-name mentions from a message in this room.

        Returns names without the ``@`` prefix, deduplicated, preserving
        order of first occurrence. Matches ``@`` only at word boundaries and
        excludes domain/email tokens: both ``user@x.com`` (email local-part)
        and a bare ``@x.com`` (domain/TLD) are rejected, while ``**@agent**``
        is accepted. Valid handles never contain a dot (see ``_MENTION_RE``).
        """
        if not content:
            return []
        seen: set[str] = set()
        result: list[str] = []
        for m in _MENTION_RE.findall(content):
            if m not in seen:
                seen.add(m)
                result.append(m)
        return result

    def resolve_mention(self, name: str) -> Optional["Agent"]:
        """Resolve an @mention name to an Agent in this room's namespace.

        Delegates to ``Agent.resolve_with_namespace`` using this chatroom's
        parent project. Does NOT filter by participant membership — callers
        that need that distinction (e.g. to log "not a participant" warnings
        separately from "not found") apply it themselves.
        """
        from .agent import Agent
        return Agent.resolve_with_namespace(name, self.project(), self.ctx)

    def list_participants(self) -> list:
        """Load participants for all chatroom participants.

        Returns:
            List of Participant objects (Agent, Assistant, or User)
        """
        result = []
        for pid in self.participants:
            participant = Participant.get(pid, self.ctx)
            if participant is not None:
                result.append(participant)
        return result

    def get_log_entries(self, limit: int = 9999999) -> list[ChatLogEntry]:
        """Load all CHATS.ndjson rows (messages + file events).

        Legacy rows without `entry_type` are parsed as ChatMessage.

        Args:
            limit: Maximum number of entries to return (most recent)

        Returns:
            List of ChatMessage | ChatFileEvent in file order
        """
        if not self.chats_path.exists():
            return []
        result: list[ChatLogEntry] = []
        for line_data in FileUtil.read(self.chats_path, "ndjson"):
            result.append(parse_log_line(line_data))
        return result[-limit:]

    def get_messages(self, limit: int = 9999999) -> list[ChatMessage]:
        """Load messages from this chatroom, filtering out file events.

        Args:
            limit: Maximum number of messages to return (most recent last)

        Returns:
            List of ChatMessage objects (most recent last)
        """
        messages = [e for e in self.get_log_entries() if isinstance(e, ChatMessage)]
        return messages[-limit:]

    def recent_history_for_prompt(
        self,
        limit: int = 10,
        exclude_message_id: Optional[str] = None,
    ) -> list[tuple[str, str]]:
        """Return the last ``limit`` non-ack messages as (sender, content) pairs.

        Used by the prompt builder's `RECENT CHAT IN THIS ROOM` section so the
        LLM sees recent context on every turn (not only batch mode).

        Args:
            limit: max number of messages to include (most recent last).
            exclude_message_id: drop this message id from the result — used
                when the trigger message will already appear in the prompt's
                `INCOMING MESSAGE` block, to avoid duplication.
        """
        messages = [m for m in self.get_messages() if not m.is_ack]
        if exclude_message_id is not None:
            messages = [m for m in messages if m.id != exclude_message_id]
        messages = messages[-limit:]
        return [
            (m.from_participant_name or m.from_participant_id, m.content)
            for m in messages
        ]

    def get_messages_since(self, since_message_id: str) -> list[ChatMessage]:
        """Get messages after a given message ID.

        Args:
            since_message_id: Return messages after this message

        Returns:
            List of ChatMessage objects after the specified message

        Raises:
            ValueError: If since_message_id not found
        """
        messages = self.get_messages()
        ids = [m.id for m in messages]
        if since_message_id not in ids:
            raise ValueError(f"Message {since_message_id!r} not found")
        idx = ids.index(since_message_id)
        return messages[idx + 1:]

    def count_messages(self) -> int:
        """Count messages in this chatroom.

        Returns:
            Number of messages in the chatroom
        """
        return len(self.get_messages())

    def list_files(self) -> list[str]:
        """List file paths in this chatroom, including subdirectories.

        Returns:
            Sorted list of file paths relative to the files directory
        """
        return FileUtil.list_dir_recursive(self.files_dir)

    def get_file(self, filename: str) -> Optional[bytes]:
        """Read file content from this chatroom.

        Args:
            filename: The filename to read

        Returns:
            File content as bytes, or None if not found
        """
        file_path = self.files_dir / filename
        return FileUtil.read(file_path, "bytes")

    def file_exists(self, filename: str) -> bool:
        """Check if a file exists in this chatroom.

        Args:
            filename: The filename to check

        Returns:
            True if file exists, False otherwise
        """
        return (self.files_dir / filename).exists()

    # -------------------------------------------------------------------------
    # Active Record: Persistence Methods
    # -------------------------------------------------------------------------

    @classmethod
    def get(
        cls,
        project_id: str,
        chatroom_name: str,
        ctx: "ModelContext",
    ) -> Optional["Chatroom"]:
        """Load chatroom by name.

        Args:
            project_id: The project ID
            chatroom_name: The chatroom name (unique within project)
            ctx: ModelContext for filesystem operations

        Returns:
            Chatroom, or None when the room doesn't exist — same Active Record
            contract as ``Project.get`` / ``Agent.get``, so route handlers'
            ``if room is None: raise HTTPException(404)`` checks work. (This
            used to raise ValueError, which surfaced as a 500 on e.g. a file
            upload to a room name an agent got slightly wrong.)
        """
        from .project import Project

        # Load project first to get project_name. Project.get raises ValueError
        # for an unknown project — fold that into the None contract here.
        try:
            project = Project.get(project_id, ctx)
        except ValueError:
            return None

        # Build path directly using project name
        meta_path = (
            ctx.metadata_dir
            / f"{project.name}-{project_id}"
            / "chatrooms"
            / chatroom_name
            / "meta.json"
        )

        data = FileUtil.read(meta_path, "json")
        if not data:
            return None
        instance = cls.model_validate(data)
        object.__setattr__(instance, "_ctx", ctx)
        return instance

    # -------------------------------------------------------------------------
    # State Access (for write operations)
    # -------------------------------------------------------------------------

    def state(self) -> "ChatroomState":
        """Get the state model for write operations.

        Returns:
            ChatroomState instance for this chatroom
        """
        return ChatroomState(self)


# =============================================================================
# ChatroomState: Write Operations
# =============================================================================


class ChatroomState:
    """Mutable state model for chatroom write operations.

    This class handles all filesystem write operations for a chatroom,
    triggered by changelog processing. The separation keeps the main
    Chatroom model immutable while allowing write operations.

    Usage:
        chatroom = Chatroom.get(project_id, chatroom_name, ctx)
        chatroom.state().append_message(message)

    For creation, use the classmethod:
        chatroom = ChatroomState.create(project_id, project_name, chatroom_name, participants, ctx)
    """

    def __init__(self, chatroom: Chatroom) -> None:
        """Initialize with a chatroom instance.

        Args:
            chatroom: The chatroom to operate on
        """
        self._chatroom = chatroom

    @classmethod
    def create(
        cls,
        project_id: str,
        project_name: str,
        chatroom_name: str,
        participants: list[dict],
        created_at: datetime,
        ctx: "ModelContext",
    ) -> Chatroom:
        """Create a new chatroom with directories and meta.json.

        Creates:
        - Data directory: projects/{name}-{id}/chatrooms/{chatroom_name}/
        - Files directory: projects/{name}-{id}/chatrooms/{chatroom_name}/files/
        - Metadata directory: metadata/projects/{name}-{id}/chatrooms/{chatroom_name}/
        - meta.json in metadata directory

        Args:
            project_id: The project ID
            project_name: The project name
            chatroom_name: The chatroom name
            participants: List of participant dicts with 'id' and 'name' keys
            created_at: Creation timestamp
            ctx: ModelContext for filesystem operations

        Returns:
            The created Chatroom instance

        Raises:
            ValueError: If chatroom_name is invalid
        """
        # Validate chatroom name
        chatroom_name = FileUtil.validate_fs_name(chatroom_name)

        # Build paths (directories created by FileUtil.write with ensure_dir=True)
        data_dir = (
            ctx.projects_dir
            / f"{project_name}-{project_id}"
            / "chatrooms"
            / chatroom_name
        )
        meta_dir = (
            ctx.metadata_dir
            / f"{project_name}-{project_id}"
            / "chatrooms"
            / chatroom_name
        )

        # Create files/ directory upfront (may never have files written to it)
        files_dir = data_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        # Write meta.json (participants derived from PARTICIPANTS.ndjson, not stored)
        chatroom_data = {
            "id": chatroom_name,  # Use name as ID for compatibility
            "name": chatroom_name,
            "project_id": project_id,
            "project_name": project_name,
            "created_at": created_at.isoformat() if created_at else None,
        }
        FileUtil.write(meta_dir / "meta.json", chatroom_data, "json", atomic=True)

        # Write initial participants to PARTICIPANTS.ndjson
        participants_path = data_dir / "PARTICIPANTS.ndjson"
        for participant in participants:
            participant_entry = {
                "agent_id": participant["id"],
                "agent_name": participant["name"],
                "invited_at": created_at.isoformat() if created_at else None,
                "external": bool(participant.get("external", False)),
            }
            FileUtil.write(
                participants_path,
                participant_entry,
                "ndjson",
                mode="a",
                ensure_dir=True,
                atomic=False,
            )

        # Return the created chatroom
        instance = Chatroom.model_validate(chatroom_data)
        object.__setattr__(instance, "_ctx", ctx)
        return instance

    def add_participant(
        self,
        participant_id: str,
        participant_name: str,
        timestamp: datetime,
        *,
        external: bool = False,
    ) -> None:
        """Add a participant to an existing chatroom's PARTICIPANTS.ndjson.

        Idempotent: no-op if participant is already a member.

        Args:
            participant_id: The participant's ID
            participant_name: The participant's display name
            timestamp: When the participant was added
            external: True if this membership is a ghost — the participant's
                runner is expected to skip processing this chatroom (the FD
                tunnel handles delivery instead).
        """
        if participant_id in self._chatroom.participants:
            return  # Already a member
        participant_entry = {
            "agent_id": participant_id,
            "agent_name": participant_name,
            "invited_at": timestamp.isoformat() if timestamp else None,
            "external": bool(external),
        }
        FileUtil.write(
            self._chatroom.participants_path,
            participant_entry,
            "ndjson",
            mode="a",
            ensure_dir=True,
            atomic=False,
        )

    def append_message(self, message: ChatMessage) -> None:
        """Append a message to CHATS.ndjson.

        Args:
            message: The ChatMessage to append
        """
        FileUtil.write(
            self._chatroom.chats_path,
            message.model_dump(by_alias=True),
            "ndjson",
            mode="a",
            ensure_dir=True,
            atomic=False,
        )

    def append_file_event(self, event: ChatFileEvent) -> None:
        """Append a file-touched event to CHATS.ndjson.

        Args:
            event: The ChatFileEvent to append
        """
        FileUtil.write(
            self._chatroom.chats_path,
            event.model_dump(by_alias=True),
            "ndjson",
            mode="a",
            ensure_dir=True,
            atomic=False,
        )

    def append_batch_timeout(self, event: ChatBatchTimeoutEvent) -> None:
        """Append a batch-timeout event to CHATS.ndjson.

        Args:
            event: The ChatBatchTimeoutEvent to append
        """
        FileUtil.write(
            self._chatroom.chats_path,
            event.model_dump(by_alias=True),
            "ndjson",
            mode="a",
            ensure_dir=True,
            atomic=False,
        )

    def write_file(self, filename: str, content: bytes) -> None:
        """Write a file to the files/ directory.

        Args:
            filename: The filename to write (may include subdirectories)
            content: The file content as bytes
        """
        file_path = self._chatroom.files_dir / filename
        FileUtil.write(file_path, content, "bytes", atomic=False)

    def clear_history(
        self,
        archive_filename: Optional[str],
        cleared_at: datetime,
        cleared_through_version: int,
    ) -> None:
        """Wipe CHATS.ndjson, archiving prior contents alongside.

        Idempotent rewrite triggered by the CHATROOM_CLEARED changelog entry —
        invoked identically on the server (where the entry was minted) and on
        every runner that syncs the entry, so all sides end up with the same
        empty CHATS.ndjson plus the same .bak sibling.

        Args:
            archive_filename: Name to move existing CHATS.ndjson to (relative
                to chatroom dir). None when CHATS.ndjson was already empty
                or missing — nothing is moved in that case.
            cleared_at: Timestamp to stamp into meta.json.
            cleared_through_version: Changelog version this clear supersedes;
                stamped into meta.json so consumers can show "cleared at vN".
        """
        chats_path = self._chatroom.chats_path
        if archive_filename and chats_path.exists() and chats_path.stat().st_size > 0:
            archive_path = chats_path.parent / archive_filename
            shutil.move(str(chats_path), str(archive_path))
        # Touch an empty CHATS.ndjson so subsequent appends land cleanly even
        # if the original was missing or just got moved.
        chats_path.parent.mkdir(parents=True, exist_ok=True)
        chats_path.write_text("", encoding="utf-8")

        # Stamp meta.json with clear info (preserves all other fields).
        meta = FileUtil.read(self._chatroom.meta_path, "json", default=None) or {}
        meta["last_cleared_at"] = cleared_at.isoformat()
        meta["last_cleared_through_version"] = cleared_through_version
        FileUtil.write(self._chatroom.meta_path, meta, "json", atomic=True)
