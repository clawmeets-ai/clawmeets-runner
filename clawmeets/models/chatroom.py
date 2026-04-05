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

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field, PrivateAttr, computed_field

from ..utils.file_io import FileUtil
from ..utils.validation import validate_name
from .chat_message import ChatMessage
from .participant import Participant

if TYPE_CHECKING:
    from .context import ModelContext


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

    @property
    def is_shared_context_room(self) -> bool:
        """Check if this is the shared-context room for project-wide knowledge."""
        return self.name.startswith("shared-context")

    @property
    def is_user_communication_room(self) -> bool:
        """Check if this is the user-communication room for user<->assistant chat."""
        return self.name.startswith("user-communication")

    @property
    def is_dm_chatroom(self) -> bool:
        """Check if this is a DM (direct message) chatroom.

        DM chatrooms are named "dm-{agent-name}" and are used for direct
        messaging between users and specific agents.
        """
        return self.name.startswith("dm-")

    @property
    def dm_agent_name(self) -> Optional[str]:
        """Extract the agent name from a DM chatroom name.

        Returns:
            The agent name if this is a DM chatroom, None otherwise.
        """
        if not self.is_dm_chatroom:
            return None
        return self.name[3:]  # Remove "dm-" prefix

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

    def get_messages(self, limit: int = 9999999) -> list:
        """Load messages from this chatroom.

        Args:
            limit: Maximum number of messages to return (None for all)

        Returns:
            List of ChatMessage objects (most recent last)
        """
        if not self.chats_path.exists():
            return []
        result = []
        for line_data in FileUtil.read(self.chats_path, "ndjson"):
            result.append(ChatMessage.model_validate(line_data))
        return result[-limit:]

    def get_messages_since(self, since_message_id: str) -> list:
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
    ) -> "Chatroom":
        """Load chatroom by name.

        Args:
            project_id: The project ID
            chatroom_name: The chatroom name (unique within project)
            ctx: ModelContext for filesystem operations

        Returns:
            Chatroom
        """
        from .project import Project

        # Load project first to get project_name
        project = Project.get(project_id, ctx)

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
            raise ValueError(f"Chatroom {chatroom_name} not found in project {project_id}")
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
        chatroom_name = validate_name(chatroom_name)

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

    def add_participant(self, participant_id: str, participant_name: str, timestamp: datetime) -> None:
        """Add a participant to an existing chatroom's PARTICIPANTS.ndjson.

        Idempotent: no-op if participant is already a member.

        Args:
            participant_id: The participant's ID
            participant_name: The participant's display name
            timestamp: When the participant was added
        """
        if participant_id in self._chatroom.participants:
            return  # Already a member
        participant_entry = {
            "agent_id": participant_id,
            "agent_name": participant_name,
            "invited_at": timestamp.isoformat() if timestamp else None,
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

    def write_file(self, filename: str, content: bytes) -> None:
        """Write a file to the files/ directory.

        Args:
            filename: The filename to write (may include subdirectories)
            content: The file content as bytes
        """
        file_path = self._chatroom.files_dir / filename
        FileUtil.write(file_path, content, "bytes", atomic=False)
