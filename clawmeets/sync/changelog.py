# SPDX-License-Identifier: MIT
"""
clawmeets/sync/changelog.py
Changelog entry types and payloads for the unified changelog.

This module is part of Layer 0 (pure - no domain model dependencies).
It defines the changelog types that are used for event sourcing.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Union

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    pass


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
    PROJECT_COMPLETED = "project_completed"  # Project completed
    BATCH_COMPLETE = "batch_complete"  # All expected agents responded
    BATCH_TIMEOUT = "batch_timeout"    # Some agents didn't respond in time
    PARTICIPANT_ADDED = "participant_added"  # Participant added to existing room


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

    Project-level entries (PROJECT_CREATED, PROJECT_COMPLETED) do NOT
    inherit from this class and have no chatroom_name.
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
    # chatroom_name inherited from ChatroomPayload
    id: str
    ts: datetime
    from_participant_id: str
    from_participant_name: str  # Required - authenticated participant must have a name
    content: str
    expects_response_from: list[str] = Field(default_factory=list)
    is_ack: bool = Field(default=False)


class FilePayload(ChatroomPayload):
    """Payload for FILE_CREATED and FILE_UPDATED entries in unified changelog."""
    filename: str
    content_b64: str  # Base64-encoded file content (required)
    sha256: str       # SHA256 hash of the content (required)


class RoomCreatedParticipant(BaseModel):
    """Participant info for room creation."""
    id: str
    name: str


class RoomCreatedPayload(ChatroomPayload):
    """Payload for ROOM_CREATED entries in unified changelog."""
    # chatroom_name inherited from ChatroomPayload
    participants: list[RoomCreatedParticipant] = Field(default_factory=list)


class ProjectCompletedPayload(BaseModel):
    """Payload for PROJECT_COMPLETED entries in unified changelog.

    Project-level entry - no chatroom_name field.
    """
    pass  # No fields needed - the entry type itself indicates completion


class BatchCompletePayload(ChatroomPayload):
    """Payload for BATCH_COMPLETE entries in unified changelog."""
    message_id: str
    coordinator_id: str
    responded_participants: list[str]


class BatchTimeoutPayload(ChatroomPayload):
    """Payload for BATCH_TIMEOUT entries in unified changelog."""
    message_id: str
    coordinator_id: str
    responded_participants: list[str]
    timed_out_participants: list[str]


class ParticipantAddedPayload(ChatroomPayload):
    """Payload for PARTICIPANT_ADDED entries in unified changelog.

    Used when adding a participant to an existing chatroom (e.g., auto-adding
    agents to shared-context when they join a project via a work room).
    """
    participant_id: str
    participant_name: str


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
    agent_pool: str = "verified"  # "owned", "verified", or "all"


# Union type for changelog payloads
ChangelogPayload = Union[
    ProjectCreatedPayload,
    MessagePayload,
    FilePayload,
    RoomCreatedPayload,
    ProjectCompletedPayload,
    BatchCompletePayload,
    BatchTimeoutPayload,
    ParticipantAddedPayload,
]


# ---------------------------------------------------------------------------
# Changelog Entry
# ---------------------------------------------------------------------------

class ChangelogEntry(BaseModel):
    """Single entry in the unified changelog.

    The unified changelog provides monotonic versioning across all events
    (messages, file changes, invites, etc.) in a project.

    Chatroom-scoped entries have chatroom_name in their payload (via ChatroomPayload).
    Project-level entries (PROJECT_CREATED, PROJECT_COMPLETED) have no chatroom_name.
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
                "project_completed": ProjectCompletedPayload,
                "batch_complete": BatchCompletePayload,
                "batch_timeout": BatchTimeoutPayload,
                "participant_added": ParticipantAddedPayload,
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
            ChangelogEntryType.PROJECT_COMPLETED: ProjectCompletedPayload,
            ChangelogEntryType.BATCH_COMPLETE: BatchCompletePayload,
            ChangelogEntryType.BATCH_TIMEOUT: BatchTimeoutPayload,
            ChangelogEntryType.PARTICIPANT_ADDED: ParticipantAddedPayload,
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
        return self.model_dump_json()

    @classmethod
    def from_log_line(cls, line: str) -> "ChangelogEntry":
        """Deserialize from NDJSON line."""
        return cls.model_validate_json(line)


