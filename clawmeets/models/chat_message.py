# SPDX-License-Identifier: MIT
"""
clawmeets/models/chat_message.py
Log entry models for CHATS.ndjson.

This module is part of Layer 1 (depends on Layer 0 sync/changelog).
It provides:
- ChatMessage: regular chat messages (entry_type="message")
- ChatFileEvent: file-touched events (entry_type="file_created" | "file_updated")
- ChatLogEntry: discriminated union for CHATS.ndjson lines
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

if TYPE_CHECKING:
    from clawmeets.sync.changelog import MessagePayload


class ChatMessage(BaseModel):
    """One chat message row in CHATS.ndjson.

    MessagePayload in sync/changelog.py (Layer 0) carries the wire-level
    fields. Use from_message_payload() to convert from payload to ChatMessage.

    version/source_version are populated from the owning ChangelogEntry —
    they let downstream consumers (frontend, CLI listeners) link a message
    to the entry that triggered it without re-reading the changelog.
    """
    model_config = {"frozen": True}

    entry_type: Literal["message"] = "message"
    id: str
    ts: datetime
    from_participant_id: str
    from_participant_name: str  # Required - authenticated participant must have a name
    content: str
    expects_response_from: list[str] = Field(default_factory=list)
    is_ack: bool = Field(default=False)  # System-only acknowledgment marker
    version: int | None = None  # This entry's changelog version
    source_version: int | None = None  # Version of the entry that triggered this message

    def to_log_line(self) -> str:
        """Serialize to NDJSON line for CHATS.ndjson."""
        return self.model_dump_json()

    @classmethod
    def from_log_line(cls, line: str) -> "ChatMessage":
        """Deserialize from NDJSON line."""
        return cls.model_validate_json(line)

    @classmethod
    def from_message_payload(
        cls,
        payload: "MessagePayload",
        *,
        version: int | None = None,
        source_version: int | None = None,
    ) -> "ChatMessage":
        """Convert from MessagePayload (Layer 0) to ChatMessage (Layer 1).

        Args:
            payload: The MessagePayload from a changelog entry
            version: The owning ChangelogEntry's version (optional)
            source_version: The owning ChangelogEntry's source_version (optional)

        Returns:
            ChatMessage instance with the same data
        """
        return cls(
            id=payload.id,
            ts=payload.ts,
            from_participant_id=payload.from_participant_id,
            from_participant_name=payload.from_participant_name,
            content=payload.content,
            expects_response_from=payload.expects_response_from,
            is_ack=payload.is_ack,
            version=version,
            source_version=source_version,
        )


class ChatFileEvent(BaseModel):
    """One file-touched row in CHATS.ndjson.

    Emitted alongside FILE_CREATED / FILE_UPDATED changelog entries so the
    chat stream carries an inline record of which files the sender touched.
    An agent's ActionBlock tags its reply and its file uploads with the
    same source_version (the triggering mention), so the web UI groups a
    file event under its sibling message by matching source_version +
    from_participant_id.
    """
    model_config = {"frozen": True}

    entry_type: Literal["file_created", "file_updated"]
    ts: datetime
    from_participant_id: str
    from_participant_name: str
    filename: str
    version: int | None = None  # This entry's changelog version
    source_version: int | None = None  # Message entry this file belongs to


# Discriminated union of log-entry rows persisted to CHATS.ndjson.
# Existing rows without `entry_type` default to ChatMessage (entry_type="message").
ChatLogEntry = Annotated[
    Union[ChatMessage, ChatFileEvent],
    Field(discriminator="entry_type"),
]

_LOG_ENTRY_ADAPTER: TypeAdapter[ChatLogEntry] = TypeAdapter(ChatLogEntry)


def parse_log_line(line_data: dict | str) -> ChatLogEntry:
    """Parse a CHATS.ndjson line into a ChatLogEntry.

    Handles legacy rows without `entry_type` by defaulting to "message".

    Args:
        line_data: Either a dict (already parsed) or a raw JSON string.

    Returns:
        ChatMessage or ChatFileEvent depending on entry_type.
    """
    if isinstance(line_data, str):
        return _LOG_ENTRY_ADAPTER.validate_json(line_data)
    data = line_data if "entry_type" in line_data else {**line_data, "entry_type": "message"}
    return _LOG_ENTRY_ADAPTER.validate_python(data)
