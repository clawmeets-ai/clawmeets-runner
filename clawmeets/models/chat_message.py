# SPDX-License-Identifier: MIT
"""
clawmeets/models/chat_message.py
Chat message model for CHATS.ndjson deserialization.

This module is part of Layer 1 (depends on Layer 0 sync/changelog).
It provides the ChatMessage model for reading messages from CHATS.ndjson files.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from clawmeets.sync.changelog import MessagePayload


class ChatMessage(BaseModel):
    """One entry in a CHATS file (newline-delimited JSON).

    This is a Layer 1 model used for:
    - Reading messages from CHATS.ndjson files
    - Passing to participant callbacks (on_message, etc.)
    - Type hints in participant methods

    MessagePayload in sync/changelog.py (Layer 0) now contains flat fields.
    Use from_message_payload() to convert from payload to ChatMessage.
    """
    model_config = {"frozen": True}

    id: str
    ts: datetime
    from_participant_id: str
    from_participant_name: str  # Required - authenticated participant must have a name
    content: str
    expects_response_from: list[str] = Field(default_factory=list)
    is_ack: bool = Field(default=False)  # System-only acknowledgment marker

    def to_log_line(self) -> str:
        """Serialize to NDJSON line for CHATS.ndjson."""
        return self.model_dump_json()

    @classmethod
    def from_log_line(cls, line: str) -> "ChatMessage":
        """Deserialize from NDJSON line."""
        return cls.model_validate_json(line)

    @classmethod
    def from_message_payload(cls, payload: "MessagePayload") -> "ChatMessage":
        """Convert from MessagePayload (Layer 0) to ChatMessage (Layer 1).

        Args:
            payload: The MessagePayload from a changelog entry

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
        )
