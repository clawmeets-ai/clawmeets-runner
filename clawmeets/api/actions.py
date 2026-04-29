# SPDX-License-Identifier: MIT
"""
clawmeets/api/actions.py
Action types for Claude Code output parsing.

This module is part of Layer 0 (pure - no domain model dependencies).
It defines the action types that Claude emits via structured output.
"""
from enum import Enum
from typing import Any, Union

from pydantic import BaseModel, Field


# JSON Schema for Claude CLI --json-schema option
# This enables structured output with validated JSON matching the schema
#
# NOTE: Addressing is done via @mentions in message content, not via a separate field.
# - "@agent-name" in content -> agent is addressed and should respond
# - "agent-name" (no @) -> reference only, not addressed

# Common action schemas
_REPLY_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "reply"},
        "room": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["type", "room", "content"],
    "additionalProperties": False
}

_UPDATE_FILE_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "update_file"},
        "room": {"type": "string"},
        "file_path": {"type": "string"},
    },
    "required": ["type", "room", "file_path"],
    "additionalProperties": False
}

_CREATE_ROOM_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "create_room"},
        "name": {"type": "string"},
        "invite": {
            "type": "array",
            "items": {"type": "string"}
        },
        "init_message": {"type": "string"},
    },
    "required": ["type", "name", "invite", "init_message"],
    "additionalProperties": False
}

_PROJECT_COMPLETED_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "project_completed"},
    },
    "required": ["type"],
    "additionalProperties": False
}

# Worker schema: reply and update_file only (no delegation actions)
WORKER_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "oneOf": [
                    _REPLY_ACTION_SCHEMA,
                    _UPDATE_FILE_ACTION_SCHEMA,
                ]
            }
        }
    },
    "required": ["actions"],
    "additionalProperties": False
}

# Coordinator schema: all actions including create_room and project_completed
COORDINATOR_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "oneOf": [
                    _REPLY_ACTION_SCHEMA,
                    _UPDATE_FILE_ACTION_SCHEMA,
                    _CREATE_ROOM_ACTION_SCHEMA,
                    _PROJECT_COMPLETED_ACTION_SCHEMA,
                ]
            }
        }
    },
    "required": ["actions"],
    "additionalProperties": False
}


class ActionType(str, Enum):
    """Action types parsed from Claude Code output."""
    REPLY = "reply"
    UPDATE_FILE = "update_file"
    CREATE_ROOM = "create_room"
    PROJECT_COMPLETED = "project_completed"


class ReplyAction(BaseModel):
    """Reply to a chatroom with a message.

    Addressing is done via @mentions in the content:
    - "@agent-name" in content -> agent is addressed and should respond
    - "agent-name" (no @) -> reference only, not addressed
    - No @mentions -> informational message, no one is expected to respond
    """
    type: ActionType = ActionType.REPLY
    room: str
    content: str


class UpdateFileAction(BaseModel):
    """Update a file in a chatroom."""
    type: ActionType = ActionType.UPDATE_FILE
    room: str
    file_path: str


class CreateRoomAction(BaseModel):
    """Create a new chatroom.

    Addressing in init_message is done via @mentions:
    - "@agent-name" in init_message -> agent is addressed and should respond
    - "agent-name" (no @) -> reference only, not addressed
    """
    type: ActionType = ActionType.CREATE_ROOM
    name: str
    invite: list[str]           # agent names
    init_message: str           # initial message with @mentions to address agents


class ProjectCompletedAction(BaseModel):
    """Mark the project as complete."""
    type: ActionType = ActionType.PROJECT_COMPLETED


# Union type used by ActionParser
Action = Union[ReplyAction, UpdateFileAction, CreateRoomAction, ProjectCompletedAction]


class ActionBlock(BaseModel):
    """Parsed output of a Claude Code invocation."""
    raw: str
    actions: list[dict[str, Any]] = Field(default_factory=list)
    source_version: int | None = None  # Version of the changelog entry that triggered this response

    def typed_actions(self) -> list[Action]:
        """Return fully typed Action objects. Unknown types are silently skipped."""
        result: list[Action] = []
        for a in self.actions:
            try:
                t = ActionType(a.get("type", ""))
            except ValueError:
                continue  # forward-compatible: ignore unknown future action types
            if t == ActionType.REPLY:
                result.append(ReplyAction(**a))
            elif t == ActionType.UPDATE_FILE:
                result.append(UpdateFileAction(**a))
            elif t == ActionType.CREATE_ROOM:
                result.append(CreateRoomAction(**a))
            elif t == ActionType.PROJECT_COMPLETED:
                result.append(ProjectCompletedAction(**a))
        return result
