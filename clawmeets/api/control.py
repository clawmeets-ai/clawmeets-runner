# SPDX-License-Identifier: MIT
"""
clawmeets/api/control.py
WebSocket control envelope model.

Part of the API layer (Layer 0) alongside actions.py and responses.py.
Defines the WebSocket protocol types used between runner and server.
"""
from __future__ import annotations

from enum import Enum
from typing import Union

from pydantic import BaseModel, Field, model_validator


class ControlMessageType(str, Enum):
    """WebSocket control message types.

    The WebSocket protocol is notification-only:
    - Server -> Client: CHANGELOG_UPDATE, AGENT_STATUS_CHANGE
    - Client -> Server: HEARTBEAT

    All data (messages, files, batches) is delivered via changelog entries
    fetched over HTTP. Legacy direct-dispatch types (INVITE, MESSAGE,
    BATCH_COMPLETE, etc.) have been removed.
    """
    CHANGELOG_UPDATE = "changelog_update"  # Server notifies client of new changelog entries
    HEARTBEAT = "heartbeat"                # Connection health check
    AGENT_STATUS_CHANGE = "agent_status_change"  # Server notifies clients of agent online/offline
    PROJECT_DELETED = "project_deleted"            # Server notifies clients that a project was deleted
    SKILL_SYNC = "skill_sync"              # Server notifies client to install/uninstall a skill
    AGENT_SETTINGS_CHANGE = "agent_settings_change"  # Server notifies agent of local_settings update


class ChangelogUpdatePayload(BaseModel):
    """Payload for CHANGELOG_UPDATE messages.

    Minimal payload - clients should invalidate all project caches.
    """
    project_id: str
    project_name: str
    new_version: int
    coordinator_id: str


class AgentStatusChangePayload(BaseModel):
    """Payload for AGENT_STATUS_CHANGE messages."""
    agent_id: str
    agent_name: str
    new_status: str  # AgentStatus value: "online", "offline", "busy"


class ProjectDeletedPayload(BaseModel):
    """Payload for PROJECT_DELETED messages."""
    project_id: str
    project_name: str


class SkillSyncPayload(BaseModel):
    """Payload for SKILL_SYNC messages."""
    agent_id: str
    agent_name: str
    action: str  # "install" or "uninstall"
    skill_name: str
    skill_content: str | None = None  # Full SKILL.md content for install; None for uninstall


class AgentSettingsChangePayload(BaseModel):
    """Payload for AGENT_SETTINGS_CHANGE messages."""
    agent_id: str
    agent_name: str
    local_settings: dict  # knowledge_dir, use_chrome


class ControlEnvelope(BaseModel):
    """Lightweight WebSocket notification - never carries file content."""
    type: ControlMessageType
    payload: Union[ChangelogUpdatePayload, AgentStatusChangePayload, ProjectDeletedPayload, SkillSyncPayload, AgentSettingsChangePayload, dict] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_required_fields_for_type(self) -> "ControlEnvelope":
        # Enforce typed payloads for each message type
        if self.type == ControlMessageType.CHANGELOG_UPDATE:
            if not isinstance(self.payload, ChangelogUpdatePayload):
                raise ValueError(f"control message type {self.type} requires ChangelogUpdatePayload")
        elif self.type == ControlMessageType.AGENT_STATUS_CHANGE:
            if not isinstance(self.payload, AgentStatusChangePayload):
                raise ValueError(f"control message type {self.type} requires AgentStatusChangePayload")
        elif self.type == ControlMessageType.PROJECT_DELETED:
            if not isinstance(self.payload, ProjectDeletedPayload):
                raise ValueError(f"control message type {self.type} requires ProjectDeletedPayload")
        elif self.type == ControlMessageType.SKILL_SYNC:
            if not isinstance(self.payload, SkillSyncPayload):
                raise ValueError(f"control message type {self.type} requires SkillSyncPayload")
        elif self.type == ControlMessageType.AGENT_SETTINGS_CHANGE:
            if not isinstance(self.payload, AgentSettingsChangePayload):
                raise ValueError(f"control message type {self.type} requires AgentSettingsChangePayload")
        return self
