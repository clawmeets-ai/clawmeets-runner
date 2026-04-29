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
    MCP_SYNC = "mcp_sync"                  # Server notifies client to install/uninstall an MCP server
    AGENT_SETTINGS_CHANGE = "agent_settings_change"  # Server notifies agent of local_settings update
    CANCEL_LLM = "cancel_llm"              # Server tells runner to kill the in-flight LLM subprocess
    ACTIVE_WORK_CHANGE = "active_work_change"  # PendingWork state changed in a chatroom


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


class McpSyncPayload(BaseModel):
    """Payload for MCP_SYNC messages.

    The manifest (`launch` + `auth` spec) is sent on install so the runner
    can cache it locally and doesn't need to round-trip to the server for
    every .mcp.json render. It's None on uninstall.
    """
    agent_id: str
    agent_name: str
    action: str  # "install" or "uninstall"
    mcp_name: str
    manifest: dict | None = None


class AgentSettingsChangePayload(BaseModel):
    """Payload for AGENT_SETTINGS_CHANGE messages."""
    agent_id: str
    agent_name: str
    local_settings: dict  # knowledge_dir, use_chrome


class CancelLLMPayload(BaseModel):
    """Payload for CANCEL_LLM messages.

    Identifies the specific in-flight LLM invocation to terminate. The runner
    keys its in-flight tasks by (project_id, chatroom_name); agent_id is
    included so the runner can defensively confirm the message was routed to
    the right participant before acting.
    """
    agent_id: str
    project_id: str
    chatroom_name: str


class ActiveWorkChangePayload(BaseModel):
    """Payload for ACTIVE_WORK_CHANGE messages.

    Sent whenever a WorkTracker PendingWork entry transitions — create, each
    individual response, clear, or project-wide clear. ``active_participants``
    is the list of expected responders who have not yet replied; an empty list
    means the batch is complete (or was cancelled/timed out).

    One signal serves both the sidebar "actively being worked on" indicator
    (is_active = len(active_participants) > 0) and the in-room typing indicator
    (which renders one chip per participant id).
    """
    project_id: str
    project_name: str
    chatroom_name: str
    active_participants: list[str]


class ControlEnvelope(BaseModel):
    """Lightweight WebSocket notification - never carries file content."""
    type: ControlMessageType
    payload: Union[ChangelogUpdatePayload, AgentStatusChangePayload, ProjectDeletedPayload, SkillSyncPayload, McpSyncPayload, AgentSettingsChangePayload, CancelLLMPayload, ActiveWorkChangePayload, dict] = Field(default_factory=dict)

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
        elif self.type == ControlMessageType.MCP_SYNC:
            if not isinstance(self.payload, McpSyncPayload):
                raise ValueError(f"control message type {self.type} requires McpSyncPayload")
        elif self.type == ControlMessageType.AGENT_SETTINGS_CHANGE:
            if not isinstance(self.payload, AgentSettingsChangePayload):
                raise ValueError(f"control message type {self.type} requires AgentSettingsChangePayload")
        elif self.type == ControlMessageType.CANCEL_LLM:
            if not isinstance(self.payload, CancelLLMPayload):
                raise ValueError(f"control message type {self.type} requires CancelLLMPayload")
        elif self.type == ControlMessageType.ACTIVE_WORK_CHANGE:
            if not isinstance(self.payload, ActiveWorkChangePayload):
                raise ValueError(f"control message type {self.type} requires ActiveWorkChangePayload")
        return self
