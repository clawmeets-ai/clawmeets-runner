# SPDX-License-Identifier: MIT
"""
clawmeets/api/responses.py
API enums and flat response models.

This module is Layer 0 - no dependencies on other clawmeets modules except sync/.
Contains AgentStatus enum and response models for HTTP API.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from clawmeets.sync.changelog import ChangelogEntry


# ---------------------------------------------------------------------------
# Enums (Layer 0 - no dependencies on models/)
# ---------------------------------------------------------------------------

class AgentStatus(str, Enum):
    """Agent online/offline status."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    RATE_LIMITED = "rate_limited"


class AgentResponse(BaseModel):
    """API response model for agent data.

    Pure DTO with no persistence methods. Used as response_model in routes.
    Replaces AgentCard as the API response type.
    """
    id: str
    name: str
    description: str
    capabilities: list[str] = Field(default_factory=list)
    status: AgentStatus = AgentStatus.OFFLINE
    registered_at: datetime
    last_heartbeat: datetime  # Epoch (1970-01-01) if no heartbeat received yet
    discoverable_through_registry: bool = True
    registered_by: Optional[str] = None
    is_verified: bool = False
    local_settings: dict = Field(default_factory=dict)  # knowledge_dir, use_chrome


class AgentSearchResponse(BaseModel):
    """Paginated search results for agent discovery."""
    agents: list[AgentResponse]
    total: int
    offset: int
    limit: int


class AgentRegistrationResponse(BaseModel):
    """Flat response from agent registration.

    Use agent_id to lookup the full Agent if needed.
    """
    agent_id: str
    agent_name: str
    token: str
    description: str
    status: AgentStatus = AgentStatus.OFFLINE
    registered_at: datetime
    discoverable_through_registry: bool = True


class CreateUserResponse(BaseModel):
    """Flat response from user creation with assistant info.

    Contains both user and assistant agent details in a flat structure.
    Use user_id or assistant_agent_id to lookup full models if needed.
    """
    user_id: str
    username: str
    is_admin: bool
    user_created_at: datetime
    assistant_agent_id: str
    assistant_agent_name: str
    assistant_token: str


class RegisterUserResponse(BaseModel):
    """Response from self-registration.

    Contains user info, assistant credentials, and verification message.
    """
    user_id: str
    username: str
    message: str
    assistant_agent_id: str
    assistant_agent_name: str
    assistant_token: str


class ChangelogBatch(BaseModel):
    """Response from changelog sync endpoint.

    Contains changelog entries between versions for a project.
    References ChangelogEntry from sync/ (Layer 0).
    """
    project_id: str
    from_version: int
    to_version: int
    entries: list[ChangelogEntry] = Field(default_factory=list)


class ParticipantProjectResponse(BaseModel):
    """Response for participant's project membership with sync info.

    Contains project metadata plus current changelog version to enable
    efficient delta sync (skip sync if local_version == current_version).

    Used by the unified /participants/{id}/projects endpoint for all
    participant types (users, agents, assistants).
    """
    id: str
    name: str
    status: str
    current_version: int
    coordinator_id: str
    git_url: str = ""
    is_viewer: bool = False
