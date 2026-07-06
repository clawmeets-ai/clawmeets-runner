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
    user_teams: list[str] = Field(default_factory=list)  # Owner-defined team labels
    # Defaults seeded into new FD-tunnel DM projects coordinated by this
    # agent (agent_names / agent_teams on the resulting project). Editing
    # them does not retroactively rewrite existing FD-tunnel projects.
    default_invitable_agents: list[str] = Field(default_factory=list)
    default_invitable_teams: list[str] = Field(default_factory=list)
    local_settings: dict = Field(default_factory=dict)  # knowledge_dir, llm_provider, llm_model
    last_reflected_at: Optional[datetime] = None  # Last successful reflection cycle
    last_synced_at: Optional[datetime] = None  # Last successful DWH sync trigger reply


class AgentSearchResponse(BaseModel):
    """Paginated search results for agent discovery."""
    agents: list[AgentResponse]
    total: int
    offset: int
    limit: int


class AgentRegistrationResponse(BaseModel):
    """Flat response from agent registration.

    Use agent_id to lookup the full Agent if needed.

    ``token`` is None on the re-register path: re-registration is meant to be
    idempotent (e.g. ``clawmeets init`` re-runs) and must not silently
    invalidate the in-memory credential of any already-running runner.
    Token rotation is a deliberate op (delete + register-fresh).
    """
    agent_id: str
    agent_name: str
    token: Optional[str] = None
    description: str
    status: AgentStatus = AgentStatus.OFFLINE
    registered_at: datetime
    discoverable_through_registry: bool = True
    user_teams: list[str] = Field(default_factory=list)


class UserResponse(BaseModel):
    """Flat response from user creation (admin and self-register).

    Used by both POST /users and POST /auth/register. The assistant agent is
    created separately via `clawmeets assistant register` after signup, so it
    is not part of this response.
    """
    user_id: str
    username: str
    user_created_at: datetime


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
    is_viewer: bool = False
    created_by: str
    surface: Optional[str] = None  # "regular" | "dm"
    display_name: Optional[str] = None  # raw model-set label; frontend renders `display_name ?? name`
    last_modified: datetime  # ISO-8601, non-null; sidebar sorts the PROJECTS list by this desc
