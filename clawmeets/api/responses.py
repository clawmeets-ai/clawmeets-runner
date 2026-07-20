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
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Valid provider values for a named model config. Kept in sync with
# ``clawmeets.models.model_config.VALID_CONFIG_PROVIDERS`` — duplicated here
# (not imported) because api/ is Layer 0 and must not depend on models/. Used
# as a Literal so a bad ``provider`` yields a Pydantic list-shape 422.
_CONFIG_PROVIDER = Literal[
    "claude", "openai", "gemini", "opencode",
    "claude-api", "openai-api", "gemini-api", "openrouter-api", "openrouter-native",
]
_CONFIG_NAME_PATTERN = r"^[A-Za-z0-9 _-]+$"
# NOTE: a per-config ``api_key`` is REQUIRED (non-empty) for the keyed (BYO-key)
# providers (``*-api`` / ``openrouter-native``) and OPTIONAL for the CLI providers
# (claude/openai/gemini/opencode). That check lives in ``model_config.add_config``
# (not a Pydantic validator here) so a missing key raises the single-string 422
# shape ``{"detail": "api_key is required for provider '<p>'"}`` via the route's
# ValueError→HTTPException(422) path — a model_validator would emit the wrong
# Pydantic list-shape detail.

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
    # NOTE: the agent-level API key was RETIRED — keys now live per model config
    # (``model_configs[].api_key``, write-only, surfaced only as ``api_key_set``).
    # Any legacy ``local_settings.llm_api_key`` is stripped from this read path
    # (``_strip_legacy_llm_api_key``); there is no agent-level key field anymore.
    # Named model configs + designated default (model-selector revamp). Carried
    # on the response so the runner can reconcile them onto its self-card and the
    # frontend settings page can render the list. Configs are ModelConfig wire
    # dicts; default is the name of the default config (null when the list is empty).
    model_configs: list[dict] = Field(default_factory=list)
    default_model_config_name: Optional[str] = None
    last_reflected_at: Optional[datetime] = None  # Last successful reflection cycle
    last_synced_at: Optional[datetime] = None  # Last successful DWH sync trigger reply


class ModelConfig(BaseModel):
    """One named model config (canonical wire shape).

    Identity/key = ``name`` (unique per agent). ``source`` and ``created_at`` are
    server-stamped and read-only. The raw per-config ``api_key`` is write-only and
    NEVER returned — read paths expose only ``api_key_set`` (True iff a non-empty
    key is stored). The frontend derives ``isDefault`` from the list response's
    ``default`` field.
    """
    model_config = ConfigDict(protected_namespaces=())  # allow a field named "model"

    name: str
    provider: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    source: str = "manual"  # "auto" (registration probe) | "manual"
    created_at: datetime
    api_key_set: bool = False  # write-only indicator; True iff a raw key is stored


class ModelConfigCreate(BaseModel):
    """POST body — create a named config.

    ``name`` / ``provider`` are validated here so a bad value returns a Pydantic
    list-shape 422 (the canonical validation error shape).
    """
    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(..., min_length=1, max_length=64, pattern=_CONFIG_NAME_PATTERN)
    provider: _CONFIG_PROVIDER
    model: Optional[str] = None
    base_url: Optional[str] = None
    # Write-only secret. REQUIRED (non-empty) for keyed providers, OPTIONAL for
    # CLI providers — enforced in ``model_config.add_config`` (single-string 422).
    api_key: Optional[str] = None


class ModelConfigPatch(BaseModel):
    """PATCH body — partial update (all fields optional; rename via ``name``).

    Only provided fields are applied (``model_dump(exclude_unset=True)``), so
    ``model``/``base_url`` can be explicitly set to null to clear them. A
    provided ``name``/``provider`` is validated (list-shape 422 on a bad value).
    """
    model_config = ConfigDict(protected_namespaces=())

    name: Optional[str] = Field(
        None, min_length=1, max_length=64, pattern=_CONFIG_NAME_PATTERN
    )
    provider: Optional[_CONFIG_PROVIDER] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    # Write-only secret, TRI-STATE (driven by ``model_dump(exclude_unset=True)``):
    # field OMITTED → leave the stored key unchanged; non-empty string → set/replace;
    # empty string "" (or null) → clear the stored key. Applied in ``update_config``.
    api_key: Optional[str] = None


class ModelConfigList(BaseModel):
    """GET / set-default response — the configs plus the default name."""
    configs: list[ModelConfig] = Field(default_factory=list)
    default: Optional[str] = None


class SetDefaultBody(BaseModel):
    """PUT .../model-configs/default body."""
    name: str


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
    report_published_at: Optional[datetime] = None  # Same meaning as Project.report_published_at: ISO ts string | null, non-null EXACTLY when the project has a real completion report. Present on every row of the desk's non-admin list (GET /participants/{id}/projects).
