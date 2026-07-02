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

    The WebSocket protocol is notification-only — every envelope is a small
    pointer/trigger; bulk data (messages, files, batches) is fetched
    separately over HTTP via the changelog.

    Direction:
    - Client -> Server: HEARTBEAT
    - Server -> Client (everything else):
        - To runners (agents) only:
            SKILL_SYNC, MCP_SYNC, AGENT_SETTINGS_CHANGE, CANCEL_LLM,
            MCP_AUTH_CODE, KNOWLEDGE_PACK_SYNC, AGENT_REGISTRY_CHANGE
        - To user UIs (web frontend) only:
            AGENT_STATUS_CHANGE, MCP_AUTH_URL_FOR_USER, AGENT_CARD_UPDATE
        - To both (fan-out via ``ws_hub.broadcast_to_project`` over every
          project participant — agents, coordinator, owner, FD requester,
          share-token viewers):
            CHANGELOG_UPDATE, PROJECT_DELETED, ACTIVE_WORK_CHANGE

    Runners act on CHANGELOG_UPDATE (wake the per-project runloop) and
    PROJECT_DELETED (tear down runloop + delete local project/sandbox/
    metadata dirs); they receive ACTIVE_WORK_CHANGE but currently no-op
    on it. UIs use the same envelopes for cache invalidation and to
    render typing / "actively working" indicators.
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
    MCP_AUTH_URL_FOR_USER = "mcp_auth_url_for_user"  # Server pushes a "Continue with Google" link to the agent's owner
    MCP_AUTH_CODE = "mcp_auth_code"        # Server forwards an OAuth code to the runner after the user finishes consent
    SKILL_AUTH_URL_FOR_USER = "skill_auth_url_for_user"  # Skill-rail sibling of MCP_AUTH_URL_FOR_USER
    SKILL_AUTH_CODE = "skill_auth_code"    # Skill-rail sibling of MCP_AUTH_CODE
    KNOWLEDGE_PACK_SYNC = "knowledge_pack_sync"  # Server notifies client to install/uninstall a knowledge pack
    AGENT_REGISTRY_CHANGE = "agent_registry_change"  # Server notifies peer runners that an agent was registered/updated
    AGENT_CARD_UPDATE = "agent_card_update"  # Server notifies the agent's owner UI of card-field bumps (last_reflected_at, last_synced_at)
    TODAY_TAB_SYNC = "today_tab_sync"  # Server notifies the owning user that a today-tab was upserted / deleted
    PROJECT_REPORT_SYNC = "project_report_sync"  # Server notifies project participants that the report was upserted / deleted


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
    action: str  # "install" | "uninstall" | "reauth"
    skill_name: str
    skill_content: str  # Full SKILL.md content — sent on both install and uninstall; consumer gates on ``action``
    # Sibling files alongside SKILL.md (template.html, render.py, etc.). Keyed
    # by forward-slash relpath under ``skills/<name>/``; each value is
    # ``{"content_b64": <base64-encoded bytes>}`` to mirror
    # ``KnowledgePackSyncPayload`` and stay JSON-safe over the wire. Empty when
    # the skill has no companion files.
    skill_files: dict[str, dict] = Field(default_factory=dict)
    # OAuth block from the registry entry, when the skill declares one (gmail
    # / gcal / gdrive / gdrive-write). Empty / None on uninstall and on skills
    # without auth. Shape: ``{"method": "google_oauth_installed", "scopes":
    # [...], "token_file": "token.json"}``. The runner reads this on install
    # to fire ``_spawn_auto_auth_skill``; consumers without an interest skip it.
    auth: dict | None = None


class McpSyncPayload(BaseModel):
    """Payload for MCP_SYNC messages.

    The manifest (`launch` + `auth` spec) is sent on install so the runner
    can cache it locally and doesn't need to round-trip to the server for
    every .mcp.json render. Empty on uninstall — consumer gates on ``action``.
    """
    agent_id: str
    agent_name: str
    action: str  # "install" or "uninstall"
    mcp_name: str
    manifest: dict = Field(default_factory=dict)


class AgentSettingsChangePayload(BaseModel):
    """Payload for AGENT_SETTINGS_CHANGE messages.

    Carries server-side card edits the runner must mirror to its local
    `card.json` so subsequent prompt builds see fresh values. ``local_settings``
    covers ``knowledge_dir`` / ``llm_provider`` / ``llm_model`` (the original
    runner config). May be ``None`` when unchanged in this envelope.
    """
    agent_id: str
    agent_name: str
    local_settings: dict | None = None  # None = unchanged in this envelope


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


class McpAuthUrlForUserPayload(BaseModel):
    """Payload for MCP_AUTH_URL_FOR_USER messages.

    Sent server -> user (the agent's owner) when the runner has built a Google
    consent URL for an MCP that needs OAuth. The user's web UI renders this
    as a "Continue with Google" link. ``state`` is the one-shot token the
    server uses to correlate the eventual /oauth/mcp/callback hit back to the
    runner; the frontend doesn't need to read it but echoing it is harmless.
    """
    agent_id: str
    agent_name: str
    mcp_name: str
    auth_url: str
    state: str


class McpAuthCodePayload(BaseModel):
    """Payload for MCP_AUTH_CODE messages.

    Sent server -> runner after the user completes consent and Google redirects
    to /oauth/mcp/callback. The runner exchanges the code for tokens locally
    via google-auth-oauthlib. The server never sees the resulting access /
    refresh tokens.
    """
    agent_id: str
    mcp_name: str
    state: str
    code: str


class SkillAuthUrlForUserPayload(BaseModel):
    """Payload for SKILL_AUTH_URL_FOR_USER messages.

    Skill-rail sibling of ``McpAuthUrlForUserPayload``. Sent server -> user
    when the runner has built a Google consent URL for a skill whose
    registry entry declares an ``auth`` block (gmail / google-calendar /
    google-drive / google-drive-write). The user's web UI renders this
    as a "Continue with Google" link; on completion the server forwards
    the code back to the runner via ``SKILL_AUTH_CODE``.
    """
    agent_id: str
    agent_name: str
    skill_name: str
    auth_url: str
    state: str


class SkillAuthCodePayload(BaseModel):
    """Payload for SKILL_AUTH_CODE messages.

    Skill-rail sibling of ``McpAuthCodePayload``. Sent server -> runner
    after the user completes consent and Google redirects to
    ``/oauth/skill/callback``. The runner exchanges the code for tokens
    locally and writes the token to
    ``{agent_dir}/skill-hub/state/<skill_name>/token.json`` (mode 0600).
    The server never sees the resulting access / refresh tokens.
    """
    agent_id: str
    skill_name: str
    state: str
    code: str


class KnowledgePackSyncPayload(BaseModel):
    """Payload for KNOWLEDGE_PACK_SYNC messages.

    Sent server -> runner whenever a user-curated knowledge pack is installed,
    re-installed (after a pack edit), or uninstalled on a specific agent. The
    pack's full file set travels in the envelope so the runner can rewrite
    ``{knowledge_dir}/knowledge-packs/{slug}/`` without a round-trip.
    """
    agent_id: str
    agent_name: str
    action: str  # "install" or "uninstall"
    pack_slug: str
    pack_name: str | None = None              # human-readable name (None on uninstall)
    pack_description: str | None = None       # one-line index hint (None on uninstall)
    # {relative_path: {"content_b64": <base64-encoded bytes>}}. Empty on
    # uninstall — runner gates on ``action``.
    pack_files: dict[str, dict] = Field(default_factory=dict)


class AgentRegistryChangePayload(BaseModel):
    """Payload for AGENT_REGISTRY_CHANGE messages.

    Fan-out to peer runners owned by the same user when an agent in the
    registry is registered, re-registered, has a peer-visible field changed
    (``user_teams``, ``description``, ``capabilities``,
    ``discoverable_through_registry``, rename), or is deleted. For
    register/update, receivers re-run ``Agent.sync_from_server`` to refresh
    their local peer-card cache. For delete, receivers prune the specific
    peer-card directory (``Agent.prune_peer_card``) so the per-turn invitable
    resolver (``_resolve_invitable_agents_for_prompt``) sees live state
    without a runner restart.

    Sibling of ``AGENT_SETTINGS_CHANGE`` (which carries runner-local state
    delivered only to the changed agent itself). The two stay orthogonal:
    peer-visible registry edits ride this envelope; runner-local config
    rides ``AGENT_SETTINGS_CHANGE``.
    """
    changed_agent_id: str
    changed_agent_name: str
    action: str  # "register" | "update" | "delete"


class AgentCardUpdatePayload(BaseModel):
    """Payload for AGENT_CARD_UPDATE messages.

    Server-side cursor bump on the agent card: trigger replies move
    ``last_reflected_at`` / ``last_synced_at``. Delivered to the agent's
    owner so the Agent Settings page's "Memory & Reflection" panel can
    live-refresh without a full page reload. Distinct from
    ``AGENT_SETTINGS_CHANGE`` (runner-local config) and
    ``AGENT_REGISTRY_CHANGE`` (peer-visible registry fan-out): these cursors
    are neither settings nor peer-visible.

    Convention mirrors ``AgentSettingsChangePayload``: only the field that
    changed in this envelope is populated; the others stay ``None``.
    """
    agent_id: str
    agent_name: str
    last_reflected_at: str | None = None  # None = unchanged in this envelope
    last_synced_at: str | None = None  # None = unchanged in this envelope


class TodayTabSyncPayload(BaseModel):
    """Payload for TODAY_TAB_SYNC messages.

    Sent server -> the tab-owning user (not the publishing agent) whenever
    any of the user's agents upserts or deletes a today tab via
    ``PUT|DELETE /me/today/tabs/{slug}``. The frontend listens and
    invalidates its ``['today-tabs']`` query so /today re-renders with the
    fresh tab.

    Carries only the per-tab cursor; the full payload (data + render code)
    is fetched separately via ``GET /me/today/tabs``.
    """
    action: str            # "upsert" | "delete"
    slug: str
    title: str | None = None
    generated_at: str | None = None


class ProjectReportSyncPayload(BaseModel):
    """Payload for PROJECT_REPORT_SYNC messages.

    Broadcast to every participant of a project whenever the project's
    interactive report is upserted or deleted via
    ``PUT|DELETE /projects/{id}/report``. The frontend listens on the
    project-detail view and invalidates ``['project-report', project_id]``
    so the Report tab re-fetches and re-renders.

    Carries only the cursor; the full payload (data + render code) is
    fetched separately via ``GET /projects/{id}/report``.
    """
    project_id: str
    action: str            # "upsert" | "delete"
    title: str | None = None
    generated_by_agent_name: str | None = None
    generated_at: str | None = None


class ControlEnvelope(BaseModel):
    """Lightweight WebSocket notification - never carries file content."""
    type: ControlMessageType
    payload: Union[ChangelogUpdatePayload, AgentStatusChangePayload, ProjectDeletedPayload, SkillSyncPayload, McpSyncPayload, AgentSettingsChangePayload, CancelLLMPayload, ActiveWorkChangePayload, McpAuthUrlForUserPayload, McpAuthCodePayload, SkillAuthUrlForUserPayload, SkillAuthCodePayload, KnowledgePackSyncPayload, AgentRegistryChangePayload, AgentCardUpdatePayload, TodayTabSyncPayload, ProjectReportSyncPayload, dict] = Field(default_factory=dict)

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
        elif self.type == ControlMessageType.MCP_AUTH_URL_FOR_USER:
            if not isinstance(self.payload, McpAuthUrlForUserPayload):
                raise ValueError(f"control message type {self.type} requires McpAuthUrlForUserPayload")
        elif self.type == ControlMessageType.MCP_AUTH_CODE:
            if not isinstance(self.payload, McpAuthCodePayload):
                raise ValueError(f"control message type {self.type} requires McpAuthCodePayload")
        elif self.type == ControlMessageType.SKILL_AUTH_URL_FOR_USER:
            if not isinstance(self.payload, SkillAuthUrlForUserPayload):
                raise ValueError(f"control message type {self.type} requires SkillAuthUrlForUserPayload")
        elif self.type == ControlMessageType.SKILL_AUTH_CODE:
            if not isinstance(self.payload, SkillAuthCodePayload):
                raise ValueError(f"control message type {self.type} requires SkillAuthCodePayload")
        elif self.type == ControlMessageType.KNOWLEDGE_PACK_SYNC:
            if not isinstance(self.payload, KnowledgePackSyncPayload):
                raise ValueError(f"control message type {self.type} requires KnowledgePackSyncPayload")
        elif self.type == ControlMessageType.AGENT_REGISTRY_CHANGE:
            if not isinstance(self.payload, AgentRegistryChangePayload):
                raise ValueError(f"control message type {self.type} requires AgentRegistryChangePayload")
        elif self.type == ControlMessageType.AGENT_CARD_UPDATE:
            if not isinstance(self.payload, AgentCardUpdatePayload):
                raise ValueError(f"control message type {self.type} requires AgentCardUpdatePayload")
        elif self.type == ControlMessageType.TODAY_TAB_SYNC:
            if not isinstance(self.payload, TodayTabSyncPayload):
                raise ValueError(f"control message type {self.type} requires TodayTabSyncPayload")
        elif self.type == ControlMessageType.PROJECT_REPORT_SYNC:
            if not isinstance(self.payload, ProjectReportSyncPayload):
                raise ValueError(f"control message type {self.type} requires ProjectReportSyncPayload")
        return self
