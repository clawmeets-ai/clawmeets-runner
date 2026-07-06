# SPDX-License-Identifier: MIT
"""
clawmeets/models/project.py
Project model with Active Record persistence methods.

## Changelog-First Architecture

Project is a **frozen** Pydantic model (`model_config = {"frozen": True}`).
All mutations flow through the changelog (acting as a redo log), ensuring:
1. Atomic recording on the server before any local writes
2. Eventual consistency across all runners via sync
3. Idempotent replay for crash recovery

Direct mutation is prevented by the frozen config. Use `project.state()`
to access ProjectState for changelog-driven writes.

## Read/Write Separation

- **Project** (frozen): Read-only data representation, path properties,
  association methods for loading related objects
- **ProjectState**: Handles all filesystem writes triggered by changelog
  processing (create, complete, add_participant)

This separation ensures changes are visible to other agents only after
they flow through the distributed changelog system.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, Field, PrivateAttr, computed_field

from ..sync.changelog import ProjectStatus
from ..utils.file_io import FileUtil
from .participant import Participant


# Project surface — the explicit project shape.
#   "regular"   — task-shaped project (PLAN.md / milestones / multi-party).
#   "dm"        — owned 1:1 DM (a user with their OWN agent), single
#                 user-communication room, DM prompt variant, auto-title.
#   "frontdesk" — ONE END of a cross-account thread (Front Desk). Each end is a
#                 distinct project OWNED (created_by) by that side and bound
#                 1-to-1 to the other end by a TunnelBinding. The requester end's
#                 coordinator is a foreign agent (an `external` ghost that never
#                 runs locally); the host end's coordinator is that same foreign
#                 agent running for real. "dm" and "frontdesk" are both
#                 "dm-shaped" (see Project.is_dm_shaped).
ProjectSurface = Literal["regular", "dm", "frontdesk"]

# Seeded label a fresh DM thread carries until its first exchange auto-titles it.
# Doubles as the fire-once latch for the auto-title trigger (agent.py) and the
# defensive ``expected_current`` sentinel on the rename endpoint — the trigger
# runs only while ``display_name == NEW_CHAT_PLACEHOLDER`` and the mutation flips
# it, so every later message reads a non-placeholder value and skips.
NEW_CHAT_PLACEHOLDER = "New chat"

# Front Desk / DM-shaped project name shape: ``{requester}-fd-{agent_short}``.
# Mirrors the frontend FRONT_DESK_NAME_RE (web/frontend/src/types/index.ts).
# Group 1 is the requester prefix (a username for user-requester channels, or a
# requester agent's full name for cross-agent delegation).
_FRONT_DESK_NAME_RE = re.compile(r"^([a-z0-9][a-z0-9_-]*)-fd-([a-z0-9][a-z0-9_-]*)$")


if TYPE_CHECKING:
    from .agent import Agent
    from .context import ModelContext
    from .chatroom import Chatroom


def _as_aware_utc(dt: datetime) -> datetime:
    """Coerce a datetime to timezone-aware UTC.

    Timestamps in this system are written as UTC-aware isoformat strings, but a
    legacy/hand-edited ``meta.json`` could carry a naive value. Normalizing
    before any ``max()``/comparison avoids a ``TypeError`` from mixing aware and
    naive datetimes.
    """
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


class Project(BaseModel):
    """Project metadata with persistence and association methods.

    A project represents a collaborative task with multiple chatrooms
    and participating agents coordinated by a single coordinator agent.
    """
    model_config = {"frozen": True}

    id: str
    name: str
    coordinator_id: str           # agent_id of coordinator
    coordinator_name: str         # name of coordinator (avoids lookup on workers)
    request: str                  # original user prompt
    participating_agents: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_by: str               # user_id of creator (derived from auth)
    agent_pool: str = Field(default="verified")  # "self", "owned", "verified", or "all". "self" = coordinator only (used by own-DM); the resolver returns an empty invitable list.
    agent_teams: list[str] = Field(default_factory=list)  # Hard allowlist by user_team; pairs with agent_names. Empty teams + empty names = no filter (everyone in pool is invitable).
    agent_names: list[str] = Field(default_factory=list)  # Hard allowlist by agent display name (id, full name, or owner-relative short name); pairs with agent_teams. OR semantics across both lists.
    surface: ProjectSurface = "regular"  # Explicit project shape; "regular" | "dm"
    display_name: Optional[str] = None    # Model-set human label (regular projects AND dm threads).
                                          # None only on legacy/pre-migration rows -> callers render `name` (the slug).
    last_request_ts: Optional[datetime] = None   # ts of the most recent non-ack USER message across the project.
    last_response_ts: Optional[datetime] = None  # ts of the most recent non-ack AGENT/ASSISTANT message.

    # Private runtime state (not serialized)
    _ctx: Optional["ModelContext"] = PrivateAttr(default=None)

    @computed_field
    @property
    def last_modified(self) -> datetime:
        """Recency key the sidebar sorts on (desc).

        ``max(last_request_ts, last_response_ts)`` with ``created_at`` as the
        floor, so a brand-new / message-less project still has a deterministic,
        non-null sort key. Decorated with ``@computed_field`` so it rides
        ``model_dump()`` into ``GET /projects`` and ``GET /projects/{id}``
        without extra plumbing.
        """
        candidates = [_as_aware_utc(self.created_at)]
        if self.last_request_ts is not None:
            candidates.append(_as_aware_utc(self.last_request_ts))
        if self.last_response_ts is not None:
            candidates.append(_as_aware_utc(self.last_response_ts))
        return max(candidates)

    @property
    def ctx(self) -> "ModelContext":
        """Get the model context (required for I/O operations).

        Raises:
            RuntimeError: If context is not set (use Project.get() to load with context)
        """
        if self._ctx is None:
            raise RuntimeError(
                "Project requires ModelContext. "
                "Use Project.get() to load with context."
            )
        return self._ctx

    # -------------------------------------------------------------------------
    # Derived Properties (always read fresh from filesystem)
    # -------------------------------------------------------------------------

    @computed_field
    @property
    def status(self) -> ProjectStatus:
        """Project status (always reads fresh from meta.json).

        Decorated with ``@computed_field`` so Pydantic includes it in
        ``model_dump()`` / serialized HTTP responses (otherwise the
        ``GET /projects/{id}`` payload would silently omit ``status``,
        and frontend gates like ``project.status === ProjectStatus.COMPLETED``
        would never match).

        Returns:
            ProjectStatus.ACTIVE or ProjectStatus.COMPLETED or ProjectStatus.FAILED
        """
        data = FileUtil.read(self.meta_path, "json")
        return ProjectStatus(data["status"])

    @property
    def data_dir(self) -> Path:
        """Path to project data directory (projects/{name}-{id}/)."""
        return self.ctx.projects_dir / f"{self.name}-{self.id}"

    @property
    def meta_dir(self) -> Path:
        """Path to project metadata directory (metadata/projects/{name}-{id}/)."""
        return self.ctx.metadata_dir / f"{self.name}-{self.id}"

    @property
    def meta_path(self) -> Path:
        """Path to project meta.json file."""
        return self.meta_dir / "meta.json"

    @property
    def chatrooms(self) -> list[str]:
        """List chatroom names by scanning metadata directory.

        Derived from filesystem: lists directories under
        metadata/projects/{name}-{id}/chatrooms/

        Returns:
            Sorted list of chatroom names
        """
        chatrooms_dir = self.meta_dir / "chatrooms"
        return sorted(d.name for d in chatrooms_dir.iterdir() if d.is_dir())

    @property
    def is_dm_project(self) -> bool:
        """Owned 1:1 DM ONLY (``surface == "dm"``).

        Does NOT include Front Desk ends (``surface == "frontdesk"``). Use
        :meth:`is_dm_shaped` wherever the old code meant "dm-shaped" rather than
        "owned DM specifically" — see the §5 surface audit in the Front Desk plan.
        """
        return self.surface == "dm"

    @property
    def is_frontdesk_project(self) -> bool:
        """One END of a cross-account Front Desk thread (``surface == "frontdesk"``)."""
        return self.surface == "frontdesk"

    @property
    def is_dm_shaped(self) -> bool:
        """True for BOTH owned DMs and Front Desk ends.

        Both carry a single ``user-communication`` room, use the DM prompt
        variant (``is_dm=True``), and are auto-title eligible. This is the
        predicate the auto-title trigger, prompt-variant selection, and the
        DM-vs-milestone-setup branch key on — the widened successor to
        ``is_dm_project`` at those callsites.
        """
        return self.surface in ("dm", "frontdesk")

    def frontdesk_is_host_side(self, ctx: "ModelContext") -> bool:
        """True when the coordinator is owned by ``created_by`` (the terminal/host end).

        Semantics: the project's coordinator agent is ``registered_by`` the same
        user who created the project — i.e. someone reached IN to MY agent. This
        is the **host / terminal** side of a cross-account thread (renders in the
        owner's Front Desk section; tunnel detection treats it as terminal so it
        never spawns an FD-of-an-FD). The **requester** side is the negation —
        the coordinator is a foreign agent I do NOT own (I reached OUT).

        Requires an Agent lookup because ``Project`` stores ``coordinator_id``,
        not the coordinator's owner. Returns ``False`` when the coordinator or
        ``created_by`` cannot be resolved (fail-open to non-terminal).

        NOTE: for an owned DM (``surface == "dm"``) this is also True (the agent
        is my own) — which is correct for the terminal-gate usage in
        ``tunnel_subscriber`` (own DMs and legacy shared ``-fd-`` rows are both
        terminal: cross-account delegation must not bleed a tunnel out of them).
        """
        from .agent import Agent
        if not self.created_by:
            return False
        coordinator = Agent.get(self.coordinator_id, ctx)
        if coordinator is None:
            return False
        return coordinator.registered_by == self.created_by

    @property
    def fd_requester_name(self) -> Optional[str]:
        """Requester prefix of a DM-shaped Front Desk project name
        (``{requester}-fd-{agent_short}``), else None.

        For user-requester channels this equals the requester's username; for
        agent-requester (cross-agent delegation) channels it's the requester
        agent's full name, which never matches a username — so it's inert in
        user-JWT authorization. ``created_by`` stays the host (foreign agent's
        owner); this lets the requester be authorized on their own channel
        without flipping ownership.
        """
        if not self.is_dm_project:
            return None
        m = _FRONT_DESK_NAME_RE.match(self.name)
        return m.group(1) if m else None

    # -------------------------------------------------------------------------
    # Invitable allowlist (the unified (teams, names) policy for who can be
    # invited to a chatroom in this project).
    # -------------------------------------------------------------------------

    @property
    def enforces_invitable_allowlist(self) -> bool:
        """Whether the chatroom-create gate should run the allowlist check.

        Allowlist is only enforced when explicitly populated; an empty
        allowlist means "no filter" (every agent in the pool is invitable).
        ``agent_pool="self"`` also forces enforcement so the resolver's
        empty-list short-circuit is honored at the prompt callsite.
        """
        return bool(self.agent_names or self.agent_teams) or self.agent_pool == "self"

    def _invitable_viewer_owner_id(self, ctx: "ModelContext") -> Optional[str]:
        """Owner whose namespace short names resolve in for the allowlist check.

        Always scopes to the project creator. Returns ``None`` when no owner is
        recorded (legacy projects).
        """
        return self.created_by

    def matches_invitable(self, invitee: "Agent", ctx: "ModelContext") -> bool:
        """Does ``invitee`` pass this project's invitable allowlist?

        Empty allowlist = nothing matches (callers that want "empty = pass"
        should gate on ``enforces_invitable_allowlist`` first).
        """
        allowed_names = set(self.agent_names)
        allowed_teams = set(self.agent_teams)
        if invitee.id in allowed_names:
            return True
        if invitee.name in allowed_names:
            return True
        viewer_owner_id = self._invitable_viewer_owner_id(ctx)
        if (
            viewer_owner_id
            and invitee.registered_by == viewer_owner_id
            and "-" in invitee.name
        ):
            short = invitee.name.split("-", 1)[1]
            if short in allowed_names:
                return True
        if allowed_teams and (allowed_teams & set(invitee.user_teams)):
            return True
        return False

    def resolve_invitable_agents(
        self,
        candidates: "Iterable[Agent]",
        ctx: "ModelContext",
    ) -> "list[Agent]":
        """Filter ``candidates`` down to those that pass this project's allowlist.

        ``agent_pool="self"`` short-circuits to an empty list — the
        coordinator is the only allowed speaker (own-DM).
        Empty allowlist = pass-through (returns the full list as-is).
        """
        if self.agent_pool == "self":
            return []
        if not self.agent_teams and not self.agent_names:
            return list(candidates)
        return [c for c in candidates if self.matches_invitable(c, ctx)]

    # -------------------------------------------------------------------------
    # Association Methods (lookup-based, no caching)
    # -------------------------------------------------------------------------

    def list_chatrooms(self) -> list:
        """Load all chatrooms in this project.

        Returns:
            List of Chatroom objects
        """
        from .chatroom import Chatroom
        result = []
        for chatroom_name in self.chatrooms:
            room = Chatroom.get(self.id, chatroom_name, self.ctx)
            if room is not None:
                result.append(room)
        return result

    def get_chatroom(self, chatroom_name: str):
        """Load a specific chatroom.

        Args:
            chatroom_name: The chatroom name

        Returns:
            Chatroom, or None when the room doesn't exist (Chatroom.get contract)
        """
        from .chatroom import Chatroom
        return Chatroom.get(self.id, chatroom_name, self.ctx)

    def get_shared_context_room(self):
        """Get the shared-context room for project-wide knowledge.

        Returns:
            Chatroom
        """
        for room in self.list_chatrooms():
            if room.is_shared_context_room:
                return room
        raise ValueError(f"shared-context room not found in project {self.id}")

    def get_user_communication_room(self):
        """Get the user-communication room for user<->assistant chat.

        Returns:
            Chatroom
        """
        for room in self.list_chatrooms():
            if room.is_user_communication_room:
                return room
        raise ValueError(f"user-communication room not found in project {self.id}")

    def get_context_files(self) -> list[str]:
        """Get list of context files from shared-context chatroom.

        Returns user-uploaded context files, excluding auto-generated files
        (PLAN.md) which is refined during project setup.

        Note: AGENTS.md is now a global file at the runner root, not per-project.

        Returns:
            List of filenames in shared-context (excluding PLAN.md)
        """
        shared_context_room = self.get_shared_context_room()
        all_files = shared_context_room.list_files()
        excluded = {"PLAN.md"}
        return [f for f in all_files if f not in excluded]

    def list_participants(self) -> list:
        """Load participants for all project participants.

        Returns:
            List of Participant objects (Agent, Assistant, or User)
        """
        result = []
        for pid in self.participating_agents:
            participant = Participant.get(pid, self.ctx)
            result.append(participant)
        return result

    def get_coordinator(self):
        """Load the coordinator (typically an Assistant).

        Returns:
            Participant (Agent, Assistant, or User) or None if not found
        """
        return Participant.get(self.coordinator_id, self.ctx)

    def get_chatrooms_for_participant(self, participant_id: str) -> list:
        """Get chatrooms where participant is a member.

        Args:
            participant_id: The participant ID to filter by

        Returns:
            List of Chatroom objects where participant is a member
        """
        return [r for r in self.list_chatrooms() if participant_id in r.participants]

    # -------------------------------------------------------------------------
    # Active Record: Persistence Methods
    # -------------------------------------------------------------------------

    @classmethod
    def get(cls, project_id: str, ctx: "ModelContext") -> "Project":
        """Load project by ID.

        Args:
            project_id: The project ID
            ctx: ModelContext for filesystem operations

        Returns:
            Project
        """
        # Find project directory by ID (glob for {name}-{id} pattern). Sort for
        # determinism: glob order is filesystem-dependent, so an unsorted matches[0]
        # could resolve to a different {name}-{id} dir across processes if a stale
        # duplicate ever exists — yielding a different project.name and thus a
        # different sandbox_dir between the writer and the action executor.
        matches = sorted(ctx.metadata_dir.glob(f"*-{project_id}"))
        if not matches:
            # Callers may pass the bare id or the full "{name}-{id}" slug (what
            # an agent sees as its synced project dir). The glob above only
            # matches the bare id; fall back to an exact dir match so the slug
            # form resolves too.
            exact = ctx.metadata_dir / project_id
            if exact.is_dir():
                matches = [exact]
        if not matches:
            raise ValueError(f"Project {project_id} not found")
        meta_path = matches[0] / "meta.json"
        data = FileUtil.read(meta_path, "json")
        if not data:
            raise ValueError(f"Project {project_id} has no metadata")
        instance = cls.model_validate(data)
        object.__setattr__(instance, "_ctx", ctx)
        return instance

    @classmethod
    def list_all(cls, ctx: "ModelContext") -> list["Project"]:
        """List all projects.

        Args:
            ctx: ModelContext for filesystem operations

        Returns:
            List of Project objects
        """
        result = []
        if not ctx.metadata_dir.exists():
            return result
        for entry in sorted(ctx.metadata_dir.iterdir()):
            if not entry.is_dir():
                continue
            meta_path = entry / "meta.json"
            data = FileUtil.read(meta_path, "json")
            if data:
                instance = cls.model_validate(data)
                object.__setattr__(instance, "_ctx", ctx)
                result.append(instance)
        return result

    @classmethod
    def get_agent_memberships(
        cls,
        agent_id: str,
        ctx: "ModelContext",
    ) -> list["Chatroom"]:
        """Return all chatrooms an agent participates in across all projects.

        Args:
            agent_id: The agent ID to find memberships for
            ctx: ModelContext for filesystem operations

        Returns:
            List of Chatroom objects
        """
        from .chatroom import Chatroom

        result: list[Chatroom] = []
        for proj in cls.list_all(ctx):
            if agent_id not in proj.participating_agents:
                continue
            for chatroom_name in proj.chatrooms:
                room = Chatroom.get(proj.id, chatroom_name, ctx)
                if room is not None and agent_id in room.participants:
                    result.append(room)
        return result

    @classmethod
    def get_projects_for_agent(
        cls,
        agent_id: str,
        ctx: "ModelContext",
    ) -> list["Project"]:
        """Return all projects an agent participates in.

        Args:
            agent_id: The agent ID to find projects for
            ctx: ModelContext for filesystem operations

        Returns:
            List of Project objects where the agent is a participant
        """
        result: list[Project] = []
        for proj in cls.list_all(ctx):
            if agent_id in proj.participating_agents:
                result.append(proj)
        return result

    # -------------------------------------------------------------------------
    # State Access (for write operations)
    # -------------------------------------------------------------------------

    def state(self) -> "ProjectState":
        """Get the state model for write operations.

        Returns:
            ProjectState instance for this project
        """
        return ProjectState(self)


# =============================================================================
# ProjectState: Write Operations
# =============================================================================


class ProjectState:
    """Mutable state model for project write operations.

    This class handles all filesystem write operations for a project,
    triggered by changelog processing. The separation keeps the main
    Project model immutable while allowing write operations.

    Usage:
        project = Project.get(project_id, ctx)
        project.state().complete()

    For creation, use the classmethod:
        project = ProjectState.create(project_id, project_name, coordinator_id, request, ctx)
    """

    def __init__(self, project: Project) -> None:
        """Initialize with a project instance.

        Args:
            project: The project to operate on
        """
        self._project = project

    @classmethod
    def create(
        cls,
        project_id: str,
        project_name: str,
        coordinator_id: str,
        coordinator_name: str,
        request: str,
        created_by: str,
        created_at: datetime,
        ctx: "ModelContext",
        agent_pool: str = "verified",
        agent_teams: list[str] | None = None,
        agent_names: list[str] | None = None,
        surface: ProjectSurface = "regular",
        display_name: Optional[str] = None,
    ) -> Project:
        """Create a new project with directories and meta.json.

        Creates:
        - Data directory: projects/{name}-{id}/
        - Metadata directory: metadata/projects/{name}-{id}/
        - meta.json in metadata directory

        Args:
            project_id: The project ID
            project_name: The project name
            coordinator_id: The coordinator agent ID
            coordinator_name: The coordinator name
            request: User request string
            created_by: User ID of creator (optional)
            created_at: Creation timestamp
            ctx: ModelContext for filesystem operations
            agent_pool: Agent pool mode ("owned", "verified", or "all")
            agent_teams: Optional list of user_teams; agents carrying any of these teams pass the allowlist (composed via OR with agent_names; empty/None on both = no filter)
            agent_names: Optional list of agent display names (id, full name, or owner-relative short name); agents matching any pass the allowlist (composed via OR with agent_teams; empty/None on both = no filter)

        Returns:
            The created Project instance

        Raises:
            ValueError: If project_name is invalid
        """
        # Validate project name
        project_name = FileUtil.validate_fs_name(project_name)

        # Build paths (directories created by FileUtil.write with ensure_dir=True)
        meta_dir = ctx.metadata_dir / f"{project_name}-{project_id}"

        # Write meta.json (chatrooms derived from filesystem, not stored)
        project_data = {
            "id": project_id,
            "name": project_name,
            "status": "active",
            "coordinator_id": coordinator_id,
            "coordinator_name": coordinator_name,
            "request": request,
            "participating_agents": [coordinator_id],
            "created_at": created_at.isoformat() if created_at else None,
            "created_by": created_by,
            "agent_pool": agent_pool,
            "agent_teams": list(agent_teams) if agent_teams else [],
            "agent_names": list(agent_names) if agent_names else [],
            "surface": surface,
            "display_name": display_name,
        }
        FileUtil.write(meta_dir / "meta.json", project_data, "json", atomic=True)

        # Return the created project
        instance = Project.model_validate(project_data)
        object.__setattr__(instance, "_ctx", ctx)
        return instance

    def complete(self) -> None:
        """Update project status to COMPLETED in meta.json."""
        meta_path = self._project.meta_path
        project_dict = FileUtil.read(meta_path, "json")
        project_dict["status"] = "completed"
        FileUtil.write(meta_path, project_dict, "json", atomic=True)

    def reactivate(self) -> None:
        """Flip project status back to ACTIVE in meta.json.

        Counterpart to ``complete()`` — runs when a user posts into a
        completed task's user-communication, resuming the conversation.
        """
        meta_path = self._project.meta_path
        project_dict = FileUtil.read(meta_path, "json")
        project_dict["status"] = "active"
        FileUtil.write(meta_path, project_dict, "json", atomic=True)

    def apply_allowlist_update(
        self,
        agent_names: list[str],
        agent_teams: list[str],
    ) -> None:
        """Overwrite the project's allowlist snapshot in meta.json.

        Idempotent: re-writes the two list fields with the supplied values.
        Fired by ``PROJECT_ALLOWLIST_UPDATED`` replay; emitted server-side
        when the coordinator's Front Desk Settings change.
        """
        meta_path = self._project.meta_path
        project_dict = FileUtil.read(meta_path, "json")
        project_dict["agent_names"] = list(agent_names)
        project_dict["agent_teams"] = list(agent_teams)
        FileUtil.write(meta_path, project_dict, "json", atomic=True)

    def add_participant(self, participant_id: str) -> None:
        """Add a participant to the project's participating_agents list.

        Args:
            participant_id: The participant ID to add
        """
        meta_path = self._project.meta_path
        project_dict = FileUtil.read(meta_path, "json")
        current_agents = project_dict.get("participating_agents", [])
        if participant_id not in current_agents:
            current_agents.append(participant_id)
            project_dict["participating_agents"] = current_agents
            FileUtil.write(meta_path, project_dict, "json", atomic=True)

    def touch_activity(self, *, ts: datetime, is_request: bool) -> None:
        """Advance ``last_request_ts`` (user message) or ``last_response_ts``
        (agent message) so the project sorts by recency in the sidebar.

        Monotonic set-if-greater / set-if-null: a timestamp is only ever moved
        forward, never backward. This makes the write idempotent under changelog
        replay and safe against an out-of-order replay or a re-run migration
        backfill regressing a value the live server already advanced.

        Args:
            ts: The message timestamp.
            is_request: True for a USER message (``last_request_ts``), False for
                an AGENT/ASSISTANT message (``last_response_ts``).
        """
        field = "last_request_ts" if is_request else "last_response_ts"
        meta_path = self._project.meta_path
        project_dict = FileUtil.read(meta_path, "json")
        new_dt = _as_aware_utc(ts)
        existing_raw = project_dict.get(field)
        existing_dt = (
            _as_aware_utc(datetime.fromisoformat(existing_raw)) if existing_raw else None
        )
        if existing_dt is not None and existing_dt >= new_dt:
            return  # never regress; no write needed
        project_dict[field] = new_dt.isoformat()
        FileUtil.write(meta_path, project_dict, "json", atomic=True)

    def set_thread_title(self, *, display_name: str, slug: Optional[str] = None) -> None:
        """Apply a model-generated title to a DM thread after its first exchange.

        Sets ``display_name`` (replacing the ``"New chat"`` placeholder). The
        stable thread identity is the project ``id`` (UUID) — see the plan's
        OPEN Q2. The ``slug`` parameter is accepted for interface stability but
        deliberately NOT applied: the on-disk layout keys the project directory
        on ``{name}-{id}``, so renaming the slug would require moving directories
        and is unsafe until Q2 is resolved. Only ``display_name`` is model-set.

        Idempotent: a second call simply overwrites ``display_name`` again.

        Args:
            display_name: The model-generated human title.
            slug: Reserved (see above); currently a no-op.
        """
        meta_path = self._project.meta_path
        project_dict = FileUtil.read(meta_path, "json")
        project_dict["display_name"] = display_name
        FileUtil.write(meta_path, project_dict, "json", atomic=True)
