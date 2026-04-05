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

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field, PrivateAttr

from ..sync.changelog import ProjectStatus
from ..utils.file_io import FileUtil
from ..utils.validation import validate_name
from .participant import Participant

if TYPE_CHECKING:
    from .context import ModelContext
    from .chatroom import Chatroom


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
    agent_pool: str = Field(default="verified")  # "owned", "verified", or "all"

    # Private runtime state (not serialized)
    _ctx: Optional["ModelContext"] = PrivateAttr(default=None)

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

    @property
    def status(self) -> ProjectStatus:
        """Project status (always reads fresh from meta.json).

        Returns:
            ProjectStatus.ACTIVE or ProjectStatus.COMPLETED
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
        """Check if this is a DM (direct message) project.

        DM projects are named "DM-{username}" and are used for direct
        messaging between users and agents.
        """
        return self.name.startswith("DM-")

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
            result.append(room)
        return result

    def get_chatroom(self, chatroom_name: str):
        """Load a specific chatroom.

        Args:
            chatroom_name: The chatroom name

        Returns:
            Chatroom
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
        # Find project directory by ID (glob for {name}-{id} pattern)
        matches = list(ctx.metadata_dir.glob(f"*-{project_id}"))
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
                if agent_id in room.participants:
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

        Returns:
            The created Project instance

        Raises:
            ValueError: If project_name is invalid
        """
        # Validate project name
        project_name = validate_name(project_name)

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
