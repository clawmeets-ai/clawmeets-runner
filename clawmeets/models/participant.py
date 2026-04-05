# SPDX-License-Identifier: MIT
"""
clawmeets/models/participant.py
Base interface for all entities that can participate in projects.

Participants (Agent, Assistant, User) receive events from the sync layer
and execute actions via their configured ActionBlockExecutor.

All callbacks receive IDs and raw data - no ReactiveProject/ReactiveChatroom
dependencies. This keeps participants decoupled from state management.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Optional

# Re-export OperationalMode from llm.prompt_builder for backward compatibility
# This enum is defined in Layer 0 (prompt_builder.py) to avoid circular imports
from ..llm.prompt_builder import OperationalMode

if TYPE_CHECKING:
    from .chat_message import ChatMessage
    from .context import ModelContext
    from .project import Project


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ParticipantRole(str, Enum):
    """Role of a participant in the system.

    Used by the reactive architecture to distinguish between
    different types of participants (agents, assistants, users).
    """
    AGENT = "agent"          # Public worker agent
    ASSISTANT = "assistant"  # Private user-linked coordinator
    USER = "user"            # Human user


# OperationalMode is imported from llm.prompt_builder and re-exported here
# for backward compatibility. See prompt_builder.py for the enum definition.


class Participant(ABC):
    """
    Base interface for all entities that can participate in projects.
    Agent, Assistant, and User all implement this interface.

    All event handlers receive IDs and raw data - no state objects.
    This keeps participants decoupled from state management.
    """

    # ─────────────────────────────────────────────────────────
    # Abstract Properties
    # ─────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name."""
        pass

    @property
    @abstractmethod
    def role(self) -> ParticipantRole:
        """Participant role (agent, assistant, user)."""
        pass

    # ─────────────────────────────────────────────────────────
    # Association Methods
    # ─────────────────────────────────────────────────────────

    @abstractmethod
    def get_project(self, project_id: str):
        """Load a project by ID.

        Args:
            project_id: The project ID

        Returns:
            Project or None if not found
        """
        pass

    # ─────────────────────────────────────────────────────────
    # Operational Mode Methods
    # ─────────────────────────────────────────────────────────

    def is_coordinator_for(self, project: "Project") -> bool:
        """Check if this participant is the coordinator for the given project.

        Coordinators orchestrate work by delegating to other agents.
        Mode is derived from project metadata, not stored on the participant.

        Args:
            project: The project to check

        Returns:
            True if this participant's ID matches project.coordinator_id
        """
        return self.id == project.coordinator_id

    def get_operational_mode(self, project: "Project") -> OperationalMode:
        """Get operational mode for this participant in the given project.

        The same participant can be a coordinator in one project and
        a worker in another, depending on project.coordinator_id.

        Args:
            project: The project context

        Returns:
            OperationalMode.COORDINATOR if this is the project coordinator,
            OperationalMode.WORKER otherwise
        """
        if self.is_coordinator_for(project):
            return OperationalMode.COORDINATOR
        return OperationalMode.WORKER

    # ─────────────────────────────────────────────────────────
    # Class Methods
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def get(participant_id: str, ctx: "ModelContext") -> Optional["Participant"]:
        """Load participant by ID, returning the correct concrete type.

        Args:
            participant_id: The participant ID to look up
            ctx: ModelContext for filesystem operations

        Returns:
            Agent, Assistant, or User instance, or None if not found
        """
        # Import concrete types here to avoid circular imports
        from clawmeets.models.agent import Agent
        from clawmeets.models.assistant import Assistant
        from clawmeets.models.user import User

        # Try each concrete type until one exists
        for concrete_cls in [Agent, Assistant, User]:
            participant = concrete_cls.get(participant_id, ctx)
            if participant is not None:
                return participant

        return None

    @staticmethod
    def get_by_name(name: str, ctx: "ModelContext") -> Optional["Participant"]:
        """Load participant by name, returning the correct concrete type.

        Searches across all participant types (Agent, Assistant, User).

        Args:
            name: The participant name to look up
            ctx: ModelContext for filesystem operations

        Returns:
            Agent, Assistant, or User instance, or None if not found
        """
        from clawmeets.models.agent import Agent
        from clawmeets.models.assistant import Assistant
        from clawmeets.models.user import User

        # Check agents
        agent = Agent.get_by_name(name, ctx)
        if agent is not None:
            return agent

        # Check assistants
        assistant = Assistant.get_by_name(name, ctx)
        if assistant is not None:
            return assistant

        # Check users (username)
        user = User.get_by_username(name, ctx)
        if user is not None:
            return user

        return None

    # ─────────────────────────────────────────────────────────
    # Lifecycle Events
    # ─────────────────────────────────────────────────────────

    async def on_registered(self) -> None:
        """Called when participant is registered with clawmeets."""
        pass

    async def on_unregistered(self) -> None:
        """Called when participant is removed from clawmeets."""
        pass

    # ─────────────────────────────────────────────────────────
    # Project Events (ID-based)
    # ─────────────────────────────────────────────────────────

    async def on_project_created(
        self,
        project_id: str,
        project_name: str,
        request: str,
    ) -> None:
        """
        Called when a new project is created.

        Args:
            project_id: The project ID
            project_name: The project name
            request: The original user request
        """
        pass

    async def on_project_completed(
        self,
        project_id: str,
    ) -> None:
        """
        Called when a project completes.

        Args:
            project_id: The project ID
        """
        pass

    # ─────────────────────────────────────────────────────────
    # Chatroom Events (ID-based)
    # ─────────────────────────────────────────────────────────

    async def on_chatroom_created(
        self,
        project_id: str,
        chatroom_name: str,
        participants: list[str],
    ) -> None:
        """
        Called when a chatroom is created.

        Args:
            project_id: The project ID
            chatroom_name: The chatroom name (unique within project)
            participants: List of participant IDs
        """
        pass

    # ─────────────────────────────────────────────────────────
    # Message Events (ID-based with raw ChatMessage)
    # ─────────────────────────────────────────────────────────

    async def on_message(
        self,
        project_id: str,
        chatroom_name: str,
        message: "ChatMessage",
        addressed_to_me: bool,
    ) -> None:
        """
        Called when a message is received in a chatroom.

        Args:
            project_id: The project ID
            chatroom_name: The chatroom ID
            message: The message content (raw ChatMessage)
            addressed_to_me: True if message.expects_response_from includes this participant
        """
        pass

    # ─────────────────────────────────────────────────────────
    # File Events (ID-based)
    # ─────────────────────────────────────────────────────────

    async def on_file_created(
        self,
        project_id: str,
        chatroom_name: str,
        filename: str,
        content: bytes,
    ) -> None:
        """
        Called when a new file is created.

        Args:
            project_id: The project ID
            chatroom_name: The chatroom name
            filename: The filename
            content: The file content as bytes
        """
        pass

    async def on_file_updated(
        self,
        project_id: str,
        chatroom_name: str,
        filename: str,
        content: bytes,
    ) -> None:
        """
        Called when an existing file is updated.

        Args:
            project_id: The project ID
            chatroom_name: The chatroom name
            filename: The filename
            content: The updated file content as bytes
        """
        pass

    # ─────────────────────────────────────────────────────────
    # Coordination Events (ID-based)
    # ─────────────────────────────────────────────────────────

    async def on_batch_complete(
        self,
        project_id: str,
        chatroom_name: str,
        message_id: str,
        responded_participants: list[str],
    ) -> None:
        """
        Called when all expected participants have responded (coordinator only).

        Args:
            project_id: The project ID
            chatroom_name: The chatroom ID
            message_id: The original message ID that started the batch
            responded_participants: List of participant IDs that responded
        """
        pass

    async def on_batch_timeout(
        self,
        project_id: str,
        chatroom_name: str,
        message_id: str,
        responded_participants: list[str],
        timed_out_participants: list[str],
    ) -> None:
        """
        Called when some participants didn't respond in time (coordinator only).

        Args:
            project_id: The project ID
            chatroom_name: The chatroom ID
            message_id: The original message ID
            responded_participants: List of participant IDs that responded
            timed_out_participants: List of participant IDs that timed out
        """
        pass

    # ─────────────────────────────────────────────────────────
    # First User Request Event (Coordinator Only)
    # ─────────────────────────────────────────────────────────

    async def on_first_user_request(
        self,
        project_id: str,
        chatroom_name: str,
        message: "ChatMessage",
        context_files: list[str],
    ) -> None:
        """
        Called when coordinator receives first user message in a project.

        This is the opportunity to:
        - Analyze uploaded context files
        - Refine AGENTS.md with specific assignments
        - Refine PLAN.md with concrete milestones
        - Immediately kick off execution

        Default implementation falls back to on_message() for non-coordinators.

        Args:
            project_id: The project ID
            chatroom_name: The user-communication chatroom ID
            message: The first user message
            context_files: List of files in shared-context (excluding AGENTS.md, PLAN.md)
        """
        # Default: fall back to normal message handling
        await self.on_message(
            project_id=project_id,
            chatroom_name=chatroom_name,
            message=message,
            addressed_to_me=True,
        )

    # ─────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id!r}, name={self.name!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Participant):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
