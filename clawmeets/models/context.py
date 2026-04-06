# SPDX-License-Identifier: MIT
"""
clawmeets/models/context.py
Context for filesystem-backed models.

ModelContext provides path resolution for filesystem operations.
It is a lightweight class that:
- Defines directory structure (projects_dir, metadata_dir, participants_dir)
- Finds directories by ID using glob patterns
- Creates participant directories

File I/O is handled by:
- Model classes (Project, Chatroom) via their .state() methods for writes
- FileUtil for low-level read/write operations

ModelContextChangelogSubscriber handles filesystem writes from changelog entries.
Each instance is bound to a specific project via the changelog_subscriber() factory.

Directory structure:
- Projects: {project_name}-{project_id}/
- Chatrooms: chatrooms/{chatroom_name}/ (no ID suffix, name is unique within project)

ModelContext is system-level - use a single instance across all projects.
All methods take project_id/project_name as explicit parameters.

Optional runtime fields for agent execution:
- cli: ClaudeCLI instance for LLM invocation
- knowledge_dirs: Additional directories for Claude access
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from clawmeets.sync.changelog import (
    ChangelogEntry,
    ChangelogEntryType,
    ProjectCreatedPayload,
    MessagePayload,
    FilePayload,
    RoomCreatedPayload,
    ParticipantAddedPayload,
)
from clawmeets.sync.subscriber import ChangelogSubscriber
from clawmeets.utils.file_io import FileUtil
from clawmeets.models.project import Project, ProjectState
from clawmeets.models.chatroom import Chatroom, ChatroomState
from clawmeets.models.chat_message import ChatMessage

from clawmeets.api.action_executor import ActionBlockExecutor
from clawmeets.api.client import ClawMeetsClient
from clawmeets.llm.claude_cli import ClaudeCLI
from clawmeets.utils.notification_center import NotificationCenter


logger = logging.getLogger(__name__)


class ModelContext:
    """Context for filesystem-backed models.

    Combines ID-based path finding with synchronous filesystem I/O.
    Works on both server and runner sides with different base directories.

    ModelContext is system-level - use a single shared instance across all projects.
    All methods take project_id/project_name as explicit parameters.

    For changelog processing, use changelog_subscriber() to create a
    project-bound ChangelogSubscriber instance.

    Usage:
        ctx = ModelContext(base_dir=Path("~/.clawmeets").expanduser())
        subscriber = ctx.changelog_subscriber(project_id, project_name)
        runloop.add_subscriber(subscriber)
    """

    def __init__(
        self,
        base_dir: Path,
        notification_center: NotificationCenter,
        cli: Optional["ClaudeCLI"] = None,
        knowledge_dirs: Optional[list[Path]] = None,
        client: Optional["ClawMeetsClient"] = None,
        git_url: Optional[str] = None,
        git_ignored_folder: str = ".bus-files",
        claude_plugin_dirs: Optional[list[Path]] = None,
    ) -> None:
        """Initialize context with a single base directory.

        All paths are derived from base_dir:
        - projects_dir: base_dir/projects/{name}-{id}/chatrooms/{chatroom_name}/...
        - metadata_dir: base_dir/metadata/projects/{name}-{id}/...
        - participants_dir: base_dir (contains agents/, assistants/, users/)

        Optional runtime fields for agent execution:
        - cli: ClaudeCLI instance for LLM invocation (None if not configured)
        - knowledge_dirs: Additional directories for Claude access (e.g., knowledge bases)
        - client: ClawMeetsClient for HTTP operations (None if not configured)
        - claude_plugin_dirs: Claude plugin directories for skill access (e.g., save-to-knowledge)

        Optional git configuration for code-aware sandbox:
        - git_url: URL or path to clone from (None = standalone git init in sandbox)
        - git_ignored_folder: Folder for deliverables that should not be git-tracked

        Args:
            base_dir: Base directory for all data
            cli: Claude CLI for LLM invocation (optional, for agent runtime)
            knowledge_dirs: Additional directories for Claude access (optional)
            client: ClawMeetsClient for server communication (optional, for agent runtime)
            git_url: Git repo URL/path to clone from (optional, for code-aware sandbox)
            git_ignored_folder: Folder name for git-ignored deliverables (default: ".bus-files")
            claude_plugin_dirs: Claude plugin directories (optional, passed as --plugin-dir)
            notification_center: In-memory pub/sub dispatcher for cross-component events
        """
        self._base_dir = base_dir
        self._cli = cli
        self._knowledge_dirs = knowledge_dirs or []
        self._client = client
        self._action_executor: Optional["ActionBlockExecutor"] = None
        self._git_url = git_url
        self._git_ignored_folder = git_ignored_folder
        self._claude_plugin_dirs = claude_plugin_dirs or []
        self._notification_center = notification_center

    @property
    def cli(self) -> Optional["ClaudeCLI"]:
        """Claude CLI for LLM invocation (None if not configured)."""
        return self._cli

    @property
    def knowledge_dirs(self) -> list[Path]:
        """Additional directories for Claude access (e.g., knowledge bases)."""
        return self._knowledge_dirs

    @property
    def claude_plugin_dirs(self) -> list[Path]:
        """Claude plugin directories for skill access (passed as --plugin-dir)."""
        return self._claude_plugin_dirs

    @property
    def client(self) -> Optional["ClawMeetsClient"]:
        """HTTP client for server communication (None if not configured)."""
        return self._client

    @property
    def git_url(self) -> Optional[str]:
        """Git repo URL/path to clone from (None if not configured)."""
        return self._git_url

    @property
    def git_ignored_folder(self) -> str:
        """Folder name for git-ignored deliverables (e.g. '.bus-files')."""
        return self._git_ignored_folder

    @property
    def notification_center(self) -> NotificationCenter:
        """In-memory pub/sub dispatcher for cross-component events."""
        return self._notification_center

    @property
    def action_executor(self) -> Optional["ActionBlockExecutor"]:
        """ActionBlockExecutor for executing agent actions (created lazily from client)."""
        if self._action_executor is not None:
            return self._action_executor
        if self._client is not None:
            self._action_executor = ActionBlockExecutor(client=self._client)
        return self._action_executor

    @property
    def projects_dir(self) -> Path:
        """Directory containing project data (projects/{name}-{id}/...)."""
        return self._base_dir / "projects"

    @property
    def metadata_dir(self) -> Path:
        """Directory containing project metadata (metadata/projects/{name}-{id}/...)."""
        return self._base_dir / "metadata" / "projects"

    @property
    def participants_dir(self) -> Path:
        """Directory containing participant data (agents/, assistants/, users/)."""
        return self._base_dir

    def changelog_dir(self, project_id: str, project_name: str) -> Path:
        """Get changelog directory for a specific project.

        Args:
            project_id: The project ID
            project_name: The project name

        Returns:
            Path to the changelog directory
        """
        return self.metadata_dir / f"{project_name}-{project_id}"

    # ─────────────────────────────────────────────────────────────────────────
    # Project-Aware Path Methods (for agent runtime)
    # ─────────────────────────────────────────────────────────────────────────

    def project_dir(self, project_id: str, project_name: str) -> Path:
        """Synced project data directory (read-only for Claude).

        Contains files synced from the server via changelog.

        Args:
            project_id: The project ID
            project_name: The project name

        Returns:
            Path to the synced project directory
        """
        return self.projects_dir / f"{project_name}-{project_id}"

    def sandbox_dir(self, project_id: str, project_name: str) -> Path:
        """Sandbox directory for Claude writes.

        Claude writes files here; they are then pushed to server
        and synced back to project_dir via changelog.

        Args:
            project_id: The project ID
            project_name: The project name

        Returns:
            Path to the sandbox directory
        """
        return self._base_dir / "sandbox" / "projects" / f"{project_name}-{project_id}"

    def llm_log_dir(self, project_id: str, project_name: str) -> Path:
        """Directory for Claude CLI logs.

        Contains cli-stdout.log, cli-stderr.log, etc.

        Args:
            project_id: The project ID
            project_name: The project name

        Returns:
            Path to the LLM log directory
        """
        return self.metadata_dir / f"{project_name}-{project_id}"

    # ─────────────────────────────────────────────────────────────────────────
    # Changelog Subscriber Factory
    # ─────────────────────────────────────────────────────────────────────────

    def changelog_subscriber(
        self, project_id: str, project_name: str
    ) -> "ModelContextChangelogSubscriber":
        """Create a changelog subscriber bound to a specific project.

        Each subscriber instance is bound to a single project and handles
        filesystem writes for changelog entries.

        Args:
            project_id: The project ID
            project_name: The project name

        Returns:
            A ChangelogSubscriber bound to this ModelContext and project
        """
        return ModelContextChangelogSubscriber(self, project_id, project_name)

    def __repr__(self) -> str:
        return f"ModelContext(base_dir={self._base_dir})"


# ─────────────────────────────────────────────────────────────────────────────
# Changelog Subscriber Implementation
# ─────────────────────────────────────────────────────────────────────────────


class ModelContextChangelogSubscriber(ChangelogSubscriber):
    """Materializes changelog entries into filesystem state.

    ## Redo Log Architecture

    The changelog functions as a redo log. This subscriber:
    1. Receives each entry in version order from ChangelogRunloop
    2. Writes to filesystem via State classes (ProjectState, ChatroomState)
    3. MUST be idempotent (same entry replayed = same result)

    Idempotency is critical for:
    - Crash recovery (replay from last_synced_version)
    - New runner joining mid-project
    - Network retries

    ## Entry → State Mapping

    - PROJECT_CREATED → ProjectState.create()
    - ROOM_CREATED → ChatroomState.create() + ProjectState.add_participant() for each participant
    - MESSAGE → ChatroomState.append_message()
    - FILE_CREATED/UPDATED → ChatroomState.write_file()
    - PROJECT_COMPLETED → ProjectState.complete()

    ## Priority

    This subscriber runs at priority 0 (default), ensuring filesystem writes
    complete before higher-priority subscribers (e.g., ParticipantNotifier
    at priority 200) fire callbacks.

    Usage:
        ctx = ModelContext(base_dir=Path("~/.clawmeets").expanduser())
        subscriber = ctx.changelog_subscriber(project_id, project_name)
        runloop.add_subscriber(subscriber)
    """

    def __init__(
        self, model_ctx: ModelContext, project_id: str, project_name: str
    ) -> None:
        """Initialize subscriber with model context and project binding.

        Args:
            model_ctx: The ModelContext for filesystem I/O
            project_id: The project ID this subscriber handles
            project_name: The project name for path resolution
        """
        self._model_ctx = model_ctx
        self._project_id = project_id
        self._project_name = project_name

    # ─────────────────────────────────────────────────────────────────────────
    # ChangelogSubscriber Interface Implementation
    # ─────────────────────────────────────────────────────────────────────────

    async def on_entry(
        self,
        entry: ChangelogEntry,
        project_id: str,
        project_name: str,
    ) -> None:
        """Process a single changelog entry and write to disk.

        This is the main ChangelogSubscriber interface method.

        Note: project_id and project_name parameters are provided by the runloop
        interface but this subscriber uses its bound values for consistency.

        Args:
            entry: The changelog entry to process
            project_id: The project ID (from runloop, ignored in favor of self._project_id)
            project_name: The project name (from runloop, ignored in favor of self._project_name)
        """
        match entry.entry_type:
            case ChangelogEntryType.PROJECT_CREATED:
                await self._handle_project_created(entry)

            case ChangelogEntryType.ROOM_CREATED:
                await self._handle_room_created(entry)

            case ChangelogEntryType.MESSAGE:
                await self._handle_message(entry)

            case ChangelogEntryType.FILE_CREATED | ChangelogEntryType.FILE_UPDATED:
                await self._handle_file_update(entry)

            case ChangelogEntryType.PROJECT_COMPLETED:
                await self._handle_project_completed(entry)

            case ChangelogEntryType.PARTICIPANT_ADDED:
                await self._handle_participant_added(entry)

            case ChangelogEntryType.BATCH_COMPLETE | ChangelogEntryType.BATCH_TIMEOUT:
                pass  # No filesystem action needed - coordination events

    async def _handle_project_created(self, entry: ChangelogEntry) -> None:
        """Create project directories and write project meta.json.

        Args:
            entry: The PROJECT_CREATED changelog entry
        """
        payload: ProjectCreatedPayload = entry.payload  # type: ignore[assignment]

        # Use ProjectState.create() to create project directories and meta.json
        ProjectState.create(
            project_id=payload.project_id,
            project_name=payload.project_name,
            coordinator_id=payload.coordinator_id,
            coordinator_name=payload.coordinator_name,
            request=payload.request,
            created_by=payload.created_by,
            created_at=entry.timestamp,
            ctx=self._model_ctx,
            agent_pool=getattr(payload, "agent_pool", "verified"),
        )

    async def _handle_room_created(self, entry: ChangelogEntry) -> None:
        """Create chatroom directories and update project metadata.

        Args:
            entry: The ROOM_CREATED changelog entry
        """
        payload: RoomCreatedPayload = entry.payload  # type: ignore[assignment]
        chatroom_name = payload.chatroom_name

        # Convert RoomCreatedParticipant list to dict format for ChatroomState.create()
        participants_data = [{"id": p.id, "name": p.name} for p in payload.participants]

        # Use ChatroomState.create() to create chatroom directories and meta.json
        # Chatrooms list is derived from filesystem, no add_chatroom() needed
        ChatroomState.create(
            project_id=self._project_id,
            project_name=self._project_name,
            chatroom_name=chatroom_name,
            participants=participants_data,
            created_at=entry.timestamp,
            ctx=self._model_ctx,
        )

        # Update project meta.json with new participants
        project = Project.get(self._project_id, self._model_ctx)
        internal = project.state()
        for p in payload.participants:
            internal.add_participant(p.id)

    async def _handle_participant_added(self, entry: ChangelogEntry) -> None:
        """Add a participant to an existing chatroom's PARTICIPANTS.ndjson.

        Args:
            entry: The PARTICIPANT_ADDED changelog entry
        """
        payload: ParticipantAddedPayload = entry.payload  # type: ignore[assignment]
        chatroom = Chatroom.get(self._project_id, payload.chatroom_name, self._model_ctx)
        if chatroom:
            chatroom.state().add_participant(
                payload.participant_id,
                payload.participant_name,
                entry.timestamp,
            )

    async def _handle_message(self, entry: ChangelogEntry) -> None:
        """Append message to CHATS.ndjson.

        No dedup check needed - runloop's per-entry version tracking handles it.

        Args:
            entry: The MESSAGE changelog entry
        """
        payload: MessagePayload = entry.payload  # type: ignore[assignment]
        chatroom_name = payload.chatroom_name

        # Load chatroom and append message via internal model
        chatroom = Chatroom.get(self._project_id, chatroom_name, self._model_ctx)
        chat_message = ChatMessage.from_message_payload(payload)
        chatroom.state().append_message(chat_message)

    async def _handle_file_update(self, entry: ChangelogEntry) -> None:
        """Create or update a data file.

        Args:
            entry: The FILE_CREATED or FILE_UPDATED changelog entry
        """
        payload: FilePayload = entry.payload  # type: ignore[assignment]
        chatroom_name = payload.chatroom_name

        # Load chatroom and write file via internal model
        chatroom = Chatroom.get(self._project_id, chatroom_name, self._model_ctx)
        content = FileUtil.from_base64(payload.content_b64)
        chatroom.state().write_file(payload.filename, content)

    async def _handle_project_completed(self, entry: ChangelogEntry) -> None:
        """Update project status to completed in meta.json.

        Args:
            entry: The PROJECT_COMPLETED changelog entry
        """
        project = Project.get(self._project_id, self._model_ctx)
        project.state().complete()

