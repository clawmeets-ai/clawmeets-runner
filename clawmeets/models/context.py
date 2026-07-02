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
- cli: LLMProvider instance for LLM invocation (e.g. ClaudeCLI, CodexCLI)
- knowledge_dirs: Additional directories for LLM access
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from clawmeets.sync.changelog import (
    BatchTimeoutPayload,
    ChangelogEntry,
    ChangelogEntryType,
    ChatroomClearedPayload,
    ProjectAllowlistUpdatedPayload,
    ProjectCreatedPayload,
    MessagePayload,
    FilePayload,
    RoomCreatedPayload,
    RoomDeletedPayload,
    ParticipantAddedPayload,
)
from clawmeets.sync.subscriber import ChangelogSubscriber
from clawmeets.utils.file_io import FileUtil
from .project import Project, ProjectState
from .chatroom import Chatroom, ChatroomState
from .chat_message import ChatBatchTimeoutEvent, ChatFileEvent, ChatMessage

from clawmeets.api.action_executor import ActionBlockExecutor
from clawmeets.api.client import ClawMeetsClient
from clawmeets.llm.base import LLMProvider
from clawmeets.utils.notification_center import NotificationCenter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clawmeets.runner.invocation_registry import InvocationRegistry


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
        cli: Optional["LLMProvider"] = None,
        knowledge_dirs: Optional[list[Path]] = None,
        client: Optional["ClawMeetsClient"] = None,
        claude_plugin_dirs: Optional[list[Path]] = None,
        dwh_dir: Optional[Path] = None,
        git_url: Optional[str] = None,
    ) -> None:
        """Initialize context with a single base directory.

        All paths are derived from base_dir:
        - projects_dir: base_dir/projects/{name}-{id}/chatrooms/{chatroom_name}/...
        - metadata_dir: base_dir/metadata/projects/{name}-{id}/...
        - participants_dir: base_dir (contains agents/, assistants/, users/)

        Optional runtime fields for agent execution:
        - cli: LLMProvider instance for LLM invocation (None if not configured)
        - knowledge_dirs: Additional directories for LLM access (e.g., knowledge bases)
        - client: ClawMeetsClient for HTTP operations (None if not configured)
        - claude_plugin_dirs: Claude plugin directories for skill access.
            Used only by ClaudeCLI; other providers ignore this field.

        Git configuration (git_url) is per-AGENT, set in card.json
        local_settings and surfaced to the LLM via $CLAWMEETS_AGENT_GIT_URL.
        The git-workflow skill — not the runner — drives clone/branch/commit/push.
        This field is rendered as a one-line binding nudge in the agent prompt.

        Args:
            base_dir: Base directory for all data
            cli: LLM provider for invocation (optional, for agent runtime)
            knowledge_dirs: Additional directories for LLM access (optional)
            client: ClawMeetsClient for server communication (optional, for agent runtime)
            claude_plugin_dirs: Claude plugin directories (optional, Claude-specific)
            notification_center: In-memory pub/sub dispatcher for cross-component events
        """
        self._base_dir = base_dir
        self._cli = cli
        self._knowledge_dirs = knowledge_dirs or []
        self._client = client
        self._action_executor: Optional["ActionBlockExecutor"] = None
        self._claude_plugin_dirs = claude_plugin_dirs or []
        self._notification_center = notification_center
        self._invocation_registry: Optional["InvocationRegistry"] = None
        self._dwh_dir = dwh_dir
        self._git_url = git_url or None

    @property
    def cli(self) -> Optional["LLMProvider"]:
        """LLM provider for invocation (None if not configured)."""
        return self._cli

    def update_cli(self, cli: "LLMProvider") -> None:
        """Replace the LLM provider. Takes effect on the next LLM invocation;
        in-flight invocations hold the prior reference and finish on it."""
        self._cli = cli

    @property
    def knowledge_dirs(self) -> list[Path]:
        """Additional directories for LLM access (e.g., knowledge bases)."""
        return self._knowledge_dirs

    def update_knowledge_dirs(self, dirs: list[Path]) -> None:
        """Replace knowledge directories. Takes effect on the next LLM invocation."""
        self._knowledge_dirs = dirs

    @property
    def dwh_dir(self) -> Optional[Path]:
        """Personal data-warehouse root (network-shared across runners). None if not configured."""
        return self._dwh_dir

    def update_dwh_dir(self, dwh_dir: Optional[Path]) -> None:
        """Replace the dwh_dir. Takes effect on the next LLM invocation."""
        self._dwh_dir = dwh_dir

    @property
    def git_url(self) -> Optional[str]:
        """Git repo this agent is bound to (None if not configured). Surfaced
        in the prompt; the actual clone/branch/commit/push is run by the
        git-workflow skill via $CLAWMEETS_AGENT_GIT_URL."""
        return self._git_url

    def update_git_url(self, git_url: Optional[str]) -> None:
        """Replace the bound git_url. Takes effect on the next LLM invocation."""
        self._git_url = git_url or None

    @property
    def claude_plugin_dirs(self) -> list[Path]:
        """Claude plugin directories for skill access (passed as --plugin-dir)."""
        return self._claude_plugin_dirs

    @property
    def client(self) -> Optional["ClawMeetsClient"]:
        """HTTP client for server communication (None if not configured)."""
        return self._client

    @property
    def notification_center(self) -> NotificationCenter:
        """In-memory pub/sub dispatcher for cross-component events."""
        return self._notification_center

    @property
    def invocation_registry(self) -> Optional["InvocationRegistry"]:
        """Per-runner registry of in-flight LLM tasks (None on the server side)."""
        return self._invocation_registry

    def set_invocation_registry(self, registry: "InvocationRegistry") -> None:
        """Attach the runner's InvocationRegistry. Called once at runner startup."""
        self._invocation_registry = registry

    @property
    def action_executor(self) -> Optional["ActionBlockExecutor"]:
        """ActionBlockExecutor for executing agent actions (created lazily from client)."""
        if self._action_executor is not None:
            return self._action_executor
        if self._client is not None:
            self._action_executor = ActionBlockExecutor(client=self._client)
        return self._action_executor

    @property
    def base_dir(self) -> Path:
        """Root directory. On the runner this is the agent's own dir
        (`~/.clawmeets/agents/<name>-<id>/`); on the server it is the
        shared data_dir. Subtree-specific properties (`projects_dir`,
        `memory_dir`, `packs_dir`, `mcp_configs_dir`, ...) compose on
        top of this path."""
        return self._base_dir

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

    # ─────────────────────────────────────────────────────────────────────────
    # Runner-only paths under `agent_dir/`
    #
    # On the runner, base_dir IS the agent's own directory
    # (`~/.clawmeets/agents/<name>-<id>/`). These three properties name the
    # purpose-specific subtrees under it:
    #   - memory/         agent-authored memory (USER.md, REFERENCES.md,
    #                     KNOWLEDGE_PACKS.md, learnings/)
    #   - knowledge_packs/  runner-synced pack installs (content; the index
    #                       lives at memory/KNOWLEDGE_PACKS.md)
    #   - mcp-hub/configs/  per-MCP runtime config files (siblings of
    #                       mcp-hub/manifests/ and mcp-hub/servers/)
    #
    # Server-side code never accesses these — base_dir there is the server's
    # data_dir and these paths would be nonsensical.
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def memory_dir(self) -> Path:
        """Agent-authored memory: USER.md, REFERENCES.md, KNOWLEDGE_PACKS.md,
        learnings/. Runner-only path; meaningless on the server."""
        return self._base_dir / "memory"

    @property
    def packs_dir(self) -> Path:
        """Runner-synced knowledge-pack installs. Content only; the index
        lives at `memory/KNOWLEDGE_PACKS.md`. Runner-only path."""
        return self._base_dir / "knowledge_packs"

    @property
    def mcp_configs_dir(self) -> Path:
        """Per-MCP runtime config files at `mcp-hub/configs/<name>.json`,
        sibling of the existing `mcp-hub/manifests/` and `mcp-hub/servers/`
        subdirs. Runner-only path."""
        return self._base_dir / "mcp-hub" / "configs"

    @property
    def mcp_dist_dir(self) -> Path:
        """Per-provider MCP configs rendered by `McpManager.render_dist` on
        MCP_SYNC. The runner passes this to `cli.invoke(mcp_config_dir=...)`
        so each provider symlinks the relevant per-format file
        (`.mcp.json` / `.gemini/settings.json` / `.codex/config.toml`) into
        the per-invocation working dir. Runner-only path."""
        return self._base_dir / "mcp-hub" / "dist"

    def skill_source_dirs(self, role: Optional[str] = None) -> list[Path]:
        """Skill source-of-truth dirs in scan order, passed to
        `cli.invoke(skill_source_dirs=...)`. Each provider's
        `_prepare_invocation` flattens these into the CLI's native
        cwd skill-discovery path (`.claude/skills` for Claude,
        `.agents/skills` for Codex + Gemini).

        Order matters: `materialize_skill_tree` lets later sources win
        on name collisions. With `role` set, the per-audience
        ``system-skill-hub/skills-<role>/`` tree is prepended as the
        BASE layer so a user-installed (skill-hub) or agent-authored
        (personal-skill-hub) skill of the same name overrides it.
        Runner-only path."""
        dirs: list[Path] = []
        if role is not None:
            dirs.append(self._base_dir / "system-skill-hub" / f"skills-{role}")
        dirs.extend([
            self._base_dir / "skill-hub" / "skills",
            self._base_dir / "personal-skill-hub" / "skills",
        ])
        return dirs

    @property
    def skill_configs_dir(self) -> Path:
        """Per-skill runtime config files at `skill-hub/configs/<name>.json`,
        sibling of the synced `skill-hub/skills/`. Runner-only path. The
        SKILL.md for a configurable skill (e.g. `etl`) reads this file
        at trigger time."""
        return self._base_dir / "skill-hub" / "configs"

    def mcp_config_path(self, mcp_name: str) -> Path:
        """Path to a specific MCP's per-agent config file. The MCP tool
        receives this path verbatim as its `config_file` argument."""
        return self.mcp_configs_dir / f"{mcp_name}.json"

    def installed_mcp_config_files(self) -> dict[str, Path]:
        """Per-installed-MCP config file paths, discovered from cached
        manifests at `{base_dir}/mcp-hub/manifests/<name>.json`.

        Returns a mapping `{mcp_name: config_file_path}` suitable for
        passing to the prompt builder's `mcp_config_files=` argument so
        the prompt's `MCP CONFIG FILES` block names each installed MCP's
        config file explicitly. Empty dict when no MCPs are installed.
        """
        manifests_dir = self._base_dir / "mcp-hub" / "manifests"
        if not manifests_dir.is_dir():
            return {}
        return {
            p.stem: self.mcp_config_path(p.stem)
            for p in sorted(manifests_dir.iterdir())
            if p.is_file() and p.suffix == ".json"
        }

    def installed_skill_config_files(self) -> dict[str, Path]:
        """Per-installed-skill config file paths, discovered from the
        synced `{base_dir}/skill-hub/skills/<name>/SKILL.md` directories.

        Returns a mapping `{skill_name: config_file_path}` suitable for
        passing to the prompt builder's `skill_config_files=` argument
        so the prompt's `SKILL CONFIG FILES` block names each installed
        skill's per-agent config file explicitly. The LLM is instructed
        to Read these files before invoking a skill so operator-set
        per-skill policy (e.g. `invoke_when`, `providers.<n>.use_for`
        for the consult skill) actually influences routing decisions.

        Skills without a sibling config file are still included — the
        path is emitted so the LLM knows where to look, and the
        SKILL.md handles the missing-config fallback. Empty dict when
        no skills are installed.

        The 4 memory-loop trigger skills (reflect/lint/references/
        personalize) ship via the runner plugin (`plugins/clawmeets/`),
        not the per-agent skill-hub — they aren't returned here; the
        prompt builder lists them separately via the existing hardcoded
        section.
        """
        skills_dir = self._base_dir / "skill-hub" / "skills"
        if not skills_dir.is_dir():
            return {}
        return {
            child.name: self.skill_configs_dir / f"{child.name}.json"
            for child in sorted(skills_dir.iterdir())
            if child.is_dir() and (child / "SKILL.md").exists()
        }

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
    - BATCH_TIMEOUT → ChatroomState.append_batch_timeout()
    - PROJECT_COMPLETED → ProjectState.complete()
    - PROJECT_REACTIVATED → ProjectState.reactivate()
    - PROJECT_ALLOWLIST_UPDATED → ProjectState.apply_allowlist_update()

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

            case ChangelogEntryType.ROOM_DELETED:
                await self._handle_room_deleted(entry)

            case ChangelogEntryType.MESSAGE:
                await self._handle_message(entry)

            case ChangelogEntryType.FILE_CREATED | ChangelogEntryType.FILE_UPDATED:
                await self._handle_file_update(entry)

            case ChangelogEntryType.PROJECT_COMPLETED:
                await self._handle_project_completed(entry)

            case ChangelogEntryType.PROJECT_REACTIVATED:
                await self._handle_project_reactivated(entry)

            case ChangelogEntryType.PARTICIPANT_ADDED:
                await self._handle_participant_added(entry)

            case ChangelogEntryType.BATCH_TIMEOUT:
                await self._handle_batch_timeout(entry)

            case ChangelogEntryType.CHATROOM_CLEARED:
                await self._handle_chatroom_cleared(entry)

            case ChangelogEntryType.PROJECT_ALLOWLIST_UPDATED:
                await self._handle_project_allowlist_updated(entry)

            case ChangelogEntryType.BATCH_COMPLETE:
                pass  # Existing reply-window logic in the chip already covers it

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
            agent_teams=getattr(payload, "agent_teams", []) or [],
            agent_names=getattr(payload, "agent_names", []) or [],
            surface=getattr(payload, "surface", "regular"),
        )

    async def _handle_room_created(self, entry: ChangelogEntry) -> None:
        """Create chatroom directories and update project metadata.

        Args:
            entry: The ROOM_CREATED changelog entry
        """
        payload: RoomCreatedPayload = entry.payload  # type: ignore[assignment]
        chatroom_name = payload.chatroom_name

        # Convert RoomCreatedParticipant list to dict format for ChatroomState.create()
        participants_data = [
            {"id": p.id, "name": p.name, "external": p.external}
            for p in payload.participants
        ]

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

    async def _handle_room_deleted(self, entry: ChangelogEntry) -> None:
        """Remove a chatroom's directories from disk (synced + sandbox + meta).

        Idempotent: missing dirs are a no-op. Defends `shared-context` and
        `user-communication` here too — those rooms are never deletable
        through this entry even if a buggy caller emitted one.
        """
        payload: RoomDeletedPayload = entry.payload  # type: ignore[assignment]
        chatroom_name = payload.chatroom_name
        if chatroom_name in ("shared-context", "user-communication"):
            logger.warning(
                f"ROOM_DELETED: refusing to delete system room {chatroom_name!r} "
                f"in project {self._project_id[:8]}"
            )
            return

        project_dir = self._model_ctx.project_dir(self._project_id, self._project_name)
        sandbox_dir = self._model_ctx.sandbox_dir(self._project_id, self._project_name)
        meta_dir = self._model_ctx.changelog_dir(self._project_id, self._project_name)
        for parent in (project_dir, sandbox_dir, meta_dir):
            target = parent / "chatrooms" / chatroom_name
            if target.exists():
                shutil.rmtree(target)
                logger.info(f"ROOM_DELETED: removed {target}")

    async def _handle_participant_added(self, entry: ChangelogEntry) -> None:
        """Add a participant to an existing chatroom's PARTICIPANTS.ndjson
        and to the project's participating_agents list.

        Updating ``participating_agents`` here makes PARTICIPANT_ADDED a single
        source-of-truth for membership at both chatroom and project levels —
        without it, the DM register-time stitch would update the chatroom but
        leave the new agent invisible to ``GET /participants/{id}/projects``
        and ``broadcast_to_project``, both of which key off
        ``project.participating_agents``. Idempotent at both levels.

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
                external=getattr(payload, "external", False),
            )
        project = Project.get(self._project_id, self._model_ctx)
        project.state().add_participant(payload.participant_id)

    async def _handle_message(self, entry: ChangelogEntry) -> None:
        """Append message to CHATS.ndjson.

        No dedup check needed - runloop's per-entry version tracking handles it.

        Args:
            entry: The MESSAGE changelog entry
        """
        payload: MessagePayload = entry.payload  # type: ignore[assignment]
        chatroom_name = payload.chatroom_name

        # Load chatroom and append message via internal model. The changelog is
        # the source of truth — a MESSAGE for a room with no prior ROOM_CREATED
        # is a corruption signal, so fail loudly.
        chatroom = Chatroom.get(self._project_id, chatroom_name, self._model_ctx)
        if chatroom is None:
            raise ValueError(
                f"Chatroom {chatroom_name} not found in project {self._project_id}"
            )
        chat_message = ChatMessage.from_message_payload(
            payload,
            version=entry.version,
            source_version=entry.source_version,
        )
        chatroom.state().append_message(chat_message)

    async def _handle_file_update(self, entry: ChangelogEntry) -> None:
        """Create or update a data file, and log the touch to CHATS.ndjson.

        Args:
            entry: The FILE_CREATED or FILE_UPDATED changelog entry
        """
        payload: FilePayload = entry.payload  # type: ignore[assignment]
        chatroom_name = payload.chatroom_name

        # Load chatroom and write file via internal model. Same corruption
        # signal as _handle_message: FILE_* without prior ROOM_CREATED.
        chatroom = Chatroom.get(self._project_id, chatroom_name, self._model_ctx)
        if chatroom is None:
            raise ValueError(
                f"Chatroom {chatroom_name} not found in project {self._project_id}"
            )
        content = FileUtil.from_base64(payload.content_b64)
        chatroom.state().write_file(payload.filename, content)

        # Log the file touch to CHATS.ndjson so the chat stream carries an
        # inline record of what the sender did. source_version links back to
        # the message entry that triggered the touch (set by the caller when
        # uploading; see ActionBlockExecutor and the files route).
        event_type = (
            "file_created"
            if entry.entry_type == ChangelogEntryType.FILE_CREATED
            else "file_updated"
        )
        file_event = ChatFileEvent(
            entry_type=event_type,
            ts=entry.timestamp,
            from_participant_id=payload.from_participant_id,
            from_participant_name=payload.from_participant_name,
            filename=payload.filename,
            version=entry.version,
            source_version=entry.source_version,
        )
        chatroom.state().append_file_event(file_event)

    async def _handle_batch_timeout(self, entry: ChangelogEntry) -> None:
        """Materialize a BATCH_TIMEOUT entry into the chatroom's CHATS.ndjson.

        The frontend chip uses this row to flip the per-recipient status
        from "AWAITING" to "TIMED OUT". source_version on the changelog
        entry already points at the @mention message that opened the
        batch, which matches what the chip keys off of.

        Args:
            entry: The BATCH_TIMEOUT changelog entry
        """
        payload: BatchTimeoutPayload = entry.payload  # type: ignore[assignment]
        chatroom = Chatroom.get(self._project_id, payload.chatroom_name, self._model_ctx)
        if chatroom is None:
            return
        event = ChatBatchTimeoutEvent(
            ts=entry.timestamp,
            message_id=payload.message_id,
            coordinator_id=payload.coordinator_id,
            responded_participants=list(payload.responded_participants),
            timed_out_participants=list(payload.timed_out_participants),
            version=entry.version,
            source_version=entry.source_version,
        )
        chatroom.state().append_batch_timeout(event)

    async def _handle_project_completed(self, entry: ChangelogEntry) -> None:
        """Update project status to completed in meta.json.

        Args:
            entry: The PROJECT_COMPLETED changelog entry
        """
        project = Project.get(self._project_id, self._model_ctx)
        project.state().complete()

    async def _handle_project_reactivated(self, entry: ChangelogEntry) -> None:
        """Flip project status back to active in meta.json.

        Mirrors ``_handle_project_completed``; runs identically server-side
        and on every runner that syncs the entry.

        Args:
            entry: The PROJECT_REACTIVATED changelog entry
        """
        project = Project.get(self._project_id, self._model_ctx)
        project.state().reactivate()

    async def _handle_project_allowlist_updated(self, entry: ChangelogEntry) -> None:
        """Replay an allowlist snapshot refresh into project meta.json."""
        payload: ProjectAllowlistUpdatedPayload = entry.payload  # type: ignore[assignment]
        project = Project.get(self._project_id, self._model_ctx)
        if project is None:
            logger.warning(
                f"PROJECT_ALLOWLIST_UPDATED: project {self._project_id[:8]} not found; "
                "skipping (likely a stale entry replayed before the project meta is on disk)"
            )
            return
        project.state().apply_allowlist_update(
            agent_names=payload.agent_names,
            agent_teams=payload.agent_teams,
        )

    async def _handle_chatroom_cleared(self, entry: ChangelogEntry) -> None:
        """Wipe CHATS.ndjson and archive prior contents to .bak sibling.

        Runs identically on the server (where the entry was minted) and on
        every runner that syncs the entry. The archive filename in the
        payload is reused on every side so the resulting .bak names line up
        across server + runners — useful for support diffing.

        Args:
            entry: The CHATROOM_CLEARED changelog entry
        """
        payload: ChatroomClearedPayload = entry.payload  # type: ignore[assignment]
        chatroom = Chatroom.get(self._project_id, payload.chatroom_name, self._model_ctx)
        if chatroom is None:
            logger.warning(
                f"CHATROOM_CLEARED: chatroom {payload.chatroom_name!r} not found "
                f"in project {self._project_id[:8]}; skipping"
            )
            return
        chatroom.state().clear_history(
            archive_filename=payload.archive_filename,
            cleared_at=payload.cleared_at,
            cleared_through_version=payload.cleared_through_version,
        )

