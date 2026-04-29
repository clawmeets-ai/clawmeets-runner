# SPDX-License-Identifier: MIT
"""
clawmeets/models/agent.py
Public worker agent implementation using composition.

Agents are registered by admins and discoverable in the registry.
They execute specific tasks when addressed by a coordinator.

All state is read from the filesystem (card.json) - no in-memory state.
Extends PersistableParticipant for Active Record persistence.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional

from .participant import ParticipantRole, OperationalMode
from .persistable import PersistableParticipant
from ..api.actions import ActionBlock, COORDINATOR_ACTION_SCHEMA, WORKER_ACTION_SCHEMA
from ..api.responses import AgentStatus
from ..llm.base import LLMInvocationError, LLMRateLimitError, LLMTimeoutError
from ..llm.prompt_builder import CoordinatorPromptBuilder, create_prompt_builder
from ..runner.invocation_registry import invoke_with_registry as _invoke_with_registry
from ..utils.agent_namespace import short_name
from ..utils.file_io import FileUtil

if TYPE_CHECKING:
    from ..api.client import ClawMeetsClient
    from .context import ModelContext
    from .chat_message import ChatMessage

logger = logging.getLogger(__name__)

# Retry configuration for transient LLM CLI failures
_MAX_RETRIES = 2  # Total attempts: 3 (1 original + 2 retries)
_INITIAL_RETRY_DELAY = 30  # seconds
_TRANSIENT_INDICATORS = ("overloaded", "rate_limit", "529", "503", "too many requests")


def _is_transient_error(error: LLMInvocationError) -> bool:
    """Check if an LLMInvocationError is likely transient (retryable)."""
    if isinstance(error, LLMRateLimitError):
        return False  # Rate limits should not be retried with short backoff
    if isinstance(error, LLMTimeoutError):
        return True
    msg = str(error).lower()
    return any(indicator in msg for indicator in _TRANSIENT_INDICATORS)


class Agent(PersistableParticipant):
    """
    Public worker agent - executes specific tasks using composition.
    Registered by admin, discoverable in registry.

    Agents only respond when explicitly addressed via expects_response_from.
    They execute tasks and report results back to the coordinator.

    All state is read from the filesystem (card.json) on each property access.
    This ensures the model always reflects the current state on disk.

    Extends PersistableParticipant for Active Record methods:
        - Agent.get(), Agent.get_by_name(), Agent.list_all()
        - Agent.register(), Agent.verify_token()
        - agent.save(), agent.update_status(), agent.heartbeat()
        - agent.to_response()

    Composition:
        - ClaudeCLI: Direct Claude CLI invocation
        - ActionBlockExecutor: Processes Claude output and executes actions via HTTP
        - Prompt builders created on-demand via create_prompt_builder()
    """

    # Active Record: directory subdirectory for agents
    _role_subdir: ClassVar[str] = "agents"

    @classmethod
    async def sync_from_server(
        cls,
        ctx: "ModelContext",
        exclude_ids: set[str],
        owner_username: Optional[str] = None,
    ) -> int:
        """Sync all worker agents from server to local filesystem.

        Fetches the agent registry from server and persists each agent's
        card.json locally. Called on startup to populate local agents/
        directory with current registry state.

        Also generates a global AGENTS.md file listing all available agents,
        which coordinators reference during prompts.

        Args:
            ctx: ModelContext for filesystem access (must have client configured)
            exclude_ids: Set of agent IDs to skip (pass empty set if none)
            owner_username: Username of the runner's owner. When set, agents
                owned by this user render with short names in AGENTS.md.

        Returns:
            Number of agents synced

        Raises:
            ValueError: If ctx.client is not configured
        """
        if ctx.client is None:
            raise ValueError("ModelContext.client must be configured for sync_from_server")
        agents = await ctx.client.list_agents()
        synced_count = 0
        owner_user_id: Optional[str] = None
        for agent_data in agents:
            agent_id = agent_data["id"]
            if agent_id in exclude_ids:
                continue

            # Sync card.json for this worker agent
            agent = cls.get_or_create(agent_id, ctx)
            agent.update_card(**agent_data)
            synced_count += 1

            # Locate the owner's user id by matching username against any
            # agent registered by that user (agent name starts with
            # ``{owner_username}-``). Avoids an extra server round-trip.
            if (
                owner_user_id is None
                and owner_username
                and agent_data.get("registered_by")
                and agent_data.get("name", "").startswith(f"{owner_username}-")
            ):
                owner_user_id = agent_data["registered_by"]

        logger.info(f"Synced {synced_count} worker agents from server")

        # Generate global AGENTS.md file after syncing all agents
        all_agents = cls.list_all(ctx, discoverable_only=True)
        cls._generate_agents_md(
            all_agents,
            ctx.participants_dir / "AGENTS.md",
            owner_username=owner_username,
            owner_user_id=owner_user_id,
        )

        return synced_count

    @classmethod
    def _generate_agents_md(
        cls,
        agents: list["Agent"],
        output_path: "Path",
        owner_username: Optional[str] = None,
        owner_user_id: Optional[str] = None,
    ) -> None:
        """Generate AGENTS.md file listing all available agents.

        This file is referenced by coordinators in their prompts to see
        which agents are available for delegation.

        When ``owner_username``/``owner_user_id`` are provided, agents owned by
        that user are listed by their short name (without the ``{owner}-``
        prefix) so the coordinator can address them concisely. Other agents
        keep their fully-qualified name.

        Args:
            agents: List of Agent objects
            output_path: Path to write AGENTS.md
            owner_username: Runner owner's username (used to strip prefixes)
            owner_user_id: Runner owner's user id (used to match registered_by)
        """
        if agents:
            rows: list[str] = []
            for a in agents:
                display = a.name
                if owner_user_id and a.registered_by == owner_user_id:
                    display = short_name(a.name, owner_username)
                if display != a.name:
                    rows.append(
                        f"| {display} | {a.description} | {a.status.value} | (full name: `{a.name}`) |"
                    )
                else:
                    rows.append(f"| {display} | {a.description} | {a.status.value} | |")
            table = "\n".join(rows)
        else:
            table = "| (no agents registered) | - | - | |"

        namespace_note = ""
        if owner_username:
            namespace_note = (
                f"- Agents you own render by **short name**; address them with "
                f"``@short-name`` (e.g. ``@researcher`` instead of "
                f"``@{owner_username}-researcher``). Fully-qualified names also work.\n"
            )

        content = f"""# Available Agents

| Agent | Role | Status | Notes |
|-------|------|--------|-------|
{table}

## Notes
- Use @mentions to delegate work: "@agent-name please do X"
- Check agent status before delegating (online agents respond faster)
- Agent names (not IDs) should be used in chatroom invites
{namespace_note}"""
        FileUtil.write(output_path, content, "text")
        logger.debug(f"Generated AGENTS.md at {output_path}")

    def __init__(
        self,
        id: str,
        model_ctx: "ModelContext",
    ) -> None:
        """Initialize an Agent.

        For Active Record operations (server-side), only id and model_ctx are needed.
        For task execution (runner-side), model_ctx should have cli and client configured.

        Runtime dependencies are accessed via model_ctx:
        - model_ctx.cli: Claude CLI for LLM invocation
        - model_ctx.knowledge_dirs: Additional directories for Claude access
        - model_ctx.client: ClawMeetsClient for HTTP operations
        - model_ctx.action_executor: ActionBlockExecutor for action execution

        Args:
            id: The agent's unique identifier
            model_ctx: ModelContext for filesystem access (may include cli/knowledge_dirs/client)
        """
        super().__init__(id, model_ctx)

    # ─────────────────────────────────────────────────────────────────────────
    # Role Property (from Participant ABC)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def role(self) -> ParticipantRole:
        """Return AGENT role."""
        return ParticipantRole.AGENT

    @property
    def linked_user_id(self) -> Optional[str]:
        """User this agent is linked to (alias for registered_by).

        Returns the user ID of whoever registered this agent. Every agent is
        owned by exactly one user; public agents are visible to all, private
        agents (``discoverable_through_registry=false``) are visible only to
        their owner's coordinator via AGENTS.md.
        """
        return self.registered_by

    async def on_message(
        self,
        project_id: str,
        chatroom_name: str,
        message: "ChatMessage",
        addressed_to_me: bool,
        trigger_version: int,
    ) -> None:
        """Route a message based on the agent's role in this project.

        If the agent is the project's coordinator, handle as coordinator
        (user-communication → user-request handler; other rooms → coordinate).
        Otherwise handle as worker (only when addressed).
        """
        from .project import Project
        from .chatroom import Chatroom

        project = Project.get(project_id, self._model_ctx)
        if project is None:
            logger.warning(
                f"Agent {self.name}: project {project_id[:8]} not found locally; skipping message"
            )
            return

        if self.is_coordinator_for(project):
            chatroom = Chatroom.get(project_id, chatroom_name, self._model_ctx)
            if chatroom is None:
                logger.warning(
                    f"Agent {self.name} (coordinator): chatroom {chatroom_name!r} not found "
                    f"for project {project_id[:8]}; skipping"
                )
                return
            if chatroom.is_user_communication_room:
                await self._handle_user_request(project_id, chatroom_name, message, trigger_version)
            elif addressed_to_me:
                await self._coordinate(project_id, chatroom_name, message, trigger_version)
            return

        # Worker mode
        if not addressed_to_me:
            return
        await self._execute_task(project_id, chatroom_name, message, trigger_version)

    async def _emit_acknowledgment(
        self,
        project_id: str,
        chatroom_name: str,
    ) -> None:
        """Emit an acknowledgment message before processing."""
        if not self._model_ctx.client:
            return

        await self._model_ctx.client.post_message(
            project_id=project_id,
            chatroom_name=chatroom_name,
            content="Message received, processing...",
            is_ack=True,
        )

    async def _execute_task(
        self,
        project_id: str,
        chatroom_name: str,
        message: "ChatMessage",
        trigger_version: int,
    ) -> None:
        """
        Execute the task requested in the message using Claude.

        Uses composition objects for prompt building, execution, and action execution.
        Requires cli and action_executor to be configured.
        Prompt builder is created on-demand based on operational mode.
        """
        agent_name = self.name  # Use property to get current name

        if not self._model_ctx.cli:
            logger.warning(f"Agent {agent_name}: CLI not configured, cannot execute task")
            return

        action_executor = self._model_ctx.action_executor
        if not action_executor:
            logger.warning(f"Agent {agent_name}: Client not configured, cannot execute task")
            return

        # Emit acknowledgment before processing
        await self._emit_acknowledgment(project_id, chatroom_name)

        # Get project for context
        from .project import Project
        project = Project.get(project_id, self._model_ctx)

        # Compute project-aware paths from ModelContext
        data_dir = self._model_ctx.project_dir(project_id, project.name)
        sandbox_dir = self._model_ctx.sandbox_dir(project_id, project.name)
        log_dir = self._model_ctx.llm_log_dir(project_id, project.name)

        # Compute additional_dirs: data_dir (if different from sandbox) + knowledge_dirs
        additional_dirs: list[Path] = []
        if data_dir != sandbox_dir:
            additional_dirs.append(data_dir)
        additional_dirs.extend(self._model_ctx.knowledge_dirs)

        # Create prompt builder on-demand for worker mode
        # Use project.coordinator_name to avoid lookup (worker may not have coordinator's card)
        prompt_builder = create_prompt_builder(
            OperationalMode.WORKER,
            coordinator_name=project.coordinator_name,
            capabilities=self.capabilities,
            git_ignored_folder=project.git_ignored_folder if project.git_url else None,
        )

        # Build prompt - extract message fields for Layer 0 compatibility
        prompt = prompt_builder.build_prompt(
            name=self.name,
            description=self.description,
            project_id=project_id,
            chatroom_name=chatroom_name,
            from_participant_name=message.from_participant_name or message.from_participant_id,
            message_content=message.content,
            data_dir=data_dir,
            project_name=project.name,
            knowledge_dirs=self._model_ctx.knowledge_dirs,
            is_dm=chatroom_name.startswith("dm-"),
        )

        # Execute using ClaudeCLI with retry for transient failures
        retry_delay = _INITIAL_RETRY_DELAY
        for attempt in range(_MAX_RETRIES + 1):
            try:
                action_block, usage = await _invoke_with_registry(
                    self._model_ctx,
                    project_id,
                    chatroom_name,
                    prompt,
                    sandbox_dir,
                    log_dir,
                    additional_dirs,
                    action_schema=WORKER_ACTION_SCHEMA,
                )
                break  # Success
            except LLMRateLimitError as e:
                logger.warning(
                    f"Agent {agent_name}: rate limited "
                    f"(type={e.rate_limit_type}, resets={e.resets_at_human})"
                )
                await self._post_error_notification(
                    project_id, chatroom_name, agent_name, e
                )
                raise
            except LLMInvocationError as e:
                if attempt < _MAX_RETRIES and _is_transient_error(e):
                    logger.warning(
                        f"Agent {agent_name}: transient failure (attempt {attempt + 1}/{_MAX_RETRIES + 1}), "
                        f"retrying in {retry_delay}s: {e}"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    await self._post_error_notification(
                        project_id, chatroom_name, agent_name, e
                    )
                    raise

        logger.info(
            f"Agent {agent_name}: Claude invocation complete "
            f"(cost=${usage.cost_usd:.4f}, tokens in={usage.input_tokens} out={usage.output_tokens})"
        )

        # Process using ActionBlockExecutor - executes actions via HTTP
        action_block.source_version = trigger_version
        replied_chatrooms = await action_executor.process(
            action_block=action_block,
            project_id=project_id,
            sandbox_dir=sandbox_dir,
        )

        # If the LLM didn't reply in the triggering chatroom, post a closure
        # so the server marks this worker as responded and clears PendingWork.
        # Without this, the typing indicator would persist until batch timeout.
        if chatroom_name not in replied_chatrooms:
            await self._model_ctx.client.post_message(
                project_id=project_id,
                chatroom_name=chatroom_name,
                content="Message processed, no further action needed at the moment!",
                source_version=trigger_version,
            )

    async def _post_error_notification(
        self,
        project_id: str,
        chatroom_name: str,
        agent_name: str,
        error: Exception,
    ) -> None:
        """Post an error notification to the chatroom when a task fails.

        Posts with is_ack=False: the error is the worker's terminal report,
        not a transient ack. Counting it as a batch response frees the
        typing chip immediately and lets the coordinator pivot in seconds
        instead of waiting for batch timeout.
        """
        if not self._model_ctx.client:
            return

        try:
            if isinstance(error, LLMRateLimitError):
                self.update_status(AgentStatus.RATE_LIMITED)
                reset_info = f" Resets at {error.resets_at_human}." if error.resets_at_human else ""
                content = f"I've hit a rate limit and cannot continue.{reset_info}"
            else:
                content = f"I encountered an error and couldn't complete this task: {type(error).__name__}"

            await self._model_ctx.client.post_message(
                project_id=project_id,
                chatroom_name=chatroom_name,
                content=content,
                is_ack=False,
            )
        except Exception:
            logger.error(f"Agent {agent_name}: failed to post error notification", exc_info=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Coordinator Callbacks (when Agent is project coordinator)
    # ─────────────────────────────────────────────────────────────────────────

    async def on_batch_complete(
        self,
        project_id: str,
        chatroom_name: str,
        message_id: str,
        responded_participants: list[str],
        trigger_version: int,
    ) -> None:
        """Handle batch completion when acting as coordinator.

        Only triggers if this agent is the project's coordinator
        (determined by project.coordinator_id).
        """
        from .project import Project
        project = Project.get(project_id, self._model_ctx)
        if not project or not self.is_coordinator_for(project):
            return  # Not the coordinator for this project

        logger.info(
            f"Agent {self.name} (as coordinator): Batch complete in {chatroom_name}, "
            f"responded: {responded_participants}"
        )
        await self._process_batch_results(
            project_id, chatroom_name, message_id, responded_participants, timed_out=[],
            trigger_version=trigger_version,
        )

    async def on_batch_timeout(
        self,
        project_id: str,
        chatroom_name: str,
        message_id: str,
        responded_participants: list[str],
        timed_out_participants: list[str],
        trigger_version: int,
    ) -> None:
        """Handle batch timeout when acting as coordinator.

        Only triggers if this agent is the project's coordinator.
        """
        from .project import Project
        project = Project.get(project_id, self._model_ctx)
        if not project or not self.is_coordinator_for(project):
            return  # Not the coordinator for this project

        logger.info(
            f"Agent {self.name} (as coordinator): Batch timeout in {chatroom_name}, "
            f"responded: {responded_participants}, timed out: {timed_out_participants}"
        )
        await self._process_batch_results(
            project_id, chatroom_name, message_id, responded_participants,
            timed_out=timed_out_participants,
            trigger_version=trigger_version,
        )

    async def on_first_user_request(
        self,
        project_id: str,
        chatroom_name: str,
        message: "ChatMessage",
        context_files: list[str],
        trigger_version: int,
    ) -> None:
        """Handle first user request when acting as coordinator.

        Uses coordinator setup prompt to analyze context, create plan,
        and delegate initial tasks.

        Only triggers if this agent is the project's coordinator.
        Falls back to normal on_message() for worker mode.
        """
        from .project import Project
        project = Project.get(project_id, self._model_ctx)
        if not project or not self.is_coordinator_for(project):
            # Not the coordinator - fall back to normal message handling
            await self.on_message(
                project_id=project_id,
                chatroom_name=chatroom_name,
                message=message,
                addressed_to_me=True,
                trigger_version=trigger_version,
            )
            return

        logger.info(
            f"Agent {self.name} (as coordinator): Handling first user request "
            f"in project {project_id[:8]}"
        )
        await self._invoke_coordinator_setup(
            project_id=project_id,
            chatroom_name=chatroom_name,
            message=message,
            context_files=context_files,
            trigger_version=trigger_version,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Coordinator Implementation Methods
    # ─────────────────────────────────────────────────────────────────────────

    async def _process_batch_results(
        self,
        project_id: str,
        chatroom_name: str,
        message_id: str,
        responded_participants: list[str],
        timed_out: list[str],
        trigger_version: int,
    ) -> None:
        """Process batch results as coordinator.

        Invokes Claude with coordinator context to decide next steps:
        - Delegate more work
        - Summarize results
        - Complete project
        """
        action_executor = self._model_ctx.action_executor
        if not self._model_ctx.cli or not action_executor:
            logger.warning(f"Agent {self.name}: Dependencies not configured for coordinator mode")
            return

        from .project import Project
        from .chatroom import Chatroom

        project = Project.get(project_id, self._model_ctx)
        chatroom = Chatroom.get(project_id, chatroom_name, self._model_ctx)

        # Compute project-aware paths from ModelContext
        data_dir = self._model_ctx.project_dir(project_id, project.name)
        sandbox_dir = self._model_ctx.sandbox_dir(project_id, project.name)
        log_dir = self._model_ctx.llm_log_dir(project_id, project.name)

        # Compute additional_dirs: data_dir (if different from sandbox) + knowledge_dirs
        additional_dirs: list[Path] = []
        if data_dir != sandbox_dir:
            additional_dirs.append(data_dir)
        additional_dirs.extend(self._model_ctx.knowledge_dirs)

        # Get recent messages for context
        messages = chatroom.get_messages()[-10:]
        context_str = "\n".join([
            f"{m.from_participant_name or m.from_participant_id}: {m.content}" for m in messages
        ])

        # Build batch status
        status_parts = [f"Batch complete in '{chatroom_name}'"]
        if responded_participants:
            status_parts.append(f"Responded: {', '.join(responded_participants)}")
        if timed_out:
            status_parts.append(f"Timed out: {', '.join(timed_out)}")
        batch_status = ". ".join(status_parts)

        # Include deliverable file listing from the chatroom
        files = chatroom.list_files() if chatroom else []
        files_section = ""
        if files:
            files_section = "\n\nDeliverable files in chatroom:\n"
            files_section += "\n".join(f"  - {f}" for f in files)
            files_section += "\nReview these files to assess whether acceptance criteria are met."

        # Create coordinator prompt builder for this task
        coordinator_builder = create_prompt_builder(
            OperationalMode.COORDINATOR,
            git_ignored_folder=project.git_ignored_folder if project.git_url else None,
        )
        assert isinstance(coordinator_builder, CoordinatorPromptBuilder)

        # Build synthetic message content with batch context
        batch_content = f"{batch_status}\n\nRecent conversation:\n{context_str}{files_section}"
        batch_content += "\n\nIMPORTANT: Update PLAN.md with your assessment (PASS/FAIL per acceptance criterion) BEFORE deciding next steps."

        # Build prompt - agents are referenced via AGENTS.md file
        prompt = coordinator_builder.build_prompt(
            name=self.name,
            description=self.description,
            project_id=project_id,
            chatroom_name=chatroom_name,
            from_participant_name="System",
            message_content=batch_content,
            data_dir=data_dir,
            project_name=project.name,
            knowledge_dirs=self._model_ctx.knowledge_dirs,
        )

        await self._emit_acknowledgment(project_id, chatroom_name)

        retry_delay = _INITIAL_RETRY_DELAY
        for attempt in range(_MAX_RETRIES + 1):
            try:
                action_block, usage = await _invoke_with_registry(
                    self._model_ctx,
                    project_id,
                    chatroom_name,
                    prompt,
                    sandbox_dir,
                    log_dir,
                    additional_dirs,
                    action_schema=COORDINATOR_ACTION_SCHEMA,
                )
                break
            except LLMRateLimitError as e:
                logger.warning(
                    f"Agent {self.name} (coordinator): rate limited "
                    f"(type={e.rate_limit_type}, resets={e.resets_at_human})"
                )
                await self._post_error_notification(
                    project_id, chatroom_name, self.name, e
                )
                raise
            except LLMInvocationError as e:
                if attempt < _MAX_RETRIES and _is_transient_error(e):
                    logger.warning(
                        f"Agent {self.name} (coordinator): transient failure "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES + 1}), retrying in {retry_delay}s: {e}"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    await self._post_error_notification(
                        project_id, chatroom_name, self.name, e
                    )
                    raise

        logger.info(
            f"Agent {self.name} (coordinator): Batch processing complete "
            f"(cost=${usage.cost_usd:.4f})"
        )

        action_block.source_version = trigger_version
        replied_chatrooms = await action_executor.process(
            action_block=action_block,
            project_id=project_id,
            sandbox_dir=sandbox_dir,
        )

        # If the coordinator moved on (created next milestone room, updated
        # PLAN.md, etc.) without replying in this room, post a closure so
        # the self-batch pending work clears and the typing chip goes away.
        if chatroom_name not in replied_chatrooms:
            await self._emit_no_action_message(project_id, chatroom_name, trigger_version)

    async def _invoke_coordinator_setup(
        self,
        project_id: str,
        chatroom_name: str,
        message: "ChatMessage",
        context_files: list[str],
        trigger_version: int,
    ) -> None:
        """Invoke Claude with coordinator setup prompt.

        Uses CoordinatorPromptBuilder.build_setup_prompt() for first request handling.
        """
        action_executor = self._model_ctx.action_executor
        if not self._model_ctx.cli or not action_executor:
            raise ValueError(f"Agent {self.name}: Dependencies not configured for coordinator setup")

        from .project import Project

        project = Project.get(project_id, self._model_ctx)

        # Compute project-aware paths from ModelContext
        data_dir = self._model_ctx.project_dir(project_id, project.name)
        sandbox_dir = self._model_ctx.sandbox_dir(project_id, project.name)
        log_dir = self._model_ctx.llm_log_dir(project_id, project.name)

        # Compute additional_dirs: data_dir (if different from sandbox) + knowledge_dirs
        additional_dirs: list[Path] = []
        if data_dir != sandbox_dir:
            additional_dirs.append(data_dir)
        additional_dirs.extend(self._model_ctx.knowledge_dirs)

        # Create coordinator prompt builder for this task
        coordinator_builder = create_prompt_builder(
            OperationalMode.COORDINATOR,
            git_ignored_folder=project.git_ignored_folder if project.git_url else None,
        )
        assert isinstance(coordinator_builder, CoordinatorPromptBuilder)

        await self._emit_acknowledgment(project_id, chatroom_name)

        # Build setup prompt - agents are referenced via AGENTS.md file
        prompt = coordinator_builder.build_setup_prompt(
            name=self.name,
            description=self.description,
            project_id=project_id,
            chatroom_name=chatroom_name,
            message_content=message.content,
            data_dir=data_dir,
            context_files=context_files,
            project_name=project.name,
            knowledge_dirs=self._model_ctx.knowledge_dirs,
        )

        retry_delay = _INITIAL_RETRY_DELAY
        for attempt in range(_MAX_RETRIES + 1):
            try:
                action_block, usage = await _invoke_with_registry(
                    self._model_ctx,
                    project_id,
                    chatroom_name,
                    prompt,
                    sandbox_dir,
                    log_dir,
                    additional_dirs,
                    action_schema=COORDINATOR_ACTION_SCHEMA,
                )
                break
            except LLMRateLimitError as e:
                logger.warning(
                    f"Agent {self.name} (coordinator): rate limited "
                    f"(type={e.rate_limit_type}, resets={e.resets_at_human})"
                )
                await self._post_error_notification(
                    project_id, chatroom_name, self.name, e
                )
                raise
            except LLMInvocationError as e:
                if attempt < _MAX_RETRIES and _is_transient_error(e):
                    logger.warning(
                        f"Agent {self.name} (coordinator): transient failure "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES + 1}), retrying in {retry_delay}s: {e}"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    await self._post_error_notification(
                        project_id, chatroom_name, self.name, e
                    )
                    raise

        logger.info(
            f"Agent {self.name} (coordinator): Setup invocation complete "
            f"(cost=${usage.cost_usd:.4f})"
        )

        action_block.source_version = trigger_version
        replied_chatrooms = await action_executor.process(
            action_block=action_block,
            project_id=project_id,
            sandbox_dir=sandbox_dir,
        )

        # Close out the triggering room if the LLM didn't reply there —
        # keeps the self-batch pending work from getting stuck and the
        # typing indicator from pinning on the coordinator until timeout.
        if chatroom_name not in replied_chatrooms:
            await self._emit_no_action_message(project_id, chatroom_name, trigger_version)

    # ─────────────────────────────────────────────────────────────────────────
    # Coordinator Live-Message Methods (when Agent is project coordinator)
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_user_request(
        self,
        project_id: str,
        chatroom_name: str,
        message: "ChatMessage",
        trigger_version: int,
    ) -> None:
        """Handle a live message in user-communication (non-first-user-request)."""
        await self._invoke_coordinator_response(project_id, chatroom_name, message, trigger_version)

    async def _coordinate(
        self,
        project_id: str,
        chatroom_name: str,
        message: "ChatMessage",
        trigger_version: int,
    ) -> None:
        """Handle coordination messages in work chatrooms (addressed to coordinator)."""
        await self._invoke_coordinator_response(project_id, chatroom_name, message, trigger_version)

    async def _invoke_coordinator_response(
        self,
        project_id: str,
        chatroom_name: str,
        message: "ChatMessage",
        trigger_version: int,
    ) -> None:
        """Invoke Claude with the coordinator prompt for a live message.

        Used for both user-communication responses and in-room coordination
        after first-user-request / batch-complete paths have been handled
        elsewhere.
        """
        name = self.name

        if not self._model_ctx.cli:
            raise RuntimeError(f"Agent {name} (coordinator): CLI not configured")

        action_executor = self._model_ctx.action_executor
        if not action_executor:
            raise RuntimeError(f"Agent {name} (coordinator): Action executor not configured")

        await self._emit_acknowledgment(project_id, chatroom_name)

        from .project import Project
        project = Project.get(project_id, self._model_ctx)

        data_dir = self._model_ctx.project_dir(project_id, project.name)
        sandbox_dir = self._model_ctx.sandbox_dir(project_id, project.name)
        log_dir = self._model_ctx.llm_log_dir(project_id, project.name)

        additional_dirs: list[Path] = []
        if data_dir != sandbox_dir:
            additional_dirs.append(data_dir)
        additional_dirs.extend(self._model_ctx.knowledge_dirs)

        prompt_builder = create_prompt_builder(
            OperationalMode.COORDINATOR,
            git_ignored_folder=project.git_ignored_folder if project.git_url else None,
        )
        assert isinstance(prompt_builder, CoordinatorPromptBuilder)

        prompt = prompt_builder.build_prompt(
            name=self.name,
            description=self.description,
            project_id=project_id,
            chatroom_name=chatroom_name,
            from_participant_name=message.from_participant_name or message.from_participant_id,
            message_content=message.content,
            data_dir=data_dir,
            project_name=project.name,
            knowledge_dirs=self._model_ctx.knowledge_dirs,
        )

        retry_delay = _INITIAL_RETRY_DELAY
        for attempt in range(_MAX_RETRIES + 1):
            try:
                action_block, usage = await _invoke_with_registry(
                    self._model_ctx,
                    project_id,
                    chatroom_name,
                    prompt,
                    sandbox_dir,
                    log_dir,
                    additional_dirs,
                    action_schema=COORDINATOR_ACTION_SCHEMA,
                )
                break
            except LLMRateLimitError as e:
                logger.warning(
                    f"Agent {name} (coordinator): rate limited "
                    f"(type={e.rate_limit_type}, resets={e.resets_at_human})"
                )
                await self._post_error_notification(project_id, chatroom_name, name, e)
                raise
            except LLMInvocationError as e:
                if attempt < _MAX_RETRIES and _is_transient_error(e):
                    logger.warning(
                        f"Agent {name} (coordinator): transient failure "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES + 1}), retrying in {retry_delay}s: {e}"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    await self._post_error_notification(project_id, chatroom_name, name, e)
                    raise

        logger.info(
            f"Agent {name} (coordinator): Claude invocation complete "
            f"(cost=${usage.cost_usd:.4f}, tokens in={usage.input_tokens} out={usage.output_tokens})"
        )

        action_block.source_version = trigger_version
        replied_chatrooms = await action_executor.process(
            action_block=action_block,
            project_id=project_id,
            sandbox_dir=sandbox_dir,
        )

        if chatroom_name not in replied_chatrooms:
            await self._emit_no_action_message(project_id, chatroom_name, trigger_version)

    async def _emit_no_action_message(
        self,
        project_id: str,
        chatroom_name: str,
        source_version: int,
    ) -> None:
        """Post a closure reply when the LLM produced no reply action for this room.

        Posted non-ack so the server's record_response marks this participant
        as responded, clears PendingWork, and fires BATCH_COMPLETE.
        """
        if not self._model_ctx.client:
            raise RuntimeError(f"Agent {self.name}: Client not configured, cannot emit no-action message")

        await self._model_ctx.client.post_message(
            project_id=project_id,
            chatroom_name=chatroom_name,
            content="Message processed, no further action needed at the moment!",
            source_version=source_version,
        )
