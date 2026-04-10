# SPDX-License-Identifier: MIT
"""
clawmeets/models/assistant.py
Private assistant agent implementation using composition.

Assistants are coordinators linked to specific users.
They orchestrate work by delegating to worker agents.

All state is read from the filesystem (card.json) - no in-memory state.
Extends PersistableParticipant for Active Record persistence.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional

from .chatroom import Chatroom
from .chat_message import ChatMessage
from .participant import ParticipantRole, OperationalMode
from .persistable import PersistableParticipant
from ..api.actions import ActionBlock
from ..api.responses import AgentStatus
from ..llm.claude_cli import ClaudeInvocationError, ClaudeRateLimitError, ClaudeTimeoutError
from ..llm.prompt_builder import CoordinatorPromptBuilder, create_prompt_builder

if TYPE_CHECKING:
    from ..api.client import ClawMeetsClient
    from .context import ModelContext

logger = logging.getLogger(__name__)

# Retry configuration for transient Claude CLI failures
_MAX_RETRIES = 2  # Total attempts: 3 (1 original + 2 retries)
_INITIAL_RETRY_DELAY = 30  # seconds
_TRANSIENT_INDICATORS = ("overloaded", "rate_limit", "529", "503", "too many requests")


def _is_transient_error(error: ClaudeInvocationError) -> bool:
    """Check if a ClaudeInvocationError is likely transient (retryable)."""
    if isinstance(error, ClaudeRateLimitError):
        return False  # Rate limits should not be retried with short backoff
    if isinstance(error, ClaudeTimeoutError):
        return True
    msg = str(error).lower()
    return any(indicator in msg for indicator in _TRANSIENT_INDICATORS)


class Assistant(PersistableParticipant):
    """
    Private assistant agent - coordinator using composition.
    Not discoverable in registry, linked to a specific user.

    Assistants handle:
    - User communication via the user-communication chatroom
    - Work delegation via expects_response_from
    - Batch completion/timeout handling

    All state is read from the filesystem (card.json) on each property access.
    This ensures the model always reflects the current state on disk.

    Extends PersistableParticipant for Active Record methods:
        - Assistant.get(), Assistant.get_by_name(), Assistant.list_all()
        - Assistant.register(), Assistant.verify_token()
        - assistant.save(), assistant.update_status(), assistant.heartbeat()
        - assistant.to_response()

    Composition:
        - ClaudeCLI: Direct Claude CLI invocation
        - ActionBlockExecutor: Processes Claude output and executes actions via HTTP
        - Prompt builders created on-demand via create_prompt_builder()
    """

    # Active Record: directory subdirectory for assistants
    _role_subdir: ClassVar[str] = "assistants"

    def __init__(
        self,
        id: str,
        model_ctx: "ModelContext",
    ) -> None:
        """Initialize an Assistant.

        For Active Record operations (server-side), only id and model_ctx are needed.
        For task execution (runner-side), model_ctx should have cli and client configured.

        Runtime dependencies are accessed via model_ctx:
        - model_ctx.cli: Claude CLI for LLM invocation
        - model_ctx.knowledge_dirs: Additional directories for Claude access
        - model_ctx.client: ClawMeetsClient for HTTP operations
        - model_ctx.action_executor: ActionBlockExecutor for action execution

        Args:
            id: The assistant's unique identifier
            model_ctx: ModelContext for filesystem access (may include cli/knowledge_dirs/client)
        """
        super().__init__(id, model_ctx)

    @classmethod
    async def sync_from_server(
        cls,
        ctx: "ModelContext",
    ) -> int:
        """Sync assistant card from server to local filesystem.

        Fetches the caller's own assistant from server and persists
        card.json locally under assistants/{name}-{id}/.

        Args:
            ctx: ModelContext for filesystem access (must have client configured)

        Returns:
            Number of assistants synced

        Raises:
            ValueError: If ctx.client is not configured
        """
        if ctx.client is None:
            raise ValueError("ModelContext.client must be configured for sync_from_server")
        assistants = await ctx.client.list_assistants()
        synced_count = 0
        for assistant_data in assistants:
            assistant_id = assistant_data["id"]
            assistant = cls.get_or_create(assistant_id, ctx)
            assistant.update_card(**assistant_data)
            synced_count += 1

        logger.info(f"Synced {synced_count} assistants from server")
        return synced_count

    async def _emit_acknowledgment(
        self,
        project_id: str,
        chatroom_name: str,
    ) -> None:
        """Emit an acknowledgment message before processing."""
        if not self._model_ctx.client:
            raise RuntimeError(f"Assistant {self.name}: Client not configured, cannot emit acknowledgment")

        await self._model_ctx.client.post_message(
            project_id=project_id,
            chatroom_name=chatroom_name,
            content="Message received, processing...",
            is_ack=True,
        )

    async def _emit_no_action_message(
        self,
        project_id: str,
        chatroom_name: str,
    ) -> None:
        """Emit a follow-up message when no reply action was needed for this chatroom."""
        if not self._model_ctx.client:
            raise RuntimeError(f"Assistant {self.name}: Client not configured, cannot emit no-action message")

        await self._model_ctx.client.post_message(
            project_id=project_id,
            chatroom_name=chatroom_name,
            content="Message processed, no further action needed at the moment!",
            is_ack=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Role Property (from Participant ABC)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def role(self) -> ParticipantRole:
        """Return ASSISTANT role."""
        return ParticipantRole.ASSISTANT

    # ─────────────────────────────────────────────────────────────────────────
    # Assistant-Specific Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def linked_user_id(self) -> Optional[str]:
        """The user this assistant is linked to (from filesystem)."""
        return self._load_card().get("linked_user_id")

    @property
    def is_discoverable(self) -> bool:
        """Assistants are never discoverable - they're private."""
        return False

    @property
    def description(self) -> str:
        """Get assistant description from filesystem (default: 'Personal assistant')."""
        return self._load_card().get("description", "Personal assistant")

    async def on_message(
        self,
        project_id: str,
        chatroom_name: str,
        message: ChatMessage,
        addressed_to_me: bool,
    ) -> None:
        """
        Assistants handle messages based on chatroom type:
        - user-communication: respond to user requests
        - work chatrooms: coordinate worker responses
        """
        chatroom = Chatroom.get(project_id, chatroom_name, self._model_ctx)
        if chatroom is None:
            logger.warning(
                f"Assistant {self.name}: Chatroom '{chatroom_name}' not found for "
                f"project {project_id[:8]}. Message skipped."
            )
            return
        if chatroom.is_user_communication_room:
            await self._handle_user_request(project_id, chatroom_name, message)
        elif addressed_to_me:
            await self._coordinate(project_id, chatroom_name, message)

    async def on_batch_complete(
        self,
        project_id: str,
        chatroom_name: str,
        message_id: str,
        responded_participants: list[str],
    ) -> None:
        """
        Assistant receives batch complete - all workers done.
        Decide next steps: more work, summarize, or complete project.
        """
        await self._process_batch_results(
            project_id, chatroom_name, message_id, responded_participants, timed_out=[]
        )

    async def on_batch_timeout(
        self,
        project_id: str,
        chatroom_name: str,
        message_id: str,
        responded_participants: list[str],
        timed_out_participants: list[str],
    ) -> None:
        """Handle batch timeout - some workers didn't respond."""
        await self._process_batch_results(
            project_id, chatroom_name, message_id, responded_participants, timed_out=timed_out_participants
        )

    async def on_first_user_request(
        self,
        project_id: str,
        chatroom_name: str,
        message: ChatMessage,
        context_files: list[str],
    ) -> None:
        """Handle first user request with special setup prompt.

        This triggers a setup-focused prompt that:
        - Analyzes context files and user request
        - Refines AGENTS.md and PLAN.md
        - Immediately delegates to workers

        Args:
            project_id: The project ID
            chatroom_name: The user-communication chatroom name
            message: The first user message
            context_files: List of files in shared-context (excluding AGENTS.md, PLAN.md)
        """
        logger.info(
            f"Assistant {self.name}: Handling first user request in project {project_id[:8]}"
        )
        await self._invoke_claude_setup(
            project_id=project_id,
            chatroom_name=chatroom_name,
            message=message,
            context_files=context_files,
        )

    # ---------------------------------------------------------
    # Implementation Methods
    # ---------------------------------------------------------

    async def _handle_user_request(
        self,
        project_id: str,
        chatroom_name: str,
        message: ChatMessage,
    ) -> None:
        """Handle a request from the linked user using Claude."""
        await self._invoke_claude(project_id, chatroom_name, message)

    async def _invoke_claude_setup(
        self,
        project_id: str,
        chatroom_name: str,
        message: ChatMessage,
        context_files: list[str],
    ) -> None:
        """Invoke Claude with setup-focused prompt for first request.

        Uses a special setup prompt that emphasizes:
        - Planning and breaking down the request
        - Learning from context files
        - Delegating sub-tasks to workers

        Requires cli and action_executor to be configured.
        Prompt builder is created on-demand.

        Args:
            project_id: The project ID
            chatroom_name: The chatroom name
            message: The first user message
            context_files: List of context files in shared-context
        """
        assistant_name = self.name

        if not self._model_ctx.cli:
            raise RuntimeError(f"Assistant {assistant_name}: CLI not configured, cannot invoke Claude")

        action_executor = self._model_ctx.action_executor
        if not action_executor:
            raise RuntimeError(f"Assistant {assistant_name}: Action executor not configured, cannot process actions")

        # Emit acknowledgment before processing
        await self._emit_acknowledgment(project_id, chatroom_name)

        # Get project
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

        # Create coordinator prompt builder on-demand
        prompt_builder = create_prompt_builder(
            OperationalMode.COORDINATOR,
            git_ignored_folder=self._model_ctx.git_ignored_folder if self._model_ctx.git_url else None,
        )
        assert isinstance(prompt_builder, CoordinatorPromptBuilder)

        # Build setup-specific prompt - agents are referenced via AGENTS.md file
        prompt = prompt_builder.build_setup_prompt(
            name=self.name,
            description=self.description,
            project_id=project_id,
            chatroom_name=chatroom_name,
            message_content=message.content,
            data_dir=data_dir,
            context_files=context_files,
            project_name=project.name,
        )

        # Execute using ClaudeCLI with retry for transient failures
        retry_delay = _INITIAL_RETRY_DELAY
        for attempt in range(_MAX_RETRIES + 1):
            try:
                action_block, usage = await self._model_ctx.cli.invoke(
                    prompt,
                    working_dir=sandbox_dir,
                    log_dir=log_dir,
                    additional_dirs=additional_dirs,
                    notification_center=self._model_ctx.notification_center,
                )
                break
            except ClaudeRateLimitError as e:
                logger.warning(
                    f"Assistant {assistant_name}: rate limited "
                    f"(type={e.rate_limit_type}, resets={e.resets_at_human})"
                )
                await self._post_error_notification(
                    project_id, chatroom_name, assistant_name, e
                )
                raise
            except ClaudeInvocationError as e:
                if attempt < _MAX_RETRIES and _is_transient_error(e):
                    logger.warning(
                        f"Assistant {assistant_name}: transient failure "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES + 1}), retrying in {retry_delay}s: {e}"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    await self._post_error_notification(
                        project_id, chatroom_name, assistant_name, e
                    )
                    raise

        logger.info(
            f"Assistant {assistant_name}: Setup invocation complete "
            f"(cost=${usage.cost_usd:.4f}, tokens in={usage.input_tokens} out={usage.output_tokens})"
        )

        # Process using ActionBlockExecutor
        replied_chatrooms = await action_executor.process(
            action_block=action_block,
            project_id=project_id,
            sandbox_dir=sandbox_dir,
        )

        # If triggering chatroom didn't receive a reply, emit a closure message
        if chatroom_name not in replied_chatrooms:
            await self._emit_no_action_message(project_id, chatroom_name)

    async def _coordinate(
        self,
        project_id: str,
        chatroom_name: str,
        message: ChatMessage,
    ) -> None:
        """Handle coordination messages in work chatrooms using Claude."""
        await self._invoke_claude(project_id, chatroom_name, message)

    async def _process_batch_results(
        self,
        project_id: str,
        chatroom_name: str,
        message_id: str,
        responded_participants: list[str],
        timed_out: list[str],
    ) -> None:
        """Process batch results - invoke Claude to decide next steps."""
        # Create synthetic message about batch with PLAN.md reminder
        batch_type = "TIMEOUT" if timed_out else "COMPLETE"
        content = f"[BATCH {batch_type}] Participants {responded_participants} have responded in chatroom '{chatroom_name}'."
        if timed_out:
            content += f" Timed out: {timed_out}."

        # Include deliverable file listing from the chatroom
        chatroom = Chatroom.get(project_id, chatroom_name, self._model_ctx)
        files = chatroom.list_files() if chatroom else []
        if files:
            content += "\n\nDELIVERABLE FILES in chatroom:\n"
            for f in files:
                content += f"  - {f}\n"
            content += "Review these files to assess whether acceptance criteria are met."

        # Add next steps guidance
        content += """

NEXT STEPS:
1. Review agent responses and deliverable files in the chatroom
2. Check PLAN.md - assess each acceptance criterion as PASS/FAIL
3. Update PLAN.md: mark milestone status, add to Review Log and Learnings
4. Decide: continue to next milestone, request revision, pivot approach, or finalize

IMPORTANT: Update PLAN.md with your assessment BEFORE deciding next steps.
If the current approach isn't working, document why in Learnings and try a different strategy."""

        synthetic = ChatMessage(
            id=message_id,
            ts=datetime.utcnow(),
            from_participant_id="system",
            from_participant_name="system",
            content=content,
        )

        await self._invoke_claude(project_id, chatroom_name, synthetic)

    async def _invoke_claude(
        self,
        project_id: str,
        chatroom_name: str,
        message: ChatMessage,
    ) -> None:
        """
        Invoke Claude with coordinator-specific prompt.

        Uses composition objects for prompt building, execution, and action execution.
        Requires cli and client to be configured via model_ctx.
        Prompt builder is created on-demand.
        """
        assistant_name = self.name  # Use property to get current name

        if not self._model_ctx.cli:
            raise RuntimeError(f"Assistant {assistant_name}: CLI not configured, cannot invoke Claude")

        action_executor = self._model_ctx.action_executor
        if not action_executor:
            raise RuntimeError(f"Assistant {assistant_name}: Action executor not configured, cannot process actions")

        # Emit acknowledgment before processing
        await self._emit_acknowledgment(project_id, chatroom_name)

        # Get project
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

        # Create coordinator prompt builder on-demand
        prompt_builder = create_prompt_builder(
            OperationalMode.COORDINATOR,
            git_ignored_folder=self._model_ctx.git_ignored_folder if self._model_ctx.git_url else None,
        )
        assert isinstance(prompt_builder, CoordinatorPromptBuilder)

        # Build prompt - agents are referenced via AGENTS.md file
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

        # Execute using ClaudeCLI with retry for transient failures
        retry_delay = _INITIAL_RETRY_DELAY
        for attempt in range(_MAX_RETRIES + 1):
            try:
                action_block, usage = await self._model_ctx.cli.invoke(
                    prompt,
                    working_dir=sandbox_dir,
                    log_dir=log_dir,
                    additional_dirs=additional_dirs,
                    notification_center=self._model_ctx.notification_center,
                )
                break
            except ClaudeRateLimitError as e:
                logger.warning(
                    f"Assistant {assistant_name}: rate limited "
                    f"(type={e.rate_limit_type}, resets={e.resets_at_human})"
                )
                await self._post_error_notification(
                    project_id, chatroom_name, assistant_name, e
                )
                raise
            except ClaudeInvocationError as e:
                if attempt < _MAX_RETRIES and _is_transient_error(e):
                    logger.warning(
                        f"Assistant {assistant_name}: transient failure "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES + 1}), retrying in {retry_delay}s: {e}"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    await self._post_error_notification(
                        project_id, chatroom_name, assistant_name, e
                    )
                    raise

        logger.info(
            f"Assistant {assistant_name}: Claude invocation complete "
            f"(cost=${usage.cost_usd:.4f}, tokens in={usage.input_tokens} out={usage.output_tokens})"
        )

        # Process using ActionBlockExecutor - executes actions via HTTP
        replied_chatrooms = await action_executor.process(
            action_block=action_block,
            project_id=project_id,
            sandbox_dir=sandbox_dir,
        )

        # If triggering chatroom didn't receive a reply, emit a closure message
        if chatroom_name not in replied_chatrooms:
            await self._emit_no_action_message(project_id, chatroom_name)

    async def _post_error_notification(
        self,
        project_id: str,
        chatroom_name: str,
        assistant_name: str,
        error: Exception,
    ) -> None:
        """Post an error notification to the chatroom when a task fails.

        Posts with is_ack=True so the message appears in chat but does not
        trigger batch completion (the batch will timeout instead, correctly
        signaling that the assistant didn't complete).
        """
        if not self._model_ctx.client:
            return

        try:
            if isinstance(error, ClaudeRateLimitError):
                self.update_status(AgentStatus.RATE_LIMITED)
                reset_info = f" Resets at {error.resets_at_human}." if error.resets_at_human else ""
                content = f"I've hit a rate limit and cannot continue.{reset_info}"
            else:
                content = f"I encountered an error and couldn't complete this task: {type(error).__name__}"

            await self._model_ctx.client.post_message(
                project_id=project_id,
                chatroom_name=chatroom_name,
                content=content,
                is_ack=True,
            )
        except Exception:
            logger.error(f"Assistant {assistant_name}: failed to post error notification", exc_info=True)

    def to_dict(self) -> dict:
        """Serialize to dictionary (reads from filesystem).

        Override to include linked_user_id.
        """
        base = super().to_dict()
        base["linked_user_id"] = self.linked_user_id
        return base
