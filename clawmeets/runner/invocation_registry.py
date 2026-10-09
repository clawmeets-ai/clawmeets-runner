# SPDX-License-Identifier: MIT
"""
clawmeets/runner/invocation_registry.py

Per-runner registry of in-flight LLM invocation tasks, keyed by
(project_id, chatroom_name).

The runner wraps each `cli.invoke(...)` call in an `asyncio.Task`, registers
it here, and unregisters in a `finally`. The reactive control loop calls
`cancel(...)` from the CANCEL_LLM dispatch path, which propagates as
`asyncio.CancelledError` into the provider; the provider's `invoke()` is
expected to terminate its subprocess on cancel.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..api.actions import ActionBlock
    from ..llm.base import LLMProvider, LLMUsage
    from ..models.context import ModelContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InFlightTurn:
    """One live LLM turn, as reported to the server in the WS auth frame.

    ``trigger_version`` is the changelog entry the turn is answering and
    ``started_at`` when it began, so a restarted server can rebuild the
    turn's pending batch with the original timeout clock
    (``server/in_flight_restore.py``).
    """
    project_id: str
    chatroom_name: str
    trigger_version: int
    started_at: datetime

    def to_wire(self) -> dict:
        return {
            "project_id": self.project_id,
            "chatroom_name": self.chatroom_name,
            "trigger_version": self.trigger_version,
            "started_at": self.started_at.isoformat(),
        }


class InvocationRegistry:
    """In-memory map of active LLM tasks for one runner."""

    def __init__(self) -> None:
        self._tasks: dict[tuple[str, str], asyncio.Task] = {}
        # Same keys as _tasks; None when the caller supplied no trigger_version.
        self._turns: dict[tuple[str, str], InFlightTurn | None] = {}

    def register(
        self,
        project_id: str,
        chatroom_name: str,
        task: asyncio.Task,
        trigger_version: int | None = None,
    ) -> None:
        key = (project_id, chatroom_name)
        existing = self._tasks.get(key)
        if existing is not None and not existing.done():
            logger.warning(
                "InvocationRegistry: replacing in-flight task for "
                f"project={project_id[:8]} room={chatroom_name} "
                "(prior invocation still running — concurrent dispatch?)"
            )
        self._tasks[key] = task
        self._turns[key] = (
            InFlightTurn(project_id, chatroom_name, trigger_version, datetime.now(UTC))
            if trigger_version is not None else None
        )

    def unregister(self, project_id: str, chatroom_name: str) -> None:
        self._tasks.pop((project_id, chatroom_name), None)
        self._turns.pop((project_id, chatroom_name), None)

    def in_flight(self) -> list[InFlightTurn]:
        """Snapshot of live (not done) turns that carry a trigger_version.

        Read synchronously on the event loop while the WS auth frame is
        built, so no lock is needed.
        """
        return [
            turn for key, turn in self._turns.items()
            if turn is not None
            and (task := self._tasks.get(key)) is not None
            and not task.done()
        ]

    def cancel(self, project_id: str, chatroom_name: str) -> bool:
        """Cancel the task for the given (project, chatroom). Returns True if a
        live task was cancelled, False if nothing was registered.
        """
        task = self._tasks.get((project_id, chatroom_name))
        if task is None or task.done():
            return False
        task.cancel()
        return True


async def invoke_with_registry(
    model_ctx: "ModelContext",
    project_id: str,
    chatroom_name: str,
    prompt: str,
    working_dir: Path,
    log_dir: Path,
    additional_dirs: list[Path],
    action_schema: dict,
    trigger_version: int,
    role: "str | None" = None,
    correction: "str | None" = None,
    override_cli: "LLMProvider | None" = None,
) -> "tuple[ActionBlock, LLMUsage]":
    """Run cli.invoke and register the task so it can be cancelled.

    If no InvocationRegistry is attached (e.g. unit tests), the call still
    runs — it just isn't cancellable from outside.

    ``role`` ("worker" | "coordinator" | "assistant") selects which
    system-skill-hub subset is prepended to skill_source_dirs. None
    means no system layer (back-compat for callsites that haven't been
    threaded yet); production callsites always pass it.

    ``correction`` is the semantic-validation feedback appended to ``prompt``
    on a retry (see ``ActionValidator`` / ``Agent._invoke_validated``). Default
    ``None`` keeps every existing caller byte-for-byte identical.

    ``override_cli`` is a one-turn provider built from a per-request
    ``model_config_name`` override (spec #3). When ``None`` (the common case)
    the invocation uses the agent's default ``model_ctx.cli``; the override
    never mutates ``model_ctx.cli``, so it applies to this turn only.
    """
    if correction is not None:
        prompt = f"{prompt}\n\n{correction}"
    cli = override_cli or model_ctx.cli
    coro = cli.invoke(
        prompt,
        working_dir=working_dir,
        log_dir=log_dir,
        additional_dirs=additional_dirs,
        notification_center=model_ctx.notification_center,
        action_schema=action_schema,
        trigger_version=trigger_version,
        mcp_config_dir=model_ctx.mcp_dist_dir,
        skill_source_dirs=model_ctx.skill_source_dirs(role=role),
        memory_dir=model_ctx.memory_dir,
    )
    registry = model_ctx.invocation_registry
    if registry is None:
        return await coro

    task = asyncio.create_task(coro)
    registry.register(project_id, chatroom_name, task, trigger_version=trigger_version)
    try:
        return await task
    finally:
        registry.unregister(project_id, chatroom_name)
