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
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..api.actions import ActionBlock
    from ..llm.base import LLMUsage
    from ..models.context import ModelContext

logger = logging.getLogger(__name__)


class InvocationRegistry:
    """In-memory map of active LLM tasks for one runner."""

    def __init__(self) -> None:
        self._tasks: dict[tuple[str, str], asyncio.Task] = {}

    def register(self, project_id: str, chatroom_name: str, task: asyncio.Task) -> None:
        key = (project_id, chatroom_name)
        existing = self._tasks.get(key)
        if existing is not None and not existing.done():
            logger.warning(
                "InvocationRegistry: replacing in-flight task for "
                f"project={project_id[:8]} room={chatroom_name} "
                "(prior invocation still running — concurrent dispatch?)"
            )
        self._tasks[key] = task

    def unregister(self, project_id: str, chatroom_name: str) -> None:
        self._tasks.pop((project_id, chatroom_name), None)

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
) -> "tuple[ActionBlock, LLMUsage]":
    """Run cli.invoke and register the task so it can be cancelled.

    If no InvocationRegistry is attached (e.g. unit tests), the call still
    runs — it just isn't cancellable from outside.
    """
    coro = model_ctx.cli.invoke(
        prompt,
        working_dir=working_dir,
        log_dir=log_dir,
        additional_dirs=additional_dirs,
        notification_center=model_ctx.notification_center,
        action_schema=action_schema,
    )
    registry = model_ctx.invocation_registry
    if registry is None:
        return await coro

    task = asyncio.create_task(coro)
    registry.register(project_id, chatroom_name, task)
    try:
        return await task
    finally:
        registry.unregister(project_id, chatroom_name)
