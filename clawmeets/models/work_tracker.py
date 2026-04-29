# SPDX-License-Identifier: MIT
"""
clawmeets/models/work_tracker.py

Work tracking for coordinator batch completion detection.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Awaitable, Callable, Optional

from pydantic import BaseModel, Field

OnChangeCallback = Callable[[str, str, list[str]], Awaitable[None]]


class PendingWork(BaseModel):
    """Tracks a coordinator's work dispatch for batch completion detection.

    When a coordinator sends a message with expects_response_from, the server
    creates a PendingWork record. As participants respond, the server updates
    responded_participants. When all expected participants have responded, the server
    sends BATCH_COMPLETE to the coordinator.
    """
    model_config = {"frozen": True}

    message_id: str                    # Coordinator's message that initiated this wave
    message_version: int               # Changelog version of the initiating message (source for workers' replies and BATCH_COMPLETE)
    chatroom_name: str
    project_id: str
    project_name: str                  # Needed for runloop lookup on timeout
    coordinator_id: str
    expected_participants: list[str]         # Participant IDs expected to respond
    responded_participants: list[str] = Field(default_factory=list)  # Participant IDs who have responded
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    timeout_seconds: int = 600         # Default 10 minutes

    @property
    def is_complete(self) -> bool:
        """Check if all expected participants have responded."""
        return set(self.expected_participants) == set(self.responded_participants)

    @property
    def timed_out_participants(self) -> list[str]:
        """Return participants who haven't responded yet."""
        return [a for a in self.expected_participants if a not in self.responded_participants]


class WorkTracker:
    """
    In-memory work tracker for coordinator batch completion detection.
    Lives only for the lifetime of the server process.

    Pending work is keyed by ``(project_id, chatroom_name)`` — chatroom names
    aren't globally unique (e.g. two projects each have a ``user-communication``
    room), so scoping to project is required for correctness.
    """

    def __init__(self, on_change: Optional[OnChangeCallback] = None) -> None:
        # (project_id, chatroom_name) -> PendingWork
        self._pending: dict[tuple[str, str], PendingWork] = {}
        # agent_id -> list of message IDs currently being processed
        self._agent_processing: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()
        # Fired after every PendingWork transition (create / record_response /
        # clear / clear_project) with the updated outstanding-participant list.
        # Signature: (project_id, chatroom_name, active_participants). Kept
        # opaque so WorkTracker stays free of ws_hub/server imports.
        self._on_change = on_change

    async def _emit_change(
        self, project_id: str, chatroom_name: str, active_participants: list[str]
    ) -> None:
        if self._on_change is None:
            return
        await self._on_change(project_id, chatroom_name, active_participants)

    async def create_pending_work(
        self,
        message_id: str,
        message_version: int,
        project_id: str,
        project_name: str,
        chatroom_name: str,
        coordinator_id: str,
        expected_participants: list[str],
        timeout_seconds: int = 600,
    ) -> PendingWork:
        key = (project_id, chatroom_name)
        async with self._lock:
            if key in self._pending:
                raise ValueError(
                    f"Pending work already exists for ({project_id!r}, {chatroom_name!r})"
                )
            work = PendingWork(
                message_id=message_id,
                message_version=message_version,
                project_id=project_id,
                project_name=project_name,
                chatroom_name=chatroom_name,
                coordinator_id=coordinator_id,
                expected_participants=expected_participants,
                timeout_seconds=timeout_seconds,
            )
            self._pending[key] = work
        await self._emit_change(
            project_id, chatroom_name, list(work.expected_participants)
        )
        return work

    async def get_pending_work(
        self, project_id: str, chatroom_name: str
    ) -> Optional[PendingWork]:
        return self._pending.get((project_id, chatroom_name))

    async def record_response(
        self, project_id: str, chatroom_name: str, participant_id: str
    ) -> Optional[PendingWork]:
        key = (project_id, chatroom_name)
        changed = False
        async with self._lock:
            work = self._pending.get(key)
            if work is None:
                return None
            if participant_id not in work.expected_participants:
                return None
            if participant_id not in work.responded_participants:
                # Use immutable update pattern
                new_responded = work.responded_participants + [participant_id]
                work = work.model_copy(update={"responded_participants": new_responded})
                self._pending[key] = work
                changed = True
        if changed:
            await self._emit_change(
                project_id, chatroom_name, work.timed_out_participants
            )
        return work

    async def clear_pending_work(self, project_id: str, chatroom_name: str) -> None:
        async with self._lock:
            existed = self._pending.pop((project_id, chatroom_name), None) is not None
        if existed:
            await self._emit_change(project_id, chatroom_name, [])

    async def clear_project(self, project_id: str) -> None:
        """Remove all pending work for a project."""
        async with self._lock:
            to_remove = [
                key for key in self._pending if key[0] == project_id
            ]
            for key in to_remove:
                del self._pending[key]
        for _, chatroom_name in to_remove:
            await self._emit_change(project_id, chatroom_name, [])

    async def get_all_pending_work(self) -> list[PendingWork]:
        return list(self._pending.values())

    async def update_agent_processing(
        self, agent_id: str, processing_message_ids: list[str]
    ) -> None:
        async with self._lock:
            self._agent_processing[agent_id] = processing_message_ids

    async def get_agent_processing(self, agent_id: str) -> list[str]:
        return self._agent_processing.get(agent_id, [])

    async def get_processing_agents(
        self, project_id: str, chatroom_name: str, participant_ids: list[str]
    ) -> list[str]:
        """Return participants of ``(project_id, chatroom_name)`` with outstanding work.

        Scoped to a single chatroom: only looks at the pending batch for this
        specific (project, chatroom) pair, not pending work elsewhere. This
        prevents a coordinator who is busy in one project's user-communication
        from lighting up the typing indicator in an unrelated DM chatroom.
        """
        work = self._pending.get((project_id, chatroom_name))
        if work is None:
            return []
        outstanding = {
            pid for pid in work.expected_participants
            if pid not in work.responded_participants
        }
        return [pid for pid in participant_ids if pid in outstanding]
