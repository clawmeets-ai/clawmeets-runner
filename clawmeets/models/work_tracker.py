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
    timeout_seconds: int = 1800        # Default 30 minutes

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
        timeout_seconds: int = 1800,
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

    async def record_response_opening_if_missing(
        self,
        project_id: str,
        project_name: str,
        chatroom_name: str,
        participant_id: str,
        *,
        coordinator_id: str,
        message_id: str,
        message_version: int,
        timeout_seconds: int = 1800,
    ) -> Optional[PendingWork]:
        """Record a response, synthesizing a one-participant batch if none is open.

        A reply can arrive in a room with no open batch — e.g. an agent posts a
        follow-up after its earlier batch already completed, or a foreign agent
        mirrors a second (final) message through the tunnel after its first reply
        closed the batch. Without an open batch the reply would be invisible to
        the batch-completion machinery and the room's coordinator would never be
        re-woken to look at it.

        When no batch is open for ``(project_id, chatroom_name)`` this synthesizes
        one expecting only ``participant_id`` (coordinator = ``coordinator_id``,
        the room's coordinator), then records the response — so the batch is
        immediately complete and the caller fires BATCH_COMPLETE as usual. When a
        batch IS already open, this is exactly ``record_response`` (the reply
        credits the existing batch).

        Callers MUST guard ``participant_id != coordinator_id`` before calling —
        synthesizing a batch whose coordinator is the replier would wake the
        replier on its own message (infinite loop).
        """
        if self._pending.get((project_id, chatroom_name)) is None:
            try:
                await self.create_pending_work(
                    message_id=message_id,
                    message_version=message_version,
                    project_id=project_id,
                    project_name=project_name,
                    chatroom_name=chatroom_name,
                    coordinator_id=coordinator_id,
                    expected_participants=[participant_id],
                    timeout_seconds=timeout_seconds,
                )
            except ValueError:
                # A concurrent reply opened a batch between the check and the
                # create — fall through and record against whatever is now open.
                pass
        return await self.record_response(project_id, chatroom_name, participant_id)

    async def remap_expected(
        self, project_id: str, chatroom_name: str, mapping: dict[str, str]
    ) -> None:
        """Swap participant ids in a pending batch's ``expected_participants``.

        Used when an agent is re-registered with a new id and we need any
        in-flight batch addressed to the deleted predecessor to credit the
        new agent's reply (otherwise ``record_response(new_id)`` no-ops
        against a stale expected id and the batch silently times out).

        No-op when there's no pending work, or no expected id matches the
        mapping's keys. Preserves the responded_participants set (if a
        stale id is in responded, that's already credited and we keep it).
        """
        if not mapping:
            return
        key = (project_id, chatroom_name)
        changed = False
        async with self._lock:
            work = self._pending.get(key)
            if work is None:
                return
            new_expected = [mapping.get(pid, pid) for pid in work.expected_participants]
            if new_expected == work.expected_participants:
                return
            work = work.model_copy(update={"expected_participants": new_expected})
            self._pending[key] = work
            changed = True
        if changed:
            await self._emit_change(
                project_id, chatroom_name, list(work.expected_participants)
            )

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

    async def is_project_idle(self, project_id: str) -> bool:
        """Return True iff no chatroom in ``project_id`` has an open batch.

        ``_pending`` only holds incomplete batches (entries are removed by
        :meth:`clear_pending_work` on BATCH_COMPLETE), so the presence of any
        key whose first element is ``project_id`` is sufficient to decide
        the project is busy.
        """
        return not any(key[0] == project_id for key in self._pending)

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
