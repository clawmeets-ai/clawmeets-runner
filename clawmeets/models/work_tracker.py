# SPDX-License-Identifier: MIT
"""
clawmeets/models/work_tracker.py

Work tracking for coordinator batch completion detection.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Optional

from pydantic import BaseModel, Field


class PendingWork(BaseModel):
    """Tracks a coordinator's work dispatch for batch completion detection.

    When a coordinator sends a message with expects_response_from, the server
    creates a PendingWork record. As participants respond, the server updates
    responded_participants. When all expected participants have responded, the server
    sends BATCH_COMPLETE to the coordinator.
    """
    model_config = {"frozen": True}

    message_id: str                    # Coordinator's message that initiated this wave
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
    """

    def __init__(self) -> None:
        # chatroom_name -> PendingWork
        self._pending: dict[str, PendingWork] = {}
        # agent_id -> list of message IDs currently being processed
        self._agent_processing: dict[str, list[str]] = {}
        self._lock = asyncio.Lock()

    async def create_pending_work(
        self,
        message_id: str,
        project_id: str,
        project_name: str,
        chatroom_name: str,
        coordinator_id: str,
        expected_participants: list[str],
        timeout_seconds: int = 600,
    ) -> PendingWork:
        async with self._lock:
            if chatroom_name in self._pending:
                raise ValueError(
                    f"Pending work already exists for chatroom {chatroom_name}"
                )
            work = PendingWork(
                message_id=message_id,
                project_id=project_id,
                project_name=project_name,
                chatroom_name=chatroom_name,
                coordinator_id=coordinator_id,
                expected_participants=expected_participants,
                timeout_seconds=timeout_seconds,
            )
            self._pending[chatroom_name] = work
            return work

    async def get_pending_work(self, chatroom_name: str) -> Optional[PendingWork]:
        return self._pending.get(chatroom_name)

    async def record_response(
        self, chatroom_name: str, participant_id: str
    ) -> Optional[PendingWork]:
        async with self._lock:
            work = self._pending.get(chatroom_name)
            if work is None:
                return None
            if participant_id not in work.expected_participants:
                return None
            if participant_id not in work.responded_participants:
                # Use immutable update pattern
                new_responded = work.responded_participants + [participant_id]
                work = work.model_copy(update={"responded_participants": new_responded})
                self._pending[chatroom_name] = work
            return work

    async def clear_pending_work(self, chatroom_name: str) -> None:
        async with self._lock:
            self._pending.pop(chatroom_name, None)

    async def clear_project(self, project_id: str) -> None:
        """Remove all pending work for a project."""
        async with self._lock:
            to_remove = [
                chatroom_name
                for chatroom_name, work in self._pending.items()
                if work.project_id == project_id
            ]
            for chatroom_name in to_remove:
                del self._pending[chatroom_name]

    async def get_all_pending_work(self) -> list[PendingWork]:
        return list(self._pending.values())

    async def update_agent_processing(
        self, agent_id: str, processing_message_ids: list[str]
    ) -> None:
        async with self._lock:
            self._agent_processing[agent_id] = processing_message_ids

    async def get_agent_processing(self, agent_id: str) -> list[str]:
        return self._agent_processing.get(agent_id, [])

    async def get_processing_agents(self, participant_ids: list[str]) -> list[str]:
        """Return agent IDs from participants that are currently processing."""
        result = []
        for agent_id in participant_ids:
            if self._agent_processing.get(agent_id):
                result.append(agent_id)
        return result
