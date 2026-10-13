# SPDX-License-Identifier: MIT
"""Durable lifecycle state for runner-managed background jobs."""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from clawmeets.utils.file_io import FileUtil

logger = logging.getLogger("clawmeets.models.managed_job")


class ManagedJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    STALLED = "stalled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = {
    ManagedJobStatus.SUCCEEDED,
    ManagedJobStatus.FAILED,
    ManagedJobStatus.CANCELLED,
}


class ManagedJob(BaseModel):
    model_config = {"frozen": True}

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    project_name: str
    chatroom_name: str
    pending_message_id: str
    pending_message_version: int
    coordinator_id: str
    owner_agent_id: str
    status: ManagedJobStatus = ManagedJobStatus.RUNNING
    step: str | None = None
    progress: str | None = None
    checkpoint: str | None = None
    failure_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    lease_expires_at: datetime
    #: None = no absolute deadline. The heartbeat lease still applies, so a
    #: job whose process died is still reported as stalled.
    deadline_at: datetime | None = None
    heartbeat_timeout_seconds: int
    attempt: int = 1
    event_sequence: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def deadline_passed(self, now: datetime | None = None) -> bool:
        if self.deadline_at is None:
            return False
        return (now or datetime.now(UTC)) >= self.deadline_at

    def _capped_lease(self, now: datetime) -> datetime:
        lease = now + timedelta(seconds=self.heartbeat_timeout_seconds)
        return lease if self.deadline_at is None else min(self.deadline_at, lease)

    @property
    def has_healthy_lease(self) -> bool:
        now = datetime.now(UTC)
        return (
            not self.is_terminal
            and self.status != ManagedJobStatus.STALLED
            and now < self.lease_expires_at
            and not self.deadline_passed(now)
        )


class ManagedJobError(ValueError):
    """A lifecycle request violated an invariant."""


class ManagedJobStore:
    """Atomic JSON persistence and state-machine enforcement for managed jobs."""

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "metadata" / "managed_jobs.json"
        self._lock = asyncio.Lock()

    def _load(self) -> list[ManagedJob]:
        rows = FileUtil.read(self._path, "json") or []
        jobs: list[ManagedJob] = []
        for row in rows:
            try:
                jobs.append(ManagedJob.model_validate(row))
            except Exception:
                logger.warning("Skipping invalid managed job row", exc_info=True)
        return jobs

    def _save(self, jobs: list[ManagedJob]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        FileUtil.write(
            self._path,
            [job.model_dump(mode="json") for job in jobs],
            "json",
        )

    async def create(
        self,
        *,
        project_id: str,
        project_name: str,
        chatroom_name: str,
        pending_message_id: str,
        pending_message_version: int,
        coordinator_id: str,
        owner_agent_id: str,
        deadline_at: datetime | None,
        heartbeat_timeout_seconds: int,
        step: str | None = None,
    ) -> ManagedJob:
        now = datetime.now(UTC)
        if deadline_at is not None:
            if deadline_at.tzinfo is None:
                raise ManagedJobError("deadline_at must include a timezone")
            deadline_at = deadline_at.astimezone(UTC)
            if deadline_at <= now:
                raise ManagedJobError("deadline_at must be in the future")
        if heartbeat_timeout_seconds < 10:
            raise ManagedJobError("heartbeat_timeout_seconds must be at least 10")
        async with self._lock:
            jobs = self._load()
            for existing in jobs:
                if (
                    existing.project_id == project_id
                    and existing.chatroom_name == chatroom_name
                    and existing.pending_message_id == pending_message_id
                    and not existing.is_terminal
                ):
                    raise ManagedJobError("an active managed job already owns this pending work")
            job = ManagedJob(
                project_id=project_id,
                project_name=project_name,
                chatroom_name=chatroom_name,
                pending_message_id=pending_message_id,
                pending_message_version=pending_message_version,
                coordinator_id=coordinator_id,
                owner_agent_id=owner_agent_id,
                deadline_at=deadline_at,
                lease_expires_at=(
                    now + timedelta(seconds=heartbeat_timeout_seconds)
                    if deadline_at is None
                    else min(deadline_at, now + timedelta(seconds=heartbeat_timeout_seconds))
                ),
                heartbeat_timeout_seconds=heartbeat_timeout_seconds,
                step=step,
            )
            jobs.append(job)
            self._save(jobs)
            return job

    async def get(self, job_id: str) -> ManagedJob | None:
        async with self._lock:
            return next((job for job in self._load() if job.id == job_id), None)

    async def active_for_pending(self, project_id: str, room: str) -> ManagedJob | None:
        async with self._lock:
            candidates = [
                job for job in self._load()
                if job.project_id == project_id and job.chatroom_name == room and not job.is_terminal
            ]
            return max(candidates, key=lambda job: job.created_at, default=None)

    async def list_active(self) -> list[ManagedJob]:
        """Every non-terminal job, STALLED included (startup restore)."""
        async with self._lock:
            return [job for job in self._load() if not job.is_terminal]

    async def update(
        self,
        job_id: str,
        *,
        owner_agent_id: str,
        expected_sequence: int,
        status: ManagedJobStatus | None = None,
        progress: str | None = None,
        step: str | None = None,
        checkpoint: str | None = None,
        failure_reason: str | None = None,
        renew_lease: bool = False,
    ) -> ManagedJob:
        async with self._lock:
            jobs = self._load()
            index = next((i for i, job in enumerate(jobs) if job.id == job_id), None)
            if index is None:
                raise KeyError(job_id)
            job = jobs[index]
            if job.owner_agent_id != owner_agent_id:
                raise PermissionError("managed job belongs to another agent")
            if job.event_sequence != expected_sequence:
                raise ManagedJobError(
                    f"stale sequence: expected {job.event_sequence}, received {expected_sequence}"
                )
            if job.is_terminal:
                raise ManagedJobError(f"job is already {job.status.value}")

            target = status or job.status
            allowed = {
                ManagedJobStatus.QUEUED: {ManagedJobStatus.RUNNING, ManagedJobStatus.CANCELLED},
                ManagedJobStatus.RUNNING: {
                    ManagedJobStatus.RETRY_WAIT, ManagedJobStatus.STALLED,
                    ManagedJobStatus.SUCCEEDED, ManagedJobStatus.FAILED,
                    ManagedJobStatus.CANCELLED,
                },
                ManagedJobStatus.RETRY_WAIT: {ManagedJobStatus.RUNNING, ManagedJobStatus.CANCELLED},
                ManagedJobStatus.STALLED: {ManagedJobStatus.RUNNING, ManagedJobStatus.FAILED, ManagedJobStatus.CANCELLED},
            }
            if status is not None and status != job.status and status not in allowed.get(job.status, set()):
                raise ManagedJobError(f"invalid transition: {job.status.value} -> {status.value}")

            now = datetime.now(UTC)
            if job.deadline_passed(now) and target not in TERMINAL_STATUSES:
                raise ManagedJobError("job deadline has passed")
            if job.status == ManagedJobStatus.STALLED and target == ManagedJobStatus.STALLED:
                # Its open request was already reported as timed out; a late
                # heartbeat must not quietly revive it.
                raise ManagedJobError("job is stalled")
            lease = job.lease_expires_at
            if renew_lease:
                lease = job._capped_lease(now)
            updated = job.model_copy(update={
                "status": target,
                "progress": progress if progress is not None else job.progress,
                "step": step if step is not None else job.step,
                "checkpoint": checkpoint if checkpoint is not None else job.checkpoint,
                "failure_reason": failure_reason if failure_reason is not None else job.failure_reason,
                "updated_at": now,
                "lease_expires_at": lease,
                "event_sequence": job.event_sequence + 1,
                "attempt": job.attempt + (1 if job.status in {ManagedJobStatus.RETRY_WAIT, ManagedJobStatus.STALLED} and target == ManagedJobStatus.RUNNING else 0),
            })
            jobs[index] = updated
            self._save(jobs)
            return updated

    async def mark_stalled_if_expired(self, job_id: str) -> ManagedJob | None:
        async with self._lock:
            jobs = self._load()
            index = next((i for i, job in enumerate(jobs) if job.id == job_id), None)
            if index is None:
                return None
            job = jobs[index]
            now = datetime.now(UTC)
            if job.is_terminal or (now < job.lease_expires_at and not job.deadline_passed(now)):
                return job
            if job.status == ManagedJobStatus.STALLED:
                return job
            updated = job.model_copy(update={
                "status": ManagedJobStatus.STALLED,
                "failure_reason": "absolute deadline reached" if job.deadline_passed(now) else "heartbeat lease expired",
                "updated_at": now,
                "event_sequence": job.event_sequence + 1,
            })
            jobs[index] = updated
            self._save(jobs)
            return updated

    async def cancel_by_server(self, job_id: str, reason: str) -> ManagedJob | None:
        """Cancel a non-terminal job on the server's own authority (the owner
        cancelled the request it is attached to). No sequence check: the
        supervisor learns of it from its next refused event."""
        async with self._lock:
            jobs = self._load()
            index = next((i for i, job in enumerate(jobs) if job.id == job_id), None)
            if index is None:
                return None
            job = jobs[index]
            if job.is_terminal:
                return job
            updated = job.model_copy(update={
                "status": ManagedJobStatus.CANCELLED,
                "failure_reason": reason,
                "updated_at": datetime.now(UTC),
                "event_sequence": job.event_sequence + 1,
            })
            jobs[index] = updated
            self._save(jobs)
            return updated
