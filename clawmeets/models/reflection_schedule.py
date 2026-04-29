"""
clawmeets/models/reflection_schedule.py
Reflection schedule model and persistence store.

A user has at most one ReflectionSchedule. When the schedule fires, the
ReflectionScheduler walks the user's agents and triggers a reflection cycle
for each agent that has unreflected activity since its last reflection.

Persisted as NDJSON (one entry per user) at:
    {server_dir}/metadata/reflection_schedules.ndjson
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Optional

from croniter import croniter
from pydantic import BaseModel

from clawmeets.models.scheduled_message import compute_next_fire, validate_cron_expression
from clawmeets.utils.file_io import FileUtil

if TYPE_CHECKING:
    from clawmeets.models.context import ModelContext

logger = logging.getLogger("clawmeets.models.reflection_schedule")


class ReflectionSchedule(BaseModel):
    """A user-scoped recurring reflection schedule.

    Carries two independent cadences:
    - ``cron_expression`` / ``next_fire_at`` / ``last_fired_at`` for the *reflect*
      pass (distill new lessons; fans out to agents with new activity).
    - ``lint_cron_expression`` / ``next_lint_fire_at`` / ``last_lint_fired_at``
      for the *lint* pass (audit existing memory; fans out to agents that have
      a memory to lint, i.e. ``last_reflected_at is not None``).

    Lint cadence is opt-in: ``lint_cron_expression == None`` disables the lint
    fan-out even when the reflect cron is active. Setting ``is_active = False``
    disables both.
    """

    user_id: str
    username: str
    cron_expression: str
    is_active: bool = True
    created_at: datetime
    last_fired_at: Optional[datetime] = None
    next_fire_at: datetime
    end_at: Optional[datetime] = None
    # Lint cadence (v2). Independent cursors so reflect and lint advance separately.
    lint_cron_expression: Optional[str] = None
    last_lint_fired_at: Optional[datetime] = None
    next_lint_fire_at: Optional[datetime] = None


class ReflectionScheduleStore:
    """File-backed persistence for ReflectionSchedule.

    One entry per user_id (upsert semantics). Mirrors ScheduledMessageStore's
    interface but keyed on user_id rather than message id.
    """

    def __init__(self, model_ctx: "ModelContext") -> None:
        self._model_ctx = model_ctx
        self._lock = asyncio.Lock()

    def _store_path(self):
        return self._model_ctx._base_dir / "metadata" / "reflection_schedules.ndjson"

    def _load_all_sync(self) -> list[ReflectionSchedule]:
        path = self._store_path()
        entries = FileUtil.read(path, "ndjson")
        if not entries:
            return []
        schedules: list[ReflectionSchedule] = []
        for entry in entries:
            try:
                schedules.append(ReflectionSchedule.model_validate(entry))
            except Exception:
                logger.warning(f"Skipping invalid reflection schedule entry: {entry}")
        return schedules

    def _save_all_sync(self, schedules: list[ReflectionSchedule]) -> None:
        path = self._store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not schedules:
            FileUtil.write(path, "", "text")
            return
        FileUtil.write(path, schedules[0].model_dump(mode="json"), "ndjson", mode="w")
        for s in schedules[1:]:
            FileUtil.write(path, s.model_dump(mode="json"), "ndjson", mode="a")

    async def get_by_user(self, user_id: str) -> Optional[ReflectionSchedule]:
        async with self._lock:
            for s in self._load_all_sync():
                if s.user_id == user_id:
                    return s
        return None

    async def list_active(self) -> list[ReflectionSchedule]:
        async with self._lock:
            return [s for s in self._load_all_sync() if s.is_active]

    async def upsert(
        self,
        user_id: str,
        username: str,
        cron_expression: str,
        is_active: bool = True,
        end_at: Optional[datetime] = None,
        lint_cron_expression: Optional[str] = None,
    ) -> ReflectionSchedule:
        """Create or update the user's reflection schedule.

        Validates both crons. Recomputes ``next_fire_at`` from now. Recomputes
        ``next_lint_fire_at`` only when ``lint_cron_expression`` differs from
        the stored value (so toggling reflect doesn't reset lint progress).
        Pass ``lint_cron_expression=None`` to clear lint cadence.
        Preserves ``last_fired_at`` / ``last_lint_fired_at`` on update.
        """
        if not validate_cron_expression(cron_expression):
            raise ValueError(f"Invalid cron expression: {cron_expression!r}")
        if lint_cron_expression is not None and not validate_cron_expression(lint_cron_expression):
            raise ValueError(f"Invalid lint cron expression: {lint_cron_expression!r}")

        now = datetime.now(UTC)
        next_fire = compute_next_fire(cron_expression, now)

        async with self._lock:
            schedules = self._load_all_sync()
            existing: Optional[ReflectionSchedule] = None
            for s in schedules:
                if s.user_id == user_id:
                    existing = s
                    break
            if existing is not None:
                existing.username = username
                existing.cron_expression = cron_expression
                existing.is_active = is_active
                existing.next_fire_at = next_fire
                existing.end_at = end_at
                # Only recompute next_lint_fire_at when the cron actually changed —
                # avoids skipping a pending lint fire just because the user
                # re-saved an unchanged settings form.
                if lint_cron_expression != existing.lint_cron_expression:
                    existing.lint_cron_expression = lint_cron_expression
                    existing.next_lint_fire_at = (
                        compute_next_fire(lint_cron_expression, now)
                        if lint_cron_expression
                        else None
                    )
                    if lint_cron_expression is None:
                        existing.last_lint_fired_at = None
                result = existing
            else:
                result = ReflectionSchedule(
                    user_id=user_id,
                    username=username,
                    cron_expression=cron_expression,
                    is_active=is_active,
                    created_at=now,
                    next_fire_at=next_fire,
                    end_at=end_at,
                    lint_cron_expression=lint_cron_expression,
                    next_lint_fire_at=(
                        compute_next_fire(lint_cron_expression, now)
                        if lint_cron_expression
                        else None
                    ),
                )
                schedules.append(result)
            self._save_all_sync(schedules)
            return result

    async def deactivate(self, user_id: str) -> bool:
        """Mark a user's reflection schedule inactive. Returns True if found."""
        async with self._lock:
            schedules = self._load_all_sync()
            for s in schedules:
                if s.user_id == user_id:
                    if not s.is_active:
                        return False
                    s.is_active = False
                    self._save_all_sync(schedules)
                    return True
        return False

    async def update_after_fire(self, user_id: str, now: datetime) -> None:
        """Advance the *reflect* cursor after a reflect fan-out fires."""
        async with self._lock:
            schedules = self._load_all_sync()
            for s in schedules:
                if s.user_id == user_id:
                    s.last_fired_at = now
                    try:
                        s.next_fire_at = compute_next_fire(s.cron_expression, now)
                    except ValueError:
                        logger.error(
                            f"Invalid cron for reflection schedule user={user_id}, deactivating"
                        )
                        s.is_active = False
                        break
                    if s.end_at and s.next_fire_at > s.end_at:
                        s.is_active = False
                    break
            self._save_all_sync(schedules)

    async def update_after_lint_fire(self, user_id: str, now: datetime) -> None:
        """Advance the *lint* cursor after a lint fan-out fires.

        Independent of the reflect cursor — they tick at their own cadences.
        If the lint cron is no longer present (cleared via upsert), this is a
        no-op so a stale loop iteration can't resurrect the cursor.
        """
        async with self._lock:
            schedules = self._load_all_sync()
            for s in schedules:
                if s.user_id == user_id:
                    if s.lint_cron_expression is None:
                        break
                    s.last_lint_fired_at = now
                    try:
                        s.next_lint_fire_at = compute_next_fire(s.lint_cron_expression, now)
                    except ValueError:
                        logger.error(
                            f"Invalid lint cron for reflection schedule user={user_id}, "
                            "clearing lint cadence"
                        )
                        s.lint_cron_expression = None
                        s.next_lint_fire_at = None
                        break
                    if s.end_at and s.next_lint_fire_at and s.next_lint_fire_at > s.end_at:
                        s.lint_cron_expression = None
                        s.next_lint_fire_at = None
                    break
            self._save_all_sync(schedules)
