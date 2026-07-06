# SPDX-License-Identifier: MIT
"""
clawmeets/models/scheduled_message.py
Scheduled message model and persistence store.

Allows users to schedule recurring messages to chatrooms or DM rooms
using standard cron expressions. Messages are persisted as NDJSON and
fired by the MessageScheduler background task.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
from pydantic import BaseModel

from clawmeets.utils.file_io import FileUtil

if TYPE_CHECKING:
    from clawmeets.models.context import ModelContext

logger = logging.getLogger("clawmeets.models.scheduled_message")


def compute_next_fire(
    cron_expression: str, after: datetime, tz: str = "UTC"
) -> datetime:
    """Compute the next fire time after a given datetime using croniter.

    The cron expression is interpreted in ``tz`` (an IANA name like
    ``"Asia/Taipei"``), so ``"0 9 * * *"`` with ``tz="Asia/Taipei"`` fires at
    9am Taipei. The returned datetime is always converted back to UTC for
    storage.

    Args:
        cron_expression: Standard cron expression or preset (@hourly, @daily, @weekly)
        after: Compute next fire time after this datetime (any tz-aware datetime)
        tz: IANA timezone the cron should be interpreted in. Defaults to UTC.

    Returns:
        Next fire datetime (UTC, tz-aware)

    Raises:
        ValueError: If cron_expression or tz is invalid
    """
    if not croniter.is_valid(cron_expression):
        raise ValueError(f"Invalid cron expression: {cron_expression!r}")
    try:
        zone = ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid timezone: {tz!r}") from exc
    after_local = after.astimezone(zone)
    cron = croniter(cron_expression, after_local)
    next_local = cron.get_next(datetime)
    # croniter may strip tzinfo on some versions; re-anchor in the user's zone
    # before converting to UTC so we never interpret a naive datetime as UTC.
    if next_local.tzinfo is None:
        next_local = next_local.replace(tzinfo=zone)
    return next_local.astimezone(UTC)


def validate_cron_expression(cron_expression: str) -> bool:
    """Check if a cron expression is valid."""
    return croniter.is_valid(cron_expression)


def validate_timezone(tz: str) -> bool:
    """Check if a string is a valid IANA timezone name."""
    try:
        ZoneInfo(tz)
        return True
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        return False


class ScheduledMessage(BaseModel):
    """A scheduled message that fires on a cron schedule.

    ``timezone`` is the IANA name the cron expression is evaluated in (default
    ``"UTC"`` for backward compat with rows written before the field existed).
    All timestamps (``next_fire_at``, ``last_fired_at``, ...) remain UTC.
    """

    id: str
    user_id: str
    username: str
    project_id: str
    chatroom_name: str
    content: str
    cron_expression: str
    timezone: str = "UTC"
    end_at: Optional[datetime] = None
    created_at: datetime
    last_fired_at: Optional[datetime] = None
    next_fire_at: datetime
    is_active: bool = True
    idle_only: bool = False


class ScheduledMessageStore:
    """Thread-safe NDJSON persistence for scheduled messages.

    Stores at {base_dir}/metadata/scheduled_messages.ndjson.
    Uses an asyncio.Lock to prevent concurrent read-modify-write races.
    """

    def __init__(self, model_ctx: "ModelContext") -> None:
        self._model_ctx = model_ctx
        self._lock = asyncio.Lock()

    def _store_path(self):
        return self._model_ctx._base_dir / "metadata" / "scheduled_messages.ndjson"

    def _load_all_sync(self) -> list[ScheduledMessage]:
        """Load all scheduled messages from disk (synchronous)."""
        path = self._store_path()
        entries = FileUtil.read(path, "ndjson")
        if not entries:
            return []
        messages = []
        for entry in entries:
            try:
                messages.append(ScheduledMessage.model_validate(entry))
            except Exception:
                logger.warning(f"Skipping invalid scheduled message entry: {entry}")
        return messages

    def _save_all_sync(self, messages: list[ScheduledMessage]) -> None:
        """Save all scheduled messages to disk (atomic rewrite)."""
        path = self._store_path()
        # Atomic rewrite: write all entries
        data = [m.model_dump(mode="json") for m in messages]
        FileUtil.write(path, data, "json")  # Write as JSON array, read back as ndjson-compatible

    def _save_all_ndjson_sync(self, messages: list[ScheduledMessage]) -> None:
        """Save all scheduled messages as NDJSON (one JSON object per line)."""
        path = self._store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write as NDJSON: truncate and write each entry
        # First clear the file, then append each entry
        if not messages:
            FileUtil.write(path, "", "text")
            return
        # Write first entry (truncate)
        FileUtil.write(path, messages[0].model_dump(mode="json"), "ndjson", mode="w")
        # Append remaining
        for msg in messages[1:]:
            FileUtil.write(path, msg.model_dump(mode="json"), "ndjson", mode="a")

    async def load_all(self) -> list[ScheduledMessage]:
        """Load all scheduled messages."""
        async with self._lock:
            return self._load_all_sync()

    async def get_active(self) -> list[ScheduledMessage]:
        """Load only active scheduled messages."""
        async with self._lock:
            return [m for m in self._load_all_sync() if m.is_active]

    async def get_by_user(self, user_id: str) -> list[ScheduledMessage]:
        """Load scheduled messages for a specific user."""
        async with self._lock:
            return [m for m in self._load_all_sync() if m.user_id == user_id]

    async def add(self, msg: ScheduledMessage) -> ScheduledMessage:
        """Add a new scheduled message."""
        async with self._lock:
            messages = self._load_all_sync()
            messages.append(msg)
            self._save_all_ndjson_sync(messages)
            return msg

    async def deactivate(self, msg_id: str, user_id: str) -> bool:
        """Deactivate a scheduled message. Returns True if found and deactivated."""
        async with self._lock:
            messages = self._load_all_sync()
            for msg in messages:
                if msg.id == msg_id and msg.user_id == user_id:
                    if not msg.is_active:
                        return False
                    msg.is_active = False
                    self._save_all_ndjson_sync(messages)
                    return True
            return False

    async def update(
        self,
        msg_id: str,
        user_id: str,
        *,
        project_id: str,
        chatroom_name: str,
        content: str,
        cron_expression: str,
        timezone: str,
        end_at: Optional[datetime],
    ) -> Optional[ScheduledMessage]:
        """Update an existing scheduled message in place (user-scoped).

        Rewrites the editable fields and recomputes ``next_fire_at`` from the new
        cron/timezone. Returns the updated message, or ``None`` if not found or
        not owned by ``user_id``. Callers must validate cron/timezone/target
        first (see the PUT route).
        """
        async with self._lock:
            messages = self._load_all_sync()
            for msg in messages:
                if msg.id == msg_id and msg.user_id == user_id:
                    msg.project_id = project_id
                    msg.chatroom_name = chatroom_name
                    msg.content = content
                    msg.cron_expression = cron_expression
                    msg.timezone = timezone
                    msg.end_at = end_at
                    msg.next_fire_at = compute_next_fire(
                        cron_expression, datetime.now(UTC), timezone
                    )
                    if msg.end_at and msg.next_fire_at > msg.end_at:
                        msg.is_active = False
                    self._save_all_ndjson_sync(messages)
                    return msg
            return None

    async def update_after_fire(self, msg_id: str, now: datetime) -> None:
        """Update a message after it has been fired.

        Sets last_fired_at, computes next_fire_at, and deactivates if past end_at.
        """
        async with self._lock:
            messages = self._load_all_sync()
            for msg in messages:
                if msg.id == msg_id:
                    msg.last_fired_at = now
                    try:
                        msg.next_fire_at = compute_next_fire(
                            msg.cron_expression, now, msg.timezone
                        )
                    except ValueError:
                        logger.error(f"Invalid cron for scheduled message {msg_id}, deactivating")
                        msg.is_active = False
                        break
                    # Check if past end_at
                    if msg.end_at and msg.next_fire_at > msg.end_at:
                        msg.is_active = False
                    break
            self._save_all_ndjson_sync(messages)

    async def update_next_fire_only(self, msg_id: str, now: datetime) -> None:
        """Advance ``next_fire_at`` without recording a fire.

        Mirrors :meth:`update_after_fire` but skips the ``last_fired_at`` write.
        Used by the idle gate when a scheduled fire is deferred because the
        target project is busy: we still need to push ``next_fire_at`` forward
        so the scheduler doesn't keep re-evaluating the same overdue cron tick.
        """
        async with self._lock:
            messages = self._load_all_sync()
            for msg in messages:
                if msg.id == msg_id:
                    try:
                        msg.next_fire_at = compute_next_fire(
                            msg.cron_expression, now, msg.timezone
                        )
                    except ValueError:
                        logger.error(f"Invalid cron for scheduled message {msg_id}, deactivating")
                        msg.is_active = False
                        break
                    if msg.end_at and msg.next_fire_at > msg.end_at:
                        msg.is_active = False
                    break
            self._save_all_ndjson_sync(messages)

    async def deactivate_by_id(self, msg_id: str) -> None:
        """Deactivate a scheduled message by ID (no user check). Used by scheduler."""
        async with self._lock:
            messages = self._load_all_sync()
            for msg in messages:
                if msg.id == msg_id:
                    msg.is_active = False
                    break
            self._save_all_ndjson_sync(messages)
