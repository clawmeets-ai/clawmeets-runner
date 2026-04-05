# SPDX-License-Identifier: MIT
"""
clawmeets/sync/runloop.py
Per-project changelog runloop with sync, enqueue, and persistence.

This module is part of Layer 0 (pure - minimal domain dependencies).
Each project gets its own runloop instance. The runloop handles:
- Thread-safe processing via asyncio.Lock
- Version tracking (last_processed_version)
- Changelog persistence
- Sync orchestration with callback-based fetch
- Subscriber management (entry-by-entry processing)
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Awaitable, Callable, Optional

from .changelog import ChangelogEntry, ChangelogPayload
from .subscriber import ChangelogSubscriber
from clawmeets.utils.file_io import FileUtil

logger = logging.getLogger(__name__)


class ChangelogRunloop:
    """Per-project changelog runloop with sync, enqueue, and persistence.

    Each project gets its own runloop instance. The runloop handles:
    - Thread-safe processing via asyncio.Lock
    - Version tracking (last_processed_version)
    - Changelog persistence
    - Sync orchestration with callback-based fetch
    - Subscriber management with entry-by-entry processing

    Subscribers are added via add_subscriber() and called in insertion order.
    Each entry is processed through ALL subscribers before the next entry,
    ensuring that files are ready before callbacks fire.

    Usage:
        runloop = ChangelogRunloop(
            project_id="abc123",
            project_name="my-project",
            changelog_dir=Path(".agents/my-agent/metadata/projects/my-project-abc123"),
        )

        # Add subscribers in order (ModelContextChangelogSubscriber first, ParticipantNotifier second)
        runloop.add_subscriber(model_ctx.changelog_subscriber(project_id, project_name))
        runloop.add_subscriber(notifier)

        await runloop.load_state()

        # Sync with server
        processed = await runloop.sync(
            new_version=10,
            fetch_callback=fetch_entries_from_server,
        )
    """

    def __init__(
        self,
        project_id: str,
        project_name: str,
        changelog_dir: Path,
    ) -> None:
        """Initialize the runloop.

        Args:
            project_id: The project ID this runloop handles
            project_name: The project name (for path resolution)
            changelog_dir: Directory for changelog persistence
        """
        self._project_id = project_id
        self._project_name = project_name
        self._changelog_dir = changelog_dir

        # Subscriber list (replaces CompositeSubscriber)
        # Subscribers are called in insertion order for each entry
        self._subscribers: list[ChangelogSubscriber] = []

        # Version tracking
        self._last_processed_version: int = 0

        # Thread safety
        self._lock = asyncio.Lock()

        # Pending entries queue
        self._pending_entries: list[ChangelogEntry] = []

    # ─────────────────────────────────────────────────────────
    # Subscriber Management
    # ─────────────────────────────────────────────────────────

    def add_subscriber(self, subscriber: ChangelogSubscriber) -> None:
        """Add a subscriber to the chain.

        Subscribers are called in insertion order for each entry.
        Add ModelContext first, then ParticipantNotifier.

        Args:
            subscriber: The subscriber to add
        """
        self._subscribers.append(subscriber)
        logger.debug(f"Added subscriber {subscriber.__class__.__name__}")

    # ─────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────

    @property
    def project_id(self) -> str:
        """Get the project ID."""
        return self._project_id

    @property
    def project_name(self) -> str:
        """Get the project name."""
        return self._project_name

    @property
    def last_processed_version(self) -> int:
        """Get last processed version number."""
        return self._last_processed_version

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    async def sync(
        self,
        new_version: int,
        fetch_callback: Callable[[int, int], Awaitable[list[ChangelogEntry]]],
    ) -> int:
        """Sync to target version using fetch callback.

        Thread-safe. Fetches new entries if needed, then processes all pending
        entries (including any from crash recovery).

        Args:
            new_version: Target version to sync to
            fetch_callback: Async function that fetches entries between versions.
                            Called as fetch_callback(file_version, target_version)

        Returns:
            Number of entries processed
        """
        async with self._lock:
            file_version = self.get_current_version()

            # Fetch new entries if server has more than our file
            if new_version > file_version:
                entries = await fetch_callback(file_version, new_version)
                if entries:
                    await self._enqueue_internal(entries)

            # Process all pending entries (including crash recovery)
            processed = await self._process_queue_internal()
            return processed

    async def load_state(self) -> None:
        """Load persisted state (call before use)."""
        state_path = self._changelog_dir / "runloop_state.json"
        if not state_path.exists():
            return

        async with self._lock:
            state = FileUtil.read(state_path, "json", default=None)
            if state is None:
                logger.warning(f"Failed to load runloop state from {state_path}")
                return

            self._last_processed_version = state.get("last_processed_version", 0)

            # Load pending entries from changelog
            await self._load_pending_entries()

            logger.debug(
                f"Loaded runloop state for project {self._project_id[:8]}: "
                f"last_processed={self._last_processed_version}, "
                f"pending={len(self._pending_entries)}"
            )

    async def save_state(self) -> None:
        """Explicitly save state (for graceful shutdown)."""
        async with self._lock:
            await self._save_state_internal()

    # ─────────────────────────────────────────────────────────
    # Server-Side Append (version assignment)
    # ─────────────────────────────────────────────────────────

    async def append(
        self,
        entry_type: ChangelogEntryType,
        payload: ChangelogPayload,
    ) -> ChangelogEntry:
        """Append entry with version assignment (server-side).

        Reads current version from changelog, assigns version+1,
        persists, and processes through subscribers.

        Args:
            entry_type: Type of changelog entry
            payload: Entry payload (chatroom_name is in payload for chatroom-scoped entries)

        Returns:
            The created ChangelogEntry with assigned version
        """
        async with self._lock:
            # Get current version from file
            current_version = self.get_current_version()

            # Create entry with assigned version
            entry = ChangelogEntry(
                version=current_version + 1,
                entry_type=entry_type,
                payload=payload,
            )

            # Persist to changelog.ndjson
            await self._persist_entries([entry])

            # Add to pending and process
            self._pending_entries.append(entry)
            await self._process_queue_internal()

            return entry

    # ─────────────────────────────────────────────────────────
    # Query Methods
    # ─────────────────────────────────────────────────────────

    def get_entries_since(
        self,
        since_version: int
    ) -> list[ChangelogEntry]:
        """Get entries with version > since_version.

        Args:
            since_version: Return entries after this version

        Returns:
            List of ChangelogEntry objects
        """
        changelog_path = self._changelog_dir / "changelog.ndjson"
        entries = []
        for line in changelog_path.read_text(encoding="utf-8").splitlines():
            entry = ChangelogEntry.model_validate_json(line)
            if entry.version <= since_version:
                continue
            entries.append(entry)
        return entries

    def get_current_version(self) -> int:
        """Get latest version number (0 if no entries)."""
        changelog_path = self._changelog_dir / "changelog.ndjson"
        if not changelog_path.exists():
            return 0
        lines = changelog_path.read_text(encoding="utf-8").strip().split('\n')
        if not lines or not lines[-1]:
            return 0
        last_entry = ChangelogEntry.model_validate_json(lines[-1])
        return last_entry.version

    # ─────────────────────────────────────────────────────────
    # Internal Methods (lock must be held)
    # ─────────────────────────────────────────────────────────

    async def _enqueue_internal(self, entries: list[ChangelogEntry]) -> None:
        """Enqueue and persist. Lock must be held."""
        sorted_entries = sorted(entries, key=lambda e: e.version)
        new_entries = [e for e in sorted_entries if e.version > self._last_processed_version]

        if not new_entries:
            return

        # Persist to local changelog
        await self._persist_entries(new_entries)

        self._pending_entries.extend(new_entries)

    async def _process_queue_internal(self) -> int:
        """Process pending entries one at a time through all subscribers.

        Each entry is processed through ALL subscribers before the next entry.
        This ensures files are ready before callbacks fire for each version.
        """
        self._pending_entries.sort(key=lambda e: e.version)

        # Process each entry through ALL subscribers before next entry
        # Pop entries as we process them for explicit "done with this entry" semantics
        count = 0
        while self._pending_entries:
            entry = self._pending_entries.pop(0)

            for subscriber in self._subscribers:
                await subscriber.on_entry(
                    entry,
                    self._project_id,
                    self._project_name,
                )

            # Update and save state AFTER each entry (crash safety)
            self._last_processed_version = entry.version
            await self._save_state_internal()
            count += 1

        logger.debug(
            f"Processed {count} entries for project {self._project_id[:8]}, "
            f"version now {self._last_processed_version}"
        )

        # Notify subscribers that sync batch is complete
        if count > 0:
            for subscriber in self._subscribers:
                await subscriber.on_sync_complete(
                    self._project_id,
                    self._project_name,
                )

        return count

    async def _persist_entries(self, entries: list[ChangelogEntry]) -> None:
        """Persist entries to changelog file."""
        changelog_path = self._changelog_dir / "changelog.ndjson"

        for entry in entries:
            # Use text format with mode="a" for appending JSON lines
            FileUtil.write(
                changelog_path,
                entry.model_dump_json(by_alias=True) + "\n",
                "text",
                mode="a",
            )

    async def _save_state_internal(self) -> None:
        """Save runloop state. Lock must be held."""
        state_path = self._changelog_dir / "runloop_state.json"

        state = {
            "project_id": self._project_id,
            "last_processed_version": self._last_processed_version,
        }

        FileUtil.write(state_path, state, "json", atomic=True)

    async def _load_pending_entries(self) -> None:
        """Load unprocessed entries from changelog."""
        changelog_path = self._changelog_dir / "changelog.ndjson"
        if not changelog_path.exists():
            return

        for line in changelog_path.read_text(encoding="utf-8").splitlines():
            entry = ChangelogEntry.model_validate_json(line)
            if entry.version <= self._last_processed_version:
                continue
            self._pending_entries.append(entry)

        self._pending_entries.sort(key=lambda e: e.version)

    def __repr__(self) -> str:
        return (
            f"ChangelogRunloop("
            f"project={self._project_name}-{self._project_id[:8]}, "
            f"version={self._last_processed_version})"
        )
