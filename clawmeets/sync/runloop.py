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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

from .changelog import (
    ChangelogEntry,
    ChangelogEntryType,
    ChangelogPayload,
    MirroredFromRef,
    iter_entries,
    ndjson_safe,
    read_tip_version_and_stat,
)
from .subscriber import ChangelogSubscriber
from clawmeets.utils.file_io import FileUtil

logger = logging.getLogger(__name__)


@dataclass
class BatchEntrySpec:
    """One entry in an :meth:`ChangelogRunloop.append_batch` call.

    ``source_version`` is an absolute reply-to link (same as ``append``).
    ``link_to_index`` instead links to the assigned version of another spec in
    the SAME batch (by its position in the list) — used to point attached FILE
    entries at the sibling MESSAGE entry whose version is only known once the
    batch assigns versions under the lock. Exactly one of the two should be set
    (``link_to_index`` wins if both are).
    """

    entry_type: ChangelogEntryType
    payload: ChangelogPayload
    source_version: int | None = None
    link_to_index: int | None = None
    mirrored_from: MirroredFromRef | None = None


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

        #: ``(inode, size, mtime_ns, version)`` for the changelog's last entry,
        #: or None. Four ints — **never parsed entries**: one project's are
        #: 1.2 GB of Python objects and a project-list request touches a
        #: thousand. See the note above ``read_tip_version_and_stat``.
        self._tip_cache: Optional[tuple[int, int, int, int]] = None

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
        """Load persisted state (call before use).

        Ends by telling every subscriber the state is loaded — **always**, on
        each of the three exits, including the two that loaded nothing. The
        callback's contract is *"on-disk state now predates every entry you will
        see"*, and that is as true of a project with no state file as of one
        resumed mid-stream; a subscriber seeded on only some of them would be
        seeded on a rule it could not state. Fired outside the lock, because a
        subscriber priming itself reads the same project state whose writes go
        through this runloop.
        """
        state_path = self._changelog_dir / "runloop_state.json"
        if not state_path.exists():
            await self._notify_state_loaded()
            return

        async with self._lock:
            state = FileUtil.read(state_path, "json", default=None)
            if state is None:
                logger.warning(f"Failed to load runloop state from {state_path}")
            else:
                self._last_processed_version = state.get("last_processed_version", 0)

                # Load pending entries from changelog
                await self._load_pending_entries()

                logger.debug(
                    f"Loaded runloop state for project {self._project_id[:8]}: "
                    f"last_processed={self._last_processed_version}, "
                    f"pending={len(self._pending_entries)}"
                )

        await self._notify_state_loaded()

    async def _notify_state_loaded(self) -> None:
        """Fire ``on_state_loaded`` on every subscriber, insertion order.

        Best-effort per subscriber: this hook only lets a subscriber PRIME
        itself, so one that raises must not stop the runloop from coming up —
        the cost of a skipped prime is the pre-fix behaviour, the cost of a
        raise here is a project that never syncs again.
        """
        for subscriber in self._subscribers:
            try:
                await subscriber.on_state_loaded(
                    self._project_id,
                    self._project_name,
                )
            except Exception:
                logger.exception(
                    f"on_state_loaded failed for project {self._project_id[:8]}"
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
        source_version: int | None = None,
        mirrored_from: MirroredFromRef | None = None,
    ) -> ChangelogEntry:
        """Append entry with version assignment (server-side).

        Reads current version from changelog, assigns version+1,
        persists, and processes through subscribers.

        Args:
            entry_type: Type of changelog entry
            payload: Entry payload (chatroom_name is in payload for chatroom-scoped entries)
            source_version: Version of the entry that triggered this one (reply-to link)
            mirrored_from: Set by TunnelSubscriber when this entry is a
                cross-project mirror of another entry. Subscribers use this
                as a loop guard.

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
                source_version=source_version,
                mirrored_from=mirrored_from,
            )

            # Persist to changelog.ndjson
            await self._persist_entries([entry])

            # Add to pending and process
            self._pending_entries.append(entry)
            await self._process_queue_internal()

            return entry

    async def append_batch(
        self,
        specs: list["BatchEntrySpec"],
    ) -> list[ChangelogEntry]:
        """Append several entries atomically under a single lock hold.

        All entries are assigned sequential versions and persisted BEFORE any
        subscriber runs, then processed in version order. This is what makes a
        "message with attachments" wake-safe: the FILE_* entries (which wake no
        agent) are given lower versions and process first, so the sibling
        MESSAGE entry — appended last — fires ``ParticipantNotifier`` only once
        the files are already on disk.

        ``link_to_index`` on a spec is resolved to the assigned version of the
        spec at that index, so callers can point attachments at the message
        without knowing the version in advance (it is assigned here, under the
        lock — never guessed).

        Returns the created entries in the same order as ``specs``.
        """
        if not specs:
            return []

        async with self._lock:
            base_version = self.get_current_version()

            # First pass: assign versions so link_to_index can resolve to a
            # sibling's final version.
            assigned_versions = [base_version + 1 + i for i in range(len(specs))]

            entries: list[ChangelogEntry] = []
            for i, spec in enumerate(specs):
                if spec.link_to_index is not None:
                    source_version = assigned_versions[spec.link_to_index]
                else:
                    source_version = spec.source_version
                entries.append(
                    ChangelogEntry(
                        version=assigned_versions[i],
                        entry_type=spec.entry_type,
                        payload=spec.payload,
                        source_version=source_version,
                        mirrored_from=spec.mirrored_from,
                    )
                )

            # Persist all, then process the whole batch in version order.
            await self._persist_entries(entries)
            self._pending_entries.extend(entries)
            await self._process_queue_internal()

            return entries

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
        # **The dominant call is `since == tip`, and it now costs one stat.**
        # A runner's reconnect catch-up asks this of every project it is in, and
        # almost always has nothing to fetch; parsing the file to discover that
        # is what made a reconnect storm self-sustaining.
        if since_version >= self.get_current_version():
            return []
        changelog_path = self._changelog_dir / "changelog.ndjson"
        return [e for e in iter_entries(changelog_path) if e.version > since_version]

    def get_entries_by_source_version(
        self,
        source_version: int,
    ) -> list[ChangelogEntry]:
        """Return all entries whose ``source_version`` matches (single pass).

        Used by the tunnel to gather a message's sibling attachments (the FILE_*
        entries appended in the same atomic batch, which carry
        ``source_version == message.version``).
        """
        # **A full scan on purpose — do not "optimise" it into a tail scan.**
        # ``append_batch`` gives the FILE entries of a batch LOWER versions than
        # the MESSAGE that links them (``link_to_index``), so
        # ``source_version > version`` is the normal shape and a backwards scan
        # stopping at ``version == source_version`` would miss exactly the
        # attachments this exists to find. Streamed, so it retains nothing.
        changelog_path = self._changelog_dir / "changelog.ndjson"
        return [
            e for e in iter_entries(changelog_path)
            if e.source_version == source_version
        ]

    def get_current_version(self) -> int:
        """Latest version number, or 0 when there are no entries.

        **Was the single most expensive call on the server.** It read the whole
        file, ``.strip()``-ed it into a second full copy and split that, to look
        at one line: 13.6 s on a 1.2 GB changelog — slower than fully parsing
        it — while ``append``, ``append_batch``, ``sync`` and a per-project loop
        in ``GET /participants/{id}/projects`` all call it. A tail read answers
        in ~4 ms; the cache below makes the repeat free.

        Two layers rather than either one: a bare tail read still costs ~0.6 s
        across 1300 projects on the list path, and a bare cache can neither
        start cold nor notice the out-of-band writes that tests and migration
        scripts perform.
        """
        changelog_path = self._changelog_dir / "changelog.ndjson"
        try:
            st = os.stat(changelog_path)
        except (FileNotFoundError, NotADirectoryError):
            self._tip_cache = None
            return 0
        cached = self._tip_cache
        if cached is not None and cached[:3] == (st.st_ino, st.st_size, st.st_mtime_ns):
            return cached[3]
        result = read_tip_version_and_stat(changelog_path)
        if result is None:
            self._tip_cache = None
            return 0
        version, read_st = result
        # Keyed on the stat taken INSIDE the read, on its own descriptor — see
        # that function's docstring for why a separate stat is unsafe here.
        self._tip_cache = (read_st.st_ino, read_st.st_size, read_st.st_mtime_ns, version)
        return version

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
                ndjson_safe(entry.model_dump_json(by_alias=True)) + "\n",
                "text",
                mode="a",
            )

        # **Invalidate; never set it to ``entries[-1].version``.** Assigning the
        # value we just wrote would assert that nothing else appends to this
        # file — and a migration script or a test that does would then hand the
        # next ``append`` a stale tip and duplicate a version. Invalidating
        # costs one tail read (~4 ms worst case) and assumes nothing.
        self._tip_cache = None

    async def _save_state_internal(self) -> None:
        """Save runloop state. Lock must be held."""
        state_path = self._changelog_dir / "runloop_state.json"

        state = {
            "project_id": self._project_id,
            "last_processed_version": self._last_processed_version,
        }

        FileUtil.write(state_path, state, "json", atomic=True)

    async def _load_pending_entries(self) -> None:
        """Load unprocessed entries from changelog. Lock must be held.

        **This, not ``get_current_version``, is what read 2.1 GB during a
        reconnect storm.** Every ``get_or_create`` for a project carrying a
        ``runloop_state.json`` came through here — and all of them do, so the
        old ``exists()`` check never short-circuited — to parse the entire file
        and, on a cleanly shut down server, append nothing.

        The guard below is the whole fix, and it introduces no new assumption:
        the file's tip IS its maximum version (append-only, ascending), which is
        what ``get_current_version`` has always relied on.
        """
        changelog_path = self._changelog_dir / "changelog.ndjson"
        if not changelog_path.exists():
            return

        if self.get_current_version() <= self._last_processed_version:
            return  # caught up — the steady state, and now free

        # Genuine crash recovery. Off the loop: a single-process server must not
        # stall every other request (and every WebSocket handshake) while it
        # replays. Safe under ``self._lock`` — the thread only reads the file.
        floor = self._last_processed_version
        self._pending_entries.extend(
            await asyncio.to_thread(
                lambda: [e for e in iter_entries(changelog_path) if e.version > floor]
            )
        )
        self._pending_entries.sort(key=lambda e: e.version)

    def __repr__(self) -> str:
        return (
            f"ChangelogRunloop("
            f"project={self._project_name}-{self._project_id[:8]}, "
            f"version={self._last_processed_version})"
        )
