# SPDX-License-Identifier: MIT
"""
clawmeets/sync/runloop_manager.py
Registry for per-project ChangelogRunloop instances.

Creates runloops on demand and manages their lifecycle.
Used by both server and runner for per-project changelog processing.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable

from .runloop import ChangelogRunloop
from .subscriber import ChangelogSubscriber

logger = logging.getLogger(__name__)

# Factory called per-project: (project_id, project_name, coordinator_id) -> (changelog_dir, subscribers)
RunloopFactory = Callable[[str, str, str], tuple[Path, list[ChangelogSubscriber]]]


class ChangelogRunloopManager:
    """Registry for per-project ChangelogRunloop instances.

    Creates runloops on demand, manages lifecycle.
    Used by both server and runner.

    Callers provide a ``runloop_factory`` that, given a project, returns the
    changelog directory and the full ordered list of subscribers for that
    project's runloop.  This keeps the manager free of domain knowledge
    (ModelContext, git sandbox, etc.).

    Usage:
        # Server
        def make_runloop(pid, pname, cid):
            return model_ctx.changelog_dir(pid, pname), [
                model_ctx.changelog_subscriber(pid, pname),
            ]
        manager = ChangelogRunloopManager(runloop_factory=make_runloop)

        # Runner
        def make_runloop(pid, pname, coordinator_id):
            subs = [model_ctx.changelog_subscriber(pid, pname)]
            # ... add git sandbox, notifier, etc.
            return model_ctx.changelog_dir(pid, pname), subs
        manager = ChangelogRunloopManager(runloop_factory=make_runloop)

        # Get or create runloop for a project
        runloop = await manager.get_or_create("abc123", "my-project")
    """

    def __init__(self, runloop_factory: RunloopFactory) -> None:
        """Initialize the manager.

        Args:
            runloop_factory: Called per-project with (project_id, project_name, coordinator_id).
                Returns (changelog_dir, subscribers) for the new runloop.
        """
        self._runloop_factory = runloop_factory
        self._runloops: dict[str, ChangelogRunloop] = {}
        # Guards ONLY the _runloops/_create_locks dicts. Never held across an
        # awaited runloop operation (load_state/save_state/processing) — doing
        # so deadlocks against any runloop whose subscriber re-enters
        # get_or_create from inside on_entry (TunnelSubscriber mirroring).
        self._lock = asyncio.Lock()
        # Per-project creation locks: serialize concurrent first-creates of the
        # SAME project without serializing different projects (and without
        # holding the global lock across load_state()).
        self._create_locks: dict[str, asyncio.Lock] = {}

    async def get_or_create(
        self,
        project_id: str,
        project_name: str,
        coordinator_id: str = "",
    ) -> ChangelogRunloop:
        """Get existing or create new runloop for project.

        Args:
            project_id: The project ID
            project_name: The project name
            coordinator_id: The project coordinator's ID (passed to the runloop
                factory; retained for the factory contract)

        Returns:
            ChangelogRunloop instance for the project
        """
        # Fast path: return the cached runloop. The global lock is held only
        # for the dict lookup — never across load_state() below.
        async with self._lock:
            existing = self._runloops.get(project_id)
            if existing is not None:
                return existing
            create_lock = self._create_locks.get(project_id)
            if create_lock is None:
                create_lock = asyncio.Lock()
                self._create_locks[project_id] = create_lock

        # Serialize first-creation of THIS project only. load_state() runs
        # OUTSIDE the global lock, so a slow/stuck runloop can never block
        # lookups for other projects, and remove()'s teardown can proceed even
        # while this awaits — breaking the manager-lock ↔ runloop-lock cycle.
        async with create_lock:
            async with self._lock:
                existing = self._runloops.get(project_id)
                if existing is not None:
                    return existing

            # Ask the factory for changelog dir and subscribers
            changelog_dir, subscribers = self._runloop_factory(
                project_id, project_name, coordinator_id,
            )

            runloop = ChangelogRunloop(
                project_id=project_id,
                project_name=project_name,
                changelog_dir=changelog_dir,
            )

            for subscriber in subscribers:
                runloop.add_subscriber(subscriber)

            await runloop.load_state()

            async with self._lock:
                # A concurrent caller may have published one while we loaded;
                # keep a single instance per project.
                existing = self._runloops.get(project_id)
                if existing is not None:
                    return existing
                self._runloops[project_id] = runloop

            logger.debug(
                f"Created runloop for project {project_name}-{project_id[:8]}, "
                f"last_version={runloop.last_processed_version}"
            )

            return runloop

    async def remove(self, project_id: str) -> None:
        """Remove a project's runloop from the registry.

        Use when a project is being deleted. Does NOT persist runloop state:
        the project's on-disk data is destroyed immediately after (rmtree), so
        the save is pointless — and the old in-lock save_state() held the
        global manager lock across the runloop's own lock, which deadlocked
        against TunnelSubscriber re-entering get_or_create from inside a
        runloop's processing. Dict mutation only, no awaited runloop I/O.
        """
        async with self._lock:
            existed = self._runloops.pop(project_id, None) is not None
            self._create_locks.pop(project_id, None)
        if existed:
            logger.info(f"Removed runloop for project {project_id[:8]}")

    async def shutdown(self) -> None:
        """Save state for all runloops and clear registry.

        Call this on graceful shutdown.
        """
        async with self._lock:
            runloops = list(self._runloops.items())
            self._runloops.clear()
            self._create_locks.clear()

        # Persist OUTSIDE the global lock (same deadlock-avoidance rationale as
        # get_or_create/remove). Shutdown is terminal, but a runloop may still
        # be mid-processing and re-enter the manager.
        for project_id, runloop in runloops:
            try:
                await runloop.save_state()
                logger.debug(f"Saved state for project {project_id[:8]}")
            except Exception as e:
                logger.error(f"Failed to save state for project {project_id}: {e}")

        logger.info("ChangelogRunloopManager shutdown complete")

    def __len__(self) -> int:
        """Return number of active runloops."""
        return len(self._runloops)

    def __repr__(self) -> str:
        return f"ChangelogRunloopManager(projects={len(self)})"
