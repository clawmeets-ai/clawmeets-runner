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

from clawmeets.sync.runloop import ChangelogRunloop
from clawmeets.sync.subscriber import ChangelogSubscriber

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
        self._lock = asyncio.Lock()

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
            coordinator_id: The project coordinator's ID (used by git sandbox)

        Returns:
            ChangelogRunloop instance for the project
        """
        async with self._lock:
            if project_id in self._runloops:
                return self._runloops[project_id]

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
            self._runloops[project_id] = runloop

            logger.debug(
                f"Created runloop for project {project_name}-{project_id[:8]}, "
                f"last_version={runloop.last_processed_version}"
            )

            return runloop

    async def remove(self, project_id: str) -> None:
        """Remove a project's runloop from the registry.

        Saves state before removal. Use when a project is being deleted.
        """
        async with self._lock:
            runloop = self._runloops.pop(project_id, None)
            if runloop:
                try:
                    await runloop.save_state()
                except Exception as e:
                    logger.warning(f"Failed to save state for deleted project {project_id}: {e}")
                logger.info(f"Removed runloop for project {project_id[:8]}")

    async def shutdown(self) -> None:
        """Save state for all runloops and clear registry.

        Call this on graceful shutdown.
        """
        async with self._lock:
            for project_id, runloop in self._runloops.items():
                try:
                    await runloop.save_state()
                    logger.debug(f"Saved state for project {project_id[:8]}")
                except Exception as e:
                    logger.error(f"Failed to save state for project {project_id}: {e}")

            self._runloops.clear()
            logger.info("ChangelogRunloopManager shutdown complete")

    def __len__(self) -> int:
        """Return number of active runloops."""
        return len(self._runloops)

    def __repr__(self) -> str:
        return f"ChangelogRunloopManager(model_ctx={self._model_ctx!r}, projects={len(self)})"
