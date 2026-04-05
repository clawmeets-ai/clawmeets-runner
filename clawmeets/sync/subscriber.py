# SPDX-License-Identifier: MIT
"""
clawmeets/sync/subscribe.py
Changelog subscription infrastructure.

This module is part of Layer 0 (pure - minimal domain dependencies).
It provides the subscriber interface for single-entry processing.

Filtering is done server-side in the API route.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .changelog import ChangelogEntry


# ---------------------------------------------------------------------------
# ChangelogSubscriber Interface
# ---------------------------------------------------------------------------

class ChangelogSubscriber(ABC):
    """
    Interface for entities that receive changelog entries.

    Subscribers are processed in insertion order (the order they were added
    to the ChangelogRunloop via add_subscriber). Each entry is processed
    through ALL subscribers before the next entry is processed.

    This ensures that for any given entry version:
    - All subscribers see entries in version order
    - When a higher-priority subscriber (e.g., ParticipantNotifier) processes
      an entry, lower-priority subscribers (e.g., ModelContext) have already
      finished processing that same entry
    """

    @abstractmethod
    async def on_entry(
        self,
        entry: "ChangelogEntry",
        project_id: str,
        project_name: str,
    ) -> None:
        """
        Process a single changelog entry.

        Args:
            entry: The changelog entry to process
            project_id: The project ID
            project_name: The project name (for path resolution)
        """
        pass

    async def on_sync_complete(
        self,
        project_id: str,
        project_name: str,
    ) -> None:
        """Called after all entries in a sync batch are processed.

        Default no-op. Override to perform batch-level operations
        like git commit after multiple file updates.

        Args:
            project_id: The project ID
            project_name: The project name
        """
        pass
