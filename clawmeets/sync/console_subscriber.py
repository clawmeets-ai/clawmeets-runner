# SPDX-License-Identifier: MIT
"""
clawmeets/sync/console_subscriber.py
Console output subscriber for changelog events.

This module provides a subscriber that prints formatted changelog events
to the console with ANSI colors.
"""
from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from .changelog import ChangelogEntryType
from .subscriber import ChangelogSubscriber

if TYPE_CHECKING:
    from .changelog import ChangelogEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ANSI Colors
# ---------------------------------------------------------------------------

class Colors:
    """ANSI color codes for console output."""
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    CYAN = '\033[0;36m'
    RED = '\033[0;31m'
    GRAY = '\033[0;90m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color / Reset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ConsoleConfig:
    """Configuration for console output."""
    colors: bool = True
    timestamps: bool = True
    output_stream: object = field(default_factory=lambda: sys.stdout)


# ---------------------------------------------------------------------------
# ConsoleOutputSubscriber
# ---------------------------------------------------------------------------

class ConsoleOutputSubscriber(ChangelogSubscriber):
    """
    Subscriber that prints formatted changelog events to the console.

    Output format:
    - [HH:MM:SS] #room-name agent-name: message content
    - [Sync] #room-name File synced: filename.md
    - [+] #room-name agent-name joined

    Can be placed between ModelContext and ParticipantNotifier in the
    subscriber chain to provide real-time console output.
    """

    def __init__(self, config: ConsoleConfig) -> None:
        """Initialize the console subscriber.

        Args:
            config: Console output configuration
        """
        self._config = config
        self._entry_counts: dict[str, int] = {}

    # ─────────────────────────────────────────────────────────
    # Color Helpers
    # ─────────────────────────────────────────────────────────

    def _c(self, color: str, text: str) -> str:
        """Apply color if colors are enabled."""
        if self._config.colors:
            return f"{color}{text}{Colors.NC}"
        return text

    def _timestamp(self, ts: datetime) -> str:
        """Format timestamp for display."""
        if not self._config.timestamps:
            return ""
        time_str = ts.strftime("%H:%M:%S")
        return self._c(Colors.GRAY, f"[{time_str}]")

    def _room_name(self, room_name: str) -> str:
        """Get formatted room name."""
        return self._c(Colors.YELLOW, f"#{room_name}")

    def _agent_name(self, name: str) -> str:
        """Format agent name with color."""
        return self._c(Colors.GREEN, "") + self._c(Colors.BOLD, name)

    def _truncate(self, text: str, max_len: int = 99999) -> str:
        """Truncate text and replace newlines with spaces."""
        # Replace newlines with spaces
        oneline = text.replace('\n', ' ').replace('\r', '')
        # Collapse multiple spaces
        oneline = re.sub(r'\s+', ' ', oneline).strip()
        if len(oneline) > max_len:
            return oneline[:max_len] + "..."
        return oneline

    # ─────────────────────────────────────────────────────────
    # Output Methods
    # ─────────────────────────────────────────────────────────

    def _print(self, message: str) -> None:
        """Print message to output stream."""
        print(message, file=self._config.output_stream, flush=True)

    # ─────────────────────────────────────────────────────────
    # ChangelogSubscriber Interface
    # ─────────────────────────────────────────────────────────

    async def on_entry(
        self,
        entry: "ChangelogEntry",
        project_id: str,
        project_name: str,
    ) -> None:
        """Process a changelog entry and print formatted output."""
        self._entry_counts[project_id] = self._entry_counts.get(project_id, 0) + 1
        payload = entry.payload
        ts = self._timestamp(entry.timestamp)
        try:
            match entry.entry_type:
                case ChangelogEntryType.MESSAGE:
                    from_name = payload.from_participant_name or (payload.from_participant_id[:8] if payload.from_participant_id else "unknown")
                    self._print(
                        f"{self._timestamp(payload.ts)} {self._room_name(payload.chatroom_name)} "
                        f"{self._agent_name(from_name)}: {self._truncate(payload.content)}"
                    )
                case ChangelogEntryType.FILE_CREATED:
                    self._print(
                        f"{self._c(Colors.CYAN, '[Sync]')} {self._room_name(payload.chatroom_name)} "
                        f"File created: {self._c(Colors.BOLD, payload.filename or 'unknown')}"
                    )
                case ChangelogEntryType.FILE_UPDATED:
                    self._print(
                        f"{self._c(Colors.CYAN, '[Sync]')} {self._room_name(payload.chatroom_name)} "
                        f"File synced: {self._c(Colors.BOLD, payload.filename or 'unknown')}"
                    )
                case ChangelogEntryType.ROOM_CREATED:
                    plus = self._c(Colors.BLUE, "[+]")
                    room = self._room_name(payload.chatroom_name)
                    for p in payload.participants:
                        self._print(f"{plus} {room} {self._agent_name(p.name or p.id[:8])} joined")
                case ChangelogEntryType.PROJECT_COMPLETED:
                    self._print("")
                    green_bold = Colors.GREEN + Colors.BOLD if self._config.colors else ""
                    nc = Colors.NC if self._config.colors else ""
                    self._print(f"{green_bold}=== Project '{project_name}' Completed! ==={nc}")
                case ChangelogEntryType.PROJECT_CREATED:
                    self._print(
                        f"{ts} {self._c(Colors.BLUE, '[+]')} Project created: "
                        f"{self._c(Colors.BOLD, payload.project_name)} "
                        f"(coordinator: {self._agent_name(payload.coordinator_name)})"
                    )
                case ChangelogEntryType.ROOM_DELETED:
                    self._print(f"{ts} {self._c(Colors.RED, '[-]')} {self._room_name(payload.chatroom_name)} deleted")
                case ChangelogEntryType.PROJECT_REACTIVATED:
                    self._print(f"{ts} Project '{self._c(Colors.BOLD, project_name)}' reactivated")
                case ChangelogEntryType.BATCH_COMPLETE:
                    self._print(
                        f"{ts} {self._c(Colors.CYAN, '[Batch]')} {self._room_name(payload.chatroom_name)} "
                        f"complete (responded: {len(payload.responded_participants)})"
                    )
                case ChangelogEntryType.BATCH_TIMEOUT:
                    self._print(
                        f"{ts} {self._c(Colors.RED, '[Batch]')} {self._room_name(payload.chatroom_name)} "
                        f"timeout (responded: {len(payload.responded_participants)}, "
                        f"timed out: {len(payload.timed_out_participants)})"
                    )
                case ChangelogEntryType.PARTICIPANT_ADDED:
                    name = payload.participant_name or payload.participant_id[:8]
                    self._print(
                        f"{ts} {self._c(Colors.BLUE, '[+]')} {self._room_name(payload.chatroom_name)} "
                        f"{self._agent_name(name)} joined"
                    )
                case ChangelogEntryType.CHATROOM_CLEARED:
                    self._print(
                        f"{ts} {self._room_name(payload.chatroom_name)} "
                        f"cleared by {payload.cleared_by_participant_id[:8]} "
                        f"(archive: {payload.archive_filename or 'empty'})"
                    )
                case _:
                    logger.debug(f"Unhandled changelog entry type: {entry.entry_type}")
        except Exception as e:
            logger.warning(f"Failed to format changelog entry: {e}")

    async def on_sync_complete(
        self,
        project_id: str,
        project_name: str,
    ) -> None:
        """Emit a one-line summary at the end of each sync batch."""
        count = self._entry_counts.pop(project_id, 0)
        sync_label = self._c(Colors.CYAN, "[Sync]")
        name = self._c(Colors.BOLD, project_name)
        self._print(f"{sync_label} {name} sync complete ({count} entries)")

    async def on_first_message(
        self,
        entry: "ChangelogEntry",
        project_id: str,
        project_name: str,
    ) -> None:
        """Process first message - same as regular message."""
        await self.on_entry(entry, project_id, project_name)
