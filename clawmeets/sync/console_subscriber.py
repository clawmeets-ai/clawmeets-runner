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
    # Entry Handlers
    # ─────────────────────────────────────────────────────────

    def _handle_message(
        self,
        entry: "ChangelogEntry",
        project_id: str,
        project_name: str,
    ) -> None:
        """Handle MESSAGE entry."""
        payload = entry.payload

        # Get display info
        from_name = payload.from_participant_name or payload.from_participant_id[:8] if payload.from_participant_id else "unknown"

        timestamp = self._timestamp(payload.ts)
        room = self._room_name(payload.chatroom_name)
        agent = self._agent_name(from_name)
        preview = self._truncate(payload.content)

        self._print(f"{timestamp} {room} {agent}: {preview}")

    def _handle_file_created(
        self,
        entry: "ChangelogEntry",
        project_id: str,
        project_name: str,
    ) -> None:
        """Handle FILE_CREATED entry."""
        payload = entry.payload
        filename = payload.filename or "unknown"

        sync_label = self._c(Colors.CYAN, "[Sync]")
        room = self._room_name(payload.chatroom_name)
        file_text = self._c(Colors.BOLD, filename)

        self._print(f"{sync_label} {room} File created: {file_text}")

    def _handle_file_updated(
        self,
        entry: "ChangelogEntry",
        project_id: str,
        project_name: str,
    ) -> None:
        """Handle FILE_UPDATED entry."""
        payload = entry.payload
        filename = payload.filename or "unknown"

        sync_label = self._c(Colors.CYAN, "[Sync]")
        room = self._room_name(payload.chatroom_name)
        file_text = self._c(Colors.BOLD, filename)

        self._print(f"{sync_label} {room} File synced: {file_text}")

    def _handle_room_created(
        self,
        entry: "ChangelogEntry",
        project_id: str,
        project_name: str,
    ) -> None:
        """Handle ROOM_CREATED entry."""
        # Print participants joining when room is created
        payload = entry.payload
        plus_label = self._c(Colors.BLUE, "[+]")
        room = self._room_name(payload.chatroom_name)

        for participant in payload.participants:
            agent_name = participant.name or participant.id[:8]
            agent = self._agent_name(agent_name)
            self._print(f"{plus_label} {room} {agent} joined")

    def _handle_project_completed(
        self,
        entry: "ChangelogEntry",
        project_id: str,
        project_name: str,
    ) -> None:
        """Handle PROJECT_COMPLETED entry."""
        self._print("")
        green_bold = Colors.GREEN + Colors.BOLD if self._config.colors else ""
        nc = Colors.NC if self._config.colors else ""
        self._print(f"{green_bold}=== Project '{project_name}' Completed! ==={nc}")

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
        try:
            match entry.entry_type:
                case ChangelogEntryType.MESSAGE:
                    self._handle_message(entry, project_id, project_name)
                case ChangelogEntryType.FILE_CREATED:
                    self._handle_file_created(entry, project_id, project_name)
                case ChangelogEntryType.FILE_UPDATED:
                    self._handle_file_updated(entry, project_id, project_name)
                case ChangelogEntryType.ROOM_CREATED:
                    self._handle_room_created(entry, project_id, project_name)
                case ChangelogEntryType.PROJECT_COMPLETED:
                    self._handle_project_completed(entry, project_id, project_name)
                case _:
                    pass  # Ignore other entry types
        except Exception as e:
            logger.warning(f"Failed to format changelog entry: {e}")

    async def on_first_message(
        self,
        entry: "ChangelogEntry",
        project_id: str,
        project_name: str,
    ) -> None:
        """Process first message - same as regular message."""
        await self.on_entry(entry, project_id, project_name)
