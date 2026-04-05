# SPDX-License-Identifier: MIT
"""
clawmeets/utils/notification_center.py
In-memory pub/sub message dispatcher.

Layer 0 utility — no dependencies on other clawmeets modules.

Usage:
    nc = NotificationCenter()
    nc.subscribe(LLM_COMPLETE, my_handler)
    await nc.publish(LLM_COMPLETE, sandbox_dir=path, usage=usage)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event name constants
# ---------------------------------------------------------------------------

LLM_COMPLETE = "llm_complete"  # kwargs: sandbox_dir: Path, usage: ClaudeUsage
LLM_ERROR = "llm_error"        # kwargs: sandbox_dir: Path, error: Exception


# ---------------------------------------------------------------------------
# NotificationCenter
# ---------------------------------------------------------------------------

class NotificationCenter:
    """In-memory pub/sub message dispatcher.

    Publishers call publish(event, **kwargs) to notify all subscribers.
    Subscribers register async callbacks via subscribe(event, callback).
    Callbacks are invoked in registration order.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[..., Awaitable[None]]]] = defaultdict(list)

    def subscribe(self, event: str, callback: Callable[..., Awaitable[None]]) -> None:
        """Register a callback for an event.

        Args:
            event: Event name (e.g. LLM_COMPLETE)
            callback: Async function to call when event fires
        """
        self._subscribers[event].append(callback)

    async def publish(self, event: str, **kwargs: Any) -> None:
        """Publish an event to all subscribers.

        Args:
            event: Event name
            **kwargs: Event-specific data passed to callbacks
        """
        for callback in self._subscribers[event]:
            try:
                await callback(**kwargs)
            except Exception:
                logger.error(
                    f"NotificationCenter: callback failed for event '{event}'",
                    exc_info=True,
                )
