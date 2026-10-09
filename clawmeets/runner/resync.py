# SPDX-License-Identifier: MIT
"""
clawmeets/runner/resync.py

Client half of RESYNC (``api/control.ResyncPayload``): the server's request to
"catch up as if you had just reconnected", sent when core restarted behind a
websocket edge that kept the socket open.

``ResyncScheduler`` owns the timing so the socket loops stay transport-only:
each request waits ``uniform(0, jitter_s)`` so a whole deployment does not
catch up in the same second, and requests coalesce — at most one catch-up is
pending or running at a time. A RESYNC arriving while one is pending is folded
into it (``full`` wins over ``changelog``); one arriving while a catch-up is
running is queued and runs after it, since the running one may already be past
the part the new request is about.

Used by the agent runner and by ``clawmeets user listen``; the browser has its
own copy of the same rule (``web/frontend/src/api/websocket.ts``).
"""
from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Literal, Optional

from ..api.control import ResyncPayload

logger = logging.getLogger(__name__)

ResyncScope = Literal["full", "changelog"]


class ResyncScheduler:
    """Run one coalesced, jittered catch-up per burst of RESYNC requests.

    ``on_full`` / ``on_changelog`` are the two catch-ups. They run on the
    scheduler's own task, never the socket's receive loop, and any exception
    they raise is logged and swallowed: a failed resync must never kill the
    runner, and the next RESYNC or reconnect retries it.

    ``sleep`` and ``uniform`` are injectable for tests.
    """

    def __init__(
        self,
        *,
        on_full: Callable[[], Awaitable[None]],
        on_changelog: Callable[[], Awaitable[None]],
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        uniform: Callable[[float, float], float] = random.uniform,
        name: str = "resync",
    ) -> None:
        self._on_full = on_full
        self._on_changelog = on_changelog
        self._sleep = sleep
        self._uniform = uniform
        self._name = name
        # The scope of the next catch-up, or None when nothing is queued. Read
        # (and cleared) only after the jitter sleep, so a request arriving
        # during the sleep is folded into the one already pending.
        self._pending_scope: Optional[ResyncScope] = None
        self._jitter_s: float = 0.0
        self._task: Optional[asyncio.Task] = None

    @property
    def busy(self) -> bool:
        """True while a catch-up is pending or running."""
        return self._task is not None and not self._task.done()

    def request(self, payload: ResyncPayload) -> None:
        """Queue a catch-up for one RESYNC. Never blocks, never raises."""
        if self._pending_scope != "full":
            self._pending_scope = payload.scope
        self._jitter_s = max(0.0, payload.jitter_s)
        if not self.busy:
            self._task = asyncio.create_task(self._run())

    def cancel(self) -> None:
        """Drop any pending or running catch-up.

        Called when the socket closes: the reconnect that follows runs the
        full catch-up itself, so finishing this one would only run it twice,
        concurrently.
        """
        self._pending_scope = None
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    async def wait(self) -> None:
        """Await the current catch-up, if any (tests and shutdown)."""
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self) -> None:
        while self._pending_scope is not None:
            await self._sleep(self._uniform(0.0, self._jitter_s))
            scope, self._pending_scope = self._pending_scope, None
            if scope is None:
                return
            try:
                if scope == "full":
                    await self._on_full()
                else:
                    await self._on_changelog()
            except Exception:  # noqa: BLE001 — never kill the runner
                logger.warning(
                    f"{self._name}: {scope} resync failed; the next RESYNC or "
                    "reconnect retries it",
                    exc_info=True,
                )
