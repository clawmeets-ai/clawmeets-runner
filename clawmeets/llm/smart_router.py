# SPDX-License-Identifier: MIT
"""Health-aware, replay-safe routing across configured LLM providers.

The router deliberately owns only provider selection.  Prompt construction,
filesystem visibility, skills, MCP, memory, and action validation remain in the
shared runner path and are passed to every selected provider unchanged.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .base import (
    LLMInvocationError,
    LLMNotFoundError,
    LLMProvider,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ..utils.notification_center import NotificationCenter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SmartRoute:
    """One named route, ordered by its position in the configured route list."""

    name: str
    provider: str
    model: Optional[str]
    build: Callable[[], LLMProvider]


@dataclass(frozen=True)
class RoutingDecision:
    """Content-free decision record suitable for operator logs and tests."""

    route: str
    outcome: str
    reason: str


def _safe_to_replay(error: BaseException) -> bool:
    """Whether a failed invocation proves no model-directed effect ran.

    Missing backends, quota/rate limits, and authentication/payment failures
    happen before a model can issue a tool call.  Timeouts, server failures and
    malformed responses are intentionally excluded: they may arrive after a
    provider already performed a non-idempotent tool action.
    """

    if isinstance(error, (LLMNotFoundError, LLMRateLimitError)):
        return True
    if isinstance(error, LLMTimeoutError):
        return False
    status = getattr(error, "status_code", None)
    return status in {401, 402, 403, 429}


class SmartRouterProvider(LLMProvider):
    """Try capable configured routes in order, with bounded safe fallback."""

    _provider_name = "Smart router"
    _log_tag = "smart-router"
    _install_hint = "Configure at least one non-smart model config"

    def __init__(
        self,
        routes: list[SmartRoute],
        *,
        max_attempts: int = 3,
        cooldown_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not routes:
            raise LLMNotFoundError("smart", self._install_hint)
        if max_attempts < 1:
            raise ValueError("smart_max_attempts must be at least 1")
        if cooldown_seconds < 0:
            raise ValueError("smart_cooldown_seconds cannot be negative")
        self._routes = tuple(routes)
        self._max_attempts = max_attempts
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._cooldowns: dict[str, float] = {}
        self.last_decisions: tuple[RoutingDecision, ...] = ()

    @classmethod
    def verify_cli(cls) -> None:
        """Individual routes are verified lazily so unavailable ones can fall back."""

    def _available_routes(self) -> list[SmartRoute]:
        now = self._clock()
        available: list[SmartRoute] = []
        decisions: list[RoutingDecision] = []
        for route in self._routes:
            until = self._cooldowns.get(route.name, 0.0)
            if until > now:
                decisions.append(RoutingDecision(route.name, "skipped", "cooldown"))
            else:
                available.append(route)
        self.last_decisions = tuple(decisions)
        return available[: self._max_attempts]

    def _record(self, route: SmartRoute, outcome: str, reason: str) -> None:
        self.last_decisions = (*self.last_decisions, RoutingDecision(route.name, outcome, reason))
        logger.info(
            "[smart-router] route=%s provider=%s model=%s outcome=%s reason=%s",
            route.name,
            route.provider,
            route.model or "default",
            outcome,
            reason,
        )

    def _cool(self, route: SmartRoute) -> None:
        self._cooldowns[route.name] = self._clock() + self._cooldown_seconds

    async def invoke(
        self,
        prompt: str,
        working_dir: Path,
        log_dir: Path,
        additional_dirs: list[Path],
        notification_center: NotificationCenter,
        action_schema: dict,
        trigger_version: int,
        mcp_config_dir: Optional[Path] = None,
        skill_source_dirs: Optional[list[Path]] = None,
        memory_dir: Optional[Path] = None,
    ):
        routes = self._available_routes()
        if not routes:
            raise LLMInvocationError("Smart routing exhausted: every configured route is cooling down")

        last_error: Optional[BaseException] = None
        for route in routes:
            try:
                provider = route.build()
                # The runner applies its configured kill window to this router;
                # each lazily-built child must inherit the same boundary.
                provider._invoke_timeout = self._invoke_timeout
                result = await provider.invoke(
                    prompt,
                    working_dir,
                    log_dir,
                    additional_dirs,
                    notification_center,
                    action_schema,
                    trigger_version,
                    mcp_config_dir=mcp_config_dir,
                    skill_source_dirs=skill_source_dirs,
                    memory_dir=memory_dir,
                )
                self._cooldowns.pop(route.name, None)
                self._record(route, "selected", "success")
                return result
            except LLMInvocationError as error:
                last_error = error
                safe = _safe_to_replay(error)
                self._cool(route)
                self._record(
                    route,
                    "failed",
                    f"{type(error).__name__}:{'fallback' if safe else 'replay-unsafe'}",
                )
                if not safe:
                    raise

        raise LLMInvocationError(
            "Smart routing exhausted all eligible routes; "
            f"last failure was {type(last_error).__name__ if last_error else 'unknown'}"
        ) from last_error

    async def generate_text(self, prompt: str, *, max_tokens: int = 32) -> str:
        """One-shot text generation has no tools, so provider fallback is replay-safe."""

        last_error: Optional[BaseException] = None
        for route in self._available_routes():
            try:
                provider = route.build()
                provider._invoke_timeout = self._invoke_timeout
                text = await provider.generate_text(prompt, max_tokens=max_tokens)
                self._record(route, "selected", "success")
                return text
            except LLMInvocationError as error:
                last_error = error
                self._cool(route)
                self._record(route, "failed", f"{type(error).__name__}:fallback")
        raise LLMInvocationError("Smart routing exhausted all eligible text routes") from last_error
