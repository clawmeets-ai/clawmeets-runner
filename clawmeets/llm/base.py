# SPDX-License-Identifier: MIT
"""
clawmeets/llm/base.py
LLM provider abstraction — shared types and interface.

Layer 0 (pure — no domain model dependencies). Any LLM CLI backend
(Claude Code, OpenAI Codex, etc.) implements LLMProvider and raises
the generic LLM* exceptions defined here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..api.actions import ActionBlock
from ..utils.notification_center import NotificationCenter


# ---------------------------------------------------------------------------
# Exception Classes (generic, shared across providers)
# ---------------------------------------------------------------------------


class LLMInvocationError(Exception):
    """Base exception for LLM CLI invocation errors."""

    def __init__(
        self,
        message: str,
        prompt_file: Optional[str] = None,
        working_dir: Optional[str] = None,
    ) -> None:
        self.prompt_file = prompt_file
        self.working_dir = working_dir
        super().__init__(message)


class LLMTimeoutError(LLMInvocationError):
    """LLM CLI invocation timed out."""

    def __init__(
        self,
        timeout_seconds: int,
        prompt_file: Optional[str] = None,
        working_dir: Optional[str] = None,
        provider: str = "LLM",
    ) -> None:
        self.timeout_seconds = timeout_seconds
        message = f"{provider} invocation timed out after {timeout_seconds} seconds"
        if prompt_file:
            message += f". Prompt file: {prompt_file}"
        super().__init__(message, prompt_file, working_dir)


class LLMNotFoundError(LLMInvocationError):
    """LLM CLI binary not found on PATH."""

    def __init__(self, binary: str, install_hint: str = "") -> None:
        self.binary = binary
        message = f"'{binary}' not found on PATH."
        if install_hint:
            message += f" {install_hint}"
        super().__init__(message)


class LLMRateLimitError(LLMInvocationError):
    """LLM provider signaled a rate limit (should back off, not retry immediately)."""

    def __init__(
        self,
        message: str,
        resets_at: Optional[float] = None,
        rate_limit_type: Optional[str] = None,
        prompt_file: Optional[str] = None,
        working_dir: Optional[str] = None,
    ) -> None:
        self.resets_at = resets_at
        self.rate_limit_type = rate_limit_type
        super().__init__(message, prompt_file, working_dir)

    @property
    def resets_at_human(self) -> Optional[str]:
        """Human-readable reset time, if provided by the backend."""
        if self.resets_at is None:
            return None
        from datetime import datetime, timezone

        dt = datetime.fromtimestamp(self.resets_at, tz=timezone.utc)
        return dt.astimezone().strftime("%I:%M %p %Z")


# ---------------------------------------------------------------------------
# LLMUsage (generic — providers fill whatever fields they report)
# ---------------------------------------------------------------------------


@dataclass
class LLMUsage:
    """Usage stats from a single LLM invocation.

    Not every provider populates every field. Unreported fields are 0 / "".
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    model: str = ""

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        """Accumulate usage across invocations."""
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            duration_ms=self.duration_ms + other.duration_ms,
            model=other.model or self.model,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cost_usd": self.cost_usd,
            "duration_ms": self.duration_ms,
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# LLMProvider interface
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract base for LLM CLI subprocess wrappers.

    Implementations invoke a terminal agent (e.g. `claude`, `codex exec`) with
    a caller-provided prompt + JSON action schema, then parse the result into
    an `ActionBlock` and `LLMUsage`. On failure they raise one of the generic
    `LLM*` exceptions defined above.
    """

    # Default — providers that support a browser tool may override.
    _use_chrome: bool = False

    @property
    def use_chrome(self) -> bool:
        """Whether browser integration is enabled for this provider."""
        return self._use_chrome

    @use_chrome.setter
    def use_chrome(self, value: bool) -> None:
        """Update browser integration. Takes effect on the next invocation.

        Providers without a browser tool should leave this as a no-op override.
        """
        self._use_chrome = value

    @classmethod
    @abstractmethod
    def verify_cli(cls) -> None:
        """Verify the CLI is available.

        Raises:
            LLMNotFoundError: If the binary is not on PATH.
            LLMTimeoutError: If probe times out.
            LLMInvocationError: If the probe returns an error.
        """
        ...

    @abstractmethod
    async def invoke(
        self,
        prompt: str,
        working_dir: Path,
        log_dir: Path,
        additional_dirs: list[Path],
        notification_center: NotificationCenter,
    ) -> tuple[ActionBlock, LLMUsage]:
        """Invoke the LLM CLI with the given prompt.

        Args:
            prompt: The prompt to send to the model
            working_dir: Directory where the CLI will run (writable sandbox)
            log_dir: Directory for stdout/stderr logs
            additional_dirs: Extra directories the model should be able to read
                (project_dir when different from working_dir, knowledge bases)
            notification_center: Dispatcher for LLM_COMPLETE / LLM_ERROR events

        Returns:
            Tuple of (action_block, usage_stats). action_block.actions may be
            empty if the model returned no structured output.
        """
        ...
