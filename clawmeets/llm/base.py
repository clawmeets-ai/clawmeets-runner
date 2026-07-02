# SPDX-License-Identifier: MIT
"""
clawmeets/llm/base.py
LLM provider abstraction — shared types and interface.

Layer 0 (pure — no domain model dependencies). Any LLM CLI backend
(Claude Code, OpenAI Codex, etc.) implements LLMProvider and raises
the generic LLM* exceptions defined here.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from ..api.actions import ActionBlock
from ..utils.notification_center import LLM_COMPLETE, LLM_ERROR, NotificationCenter

logger = logging.getLogger(__name__)


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
        dt = datetime.fromtimestamp(self.resets_at, tz=UTC)
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
    # Telemetry: model round-trips this invocation, and per-tool call counts
    # (function tools + native/builtin tools, keyed by tool name). Best-effort —
    # providers populate them where the underlying output exposes them.
    requests: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        """Accumulate usage across invocations."""
        merged_tools: dict[str, int] = dict(self.tool_calls)
        for name, count in other.tool_calls.items():
            merged_tools[name] = merged_tools.get(name, 0) + count
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            duration_ms=self.duration_ms + other.duration_ms,
            model=other.model or self.model,
            requests=self.requests + other.requests,
            tool_calls=merged_tools,
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
            "requests": self.requests,
            "tool_calls": self.tool_calls,
        }


# ---------------------------------------------------------------------------
# Per-invocation data carriers (Template Method hooks)
# ---------------------------------------------------------------------------


def link_mcp_config_into(source: Path, target: Path) -> bool:
    """Idempotently symlink ``source`` → ``target``.

    Returns ``True`` if a link now points at the source, ``False`` if the
    source doesn't exist (in which case any stale target is removed so a
    previous run's link doesn't survive an uninstall).

    Used by each LLM provider's ``_prepare_invocation`` to surface the
    per-provider file from the agent's ``mcp-hub/dist/`` into the
    per-invocation working dir. Replaces any existing target — symlinks,
    plain files, or broken links — so callers can re-run safely after
    ``McpManager.render_dist`` rewrites the source.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    if not source.exists():
        return False
    target.symlink_to(source)
    return True


def materialize_skill_tree(target_dir: Path, source_dirs: list[Path]) -> None:
    """Recreate ``target_dir`` as a flat tree of ``<name> → source/<name>``
    symlinks.

    Walks each source dir in order. Any subdir containing ``SKILL.md`` is
    surfaced as ``target_dir/<name>`` symlinked at the source. Later
    sources win on name collisions, so callers should order
    ``source_dirs`` as ``[platform-curated, personal]`` — the agent's
    own learned version overrides the curated default.

    Idempotent: wipes ``target_dir`` and recreates from scratch on each
    call. Used at invoke time by each LLM provider's
    ``_prepare_invocation`` so the CLI's native skill auto-discovery
    (Claude scans ``.claude/skills/``; Codex + Gemini scan
    ``.agents/skills/``) sees the agent's installed + personal skills as
    a single flat list at the cwd-relative path it already expects.
    """
    if target_dir.is_symlink() or target_dir.exists():
        if target_dir.is_dir() and not target_dir.is_symlink():
            shutil.rmtree(target_dir)
        else:
            target_dir.unlink()
    target_dir.mkdir(parents=True, exist_ok=True)

    chosen: dict[str, Path] = {}
    for src in source_dirs:
        if not src.is_dir():
            continue
        for entry in sorted(src.iterdir()):
            if not entry.is_dir():
                continue
            if not (entry / "SKILL.md").exists():
                continue
            chosen[entry.name] = entry  # later wins
    for name, src in chosen.items():
        (target_dir / name).symlink_to(src.resolve())


@dataclass
class PreparedInvocation:
    """Provider-built launch spec for a single subprocess invocation.

    `stdin_bytes is None` ⇒ stdin is wired to DEVNULL (Gemini); bytes ⇒ PIPE +
    fed via `proc.communicate(input=...)` (Claude, Codex).

    `extras` is provider-private scratch — e.g. Codex stashes the path of its
    `-o <last_message>` file so its `_parse_result` can read it back.
    """

    cmd: list[str]
    cwd: str
    prompt_file_abs: str
    stdin_bytes: Optional[bytes] = None
    extras: dict = field(default_factory=dict)


@dataclass
class ParsedResult:
    """Provider-parsed success-path outcome of a single invocation.

    Only built when both `_check_rate_limit` and `_build_error_detail` returned
    None and `returncode == 0` — i.e. the invocation completed cleanly.
    """

    usage: LLMUsage
    actions: list[dict]
    raw_text: str


# ---------------------------------------------------------------------------
# LLMProvider — Template Method base
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract base for any LLM backend — execution-model agnostic.

    Defines only the public contract every backend must honor:

    - ``invoke(...) -> (ActionBlock, LLMUsage)`` — run one agent turn and
      return the structured action block plus usage. Implementations MUST be
      cancellable (await points so the surrounding task can be cancelled) and
      MUST publish ``LLM_COMPLETE`` (on success) / ``LLM_ERROR`` (on failure)
      to the provided ``notification_center``.
    - ``verify_cli()`` — preflight that the backend is usable (binary on PATH
      for subprocess backends; SDK importable + key present for API backends).

    Two execution models implement this:

    - :class:`SubprocessLLMProvider` — shells a Code CLI (``claude`` / ``codex``
      / ``gemini``); the CLI harness provides file tools, skill discovery, and
      MCP natively.
    - the in-process API providers — drive the model HTTP APIs directly
      (BYO-key) and reimplement the harness (file/bash tools, skills, MCP) in
      the runner process, so the runner stays portable across local/hosted
      environments without the CLI binaries.

    Subclass identity attrs (``_provider_name`` / ``_log_tag`` /
    ``_install_hint``) are shared by both.
    """

    # Per-invocation kill window; aligned with server's --batch-timeout.
    _invoke_timeout: int = 1800

    # Subclass identity — override as class attributes.
    _provider_name: str = ""
    _log_tag: str = ""
    _install_hint: str = ""

    @classmethod
    @abstractmethod
    def verify_cli(cls) -> None:
        """Verify the backend is usable.

        Raises:
            LLMNotFoundError: If the backend is unavailable (binary not on
                PATH for subprocess backends; SDK/key missing for API backends).
            LLMTimeoutError: If a preflight probe times out.
            LLMInvocationError: If a preflight probe returns an error.
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
        action_schema: dict,
        trigger_version: int,
        mcp_config_dir: Optional[Path] = None,
        skill_source_dirs: Optional[list[Path]] = None,
    ) -> tuple[ActionBlock, LLMUsage]:
        """Run one agent turn and translate the result into an ActionBlock.

        Args:
            prompt: The prompt to send to the model.
            working_dir: Writable sandbox the agent operates in (cwd for
                subprocess backends / root of file tools for API backends).
            log_dir: Directory for per-invocation logs.
            additional_dirs: Extra dirs the model may read (project_dir when
                different from working_dir, knowledge bases, dwh).
            notification_center: Dispatcher for LLM_COMPLETE / LLM_ERROR.
            action_schema: JSON schema for the structured action output,
                selected per invocation by the caller (WORKER vs COORDINATOR).
            trigger_version: Changelog version that triggered this invocation;
                stamped onto ``ActionBlock.source_version``.
            mcp_config_dir: Optional ``{agent_dir}/mcp-hub/dist/`` pointer for
                installed MCP servers.
            skill_source_dirs: Optional skill-source dirs to surface to the
                model (installed + personal skill hubs).

        Returns:
            Tuple of (action_block, usage_stats). ``action_block.actions`` may
            be empty if the model returned no structured output.
        """
        ...


class SubprocessLLMProvider(LLMProvider):
    """Base for LLM CLI subprocess wrappers.

    Implements the shared subprocess workflow (spawn → communicate → reap →
    log → publish) as a final `invoke()`. Subclasses supply per-provider
    identity (`_provider_name`, `_log_tag`, `_install_hint`) plus two hooks:
    `_prepare_invocation` (build cmd + scratch files) and `_parse_result`
    (turn stdout/stderr into usage, actions, raw_text, and optional rate-limit
    or error_detail).

    Subclasses also keep `self._bin` and `self._agent_env`, set in `__init__`.
    """

    _bin: str
    _agent_env: dict[str, str]

    # --- abstract hooks -----------------------------------------------------

    @abstractmethod
    def _prepare_invocation(
        self,
        prompt: str,
        working_dir: Path,
        additional_dirs: list[Path],
        action_schema: dict,
        mcp_config_dir: Optional[Path] = None,
        skill_source_dirs: Optional[list[Path]] = None,
    ) -> PreparedInvocation:
        """Build the launch spec for a single invocation.

        Implementations write any per-invocation files (prompt, schema, sentinel
        outputs) under `working_dir` and return the cmd + cwd + stdin bytes
        plus any provider-private scratch the parser will need.

        ``mcp_config_dir`` — when set, points at ``{agent_dir}/mcp-hub/dist/``
        which holds the pre-rendered per-provider MCP configs (refreshed on
        MCP_SYNC, not per invocation). Subclasses surface the relevant file
        into their working dir (Claude: ``.mcp.json``; Gemini:
        ``.gemini/settings.json``; Codex: ``.codex/config.toml``) — typically
        via an idempotent symlink — so the spawned CLI auto-discovers the
        installed MCP servers from cwd.

        ``skill_source_dirs`` — list of skill-source directories (typically
        ``[skill-hub/skills, personal-skill-hub/skills]``). Subclasses call
        ``materialize_skill_tree`` to flatten these into the provider's
        native cwd skill-discovery path (Claude: ``.claude/skills``; Codex
        + Gemini: ``.agents/skills``) so each CLI's own auto-loader picks
        the skills up — the prompt no longer carries an INDEX.
        """
        ...

    @abstractmethod
    def _check_rate_limit(
        self,
        prepared: PreparedInvocation,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> Optional["LLMRateLimitError"]:
        """Return a partial `LLMRateLimitError` (no prompt_file / working_dir —
        the base attaches those) when the provider's rate-limit signature
        matches, otherwise `None`.

        Called first; takes precedence over `_build_error_detail` and the
        success path so rate limits never get retried with short backoff.
        """
        ...

    @abstractmethod
    def _build_error_detail(
        self,
        prepared: PreparedInvocation,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> Optional[str]:
        """Return a fully-formed error-detail string when the provider observed
        a logical error (envelope.error, error events, non-zero exit, etc.),
        otherwise `None`.

        Implementations merge stderr with any provider-specific bits in the
        order they want. The base still raises `LLMInvocationError` whenever
        `returncode != 0` even if this returns `None`, falling back to bare
        stderr / `"(no error detail)"` for the message body.
        """
        ...

    @abstractmethod
    def _parse_result(
        self,
        prepared: PreparedInvocation,
        stdout: str,
        stderr: str,
    ) -> ParsedResult:
        """Success-path parser. Returns usage, actions, raw_text.

        Only called after `_check_rate_limit` and `_build_error_detail` both
        returned `None` and `returncode == 0`.
        """
        ...

    # --- shared helpers (overridable) ---------------------------------------

    def _build_env(self) -> dict[str, str]:
        """Build the environment passed to the subprocess.

        Default: process env + `agent_env` (CLAWMEETS_AGENT_ID / TOKEN /
        SERVER_URL / AGENT_DIR). Subclasses override to add per-provider
        flags (e.g. ClaudeCLI sets CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD).
        """
        env = os.environ.copy()
        env.update(self._agent_env)
        return env

    def _write_invocation_logs(
        self, log_dir: Path, stdout: str, stderr: str
    ) -> None:
        """Append stdout/stderr to log files with timestamps."""
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).isoformat()
        separator = f"\n{'='*60}\n[{timestamp}]\n{'='*60}\n"
        for name, content in (("cli-stdout.log", stdout), ("cli-stderr.log", stderr or "(empty)")):
            with open(log_dir / name, "a", encoding="utf-8") as f:
                f.write(separator)
                f.write(content)

    # --- the shared workflow ------------------------------------------------

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
    ) -> tuple[ActionBlock, LLMUsage]:
        """Run the CLI subprocess and translate its output into an ActionBlock.

        Uses asyncio.create_subprocess_exec so cancelling the awaiting task
        (e.g. via InvocationRegistry.cancel) terminates the subprocess.

        Args:
            prompt: The prompt to send to the model
            working_dir: Directory where the CLI will run (writable sandbox)
            log_dir: Directory for stdout/stderr logs
            additional_dirs: Extra directories the model should be able to read
                (project_dir when different from working_dir, knowledge bases)
            notification_center: Dispatcher for LLM_COMPLETE / LLM_ERROR events
            action_schema: JSON schema for structured output. Selected per
                invocation by the caller based on operational mode
                (WORKER_ACTION_SCHEMA vs COORDINATOR_ACTION_SCHEMA).
            trigger_version: Version of the changelog entry that triggered this
                invocation. Stamped onto the returned `ActionBlock.source_version`
                so every agent-authored action links back to its trigger.
            mcp_config_dir: Optional pointer at the agent's pre-rendered MCP
                dist dir (``{agent_dir}/mcp-hub/dist/``). When set, the
                provider's ``_prepare_invocation`` surfaces the relevant
                per-format file into ``working_dir`` so the spawned CLI
                auto-discovers installed MCP servers from cwd.
            skill_source_dirs: Optional list of skill-source dirs (typically
                the agent's two skill hubs). When set, the provider's
                ``_prepare_invocation`` flattens them into the CLI's native
                cwd skill-discovery path so the spawned CLI auto-loads them
                without a prompt-side INDEX.

        Returns:
            Tuple of (action_block, usage_stats). action_block.actions may be
            empty if the model returned no structured output.
        """
        prepared = self._prepare_invocation(
            prompt, working_dir, additional_dirs, action_schema,
            mcp_config_dir=mcp_config_dir,
            skill_source_dirs=skill_source_dirs,
        )
        env = self._build_env()

        start_time = time.time()
        proc: Optional[asyncio.subprocess.Process] = None
        try:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *prepared.cmd,
                    stdin=(
                        asyncio.subprocess.PIPE
                        if prepared.stdin_bytes is not None
                        else asyncio.subprocess.DEVNULL
                    ),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=prepared.cwd,
                    env=env,
                )
            except FileNotFoundError:
                raise LLMNotFoundError(self._bin, install_hint=self._install_hint)

            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(input=prepared.stdin_bytes),
                    timeout=self._invoke_timeout,
                )
            except asyncio.CancelledError:
                elapsed = time.time() - start_time
                logger.info(f"[{self._log_tag}] CANCELLED after {elapsed:.1f}s")
                raise
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                logger.error(f"[{self._log_tag}] TIMEOUT after {elapsed:.1f}s")
                logger.error(
                    f"[{self._log_tag}] prompt file saved at: {prepared.prompt_file_abs}"
                )
                error = LLMTimeoutError(
                    timeout_seconds=self._invoke_timeout,
                    prompt_file=prepared.prompt_file_abs,
                    working_dir=prepared.cwd,
                    provider=self._provider_name,
                )
                await notification_center.publish(
                    LLM_ERROR, sandbox_dir=working_dir, error=error
                )
                raise error

            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            returncode = proc.returncode if proc.returncode is not None else -1

            elapsed = time.time() - start_time
            logger.info(
                f"[{self._log_tag}] FINISHED in {elapsed:.1f}s, returncode={returncode}"
            )
            logger.info(f"[{self._log_tag}] stdout length={len(stdout)} chars")
            if stderr:
                # Warn only when the process actually failed; on success many
                # CLIs emit benign stderr (e.g. codex "failed to record rollout
                # items"), so log that at debug to avoid noise.
                log = logger.warning if returncode != 0 else logger.debug
                log(f"[{self._log_tag}] stderr ({len(stderr)} chars): {stderr[:1000]}")

            self._write_invocation_logs(log_dir, stdout, stderr)
        finally:
            # Defensive reap on any unexpected exit path
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass

        # Rate limits take precedence over generic invocation errors —
        # they should not be retried with short backoff.
        rate_limit = self._check_rate_limit(prepared, stdout, stderr, returncode)
        if rate_limit is not None:
            rate_limit.prompt_file = prepared.prompt_file_abs
            rate_limit.working_dir = prepared.cwd
            await notification_center.publish(
                LLM_ERROR, sandbox_dir=working_dir, error=rate_limit
            )
            raise rate_limit

        error_detail = self._build_error_detail(prepared, stdout, stderr, returncode)
        if returncode != 0 or error_detail:
            detail = error_detail or stderr or "(no error detail)"
            logger.error(f"[{self._log_tag}] error detail: {detail[:2000]}")
            error = LLMInvocationError(
                f"{self._provider_name} exited with code {returncode}:\n{detail}",
                prompt_file=prepared.prompt_file_abs,
                working_dir=prepared.cwd,
            )
            await notification_center.publish(
                LLM_ERROR, sandbox_dir=working_dir, error=error
            )
            raise error

        parsed = self._parse_result(prepared, stdout, stderr)

        await notification_center.publish(
            LLM_COMPLETE, sandbox_dir=working_dir, usage=parsed.usage
        )

        return (
            ActionBlock(
                raw=parsed.raw_text,
                actions=parsed.actions,
                source_version=trigger_version,
            ),
            parsed.usage,
        )
