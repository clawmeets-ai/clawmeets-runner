# SPDX-License-Identifier: MIT
"""
clawmeets/llm/claude_cli.py
Claude CLI integration - subprocess invocation and parsing.

This module is part of Layer 0 (pure - no domain model dependencies).
It provides Claude CLI integration for agent participants.

Classes defined here:
- ClaudeCLI: Direct CLI invocation
- ClaudeUsage: Usage stats from invocation
- ClaudeInvocationError: Base exception for CLI errors
- ClaudeTimeoutError: Timeout during invocation
- ClaudeNotFoundError: CLI not found on PATH
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..api.actions import ActionBlock
from ..utils.notification_center import LLM_COMPLETE, LLM_ERROR, NotificationCenter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception Classes
# ---------------------------------------------------------------------------


class ClaudeInvocationError(Exception):
    """Base exception for Claude CLI invocation errors."""

    def __init__(
        self,
        message: str,
        prompt_file: str | None = None,
        working_dir: str | None = None,
    ) -> None:
        self.prompt_file = prompt_file
        self.working_dir = working_dir
        super().__init__(message)


class ClaudeTimeoutError(ClaudeInvocationError):
    """Claude CLI invocation timed out."""

    def __init__(
        self,
        timeout_seconds: int,
        prompt_file: str | None = None,
        working_dir: str | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        message = f"Claude Code invocation timed out after {timeout_seconds} seconds"
        if prompt_file:
            message += f". Prompt file: {prompt_file}"
        super().__init__(message, prompt_file, working_dir)


class ClaudeNotFoundError(ClaudeInvocationError):
    """Claude CLI not found on PATH."""

    def __init__(self, claude_bin: str = "claude") -> None:
        self.claude_bin = claude_bin
        message = (
            f"'{claude_bin}' not found on PATH. "
            "Install Claude Code: https://docs.anthropic.com/claude-code"
        )
        super().__init__(message)


class ClaudeRateLimitError(ClaudeInvocationError):
    """Claude CLI session hit a rate limit (returncode=0, is_error=true)."""

    def __init__(
        self,
        message: str,
        resets_at: float | None = None,
        rate_limit_type: str | None = None,
        prompt_file: str | None = None,
        working_dir: str | None = None,
    ) -> None:
        self.resets_at = resets_at
        self.rate_limit_type = rate_limit_type
        super().__init__(message, prompt_file, working_dir)

    @property
    def resets_at_human(self) -> str | None:
        """Human-readable reset time."""
        if self.resets_at is None:
            return None
        from datetime import datetime, timezone

        dt = datetime.fromtimestamp(self.resets_at, tz=timezone.utc)
        return dt.astimezone().strftime("%I:%M %p %Z")


# ---------------------------------------------------------------------------
# ClaudeUsage (Claude-specific)
# ---------------------------------------------------------------------------

@dataclass
class ClaudeUsage:
    """Usage stats from a single Claude invocation."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    model: str = ""

    def __add__(self, other: "ClaudeUsage") -> "ClaudeUsage":
        """Accumulate usage across invocations."""
        return ClaudeUsage(
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
# ClaudeCLI
# ---------------------------------------------------------------------------

class ClaudeCLI:
    """
    Invokes the `claude` CLI (Claude Code) as a subprocess.
    Expects `claude` to be available on PATH.
    """

    def __init__(
        self,
        action_schema: dict,
        claude_bin: str = "claude",
        claude_plugin_dirs: Optional[list[Path]] = None,
        use_chrome: bool = False,
    ) -> None:
        """Initialize ClaudeCLI.

        Args:
            action_schema: JSON schema for structured output validation.
                          Use WORKER_ACTION_SCHEMA for workers (reply, update_file only).
                          Use COORDINATOR_ACTION_SCHEMA for coordinators (all actions).
            claude_bin: Path to claude CLI binary
            claude_plugin_dirs: Directories to load as Claude plugins via --plugin-dir
            use_chrome: Enable Chrome browser integration via --chrome flag
        """
        self._bin = claude_bin
        self._action_schema = action_schema
        self._claude_plugin_dirs = claude_plugin_dirs or []
        self._use_chrome = use_chrome

    @classmethod
    def verify_cli(cls, claude_bin: str = "claude") -> None:
        """Verify Claude CLI is available.

        Call this before creating ClaudeCLI instances if you want
        to fail early when the CLI is not installed.

        Raises:
            ClaudeNotFoundError: If CLI not found on PATH
            ClaudeTimeoutError: If --version times out
            ClaudeInvocationError: If --version returns error
        """
        try:
            result = subprocess.run(
                [claude_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise ClaudeInvocationError(
                    f"Claude Code CLI returned error: {result.stderr}"
                )
            logger.info(f"Claude CLI verified: {result.stdout.strip()}")
        except FileNotFoundError:
            raise ClaudeNotFoundError(claude_bin)
        except subprocess.TimeoutExpired:
            raise ClaudeTimeoutError(timeout_seconds=10)

    def _prepare_invocation(
        self,
        prompt: str,
        working_dir: Path,
        additional_dirs: list[Path],
    ) -> tuple[str, str, list[str]]:
        """Set up directories, write prompt file, and build command.

        Args:
            prompt: The prompt to send to Claude
            working_dir: Claude's working directory (cwd)
            additional_dirs: Additional directories to include via --add-dir
                             (e.g., data_dir when different from working_dir, knowledge bases)

        Returns:
            Tuple of (prompt_file_abs, claude_cwd, cmd)
        """
        working_dir.mkdir(parents=True, exist_ok=True)

        # Write prompt to a temp file for debugging
        prompt_file = working_dir / ".agent-prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        prompt_file_abs = str(prompt_file.resolve())
        claude_cwd = str(working_dir)

        cmd = [
            self._bin,
            "--print",
            "--verbose",
            "--permission-mode", "bypassPermissions",
            "--no-session-persistence",
            "--output-format", "json",
            "--json-schema", json.dumps(self._action_schema),
        ]

        # Add all additional directories via --add-dir
        # Use expanduser() to handle ~ in paths before resolve()
        for d in additional_dirs:
            cmd.extend(["--add-dir", str(d.expanduser().resolve())])

        # Add Claude plugin directories via --plugin-dir
        for d in self._claude_plugin_dirs:
            cmd.extend(["--plugin-dir", str(d.expanduser().resolve())])

        # Enable Chrome browser integration
        if self._use_chrome:
            cmd.append("--chrome")

        logger.info(f"[claude-invoke] START: invoking Claude CLI via stdin")
        logger.info(f"[claude-invoke] command: {' '.join(cmd)}")
        logger.info(f"[claude-invoke] prompt size={len(prompt)} chars")
        logger.info(f"[claude-invoke] prompt file saved at: {prompt_file_abs}")
        logger.info(f"[claude-invoke] cwd={claude_cwd}")
        if additional_dirs:
            logger.info(f"[claude-invoke] additional-dirs={[str(d.expanduser().resolve()) for d in additional_dirs]}")
        if self._claude_plugin_dirs:
            logger.info(f"[claude-invoke] plugin-dirs={[str(d.expanduser().resolve()) for d in self._claude_plugin_dirs]}")
        logger.info(f"[claude-invoke] To test manually: cd {claude_cwd} && cat {prompt_file_abs} | {' '.join(cmd)}")
        logger.debug(f"[claude-invoke] prompt content:\n{prompt[:500]}...")

        return prompt_file_abs, claude_cwd, cmd

    def _write_invocation_logs(
        self, log_dir: Path, stdout: str, stderr: str
    ) -> None:
        """Append stdout/stderr to log files with timestamps."""
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_file = log_dir / "cli-stdout.log"
        stderr_file = log_dir / "cli-stderr.log"

        timestamp = datetime.now(UTC).isoformat()
        separator = f"\n{'='*60}\n[{timestamp}]\n{'='*60}\n"

        with open(stdout_file, "a", encoding="utf-8") as f:
            f.write(separator)
            f.write(stdout)
        with open(stderr_file, "a", encoding="utf-8") as f:
            f.write(separator)
            f.write(stderr or "(empty)")

    def _parse_json_output(self, raw_output: str) -> tuple[str, ClaudeUsage, list[dict]]:
        """Parse JSON output from Claude CLI and extract result text, usage stats, and actions.

        Returns:
            Tuple of (result_text, usage_stats, actions_list)
            - actions_list is extracted from structured_output.actions if present, empty list otherwise
        """
        try:
            data = json.loads(raw_output.strip())

            # Handle both array format and single object format
            if isinstance(data, list):
                result_data = None
                for item in data:
                    if isinstance(item, dict) and item.get("type") == "result":
                        result_data = item
                        break
                if not result_data:
                    logger.warning("[claude-invoke] No 'result' message found in JSON array output")
                    return raw_output, ClaudeUsage(), []
            elif isinstance(data, dict):
                result_data = data
            else:
                logger.warning(f"[claude-invoke] Unexpected JSON type: {type(data)}")
                return raw_output, ClaudeUsage(), []

            result_text = result_data.get("result", "")
            model_usage = result_data.get("modelUsage", {})

            # Sum up tokens across all models used in the session
            # modelUsage contains cumulative totals, unlike "usage" which is final response only
            total_input = 0
            total_output = 0
            total_cache_read = 0
            total_cache_creation = 0
            model_name = ""
            for model, stats in model_usage.items():
                model_name = model  # Use last model seen
                total_input += stats.get("inputTokens", 0)
                total_output += stats.get("outputTokens", 0)
                total_cache_read += stats.get("cacheReadInputTokens", 0)
                total_cache_creation += stats.get("cacheCreationInputTokens", 0)

            usage = ClaudeUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                cache_read_tokens=total_cache_read,
                cache_creation_tokens=total_cache_creation,
                cost_usd=result_data.get("total_cost_usd", 0.0),
                duration_ms=result_data.get("duration_ms", 0),
                model=model_name,
            )

            logger.info(
                f"[claude-invoke] usage: cost=${usage.cost_usd:.4f} "
                f"in={usage.input_tokens} out={usage.output_tokens} "
                f"cache_read={usage.cache_read_tokens} cache_create={usage.cache_creation_tokens}"
            )

            # Extract structured_output.actions from result item
            structured = result_data.get("structured_output")
            if structured and isinstance(structured, dict):
                actions = structured.get("actions", [])
                if actions:
                    logger.info(f"[claude-invoke] structured_output.actions: {len(actions)} action(s)")
                return result_text, usage, actions

            return result_text, usage, []

        except json.JSONDecodeError as e:
            logger.warning(f"[claude-invoke] Failed to parse JSON output: {e}, returning raw stdout")
            return raw_output, ClaudeUsage(), []

    async def invoke(
        self,
        prompt: str,
        working_dir: Path,
        log_dir: Path,
        additional_dirs: list[Path],
        notification_center: NotificationCenter,
    ) -> tuple[ActionBlock, ClaudeUsage]:
        """Invoke Claude CLI with the given prompt.

        Args:
            prompt: The prompt to send to Claude
            working_dir: Directory where Claude will run
            log_dir: Directory for log files
            additional_dirs: Additional directories to include via --add-dir
                             (e.g., data_dir when different from working_dir, knowledge bases)
            notification_center: Dispatcher for LLM lifecycle events (LLM_COMPLETE, LLM_ERROR)

        Returns:
            Tuple of (action_block, usage_stats)
            - action_block.actions is empty list if no structured actions were returned
        """
        # Prepare invocation (setup dirs, write prompt, build command)
        prompt_file_abs, claude_cwd, cmd = self._prepare_invocation(
            prompt, working_dir, additional_dirs
        )

        # Always enable CLAUDE.md loading from additional directories
        # (harmless if no additional dirs - no CLAUDE.md files to load)
        env = os.environ.copy()
        env["CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD"] = "1"

        # Chrome browsing takes longer (navigating, filling forms, extracting)
        invoke_timeout = 600 if self._use_chrome else 300

        # Run subprocess
        start_time = time.time()
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                cwd=claude_cwd,
                env=env,
                timeout=invoke_timeout,
            )
            elapsed = time.time() - start_time
            logger.info(f"[claude-invoke] FINISHED in {elapsed:.1f}s, returncode={result.returncode}")
            logger.info(f"[claude-invoke] stdout length={len(result.stdout)} chars")
            if result.stderr:
                logger.warning(f"[claude-invoke] stderr ({len(result.stderr)} chars): {result.stderr[:1000]}")

            self._write_invocation_logs(log_dir, result.stdout, result.stderr)

        except FileNotFoundError:
            raise ClaudeNotFoundError(self._bin)
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start_time
            logger.error(f"[claude-invoke] TIMEOUT after {elapsed:.1f}s")
            logger.error(f"[claude-invoke] partial stdout: {getattr(e, 'stdout', 'N/A')}")
            logger.error(f"[claude-invoke] partial stderr: {getattr(e, 'stderr', 'N/A')}")
            logger.error(f"[claude-invoke] prompt file saved at: {prompt_file_abs}")
            error = ClaudeTimeoutError(
                timeout_seconds=invoke_timeout,
                prompt_file=prompt_file_abs,
                working_dir=claude_cwd,
            )
            await notification_center.publish(LLM_ERROR, sandbox_dir=working_dir, error=error)
            raise error

        if result.returncode != 0:
            # Log stdout content on error for debugging
            logger.error(f"[claude-invoke] stdout on error: {result.stdout[:2000]}...")
            # Try to extract any useful error message from JSON output
            error_detail = result.stderr or "(no stderr)"
            try:
                data = json.loads(result.stdout.strip())
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get("type") == "result":
                            if "error" in item:
                                error_detail = f"{error_detail}\nJSON error: {item['error']}"
                            break
            except (json.JSONDecodeError, KeyError):
                pass
            error = ClaudeInvocationError(
                f"Claude Code exited with code {result.returncode}:\n{error_detail}",
                prompt_file=prompt_file_abs,
                working_dir=claude_cwd,
            )
            await notification_center.publish(LLM_ERROR, sandbox_dir=working_dir, error=error)
            raise error

        # Parse output, extract usage and actions, return ActionBlock directly
        raw_output, usage, actions = self._parse_json_output(result.stdout)

        # Detect rate limit: returncode=0 but is_error=true with error="rate_limit"
        try:
            data = json.loads(result.stdout.strip())
            if isinstance(data, list):
                for item in data:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "result"
                        and item.get("is_error") is True
                        and item.get("error") == "rate_limit"
                    ):
                        resets_at = None
                        rate_limit_type = None
                        for entry in data:
                            if isinstance(entry, dict) and entry.get("type") == "rate_limit_event":
                                info = entry.get("rate_limit_info", {})
                                resets_at = info.get("resetsAt")
                                rate_limit_type = info.get("rateLimitType")
                                break
                        error = ClaudeRateLimitError(
                            message=f"Rate limited: {result_text}",
                            resets_at=resets_at,
                            rate_limit_type=rate_limit_type,
                            prompt_file=prompt_file_abs,
                            working_dir=claude_cwd,
                        )
                        await notification_center.publish(LLM_ERROR, sandbox_dir=working_dir, error=error)
                        raise error
        except json.JSONDecodeError:
            pass  # Already handled by _parse_json_output

        await notification_center.publish(LLM_COMPLETE, sandbox_dir=working_dir, usage=usage)

        return ActionBlock(raw=raw_output, actions=actions), usage
