# SPDX-License-Identifier: MIT
"""
clawmeets/llm/claude_cli.py
Claude Code CLI provider — subprocess invocation and parsing.

Implements LLMProvider against the `claude` CLI. Raises the generic
LLM* exceptions from clawmeets.llm.base.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..api.actions import ActionBlock
from ..utils.notification_center import LLM_COMPLETE, LLM_ERROR, NotificationCenter

if TYPE_CHECKING:
    from ..runner.mcp_manager import McpManager
from .base import (
    LLMInvocationError,
    LLMNotFoundError,
    LLMProvider,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUsage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ClaudeCLI
# ---------------------------------------------------------------------------


class ClaudeCLI(LLMProvider):
    """Invokes the `claude` CLI (Claude Code) as a subprocess.

    Expects `claude` to be available on PATH.
    """

    def __init__(
        self,
        claude_bin: str = "claude",
        claude_plugin_dirs: Optional[list[Path]] = None,
        mcp_manager: Optional["McpManager"] = None,
        agent_env: Optional[dict[str, str]] = None,
        model: Optional[str] = None,
    ) -> None:
        """Initialize ClaudeCLI.

        Args:
            claude_bin: Path to claude CLI binary
            claude_plugin_dirs: Directories to load as Claude plugins via --plugin-dir
            mcp_manager: If provided, renders .mcp.json into each invocation's
                working_dir so Claude Code picks up installed MCP servers.
            agent_env: Extra environment variables exposed to every Claude
                subprocess. Used to surface the running agent's identity
                (CLAWMEETS_AGENT_ID / CLAWMEETS_AGENT_TOKEN /
                CLAWMEETS_SERVER_URL) so Bash-shelled commands inside Claude
                (e.g. ``clawmeets project create ... --token $...``) can
                authenticate as this agent without re-resolving credentials.
            model: If set, passed to the Claude CLI as ``--model <name>`` per
                invocation. None falls back to Claude Code's configured default.

        The JSON action schema is selected per invocation and passed to
        ``invoke(action_schema=...)``.
        """
        self._bin = claude_bin
        self._claude_plugin_dirs = claude_plugin_dirs or []
        self._mcp_manager = mcp_manager
        self._agent_env = dict(agent_env or {})
        self._model = model

    @classmethod
    def verify_cli(cls, claude_bin: str = "claude") -> None:
        """Verify Claude CLI is available.

        Raises:
            LLMNotFoundError: If CLI not found on PATH
            LLMTimeoutError: If --version times out
            LLMInvocationError: If --version returns error
        """
        try:
            result = subprocess.run(
                [claude_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise LLMInvocationError(
                    f"Claude Code CLI returned error: {result.stderr}"
                )
            logger.info(f"Claude CLI verified: {result.stdout.strip()}")
        except FileNotFoundError:
            raise LLMNotFoundError(
                claude_bin,
                install_hint="Install Claude Code: https://docs.anthropic.com/claude-code",
            )
        except subprocess.TimeoutExpired:
            raise LLMTimeoutError(timeout_seconds=10, provider="Claude Code")

    def _prepare_invocation(
        self,
        prompt: str,
        working_dir: Path,
        additional_dirs: list[Path],
        action_schema: dict,
    ) -> tuple[str, str, list[str]]:
        """Set up directories, write prompt file, and build command.

        Returns:
            Tuple of (prompt_file_abs, claude_cwd, cmd)
        """
        working_dir.mkdir(parents=True, exist_ok=True)

        # Render .mcp.json into the cwd so Claude Code picks up installed MCP
        # servers. No-op if no MCP manager or no installed servers.
        if self._mcp_manager is not None:
            self._mcp_manager.render_mcp_json(working_dir)

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
            "--json-schema", json.dumps(action_schema),
        ]

        if self._model:
            cmd.extend(["--model", self._model])

        for d in additional_dirs:
            cmd.extend(["--add-dir", str(d.expanduser().resolve())])

        for d in self._claude_plugin_dirs:
            cmd.extend(["--plugin-dir", str(d.expanduser().resolve())])

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

    def _parse_json_output(self, raw_output: str) -> tuple[str, LLMUsage, list[dict]]:
        """Parse JSON output from Claude CLI and extract result text, usage stats, and actions."""
        try:
            data = json.loads(raw_output.strip())

            if isinstance(data, list):
                result_data = None
                for item in data:
                    if isinstance(item, dict) and item.get("type") == "result":
                        result_data = item
                        break
                if not result_data:
                    logger.warning("[claude-invoke] No 'result' message found in JSON array output")
                    return raw_output, LLMUsage(), []
            elif isinstance(data, dict):
                result_data = data
            else:
                logger.warning(f"[claude-invoke] Unexpected JSON type: {type(data)}")
                return raw_output, LLMUsage(), []

            result_text = result_data.get("result", "")
            model_usage = result_data.get("modelUsage", {})

            total_input = 0
            total_output = 0
            total_cache_read = 0
            total_cache_creation = 0
            model_name = ""
            for model, stats in model_usage.items():
                model_name = model
                total_input += stats.get("inputTokens", 0)
                total_output += stats.get("outputTokens", 0)
                total_cache_read += stats.get("cacheReadInputTokens", 0)
                total_cache_creation += stats.get("cacheCreationInputTokens", 0)

            usage = LLMUsage(
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

            # Aggregate actions across EVERY StructuredOutput tool_use, not just
            # the last. Claude can split actions across multiple calls in one
            # turn; result.structured_output only carries the most recent call,
            # which silently drops earlier actions.
            merged_actions: list[dict] = []
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict) or item.get("type") != "assistant":
                        continue
                    msg = item.get("message") or {}
                    if not isinstance(msg, dict):
                        continue
                    for c in msg.get("content", []):
                        if not isinstance(c, dict):
                            continue
                        if c.get("type") != "tool_use" or c.get("name") != "StructuredOutput":
                            continue
                        inp = c.get("input") or {}
                        actions = inp.get("actions") if isinstance(inp, dict) else None
                        if isinstance(actions, list):
                            merged_actions.extend(actions)

            structured = result_data.get("structured_output")
            final_actions = (
                structured.get("actions", [])
                if isinstance(structured, dict)
                else []
            )

            if merged_actions:
                if len(merged_actions) != len(final_actions):
                    logger.info(
                        f"[claude-invoke] merged {len(merged_actions)} action(s) across "
                        f"StructuredOutput tool_use events; result.structured_output only "
                        f"reported {len(final_actions)} (using merged set)"
                    )
                else:
                    logger.info(f"[claude-invoke] structured_output.actions: {len(merged_actions)} action(s)")
                return result_text, usage, merged_actions

            if final_actions:
                logger.info(f"[claude-invoke] structured_output.actions: {len(final_actions)} action(s)")
            return result_text, usage, final_actions

        except json.JSONDecodeError as e:
            logger.warning(f"[claude-invoke] Failed to parse JSON output: {e}, returning raw stdout")
            return raw_output, LLMUsage(), []

    async def invoke(
        self,
        prompt: str,
        working_dir: Path,
        log_dir: Path,
        additional_dirs: list[Path],
        notification_center: NotificationCenter,
        action_schema: dict,
    ) -> tuple[ActionBlock, LLMUsage]:
        """Invoke Claude CLI with the given prompt.

        Uses asyncio.create_subprocess_exec so that cancelling the awaiting
        task (e.g. via InvocationRegistry.cancel) terminates the subprocess.
        """
        prompt_file_abs, claude_cwd, cmd = self._prepare_invocation(
            prompt, working_dir, additional_dirs, action_schema
        )

        env = os.environ.copy()
        env["CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD"] = "1"
        # Surface the agent's identity to any Bash-shelled CLI calls inside
        # Claude (e.g. /personal:<project-skill> shelling `clawmeets project
        # create ... --token $CLAWMEETS_AGENT_TOKEN`).
        env.update(self._agent_env)

        invoke_timeout = 1800

        start_time = time.time()
        proc: Optional[asyncio.subprocess.Process] = None
        try:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=claude_cwd,
                    env=env,
                )
            except FileNotFoundError:
                raise LLMNotFoundError(
                    self._bin,
                    install_hint="Install Claude Code: https://docs.anthropic.com/claude-code",
                )

            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(input=prompt.encode("utf-8")),
                    timeout=invoke_timeout,
                )
            except asyncio.CancelledError:
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass
                elapsed = time.time() - start_time
                logger.info(f"[claude-invoke] CANCELLED after {elapsed:.1f}s")
                raise
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass
                elapsed = time.time() - start_time
                logger.error(f"[claude-invoke] TIMEOUT after {elapsed:.1f}s")
                logger.error(f"[claude-invoke] prompt file saved at: {prompt_file_abs}")
                error = LLMTimeoutError(
                    timeout_seconds=invoke_timeout,
                    prompt_file=prompt_file_abs,
                    working_dir=claude_cwd,
                    provider="Claude Code",
                )
                await notification_center.publish(LLM_ERROR, sandbox_dir=working_dir, error=error)
                raise error

            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            returncode = proc.returncode if proc.returncode is not None else -1

            elapsed = time.time() - start_time
            logger.info(f"[claude-invoke] FINISHED in {elapsed:.1f}s, returncode={returncode}")
            logger.info(f"[claude-invoke] stdout length={len(stdout)} chars")
            if stderr:
                logger.warning(f"[claude-invoke] stderr ({len(stderr)} chars): {stderr[:1000]}")

            self._write_invocation_logs(log_dir, stdout, stderr)
        finally:
            # Defensive reap on any unexpected exit path
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass

        if returncode != 0:
            logger.error(f"[claude-invoke] stdout on error: {stdout[:2000]}...")
            error_detail = stderr or "(no stderr)"
            try:
                data = json.loads(stdout.strip())
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get("type") == "result":
                            if "error" in item:
                                error_detail = f"{error_detail}\nJSON error: {item['error']}"
                            break
            except (json.JSONDecodeError, KeyError):
                pass
            error = LLMInvocationError(
                f"Claude Code exited with code {returncode}:\n{error_detail}",
                prompt_file=prompt_file_abs,
                working_dir=claude_cwd,
            )
            await notification_center.publish(LLM_ERROR, sandbox_dir=working_dir, error=error)
            raise error

        result_text, usage, actions = self._parse_json_output(stdout)

        # Detect rate limit: returncode=0 but is_error=true with error="rate_limit"
        try:
            data = json.loads(stdout.strip())
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
                        error = LLMRateLimitError(
                            message=f"Rate limited: {result_text}",
                            resets_at=resets_at,
                            rate_limit_type=rate_limit_type,
                            prompt_file=prompt_file_abs,
                            working_dir=claude_cwd,
                        )
                        await notification_center.publish(LLM_ERROR, sandbox_dir=working_dir, error=error)
                        raise error
        except json.JSONDecodeError:
            pass

        await notification_center.publish(LLM_COMPLETE, sandbox_dir=working_dir, usage=usage)

        return ActionBlock(raw=result_text, actions=actions), usage
