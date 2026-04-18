# SPDX-License-Identifier: MIT
"""
clawmeets/llm/codex_cli.py
OpenAI Codex CLI provider — subprocess invocation and parsing.

Implements LLMProvider against the `codex` CLI (github.com/openai/codex).
Uses `codex exec --json --output-schema <file>` for schema-enforced JSON
output, and `-o <file>` to capture the final message cleanly.

Raises the generic LLM* exceptions from clawmeets.llm.base.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from ..api.actions import ActionBlock
from ..utils.notification_center import LLM_COMPLETE, LLM_ERROR, NotificationCenter
from .base import (
    LLMInvocationError,
    LLMNotFoundError,
    LLMProvider,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUsage,
)

logger = logging.getLogger(__name__)


# Markers for detecting rate limits in error events. Codex surfaces provider
# errors as JSONL events with free-form messages, so we pattern-match.
_RATE_LIMIT_MARKERS = ("rate_limit", "rate limit", "429", "too many requests")


def _adapt_schema_for_codex(schema: dict) -> dict:
    """Rewrite a JSON schema to satisfy OpenAI strict-schema mode.

    Verified empirically:
    - `oneOf` is rejected → rewritten as `anyOf`.
    - `{"const": "x"}` shorthand is rejected (no `type`) → expanded to
      `{"type": "string", "const": "x"}`.

    clawmeets schemas already set `additionalProperties: false` and list
    every property in `required`, which strict mode also requires.
    """
    result = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "oneOf" in node and "anyOf" not in node:
                node["anyOf"] = node.pop("oneOf")
            if "const" in node and "type" not in node:
                const = node["const"]
                if isinstance(const, bool):
                    node["type"] = "boolean"
                elif isinstance(const, int):
                    node["type"] = "integer"
                elif isinstance(const, float):
                    node["type"] = "number"
                elif isinstance(const, str):
                    node["type"] = "string"
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(result)
    return result


class CodexCLI(LLMProvider):
    """Invokes the `codex` CLI (OpenAI Codex) as a subprocess.

    Expects `codex` to be available on PATH and authenticated
    (via `codex login` or OPENAI_API_KEY env var).
    """

    def __init__(
        self,
        action_schema: dict,
        codex_bin: str = "codex",
        model: Optional[str] = None,
        sandbox_mode: str = "workspace-write",
    ) -> None:
        """Initialize CodexCLI.

        Args:
            action_schema: JSON schema for `--output-schema`. Forces the final
                response to match this shape.
            codex_bin: Path to codex CLI binary
            model: Optional model override (e.g. "o3"); None uses Codex default.
            sandbox_mode: Codex sandbox policy. Default "workspace-write" lets
                the agent modify its own sandbox (clawmeets already isolates
                sandboxes per agent). Other options: "read-only",
                "danger-full-access".
        """
        self._bin = codex_bin
        self._action_schema = action_schema
        self._model = model
        self._sandbox_mode = sandbox_mode
        # Codex has no Chrome integration; use_chrome stays False (inherited).

    @classmethod
    def verify_cli(cls, codex_bin: str = "codex") -> None:
        """Verify Codex CLI is available.

        Raises:
            LLMNotFoundError: If CLI not found on PATH
            LLMTimeoutError: If --version times out
            LLMInvocationError: If --version returns error
        """
        try:
            result = subprocess.run(
                [codex_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise LLMInvocationError(
                    f"Codex CLI returned error: {result.stderr}"
                )
            logger.info(f"Codex CLI verified: {result.stdout.strip()}")
        except FileNotFoundError:
            raise LLMNotFoundError(
                codex_bin,
                install_hint="Install Codex: https://github.com/openai/codex",
            )
        except subprocess.TimeoutExpired:
            raise LLMTimeoutError(timeout_seconds=10, provider="Codex")

    def _prepare_invocation(
        self,
        prompt: str,
        working_dir: Path,
        additional_dirs: list[Path],
    ) -> tuple[str, str, str, str, list[str]]:
        """Set up directories, write prompt + schema files, build command.

        Returns:
            Tuple of (prompt_file_abs, schema_file_abs, last_message_abs,
                      codex_cwd, cmd)
        """
        working_dir.mkdir(parents=True, exist_ok=True)

        prompt_file = working_dir / ".agent-prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        prompt_file_abs = str(prompt_file.resolve())

        schema_file = working_dir / ".agent-schema.json"
        adapted_schema = _adapt_schema_for_codex(self._action_schema)
        schema_file.write_text(json.dumps(adapted_schema), encoding="utf-8")
        schema_file_abs = str(schema_file.resolve())

        last_message_file = working_dir / ".agent-last-message.json"
        # Remove any stale file from a prior invocation so we never read a previous run's output.
        if last_message_file.exists():
            last_message_file.unlink()
        last_message_abs = str(last_message_file.resolve())

        codex_cwd = str(working_dir)

        cmd = [
            self._bin,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--sandbox", self._sandbox_mode,
            "--output-schema", schema_file_abs,
            "-o", last_message_abs,
            "-C", codex_cwd,
        ]

        if self._model:
            cmd.extend(["-m", self._model])

        # Codex's --add-dir makes dirs writable (different semantics from
        # Claude's --add-dir). Acceptable for agent sandboxes; knowledge dirs
        # stay read-through-prompt-only since the prompt lists their contents.
        for d in additional_dirs:
            cmd.extend(["--add-dir", str(d.expanduser().resolve())])

        logger.info(f"[codex-invoke] START: invoking Codex CLI via stdin")
        logger.info(f"[codex-invoke] command: {' '.join(cmd)}")
        logger.info(f"[codex-invoke] prompt size={len(prompt)} chars")
        logger.info(f"[codex-invoke] prompt file saved at: {prompt_file_abs}")
        logger.info(f"[codex-invoke] schema file saved at: {schema_file_abs}")
        logger.info(f"[codex-invoke] cwd={codex_cwd}")
        if additional_dirs:
            logger.info(f"[codex-invoke] additional-dirs={[str(d.expanduser().resolve()) for d in additional_dirs]}")
        logger.debug(f"[codex-invoke] prompt content:\n{prompt[:500]}...")

        return prompt_file_abs, schema_file_abs, last_message_abs, codex_cwd, cmd

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

    def _parse_events(self, raw_output: str) -> tuple[LLMUsage, list[dict]]:
        """Parse Codex JSONL event stream for usage + error events.

        Returns:
            (usage, error_events) — usage is empty LLMUsage if no
            turn.completed seen; error_events is a list of error payloads.
        """
        usage = LLMUsage()
        errors: list[dict] = []

        for line in raw_output.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            if etype == "turn.completed":
                # Usage shape (empirically verified against codex docs):
                # {"type":"turn.completed","usage":{"input_tokens":N,
                #  "output_tokens":M,"cached_input_tokens":K,...}}
                u = event.get("usage") or {}
                usage = LLMUsage(
                    input_tokens=u.get("input_tokens", 0),
                    output_tokens=u.get("output_tokens", 0),
                    cache_read_tokens=u.get("cached_input_tokens", 0),
                    cache_creation_tokens=0,  # Codex doesn't surface this
                    cost_usd=u.get("total_cost_usd", 0.0),
                    duration_ms=event.get("duration_ms", 0),
                    model=event.get("model", "") or u.get("model", ""),
                )
            elif etype in ("error", "turn.failed"):
                errors.append(event)

        return usage, errors

    def _read_last_message(self, last_message_abs: str) -> Optional[dict]:
        """Read the schema-conformant final message from Codex's -o file."""
        path = Path(last_message_abs)
        if not path.exists():
            return None
        try:
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                return None
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"[codex-invoke] last message is not valid JSON: {e}")
            return None

    def _extract_actions(self, final_message: Optional[dict]) -> list[dict]:
        """Extract the actions array from the schema-validated final message."""
        if not isinstance(final_message, dict):
            return []
        actions = final_message.get("actions")
        if isinstance(actions, list):
            return actions
        return []

    def _detect_rate_limit(self, errors: list[dict], stderr: str) -> Optional[LLMRateLimitError]:
        """Check whether any error event signals a rate limit."""
        def _matches(text: str) -> bool:
            lower = text.lower()
            return any(marker in lower for marker in _RATE_LIMIT_MARKERS)

        for event in errors:
            msg = event.get("message", "") or ""
            err = event.get("error") or {}
            err_msg = err.get("message", "") if isinstance(err, dict) else ""
            combined = f"{msg} {err_msg}"
            if _matches(combined):
                return LLMRateLimitError(
                    message=f"Rate limited: {combined.strip() or 'rate limit'}",
                    rate_limit_type=None,
                )

        if _matches(stderr):
            return LLMRateLimitError(
                message=f"Rate limited: {stderr.strip()[:500]}",
                rate_limit_type=None,
            )

        return None

    async def invoke(
        self,
        prompt: str,
        working_dir: Path,
        log_dir: Path,
        additional_dirs: list[Path],
        notification_center: NotificationCenter,
    ) -> tuple[ActionBlock, LLMUsage]:
        """Invoke Codex CLI with the given prompt."""
        (
            prompt_file_abs,
            _schema_file_abs,
            last_message_abs,
            codex_cwd,
            cmd,
        ) = self._prepare_invocation(prompt, working_dir, additional_dirs)

        env = os.environ.copy()

        invoke_timeout = 300

        start_time = time.time()
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                cwd=codex_cwd,
                env=env,
                timeout=invoke_timeout,
            )
            elapsed = time.time() - start_time
            logger.info(f"[codex-invoke] FINISHED in {elapsed:.1f}s, returncode={result.returncode}")
            logger.info(f"[codex-invoke] stdout length={len(result.stdout)} chars")
            if result.stderr:
                logger.warning(f"[codex-invoke] stderr ({len(result.stderr)} chars): {result.stderr[:1000]}")

            self._write_invocation_logs(log_dir, result.stdout, result.stderr)

        except FileNotFoundError:
            raise LLMNotFoundError(
                self._bin,
                install_hint="Install Codex: https://github.com/openai/codex",
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            logger.error(f"[codex-invoke] TIMEOUT after {elapsed:.1f}s")
            error = LLMTimeoutError(
                timeout_seconds=invoke_timeout,
                prompt_file=prompt_file_abs,
                working_dir=codex_cwd,
                provider="Codex",
            )
            await notification_center.publish(LLM_ERROR, sandbox_dir=working_dir, error=error)
            raise error

        usage, error_events = self._parse_events(result.stdout)

        # Rate limit detection takes precedence — they look like invocation errors
        # otherwise, and should not be retried with short backoff.
        rate_limit_error = self._detect_rate_limit(error_events, result.stderr)
        if rate_limit_error is not None:
            rate_limit_error.prompt_file = prompt_file_abs
            rate_limit_error.working_dir = codex_cwd
            await notification_center.publish(LLM_ERROR, sandbox_dir=working_dir, error=rate_limit_error)
            raise rate_limit_error

        if result.returncode != 0 or error_events:
            logger.error(f"[codex-invoke] stdout tail: {result.stdout[-2000:]}")
            detail_parts = []
            if error_events:
                for ev in error_events[:3]:
                    detail_parts.append(json.dumps(ev)[:500])
            if result.stderr:
                detail_parts.append(result.stderr[:500])
            detail = "\n".join(detail_parts) or "(no error detail)"
            error = LLMInvocationError(
                f"Codex exited with code {result.returncode}:\n{detail}",
                prompt_file=prompt_file_abs,
                working_dir=codex_cwd,
            )
            await notification_center.publish(LLM_ERROR, sandbox_dir=working_dir, error=error)
            raise error

        final_message = self._read_last_message(last_message_abs)
        actions = self._extract_actions(final_message)
        if actions:
            logger.info(f"[codex-invoke] final message actions: {len(actions)} action(s)")
        else:
            logger.warning(f"[codex-invoke] no actions parsed from final message")

        logger.info(
            f"[codex-invoke] usage: cost=${usage.cost_usd:.4f} "
            f"in={usage.input_tokens} out={usage.output_tokens} "
            f"cache_read={usage.cache_read_tokens}"
        )

        await notification_center.publish(LLM_COMPLETE, sandbox_dir=working_dir, usage=usage)

        raw_output = json.dumps(final_message) if final_message is not None else ""
        return ActionBlock(raw=raw_output, actions=actions), usage
