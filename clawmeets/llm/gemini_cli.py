# SPDX-License-Identifier: MIT
"""
clawmeets/llm/gemini_cli.py
Google Gemini CLI provider — subprocess invocation and parsing.

Implements LLMProvider against the `gemini` CLI (github.com/google-gemini/gemini-cli).

Unlike Claude and Codex, Gemini has no schema-enforcement flag — `-o json`
only wraps the free-text model output inside a `{response, stats, ...}`
envelope. We compensate by:

1. Appending a strict "output only JSON" suffix to the prompt.
2. Parsing the envelope's `response` field; first as-is, then with
   markdown-fence stripping before giving up.
3. Letting the Agent/Assistant retry loop handle transient misses.

Raises the generic LLM* exceptions from clawmeets.llm.base.
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


# Markers for detecting rate limits in stderr or error envelopes.
_RATE_LIMIT_MARKERS = (
    "rate_limit",
    "rate limit",
    "429",
    "too many requests",
    "quota",
    "resource_exhausted",
)


_JSON_ONLY_SUFFIX = (
    "\n\n=== OUTPUT FORMAT (STRICT) ===\n"
    "Respond with ONLY the raw JSON object matching the action schema above. "
    "Do not wrap it in markdown fences (no ```json, no ```). "
    "Do not include any prose, explanation, or trailing text. "
    "Your entire response must be a single parseable JSON object."
)


def _strip_markdown_fences(text: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` fences if the text is wrapped in them.

    Returns the inner content unchanged if no fence is detected.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    first_newline = stripped.find("\n")
    if first_newline == -1:
        return stripped  # No newline after fence — not a valid block, leave it.

    inner = stripped[first_newline + 1 :].rstrip()
    if inner.endswith("```"):
        inner = inner[:-3].rstrip()
    return inner


class GeminiCLI(LLMProvider):
    """Invokes the `gemini` CLI (Google Gemini) as a subprocess.

    Expects `gemini` to be available on PATH and authenticated
    (via `gemini auth` or GEMINI_API_KEY env var).
    """

    def __init__(
        self,
        gemini_bin: str = "gemini",
        model: Optional[str] = None,
    ) -> None:
        """Initialize GeminiCLI.

        Args:
            gemini_bin: Path to gemini CLI binary
            model: Optional model override (e.g. "gemini-2.5-pro"); None uses
                Gemini's default.

        Gemini cannot enforce a JSON schema at the CLI level; the schema is
        embedded in the prompt (via the existing prompt builder) and parsed
        post-hoc. The schema is still passed to ``invoke(action_schema=...)``
        for symmetry with the other providers, but is ignored at this layer.
        """
        self._bin = gemini_bin
        self._model = model
        # Gemini has no Chrome integration; use_chrome stays False (inherited).

    @classmethod
    def verify_cli(cls, gemini_bin: str = "gemini") -> None:
        """Verify Gemini CLI is available.

        Raises:
            LLMNotFoundError: If CLI not found on PATH
            LLMTimeoutError: If --version times out
            LLMInvocationError: If --version returns error
        """
        try:
            result = subprocess.run(
                [gemini_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise LLMInvocationError(
                    f"Gemini CLI returned error: {result.stderr}"
                )
            logger.info(f"Gemini CLI verified: {result.stdout.strip()}")
        except FileNotFoundError:
            raise LLMNotFoundError(
                gemini_bin,
                install_hint="Install Gemini CLI: https://github.com/google-gemini/gemini-cli",
            )
        except subprocess.TimeoutExpired:
            raise LLMTimeoutError(timeout_seconds=10, provider="Gemini")

    def _prepare_invocation(
        self,
        prompt: str,
        working_dir: Path,
        additional_dirs: list[Path],
    ) -> tuple[str, str, str, list[str]]:
        """Set up directories, write prompt file, build command.

        Returns:
            Tuple of (prompt_file_abs, gemini_cwd, full_prompt, cmd)
        """
        working_dir.mkdir(parents=True, exist_ok=True)

        full_prompt = prompt + _JSON_ONLY_SUFFIX

        prompt_file = working_dir / ".agent-prompt.txt"
        prompt_file.write_text(full_prompt, encoding="utf-8")
        prompt_file_abs = str(prompt_file.resolve())
        gemini_cwd = str(working_dir)

        cmd = [
            self._bin,
            "-o", "json",
            "--approval-mode", "yolo",
        ]

        if self._model:
            cmd.extend(["-m", self._model])

        # --include-directories accepts either comma-separated or repeated args.
        # Use repeated args so paths with commas (unlikely but possible) work.
        for d in additional_dirs:
            cmd.extend(["--include-directories", str(d.expanduser().resolve())])

        cmd.extend(["-p", full_prompt])

        # Log a sanitized command — the full prompt would be unreadably long.
        log_cmd = cmd[:-1] + [f"<prompt:{len(full_prompt)} chars from {prompt_file_abs}>"]
        logger.info(f"[gemini-invoke] START")
        logger.info(f"[gemini-invoke] command: {' '.join(log_cmd)}")
        logger.info(f"[gemini-invoke] cwd={gemini_cwd}")
        if additional_dirs:
            logger.info(
                f"[gemini-invoke] include-dirs="
                f"{[str(d.expanduser().resolve()) for d in additional_dirs]}"
            )
        logger.debug(f"[gemini-invoke] prompt content:\n{full_prompt[:500]}...")

        return prompt_file_abs, gemini_cwd, full_prompt, cmd

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

    def _parse_envelope(self, raw_output: str) -> Optional[dict]:
        """Parse Gemini's outer JSON envelope from stdout."""
        text = raw_output.strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"[gemini-invoke] outer envelope is not valid JSON: {e}")
            return None
        if not isinstance(data, dict):
            logger.warning(
                f"[gemini-invoke] unexpected envelope type: {type(data).__name__}"
            )
            return None
        return data

    def _parse_response_field(self, response_text: str) -> Optional[Any]:
        """Parse the envelope's `response` string as JSON.

        Tries raw first, then markdown-fence-stripped before giving up.
        """
        if not response_text:
            return None

        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        stripped = _strip_markdown_fences(response_text)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as e:
            logger.warning(
                f"[gemini-invoke] response field is not valid JSON after fence "
                f"strip: {e}; head={response_text[:200]!r}"
            )
            return None

    def _extract_usage(self, envelope: dict) -> LLMUsage:
        """Aggregate token usage across all models in stats.models.*"""
        stats = envelope.get("stats") or {}
        models = stats.get("models") or {}
        if not isinstance(models, dict):
            return LLMUsage()

        input_tokens = 0
        output_tokens = 0
        cached = 0
        model_names: list[str] = []

        for name, info in models.items():
            if not isinstance(info, dict):
                continue
            model_names.append(name)
            tokens = info.get("tokens") or {}
            if not isinstance(tokens, dict):
                continue
            input_tokens += tokens.get("input", 0) or 0
            output_tokens += tokens.get("candidates", 0) or 0
            cached += tokens.get("cached", 0) or 0

        # Gemini returns multiple models (routing utility + main). Report the
        # "main" one if present, else any; falls back to empty string.
        model_name = ""
        if "main" in model_names:
            model_name = "main"
        elif model_names:
            model_name = model_names[-1]

        return LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cached,
            cache_creation_tokens=0,
            cost_usd=0.0,  # Gemini does not report cost
            duration_ms=0,  # Latency is in stats but not summed here
            model=model_name,
        )

    def _extract_actions(self, parsed: Optional[Any]) -> list[dict]:
        """Extract the actions array from the parsed response."""
        if not isinstance(parsed, dict):
            return []
        actions = parsed.get("actions")
        if isinstance(actions, list):
            return actions
        return []

    def _detect_rate_limit(self, *haystacks: str) -> Optional[LLMRateLimitError]:
        """Check each text blob for rate-limit markers."""
        def _matches(text: str) -> bool:
            lower = text.lower()
            return any(marker in lower for marker in _RATE_LIMIT_MARKERS)

        for hay in haystacks:
            if hay and _matches(hay):
                return LLMRateLimitError(
                    message=f"Rate limited: {hay.strip()[:500]}",
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
        action_schema: dict,
    ) -> tuple[ActionBlock, LLMUsage]:
        """Invoke Gemini CLI with the given prompt.

        ``action_schema`` is accepted for interface symmetry but not used —
        Gemini has no CLI-level schema enforcement; the schema is embedded
        in the prompt by the caller's prompt builder.
        """
        (
            prompt_file_abs,
            gemini_cwd,
            _full_prompt,
            cmd,
        ) = self._prepare_invocation(prompt, working_dir, additional_dirs)

        env = os.environ.copy()

        invoke_timeout = 300

        start_time = time.time()
        proc: Optional[asyncio.subprocess.Process] = None
        try:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=gemini_cwd,
                    env=env,
                )
            except FileNotFoundError:
                raise LLMNotFoundError(
                    self._bin,
                    install_hint="Install Gemini CLI: https://github.com/google-gemini/gemini-cli",
                )

            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=invoke_timeout,
                )
            except asyncio.CancelledError:
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass
                elapsed = time.time() - start_time
                logger.info(f"[gemini-invoke] CANCELLED after {elapsed:.1f}s")
                raise
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass
                elapsed = time.time() - start_time
                logger.error(f"[gemini-invoke] TIMEOUT after {elapsed:.1f}s")
                error = LLMTimeoutError(
                    timeout_seconds=invoke_timeout,
                    prompt_file=prompt_file_abs,
                    working_dir=gemini_cwd,
                    provider="Gemini",
                )
                await notification_center.publish(LLM_ERROR, sandbox_dir=working_dir, error=error)
                raise error

            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            returncode = proc.returncode if proc.returncode is not None else -1

            elapsed = time.time() - start_time
            logger.info(
                f"[gemini-invoke] FINISHED in {elapsed:.1f}s, "
                f"returncode={returncode}"
            )
            logger.info(f"[gemini-invoke] stdout length={len(stdout)} chars")
            if stderr:
                logger.warning(
                    f"[gemini-invoke] stderr ({len(stderr)} chars): "
                    f"{stderr[:1000]}"
                )

            self._write_invocation_logs(log_dir, stdout, stderr)
        finally:
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass

        # Rate limit check — look at stderr first (most likely location).
        envelope = self._parse_envelope(stdout)
        envelope_error_text = ""
        if envelope is not None:
            err = envelope.get("error")
            if isinstance(err, str):
                envelope_error_text = err
            elif isinstance(err, dict):
                envelope_error_text = err.get("message", "") or json.dumps(err)

        rate_limit_error = self._detect_rate_limit(stderr, envelope_error_text)
        if rate_limit_error is not None:
            rate_limit_error.prompt_file = prompt_file_abs
            rate_limit_error.working_dir = gemini_cwd
            await notification_center.publish(LLM_ERROR, sandbox_dir=working_dir, error=rate_limit_error)
            raise rate_limit_error

        # Non-zero exit with no rate-limit signature is a generic invocation error.
        if returncode != 0:
            logger.error(f"[gemini-invoke] stdout tail: {stdout[-2000:]}")
            detail_parts = []
            if envelope_error_text:
                detail_parts.append(envelope_error_text[:500])
            if stderr:
                detail_parts.append(stderr[:500])
            detail = "\n".join(detail_parts) or "(no error detail)"
            error = LLMInvocationError(
                f"Gemini exited with code {returncode}:\n{detail}",
                prompt_file=prompt_file_abs,
                working_dir=gemini_cwd,
            )
            await notification_center.publish(LLM_ERROR, sandbox_dir=working_dir, error=error)
            raise error

        # Returncode 0 but no envelope → treat as invocation error (malformed output).
        if envelope is None:
            error = LLMInvocationError(
                f"Gemini returned non-JSON stdout:\n{stdout[:500]}",
                prompt_file=prompt_file_abs,
                working_dir=gemini_cwd,
            )
            await notification_center.publish(LLM_ERROR, sandbox_dir=working_dir, error=error)
            raise error

        # Envelope reports an error even with exit code 0 (shouldn't normally happen).
        if envelope_error_text:
            error = LLMInvocationError(
                f"Gemini envelope error: {envelope_error_text[:500]}",
                prompt_file=prompt_file_abs,
                working_dir=gemini_cwd,
            )
            await notification_center.publish(LLM_ERROR, sandbox_dir=working_dir, error=error)
            raise error

        usage = self._extract_usage(envelope)

        response_text = envelope.get("response")
        if not isinstance(response_text, str):
            response_text = ""
        parsed = self._parse_response_field(response_text)
        actions = self._extract_actions(parsed)

        if actions:
            logger.info(f"[gemini-invoke] parsed {len(actions)} action(s)")
        else:
            logger.warning(
                f"[gemini-invoke] no actions parsed — response head: "
                f"{response_text[:300]!r}"
            )

        logger.info(
            f"[gemini-invoke] usage: in={usage.input_tokens} "
            f"out={usage.output_tokens} cache_read={usage.cache_read_tokens}"
        )

        await notification_center.publish(LLM_COMPLETE, sandbox_dir=working_dir, usage=usage)

        raw_output = (
            json.dumps(parsed) if parsed is not None else (response_text or "")
        )
        return ActionBlock(raw=raw_output, actions=actions), usage
