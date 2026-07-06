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

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

from ._structured_text import JSON_ONLY_SUFFIX as _JSON_ONLY_SUFFIX
from ._structured_text import normalize_actions, parse_json_object
from .base import (
    LLMInvocationError,
    LLMNotFoundError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUsage,
    ParsedResult,
    PreparedInvocation,
    SubprocessLLMProvider,
    link_mcp_config_into,
    materialize_skill_tree,
)
from .pricing import price_usd

logger = logging.getLogger(__name__)

# The `gemini` CLI's default model, used to price token usage when no explicit
# --llm-model is set (the CLI reports tokens but not cost, and its usage stats
# key models by ROLE, not model ID). Exact pricing requires --llm-model.
_DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"


# Markers for detecting rate limits in stderr or error envelopes.
_RATE_LIMIT_MARKERS = (
    "rate_limit",
    "rate limit",
    "429",
    "too many requests",
    "quota",
    "resource_exhausted",
)

# Auth / ineligible-tier failures (e.g. deprecated "Gemini Code Assist for
# individuals"). Checked first so a retry-backoff line in the auth stack trace
# isn't misclassified as a rate limit.
_AUTH_MARKERS = (
    "ineligibletier",
    "unsupported_client",
    "no longer supported",
    "error authenticating",
    "gemini code assist",
)


class GeminiCLI(SubprocessLLMProvider):
    """Invokes the `gemini` CLI (Google Gemini) as a subprocess.

    Expects `gemini` to be available on PATH and authenticated
    (via `gemini auth` or GEMINI_API_KEY env var).
    """

    _provider_name = "Gemini"
    _log_tag = "gemini-invoke"
    _install_hint = "Install Gemini CLI: https://github.com/google-gemini/gemini-cli"

    def __init__(
        self,
        *,
        agent_env: dict[str, str],
        gemini_bin: str = "gemini",
        model: Optional[str] = None,
        skill_dirs: Optional[list[Path]] = None,
    ) -> None:
        """Initialize GeminiCLI.

        Args:
            agent_env: Environment variables exposed to every Gemini
                subprocess (CLAWMEETS_AGENT_ID / CLAWMEETS_AGENT_TOKEN /
                CLAWMEETS_SERVER_URL / CLAWMEETS_AGENT_DIR). See ClaudeCLI
                for rationale. Required — pass ``{}`` explicitly to opt out.
            gemini_bin: Path to gemini CLI binary
            model: Optional model override (e.g. "gemini-2.5-pro"); None uses
                Gemini's default.
            skill_dirs: Static directories merged into ``--include-directories``
                on every invocation, so the prompt's absolute SKILL.md paths
                (under skill-hub / personal-skill-hub) resolve through
                Gemini's sandboxed file access. The plumbing is uniform
                across all skill kinds — Gemini has no equivalent to
                Claude's plugin auto-discovery.

        Gemini cannot enforce a JSON schema at the CLI level; the schema is
        embedded in the prompt (via the existing prompt builder) and parsed
        post-hoc. The schema is still passed to ``invoke(action_schema=...)``
        for symmetry with the other providers, but is ignored at this layer.
        """
        self._bin = gemini_bin
        self._model = model
        self._agent_env = dict(agent_env)
        self._skill_dirs = list(skill_dirs or [])

    def _build_env(self) -> dict[str, str]:
        env = super()._build_env()
        # Gemini CLI's trusted-folders feature refuses headless runs in an
        # untrusted cwd (exit 55: "not running in a trusted directory") —
        # and the agent's sandbox dir is never interactively trusted.
        # Trusting the workspace is the same decision --approval-mode yolo
        # already encodes.
        env.setdefault("GEMINI_CLI_TRUST_WORKSPACE", "true")
        return env

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
        action_schema: dict,
        mcp_config_dir: Optional[Path] = None,
        skill_source_dirs: Optional[list[Path]] = None,
    ) -> PreparedInvocation:
        """Set up directories, write prompt file, build command.

        ``action_schema`` is accepted for interface symmetry but not used —
        Gemini has no CLI-level schema enforcement; the schema is embedded
        in the prompt by the caller's prompt builder.
        """
        working_dir.mkdir(parents=True, exist_ok=True)

        # Gemini auto-discovers .gemini/settings.json from cwd (project scope).
        # Symlink it from the agent's pre-rendered dist so installed MCP
        # servers appear under `mcpServers`.
        if mcp_config_dir is not None:
            link_mcp_config_into(
                mcp_config_dir / ".gemini" / "settings.json",
                working_dir / ".gemini" / "settings.json",
            )

        # Gemini auto-discovers `.agents/skills/<name>/SKILL.md` from cwd
        # (workspace tier, Agent Skills open-standard alias). Materialize
        # the flat tree so gemini's native skill-loader picks up installed
        # + personal skills. --include-directories below keeps the symlink
        # *targets* (the two hub dirs) Read-accessible inside Gemini's
        # sandbox so following a symlink to its absolute target doesn't
        # trip the allow-list.
        if skill_source_dirs is not None:
            materialize_skill_tree(
                working_dir / ".agents" / "skills",
                skill_source_dirs,
            )

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
        # Static skill_dirs (skill-hub / personal-skill-hub) merge with the
        # per-invocation additional_dirs (project / knowledge bases). Dedupe
        # by resolved absolute path so a project that happens to share a
        # prefix with a skill dir doesn't appear twice.
        seen: set[str] = set()
        for d in list(self._skill_dirs) + list(additional_dirs):
            abs_d = str(d.expanduser().resolve())
            if abs_d in seen:
                continue
            seen.add(abs_d)
            cmd.extend(["--include-directories", abs_d])

        cmd.extend(["-p", full_prompt])

        # Log a sanitized command — the full prompt would be unreadably long.
        log_cmd = cmd[:-1] + [f"<prompt:{len(full_prompt)} chars from {prompt_file_abs}>"]
        logger.info(f"[gemini-invoke] START")
        logger.info(f"[gemini-invoke] command: {' '.join(log_cmd)}")
        logger.info(f"[gemini-invoke] cwd={gemini_cwd}")
        if seen:
            logger.info(f"[gemini-invoke] include-dirs={sorted(seen)}")
        logger.debug(f"[gemini-invoke] prompt content:\n{full_prompt[:500]}...")

        # stdin_bytes=None → base wires stdin to DEVNULL (prompt rides in -p).
        return PreparedInvocation(
            cmd=cmd,
            cwd=gemini_cwd,
            prompt_file_abs=prompt_file_abs,
            stdin_bytes=None,
        )

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
        return parse_json_object(response_text, log_tag="gemini-invoke")

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

        # stats.models keys are ROLES ("main"/"routing utility"), not model IDs,
        # so they can't price. Use the configured model (--llm-model) when set,
        # else the CLI's default model — a best-effort price (exact when pinned).
        del model_names  # (role keys, unused for pricing/reporting)
        priced_model = self._model or _DEFAULT_GEMINI_MODEL

        # Best-effort tool telemetry from stats.tools (shape varies by version);
        # stays empty if absent — the eval harness falls back to output-inference.
        tool_counts: dict[str, int] = {}
        tools = stats.get("tools")
        if isinstance(tools, dict):
            for tname, info in tools.items():
                if isinstance(info, dict):
                    n = info.get("count") or info.get("calls") or info.get("totalCalls")
                    if isinstance(n, int) and n > 0:
                        tool_counts[str(tname)] = n
                elif isinstance(info, int) and info > 0:
                    tool_counts[str(tname)] = info

        return LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cached,
            cache_creation_tokens=0,
            # Gemini CLI reports no cost — compute it from the captured tokens.
            # The CLI's `tokens.input` is the FRESH (uncached) count with `cached`
            # SEPARATE (cache_read routinely exceeds input via context caching), but
            # genai-prices wants the TOTAL input — so add `cached` here, else the
            # "uncached = input − cache_read" goes negative and prices to $0.0.
            cost_usd=price_usd(
                priced_model, "google", input_tokens + cached, output_tokens, cached, 0
            ),
            duration_ms=0,  # Latency is in stats but not summed here
            model=priced_model,
            tool_calls=tool_counts,
        )

    def _envelope_error_text(self, envelope: Optional[dict]) -> str:
        """Extract the envelope-level error message (str or dict.message)."""
        if envelope is None:
            return ""
        err = envelope.get("error")
        if isinstance(err, str):
            return err
        if isinstance(err, dict):
            return err.get("message", "") or json.dumps(err)
        return ""

    def _check_rate_limit(
        self,
        prepared: PreparedInvocation,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> Optional[LLMRateLimitError]:
        """Pattern-match `_RATE_LIMIT_MARKERS` against stderr and the envelope
        error message (Gemini's `gemini-cli` surfaces rate limits in either
        channel depending on whether the upstream HTTP status propagates)."""
        envelope = self._parse_envelope(stdout)
        envelope_error_text = self._envelope_error_text(envelope)

        # Auth/ineligible-tier failures are NOT rate limits — let them fall
        # through to a generic LLMInvocationError instead of being retried.
        for hay in (stderr, envelope_error_text):
            if hay and any(m in hay.lower() for m in _AUTH_MARKERS):
                return None

        for hay in (stderr, envelope_error_text):
            if not hay:
                continue
            lower = hay.lower()
            if any(marker in lower for marker in _RATE_LIMIT_MARKERS):
                return LLMRateLimitError(
                    message=f"Rate limited: {hay.strip()[:500]}",
                    rate_limit_type=None,
                )
        return None

    def _build_error_detail(
        self,
        prepared: PreparedInvocation,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> Optional[str]:
        # Auth / ineligible-tier failure → a concise, actionable message instead
        # of the raw ~1.6 KB CLI stack trace.
        if stderr and any(m in stderr.lower() for m in _AUTH_MARKERS):
            return (
                "gemini CLI auth failed: it is using the deprecated 'Gemini Code "
                "Assist for individuals' OAuth tier and ignoring GEMINI_API_KEY. "
                "Reconfigure the gemini CLI to API-key auth (or clear its cached "
                "OAuth login). gemini-api (the in-process tier) is unaffected."
            )

        envelope = self._parse_envelope(stdout)
        envelope_error_text = self._envelope_error_text(envelope)

        # Malformed envelope (no parseable JSON object) — treat as invocation
        # error so the runner doesn't ingest a half-formed result.
        if envelope is None:
            if returncode != 0:
                return (stderr[:500] if stderr else "(no error detail)")
            return f"Gemini returned non-JSON stdout:\n{stdout[:500]}"

        if returncode == 0 and not envelope_error_text:
            return None

        logger.error(f"[gemini-invoke] stdout tail: {stdout[-2000:]}")
        parts: list[str] = []
        if envelope_error_text:
            parts.append(envelope_error_text[:500])
        if stderr:
            parts.append(stderr[:500])
        return "\n".join(parts) or "(no error detail)"

    def _parse_result(
        self,
        prepared: PreparedInvocation,
        stdout: str,
        stderr: str,
    ) -> ParsedResult:
        # Reachable only when _build_error_detail returned None, which already
        # rules out envelope is None and envelope.error. Belt-and-braces guard
        # in case the envelope vanishes between calls.
        envelope = self._parse_envelope(stdout)
        if envelope is None:
            return ParsedResult(usage=LLMUsage(), actions=[], raw_text="")

        usage = self._extract_usage(envelope)

        response_text = envelope.get("response")
        if not isinstance(response_text, str):
            response_text = ""
        parsed = self._parse_response_field(response_text)

        actions: list[dict] = []
        if isinstance(parsed, dict):
            actions = normalize_actions(parsed.get("actions"))

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

        raw_output = (
            json.dumps(parsed) if parsed is not None else (response_text or "")
        )
        return ParsedResult(usage=usage, actions=actions, raw_text=raw_output)
