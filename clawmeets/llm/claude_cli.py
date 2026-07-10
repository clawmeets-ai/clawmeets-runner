# SPDX-License-Identifier: MIT
"""
clawmeets/llm/claude_cli.py
Claude Code CLI provider — subprocess invocation and parsing.

Implements LLMProvider against the `claude` CLI. Raises the generic
LLM* exceptions from clawmeets.llm.base.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ClaudeCLI
# ---------------------------------------------------------------------------


class ClaudeCLI(SubprocessLLMProvider):
    """Invokes the `claude` CLI (Claude Code) as a subprocess.

    Expects `claude` to be available on PATH.
    """

    _provider_name = "Claude Code"
    _log_tag = "claude-invoke"
    _install_hint = "Install Claude Code: https://docs.anthropic.com/claude-code"

    def __init__(
        self,
        *,
        agent_env: dict[str, str],
        claude_bin: str = "claude",
        claude_plugin_dirs: Optional[list[Path]] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize ClaudeCLI.

        Args:
            agent_env: Environment variables exposed to every Claude
                subprocess. The runner threads the agent's identity
                (CLAWMEETS_AGENT_ID / CLAWMEETS_AGENT_TOKEN /
                CLAWMEETS_SERVER_URL / CLAWMEETS_AGENT_DIR) through here
                so Bash-shelled commands inside Claude (e.g.
                ``clawmeets project create ... --token $...``) can
                authenticate as this agent. Required — pass ``{}``
                explicitly to opt out, knowing that skills depending on
                agent identity will fail.
            claude_bin: Path to claude CLI binary
            claude_plugin_dirs: Directories to load as Claude plugins via --plugin-dir
            model: If set, passed to the Claude CLI as ``--model <name>`` per
                invocation. None falls back to Claude Code's configured default.
            base_url: If set, retargets the Claude CLI at a **local** Anthropic
                Messages-API-compatible endpoint via ``ANTHROPIC_BASE_URL`` — e.g.
                ollama's ``http://localhost:11434`` (no ``/v1``), or a gateway like
                claude-code-router / LiteLLM. Lets the full Claude Code harness
                (native skills / plugins / MCP) run on a local model. None ⇒ the
                runner's default auth (subscription / ``ANTHROPIC_API_KEY``).
            api_key: Bearer token for the ``base_url`` endpoint (``ANTHROPIC_AUTH_TOKEN``
                / ``ANTHROPIC_API_KEY``). Optional — local servers like ollama accept
                any value ("required but ignored"), so a placeholder is used when None.
                Ignored unless ``base_url`` is set.

        MCP servers are surfaced via the ``mcp_config_dir`` argument to
        ``invoke()`` — the runner points at ``{agent_dir}/mcp-hub/dist/`` and
        ``_prepare_invocation`` symlinks ``.mcp.json`` into the working dir.

        The JSON action schema is selected per invocation and passed to
        ``invoke(action_schema=...)``.
        """
        self._bin = claude_bin
        self._claude_plugin_dirs = claude_plugin_dirs or []
        self._agent_env = dict(agent_env)
        self._model = model
        self._base_url = base_url or None
        self._api_key = api_key or None

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

    def _build_env(self) -> dict[str, str]:
        env = super()._build_env()
        env["CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD"] = "1"
        # Local-model route: point the CLI at a local Anthropic Messages-API
        # endpoint (ollama http://localhost:11434, claude-code-router, …). Set
        # both auth vars to cover gateways keyed on either header; ollama-style
        # servers ignore the value ("required but ignored"), so a placeholder
        # suffices when no key is configured.
        if self._base_url:
            token = self._api_key or "clawmeets-local"
            env["ANTHROPIC_BASE_URL"] = self._base_url
            env["ANTHROPIC_AUTH_TOKEN"] = token
            env["ANTHROPIC_API_KEY"] = token
        return env

    def _prepare_invocation(
        self,
        prompt: str,
        working_dir: Path,
        additional_dirs: list[Path],
        action_schema: dict,
        mcp_config_dir: Optional[Path] = None,
        skill_source_dirs: Optional[list[Path]] = None,
    ) -> PreparedInvocation:
        """Build cmd, link .mcp.json + .claude/skills from the agent dir, write prompt file."""
        working_dir.mkdir(parents=True, exist_ok=True)

        # Claude Code auto-discovers .mcp.json from cwd. Symlink it from the
        # agent's pre-rendered dist (refreshed on MCP_SYNC) so the spawned
        # subprocess picks up installed MCP servers.
        if mcp_config_dir is not None:
            link_mcp_config_into(
                mcp_config_dir / ".mcp.json",
                working_dir / ".mcp.json",
            )

        # Claude Code auto-discovers `.claude/skills/<name>/SKILL.md` from
        # cwd. Materialize a flat tree of symlinks to the agent's installed
        # + personal skills so the CLI's native skill-loader picks them up.
        if skill_source_dirs is not None:
            materialize_skill_tree(
                working_dir / ".claude" / "skills",
                skill_source_dirs,
            )

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

        return PreparedInvocation(
            cmd=cmd,
            cwd=claude_cwd,
            prompt_file_abs=prompt_file_abs,
            stdin_bytes=prompt.encode("utf-8"),
        )

    def _prepare_text_invocation(
        self, prompt: str, working_dir: Path, max_tokens: int
    ) -> PreparedInvocation:
        """One-shot plain-text completion via ``claude --print``.

        No ``--json-schema`` (so ``--print`` streams the assistant text straight
        to stdout in its default text format), no MCP/skills/add-dir, no session
        persistence — the cheapest way to turn a one-line titling prompt into a
        title. Reads the prompt from stdin like the main invoke path.
        """
        working_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._bin,
            "--print",
            "--permission-mode", "bypassPermissions",
            "--no-session-persistence",
        ]
        if self._model:
            cmd.extend(["--model", self._model])
        return PreparedInvocation(
            cmd=cmd,
            cwd=str(working_dir),
            prompt_file_abs="",
            stdin_bytes=prompt.encode("utf-8"),
        )

    def _check_rate_limit(
        self,
        prepared: PreparedInvocation,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> Optional[LLMRateLimitError]:
        """Detect Claude's rate-limit signature: returncode=0 + is_error=true + error="rate_limit"."""
        try:
            data = json.loads(stdout.strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(data, list):
            return None
        for item in data:
            if not (
                isinstance(item, dict)
                and item.get("type") == "result"
                and item.get("is_error") is True
                and item.get("error") == "rate_limit"
            ):
                continue
            result_text = item.get("result", "")
            resets_at = None
            rate_limit_type = None
            for entry in data:
                if isinstance(entry, dict) and entry.get("type") == "rate_limit_event":
                    info = entry.get("rate_limit_info", {})
                    resets_at = info.get("resetsAt")
                    rate_limit_type = info.get("rateLimitType")
                    break
            return LLMRateLimitError(
                message=f"Rate limited: {result_text}",
                resets_at=resets_at,
                rate_limit_type=rate_limit_type,
            )
        return None

    def _build_error_detail(
        self,
        prepared: PreparedInvocation,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> Optional[str]:
        """Stderr-primary detail; if the JSON output carries a top-level result
        `error` field, append it. Returns None when returncode == 0 — Claude
        does not synthesize invocation errors from successful exits (rate
        limits use their own signal)."""
        if returncode == 0:
            return None
        logger.error(f"[claude-invoke] stdout on error: {stdout[:2000]}...")
        detail = stderr or "(no stderr)"
        try:
            data = json.loads(stdout.strip())
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("type") == "result":
                        if "error" in item:
                            detail = f"{detail}\nJSON error: {item['error']}"
                        break
        except (json.JSONDecodeError, KeyError):
            pass
        return detail

    def _parse_result(
        self,
        prepared: PreparedInvocation,
        stdout: str,
        stderr: str,
    ) -> ParsedResult:
        """Parse JSON output from Claude CLI into ParsedResult.

        Aggregates actions across EVERY StructuredOutput tool_use, not just the
        last. Claude can split actions across multiple calls in one turn;
        result.structured_output only carries the most recent call, which would
        silently drop earlier actions.
        """
        try:
            data = json.loads(stdout.strip())
        except json.JSONDecodeError as e:
            logger.warning(
                f"[claude-invoke] Failed to parse JSON output: {e}, returning raw stdout"
            )
            return ParsedResult(usage=LLMUsage(), actions=[], raw_text=stdout)

        if isinstance(data, list):
            result_data = None
            for item in data:
                if isinstance(item, dict) and item.get("type") == "result":
                    result_data = item
                    break
            if not result_data:
                logger.warning("[claude-invoke] No 'result' message found in JSON array output")
                return ParsedResult(usage=LLMUsage(), actions=[], raw_text=stdout)
        elif isinstance(data, dict):
            result_data = data
        else:
            logger.warning(f"[claude-invoke] Unexpected JSON type: {type(data)}")
            return ParsedResult(usage=LLMUsage(), actions=[], raw_text=stdout)

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

        # Count real tool invocations by name (excluding the StructuredOutput
        # action-emit mechanism) for capability telemetry.
        tool_counts: dict[str, int] = {}
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict) or item.get("type") != "assistant":
                    continue
                msg = item.get("message") or {}
                content = msg.get("content", []) if isinstance(msg, dict) else []
                for c in content:
                    if not isinstance(c, dict) or c.get("type") != "tool_use":
                        continue
                    name = c.get("name") or "tool"
                    if name == "StructuredOutput":
                        continue
                    tool_counts[name] = tool_counts.get(name, 0) + 1

        usage = LLMUsage(
            input_tokens=total_input,
            output_tokens=total_output,
            cache_read_tokens=total_cache_read,
            cache_creation_tokens=total_cache_creation,
            cost_usd=result_data.get("total_cost_usd", 0.0),
            duration_ms=result_data.get("duration_ms", 0),
            model=model_name,
            requests=result_data.get("num_turns", 0) or 0,
            tool_calls=tool_counts,
        )

        logger.info(
            f"[claude-invoke] usage: cost=${usage.cost_usd:.4f} "
            f"in={usage.input_tokens} out={usage.output_tokens} "
            f"cache_read={usage.cache_read_tokens} cache_create={usage.cache_creation_tokens}"
        )

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

        # Dedup exact-duplicate actions: the model occasionally restates the same
        # reply/update_file across multiple StructuredOutput calls in one turn,
        # which the cross-call merge above would otherwise post twice (each emit
        # mints a fresh server id). Exact-match only, first-seen order preserved,
        # so genuinely distinct split actions are untouched.
        if merged_actions:
            seen: set[str] = set()
            deduped: list[dict] = []
            for a in merged_actions:
                sig = json.dumps(a, sort_keys=True, default=str)
                if sig in seen:
                    continue
                seen.add(sig)
                deduped.append(a)
            if len(deduped) != len(merged_actions):
                logger.info(
                    f"[claude-invoke] dropped {len(merged_actions) - len(deduped)} "
                    f"duplicate action(s) across StructuredOutput tool_use events"
                )
            merged_actions = deduped

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
            return ParsedResult(usage=usage, actions=merged_actions, raw_text=result_text)

        if final_actions:
            logger.info(f"[claude-invoke] structured_output.actions: {len(final_actions)} action(s)")
        return ParsedResult(usage=usage, actions=final_actions, raw_text=result_text)
