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

import copy
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

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

# Codex CLI's default model family, used to price token usage when codex reports
# neither a cost nor a model name and no --llm-model is set (best-effort; exact
# when codex names the model or --llm-model is pinned).
_DEFAULT_CODEX_MODEL = "gpt-5"


# Markers for detecting rate limits in error events. Codex surfaces provider
# errors as JSONL events with free-form messages, so we pattern-match.
_RATE_LIMIT_MARKERS = ("rate_limit", "rate limit", "429", "too many requests")


def _ensure_codex_project_trust(working_dir: Path) -> None:
    """Auto-trust ``working_dir`` in ``~/.codex/config.toml`` so codex loads
    its project-level ``.codex/config.toml`` (including our symlinked
    ``[mcp_servers.*]`` blocks).

    Codex's discovery rules ignore project config unless the project path
    appears as ``[projects."<abs-path>"] trust_level = "trusted"`` in the
    user's user-config — normally established interactively on first run.
    We elide the prompt by appending the block when absent.

    Skipped if ``~/.codex/config.toml`` doesn't exist yet (fresh codex
    install — the user's first interactive run will create it and the trust
    prompt mechanism will work as designed).
    """
    user_config = Path.home() / ".codex" / "config.toml"
    if not user_config.exists():
        return
    abs_path = str(working_dir.resolve())
    header = f'[projects."{abs_path}"]'
    try:
        existing = user_config.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning(f"[codex-invoke] could not read {user_config}: {e}")
        return
    if header in existing:
        return
    suffix = "" if existing.endswith("\n") else "\n"
    suffix += f'\n{header}\ntrust_level = "trusted"\n'
    try:
        with open(user_config, "a", encoding="utf-8") as f:
            f.write(suffix)
    except OSError as e:
        logger.warning(f"[codex-invoke] could not append trust entry to {user_config}: {e}")
        return
    logger.info(f"[codex-invoke] auto-trusted codex project: {abs_path}")


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


class CodexCLI(SubprocessLLMProvider):
    """Invokes the `codex` CLI (OpenAI Codex) as a subprocess.

    Expects `codex` to be available on PATH and authenticated
    (via `codex login` or OPENAI_API_KEY env var).
    """

    _provider_name = "Codex"
    _log_tag = "codex-invoke"
    _install_hint = "Install Codex: https://github.com/openai/codex"

    def __init__(
        self,
        *,
        agent_env: dict[str, str],
        codex_bin: str = "codex",
        model: Optional[str] = None,
        sandbox_mode: str = "workspace-write",
    ) -> None:
        """Initialize CodexCLI.

        Args:
            agent_env: Environment variables exposed to every Codex
                subprocess (CLAWMEETS_AGENT_ID / CLAWMEETS_AGENT_TOKEN /
                CLAWMEETS_SERVER_URL / CLAWMEETS_AGENT_DIR). See ClaudeCLI
                for rationale. Required — pass ``{}`` explicitly to opt out.
            codex_bin: Path to codex CLI binary
            model: Optional model override (e.g. "o3"); None uses Codex default.
            sandbox_mode: Codex sandbox policy. Default "workspace-write" lets
                the agent modify its own sandbox (clawmeets already isolates
                sandboxes per agent). Other options: "read-only",
                "danger-full-access".

        The JSON action schema is selected per invocation and passed to
        ``invoke(action_schema=...)``.
        """
        self._bin = codex_bin
        self._model = model
        self._sandbox_mode = sandbox_mode
        self._agent_env = dict(agent_env)

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
        action_schema: dict,
        mcp_config_dir: Optional[Path] = None,
        skill_source_dirs: Optional[list[Path]] = None,
    ) -> PreparedInvocation:
        """Set up directories, write prompt + schema files, build command."""
        working_dir.mkdir(parents=True, exist_ok=True)

        # Codex loads project-level .codex/config.toml from cwd (or any
        # ancestor) when the project is "trusted" in ~/.codex/config.toml.
        # Symlink the rendered config from the agent's pre-rendered dist,
        # then idempotently auto-trust the sandbox path so the prompt is
        # elided for fresh sandboxes.
        if mcp_config_dir is not None:
            link_mcp_config_into(
                mcp_config_dir / ".codex" / "config.toml",
                working_dir / ".codex" / "config.toml",
            )
            _ensure_codex_project_trust(working_dir)

        # Codex auto-discovers `.agents/skills/<name>/SKILL.md` from cwd
        # (Agent Skills open-standard path). Materialize the flat tree so
        # codex's native skill-loader picks up installed + personal skills.
        if skill_source_dirs is not None:
            materialize_skill_tree(
                working_dir / ".agents" / "skills",
                skill_source_dirs,
            )

        prompt_file = working_dir / ".agent-prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        prompt_file_abs = str(prompt_file.resolve())

        schema_file = working_dir / ".agent-schema.json"
        adapted_schema = _adapt_schema_for_codex(action_schema)
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

        return PreparedInvocation(
            cmd=cmd,
            cwd=codex_cwd,
            prompt_file_abs=prompt_file_abs,
            stdin_bytes=prompt.encode("utf-8"),
            extras={"last_message_abs": last_message_abs},
        )

    def _parse_events(self, raw_output: str) -> tuple[LLMUsage, list[dict]]:
        """Parse Codex JSONL event stream for usage + error events.

        Returns:
            (usage, error_events) — usage is empty LLMUsage if no
            turn.completed seen; error_events is a list of error payloads.
        """
        usage = LLMUsage()
        errors: list[dict] = []
        # Best-effort tool telemetry: count Responses-style tool/command items
        # (`item.completed` events). Stays empty if the schema differs, in which
        # case the eval harness falls back to output-inference.
        tool_counts: dict[str, int] = {}
        _tool_item_types = {
            "function_call", "local_shell_call", "web_search_call",
            "mcp_tool_call", "command_execution", "tool_call",
        }

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
                in_tok = u.get("input_tokens", 0)
                out_tok = u.get("output_tokens", 0)
                cached = u.get("cached_input_tokens", 0)
                # Prefer codex's reported cost; it usually omits it, so fall back
                # to pricing the captured tokens. Model: what codex reports, else
                # the configured --llm-model, else the CLI default (best-effort).
                model = (event.get("model", "") or u.get("model", "")
                         or self._model or _DEFAULT_CODEX_MODEL)
                reported = u.get("total_cost_usd", 0.0) or 0.0
                cost = reported if reported > 0 else price_usd(
                    model, "openai", in_tok, out_tok, cached, 0)
                usage = LLMUsage(
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cache_read_tokens=cached,
                    cache_creation_tokens=0,  # Codex doesn't surface this
                    cost_usd=cost,
                    duration_ms=event.get("duration_ms", 0),
                    model=model,
                )
            elif etype in ("error", "turn.failed"):
                errors.append(event)
            elif etype in ("item.completed", "item.done"):
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") in _tool_item_types:
                    nm = item.get("name") or item.get("type")
                    tool_counts[nm] = tool_counts.get(nm, 0) + 1

        usage.tool_calls = tool_counts
        return usage, errors

    def _check_rate_limit(
        self,
        prepared: PreparedInvocation,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> Optional[LLMRateLimitError]:
        """Codex surfaces provider errors as JSONL events with free-form
        messages — pattern-match against `_RATE_LIMIT_MARKERS`."""
        _, error_events = self._parse_events(stdout)

        def _matches(text: str) -> bool:
            lower = text.lower()
            return any(marker in lower for marker in _RATE_LIMIT_MARKERS)

        for event in error_events:
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

    def _build_error_detail(
        self,
        prepared: PreparedInvocation,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> Optional[str]:
        _, error_events = self._parse_events(stdout)
        if returncode == 0 and not error_events:
            return None
        logger.error(f"[codex-invoke] stdout tail: {stdout[-2000:]}")
        parts: list[str] = []
        for ev in error_events[:3]:
            parts.append(json.dumps(ev)[:500])
        if stderr:
            parts.append(stderr[:500])
        return "\n".join(parts) or "(no error detail)"

    def _parse_result(
        self,
        prepared: PreparedInvocation,
        stdout: str,
        stderr: str,
    ) -> ParsedResult:
        usage, _ = self._parse_events(stdout)

        # Read the schema-conformant final message from Codex's -o file.
        final_message: Optional[dict] = None
        last_message_path = Path(prepared.extras["last_message_abs"])
        if last_message_path.exists():
            try:
                content = last_message_path.read_text(encoding="utf-8").strip()
                if content:
                    final_message = json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"[codex-invoke] last message is not valid JSON: {e}")

        actions: list[dict] = []
        if isinstance(final_message, dict):
            raw_actions = final_message.get("actions")
            if isinstance(raw_actions, list):
                actions = raw_actions

        if actions:
            logger.info(f"[codex-invoke] final message actions: {len(actions)} action(s)")
        else:
            logger.warning(f"[codex-invoke] no actions parsed from final message")

        logger.info(
            f"[codex-invoke] usage: cost=${usage.cost_usd:.4f} "
            f"in={usage.input_tokens} out={usage.output_tokens} "
            f"cache_read={usage.cache_read_tokens}"
        )

        raw_output = json.dumps(final_message) if final_message is not None else ""
        return ParsedResult(usage=usage, actions=actions, raw_text=raw_output)
