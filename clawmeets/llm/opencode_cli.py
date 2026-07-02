# SPDX-License-Identifier: MIT
"""
clawmeets/llm/opencode_cli.py
OpenCode CLI provider — subprocess invocation and parsing.

Implements LLMProvider against the `opencode` CLI (opencode.ai), shelling its
headless `opencode run` command. The draw is OpenCode Zen — a hosted gateway
(incl. a rotating pool of *free* models) whose auth the binary manages itself
(`opencode auth login` → ~/.local/share/opencode/auth.json), so no key flows
through clawmeets. Models are addressed as ``<provider>/<model>`` slugs (e.g.
``opencode/deepseek-v4-flash-free``).

Like Gemini — and unlike Claude/Codex — OpenCode has no schema-enforcement flag.
`--format json` only streams raw JSONL *events*; it does not constrain output.
So we use the Gemini pattern: the prompt builder already renders the
``== OUTPUT CONTRACT ==`` block, we append a strict "JSON only" suffix, then
parse the final assistant text out of the event stream and json.loads it.

Event stream (`opencode run ... --format json`), one JSON object per line:
  - ``type:"text"``        → ``part.text``   (the assistant reply; concatenated)
  - ``type:"tool_use"``    → ``part.tool`` / ``part.callID`` (counted per call)
  - ``type:"step_finish"`` → ``part.tokens`` {total,input,output,reasoning,
                             cache:{write,read}} + ``part.cost``. Multi-step
                             turns emit several — summed.

Token convention (verified against real runs): ``total = input + output +
reasoning + cache.read``, i.e. ``input`` is FRESH/uncached and ``cache.read`` is
SEPARATE (cache_read can exceed input). genai-prices wants TOTAL input, so we add
cache.read before pricing — same handling as the gemini CLI. Cost comes from
``part.cost`` when present (0 on free models); otherwise a best-effort price.

Raises the generic LLM* exceptions from clawmeets.llm.base.
"""
from __future__ import annotations

import json
import logging
import subprocess
from collections import Counter
from pathlib import Path
from typing import Optional

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
    materialize_skill_tree,
)
from .pricing import price_usd
from ..utils.file_io import FileUtil

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Install OpenCode: brew install anomalyco/tap/opencode  "
    "(or: curl -fsSL https://opencode.ai/install | bash), then `opencode auth login`"
)

# Markers for detecting rate limits in stderr or error events.
_RATE_LIMIT_MARKERS = (
    "rate_limit",
    "rate limit",
    "429",
    "too many requests",
    "quota",
    "resource_exhausted",
    "overloaded",
)


def _translate_mcp_to_opencode(mcp_servers: dict) -> dict:
    """Translate a Claude-format ``{name: server}`` map into opencode's ``mcp``
    schema. opencode (verified via ``opencode mcp add --help`` + the binary):
    ``type: "local"`` = command-based (``command`` is an ARRAY, optional
    ``environment``); ``type: "remote"`` = ``url``-based (optional ``headers``);
    both carry ``"enabled": true``. Claude's ``.mcp.json`` uses a string
    ``command`` + separate ``args`` array, so we join them into one array."""
    out: dict = {}
    for name, cfg in (mcp_servers or {}).items():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("url"):
            entry = {"type": "remote", "url": cfg["url"], "enabled": True}
            if cfg.get("headers"):
                entry["headers"] = cfg["headers"]
        elif cfg.get("command"):
            command = [cfg["command"], *cfg.get("args", [])]
            entry = {"type": "local", "command": command, "enabled": True}
            if cfg.get("env"):
                entry["environment"] = cfg["env"]
        else:
            continue  # neither url nor command → not a launchable server
        out[name] = entry
    return out


def _write_opencode_mcp_config(mcp_config_dir: Optional[Path], working_dir: Path) -> None:
    """Surface installed MCP servers to opencode by translating the rendered
    Claude ``.mcp.json`` into ``working_dir/opencode.json`` (opencode reads its
    project config from the ``--dir`` root). No-op when there's no config dir,
    no file, or no servers — so a normal sandbox stays config-free. Mirrors the
    sibling CLIs' ``link_mcp_config_into`` (claude/codex/gemini), which opencode
    previously lacked."""
    if mcp_config_dir is None:
        return
    raw = FileUtil.read(mcp_config_dir / ".mcp.json", "json")
    servers = (raw or {}).get("mcpServers") if isinstance(raw, dict) else None
    translated = _translate_mcp_to_opencode(servers) if servers else {}
    if not translated:
        return
    FileUtil.write(
        working_dir / "opencode.json",
        {"$schema": "https://opencode.ai/config.json", "mcp": translated},
        "json",
    )


class OpenCodeCLI(SubprocessLLMProvider):
    """Invokes the `opencode` CLI (opencode.ai) as a subprocess.

    Expects `opencode` to be on PATH and authenticated (`opencode auth login`).
    """

    _provider_name = "OpenCode"
    _log_tag = "opencode-invoke"
    _install_hint = _INSTALL_HINT

    def __init__(
        self,
        *,
        agent_env: dict[str, str],
        opencode_bin: str = "opencode",
        model: Optional[str] = None,
        skill_dirs: Optional[list[Path]] = None,
    ) -> None:
        """Initialize OpenCodeCLI.

        Args:
            agent_env: Environment variables exposed to every opencode
                subprocess (CLAWMEETS_AGENT_ID / _TOKEN / _SERVER_URL /
                _AGENT_DIR). Required — pass ``{}`` explicitly to opt out.
            opencode_bin: Path to the opencode CLI binary.
            model: Optional ``<provider>/<model>`` slug (e.g.
                ``opencode/deepseek-v4-flash-free``); None uses opencode's own
                configured default.
            skill_dirs: Static skill-content roots; materialized into the
                working dir's ``.agents/skills`` so the prompt's absolute
                SKILL.md paths resolve. opencode's tools read absolute paths
                unrestricted (verified), so no include-directories plumbing is
                needed — unlike Gemini.

        opencode cannot enforce a JSON schema at the CLI level; the schema is
        embedded in the prompt (by the existing prompt builder) and parsed
        post-hoc. ``action_schema`` is still accepted by ``invoke()`` for
        symmetry with the other providers but is ignored at this layer.
        """
        self._bin = opencode_bin
        self._model = model
        self._agent_env = dict(agent_env)
        self._skill_dirs = list(skill_dirs or [])

    @classmethod
    def verify_cli(cls, opencode_bin: str = "opencode") -> None:
        """Verify the OpenCode CLI is available.

        Raises:
            LLMNotFoundError: If the CLI isn't on PATH.
            LLMTimeoutError: If --version times out.
            LLMInvocationError: If --version returns an error.
        """
        try:
            result = subprocess.run(
                [opencode_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise LLMInvocationError(
                    f"OpenCode CLI returned error: {result.stderr}"
                )
            logger.info(f"OpenCode CLI verified: {result.stdout.strip()}")
        except FileNotFoundError:
            raise LLMNotFoundError(opencode_bin, install_hint=_INSTALL_HINT)
        except subprocess.TimeoutExpired:
            raise LLMTimeoutError(timeout_seconds=10, provider="OpenCode")

    def _prepare_invocation(
        self,
        prompt: str,
        working_dir: Path,
        additional_dirs: list[Path],
        action_schema: dict,
        mcp_config_dir: Optional[Path] = None,
        skill_source_dirs: Optional[list[Path]] = None,
    ) -> PreparedInvocation:
        """Set up the working dir, write the prompt file, build the command.

        ``action_schema`` is accepted for interface symmetry but not used —
        opencode has no CLI-level schema enforcement; the schema is embedded in
        the prompt by the caller's prompt builder. ``additional_dirs`` is
        ignored: opencode's tools read absolute paths without an allow-list, so
        no per-invocation directory flag is required (and `opencode run` exposes
        none). ``mcp_config_dir`` is translated into ``opencode.json`` (below).
        """
        working_dir.mkdir(parents=True, exist_ok=True)

        # opencode discovers `.agents/skills/<name>/SKILL.md` from cwd (Agent
        # Skills open-standard alias). Materialize the flat tree so installed +
        # personal skills are present; the prompt also carries absolute paths
        # opencode can read directly.
        if skill_source_dirs is not None:
            materialize_skill_tree(
                working_dir / ".agents" / "skills",
                skill_source_dirs,
            )

        # Translate installed MCP servers (rendered Claude `.mcp.json`) into
        # opencode's own `opencode.json` project config, which it reads from the
        # `--dir` root. Parity with claude/codex/gemini, which opencode lacked.
        _write_opencode_mcp_config(mcp_config_dir, working_dir)

        full_prompt = prompt + _JSON_ONLY_SUFFIX

        prompt_file = working_dir / ".agent-prompt.txt"
        prompt_file.write_text(full_prompt, encoding="utf-8")
        prompt_file_abs = str(prompt_file.resolve())
        cwd = str(working_dir)

        # Prompt rides as a positional arg (no stdin); --format json streams
        # JSONL events that _parse_stream consumes. `--dir` pins opencode's
        # working directory explicitly — it resolves cwd from $PWD, which
        # asyncio's create_subprocess_exec does NOT update from the `cwd=`
        # arg, so without this opencode reads/writes in the runner's stale PWD
        # instead of the sandbox.
        # `--dangerously-skip-permissions` auto-approves tool use: in headless
        # `run` mode opencode's permission gate denies every tool call (no
        # interactive approver), so without it the agent can't even read the
        # roster and the project stalls on turn one. Safe here — clawmeets
        # already isolates each agent in its own sandbox dir (pinned via --dir)
        # and reads are unrestricted by the harness design. Mirrors claude's
        # `--permission-mode bypassPermissions` and codex's `--sandbox
        # workspace-write`.
        cmd = [self._bin, "run", full_prompt, "--format", "json",
               "--dangerously-skip-permissions", "--dir", cwd]
        if self._model:
            cmd.extend(["-m", self._model])

        log_cmd = [self._bin, "run", f"<prompt:{len(full_prompt)} chars from "
                   f"{prompt_file_abs}>", "--format", "json",
                   "--dangerously-skip-permissions", "--dir", cwd]
        if self._model:
            log_cmd.extend(["-m", self._model])
        logger.info(f"[opencode-invoke] START")
        logger.info(f"[opencode-invoke] command: {' '.join(log_cmd)}")
        logger.info(f"[opencode-invoke] cwd={cwd}")
        logger.debug(f"[opencode-invoke] prompt content:\n{full_prompt[:500]}...")

        # stdin_bytes=None → base wires stdin to DEVNULL (prompt rides positionally).
        return PreparedInvocation(
            cmd=cmd,
            cwd=cwd,
            prompt_file_abs=prompt_file_abs,
            stdin_bytes=None,
        )

    def _iter_events(self, stdout: str):
        """Yield parsed JSON objects from the JSONL stream, tolerating noise."""
        for line in stdout.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(ev, dict):
                yield ev

    def _parse_stream(self, stdout: str) -> tuple[LLMUsage, list[dict], str]:
        """Parse the JSONL event stream into (usage, actions, raw_reply_text)."""
        text_parts: list[str] = []
        input_t = output_t = reasoning_t = cache_read = cache_write = 0
        cost = 0.0
        n_steps = 0
        tool_by_call: dict[str, str] = {}

        for ev in self._iter_events(stdout):
            etype = ev.get("type")
            part = ev.get("part")
            if not isinstance(part, dict):
                part = {}

            if etype == "text":
                t = part.get("text")
                if isinstance(t, str):
                    text_parts.append(t)
            elif etype == "step_finish":
                n_steps += 1
                tokens = part.get("tokens")
                if isinstance(tokens, dict):
                    input_t += tokens.get("input", 0) or 0
                    output_t += tokens.get("output", 0) or 0
                    reasoning_t += tokens.get("reasoning", 0) or 0
                    cache = tokens.get("cache")
                    if isinstance(cache, dict):
                        cache_read += cache.get("read", 0) or 0
                        cache_write += cache.get("write", 0) or 0
                c = part.get("cost")
                if isinstance(c, (int, float)):
                    cost += float(c)
            elif etype == "tool_use":
                tool = part.get("tool")
                if isinstance(tool, str) and tool:
                    # Dedupe by callID so a running+completed pair for one call
                    # counts once (last write wins).
                    call_id = part.get("callID") or f"_{len(tool_by_call)}"
                    tool_by_call[str(call_id)] = tool

        reply_text = "".join(text_parts)
        priced_model = self._model or ""
        # genai-prices wants TOTAL input (cache_read is a subset); opencode
        # reports input as FRESH with cache.read SEPARATE, so add it. Reasoning
        # bills as output. Zen slugs are typically unpriceable → ~0 (expected;
        # free models report cost 0 directly anyway).
        total_input = input_t + cache_read
        total_output = output_t + reasoning_t
        cost_usd = cost if cost > 0 else price_usd(
            priced_model, None, total_input, total_output, cache_read, cache_write
        )

        usage = LLMUsage(
            input_tokens=total_input,
            output_tokens=total_output,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_write,
            cost_usd=cost_usd,
            duration_ms=0,
            model=priced_model,
            requests=n_steps,
            tool_calls=dict(Counter(tool_by_call.values())),
        )

        parsed = parse_json_object(reply_text, log_tag="opencode-invoke")
        actions: list[dict] = []
        if isinstance(parsed, dict):
            actions = normalize_actions(parsed.get("actions"))
        return usage, actions, reply_text

    def _collect_error_text(self, stdout: str, stderr: str) -> str:
        """Gather any explicit error message from error events / stderr."""
        parts: list[str] = []
        for ev in self._iter_events(stdout):
            if ev.get("type") == "error" or ev.get("error"):
                err = ev.get("error") or ev.get("message") or ev
                parts.append(err if isinstance(err, str) else json.dumps(err))
        if stderr:
            parts.append(stderr)
        return "\n".join(parts)

    def _check_rate_limit(
        self,
        prepared: PreparedInvocation,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> Optional[LLMRateLimitError]:
        hay = self._collect_error_text(stdout, stderr).lower()
        if hay and any(m in hay for m in _RATE_LIMIT_MARKERS):
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
        error_text = self._collect_error_text(stdout, stderr)
        if returncode != 0:
            return (error_text[:1000] if error_text else "(no error detail)")
        # returncode == 0 but an explicit error event surfaced → still a failure.
        if self._has_error_event(stdout):
            return error_text[:1000] or "(no error detail)"
        return None

    def _has_error_event(self, stdout: str) -> bool:
        return any(
            ev.get("type") == "error" or ev.get("error")
            for ev in self._iter_events(stdout)
        )

    def _parse_result(
        self,
        prepared: PreparedInvocation,
        stdout: str,
        stderr: str,
    ) -> ParsedResult:
        usage, actions, reply_text = self._parse_stream(stdout)

        if actions:
            logger.info(f"[opencode-invoke] parsed {len(actions)} action(s)")
        else:
            logger.warning(
                f"[opencode-invoke] no actions parsed — reply head: "
                f"{reply_text[:300]!r}"
            )
        logger.info(
            f"[opencode-invoke] usage: in={usage.input_tokens} "
            f"out={usage.output_tokens} cache_read={usage.cache_read_tokens} "
            f"cost=${usage.cost_usd:.4f}"
        )
        return ParsedResult(usage=usage, actions=actions, raw_text=reply_text)
