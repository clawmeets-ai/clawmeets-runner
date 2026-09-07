# SPDX-License-Identifier: MIT
"""
clawmeets/llm/antigravity_cli.py
Antigravity CLI provider — subprocess invocation and parsing.

Shells Google's `agy` binary (antigravity.google) headlessly via
``agy -p … --output-format json``. Auth is Google OAuth held in the system
keyring — a human signs `agy` in once, interactively, on each runner box and no
key flows through clawmeets. That is exactly the `opencode` model, which is why
this lands in the CLI tier (``VALID_CONFIG_PROVIDERS``) and never in
``KEYED_PROVIDERS``.

UNLIKE gemini/opencode, `agy` has a real ``--json-schema`` flag and returns a
PRE-PARSED ``structured_output`` field. So there is no ``JSON_ONLY_SUFFIX``, no
fence-stripping, and no new normalizer in ``_structured_text.py``.
``response`` is deliberately NOT read on the success path: it was observed
carrying keys OUTSIDE the supplied schema (``toolAction``/``toolSummary``)
while ``structured_output`` was correctly filtered to schema shape.
``parse_json_object`` survives only as the ``status != SUCCESS`` fallback.

TOKENS — the gemini convention, NOT opencode's. ``cache_read_tokens`` routinely
EXCEEDS ``input_tokens`` and the envelope's ``total_tokens`` EXCLUDES it. So
``total_tokens`` is ignored outright and ``cache_read_tokens`` is ADDED to the
input before pricing; otherwise genai-prices' "uncached = input − cache_read"
goes negative, raises, and is swallowed to $0.0 — the same silent-zero bug
``gemini_cli`` already carries a regression test for. Copying `opencode_cli`'s
arithmetic (which treats ``tokens.input`` as fresh with ``cache.read``
separate) under-bills by ~4.6x on a captured sample.

MCP is deliberately NOT plumbed in this version, by the user's decision:
``agy mcp add|remove|list`` exists, but the on-disk format of the config file
the runner would have to write was never captured, and `opencode`'s MCP support
was only built because that file's shape was verified first. See
``_prepare_invocation``. Consequence: an agent running on Antigravity sees no
installed MCP servers.

Raises the generic LLM* exceptions from clawmeets.llm.base.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

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

logger = logging.getLogger(__name__)

# `agy models` slugs are BARE (no `provider/` prefix, unlike opencode) and carry
# the reasoning effort as a `-high`/`-medium`/`-low` suffix — which is why the
# wrapper never passes `--effort`: effort is expressed in exactly one place.
_DEFAULT_ANTIGRAVITY_MODEL = "gemini-3.1-pro-high"

# `agy` EXITS at `--print-timeout` (its default is 5m). Our own kill window is
# `_invoke_timeout` (1800s class default, raisable per agent). Give agy strictly
# MORE than the runner so the runner's own LLMTimeoutError is always the thing
# that fires — an agy-side exit at its bound would otherwise look like a clean,
# short answer rather than a truncated turn.
_PRINT_TIMEOUT_HEADROOM_SECONDS = 300

_INSTALL_HINT = (
    "Install the Antigravity CLI: curl -fsSL https://antigravity.google/cli/install.sh | bash "
    "(installs `agy` to ~/.local/bin and appends a PATH export to your shell rc files), "
    "then run `agy` once interactively on this machine to complete the Google sign-in"
)

# Markers for detecting rate limits in stderr or the envelope's error field.
_RATE_LIMIT_MARKERS = (
    "rate_limit",
    "rate limit",
    "429",
    "too many requests",
    "quota",
    "resource_exhausted",
    "overloaded",
)

# `agy`'s catalog spans three vendors (gemini-*, claude-*, gpt-oss-*), so a
# single hardcoded provider_id would mis-price two thirds of it. This is a
# model-slug prefix map, NOT a provider enum — it adds no drift anchor. An
# unmatched slug falls through to None so genai-prices infers from the name.
_PRICE_VENDOR_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("gemini-", "google"),
    ("claude-", "anthropic"),
    ("gpt-", "openai"),
)


class AntigravityCLI(SubprocessLLMProvider):
    """Invokes the `agy` CLI (Google Antigravity) as a subprocess.

    Expects `agy` on PATH and already OAuth-signed-in on this machine.
    """

    _provider_name = "Antigravity"
    _log_tag = "antigravity-invoke"
    _install_hint = _INSTALL_HINT

    def __init__(
        self,
        *,
        agent_env: dict[str, str],
        antigravity_bin: str = "agy",
        model: Optional[str] = None,
        skill_dirs: Optional[list[Path]] = None,
    ) -> None:
        """Initialize AntigravityCLI.

        Args:
            agent_env: Environment variables exposed to every `agy` subprocess
                (CLAWMEETS_AGENT_ID / _TOKEN / _SERVER_URL / _AGENT_DIR).
                Required — pass ``{}`` explicitly to opt out.
            antigravity_bin: Path to the Antigravity CLI binary. The binary is
                ``agy``, NOT ``antigravity``.
            model: Optional BARE model slug (e.g. ``gemini-3.1-pro-high``); None
                uses agy's own configured default. No ``provider/`` prefix, and
                ``--effort`` is never passed — the slug suffix already carries it.
            skill_dirs: Static skill-content roots, materialized into the working
                dir's ``.agents/skills`` (the Agent Skills open-standard path) and
                passed via ``--add-dir`` so the prompt's absolute SKILL.md paths
                resolve regardless of how agy scopes workspace reads.
        """
        self._bin = antigravity_bin
        self._model = model
        self._agent_env = dict(agent_env)
        self._skill_dirs = list(skill_dirs or [])

    @classmethod
    def verify_cli(cls, antigravity_bin: str = "agy") -> None:
        """Verify the Antigravity CLI is available.

        Raises:
            LLMNotFoundError: If the CLI isn't on PATH.
            LLMTimeoutError: If --version times out.
            LLMInvocationError: If --version returns an error.
        """
        try:
            result = subprocess.run(
                [antigravity_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise LLMInvocationError(
                    f"Antigravity CLI returned error: {result.stderr}"
                )
            logger.info(f"Antigravity CLI verified: {result.stdout.strip()}")
        except FileNotFoundError:
            raise LLMNotFoundError(antigravity_bin, install_hint=_INSTALL_HINT)
        except subprocess.TimeoutExpired:
            raise LLMTimeoutError(timeout_seconds=10, provider="Antigravity")

    def _print_timeout_arg(self) -> str:
        """The ``--print-timeout`` value for THIS invocation, as a Go duration.

        Read at invocation time, never cached in ``__init__``: the runner stamps
        ``_invoke_timeout`` onto the *instance* after construction
        (``cli_runner._build_llm_provider``), so a value cached in the
        constructor would silently ignore a raised
        ``local_settings.invoke_timeout_seconds`` and use the 1800s class default.
        """
        return f"{self._invoke_timeout + _PRINT_TIMEOUT_HEADROOM_SECONDS}s"

    def _prepare_invocation(
        self,
        prompt: str,
        working_dir: Path,
        additional_dirs: list[Path],
        action_schema: dict,
        mcp_config_dir: Optional[Path] = None,
        skill_source_dirs: Optional[list[Path]] = None,
    ) -> PreparedInvocation:
        """Set up the working dir, write the prompt + schema files, build the command.

        Unlike gemini/opencode the schema is enforced by the binary itself
        (``--json-schema``), so no strict-JSON suffix is appended to the prompt.
        The schema goes to a *file* rather than an inline string so a large
        action schema can never hit an argv length limit.
        """
        working_dir.mkdir(parents=True, exist_ok=True)

        # `.agents/skills/<name>/SKILL.md` is the Agent Skills open-standard
        # location the codex/gemini/opencode siblings also materialize. Note
        # `--disable-slash-commands` turns off agy's own slash/skill expansion,
        # so the load-bearing mechanism here is the prompt's absolute SKILL.md
        # paths plus the `--add-dir` entries below; the tree is parity, not proof.
        if skill_source_dirs is not None:
            materialize_skill_tree(
                working_dir / ".agents" / "skills",
                skill_source_dirs,
            )

        # MCP is deferred for antigravity in this version, by the user's
        # decision: `agy mcp add|remove|list` exists, but the on-disk format of
        # the config file we would have to write was never captured, and
        # opencode's MCP support was only built because `opencode.json`'s shape
        # was verified first. Rather than invent an unverified file format the
        # setting is accepted and ignored — an antigravity-backed agent sees no
        # installed MCP servers.
        if mcp_config_dir is not None:
            logger.debug(
                f"[{self._log_tag}] mcp_config_dir={mcp_config_dir} ignored — "
                "MCP is not plumbed for antigravity in this version"
            )

        prompt_file = working_dir / ".agent-prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        prompt_file_abs = str(prompt_file.resolve())
        cwd = str(working_dir)

        cmd = [
            self._bin,
            "-p", prompt,
            "--output-format", "json",
            # Headless: no interactive approver exists, so without this every
            # tool call is denied and the turn stalls. Mirrors opencode's flag
            # of the same name and claude's `--permission-mode bypassPermissions`.
            "--dangerously-skip-permissions",
            # Our prompts QUOTE skill names, and agy would otherwise expand a
            # `/foo` appearing inside prompt text as its own slash command.
            "--disable-slash-commands",
            "--print-timeout", self._print_timeout_arg(),
        ]

        # An empty schema means the caller wants no structured output; agy
        # rejects an empty schema, so the flag is omitted entirely.
        schema_file_abs: Optional[str] = None
        if action_schema:
            schema_file = working_dir / ".agent-schema.json"
            schema_file.write_text(json.dumps(action_schema), encoding="utf-8")
            schema_file_abs = str(schema_file.resolve())
            cmd.extend(["--json-schema", schema_file_abs])

        if self._model:
            # No `--effort`: the `-high`/`-medium`/`-low` slug suffix IS the
            # reasoning effort, and expressing it twice invites disagreement.
            cmd.extend(["--model", self._model])

        # `--add-dir` pins extra workspace dirs (repeatable). Static skill_dirs
        # merge with the per-invocation additional_dirs (project / knowledge /
        # dwh / memory); dedupe by resolved absolute path. Passed defensively —
        # mirroring gemini's `--include-directories` — because whether agy
        # restricts reads to its workspace is unverified; harmless if it doesn't.
        seen: set[str] = set()
        for d in list(self._skill_dirs) + list(additional_dirs):
            abs_d = str(d.expanduser().resolve())
            if abs_d in seen:
                continue
            seen.add(abs_d)
            cmd.extend(["--add-dir", abs_d])

        # Log a sanitized command — the full prompt would be unreadably long.
        log_cmd = [
            c if c != prompt else f"<prompt:{len(prompt)} chars from {prompt_file_abs}>"
            for c in cmd
        ]
        logger.info(f"[{self._log_tag}] START")
        logger.info(f"[{self._log_tag}] command: {' '.join(log_cmd)}")
        logger.info(f"[{self._log_tag}] cwd={cwd}")
        if schema_file_abs:
            logger.info(f"[{self._log_tag}] schema file saved at: {schema_file_abs}")
        if seen:
            logger.info(f"[{self._log_tag}] add-dirs={sorted(seen)}")
        logger.debug(f"[{self._log_tag}] prompt content:\n{prompt[:500]}...")

        # stdin_bytes=None → base wires stdin to DEVNULL (prompt rides in -p).
        return PreparedInvocation(
            cmd=cmd,
            cwd=cwd,
            prompt_file_abs=prompt_file_abs,
            stdin_bytes=None,
        )

    def _envelope(self, stdout: str) -> Optional[dict]:
        """Parse agy's single JSON envelope from stdout, tolerating noise.

        ``--output-format json`` emits exactly one object, but a stray banner or
        trailing newline shouldn't cost us the whole turn: try the whole string
        first, then the first line that parses to a dict.
        """
        text = stdout.strip()
        if not text:
            return None
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        logger.warning(f"[{self._log_tag}] no parseable JSON envelope in stdout")
        return None

    def _priced_provider_id(self, slug: str) -> Optional[str]:
        """genai-prices provider id for a bare agy slug, or None to let it infer."""
        for prefix, vendor in _PRICE_VENDOR_BY_PREFIX:
            if slug.startswith(prefix):
                return vendor
        return None

    def _extract_usage(self, envelope: dict) -> LLMUsage:
        """Build LLMUsage from the envelope's ``usage`` block.

        The gemini cache convention: ``input_tokens`` is FRESH/uncached with
        ``cache_read_tokens`` SEPARATE (and routinely larger), and the envelope's
        ``total_tokens`` EXCLUDES the cache read. genai-prices wants the TOTAL
        input, so cache_read is ADDED here; ``total_tokens`` is ignored outright
        rather than trusted as an input figure.
        """
        usage = envelope.get("usage")
        if not isinstance(usage, dict):
            return LLMUsage(model=self._model or _DEFAULT_ANTIGRAVITY_MODEL)

        def _n(key: str) -> int:
            v = usage.get(key, 0)
            return v if isinstance(v, int) else 0

        cache_read = _n("cache_read_tokens")
        total_input = _n("input_tokens") + cache_read
        # Thinking tokens bill as output, matching opencode's `reasoning`.
        total_output = _n("output_tokens") + _n("thinking_tokens")

        priced_model = self._model or _DEFAULT_ANTIGRAVITY_MODEL
        duration = envelope.get("duration_seconds")
        turns = envelope.get("num_turns")

        return LLMUsage(
            input_tokens=total_input,
            output_tokens=total_output,
            cache_read_tokens=cache_read,
            cache_creation_tokens=0,   # agy reports no cache-write figure
            # agy surfaces no USD field — the gemini/codex precedent; pricing.py
            # handles it with no new code path.
            cost_usd=price_usd(
                priced_model,
                self._priced_provider_id(priced_model),
                total_input,
                total_output,
                cache_read,
                0,
            ),
            duration_ms=int(duration * 1000) if isinstance(duration, (int, float)) else 0,
            model=priced_model,
            requests=turns if isinstance(turns, int) else 0,
            # The `json` envelope carries no tool list (only `stream-json` does),
            # so tool telemetry stays empty and the eval harness falls back to
            # output-inference, same as gemini before stats.tools.
            tool_calls={},
        )

    def _extract_actions(self, envelope: dict) -> tuple[list[dict], str]:
        """Return (actions, raw_reply_text) from a completed envelope.

        On SUCCESS the pre-parsed, schema-filtered ``structured_output`` is
        authoritative. ``response`` is NOT read on that path — it was observed
        carrying keys outside the supplied schema. It is only fallen back to
        (via ``parse_json_object``) when the status is not SUCCESS, where no
        ``structured_output`` is emitted at all.
        """
        response = envelope.get("response")
        raw_text = response if isinstance(response, str) else ""

        if envelope.get("status") == "SUCCESS":
            structured = envelope.get("structured_output")
            if isinstance(structured, dict):
                return normalize_actions(structured.get("actions")), raw_text

        parsed = parse_json_object(raw_text, log_tag=self._log_tag)
        if isinstance(parsed, dict):
            return normalize_actions(parsed.get("actions")), raw_text
        return [], raw_text

    def _collect_error_text(self, envelope: Optional[dict], stderr: str) -> str:
        """Gather any explicit error message from the envelope and stderr."""
        parts: list[str] = []
        if envelope is not None:
            err = envelope.get("error")
            if isinstance(err, str) and err:
                parts.append(err)
            elif isinstance(err, dict):
                parts.append(err.get("message") or json.dumps(err))
            status = envelope.get("status")
            if isinstance(status, str) and status and status != "SUCCESS":
                parts.append(f"status={status}")
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
        hay = self._collect_error_text(self._envelope(stdout), stderr).lower()
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
        envelope = self._envelope(stdout)
        error_text = self._collect_error_text(envelope, stderr)

        if returncode != 0:
            return error_text[:1000] if error_text else "(no error detail)"
        if envelope is None:
            return f"unparseable agy envelope: {stdout[:500]}"
        # A clean exit can still be a logical failure — agy reports that in
        # `status`, and `structured_output` is absent on those runs.
        if envelope.get("status") != "SUCCESS":
            return error_text[:1000] or "(no error detail)"
        return None

    def _parse_result(
        self,
        prepared: PreparedInvocation,
        stdout: str,
        stderr: str,
    ) -> ParsedResult:
        # Reachable only when _build_error_detail returned None, which already
        # rules out a missing envelope. Belt-and-braces guard regardless.
        envelope = self._envelope(stdout)
        if envelope is None:
            return ParsedResult(usage=LLMUsage(), actions=[], raw_text="")

        usage = self._extract_usage(envelope)
        actions, raw_text = self._extract_actions(envelope)

        if actions:
            logger.info(f"[{self._log_tag}] parsed {len(actions)} action(s)")
        else:
            logger.warning(
                f"[{self._log_tag}] no actions parsed — response head: {raw_text[:300]!r}"
            )
        logger.info(
            f"[{self._log_tag}] usage: in={usage.input_tokens} "
            f"out={usage.output_tokens} cache_read={usage.cache_read_tokens} "
            f"cost=${usage.cost_usd:.4f}"
        )
        return ParsedResult(usage=usage, actions=actions, raw_text=raw_text)
