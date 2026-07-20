# SPDX-License-Identifier: MIT
"""
clawmeets/llm/native_harness.py
The ``openrouter-native`` provider — a native, in-process tool-calling loop over
OpenRouter's OpenAI-compatible Chat Completions API, owned by clawmeets (no
Pydantic-AI).

It conforms exactly to :class:`~clawmeets.llm.base.LLMProvider` (``invoke`` →
``(ActionBlock, LLMUsage)``, ``verify_cli``, ``generate_text``), so no caller
changes. It coexists with the untouched ``openrouter-api`` (Pydantic-AI) path as a
directly A/B-comparable alternative on the same model + key.

The loop offers the real harness tools (file / bash / skill / web) so the agent does
work, plus the ``emit_*`` / ``finalize`` "JSON via tools" surface so the action-block
JSON comes from tool schemas, not model formatting. The wire deltas (string tool
args, null content) are owned by :mod:`clawmeets.llm._openai_wire`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from ..api.actions import ActionBlock, COORDINATOR_ACTION_SCHEMA
from ..utils import env_store
from ..utils.notification_center import LLM_COMPLETE, LLM_ERROR, NotificationCenter
from ._harness_tools import (
    FINALIZE_TOOL_NAME,
    HarnessTool,
    assemble,
    build_emitter_schemas,
    build_harness_tools,
    build_mcp_tools,
    emitters_for,
)
from ._openai_wire import AssistantTurn, OpenAIChatClient, ToolCall, _coerce_content
from .base import (
    LLMInvocationError,
    LLMNotFoundError,
    LLMProvider,
    LLMTimeoutError,
    LLMUsage,
)
from .pricing import price_openrouter_usd

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "qwen/qwen-2.5-72b-instruct"


class NativeHarnessProvider(LLMProvider):
    """In-process, BYO-key OpenRouter provider that owns its own tool-calling loop."""

    _log_tag = "native-invoke"
    _provider_name = "OpenRouter (native harness)"
    _install_hint = "set OPENROUTER_API_KEY (or local_settings.llm_api_key)"
    _text_gen_timeout: int = 60

    def __init__(
        self,
        *,
        api_key: str,
        agent_env: dict[str, str],
        model: Optional[str] = None,
        base_url: str = _DEFAULT_BASE_URL,
        # Per-request completion (output) ceiling. Kept modest on purpose: this
        # provider targets smaller-context OpenRouter models (e.g. qwen-2.5-72b's
        # 32768-token context), where a huge output cap would starve the input and
        # 400. An action turn needs little output; ``min_output_tokens`` floors it.
        max_tokens: int = 4_096,
        max_requests: int = 50,
        max_total_tokens: Optional[int] = None,
        max_turns: int = 40,
        enable_web: bool = True,
        web_max_uses: int = 5,
        coordinator_web_max_uses: int = 2,
        web_fetch_max_uses: int = 12,
        min_output_tokens: int = 512,
        referer: Optional[str] = None,
        title: Optional[str] = None,
        **_ignored_caps: object,
    ) -> None:
        # ``_ignored_caps`` swallows forward cap keys the ``-api`` rail forwards but
        # this provider does not use (e.g. reasoning_effort / output_mode), so the
        # shared ``_API_CAP_KEYS`` factory call never raises a TypeError.
        self._api_key = api_key or ""
        self._agent_env = agent_env or {}
        self._model_name = model or _DEFAULT_MODEL
        self._base_url = base_url or _DEFAULT_BASE_URL
        self._max_tokens = max_tokens
        self._max_requests = max_requests
        self._max_total_tokens = max_total_tokens
        self._max_turns = max_turns
        self._enable_web = enable_web
        self._web_max_uses = web_max_uses
        self._coordinator_web_max_uses = coordinator_web_max_uses
        self._web_fetch_max_uses = web_fetch_max_uses
        self._min_output_tokens = min_output_tokens
        self._referer = referer
        self._title = title
        self._client_obj: Optional[OpenAIChatClient] = None

    # --- preflight ----------------------------------------------------------

    @classmethod
    def verify_cli(cls) -> None:
        """Assert httpx is importable. Key presence is checked lazily in ``invoke``
        so a local misconfig doesn't stop the runner from booting."""
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise LLMNotFoundError("httpx", install_hint=cls._install_hint) from exc

    # --- client -------------------------------------------------------------

    def _models(self) -> list[str]:
        """Model chain from a (possibly comma-separated) ``llm_model``; non-empty."""
        models = [m.strip() for m in (self._model_name or "").split(",") if m.strip()]
        return models or [self._model_name]

    def _client(self) -> OpenAIChatClient:
        """Build/cache the OpenAIChatClient (one per provider, not per invocation)."""
        if self._client_obj is None:
            if not self._api_key:
                raise LLMInvocationError(
                    f"{self._provider_name}: no API key "
                    "(set OPENROUTER_API_KEY or local_settings.llm_api_key)."
                )
            self._client_obj = OpenAIChatClient(
                base_url=self._base_url,
                api_key=self._api_key,
                models=self._models(),
                referer=self._referer,
                title=self._title,
            )
        return self._client_obj

    def _env(self) -> dict[str, str]:
        """Process env + runner-local env store + agent identity (CLAWMEETS_* wins).

        Mirrors ``SubprocessLLMProvider._build_env`` so the reused bash/skill tools
        see the same environment the CLI tier would.
        """
        env = os.environ.copy()
        agent_dir = self._agent_env.get("CLAWMEETS_AGENT_DIR")
        if agent_dir:
            env.update(env_store.load(Path(agent_dir)))
        env.update(self._agent_env)
        return env

    # --- the agent turn -----------------------------------------------------

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
        memory_dir: Optional[Path] = None,
    ) -> tuple[ActionBlock, LLMUsage]:
        """Run one agent turn and translate the result into an ``ActionBlock``.

        Publishes ``LLM_COMPLETE`` on success / ``LLM_ERROR`` on failure. Cancellable
        at every ``await``; the whole loop is bounded by ``_invoke_timeout``.
        """
        start = time.time()
        working_dir.mkdir(parents=True, exist_ok=True)
        is_coordinator = action_schema == COORDINATOR_ACTION_SCHEMA
        try:
            client = self._client()
            env = self._env()
            emitters = emitters_for(action_schema)
            web_budget = (
                self._coordinator_web_max_uses
                if is_coordinator
                else self._web_max_uses
            )
            tools = build_harness_tools(
                working_dir,
                env,
                list(additional_dirs or []),
                list(skill_source_dirs or []),
                web_budget,
                self._web_fetch_max_uses,
                self._enable_web,
                write_roots=[memory_dir] if memory_dir else None,
            )
            tools += build_mcp_tools(mcp_config_dir, env)
            tools_by_name = {t.name: t for t in tools}
            schemas = [t.schema for t in tools] + build_emitter_schemas(emitters)

            messages = [
                {
                    "role": "user",
                    "content": self._augment_prompt(
                        prompt, skill_source_dirs, is_coordinator
                    ),
                }
            ]

            actions, usage = await asyncio.wait_for(
                self._run_loop(
                    client, messages, schemas, tools_by_name, emitters, log_dir
                ),
                timeout=self._invoke_timeout,
            )
            usage.duration_ms = int((time.time() - start) * 1000)
            usage.model = self._model_name

            raw = json.dumps({"actions": actions})
            await notification_center.publish(
                LLM_COMPLETE, sandbox_dir=working_dir, usage=usage
            )
            return (
                ActionBlock(raw=raw, actions=actions, source_version=trigger_version),
                usage,
            )
        except asyncio.CancelledError:
            logger.info("[%s] CANCELLED after %.1fs", self._log_tag, time.time() - start)
            raise
        except asyncio.TimeoutError:
            error = LLMTimeoutError(
                timeout_seconds=self._invoke_timeout,
                working_dir=str(working_dir),
                provider=self._provider_name,
            )
            await notification_center.publish(
                LLM_ERROR, sandbox_dir=working_dir, error=error
            )
            raise error
        except LLMInvocationError as error:
            # Covers LLMRateLimitError too (subclass) — preserves the rate-limit type.
            await notification_center.publish(
                LLM_ERROR, sandbox_dir=working_dir, error=error
            )
            raise
        except Exception as exc:  # noqa: BLE001 — normalize any other failure
            error = LLMInvocationError(
                f"{self._provider_name} failed: {exc}", working_dir=str(working_dir)
            )
            logger.error("[%s] error: %s", self._log_tag, exc, exc_info=True)
            await notification_center.publish(
                LLM_ERROR, sandbox_dir=working_dir, error=error
            )
            raise error

    async def _run_loop(
        self,
        client: OpenAIChatClient,
        messages: list[dict],
        schemas: list[dict],
        tools_by_name: dict[str, HarnessTool],
        emitters: tuple,
        log_dir: Path,
    ) -> tuple[list[dict], LLMUsage]:
        """The multi-turn tool-execution loop.

        Stop reasons: ``finalize`` called (STOP_FINALIZE); a turn with no tool calls
        (STOP_NO_TOOL_CALL); or ``max_turns`` reached. Emitters are captured in order;
        real tools are dispatched and their results fed back. A finalize-less emit
        tail is still assembled, so real actions are never lost. Caps
        (``max_requests`` / ``max_total_tokens``) raise a recoverable
        :class:`LLMInvocationError`.
        """
        emitter_names = {s.tool_name for s in emitters}
        emit_calls: list[ToolCall] = []
        usage = LLMUsage()

        for _turn in range(self._max_turns):
            turn = await client.chat(
                messages,
                tools=schemas,
                max_tokens=max(self._min_output_tokens, self._max_tokens),
                extra_body={"usage": {"include": True}},
            )
            usage += self._map_turn_usage(turn)
            if usage.requests > self._max_requests:
                raise LLMInvocationError(
                    f"{self._provider_name}: exceeded max_requests={self._max_requests}"
                )
            if (
                self._max_total_tokens is not None
                and usage.input_tokens + usage.output_tokens > self._max_total_tokens
            ):
                raise LLMInvocationError(
                    f"{self._provider_name}: exceeded "
                    f"max_total_tokens={self._max_total_tokens}"
                )

            messages.append(self._assistant_msg(turn))

            if not turn.tool_calls:
                break  # STOP_NO_TOOL_CALL — prose reply, no tool

            finalize_seen = False
            for call in turn.tool_calls:
                if call.name == FINALIZE_TOOL_NAME:
                    finalize_seen = True
                    continue
                if call.name in emitter_names:
                    emit_calls.append(call)
                    messages.append(self._tool_msg(call, "ok"))
                elif call.name in tools_by_name:
                    out = await self._dispatch(tools_by_name[call.name], call)
                    messages.append(self._tool_msg(call, out))
                    usage.tool_calls[call.name] = usage.tool_calls.get(call.name, 0) + 1
                else:
                    messages.append(
                        self._tool_msg(
                            call,
                            f"ERROR: unknown tool {call.name}. Use only the "
                            "offered tools.",
                        )
                    )
            if finalize_seen:
                break  # STOP_FINALIZE

        self._write_trace(messages, log_dir)
        actions = assemble(emit_calls, emitters)
        return actions, usage

    # --- one-shot text (DM auto-title) --------------------------------------

    async def generate_text(self, prompt: str, *, max_tokens: int = 32) -> str:
        """Real one-shot completion (no tools); any failure → deterministic default."""
        try:
            client = self._client()
            turn = await asyncio.wait_for(
                client.chat(
                    [{"role": "user", "content": prompt}],
                    tools=None,
                    max_tokens=max(16, max_tokens),
                ),
                timeout=self._text_gen_timeout,
            )
            text = (turn.content or "").strip()
            if text:
                return text
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — fall back to the deterministic default
            logger.warning(
                "[%s] generate_text failed; deterministic fallback",
                self._log_tag,
                exc_info=True,
            )
        return await super().generate_text(prompt, max_tokens=max_tokens)

    # --- helpers ------------------------------------------------------------

    def _augment_prompt(
        self,
        prompt: str,
        skill_source_dirs: Optional[list[Path]],
        is_coordinator: bool,
    ) -> str:
        """Append the skill INDEX (reused from api_provider) + the tool contract."""
        from .api_provider import _skill_index

        parts = [prompt]
        index = _skill_index(list(skill_source_dirs or []))
        if index:
            parts.append(
                "\n\n== AVAILABLE SKILLS ==\n"
                "Load a skill's full instructions on demand with the `skill` tool "
                "(skill(name)).\n" + index
            )
        emit_list = "emit_reply, emit_update_file"
        if is_coordinator:
            emit_list += ", emit_create_room, emit_project_completed"
        parts.append(
            "\n\n== HOW TO RESPOND ==\n"
            "Use the provided tools to do the work (read/write files, run bash, load "
            "skills, search the web). Then produce EACH action block by calling its "
            f"emit_* tool ({emit_list}) — one call per block, its parameters mapping "
            "1:1 onto that block. Emit ALL of your action blocks, then call `finalize` "
            "to end your turn. Restate concrete tool-result values verbatim in your "
            "reply content."
        )
        return "".join(parts)

    async def _dispatch(self, tool: HarnessTool, call: ToolCall) -> str:
        """Call a real tool with only its declared args; errors degrade to text."""
        kwargs = {k: v for k, v in (call.arguments or {}).items() if k in tool.arg_names}
        try:
            if tool.is_async:
                result = await tool.fn(**kwargs)
            else:
                result = await asyncio.to_thread(tool.fn, **kwargs)
            return str(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a tool error must not crash the turn
            logger.warning("[%s] tool %s failed: %s", self._log_tag, tool.name, exc)
            return f"ERROR: {exc}"

    def _assistant_msg(self, turn: AssistantTurn) -> dict:
        """Echo the assistant turn back — content coerced to '' (never null)."""
        msg: dict = {"role": "assistant", "content": turn.content or ""}
        if turn.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in turn.tool_calls
            ]
        return msg

    def _tool_msg(self, call: ToolCall, content: object) -> dict:
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "content": _coerce_content(content) or "",
        }

    def _map_turn_usage(self, turn: AssistantTurn) -> LLMUsage:
        """OpenRouter usage → LLMUsage (+ cost: reported when present, else priced)."""
        u = turn.usage or {}
        inp = int(u.get("prompt_tokens") or 0)
        out = int(u.get("completion_tokens") or 0)
        cost = u.get("cost")
        if cost is None:
            cost = price_openrouter_usd(self._model_name, inp, out)
        return LLMUsage(
            input_tokens=inp,
            output_tokens=out,
            cost_usd=float(cost or 0.0),
            model=self._model_name,
            requests=1,
        )

    def _write_trace(self, messages: list[dict], log_dir: Path) -> None:
        """Best-effort message-transcript dump, gated on CLAWMEETS_NATIVE_TRACE."""
        if not os.environ.get("CLAWMEETS_NATIVE_TRACE"):
            return
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "native-trace.json").write_text(
                json.dumps(messages, indent=2, default=str), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001 — tracing is never load-bearing
            logger.debug("[%s] trace write failed", self._log_tag, exc_info=True)
