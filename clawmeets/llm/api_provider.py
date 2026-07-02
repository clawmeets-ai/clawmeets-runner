# SPDX-License-Identifier: MIT
"""
clawmeets/llm/api_provider.py
In-process, bring-your-own-key (BYO-key) LLM provider built on Pydantic AI.

Unlike the subprocess providers (claude_cli / codex_cli / gemini_cli) which shell
a Code CLI, this provider drives the model HTTP APIs **directly, in-process**, and
reimplements the agentic harness — local file tools, a guarded bash tool, MCP, and
skill exposure — inside the runner. That keeps the runner portable: it runs
identically whether local or on a hosted env, authenticated only by an API key, with
no `claude`/`codex`/`gemini` binary required.

File I/O still happens on the runner's own disk (the bash tool runs in-process with
the agent's `agent_env`, so skills that shell `clawmeets <cmd>` with
`$CLAWMEETS_AGENT_DIR` keep working). One class drives Anthropic / OpenAI / Gemini —
selected by ``provider`` — so there is a single skill, MCP, and structured-output
path across all three.

``pydantic-ai`` and ``genai-prices`` are core dependencies (no optional extra), so
this provider is always importable; the imports stay lazy only to keep module
import cheap.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time

import httpx
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from ..api.actions import ActionBlock, COORDINATOR_ACTION_SCHEMA
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

# Bound any single bash command so a runaway can't outlive the invocation.
_BASH_TIMEOUT = 600
# Cap how much tool output we feed back into the model context per call.
_MAX_TOOL_OUTPUT = 60_000
# Web pages are fetched many times per research turn and are the bulk of its
# tokens; clip them far tighter than a code-file read (60k chars ≈ 15k tokens
# each would blow the cumulative per-turn token cap after a handful of fetches).
_MAX_WEB_FETCH_OUTPUT = 12_000

# provider key -> (display name, default model)
_PROVIDERS: dict[str, tuple[str, str]] = {
    "anthropic": ("Anthropic API", "claude-opus-4-8"),
    "openai": ("OpenAI API", "gpt-5"),
    # Newer-gen Gemini (gemini-3-pro-preview was deprecated → 404). 3.5-flash is
    # current and stronger than 2.5-pro at the agentic harness; override per-agent
    # via `llm_model` (e.g. gemini-3.1-pro-preview for max quality).
    "google": ("Gemini API", "gemini-3.5-flash"),
    # OpenAI-wire-compatible gateway (400+ models). ``llm_model`` should be an
    # OpenRouter slug, e.g. "anthropic/claude-opus-4-8", "openai/gpt-5",
    # "google/gemini-2.5-pro", or a "…:free" model. The default is a cheap,
    # tool-capable model; users typically override it via ``llm_model``.
    "openrouter": ("OpenRouter API", "openai/gpt-4o-mini"),
}


# ---------------------------------------------------------------------------
# Structured output — self-contained, Literal-discriminated for cross-provider
# robustness (OpenAI strict mode / Gemini both prefer a clean discriminator).
# Dumped back to list[dict] so the ActionBlock contract is unchanged.
# ---------------------------------------------------------------------------


class _Reply(BaseModel):
    type: Literal["reply"]
    room: str
    content: str


class _UpdateFile(BaseModel):
    type: Literal["update_file"]
    room: str
    file_path: str


class _CreateRoom(BaseModel):
    type: Literal["create_room"]
    name: str
    invite: list[str]
    init_message: str


class _ProjectCompleted(BaseModel):
    type: Literal["project_completed"]


_WorkerAction = Annotated[Union[_Reply, _UpdateFile], Field(discriminator="type")]
_CoordinatorAction = Annotated[
    Union[_Reply, _UpdateFile, _CreateRoom, _ProjectCompleted],
    Field(discriminator="type"),
]


def _balanced_array(s: str) -> Optional[str]:
    """Return the first top-level ``[...]`` slice of ``s`` (string-literal aware),
    or None. Salvages a real array out of a stringified ``actions`` that carries
    trailing junk — e.g. opus emits ``"[ {...} ]\\n}"`` (a stray ``}`` after the
    close) which a plain ``json.loads`` rejects as 'Extra data'. The literal
    tracking keeps a ``]`` inside an ``init_message`` from closing early."""
    start = s.find("[")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _loads_actions_list(s: str) -> Optional[list]:
    """Parse a stringified ``actions`` value into a list, or None if it can't be
    salvaged (a non-JSON string then falls through to normal validation, which
    raises rather than silently swallowing). Tolerates a double-wrapped
    ``{"actions": [...]}`` string and trailing junk after the array."""
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
        if isinstance(v, dict) and isinstance(v.get("actions"), list):
            return v["actions"]
    except (json.JSONDecodeError, TypeError):
        pass
    arr = _balanced_array(s)
    if arr is not None:
        try:
            v = json.loads(arr)
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            pass
    return None


class _ActionsOutput(BaseModel):
    """Base for the structured-output models — tolerates a stringified ``actions``.

    Claude (and occasionally the other families) sometimes serializes the nested
    ``actions`` list as a JSON **string** in the StructuredOutput tool call rather
    than a real array. Pydantic AI would reject that (`list_type`), retry once, get
    the same mistake, and raise `UnexpectedModelBehavior` — crashing the whole turn
    (see the coordinator crash in the in-process harness). We parse the string back
    into a list here before the discriminated-union validators run. This is the
    in-process analogue of the CLI path's "Claude action merging" tolerance.
    """

    @model_validator(mode="before")
    @classmethod
    def _coerce_actions(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("actions"), str):
            parsed = _loads_actions_list(data["actions"])
            if parsed is not None:
                data = {**data, "actions": parsed}
            # else: leave the string in place — normal validation surfaces it
        return data


class _WorkerOutput(_ActionsOutput):
    """Actions a worker agent may emit."""

    actions: list[_WorkerAction] = Field(default_factory=list)


class _CoordinatorOutput(_ActionsOutput):
    """Actions a coordinator agent may emit."""

    actions: list[_CoordinatorAction] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Local file + bash toolset (the harness the CLI gave us for free)
# ---------------------------------------------------------------------------


def _clip(text: str, limit: int = _MAX_TOOL_OUTPUT) -> str:
    if len(text) > limit:
        return text[:limit] + f"\n... [truncated, {len(text)} chars total]"
    return text


def _tool_call_names(messages: Any) -> list[str]:
    """Ordered tool-call names (function + native/builtin) across a Pydantic AI
    run's messages. Best-effort — returns ``[]`` for a missing/odd message list.
    Order is preserved so callers can inspect the tail of a runaway loop."""
    names: list[str] = []
    for msg in messages or ():
        for part in getattr(msg, "parts", ()):  # ModelResponse.parts
            if "ToolCall" not in type(part).__name__:
                continue
            name = getattr(part, "tool_name", None)
            if name:
                names.append(name)
    return names


def _count_tool_calls(messages: Any) -> dict[str, int]:
    """Tool-call histogram (name -> count) from a Pydantic AI message list."""
    return dict(Counter(_tool_call_names(messages)))


def _build_file_tools(
    write_root: Path,
    env: dict[str, str],
    read_roots: Optional[list[Path]] = None,
) -> list:
    """Build per-invocation file + bash tools for this turn.

    READS (``read_file`` / ``list_dir`` / ``grep``) are unrestricted — they can
    reach the synced project dir, knowledge bases, and the dwh, matching the
    CLI's bypass-permissions read model (and the ``bash`` tool below can already
    read anything, so guarding reads would be theater). WRITES/EDITS are
    confined to ``write_root`` (the sandbox), so the agent can only mutate its
    own working area. ``bash`` is unrestricted (cwd = sandbox, full ``env``) —
    it mirrors the CLI shell and is what lets ``clawmeets``-shelling skills run
    in-process. Relative paths resolve against the sandbox cwd.

    ``read_roots`` are the additional first-class read roots (the synced project
    dir + knowledge dirs the caller passes as ``additional_dirs``) — mirroring
    the CLI's ``--add-dir``. ``glob`` spans them, and ``read_file``/``list_dir``/
    ``grep`` resolve a RELATIVE path against the sandbox first then those read
    roots (via ``_abs_read``) — so an agent that reads a synced file by its
    relative ``chatrooms/...`` path resolves to the project dir instead of
    failing. Absolute paths reach anywhere. WRITES stay sandbox-only (``_abs``).
    The sandbox (``write_root``) is always included.
    """
    write_root = write_root.resolve()
    extra_read_roots = [
        r.resolve() for r in (read_roots or []) if r.resolve() != write_root
    ]
    glob_roots = [write_root] + extra_read_roots

    def _abs(path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (write_root / p)

    def _abs_read(path: str) -> Path:
        """Resolve a READ path: absolute as-is; relative against the sandbox
        first, then the read roots (project dir + knowledge dirs). Returns the
        first existing candidate, else the sandbox candidate (for a sensible
        "not found" error). Reads are unrestricted, so spanning read roots just
        makes relative paths forgiving — what the sandbox-only `_abs` was not."""
        p = Path(path)
        if p.is_absolute():
            return p
        sandbox_candidate = write_root / p
        for root in (write_root, *extra_read_roots):
            cand = root / p
            if cand.exists():
                return cand
        return sandbox_candidate

    def _guard_write(path: str) -> Path:
        p = _abs(path).resolve()
        if p == write_root or write_root in p.parents:
            return p
        raise ValueError(f"path '{path}' is outside the writable sandbox")

    def read_file(path: str) -> str:
        """Read a UTF-8 text file (project / knowledge / dwh / sandbox)."""
        p = _abs_read(path)
        if not p.is_file():
            return f"ERROR: not a file: {path}"
        return _clip(p.read_text(encoding="utf-8", errors="replace"))

    def write_file(path: str, content: str) -> str:
        """Create or overwrite a UTF-8 text file in the sandbox."""
        try:
            p = _guard_write(path)
        except ValueError as e:
            # Recoverable: let the model retry with a sandbox-relative path
            # instead of crashing the turn.
            return f"ERROR: {e}. Write under the sandbox working dir (a relative path)."
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        # Trace the ABSOLUTE path actually written so we can reconcile against the
        # action executor's sandbox_dir when update_file later reports "not found".
        logger.debug("[api-write] wrote %d chars to %s", len(content), p)
        return f"wrote {len(content)} chars to {path}"

    def edit_file(path: str, old: str, new: str) -> str:
        """Replace the first exact occurrence of `old` with `new` in a sandbox file."""
        try:
            p = _guard_write(path)
        except ValueError as e:
            return f"ERROR: {e}. Edit a file under the sandbox working dir."
        if not p.is_file():
            return f"ERROR: not a file: {path}"
        text = p.read_text(encoding="utf-8")
        if old not in text:
            return "ERROR: `old` text not found; no change made"
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"edited {path}"

    def list_dir(path: str = ".") -> str:
        """List entries of a directory (relative paths resolve against the
        sandbox first, then the project/knowledge read roots)."""
        p = _abs_read(path)
        if not p.is_dir():
            return f"ERROR: not a directory: {path}"
        names = sorted(
            e.name + ("/" if e.is_dir() else "") for e in p.iterdir()
        )
        return _clip("\n".join(names) or "(empty)")

    def glob(pattern: str) -> str:
        """Glob for files under the sandbox + the project/knowledge read roots
        (e.g. '**/*.py'). Spans every read root so the agent can locate synced
        project files and reference material, not just its sandbox."""
        matches = sorted(
            str(m) for root in glob_roots for m in root.glob(pattern)
        )
        return _clip("\n".join(matches) or "(no matches)")

    def grep(pattern: str, path: str = ".") -> str:
        """Regex-search files under a directory; returns matching `file:line: text`.
        Relative paths resolve against the sandbox first, then the read roots."""
        root = _abs_read(path)
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"ERROR: bad regex: {e}"
        out: list[str] = []
        files = [root] if root.is_file() else root.rglob("*")
        for f in files:
            if not f.is_file():
                continue
            try:
                for i, line in enumerate(
                    f.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if rx.search(line):
                        out.append(f"{f}:{i}: {line.strip()}")
                        if len(out) >= 500:
                            return _clip("\n".join(out))
            except OSError:
                continue
        return _clip("\n".join(out) or "(no matches)")

    async def bash(command: str) -> str:
        """Run a shell command in the sandbox (cwd = sandbox, agent env)."""
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(write_root),
            env=env,
        )
        try:
            out_b, _ = await asyncio.wait_for(
                proc.communicate(), timeout=_BASH_TIMEOUT
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            return f"ERROR: command timed out after {_BASH_TIMEOUT}s"
        rc = proc.returncode
        out = out_b.decode("utf-8", errors="replace")
        return _clip(f"(exit {rc})\n{out}")

    return [read_file, write_file, edit_file, list_dir, glob, grep, bash]


# ---------------------------------------------------------------------------
# Local OpenAI-compatible (ollama) message-content compatibility
# ---------------------------------------------------------------------------


def _stringify_message_content(content: Any) -> str:
    """Coerce a chat message's ``content`` to a plain string.

    Strict local OpenAI-compatible servers (ollama ``/v1``) reject the content
    shapes the OpenAI SDK legitimately emits across a tool-calling turn — a
    thinking model like qwen3 can produce an assistant message whose content is
    ``null``, missing, or an array containing a nil/empty part, which ollama
    rejects with ``invalid message content type: <nil>``. Flattening every
    message's content to a string (join text parts, drop non-text/empty parts,
    ``None`` → ``""``) makes any such server accept the flow. Tool-call metadata
    (``tool_calls`` / ``tool_call_id``) is left untouched by the caller.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                text = p.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(content)


class _LocalOpenAICompatTransport(httpx.AsyncBaseTransport):
    """Rewrites outgoing ``/chat/completions`` message content to plain strings.

    Wraps the httpx transport used ONLY by the local ``base_url`` client (hosted
    OpenAI is never touched). See :func:`_stringify_message_content` for why —
    it's the fix for ollama's ``invalid message content type: <nil>`` 400 on
    tool-calling turns with a reasoning model.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions") and request.content:
            try:
                body = json.loads(request.content)
                messages = body.get("messages")
                mutated = False
                if isinstance(messages, list):
                    for m in messages:
                        if not isinstance(m, dict):
                            continue
                        new_c = _stringify_message_content(m.get("content"))
                        if m.get("content") != new_c:
                            m["content"] = new_c
                            mutated = True
                if mutated:
                    raw = json.dumps(body).encode()
                    headers = {
                        k: v for k, v in request.headers.items()
                        if k.lower() != "content-length"
                    }
                    request = httpx.Request(
                        request.method, request.url, headers=headers, content=raw
                    )
            except (ValueError, TypeError):
                pass  # not JSON we understand — pass through untouched
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class ApiLLMProvider(LLMProvider):
    """In-process, BYO-key provider for Anthropic / OpenAI / Gemini via Pydantic AI."""

    _log_tag = "api-invoke"
    _install_hint = "pydantic-ai not importable — reinstall clawmeets"

    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        agent_env: dict[str, str],
        model: Optional[str] = None,
        model_obj: Optional[Any] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 32_000,
        enable_web: bool = True,
        web_max_uses: int = 5,
        coordinator_web_max_uses: int = 2,
        web_fetch_max_uses: int = 12,
        max_requests: int = 50,
        max_total_tokens: Optional[int] = None,
    ) -> None:
        provider = (provider or "anthropic").lower()
        if provider not in _PROVIDERS:
            raise LLMNotFoundError(
                provider, install_hint=f"unknown API provider '{provider}'"
            )
        display, default_model = _PROVIDERS[provider]
        self._provider_name = display
        self._provider = provider
        self._api_key = api_key
        self._agent_env = agent_env
        self._model_name = model or default_model
        self._model_obj = model_obj  # test injection / pre-built model
        # Custom OpenAI-compatible endpoint (ollama /v1, vLLM, LM Studio, …).
        # Only consulted by the openai family: switches that branch from the
        # Responses API to Chat Completions against this base_url so a LOCAL
        # model gets schema-enforced structured output. Local endpoints need no
        # key, so an empty api_key is tolerated when this is set.
        self._base_url = base_url
        # Lazily-built httpx client (content-sanitizing) for the local base_url
        # path; cached so we don't leak a client per invocation.
        self._local_http_client: Optional[httpx.AsyncClient] = None
        self._max_tokens = max_tokens
        self._enable_web = enable_web
        # Per-turn caps (tunable via eval "profiles"): how many native/local web
        # tool uses are allowed, and the Pydantic AI request_limit backstop.
        self._web_max_uses = web_max_uses
        # A coordinator plans + delegates; it should only get a small web budget
        # for a quick scoping check, not run a worker's full research loop.
        self._coordinator_web_max_uses = coordinator_web_max_uses
        # Per-turn web_fetch budget (local fetch path). Fetched pages are the bulk
        # of a research turn's tokens, so an unbounded fetch loop trips the
        # cumulative max_total_tokens cap and aborts the turn. Budget it like
        # web_max_uses (graceful "stop fetching" instead of a fatal abort).
        self._web_fetch_max_uses = web_fetch_max_uses
        self._max_requests = max_requests
        # Cumulative per-turn token ceiling (None ⇒ unbounded). A runaway loop
        # (e.g. an agent re-reading its own deliverable each step) grows the
        # re-sent transcript until it trips this — aborting as a recoverable
        # UsageLimitExceeded BEFORE a single request blows the model's hard
        # context window (an unrecoverable 400 on OpenAI). Caps step COUNT via
        # max_requests; this caps the token blast radius the step count can't.
        self._max_total_tokens = max_total_tokens

    # --- preflight ----------------------------------------------------------

    @classmethod
    def verify_cli(cls) -> None:
        try:
            import pydantic_ai  # noqa: F401
        except ImportError as e:
            raise LLMNotFoundError("pydantic-ai", install_hint=cls._install_hint) from e

    # --- model construction (BYO key) --------------------------------------

    def _build_model(self) -> Any:
        if self._model_obj is not None:
            return self._model_obj
        # A local OpenAI-compatible endpoint (base_url) needs no key; everything
        # else does.
        if not self._api_key and not self._base_url:
            raise LLMNotFoundError(
                self._provider_name,
                install_hint="no API key configured (set local_settings.llm_api_key)",
            )
        if self._provider == "anthropic":
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider

            return AnthropicModel(
                self._model_name, provider=AnthropicProvider(api_key=self._api_key)
            )
        if self._provider == "openai":
            from pydantic_ai.providers.openai import OpenAIProvider

            if self._base_url:
                # Custom OpenAI-compatible endpoint (ollama /v1, vLLM, …). These
                # speak Chat Completions, not the Responses API, so use the Chat
                # model (same class as the openrouter branch). Structured output
                # then rides tool calls, which local endpoints support — and the
                # key is a placeholder ollama ignores. No native web search here
                # (WebSearch/WebFetch fall back to local DDG/markdownify).
                # The client wraps a content-sanitizing transport so ollama's
                # strict /v1 accepts the tool-calling flow (see the transport).
                from pydantic_ai.models.openai import OpenAIChatModel

                if self._local_http_client is None:
                    self._local_http_client = httpx.AsyncClient(
                        transport=_LocalOpenAICompatTransport(
                            httpx.AsyncHTTPTransport()
                        ),
                        timeout=httpx.Timeout(600.0),
                    )
                return OpenAIChatModel(
                    self._model_name,
                    provider=OpenAIProvider(
                        base_url=self._base_url,
                        api_key=self._api_key or "ollama",
                        http_client=self._local_http_client,
                    ),
                )
            # Responses API (not Chat Completions) so the direct OpenAI provider
            # gets native server-side web search via the WebSearch capability.
            from pydantic_ai.models.openai import OpenAIResponsesModel

            return OpenAIResponsesModel(
                self._model_name, provider=OpenAIProvider(api_key=self._api_key)
            )
        if self._provider == "openrouter":
            # OpenRouter speaks OpenAI Chat Completions (not Responses); the
            # dedicated provider sets base_url + optional app headers. No native
            # web search — WebSearch/WebFetch fall back to local DDG/markdownify.
            # `llm_model` may be a comma-separated ordered fallback chain; the
            # FIRST slug is the primary, the rest ride OpenRouter's `models`
            # fallback array (see _model_settings).
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openrouter import OpenRouterProvider

            return OpenAIChatModel(
                self._openrouter_models()[0],
                provider=OpenRouterProvider(api_key=self._api_key),
            )
        # google
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleModel(
            self._model_name, provider=GoogleProvider(api_key=self._api_key)
        )

    def _openrouter_models(self) -> list[str]:
        """OpenRouter model chain from a (possibly comma-separated) ``llm_model``.
        First entry is the primary; the rest are OpenRouter fallbacks tried in
        order on error (incl. 429 rate limits). Always non-empty."""
        models = [m.strip() for m in (self._model_name or "").split(",") if m.strip()]
        return models or [self._model_name]

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self._agent_env)
        return env

    def _resolve_web_budget(self, is_coordinator: bool) -> int:
        """Per-turn web-search budget. Clamped for a COORDINATOR: its job is to
        plan + delegate, so it gets only `coordinator_web_max_uses` searches
        (enough to scope the task), not a worker's full research loop — this is
        what stops a coordinator from burning a long web-heavy turn."""
        if is_coordinator:
            return min(self._web_max_uses, self._coordinator_web_max_uses)
        return self._web_max_uses

    def _resolve_web_fetch_budget(self, is_coordinator: bool) -> int:
        """Per-turn web_fetch budget. Clamped for a coordinator (it shouldn't
        run a worker's research loop) the same way as the search budget."""
        if is_coordinator:
            return min(self._web_fetch_max_uses, self._coordinator_web_max_uses)
        return self._web_fetch_max_uses

    def _web_capabilities(self, is_coordinator: bool) -> list:
        """Provider-aware web search/fetch capabilities for one invocation.

        Anthropic uses a NATIVE built-in web search (it honors `max_uses`).
        OpenAI's Responses API exposes a native web search too, but IGNORES
        `max_uses` — so an unbounded research turn can run dozens of slow
        server-side searches (observed: 27 searches → a 17-min turn). To bound
        it deterministically, OpenAI drops the native search and instead gets a
        budget-wrapped LOCAL DuckDuckGo `web_search` tool (added in `invoke`);
        only its WebFetch capability survives here. Gemini rejects mixing
        function + built-in tools, and OpenRouter's routed models often don't
        support native web — so those use LOCAL DuckDuckGo + markdownify.
        """
        if not self._enable_web:
            return []
        from pydantic_ai.capabilities import WebFetch, WebSearch

        web_max = self._resolve_web_budget(is_coordinator)
        if self._provider == "anthropic":
            # Native web SEARCH honors `max_uses` on Anthropic. WebFetch is
            # native where supported.
            return [
                WebSearch(native=True, local="duckduckgo", max_uses=web_max),
                WebFetch(local=True),
            ]
        # Everything else (openai, gemini, openrouter) gets NO capability tools.
        # The stock local WebSearch/WebFetch are unbudgeted and un-clipped (so a
        # research turn blows the cumulative token cap) and the stock WebFetch
        # RAISES on a 403/404/timeout (crashing the turn). Instead `invoke` adds
        # error-tolerant, budgeted, tightly-clipped custom function tools
        # (_build_web_search_tool / _build_web_fetch_tool) — plain function tools,
        # so no Gemini function-vs-builtin / OpenRouter unsupported-native 400.
        return []

    def _model_settings(self) -> Any:
        """Per-provider model settings. Anthropic opts into prompt caching of the
        system prompt + tool definitions AND the growing message history
        (`cache_control` breakpoints) — the last is critical for the agentic
        loop: without it every step re-pays full input price on the entire
        accumulated transcript (tool calls + web-search results), which is the
        dominant cost in a long, web-heavy turn. The other providers auto-cache
        server-side, and OpenRouter relies on the routed provider's own caching,
        so a plain ``max_tokens`` suffices there."""
        if self._provider == "anthropic":
            from pydantic_ai.models.anthropic import AnthropicModelSettings

            return AnthropicModelSettings(
                max_tokens=self._max_tokens,
                anthropic_cache_instructions=True,
                anthropic_cache_tool_definitions=True,
                anthropic_cache_messages=True,
            )
        if self._provider == "openrouter":
            # A multi-model `llm_model` becomes OpenRouter's native fallback chain:
            # `extra_body={"models": [...]}` makes OpenRouter try each model in
            # order, per request, routing to the next on error (incl. 429 rate
            # limits). Single-model chains send no `models` array (clean default).
            models = self._openrouter_models()
            settings: dict = {"max_tokens": self._max_tokens}
            if len(models) > 1:
                settings["extra_body"] = {"models": models}
            return settings
        return {"max_tokens": self._max_tokens}

    # --- the in-process agentic loop ---------------------------------------

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
    ) -> tuple[ActionBlock, LLMUsage]:
        from pydantic_ai import Agent

        working_dir.mkdir(parents=True, exist_ok=True)
        is_coordinator = action_schema == COORDINATOR_ACTION_SCHEMA
        output_type = _CoordinatorOutput if is_coordinator else _WorkerOutput

        web_caps = self._web_capabilities(is_coordinator)

        env = self._build_env()
        # additional_dirs (synced project dir + knowledge dirs) become first-class
        # read roots so glob can span them — CLI `--add-dir` parity. Previously
        # accepted but unused, leaving glob sandbox-only.
        tools = _build_file_tools(working_dir, env, read_roots=additional_dirs)
        tools.append(_build_skill_tool(skill_source_dirs or []))
        # All non-Anthropic providers (openai/gemini/openrouter) get the local
        # budget-wrapped, tightly-clipped, error-tolerant web tools instead of a
        # native or stock-local capability — so a research turn can't blow the
        # cumulative token cap or crash on a bad fetch. Anthropic keeps its native
        # web search (it honors max_uses server-side).
        if self._provider != "anthropic" and self._enable_web:
            tools.append(_build_web_search_tool(self._resolve_web_budget(is_coordinator)))
            tools.append(_build_web_fetch_tool(self._resolve_web_fetch_budget(is_coordinator)))
        toolsets = self._build_mcp_toolsets(mcp_config_dir, env)

        prompt = self._augment_prompt(prompt, skill_source_dirs)

        agent = Agent(
            self._build_model(),
            output_type=output_type,
            tools=tools,
            toolsets=toolsets,
            capabilities=web_caps,
            model_settings=self._model_settings(),
            # Backstop: give a malformed StructuredOutput one extra correction pass
            # before it crashes the turn (default is 1). The _ActionsOutput coercion
            # above is the real fix; this just hardens genuinely-bad first attempts.
            retries={"output": 2},
        )

        from pydantic_ai import capture_run_messages
        from pydantic_ai.usage import RunUsage, UsageLimits

        # Passed into agent.run so it accumulates in place — survives a mid-loop
        # exception, letting us record the tokens (and $) a FAILED turn burned
        # (e.g. hitting request_limit, or a billing 400 after many web steps).
        run_usage = RunUsage()
        start = time.time()
        # capture_run_messages() exposes the transcript even when the run raises,
        # so error paths can attribute tool_calls + dump a trace (the loop is
        # otherwise a black box on failure).
        with capture_run_messages() as captured:
            try:
                result = await asyncio.wait_for(
                    agent.run(
                        prompt,
                        usage=run_usage,
                        usage_limits=UsageLimits(
                            request_limit=self._max_requests,
                            total_tokens_limit=self._max_total_tokens,
                        ),
                    ),
                    timeout=self._invoke_timeout,
                )
            except asyncio.CancelledError:
                logger.info(f"[{self._log_tag}] CANCELLED after {time.time()-start:.1f}s")
                raise
            except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001
                if isinstance(e, asyncio.TimeoutError):
                    error: Exception = LLMTimeoutError(
                        timeout_seconds=self._invoke_timeout,
                        working_dir=str(working_dir),
                        provider=self._provider_name,
                    )
                else:
                    error = self._normalize_error(e, working_dir)
                seq = _tool_call_names(captured)
                logger.warning(
                    f"[{self._log_tag}] FAILED after {getattr(run_usage, 'requests', 0)} "
                    f"req in {time.time()-start:.1f}s; tools={dict(Counter(seq))}; "
                    f"last={seq[-8:]}; err={type(error).__name__}: {error}"
                )
                self._write_trace(captured, log_dir)
                await notification_center.publish(
                    LLM_ERROR, sandbox_dir=working_dir, error=error,
                    usage=self._map_usage(run_usage, time.time() - start, messages=captured),
                )
                raise error from e

        self._write_trace(result.all_messages(), log_dir)
        usage = self._map_usage(
            run_usage, time.time() - start, messages=result.all_messages()
        )
        actions = [a.model_dump(mode="json") for a in result.output.actions]
        raw = result.output.model_dump_json()

        logger.info(
            f"[{self._log_tag}] FINISHED in {usage.duration_ms/1000:.1f}s, "
            f"{len(actions)} action(s), tokens in={usage.input_tokens} "
            f"out={usage.output_tokens}"
        )

        await notification_center.publish(
            LLM_COMPLETE, sandbox_dir=working_dir, usage=usage
        )
        return (
            ActionBlock(raw=raw, actions=actions, source_version=trigger_version),
            usage,
        )

    # --- helpers ------------------------------------------------------------

    def _augment_prompt(
        self, prompt: str, skill_source_dirs: Optional[list[Path]]
    ) -> str:
        """Surface installed skills as a name+description INDEX (progressive
        disclosure level 1). The model loads a skill's full body on demand via
        the ``skill`` tool (level 2), mirroring the CLI's native skill loader —
        descriptions are always visible, full instructions only when relevant.
        """
        index = _skill_index(skill_source_dirs or [])
        if not index:
            return prompt
        return (
            f"{prompt}\n\n=== AVAILABLE SKILLS ===\n"
            "Call the `skill` tool with a skill's name to load its full "
            "instructions when a task matches its description.\n" + index
        )

    def _build_mcp_toolsets(
        self, mcp_config_dir: Optional[Path], env: dict[str, str]
    ) -> list:
        """Build Pydantic AI MCP server toolsets from the rendered .mcp.json.

        Reuses the agent's pre-rendered ``mcp-hub/dist/.mcp.json`` (Claude wire
        format: ``{"mcpServers": {name: {command, args, env}}}``) so the API
        provider speaks to the same installed MCP servers as the CLI tier.
        """
        if mcp_config_dir is None:
            return []
        from ..utils.file_io import FileUtil

        spec = FileUtil.read(mcp_config_dir / ".mcp.json", "json") or {}
        servers = spec.get("mcpServers") if isinstance(spec, dict) else None
        if not isinstance(servers, dict):
            return []
        from fastmcp.client.transports import StdioTransport
        from pydantic_ai.mcp import MCPToolset

        toolsets: list = []
        for name, cfg in servers.items():
            if not isinstance(cfg, dict):
                continue
            url = cfg.get("url")
            if url:
                toolsets.append(MCPToolset(url, id=name))
                continue
            command = cfg.get("command")
            if not command:
                continue
            server_env = {**env, **(cfg.get("env") or {})}
            toolsets.append(
                MCPToolset(
                    StdioTransport(
                        command=command,
                        args=list(cfg.get("args") or []),
                        env=server_env,
                    ),
                    id=name,
                )
            )
        return toolsets

    def _map_usage(
        self, u: Any, elapsed_s: float, messages: Any = None
    ) -> LLMUsage:
        # ``u`` is the run's accumulating RunUsage (passed into agent.run via
        # ``usage=``), so it holds partial token counts even when the run raised
        # mid-loop and there is no ``result``. ``messages`` (the captured
        # transcript, available on success AND failure) attributes tool_calls —
        # so a failed turn's cost row shows the real histogram, not ``{}``.
        input_tokens = getattr(u, "input_tokens", 0) or 0
        output_tokens = getattr(u, "output_tokens", 0) or 0
        cache_read = getattr(u, "cache_read_tokens", 0) or 0
        cache_write = getattr(u, "cache_write_tokens", 0) or 0
        return LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_write,
            cost_usd=self._price(input_tokens, output_tokens, cache_read, cache_write),
            duration_ms=int(elapsed_s * 1000),
            model=self._model_name,
            requests=getattr(u, "requests", 0) or 0,
            tool_calls=_count_tool_calls(messages) if messages else {},
        )

    def _write_trace(self, messages: Any, log_dir: Path) -> None:
        """Dump the full run transcript to ``{log_dir}/api-trace.json`` for
        debugging the in-process loop. Env-gated (``CLAWMEETS_API_TRACE``) so
        production runners pay nothing; best-effort — tracing must never break a
        turn. Overwrites per turn (a stalling turn is terminal, so it persists)."""
        if not os.environ.get("CLAWMEETS_API_TRACE"):
            return
        try:
            from pydantic_ai.messages import ModelMessagesTypeAdapter

            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "api-trace.json").write_bytes(
                ModelMessagesTypeAdapter.dump_json(list(messages or []), indent=2)
            )
        except Exception:  # noqa: BLE001 — never let tracing break a turn
            logger.debug(f"[{self._log_tag}] trace dump failed", exc_info=True)

    def _price(
        self, input_tokens: int, output_tokens: int, cache_read: int, cache_write: int
    ) -> float:
        """Best-effort USD cost via genai-prices; 0.0 if the model isn't priced
        (e.g. an unmapped OpenRouter slug)."""
        from .pricing import PRICE_PROVIDER_ID, price_openrouter_usd, price_usd

        if self._provider == "openrouter":
            # genai-prices can't price a routed `<vendor>/<model>` slug under the
            # `openrouter` id; split it and price the underlying model instead.
            return price_openrouter_usd(
                self._model_name, input_tokens, output_tokens, cache_read, cache_write
            )
        return price_usd(
            self._model_name,
            PRICE_PROVIDER_ID.get(self._provider),
            input_tokens,
            output_tokens,
            cache_read,
            cache_write,
        )

    def _normalize_error(
        self, e: Exception, working_dir: Path
    ) -> LLMInvocationError:
        blob = f"{type(e).__name__}: {e}".lower()
        status = getattr(e, "status_code", None) or getattr(e, "status", None)
        if status == 429 or "rate limit" in blob or "rate_limit" in blob or (
            "quota" in blob or "resource_exhausted" in blob
        ):
            return LLMRateLimitError(
                f"{self._provider_name} rate limited: {e}",
                working_dir=str(working_dir),
            )
        return LLMInvocationError(
            f"{self._provider_name} invocation failed: {e}",
            working_dir=str(working_dir),
        )


def _skill_map(skill_source_dirs: list[Path]) -> dict[str, Path]:
    """Map skill name -> its ``SKILL.md`` path, honoring source precedence (later
    sources win on name collisions, matching ``materialize_skill_tree`` — personal
    overrides platform overrides system)."""
    chosen: dict[str, Path] = {}
    for src in skill_source_dirs:
        if not src.is_dir():
            continue
        for entry in sorted(src.iterdir()):
            skill_md = entry / "SKILL.md"
            if entry.is_dir() and skill_md.exists():
                chosen[entry.name] = skill_md
    return chosen


def _skill_index(skill_source_dirs: list[Path]) -> str:
    """One line per discoverable skill: '- <name>: <description>' (no paths) —
    progressive-disclosure level 1. The full body is pulled via the ``skill`` tool."""
    lines: list[str] = []
    for name, skill_md in _skill_map(skill_source_dirs).items():
        desc = _skill_description(skill_md)
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def _build_skill_tool(skill_source_dirs: list[Path]):
    """Build a ``skill(name)`` tool (progressive-disclosure level 2): returns a
    skill's full ``SKILL.md`` body plus a listing of the skill dir's sibling files
    (bundled resources/scripts the body may reference), resolved by name with
    source precedence — so the model loads a skill on demand instead of guessing
    an absolute path."""

    def skill(name: str) -> str:
        """Load an installed skill by name: its SKILL.md instructions plus the
        paths of the skill's other files (read those with read_file as directed)."""
        chosen = _skill_map(skill_source_dirs)
        skill_md = chosen.get(name)
        if skill_md is None:
            avail = ", ".join(sorted(chosen)) or "(none)"
            return f"ERROR: no skill named '{name}'. Available: {avail}"
        skill_dir = skill_md.parent
        body = _clip(skill_md.read_text(encoding="utf-8", errors="replace"))
        siblings = sorted(
            str((skill_dir / p).resolve())
            for p in (
                q.relative_to(skill_dir)
                for q in skill_dir.rglob("*")
                if q.is_file() and q.name != "SKILL.md"
            )
        )
        if not siblings:
            return body
        return body + "\n\nSkill files (read with read_file as needed):\n" + "\n".join(
            f"- {s}" for s in siblings
        )

    return skill


def _build_web_search_tool(max_uses: int, *, search_fn=None):
    """Build a per-invocation budget-wrapped local web-search tool.

    Each turn gets a fresh ``{"used": 0}`` counter: the first ``max_uses`` calls
    run a local DuckDuckGo search; once the budget is spent the tool returns a
    "synthesize from what you have" note WITHOUT touching the network. This is
    how we bound web on the OpenAI Responses path — its native web search ignores
    ``WebSearch(max_uses=…)``, so an unbounded research turn can run dozens of
    slow server-side searches. ``search_fn`` is injectable so tests run offline.
    """

    def _default_search(query: str) -> list[dict]:
        from ddgs import DDGS

        return DDGS().text(query, max_results=5)

    run_search = search_fn or _default_search
    state = {"used": 0}

    def web_search(query: str) -> str:
        """Search the web (DuckDuckGo) and return the top results as text.
        Search broadly first, then synthesize — re-running similar queries
        wastes the per-turn budget."""
        if state["used"] >= max_uses:
            return (
                f"Web-search budget ({max_uses}) reached this turn. Synthesize "
                "from the results you already have — do not search again."
            )
        state["used"] += 1
        try:
            results = run_search(query) or []
        except Exception as e:  # noqa: BLE001
            return f"web_search error: {e}"
        if not results:
            return f"No results for: {query}"
        lines = []
        for r in results:
            title = r.get("title") or ""
            href = r.get("href") or r.get("url") or ""
            body = r.get("body") or r.get("snippet") or ""
            lines.append(f"- {title}\n  {href}\n  {body}")
        return _clip("\n".join(lines))

    return web_search


def _build_web_fetch_tool(max_uses: int, *, fetch_fn=None):
    """Build a budgeted, error-tolerant local web-fetch tool.

    Mirrors ``_build_web_search_tool``: a fresh ``{"used": 0}`` counter caps the
    fetches per turn (graceful "synthesize" note past the budget, so a fetch loop
    can't trip the cumulative token cap), and ANY fetch failure
    (403/404/timeout/parse) is returned as a ``web_fetch error: …`` string rather
    than raised. pydantic-ai's stock local WebFetch RAISES on errors — exhausting
    its tool-retry and crashing the whole turn — and is unbounded; both are fatal
    on the OpenAI path where snippet-only local search makes the model fetch many
    real-world pages. Fetched pages are clipped to ``_MAX_WEB_FETCH_OUTPUT`` (far
    tighter than a code-file read). ``fetch_fn`` is injectable so tests run offline.
    """

    def _default_fetch(url: str) -> str:
        import httpx
        import markdownify

        resp = httpx.get(
            url,
            follow_redirects=True,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (clawmeets research bot)"},
        )
        resp.raise_for_status()
        return markdownify.markdownify(resp.text)

    run_fetch = fetch_fn or _default_fetch
    state = {"used": 0}

    def web_fetch(url: str) -> str:
        """Fetch a web page and return its main text (markdown). On any error
        (forbidden, not found, timeout) returns an error note — pick another
        source rather than retrying the same URL."""
        if state["used"] >= max_uses:
            return (
                f"Web-fetch budget ({max_uses}) reached this turn. Synthesize "
                "from the pages you already fetched — do not fetch more."
            )
        state["used"] += 1
        try:
            text = run_fetch(url) or f"web_fetch: empty response for {url}"
        except Exception as e:  # noqa: BLE001
            return f"web_fetch error for {url}: {e}"
        return _clip(text, limit=_MAX_WEB_FETCH_OUTPUT)

    return web_fetch


def _skill_description(skill_md: Path) -> str:
    """Pull the `description:` from SKILL.md YAML frontmatter (best-effort).

    Handles both inline values and block scalars (``description: >`` / ``|``
    with optional chomping indicator, the form nearly every bundled skill
    uses) — the old inline-only regex captured a literal ``>`` for those, so
    the -api tier's skill INDEX rendered every description blank and models
    never knew when to invoke a skill.
    """
    try:
        text = skill_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = re.search(r"^description:[ \t]*(.*)$", text, re.MULTILINE)
    if not m:
        return ""
    inline = m.group(1).strip()
    if inline and not re.fullmatch(r"[>|][+-]?", inline):
        return inline.strip("\"'")
    # Block scalar: gather the following more-indented lines until the next
    # top-level key or the closing frontmatter fence.
    parts: list[str] = []
    for line in text[m.end():].splitlines():
        if line.startswith((" ", "\t")) and line.strip():
            parts.append(line.strip())
        elif not line.strip():
            continue
        else:
            break  # next top-level key or ---
    return " ".join(parts)
