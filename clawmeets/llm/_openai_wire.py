# SPDX-License-Identifier: MIT
"""
clawmeets/llm/_openai_wire.py
Native OpenAI-compatible chat transport for the ``openrouter-native`` provider.

Pydantic-AI used to normalize the OpenAI wire for us; this module owns it instead.
One request in, one parsed :class:`AssistantTurn` out — it deliberately does NOT
own the agent loop (that is :class:`~clawmeets.llm.native_harness.NativeHarnessProvider`).

Two wire facts, both reproduced live against ``qwen/qwen-2.5-72b-instruct`` on
OpenRouter (2026-07-16), are owned here at a single choke point each:

1. ``tool_calls[].function.arguments`` arrives as a JSON **string** (not an object)
   → :func:`_coerce_args` (``json.loads`` + normalize to ``dict``; junk/scalar/empty
   → ``{}`` rather than crashing; a dict passes through, so it is idempotent and
   also covers models that already send an object).
2. ``choices[0].message.content`` arrives **null** on a tool-call turn →
   :func:`_coerce_content` (``null``/list-of-parts → ``str``). Used BOTH when reading
   the reply AND when echoing the assistant message back, so no request ever carries
   a nil content part (some upstreams 400 on it).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from .base import LLMInvocationError, LLMRateLimitError

logger = logging.getLogger(__name__)


def _coerce_args(args: object) -> dict:
    """Normalize a tool call's ``arguments`` to a ``dict`` (wire-fact 1).

    OpenRouter/qwen send a JSON *string*; some routed models send an object.
    ``json.loads`` a string; a non-JSON / scalar / empty string → ``{}`` (never
    raises). A ``dict`` passes through unchanged, so this is idempotent and safe
    to apply regardless of which shape the upstream chose.
    """
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        s = args.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _coerce_content(content: object) -> str:
    """Coerce a message ``content`` to a plain string (wire-fact 2).

    ``null`` → ``""``; a list of content parts → concat of their ``text`` fields;
    anything else → ``str()``. Applied both when reading the model's reply and when
    echoing the assistant message back into ``messages`` so no request carries a
    null/nil content part.
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


@dataclass(frozen=True)
class ToolCall:
    """One normalized function call from the model (args already coerced to dict)."""

    id: str
    name: str
    arguments: dict
    raw: dict


@dataclass(frozen=True)
class AssistantTurn:
    """Parsed ``choices[0].message`` + usage for one model round-trip."""

    content: str
    tool_calls: list[ToolCall]
    finish_reason: str
    usage: dict
    model: str
    raw_message: dict


class OpenAIChatClient:
    """Thin async client over an OpenAI-compatible ``/chat/completions`` endpoint.

    Owns the endpoint, Bearer auth, optional app-attribution headers, the
    OpenRouter multi-model fallback body (``{"models": [...]}`` when more than one
    model is given), and bounded transient-retry (network / 5xx). It parses one
    response into an :class:`AssistantTurn`, applying :func:`_coerce_content` and
    :func:`_coerce_args`. It does NOT own the agent loop.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        models: list[str],
        timeout: float = 600.0,
        retries: int = 2,
        http_client: Optional[httpx.AsyncClient] = None,
        referer: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._models = list(models) or [""]
        self._retries = max(0, retries)
        self._referer = referer
        self._title = title
        # An injected client (tests / a shared pool) is NOT owned — we never close it.
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._referer:
            headers["HTTP-Referer"] = self._referer
        if self._title:
            headers["X-Title"] = self._title
        return headers

    async def chat(
        self,
        messages: list[dict],
        *,
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
        max_tokens: int = 32_000,
        temperature: float = 0.0,
        extra_body: Optional[dict] = None,
    ) -> AssistantTurn:
        """POST one request and return the parsed :class:`AssistantTurn`.

        A 429 raises :class:`LLMRateLimitError` (honoring ``Retry-After``); other
        4xx raise :class:`LLMInvocationError` immediately (no retry); network
        errors and 5xx retry with exponential backoff before raising. Cancellation
        propagates (the awaits are real await points).
        """
        body: dict = {
            "model": self._models[0],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        if len(self._models) > 1:
            body["models"] = self._models  # OpenRouter native fallback chain
        if extra_body:
            body.update(extra_body)

        last_exc: Optional[Exception] = None
        for attempt in range(self._retries + 1):
            try:
                resp = await self._client.post(
                    self._url, headers=self._headers(), json=body
                )
            except asyncio.CancelledError:
                raise
            except httpx.HTTPError as exc:  # network / timeout — transient
                last_exc = exc
                if attempt < self._retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
                raise LLMInvocationError(f"OpenRouter request failed: {exc}") from exc

            if resp.status_code == 429:
                raise LLMRateLimitError(
                    f"OpenRouter rate limit: {resp.text[:500]}",
                    resets_at=_parse_retry_after(resp),
                )
            if resp.status_code >= 500:
                last_exc = LLMInvocationError(
                    f"OpenRouter {resp.status_code}: {resp.text[:500]}"
                )
                if attempt < self._retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
                raise last_exc
            if resp.status_code >= 400:
                raise LLMInvocationError(
                    f"OpenRouter {resp.status_code}: {resp.text[:1000]}"
                )
            return self._parse(resp.json())

        # Unreachable (loop either returns or raises), but keeps the type checker happy.
        raise last_exc or LLMInvocationError("OpenRouter request failed")

    def _parse(self, data: dict) -> AssistantTurn:
        choices = data.get("choices") or [{}]
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        raw_calls = message.get("tool_calls") or []
        tool_calls: list[ToolCall] = []
        for rc in raw_calls:
            fn = (rc or {}).get("function") or {}
            tool_calls.append(
                ToolCall(
                    id=str(rc.get("id") or ""),
                    name=str(fn.get("name") or ""),
                    arguments=_coerce_args(fn.get("arguments")),
                    raw=rc,
                )
            )
        return AssistantTurn(
            content=_coerce_content(message.get("content")),
            tool_calls=tool_calls,
            finish_reason=str(choice.get("finish_reason") or ""),
            usage=data.get("usage") or {},
            model=str(data.get("model") or self._models[0]),
            raw_message=message,
        )

    async def aclose(self) -> None:
        """Close the owned httpx client (no-op for an injected one)."""
        if self._owns_client:
            await self._client.aclose()


def _parse_retry_after(resp: httpx.Response) -> Optional[float]:
    """Best-effort ``Retry-After`` (delta seconds) → absolute epoch, else ``None``."""
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return time.time() + float(raw)
    except (TypeError, ValueError):
        return None
