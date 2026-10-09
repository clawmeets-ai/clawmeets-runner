# SPDX-License-Identifier: MIT
"""
clawmeets/api/retry_transport.py

httpx transports that ride out a server restart: every request is retried
with jittered backoff while the server is unreachable, and every write
carries an ``Idempotency-Key`` so a retry of a write that DID land is
answered from the server's record instead of being applied twice.

Part of the API layer (Layer 0): imports only httpx and the stdlib.

## What is retried

Only failures that mean "the server never answered":

- connection refused / reset / dropped mid-response (``ConnectError``,
  ``ConnectTimeout``, ``RemoteProtocolError``, ``ReadError``);
- 502 / 503 / 504 whose body is NOT JSON. A restart shows up as nginx's
  HTML error page; the app's own 5xx are JSON (``{"detail": ...}``) and
  carry a meaning — "agent runner is not connected", "result unknown" —
  that a retry would only delay.

Reads are retried unconditionally. Writes (POST / PUT / PATCH / DELETE) are
retried only when they carry an ``Idempotency-Key``; the transport adds one
(uuid4) when the caller didn't, and reuses it on every attempt. ``ReadTimeout``
is never retried: the server got the request and is slow, and a second copy
would only add load.

After the budget (60 s by default) the last error is raised, or the last
response returned, exactly as the caller would have seen it without retries.

## Known limit

A custom transport replaces only the client's default transport. Proxies
picked up from the environment (``HTTPS_PROXY``) are mounted separately by
httpx and bypass this wrapper.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from typing import NamedTuple, Optional

import httpx

logger = logging.getLogger(__name__)

IDEMPOTENCY_HEADER = "Idempotency-Key"
REPLAYED_HEADER = "Idempotent-Replayed"

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_RETRY_STATUSES = frozenset({502, 503, 504})
_RETRY_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.RemoteProtocolError,
    httpx.ReadError,
)


class RetryPolicy(NamedTuple):
    budget_s: float = 60.0
    base_s: float = 0.5
    cap_s: float = 8.0

    def delay(self, attempt: int, remaining_s: float) -> Optional[float]:
        """Full-jitter backoff for ``attempt`` (0-based), or None when the
        budget cannot cover another attempt."""
        if remaining_s <= 0:
            return None
        return min(random.uniform(0, min(self.cap_s, self.base_s * (2 ** attempt))), remaining_s)


def _stamp_key(request: httpx.Request) -> None:
    """Give a write that has no key one, so every retry of it is safe."""
    if request.method not in _SAFE_METHODS and IDEMPOTENCY_HEADER not in request.headers:
        request.headers[IDEMPOTENCY_HEADER] = str(uuid.uuid4())


def _is_retryable_response(response: httpx.Response) -> bool:
    if response.status_code not in _RETRY_STATUSES:
        return False
    return not response.headers.get("content-type", "").startswith("application/json")


def _describe(request: httpx.Request, failure: str) -> str:
    return f"{request.method} {request.url.path}: {failure}"


class RetryingTransport(httpx.AsyncBaseTransport):
    """Async transport for the runner's ``httpx.AsyncClient``s."""

    def __init__(
        self,
        inner: httpx.AsyncBaseTransport | None = None,
        policy: RetryPolicy = RetryPolicy(),
    ) -> None:
        self._inner = inner or httpx.AsyncHTTPTransport()
        self._policy = policy

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        _stamp_key(request)
        # A retry re-sends the body, so it must be replayable. Every caller
        # here already holds the body in memory; this only pins it.
        await request.aread()
        deadline = time.monotonic() + self._policy.budget_s
        attempt = 0
        while True:
            try:
                response = await self._inner.handle_async_request(request)
            except _RETRY_ERRORS as e:
                wait = self._policy.delay(attempt, deadline - time.monotonic())
                if wait is None:
                    raise
                failure = type(e).__name__
            else:
                if not _is_retryable_response(response):
                    if attempt:
                        logger.info("Server reachable again after %d retries: %s", attempt, _describe(request, str(response.status_code)))
                    return response
                wait = self._policy.delay(attempt, deadline - time.monotonic())
                if wait is None:
                    return response
                failure = str(response.status_code)
                await response.aclose()
            if attempt == 0:
                logger.warning("Server unreachable, retrying for up to %.0fs: %s", self._policy.budget_s, _describe(request, failure))
            attempt += 1
            await asyncio.sleep(wait)

    async def aclose(self) -> None:
        await self._inner.aclose()


class SyncRetryingTransport(httpx.BaseTransport):
    """The same policy for the sync ``httpx.Client`` the CLIs build."""

    def __init__(
        self,
        inner: httpx.BaseTransport | None = None,
        policy: RetryPolicy = RetryPolicy(),
    ) -> None:
        self._inner = inner or httpx.HTTPTransport()
        self._policy = policy

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        _stamp_key(request)
        request.read()
        deadline = time.monotonic() + self._policy.budget_s
        attempt = 0
        while True:
            try:
                response = self._inner.handle_request(request)
            except _RETRY_ERRORS as e:
                wait = self._policy.delay(attempt, deadline - time.monotonic())
                if wait is None:
                    raise
                failure = type(e).__name__
            else:
                if not _is_retryable_response(response):
                    if attempt:
                        logger.info("Server reachable again after %d retries: %s", attempt, _describe(request, str(response.status_code)))
                    return response
                wait = self._policy.delay(attempt, deadline - time.monotonic())
                if wait is None:
                    return response
                failure = str(response.status_code)
                response.close()
            if attempt == 0:
                logger.warning("Server unreachable, retrying for up to %.0fs: %s", self._policy.budget_s, _describe(request, failure))
            attempt += 1
            time.sleep(wait)

    def close(self) -> None:
        self._inner.close()
