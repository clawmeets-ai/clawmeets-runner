# SPDX-License-Identifier: MIT
"""
clawmeets/utils/invoke_timeout.py

Single owner of the per-agent LLM invocation timeout: its ``local_settings``
key, its default, its clamp band, and its parse rules.

A turn is bounded by TWO independent ceilings and the lower one wins:

  1. the runner's kill window — ``LLMProvider._invoke_timeout``, the
     ``asyncio.wait_for`` around the invocation;
  2. the server's batch window — ``PendingWork.timeout_seconds``, which
     ``BatchTimeoutChecker`` enforces by sending ``CANCEL_LLM``.

Both derive from the same ``local_settings.invoke_timeout_seconds`` on the
agent's card, and both import this module, so they cannot drift. Raising one
without the other is a knob wired to nothing.

Everything here is total: the runner reads it on the invocation hot path and
the server on the batch-open path, and neither may raise over a hand-edited
card. Out-of-band numbers clamp (the intent was clear); unparseable values fall
back to the default with a warning.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ``local_settings`` key carrying the agent's chosen window.
SETTINGS_KEY = "invoke_timeout_seconds"

# Env var the runner injects into each agent subprocess so an agent can read
# the window currently in force without an HTTP round-trip. Read-only: writing
# it changes nothing, because the provider and the server both read the card.
ENV_VAR = "CLAWMEETS_INVOKE_TIMEOUT_SECONDS"

DEFAULT_SECONDS = 1800   # 30 min — the historical hardcoded value
MIN_SECONDS = 60         # below this a turn cannot realistically finish
MAX_SECONDS = 21600      # 6 h — bounds how long a stuck agent stays stuck


def clamp(seconds: int) -> int:
    """Force ``seconds`` into the supported band."""
    return max(MIN_SECONDS, min(MAX_SECONDS, seconds))


def parse(raw: object) -> Optional[int]:
    """Coerce a stored/typed value to a clamped second count.

    Accepts ints, floats, and numeric strings (``"7200"``, ``" 7200 "``,
    ``"7200.0"``). Returns ``None`` when the value is missing or not a number,
    leaving the caller to decide between "fall back to the default" (readers)
    and "tell the human they typo'd" (the CLI).
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return clamp(int(value))


def resolve(local_settings: Optional[dict]) -> int:
    """Read the window out of an agent's ``local_settings``.

    Total function — a missing key, a ``None``, a non-numeric string, a float,
    or an out-of-band number all resolve to something usable, so callers on the
    hot paths never have to guard.
    """
    if not local_settings:
        return DEFAULT_SECONDS
    raw = local_settings.get(SETTINGS_KEY)
    if raw is None:
        return DEFAULT_SECONDS
    parsed = parse(raw)
    if parsed is None:
        logger.warning(
            "invoke_timeout: ignoring unparseable %s=%r — using default %ds",
            SETTINGS_KEY, raw, DEFAULT_SECONDS,
        )
        return DEFAULT_SECONDS
    return parsed
