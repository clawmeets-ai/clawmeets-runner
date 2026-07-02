# SPDX-License-Identifier: MIT
"""Persist per-invocation LLM usage to disk.

The LLM providers publish an ``LLM_COMPLETE`` notification ``(sandbox_dir, usage)``
on every successful invocation, but historically nothing recorded it (only the git
subscriber listened). :class:`CostRecorder` subscribes once per runner and appends
each invocation's usage to the owning project's
``metadata/projects/{name}-{id}/cost.ndjson`` (one JSON row per invocation), so a
project's total cost / tokens can be aggregated across all participating agents.

The owning project is derived from the notification's ``sandbox_dir`` (the invoke
cwd, always ``{base}/sandbox/projects/{name}-{id}/...``) — so one runner-level
recorder routes every project's invocations without per-project wiring.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..llm.base import LLMUsage
from ..utils.file_io import FileUtil

logger = logging.getLogger(__name__)


class CostRecorder:
    """Append each invocation's :class:`LLMUsage` to its project's ``cost.ndjson``."""

    def __init__(self, metadata_projects_dir: Path) -> None:
        # ``{base}/metadata/projects`` (== ModelContext.metadata_dir)
        self._metadata_dir = Path(metadata_projects_dir)

    async def on_llm_complete(
        self, sandbox_dir: Path, usage: LLMUsage, **kwargs
    ) -> None:
        self._record(sandbox_dir, usage)

    async def on_llm_error(
        self, sandbox_dir: Path, error: object = None,
        usage: Optional[LLMUsage] = None, **kwargs
    ) -> None:
        # A failed turn still burned tokens (it ran N steps before erroring) —
        # record them so spend stays visible even when nothing "completed".
        # The provider attaches partial usage to LLM_ERROR; older providers /
        # paths may not, so guard on its presence.
        if usage is not None:
            self._record(sandbox_dir, usage, error=str(error) if error else "error")

    def _record(
        self, sandbox_dir: Path, usage: LLMUsage, error: Optional[str] = None
    ) -> None:
        proj = _project_dirname(sandbox_dir)
        if proj is None:  # not a project invocation (e.g. eval temp sandbox)
            return
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": usage.model,
            "cost_usd": usage.cost_usd,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_creation_tokens": usage.cache_creation_tokens,
            "requests": usage.requests,
            "tool_calls": usage.tool_calls,
            "duration_ms": usage.duration_ms,
            "room": _room(sandbox_dir),
        }
        if error is not None:
            row["error"] = error  # failed/partial turn — still counts toward spend
        try:
            FileUtil.write(
                self._metadata_dir / proj / "cost.ndjson", row, "ndjson", mode="a"
            )
        except Exception:  # noqa: BLE001 — cost recording must never break a turn
            logger.warning("CostRecorder: failed to write cost.ndjson", exc_info=True)


def _project_dirname(sandbox_dir: Path) -> Optional[str]:
    """Extract ``{name}-{id}`` from ``.../sandbox/projects/{name}-{id}/...``."""
    parts = Path(sandbox_dir).parts
    for i in range(len(parts) - 2):
        if parts[i] == "sandbox" and parts[i + 1] == "projects":
            return parts[i + 2]
    return None


def _room(sandbox_dir: Path) -> Optional[str]:
    """Extract the chatroom name from ``.../chatrooms/{room}/...`` if present."""
    parts = Path(sandbox_dir).parts
    for i in range(len(parts) - 1):
        if parts[i] == "chatrooms":
            return parts[i + 1]
    return None
