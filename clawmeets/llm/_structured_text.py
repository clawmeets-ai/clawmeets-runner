# SPDX-License-Identifier: MIT
"""Shared helpers for CLI providers that have no schema-enforcement flag and must
coax raw JSON out of free-text model output (gemini, opencode).

Both providers append :data:`JSON_ONLY_SUFFIX` to the prompt (the prompt builder
already renders the ``== OUTPUT CONTRACT ==`` block describing the action shapes,
so this only pins the *format*), then parse the model's reply with
:func:`parse_json_object` — raw first, markdown-fence-stripped as a fallback.

Single-sourced here so the strict-JSON contract stays identical across providers.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# A ```json ... ``` (or bare ```) fenced block appearing ANYWHERE in the text —
# weaker models sometimes prepend prose before the block despite instructions.
_FENCE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


JSON_ONLY_SUFFIX = (
    "\n\n=== OUTPUT FORMAT (STRICT) ===\n"
    "Respond with ONLY the raw JSON object matching the action schema above. "
    "Do not wrap it in markdown fences (no ```json, no ```). "
    "Do not include any prose, explanation, or trailing text. "
    "Your entire response must be a single parseable JSON object."
)


def strip_markdown_fences(text: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` fences if the text is wrapped in them.

    Returns the inner content unchanged if no fence is detected.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    first_newline = stripped.find("\n")
    if first_newline == -1:
        return stripped  # No newline after fence — not a valid block, leave it.

    inner = stripped[first_newline + 1 :].rstrip()
    if inner.endswith("```"):
        inner = inner[:-3].rstrip()
    return inner


def _json_candidates(text: str) -> list[str]:
    """Ordered candidate substrings to attempt JSON parsing on.

    Most-faithful first: the whole text, a leading-fence-stripped variant, any
    fenced ```json block embedded mid-text, and finally the outermost ``{...}``
    brace span (handles models that wrap the object in prose).
    """
    s = text.strip()
    out: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        c = candidate.strip()
        if c and c not in seen:
            seen.add(c)
            out.append(c)

    add(s)
    add(strip_markdown_fences(s))
    for m in _FENCE_BLOCK_RE.finditer(s):
        add(m.group(1))
    first, last = s.find("{"), s.rfind("}")
    if first != -1 and last > first:
        add(s[first : last + 1])
    return out


def normalize_actions(raw_actions: Any) -> list[dict[str, Any]]:
    """Coerce a schema-less model's ``actions`` array into well-formed dicts.

    Gemini / opencode have no schema-enforcement flag, so weaker (esp. local)
    models emit near-miss action shapes that would otherwise crash
    ``ActionBlock`` construction or silently drop *sibling* actions in the same
    turn. Two salvageable slips are coerced element-by-element:

      - a bare string ``"project_completed"``      → ``{"type": "project_completed"}``
      - a dict whose ``content`` is an object/list  → ``content`` JSON-encoded to a string

    Elements that can't be salvaged to a dict are dropped (so one bad element
    no longer nukes the whole block). Unknown action *types* are left intact —
    ``ActionBlock.typed_actions`` already skips them.
    """
    if not isinstance(raw_actions, list):
        return []
    out: list[dict[str, Any]] = []
    for a in raw_actions:
        if isinstance(a, str):
            out.append({"type": a})
        elif isinstance(a, dict):
            norm = dict(a)
            content = norm.get("content")
            if content is not None and not isinstance(content, str):
                norm["content"] = json.dumps(content, ensure_ascii=False)
            out.append(norm)
        # anything else (number, list, None) is unsalvageable → drop
    return out


def parse_json_object(text: str, *, log_tag: str = "") -> Optional[Any]:
    """Parse ``text`` as a JSON object, tolerating markdown fences and prose.

    Tries, in order: the raw string, a leading-fence-stripped variant, any
    embedded ```json block, and the outermost ``{...}`` span. Returns ``None``
    (and logs at warning) if none parse — schema-less CLI providers (gemini,
    opencode) lean on this since their binaries don't constrain output.
    """
    if not text:
        return None

    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    prefix = f"[{log_tag}] " if log_tag else ""
    logger.warning(
        f"{prefix}response is not valid JSON (tried raw/fence/brace-span); "
        f"head={text[:200]!r}"
    )
    return None
