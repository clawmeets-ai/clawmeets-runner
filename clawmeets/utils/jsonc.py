# SPDX-License-Identifier: MIT
"""
clawmeets/utils/jsonc.py

Tiny JSONC (JSON-with-comments) helpers used by the MCP starter_config
pipeline. JSONC files live at ``mcps/<name>/starter_config.jsonc``; the
runner/server reads them raw to seed the Configure modal in the web UI, and
also parses them (after stripping comments) for any consumer that needs a
dict view.

The stripper is **string-aware** — it walks character-by-character, tracking
whether the scanner is inside a JSON string literal (with backslash
escaping), so a ``//`` or ``/* … */`` sequence inside a JSON string value
(e.g. an ``"https://…"`` URL) is preserved verbatim. Trailing commas before
``}`` and ``]`` are also stripped so JSONC authors don't have to remember
to remove them when commenting items out.

Not a full JSONC parser — it does not allow unquoted keys, single-quoted
strings, hex literals, or any other JSON5 superset feature. Just comments
+ trailing commas on top of standard JSON.
"""
from __future__ import annotations

import json
import re
from typing import Any

_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def strip_jsonc(text: str) -> str:
    """Strip ``//`` line comments and ``/* … */`` block comments that occur
    OUTSIDE JSON string literals, then strip trailing commas before ``}``
    and ``]``. Returns standard JSON suitable for ``json.loads``.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    while i < n:
        c = text[i]
        if in_string:
            if c == "\\" and i + 1 < n:
                # Preserve the backslash escape verbatim.
                out.append(text[i:i + 2])
                i += 2
                continue
            out.append(c)
            if c == '"':
                in_string = False
            i += 1
            continue
        # Not in a string.
        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "/":
                # Line comment — skip to end of line (preserve the newline).
                j = text.find("\n", i + 2)
                i = n if j == -1 else j
                continue
            if nxt == "*":
                # Block comment — skip to closing */.
                j = text.find("*/", i + 2)
                i = n if j == -1 else j + 2
                continue
        out.append(c)
        i += 1
    return _TRAILING_COMMA.sub(r"\1", "".join(out))


def parse_jsonc(text: str) -> Any:
    """Parse a JSONC string into Python objects."""
    return json.loads(strip_jsonc(text))
