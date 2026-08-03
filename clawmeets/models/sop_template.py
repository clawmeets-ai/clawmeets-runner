# SPDX-License-Identifier: MIT
"""
clawmeets/models/sop_template.py

The SOP ``{{blank}}`` grammar, in Python — the parse/fill half of what
``web/frontend/src/components/composer/placeholders.ts`` does for the browser.

A blank is written ``{{Label}}`` or ``{{Label|kind:config}}`` — label before the
pipe, kind after, config after the colon::

    {{Company}}                          free text
    {{Threshold|text:$50}}               free text, $50 pre-offered
    {{Count|number:6}}                   numeric, default 6
    {{Voice|select:warm,punchy,expert}}  pick one of a set
    {{Deadline|date}}                    date quick-picks
    {{Deadline|date:Today,Friday}}       custom quick-picks
    {{Approver|agent}}                   pick from the agent roster

Every kind also accepts a typed-in custom value, so a set is a shortcut, never a
cage. Unknown kinds fall back to text.

**Two implementations of one grammar is the risk this module introduces.** If
they diverge, the message ``clawmeets sop trigger`` sends stops matching the
blanks the SOP editor showed the user. They are pinned to each other by the
shared corpus at ``tests/fixtures/sop_templates.json``, read by both
``tests/test_sop_template.py`` (pytest) and ``placeholders.test.ts`` (vitest) —
so a divergence is a red test on whichever side drifted, not a wrong DM.
Shelling the TS parser from Python was the rejected alternative: it would put a
Node dependency in the runner wheel.

Lives in ``models/`` rather than beside ``cli_sop.py`` because ``models/`` is
rsync'd wholesale into the runner wheel — no build-manifest row to keep in sync —
and because it is a property of the stored ``DeskSop.body``, not of one CLI.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

PhKind = Literal["text", "number", "select", "date", "agent"]

# Kinds, in the order placeholders.ts declares PH_TAGS. Anything else degrades
# to "text".
KINDS: tuple[str, ...] = ("text", "number", "select", "date", "agent")

DATE_DEFAULTS: list[str] = ["Today", "Tomorrow", "End of week", "Next Monday"]

# Ported verbatim from placeholders.ts. The label group excludes `|` and `}` so
# it cannot swallow the spec; the spec group excludes only `}`, so option values
# may contain colons and the pipe is already consumed.
PH_RE = re.compile(r"\{\{\s*([^}|]+?)\s*(?:\|([^}]*))?\}\}")


class SopBlank(BaseModel):
    """One blank in an SOP body — the question the assistant has to ask."""

    label: str
    kind: PhKind
    # Offered values. Empty for text/number, AND for a bare ``agent`` blank:
    # that roster is resolved at fill time, not parse time.
    options: list[str] = []
    # Pre-offered default, for text/number only.
    default: str = ""
    # The original ``{{…}}`` source, so a caller can round-trip a partially
    # filled body back into a template.
    raw: str


def _split_list(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def _make_blank(label: str, spec: str | None, raw: str) -> SopBlank:
    kind = "text"
    cfg = ""
    if spec:
        # Split at the FIRST colon only, so `select:9:16,1:1` keeps its colons.
        i = spec.find(":")
        kind = (spec if i < 0 else spec[:i]).strip().lower()
        cfg = "" if i < 0 else spec[i + 1:].strip()
    if kind not in KINDS:
        kind = "text"

    options: list[str] = []
    default = ""
    if kind in ("select", "agent"):
        options = _split_list(cfg)
    elif kind == "date":
        custom = _split_list(cfg)
        options = custom if custom else list(DATE_DEFAULTS)
    else:
        default = cfg

    return SopBlank(
        label=label.strip(),
        kind=kind,  # type: ignore[arg-type]
        options=options,
        default=default,
        raw=raw,
    )


def has_blanks(body: str) -> bool:
    """True if ``body`` contains at least one blank. Cheap pre-check for
    callers that only need to branch (template vs plain text)."""
    return PH_RE.search(body or "") is not None


def parse_blanks(body: str) -> list[SopBlank]:
    """Parse an SOP body into its ordered, de-duplicated blanks.

    De-duplicated by label (case-insensitively, after trimming) because a
    template naming ``{{Region}}`` twice is asking ONE question: the assistant
    must not ask the user twice, and one ``--set Region=…`` fills both
    occurrences. First occurrence wins, so the earliest spec defines the kind.
    """
    out: list[SopBlank] = []
    seen: set[str] = set()
    for m in PH_RE.finditer(body or ""):
        blank = _make_blank(m.group(1), m.group(2), m.group(0))
        key = blank.label.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(blank)
    return out


def fill(body: str, values: dict[str, str]) -> tuple[str, list[str]]:
    """Substitute ``values`` into every occurrence of every blank.

    Keys match a blank's label case-insensitively after trimming, so the
    assistant may echo the label back with whatever casing the user typed.

    Returns ``(filled_text, missing_labels)`` rather than raising, because
    "which blanks are still open?" is the useful answer for both the CLI's error
    message and the skill's follow-up question. A blank with no value is left as
    its literal ``{{…}}`` source and reported in ``missing_labels`` — the caller
    decides whether that is fatal.

    A ``select`` value outside its option list is ACCEPTED (the frontend's rule:
    a set is a shortcut, never a cage); ``cli_sop`` warns on stderr instead.
    """
    lookup = {k.strip().lower(): v for k, v in (values or {}).items()}
    missing: list[str] = []
    seen_missing: set[str] = set()

    def _sub(m: re.Match[str]) -> str:
        blank = _make_blank(m.group(1), m.group(2), m.group(0))
        key = blank.label.strip().lower()
        val = lookup.get(key)
        if val is None or val == "":
            if key not in seen_missing:
                seen_missing.add(key)
                missing.append(blank.label)
            return m.group(0)
        return val

    return PH_RE.sub(_sub, body or ""), missing


def unknown_labels(body: str, values: dict[str, str]) -> list[str]:
    """The ``values`` keys that name no blank in ``body``.

    Typo protection for ``sop trigger --set``: a value silently dropped because
    its label was misspelled would send a half-filled template, so the CLI
    fails on this instead of on the resulting ``missing_labels``, where the real
    cause would be invisible.
    """
    known = {b.label.strip().lower() for b in parse_blanks(body)}
    return [k for k in (values or {}) if k.strip().lower() not in known]
