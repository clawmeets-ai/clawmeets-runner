# SPDX-License-Identifier: MIT
"""
clawmeets/models/plan_markdown.py

The PLAN.md grammar — pure text surgery over ``str``.

Every structural fact about a plan document is **derived on every read and never
written**: sections, ids, box counts, digests. The file on disk stays plain
Markdown, which is the property the whole design rests on — a user opens
PLAN.md in any editor and sees a plan, not a serialization format.

**No I/O, no async, no clawmeets imports, no third-party dependency beyond
pydantic.** That is what makes this layer testable against a JSON corpus with no
server running, and it is why the layer above (``models/project_plan.py``) can
be the only place a lock or a changelog append appears.

``difflib`` is imported here and **nowhere else in the repository**. The diff is
a *display artifact*: it is rendered into a review message, a note view and the
tab, and it is never parsed, never stored and never accepted as an input. The
unit of change is a whole-section replacement (§2.3), located by heading text at
apply time, so it lands correctly at any revision — a diff would carry line
positions and applying one to a moved document is guesswork.

Two things earlier revisions of the specification carried and this module
deliberately does not have: ``derive_scope()`` and a ``scope=`` parameter on
``render_clean``. There is no classification of sections and no fixed heading
vocabulary; a reviewer looking for them should find this paragraph instead.

Lives in ``models/`` rather than beside the routes because ``models/`` is
rsync'd wholesale into the runner wheel by ``scripts/build-runner-package.sh``,
so a new module here needs no build-manifest row.
"""
from __future__ import annotations

import difflib
import hashlib
import re

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Limits (§3.4)
# ---------------------------------------------------------------------------

MAX_SECTIONS = 200

#: Slug of the headless section holding prose that precedes the first heading.
#: A preamble survives as an ordinary, addressable section instead of being
#: dropped. Leading underscore so it can never collide with a derived slug,
#: which is always ``[a-z0-9-]``.
LEDE_ID = "_lede"

#: Fallback slug for a heading whose text contains no ASCII alphanumerics at
#: all (a fully non-Latin heading). De-duplicated like any other slug.
FALLBACK_SLUG = "section"

SLUG_MAX_LEN = 64

#: Mirrors ``planAnchor.tsx:52`` / ``:59``. A quote crosses the language
#: boundary and cannot import across it, so these two bounds have a second home
#: here — knowingly. They are **bounds, not behaviour**: drift degrades a cap,
#: it does not break a match, which is why the duplication is acceptable where
#: a duplicated *matching rule* would not be.
#:
#: The two sides do different things at the ceiling and that is deliberate.
#: ``capQuote`` TRUNCATES a human selection, because a truncated selection is
#: still the best available record of what a person highlighted.
#: :func:`quote_from_line` REFUSES above it and returns ``""``, because a
#: truncated *derived* quote is guaranteed not to occur in the document — it
#: would manufacture the loose note the derivation exists to remove.
PLAN_QUOTE_CHARS = 4000
MIN_QUOTE_CHARS = 3

#: ``AC-<m>.<n>`` — the acceptance-criterion label the ``plan``,
#: ``create-project`` and both completion-report skills all teach. Until this
#: parser it was a convention nothing read, so the finest address a note could
#: carry was the enclosing section: an argument about one criterion pointed at
#: a whole milestone. Parsing it does **not** validate it — a plan that ignores
#: the convention parses exactly as it did before and simply has no criteria.
CRITERION_RE = re.compile(r"\bAC-(\d+)\.(\d+)\b")

#: The two layers a section can be in. **Spec is the default and the absence of
#: a marker means spec**, which is the single property that makes this change
#: invisible to every plan already on disk: an unmarked document is locked
#: exactly as strictly as it was before layers existed.
SPEC = "spec"
DETAIL = "detail"

#: ``## Milestones <!-- layer: detail -->`` — a section declares its own layer
#: on its heading line, and descendants inherit it (:func:`section_layers`).
#:
#: **A per-heading marker rather than a ``## Spec`` / ``## Execution`` document
#: split**, because nesting would let the keeper move the boundary by
#: restructuring headings, while a marker cannot be moved without moving
#: :func:`_layer_manifest` and therefore the digest.
LAYER_RE = re.compile(r"<!--\s*layer:\s*(spec|detail)\s*-->", re.I)

#: ``### M2: Session layer <!-- advances: AC-2.1, AC-2.3 -->`` — which criteria
#: a section of the detail layer claims to advance. Read by
#: :func:`parse_coverage`; the ids inside are found with ``CRITERION_RE``, so
#: "what is a criterion label" keeps exactly one answer.
ADVANCES_RE = re.compile(r"<!--\s*advances:\s*([^>]*?)\s*-->", re.I)

#: Block-level markers only, stripped off a source line to get the text a
#: rendered block will actually contain: nested blockquote ``>``, list bullet,
#: ordered marker, task box. ATX headings are handled by ``HEADING_RE``, which
#: also has to eat the closing ``#`` run.
_BLOCKQUOTE_RE = re.compile(r"^[ \t]*(?:>[ \t]?)+")
_BULLET_RE = re.compile(r"^[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+")
_TASKBOX_RE = re.compile(r"^\[[ xX]\][ \t]+")


class PlanLimitError(ValueError):
    """A document exceeded a §3.4 parse limit. Callers surface it as ``413``."""


# ---------------------------------------------------------------------------
# Wire shapes (§3.3). Response-only — neither is ever persisted.
# ---------------------------------------------------------------------------


class PlanSection(BaseModel):
    """One derived section of a plan document.

    ``boxes`` is **display only** (§2.1): there is no milestone model and no
    checkbox state machine. Counting yes, understanding no.
    """

    id: str
    heading: str
    depth: int
    #: ``SPEC`` or ``DETAIL``, **resolved** — the section's own
    #: ``<!-- layer: … -->`` marker if it has one, else the nearest marked
    #: ancestor's, else ``SPEC``. Additive with a default on a response-only
    #: model, so no client breaks and no stored document is read differently.
    layer: str = SPEC
    boxes: tuple[int, int]
    char_start: int
    char_end: int


class Shorthand(BaseModel):
    """One ``{@agent: instruction}`` found inside an HTML comment (§2.2).

    ``checked`` is the task-list marker state of the line the shorthand sits on,
    or ``None`` when it is not on a task-list line. Display only, like
    ``PlanSection.boxes``.

    ``quote`` is the line the shorthand was WRITTEN ON, with the comment holding
    it removed and :func:`quote_from_line` applied — the anchor the note filed
    from this shorthand carries, so it renders beside the task it is about
    instead of at the end of the whole milestone.

    **It is not display-only, and it is the one member here that is not.** The
    other two describe the line; this one is an ADDRESS, and it is the same
    ``(section, quote)`` pair :class:`PlanCriterion` carries, computed by the
    same function. That symmetry is the argument for the field: a shorthand and
    a criterion are both a line of the document a note wants to point at, and
    there is no reason one of them should know how and the other should not.

    ``""`` when the shorthand was ALONE on its line — extraction deletes that
    line outright, so there is nothing left to point at and the note settles at
    the end of its section, which is the honest rung.
    """

    section: str
    owner: str
    text: str
    checked: bool | None = None
    quote: str = ""


class PlanCriterion(BaseModel):
    """One ``AC-<m>.<n>`` occurrence — a criterion's **address**, not a record
    of it.

    ``id`` is the label as the plan writes it, ``section`` the enclosing
    section's slug, ``text`` the source line verbatim, and ``quote`` that line
    reduced to anchorable form (:func:`quote_from_line`). The pair
    ``(section, quote)`` is the whole point: it is exactly what a note carries,
    so a criterion id resolves to an ordinary quoted note and every predicate
    that already reads ``section`` and ``quote`` keeps working untouched.

    ``checked`` is the task-list state of the line, or ``None`` when it is not
    on one. Display only, like ``PlanSection.boxes`` and ``Shorthand.checked``:
    a plan writes criteria on checkboxes about as often as it writes them in
    prose, and neither is a state machine.
    """

    id: str
    section: str
    text: str
    quote: str
    checked: bool | None = None


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------

#: ATX headings only, at most three leading spaces (CommonMark). Setext
#: (``===`` underline) headings are not recognised; no plan writes them and
#: recognising them would make a table separator ambiguous with a heading.
HEADING_RE = re.compile(r"^([ \t]{0,3})(#{1,6})(?:[ \t]+(.*?))?[ \t]*#*[ \t]*$", re.MULTILINE)

#: Fenced code block delimiters, used to make every scanner below fence-aware.
FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})([^\n]*)$", re.MULTILINE)

#: A task-list marker at the head of a list item, either state.
BOX_RE = re.compile(r"^[ \t]*[-*+][ \t]+\[([ xX])\]", re.MULTILINE)

#: Any HTML comment run, including multi-line ones. Non-greedy, so two comments
#: on one line are two matches rather than one that swallows the text between.
COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)

#: §2.2, byte-identical to the mention rule at ``models/chatroom.py:59`` — a
#: name that routes here ``@mentions`` there.
SHORTHAND_RE = re.compile(r"\{@([a-zA-Z][a-zA-Z0-9_-]*)\s*:\s*([^{}]{1,2000})\}")

#: Digest normalization 1 (§2.1): every task-list marker folds to one form, so a
#: coordinator ticking a box on a default-shaped plan does not read as a spec
#: change. The seed template puts its checkboxes inside ``## Milestones``, which
#: is now the only place progress lives.
_MARKER_RE = re.compile(r"^([ \t]*[-*+][ \t]+\[)[ xX](\])", re.MULTILINE)


def _fence_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges of fenced code blocks, fences included.

    Every scanner in this module skips these. A plan that documents this very
    grammar contains ``## Goal`` and ``{@agent: …}`` inside code samples, and
    those must be prose, not structure — without this a plan describing the
    shorthand would file notes about itself on save.
    """
    spans: list[tuple[int, int]] = []
    open_at: int | None = None
    open_char = ""
    open_len = 0
    for m in FENCE_RE.finditer(text):
        marker, info = m.group(1), m.group(2)
        char, length = marker[0], len(marker)
        if open_at is None:
            # A backtick fence's info string may not contain a backtick.
            if char == "`" and "`" in info:
                continue
            open_at, open_char, open_len = m.start(), char, length
        elif char == open_char and length >= open_len and not info.strip():
            spans.append((open_at, m.end()))
            open_at = None
    if open_at is not None:  # unclosed fence runs to EOF
        spans.append((open_at, len(text)))
    return spans


def _outside_fences(pos: int, spans: list[tuple[int, int]]) -> bool:
    return not any(start <= pos < end for start, end in spans)


def heading_text(raw: str) -> str:
    """A heading's text with its HTML comments folded out, whitespace collapsed.

    **The one thing that lets a marker live on a heading line at all.** Slugs
    come from heading text, so ``## Milestones <!-- layer: detail -->`` would
    otherwise slug to ``milestones-layer-detail``: every note anchored to the
    section goes dangling, and marking a section would silently rename it.

    Folding here rather than special-casing ``LAYER_RE`` keeps the rule the one
    this module already states everywhere else — **a comment is inert** — and it
    closes a hole that predates layers: shorthand lives inside comments, so a
    ``{@bob: …}`` written on a heading line changed that section's slug until
    :func:`extract_shorthand` removed it, and then changed it back.
    """
    return re.sub(r"[ \t]+", " ", COMMENT_RE.sub("", raw)).strip()


def slugify_heading(text: str, taken: set[str]) -> str:
    """Derive a section id from heading text (§2.1).

    Lowercased, non-alphanumerics folded to ``-``, collapsed, trimmed, truncated
    to 64 characters, then de-duplicated as ``x`` / ``x-2`` / ``x-3``. A heading
    that slugs to nothing — a fully non-Latin one, say — falls back to
    ``section``, then ``section-2``. Ids stay unique, the heading text
    round-trips verbatim, and nothing is lost.

    ``taken`` is read, never mutated: the caller owns the accumulator.

    The id is **advisory** — it names what a note is *about*. Nothing depends on
    it to apply anything, because a change is located by heading text at apply
    time, so a dangling id costs a stale label and nothing else.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    slug = slug[:SLUG_MAX_LEN].strip("-") or FALLBACK_SLUG
    if slug not in taken:
        return slug
    n = 2
    while f"{slug}-{n}" in taken:
        n += 1
    return f"{slug}-{n}"


def count_boxes(text: str) -> tuple[int, int]:
    """``(checked, total)`` over ``- [x]`` / ``- [ ]`` at any nesting depth.

    Display only (§2.1, §11). Markers inside fenced code are not counted — a
    code sample of a checklist is prose about a checklist.
    """
    spans = _fence_spans(text)
    checked = total = 0
    for m in BOX_RE.finditer(text):
        if not _outside_fences(m.start(), spans):
            continue
        total += 1
        if m.group(1) in ("x", "X"):
            checked += 1
    return checked, total


def parse_sections(body: str) -> list[PlanSection]:
    """Derive every section of ``body``, in **source order**, at the depth read.

    A section is a heading plus every line up to the next heading of the same or
    shallower depth, or EOF; nested headings are nested sections and appear in
    the flat list too. No vocabulary is declared and **no slug is required to
    exist, including ``milestones``** — a coordinator writes whatever sections
    the work needs, and a client that renders this list in the order it is
    returned renders a plan it has never seen before correctly (§2.4).

    Two shapes get named handling, because both occur in documents this system
    must read:

    * **The document title.** A leading ``# Heading`` on a document that also
      has deeper headings is the *title*, not a section: ``# Project Plan: Acme``
      followed by five ``##`` headings yields **five** sections, not six, and a
      section extent never covers the title line. Otherwise the title's extent
      would swallow the whole document and ``replace_section`` on it would
      overwrite every other section.
    * **The lede.** Prose before the first section becomes a headless section
      ``_lede`` so a preamble survives (§2.4). It is emitted only when that
      region holds something other than the title line and whitespace.

    Raises ``PlanLimitError`` past ``MAX_SECTIONS`` (§3.4).
    """
    fences = _fence_spans(body)
    heads = [
        (m.start(), m.end(), len(m.group(2)), (m.group(3) or "").strip())
        for m in HEADING_RE.finditer(body)
        if _outside_fences(m.start(), fences)
    ]

    title_end = 0
    if heads and heads[0][2] == 1 and not body[: heads[0][0]].strip() and any(h[2] >= 2 for h in heads):
        title_end = heads[0][1]
        heads = heads[1:]

    if len(heads) > MAX_SECTIONS:
        raise PlanLimitError(f"plan has {len(heads)} sections, limit is {MAX_SECTIONS}")

    first_start = heads[0][0] if heads else len(body)
    sections: list[PlanSection] = []
    taken: set[str] = set()

    lede = body[title_end:first_start]
    if lede.strip():
        sections.append(
            PlanSection(
                id=LEDE_ID,
                heading="",
                depth=0,
                boxes=count_boxes(lede),
                char_start=title_end,
                char_end=first_start,
            )
        )
        taken.add(LEDE_ID)

    # Resolved layer, carried down the heading stack. A section's own marker
    # wins; otherwise it inherits from the nearest marked ancestor; otherwise
    # SPEC. **Inheritance is what makes a nested milestone free**: a rule that
    # only knew depth 2 would mis-file every ``### M2`` edit, because
    # :func:`split_by_section` reports the most specific section that changed.
    stack: list[tuple[int, str]] = []

    for i, (start, _end, depth, heading) in enumerate(heads):
        stop = len(body)
        for nxt_start, _e, nxt_depth, _h in heads[i + 1 :]:
            if nxt_depth <= depth:
                stop = nxt_start
                break
        while stack and stack[-1][0] >= depth:
            stack.pop()
        own = LAYER_RE.search(heading)
        layer = own.group(1).lower() if own else (stack[-1][1] if stack else SPEC)
        stack.append((depth, layer))
        slug = slugify_heading(heading_text(heading), taken)
        taken.add(slug)
        sections.append(
            PlanSection(
                id=slug,
                heading=heading_text(heading),
                depth=depth,
                layer=layer,
                boxes=count_boxes(body[start:stop]),
                char_start=start,
                char_end=stop,
            )
        )
    return sections


def section_layers(body: str) -> dict[str, str]:
    """``{slug: "spec" | "detail"}`` for every section, at every depth.

    A thin read over :func:`parse_sections` — the resolution itself happens
    there, because a section's layer is derived from the same heading walk as
    its depth and its extent, and deriving it twice is two answers waiting to
    disagree. Fence-aware for free, which matters: a plan documenting this very
    convention must not classify itself out of the lock.
    """
    return {s.id: s.layer for s in parse_sections(body)}


def section_extent(body: str, slug: str) -> tuple[int, int] | None:
    """``(start, end)`` character offsets of a section, heading line included.

    ``None`` when the slug does not resolve — which is ``section_missing``
    (§3.3). Callers that owe a ``409`` check this first.
    """
    for sec in parse_sections(body):
        if sec.id == slug:
            return sec.char_start, sec.char_end
    return None


def _append_section(body: str, text: str) -> str:
    """Put a new section at the end of the document, adding separators only.

    An unresolvable anchor **never relocates an insert** and never guesses a
    position (§15.4): a create lands at the end, where a human can see it and
    move it, rather than somewhere a stale slug happened to point.
    """
    block = text.strip("\n")
    if not block.strip():
        return body
    if not body:
        return block + "\n"
    if body.endswith("\n\n"):
        sep = ""
    elif body.endswith("\n"):
        sep = "\n"
    else:
        sep = "\n\n"
    return body + sep + block + "\n"


def replace_section(body: str, slug: str, text: str) -> str:
    """Splice ``text`` over the section's extent.

    ``text`` is the section's **full** replacement, heading line included; an
    empty ``text`` deletes the section, which is what ``SectionEdit.text == ""``
    means (§2.3). An unresolved slug appends at the end of the document.

    The splice is **position-independent** (AC-1.3): the extent is found by
    heading text at apply time, so replacing ``### M2`` gives the same result
    whether every other section is untouched or all rewritten, and no byte
    outside the extent moves. The trailing newline run of the old extent is
    carried onto the new text so the document's shape survives a replacement
    whose text was written without one.
    """
    if not text.strip():
        return delete_section(body, slug)
    extent = section_extent(body, slug)
    if extent is None:
        return _append_section(body, text)
    start, end = extent
    old = body[start:end]
    tail = old[len(old.rstrip("\n")) :]
    return body[:start] + text.rstrip("\n") + tail + body[end:]


def extent_key(text: str) -> str:
    """A section extent reduced to what a splice can actually preserve.

    **The one place "did this section change?" is allowed to be answered**, and
    it lives here because the two lines directly above it are its whole proof:
    :func:`replace_section` throws away the caller's trailing newline run and
    re-attaches the document's own. Two extents that differ only in that run are
    therefore interchangeable for every operation the question gates — the run
    is the document's shape, not the section's content.

    That matters because ``## Approval`` is LAST in the seed template, so its
    extent runs to EOF and ends in one newline; the moment any section is
    appended after it the extent stops at that heading and ends in two. A raw
    comparison called that an edit, and since the go-note's ``base_section`` is
    captured at project creation it called it an edit on **every** project — the
    Accept that releases the execution gate refused to stage, with the plan's
    approval text byte-identical the whole time.

    ``rstrip("\n")`` and **not** ``.rstrip()``. Trailing *spaces* are a markdown
    hard break and :func:`replace_section` takes them from the caller verbatim,
    so a trailing-space difference is real content and must keep reading as a
    change. The normalization is exactly as wide as the splice's own tolerance
    and not one byte wider.

    It deliberately does **not** truncate at a nested heading the way
    :func:`_section_texts`'s *own text* does. A proposal replaces the whole
    extent, so an edit inside ``### M2`` really does make a ``## Milestones``
    proposal stale — reporting that as unchanged would let an apply delete the
    nested section.
    """
    return text.rstrip("\n")


def append_to_section(body: str, slug: str, text: str) -> str:
    """Append ``text`` inside the section's extent, before the next heading.

    Exactly one newline joins it to the section's last non-blank line, so
    appending a list item extends the list; supply your own leading blank line
    in ``text`` for a new paragraph. Same unresolved-slug rule as
    ``replace_section``.
    """
    if not text.strip():
        return body
    extent = section_extent(body, slug)
    if extent is None:
        return _append_section(body, text)
    start, end = extent
    old = body[start:end]
    core = old.rstrip("\n")
    tail = old[len(core) :]
    return body[:start] + core + "\n" + text.strip("\n") + tail + body[end:]


def retitle_section(body: str, slug: str, title: str) -> str:
    """Rewrite the section's heading **text**, keeping its depth and its body.

    The one section operation :func:`replace_section` cannot express from
    outside this module, because building the new heading line needs the old
    one's depth — and depth is grammar, so deriving it in a caller would be the
    second heading parser this module exists to prevent.

    Re-slugs the section, which is §2.1's one known cost: a note pinned to the
    old slug goes dangling and ``list-notes --dangling`` is how it is found. A
    slug that does not resolve, or one whose extent opens on something that is
    not an ATX heading (the ``_lede``), is a **no-op, never an error** — same
    rule as :func:`delete_section`, and for the same reason: this function's job
    is the surgery, not the policy.
    """
    extent = section_extent(body, slug)
    if extent is None:
        return body
    start, end = extent
    old = body[start:end]
    head, sep, rest = old.partition("\n")
    match = HEADING_RE.match(head)
    if match is None:
        return body
    indent, hashes, text = match.groups()
    # **The markers on the heading line survive a retitle**, because they are
    # grammar and not title: ``<!-- layer: detail -->`` and
    # ``<!-- advances: … -->`` say what the section IS, while ``title`` says
    # what it is called. Dropping them would silently unlock a detail section —
    # or, on a milestone, drop its coverage claim — through an operation whose
    # whole contract is *"keeping its depth and its body"*.
    #
    # A ``title`` that carries its own comment is taken at its word and nothing
    # is re-appended — otherwise a caller that reads the heading, edits it and
    # sends it back doubles every marker on it.
    incoming = title.strip()
    kept = (
        ""
        if COMMENT_RE.search(incoming)
        else "".join(m.group(0) for m in COMMENT_RE.finditer(text or ""))
    )
    head = f"{indent}{hashes} {incoming}"
    return body[:start] + (f"{head} {kept}" if kept else head) + sep + rest + body[end:]


def delete_section(body: str, slug: str) -> str:
    """Remove the section's extent. A missing slug is a **no-op, never an error**.

    Callers that owe a ``409`` on a missing section check ``section_extent``
    first (§5.5); this function's job is the surgery, not the policy.
    """
    extent = section_extent(body, slug)
    if extent is None:
        return body
    start, end = extent
    return body[:start] + body[end:]


def _section_texts(body: str) -> dict[str, tuple[str, str]]:
    """``{slug: (own_text, full_text)}`` for every section of ``body``.

    ``own_text`` stops at the first nested section, ``full_text`` runs to the end
    of the extent. Every descendant of a section is contiguous at the tail of
    its extent, so the split is a single offset.
    """
    sections = parse_sections(body)
    out: dict[str, tuple[str, str]] = {}
    for i, sec in enumerate(sections):
        own_end = sec.char_end
        for nxt in sections[i + 1 :]:
            if sec.char_start < nxt.char_start < sec.char_end:
                own_end = nxt.char_start
                break
        out[sec.id] = (body[sec.char_start : own_end], body[sec.char_start : sec.char_end])
    return out


def rename_pair(before: str, after: str) -> tuple[str, str] | None:
    """``(old_slug, new_slug)`` when ``after`` is ``before`` with one section
    retitled, else ``None``.

    **Because a rename is the one edit :func:`split_by_section` cannot state.**
    Slugs come from heading text, so retitling re-slugs, and the split reports
    the change as a deletion plus an addition — two rows, neither of which is the
    edit anyone made. A caller presenting those rows to a human shows them
    *"remove Goal"* and *"add Objective"* and asks them to decide each on its
    own, when the only coherent decision is on the pair.

    **Deliberately conservative, and it answers ``None`` far more often than a
    similarity test would.** Three conditions, all structural, none of them a
    guess about content: exactly one slug disappears, exactly one appears, and
    the two sit at the same index and the same depth in their own documents. Two
    simultaneous renames, or a rename beside an unrelated create, answer ``None``
    and the caller keeps the two rows it already had — a worse presentation, and
    never a wrong one. That asymmetry is the whole design: mispairing two
    sections silently rewrites the wrong heading when the pair is accepted.

    Position is the signal rather than the text, because the text is exactly what
    changed. A retitle moves no section, so a renamed heading is still the *n*th
    heading of its document; anything that also reorders is not a retitle this
    function is willing to name.
    """
    b = parse_sections(before)
    a = parse_sections(after)
    b_slugs = [s.id for s in b]
    a_slugs = [s.id for s in a]
    gone = [s for s in b_slugs if s not in set(a_slugs)]
    added = [s for s in a_slugs if s not in set(b_slugs)]
    if len(gone) != 1 or len(added) != 1:
        return None
    old, new = gone[0], added[0]
    if b_slugs.index(old) != a_slugs.index(new):
        return None
    old_sec = next(s for s in b if s.id == old)
    new_sec = next(s for s in a if s.id == new)
    if old_sec.depth != new_sec.depth:
        return None
    return old, new


def split_by_section(before: str, after: str) -> list[tuple[str, str, str]]:
    """Per-section ``(slug, before_text, after_text)`` for every changed section.

    The engine behind the PLAN.md upload interception (§6.1): a non-keeper's
    upload becomes one note per differing section, and a keeper's stale upload
    applies only what it actually changed. A section present on one side only
    appears with ``""`` on the other — ``base: ""`` for an addition, ``text: ""``
    for a deletion, and a heading rename is **both**, because a rename re-slugs
    and is a delete plus a create (§2.3).

    **A section is judged on its own text, not its extent, but is returned in
    full.** Editing ``### M2`` also changes ``## Milestones``'s extent, because
    the extent contains it; reporting both would file two notes for one edit and
    put the whole milestone list in the second one. Judging on own text reports
    the most specific section that actually changed, while the ``base`` and
    ``text`` returned are the whole sections — which is what
    ``SectionEdit``/``replace_section`` operate on (§2.3).

    Deletions and edits come first in ``before``'s document order, then
    additions in ``after``'s. Comparison ignores trailing whitespace: a section
    that merely became the last one in the document differs only in its trailing
    newline run, and that is not an edit anyone wrote.
    """
    b_secs = _section_texts(before)
    a_secs = _section_texts(after)

    out: list[tuple[str, str, str]] = []
    for slug, (b_own, b_full) in b_secs.items():
        entry = a_secs.get(slug)
        if entry is None:
            out.append((slug, b_full, ""))
        elif b_own.rstrip() != entry[0].rstrip():
            out.append((slug, b_full, entry[1]))
    for slug, (_a_own, a_full) in a_secs.items():
        if slug not in b_secs:
            out.append((slug, "", a_full))
    return out


def _enclosing_section(sections: list[PlanSection], pos: int) -> str:
    """Id of the deepest section containing ``pos``, or ``""``."""
    found = ""
    for sec in sections:
        if sec.char_start <= pos < sec.char_end:
            found = sec.id
    return found


def _line_bounds(text: str, pos: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return start, len(text) if end == -1 else end


def strip_block_markers(line: str) -> str:
    """A source line with the markers remark CONSUMES taken off the front.

    Blockquote ``>``, ATX ``#``s, list bullet, ordered ``1.``, task box
    ``[ ]``/``[x]``. Those never survive into the rendered DOM, so text carrying
    one cannot be compared against what a reader sees.

    Named rather than left inline in :func:`quote_from_line` because
    :func:`section_holding_quote` asks the same question of a whole section and
    a second copy of this list is a second answer to *"what is a marker"*. It
    does NOT collapse whitespace or apply :func:`quote_from_line`'s length
    bounds — those are that function's own policy, not part of the grammar.
    """
    text = _BLOCKQUOTE_RE.sub("", line)
    heading = HEADING_RE.match(text)
    if heading is not None:
        return heading.group(3) or ""
    return _TASKBOX_RE.sub("", _BULLET_RE.sub("", text))


def _match_view(text: str) -> str:
    """``text`` reduced to what a quote can be compared against.

    Every line stripped of its block markers, whitespace collapsed across the
    whole run — the server-side twin of ``planAnchor``'s ``normalizeText``, so a
    quote that anchors in the browser is a quote this can find.
    """
    stripped = " ".join(strip_block_markers(ln) for ln in text.splitlines())
    return " ".join(stripped.split())


def section_holding_quote(body: str, quote: str) -> str:
    """The slug of the one section whose OWN text contains ``quote``, or ``""``.

    **Own text, not the extent**, and that is what makes the answer unique
    rather than merely first. :func:`parse_sections` nests: a line inside a
    child section is also inside its parent's extent, so a containment test over
    extents matches every ancestor and has to break the tie by depth. Own texts
    partition the document instead — every character of the body belongs to
    exactly one of them — so "which section is this line in" has one answer and
    no tie to break.

    ``""`` on **no match and on more than one**, and the second is not
    over-caution. A short excerpt genuinely can occur in two sections, and a
    caller filing a note against a guess would anchor an argument to the wrong
    passage; the honest degrade is the un-sectioned note the caller already
    wrote, which the editor still shows — just at the bottom of the page rather
    than in the body.
    """
    needle = _match_view(quote)
    if not needle:
        return ""
    hits = [
        slug for slug, (own, _full) in _section_texts(body).items()
        if needle in _match_view(own)
    ]
    return hits[0] if len(hits) == 1 else ""


def quote_from_line(line: str) -> str:
    """A source line reduced to what a rendered block will actually contain.

    Strips **block-level markers only**, via :func:`strip_block_markers` — those
    are the markers remark consumes, so a quote carrying one can never match the
    flattened DOM text ``locate`` searches, and the note it anchors is loose
    forever.

    Deliberately leaves **inline** emphasis alone. ``locate``'s ``stripMarkers``
    fallback and :func:`_comparison_view` already normalise ``*``/``_``/`` ` ``
    on both sides of every comparison, and stripping them here would make the
    stored quote diverge from the document's own bytes — the quote would no
    longer be an excerpt of anything.

    Whitespace collapses, matching ``normalizeText`` on the other side.

    Returns ``""`` in the two cases where an anchor would be a lie rather than
    a miss: below ``MIN_QUOTE_CHARS``, because a two-character anchor matches
    everywhere; and above ``PLAN_QUOTE_CHARS``, because truncating to fit would
    produce a string that does not occur in the document. Both degrade to the
    unanchored note that is today's default, which is the correct failure
    direction — a missing anchor costs a pane entry, a wrong one costs trust.
    """
    text = " ".join(strip_block_markers(line).split())
    if len(text) < MIN_QUOTE_CHARS or len(text) > PLAN_QUOTE_CHARS:
        return ""
    return text


def heading_line(section_text: str) -> str:
    """The ATX heading line a section's text opens with, or ``""``.

    Three lines, and it is here rather than in its caller for the reason
    :func:`retitle_section` gives at length: heading shape is grammar, and a
    caller deciding for itself what counts as a heading is the second heading
    parser this module exists to prevent.

    ``""`` for the ``_lede``, which opens on prose and has no heading — which is
    why a comment anchored to the preamble derives no quote and keeps today's
    behaviour.
    """
    head = section_text.split("\n", 1)[0]
    return head if HEADING_RE.match(head) else ""


def heading_depth(section_text: str) -> int:
    """The LEVEL of the ATX heading a section's text opens with, or ``0``.

    Beside :func:`heading_line` and for the same reason: heading shape is
    grammar, and a caller counting ``#`` characters for itself is the second
    heading parser this module exists to prevent. It reads ``HEADING_RE``'s
    ``#``-run group, which is the same expression :func:`parse_sections` uses to
    fill ``PlanSection.depth``, so the two can never disagree about what level a
    heading is at.

    ``0`` for the ``_lede``, matching the ``depth=0`` that section is given —
    prose is not at a level.
    """
    head = heading_line(section_text)
    m = HEADING_RE.match(head) if head else None
    return len(m.group(2)) if m else 0


def heading_slug(section_text: str) -> str:
    """The id the section ``section_text`` opens with WOULD be given, or ``""``.

    Beside :func:`heading_line` and :func:`heading_depth`, and for the third
    time the same reason: heading shape is grammar. This is the composition
    those two leave un-said, and it is :func:`parse_sections`' own — the ``#``
    run off via :func:`strip_block_markers`, then :func:`heading_text` to fold
    out the inert comment, then :func:`slugify_heading`.

    **All three steps, in that order, because two of them are individually
    survivable and that is the trap.** Skipping ``heading_text`` slugs
    ``## Milestones <!-- layer: detail -->`` to ``milestones-layer-detail``,
    which is the bug that once made marking a section silently rename it.
    Skipping ``strip_block_markers`` looks harmless — ``#`` folds to ``-`` and
    :func:`slugify_heading` strips a leading one, so the answer comes out RIGHT
    by accident on every heading anyone has written. An answer that is only
    accidentally the same as ``parse_sections``' is exactly the second heading
    parser this module exists to prevent, so the accident is not relied on.

    ``taken`` is EMPTY, deliberately, so the answer is the slug this heading
    takes **on its own** and never an ``x-2`` de-duplication against a document
    it is not in yet. The one caller that needs it — the create test in
    :func:`add_note` — is asking whether a proposal means to write the section
    it names, and a proposal is one heading rather than a document.

    ``""`` for text that does not open on a heading, which is the honest answer
    to *"which section does this create"* from a replacement that creates none.
    """
    head = heading_line(section_text)
    if not head:
        return ""
    return slugify_heading(heading_text(strip_block_markers(head)), set())


def drops_heading(current: str, replacement: str) -> bool:
    """Would replacing ``current`` with ``replacement`` delete the heading?

    :func:`replace_section`'s docstring asks for *"the section's **full**
    replacement, heading line included"* — but that was a request with nothing
    enforcing it, and an agent that ticks two boxes and sends back only the two
    bullet lines silently deletes ``### M1``. The section's content is then
    absorbed into the section above it, the enclosing ``##`` digest moves, and
    the spec lock refuses the correcting write as a spec edit. The whole chain
    starts here, so this is where it is cut.

    The grammar question lives in this module rather than in the caller for the
    reason :func:`heading_line` and :func:`retitle_section` both give: a caller
    deciding for itself what counts as a heading is the second heading parser
    this module exists to prevent.

    **An ATX heading, not the same one.** Retitling is legitimate (that is what
    :func:`retitle_section` and ``plan update --title`` do), so the question is
    whether the replacement opens with a heading at all — never whether it
    matches.

    **And "at all" is not the whole grammar**, which is what the incident that
    added :func:`relevels_heading` proved. A coordinator sent the body of
    ``## Milestones`` back with the ``##`` line stripped, so the text opened on
    ``### M1`` — a heading, so this function said ``False`` — and the splice put
    a level-3 subtree over a level-2 extent. Whether the LEVEL may move is the
    other half of the question and is asked next door; the two are kept apart so
    each stays a single sentence, and they never both fire on one input.

    ``False`` in the three cases that have nothing to drop: an empty
    ``replacement`` (that is a delete, and :func:`delete_section`'s job), an
    empty ``current``, and the ``_lede``, which opens on prose and carries no
    heading. Judged on the first **non-blank** line, because a replacement
    written with a leading blank line still carries its heading — the splice
    keeps that blank line and the document still parses.
    """
    if not current.strip() or not replacement.strip():
        return False
    if not heading_line(current):
        return False
    first = next(line for line in replacement.split("\n") if line.strip())
    return not heading_line(first)


def relevels_heading(current: str, replacement: str) -> bool:
    """Would replacing ``current`` with ``replacement`` change the section's
    LEVEL?

    :func:`drops_heading` asks whether the replacement opens with a heading at
    all. This asks the second half of the same question, and it is the half that
    let a whole section leave the document: the body of ``## Milestones`` was
    sent back with its ``##`` line stripped, so the text opened on ``### M1``.
    That IS a heading, so the first guard passed, and the splice put a level-3
    subtree over a level-2 extent — the slug ``milestones`` left the document,
    its four ``###`` children were re-parented, and every correcting write after
    it was then refused by the spec lock as an edit to what the plan says.

    **Both directions, and promotion is not the milder one.** A section's extent
    runs to the next heading of the same or shallower depth
    (:func:`parse_sections`), so a DEEPER replacement is absorbed by the section
    above it, and a SHALLOWER one terminates its own parent early and adopts
    every sibling that followed it. Demotion loses the section that was written;
    promotion silently moves sections that were not. Refusing one and allowing
    the other would leave the worse of the two open.

    **A retitle at the same level stays legal**, which is the whole reason this
    is a sibling of :func:`drops_heading` rather than a widening of it: the
    question is the level, never the words. It is the same invariant
    :func:`retitle_section` already keeps from the other door — *"keeping its
    depth"* — stated for the door that could not express it.

    ``False`` on the three shapes with no level to change, exactly as
    :func:`drops_heading` short-circuits them: an empty ``replacement`` (a
    delete, and :func:`delete_section`'s job), an empty ``current``, and the
    ``_lede``, whose depth is ``0`` because it has no heading at all. And
    ``False`` on a headingless replacement, because that one is
    :func:`drops_heading`'s answer — the two predicates are disjoint on purpose,
    so their caller's two refusals can never race for the same input.
    """
    if not current.strip() or not replacement.strip():
        return False
    depth = heading_depth(current)
    if not depth:
        return False
    first = next(line for line in replacement.split("\n") if line.strip())
    incoming = heading_depth(first)
    return bool(incoming) and incoming != depth


def _heading_tally(body: str) -> dict[str, int]:
    """How many headings the document gives each **pre-dedup** slug.

    Keyed on the slug rather than the heading text, because the slug is the
    thing that collides: ``## Approval`` and ``## approval!`` are two headings
    and one id, and it is the id every caller addresses a section by.
    :func:`slugify_heading` is handed an EMPTY ``taken`` on purpose — the
    de-duplicated ids are what this function exists to detect, so counting them
    would count each one once and find nothing.

    The ``_lede`` is skipped (``depth == 0``): it has no heading to collide.

    A plain ``dict`` rather than a ``Counter``: this module's import list is
    pinned to five names by
    ``test_the_grammar_module_imports_nothing_from_clawmeets_and_does_no_io``,
    and a tally is not worth the sixth.
    """
    tally: dict[str, int] = {}
    for sec in parse_sections(body):
        if not sec.depth:
            continue
        slug = slugify_heading(sec.heading, set())
        tally[slug] = tally.get(slug, 0) + 1
    return tally


def duplicated_heading(before: str, after: str) -> str:
    """The first heading ``after`` duplicates and ``before`` did not, or ``""``.

    **The ``duplicate id`` state, asked as a question a write can be refused
    for.** Two headings under one slug is a state this module tolerates —
    :func:`slugify_heading` de-duplicates as ``x`` / ``x-2`` so nothing is ever
    lost, and ``plan show --sections`` renders a *"duplicate heading"* warning
    beside it — but it is damage, not a feature, and ``add_note`` already
    refuses to create it from the note door
    (``test_the_typo_that_used_to_append_a_second_heading_under_one_slug``:
    *"the only one of these with real damage"*). The write door never asked.

    The incident that closed it: a coordinator regenerated ``## Milestones``
    into a file, copied one section too far, and the file ended with the
    document's own ``## Approval`` block. The splice gave the plan two of them.
    Ids are assigned in **source order**, so the injected copy took ``approval``
    and the user's real, accepted section was demoted to ``approval-2`` — and
    then reported to them, by a spec lock doing its job, as a brand-new section
    to approve. Worse on an unaccepted plan, where the write simply lands:
    :func:`section_extent` resolves a slug to the FIRST match, so every reader
    of ``approval`` — including the coordinator's own go signal — would read the
    injected copy.

    **Only a duplicate the write INTRODUCES.** A plan that already holds two
    ``### Notes`` keeps working, and every write to it stays possible; refusing
    on the state rather than on the transition would make a document that
    already exists unwritable and offer no way back. Counted per slug, so
    adding a THIRD copy is caught as surely as the second.

    Returns the offending heading rebuilt at its own depth (``"## Approval"``),
    because the refusal owes the writer the line — an agent told only *"a
    duplicate heading"* has to bisect a section it just generated to find which
    one. Falsy when there is nothing to report, which is what makes it readable
    as the predicate as well as the message.
    """
    def _lines(rows: list[PlanSection]) -> dict[tuple[int, str], int]:
        """How many times each heading LINE — depth and text together — occurs."""
        out: dict[tuple[int, str], int] = {}
        for s in rows:
            out[(s.depth, s.heading)] = out.get((s.depth, s.heading), 0) + 1
        return out

    was, now = _heading_tally(before), _heading_tally(after)
    sections = [s for s in parse_sections(after) if s.depth]
    seen = _lines([s for s in parse_sections(before) if s.depth])
    lines = _lines(sections)

    for sec in sections:
        base = slugify_heading(sec.heading, set())
        # `was.get`, NOT `was[...]`: a write may introduce BOTH copies at once
        # — two new sections under one slug, the purest form of the copy-paste
        # this exists to catch — and that slug is absent from the before-tally.
        if not (now[base] > 1 and now[base] > was.get(base, 0)):
            continue
        # **NAME THE OCCURRENCE THE WRITE ADDED, not the first one under the
        # slug.** They collide on the id but need not share a line: `## A` and
        # `## a!` are one slug and two headings, and quoting the one that was
        # already there sends the writer looking for text they did not send.
        # Identified by the heading LINE — depth and text together, since
        # `## A` and `### A` are also one slug and two lines — rather than by
        # position, because position is exactly what a copied-in section makes
        # unreliable: the injected copy in the incident landed mid-document,
        # ahead of the original. When every occurrence is byte-identical there
        # is nothing to choose between them and the first is named.
        colliding = [
            s for s in sections if slugify_heading(s.heading, set()) == base
        ]
        added = next(
            (
                s for s in colliding
                if lines[(s.depth, s.heading)] > seen.get((s.depth, s.heading), 0)
            ),
            colliding[0],
        )
        return f"{'#' * added.depth} {added.heading}"
    return ""


def parse_criteria(body: str) -> list[PlanCriterion]:
    """Every ``AC-<m>.<n>`` in the document, in source order, fence-aware.

    **Occurrences, not definitions.** An id written twice yields two rows, so a
    duplicate is visible to a client as two rows sharing an ``id`` — exactly the
    way two sections sharing a ``heading`` *are* the duplicate-heading warning
    (:func:`slugify_heading`, ``_show_sections``). §6.B's index is a closed
    enumeration precisely so a fact the response already carries does not get a
    second name, and "this id is ambiguous" is such a fact.

    One row per occurrence also means a criterion **referred to** from another
    section — a completion report citing ``AC-2.3``, say — is a row of its own
    rather than a silent overwrite of the definition. First in document order is
    the definition; callers that want one answer take the first.

    Criteria inside fenced code are prose about criteria, like every other
    scanner here: a plan documenting this convention must not file notes about
    itself.
    """
    fences = _fence_spans(body)
    sections = parse_sections(body)
    out: list[PlanCriterion] = []
    for m in CRITERION_RE.finditer(body):
        if not _outside_fences(m.start(), fences):
            continue
        line_start, line_end = _line_bounds(body, m.start())
        line = body[line_start:line_end]
        box = BOX_RE.match(line)
        out.append(
            PlanCriterion(
                id=m.group(0),
                section=_enclosing_section(sections, m.start()),
                text=line,
                quote=quote_from_line(line),
                checked=box.group(1) in ("x", "X") if box else None,
            )
        )
    return out


def quote_names_criterion(text: str, ac_id: str) -> bool:
    """Does ``text`` name this exact criterion?

    A substring test is wrong here and wrong in a way that looks right:
    ``"AC-2.1" in "AC-2.10 …"`` is ``True``, so filtering notes on ``AC-2.1``
    would silently fold in every note about ``AC-2.10`` through ``AC-2.19``.
    Matching through ``CRITERION_RE`` and comparing whole ids gets the boundary
    from the one pattern that already defines what a criterion label is, rather
    than from a second, hand-rolled idea of where one ends.

    Reads a **quote**, not a document, so it is deliberately not fence-aware:
    an excerpt has no fences of its own.
    """
    wanted = ac_id.strip().upper()
    return any(m.group(0).upper() == wanted for m in CRITERION_RE.finditer(text))


def named_criterion(text: str) -> str:
    """The first ``AC-<m>.<n>`` ``text`` names, or ``""``.

    The other direction of :func:`quote_names_criterion` — that one is handed an
    id and asks whether a string names it, this one is handed a string and asks
    which id it names — and both live here for the same reason: where a
    criterion label starts and ends is grammar, and ``CRITERION_RE`` is the one
    expression that says so.

    **First, not every.** A note's prose routinely names a second criterion in
    passing, and its subject is what it opens with; a caller wanting a set can
    have ``CRITERION_RE`` itself.

    **Case is not folded, and that is the same answer**
    :func:`quote_names_criterion` **gives about its haystack.** ``CRITERION_RE``
    is upper-case only, so a lower-case ``ac-2.3`` is not a criterion label
    anywhere in this module — :func:`parse_criteria` does not report one either,
    and a note anchored to a label the document cannot contain would find
    nothing to anchor to. The leniency ``--ac ac-2.3`` enjoys is a NEEDLE being
    normalised, which is a different act.

    Fence-aware, and that is where it parts company with
    :func:`quote_names_criterion`. That one reads a QUOTE — an excerpt, which
    has no fences of its own — while this reads a note's whole comment, which is
    Markdown a writer may well have put a code sample in. A criterion id inside
    one is prose about a criterion, exactly as it is everywhere else in this
    module.
    """
    fences = _fence_spans(text)
    for m in CRITERION_RE.finditer(text):
        if _outside_fences(m.start(), fences):
            return m.group(0)
    return ""


def extract_shorthand(body: str) -> tuple[list[Shorthand], str]:
    """Pull every ``{@agent: instruction}`` out of the document's HTML comments.

    Returns the shorthands found **and the body with them removed**, because
    §2.2 makes the removal part of the rule: each match becomes a note ``to=``
    that agent, with the enclosing section's id, and is deleted from the saved
    body. One-way; there is nothing to be idempotent about, and a second save
    finds none.

    A match is only a shorthand **inside an HTML comment** — the same text in
    ordinary prose is left alone, and so is a code sample of the syntax.

    **This is the only case in which the system alters prose the writer sent,
    and it only ever deletes.** A comment left holding nothing but whitespace
    goes with it, and so does the line if the comment was all it held —
    otherwise every extraction leaves ``<!--  -->`` litter in a document whose
    whole promise is that it is plain Markdown.

    One live cost, recorded rather than fixed (§2.2): this rule reads inside
    comments, so a source citation that ever contains a brace-at-name shape —
    ``<!-- src: Provi portal {@backend: re-check this} -->`` — loses its tail and
    files a note. Carving ``src:`` out is what would make an inert comment stop
    being inert, so the collision stands.
    """
    fences = _fence_spans(body)
    sections = parse_sections(body)
    found: list[Shorthand] = []
    edits: list[tuple[int, int, str]] = []  # (start, end, replacement)

    for comment in COMMENT_RE.finditer(body):
        if not _outside_fences(comment.start(), fences):
            continue
        inner = comment.group(1)
        matches = list(SHORTHAND_RE.finditer(inner))
        if not matches:
            continue

        line_start, line_end = _line_bounds(body, comment.start())
        line = body[line_start:line_end]
        box = BOX_RE.match(line)
        checked = box.group(1) in ("x", "X") if box else None
        section = _enclosing_section(sections, comment.start())
        # THE LINE MINUS THE COMMENT, which is exactly what survives this
        # function — the removal below deletes the comment and nothing else, so
        # `- [ ] Wire the ingest endpoint <!-- {@backend: …} -->` is still
        # `- [ ] Wire the ingest endpoint` in the document the note is read
        # against. Sliced rather than run through `strip_comments` because a
        # MULTI-LINE comment's opener is all this line holds of it: the stripper
        # would leave the dangling `<!--` in the quote, and an anchor carrying
        # one provably cannot be found.
        after = body[comment.end():line_end] if comment.end() <= line_end else ""
        quote = quote_from_line(body[line_start:comment.start()] + after)
        for m in matches:
            found.append(
                Shorthand(
                    section=section,
                    owner=m.group(1),
                    text=m.group(2).strip(),
                    checked=checked,
                    quote=quote,
                )
            )

        kept, cursor = [], 0
        for m in matches:
            kept.append(inner[cursor : m.start()])
            cursor = m.end()
        kept.append(inner[cursor:])
        remainder = "".join(kept)
        if remainder.strip():
            edits.append((comment.start(), comment.end(), f"<!--{remainder}-->"))
        else:
            edits.append((comment.start(), comment.end(), ""))

    out = body
    for start, end, replacement in reversed(edits):
        if replacement:
            out = out[:start] + replacement + out[end:]
            continue
        # The comment held nothing but shorthand, so it goes — and with it the
        # horizontal whitespace that only existed to separate it from the prose,
        # and the whole line if that is all the line held. Otherwise every
        # extraction leaves either `<!--  -->` or a trailing space behind.
        ws = start
        while ws > 0 and out[ws - 1] in " \t":
            ws -= 1
        out = out[:ws] + out[end:]
        line_start, line_end = _line_bounds(out, ws)
        if not out[line_start:line_end].strip():
            out = out[:line_start] + out[min(line_end + 1, len(out)) :]
    return found, out


def strip_comments(text: str) -> str:
    """Fold out every ``<!-- … -->`` run, leaving everything else untouched.

    **Not** a rewrite of the body — the caller decides what to do with the
    result. Used by ``spec_digest`` (where the fold is what makes a comment
    inert) and by ``render_clean`` (where it is the reading view). Comments
    inside fenced code are prose about comments and survive.
    """
    fences = _fence_spans(text)
    out, cursor = [], 0
    for m in COMMENT_RE.finditer(text):
        if not _outside_fences(m.start(), fences):
            continue
        out.append(text[cursor : m.start()])
        cursor = m.end()
    out.append(text[cursor:])
    return "".join(out)


def render_clean(body: str) -> str:
    """The reading view: comments folded out and the gaps they leave closed up.

    There is **no** ``scope`` parameter and **no** carve-out for ``src:``
    citations (§5.2, N1) — a carve-out is what would make an inert comment stop
    being inert. Shorthand needs no separate handling: it lives inside comments
    and goes with them.

    This is a display transform. It is never written back — the bytes on disk
    keep every comment the writer put there.
    """
    text = strip_comments(body)
    lines = [ln.rstrip() for ln in text.split("\n")]
    out: list[str] = []
    for line in lines:
        if not line and out and not out[-1]:
            continue
        out.append(line)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n" if out else ""


def normalize_spec_text(text: str) -> str:
    """The three normalizations :func:`spec_digest` applies, on any text.

    Split out of :func:`_digest_material` so that *"does this move the spec?"*
    has ONE implementation, askable of a whole document or of a single section.
    The digest asks it of the document; the spec lock's note-filing asks it of
    each section it is about to put in front of the user.

    That second caller is the reason this is a function rather than four lines
    inside the digest. The lock's *test* is normalized and document-wide, but the
    rows it files notes from come from :func:`split_by_section`, which is a raw
    byte comparison — so a write that legitimately moved the digest in one
    section also filed a ``to=user`` note for every other section whose bytes
    changed, including ones normalization folds away entirely. The user was asked
    to accept a diff containing nothing but ``- [ ]`` becoming ``- [x]``, which is
    the exact edit rule 2 below exists to declare inert.

    1. **HTML comments** fold out.
    2. **Task-list marker state** folds to one form.
    3. **Whitespace** collapses: trailing spaces go, runs of spaces and tabs
       become one, blank lines drop out.

    Never applied to the bytes on disk — this is a comparison key, not a
    rewriter.
    """
    text = strip_comments(text)
    text = _MARKER_RE.sub(r"\1 \2", text)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def parse_coverage(body: str) -> dict[str, tuple[str, ...]]:
    """``{slug: (ac_id, …)}`` — which criteria each section CLAIMS to advance.

    Reads ``<!-- advances: AC-2.1, AC-2.3 -->`` off any section, at any depth,
    fence-aware. The ids inside are found with ``CRITERION_RE``, so *"what is a
    criterion label"* keeps the one answer it has had since criteria were first
    parsed — a second, hand-rolled idea of where a label ends is exactly the
    ``AC-2.1`` / ``AC-2.10`` boundary bug :func:`quote_names_criterion` exists
    to name.

    **Occurrences, not validation**, matching :func:`parse_criteria`: a claim on
    an id that does not exist is a row, not an error. Sections with no marker do
    not appear. Only sections with at least one claim are keys.
    """
    fences = _fence_spans(body)
    sections = parse_sections(body)
    out: dict[str, list[str]] = {}
    for m in ADVANCES_RE.finditer(body):
        if not _outside_fences(m.start(), fences):
            continue
        slug = _enclosing_section(sections, m.start())
        seen = out.setdefault(slug, [])
        for ac in CRITERION_RE.finditer(m.group(1)):
            if ac.group(0) not in seen:
                seen.append(ac.group(0))
    return {slug: tuple(ids) for slug, ids in out.items() if ids}


def unclaimed_criteria(body: str) -> frozenset[str]:
    """Criteria DEFINED in the spec layer that no section claims to advance.

    The input to the coverage check that keeps the detail layer honest: if
    milestones are freely editable, a keeper can quietly drop the work behind a
    criterion and the user only finds out at the end.

    **Definition is the first occurrence in document order**
    (:func:`parse_criteria`'s rule), and only a definition inside a ``SPEC``
    section counts — a criterion the keeper is free to rewrite is not one the
    keeper can be held to.

    Note what this returns on a plan that carries no ``advances:`` markers at
    all: *every* criterion, both before and after any write. That is the whole
    migration story — :func:`~clawmeets.models.project_plan._coverage_regressed`
    compares two of these sets, so an unmarked plan can never regress.
    """
    layers = section_layers(body)
    defined: list[str] = []
    for c in parse_criteria(body):
        if c.id not in defined and layers.get(c.section, SPEC) == SPEC:
            defined.append(c.id)
    claimed = {ac for ids in parse_coverage(body).values() for ac in ids}
    return frozenset(ac for ac in defined if ac not in claimed)


def _top_level(sections: list[PlanSection]) -> list[PlanSection]:
    """The document's top-level sections — ``depth == 2``, with the fallback.

    Shared by :func:`_digest_material` and :func:`_layer_manifest` so the two
    cannot disagree about what the skeleton IS. ``depth == 2`` is the rule
    (§3.2) because every document this system writes puts its title at ``#`` and
    its sections at ``##``; when a document has no ``##`` at all the rule would
    select nothing and every such document would share a digest, so it falls
    back to the shallowest depth present. The ``_lede`` is top-level too and
    leads when there is one.
    """
    headed = [s for s in sections if s.id != LEDE_ID]
    chosen = [s for s in headed if s.depth == 2]
    if not chosen and headed:
        shallowest = min(s.depth for s in headed)
        chosen = [s for s in headed if s.depth == shallowest]
    return [s for s in sections if s.id == LEDE_ID] + chosen


def _layer_manifest(body: str) -> str:
    """The document's top-level skeleton as a comparison key: one
    ``<slug>:<layer>`` per section, in document order, newline-joined.

    **THIS IS WHAT MAKES THE MARKER IMMUTABLE, and without it the whole design
    is a hole.** HTML comments fold out of the digest by design
    (:func:`normalize_spec_text`), so a keeper could flip ``spec`` to ``detail``
    in one write that looks inert, and unlock the section on the next one.
    Hashing the manifest means the flip MOVES the digest and comes back to the
    user as a proposal — no new enforcement path, the existing lock does it.
    The transition to layered locking is itself a user decision.

    It also means **adding or removing a top-level section, or retitling one,
    needs the user's accept even when the section is detail** — the section list
    is part of what was approved. Deliberate, and the conservative half of a
    choice: the looser rule (manifest covers only the sections that existed at
    acceptance) is more code and a second rule to hold in mind.

    Nested sections are absent on purpose. ``### M2``'s layer is inherited, so
    re-cutting, splitting, merging and renaming milestones inside a detail
    parent moves nothing here — which is exactly the freedom this change is for.
    """
    return "\n".join(f"{s.id}:{s.layer}" for s in _top_level(parse_sections(body)))


def _legacy_digest_material(body: str) -> str:
    """:func:`_digest_material` **as it read before layers existed**, frozen.

    One caller, forever:
    :func:`~clawmeets.models.project_plan.changed_since_acceptance`'s
    compatibility arm. Every plan already accepted carries an
    ``accepted_spec_digest`` computed by this text; left alone, narrowing the
    digest would make *every accepted plan in the system* announce "the plan
    changed since approval" on its next load, having changed nothing.

    **Never grow a feature here, and never tidy it.** Its correctness is
    "identical to a hash computed months ago", so a refactor that reads better
    and hashes differently is a silent regression across every live project.
    """
    sections = parse_sections(body)
    headed = [s for s in sections if s.id != LEDE_ID]
    chosen = [s for s in headed if s.depth == 2]
    if not chosen and headed:
        shallowest = min(s.depth for s in headed)
        chosen = [s for s in headed if s.depth == shallowest]
    chosen = [s for s in sections if s.id == LEDE_ID] + chosen

    text = "\n".join(body[s.char_start : s.char_end] for s in chosen) if chosen else body
    return normalize_spec_text(text)


def legacy_spec_digest(body: str) -> str:
    """:func:`spec_digest` under the pre-layers formula. See
    :func:`_legacy_digest_material` — read-time compatibility only."""
    return hashlib.sha256(_legacy_digest_material(body).encode("utf-8")).hexdigest()


def _digest_material(body: str) -> str:
    """The exact text ``spec_digest`` hashes. Split out so tests can read it.

    Two parts, and the first is the smaller but load-bearing one.

    **The layer manifest** (:func:`_layer_manifest`) leads: the top-level
    skeleton and each section's layer. It is here so that changing a section's
    layer, or the set of top-level sections, moves the digest — a marker that
    was not hashed could be flipped in one inert-looking write, and the section
    would be unlocked on the next.

    **Then the SPEC-layer top-level sections**, in document order, so a nested
    milestone counts exactly once — its text is already inside its parent's
    extent. A ``DETAIL`` section contributes no text at all: that omission IS
    the narrowing, and it is the whole of the enforcement change. **The lock did
    not move; what it hashes did.** ``_top_level`` owns the depth rule and its
    no-``##`` fallback.

    Three normalizations, applied before hashing and **never to the bytes on
    disk**:

    1. **HTML comments** fold out. This is what makes the word *inert* true of
       the system and not only of the intent: without it, correcting a typo
       inside ``<!-- src: Provi portal -->`` moves the digest, raises "the spec
       changed since acceptance", and spends a coordinator turn on a comment the
       user was told does not affect anything.
    2. **Task-list marker state** folds to one form, so ticking a box on a
       default-shaped plan is not a spec change — which is what lets one
       ``## Milestones`` section carry both the definition and the progress.
    3. **Whitespace** collapses: trailing spaces go, runs of spaces and tabs
       become one, blank lines drop out. Reflowing a table or reindenting a
       nested list is not a change to what the plan says, and folding a comment
       out of the middle of a line leaves whitespace behind that nobody typed.
    """
    sections = parse_sections(body)
    chosen = _top_level(sections)
    manifest = "\n".join(f"{s.id}:{s.layer}" for s in chosen)
    spec = [s for s in chosen if s.layer == SPEC]

    text = "\n".join(body[s.char_start : s.char_end] for s in spec) if chosen else body
    return manifest + "\n" + normalize_spec_text(text)


def spec_digest(body: str) -> str:
    """Hash of what the plan *says*, normalized so that inert edits do not move it.

    Read by ``changed_since_acceptance`` (§7.4). It is **not** what the execution
    gate reads — the gate reads ``open_notes_for_you`` — so a moved digest is two
    hops from stopping work, and both normalizations above exist to keep the
    first hop honest.
    """
    return hashlib.sha256(_digest_material(body).encode("utf-8")).hexdigest()


def body_sha(body: str) -> str:
    """Hash of the whole file. **Identity only** — display and staleness.

    It is a precondition on nothing. The section is the unit of precondition
    (§3.2): a writer's ``base`` is the section text it read, which it already
    has, so no writer ever needs a hash to write.
    """
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def render_diff(base: str, text: str, *, label: str = "") -> str:
    """A unified diff of two section texts, **for display**.

    The only ``difflib`` call site in the repository. Generated per request,
    never stored, never an input: no surface anywhere accepts a diff (AC-4.4),
    because the unit of change is a whole-section replacement located by heading
    (§2.3).
    """
    name = label or "section"
    return "\n".join(
        difflib.unified_diff(
            base.splitlines(),
            text.splitlines(),
            fromfile=f"{name} (base)",
            tofile=f"{name} (proposed)",
            lineterm="",
        )
    )


def _is_anchorable(line: str) -> bool:
    """Can a reader point at this source line? **Non-blank AND not comment-only.**

    The predicate behind :func:`first_changed_line`'s choice of anchor, named
    rather than inlined because it is asked in three places there and a rule with
    three copies has three answers.

    ``strip_comments`` is deliberately the same function
    :func:`normalize_spec_text` uses to declare a comment inert to the spec
    digest — the two are the same claim about the same bytes, said to two
    different readers: a comment changes neither what the plan *says* nor what a
    reader can *see*.

    An unterminated ``<!--`` is left alone and reads as anchorable. Judging it
    would mean carrying comment state across lines, and the answer this function
    exists to give is about one line at a time; a stray opener is rare, and the
    cost of getting it wrong is one unanchored note rather than a wrong anchor.
    """
    return bool(strip_comments(line).strip())


def first_changed_line(base: str, text: str) -> str:
    """The line of ``base`` that ``text`` first changes — a proposal's anchor.

    Lives here, beside :func:`render_diff`, because this is the second
    ``difflib`` caller and the module docstring's claim that the import appears
    nowhere else in the repository is worth keeping true. Same discipline, same
    reason: a comparison is a **display artifact** and neither of these results
    is ever stored as structure. This one is stored as a *quote*, which is text
    cut from the base and nothing more.

    Three shapes, and the ``insert`` rule is the one that is not obvious:

    * ``replace`` / ``delete`` — the first ANCHORABLE base line in the range.
    * ``insert`` — the last ANCHORABLE base line **before** it. An insertion has
      no base line of its own, and the line it follows is what a reader is
      looking at when they read the proposal. An insertion at the very top
      falls back to the base's first anchorable line.
    * no change, or a base with no anchorable line at all (a section create) —
      ``""``. That is honest: there is nothing on screen yet to anchor to.

    **A BLANK-ONLY CHANGE RANGE IS AN INSERTION WEARING A ``replace`` TAG**, and
    treating it as neither is what put a proposal in the page-bottom pane. A
    section's captured base runs to the next heading, so it ends on a blank line;
    a proposal that only APPENDS — every *"add a new acceptance criterion"* note
    — lines up against that trailing blank, and ``SequenceMatcher`` absorbs it
    into the change rather than reporting a clean ``insert``. The range then held
    no non-blank line, the loop fell through, and the function returned ``""``,
    which :func:`_derive_quote` stores and ``PlanEditor``'s
    ``!!n.quote && shownIds.has(n.section)`` reads as *"not anchorable, send it
    to the bottom of the page"*. Measured on ``plan-block-comment-simplify``
    note ``n-7a43c9``, whose single opcode was ``replace base[15:16] == ['']``.

    So the blank range no longer ENDS the search — it is remembered and the walk
    continues, because a later opcode naming a real line is still the better
    anchor. Only the terminal answer changes: where there is no such line, the
    insertion is anchored the way :func:`insert` already anchors one.

    **ANCHORABLE IS NARROWER THAN NON-BLANK, and it has to be — on EVERY arm,
    which is a repair rather than a restatement.** The rule below was written
    for the insertion arm and applied only there; the ``replace``/``delete`` arm
    kept a plain ``ln.strip()`` and could therefore hand back the very line the
    rule exists to refuse. It is not a corner: a refused rewrite of a criterion
    diffs as a ``replace`` over a range that opens on that criterion's
    ``<!-- evidence: … -->`` line, so the arm most likely to see a comment-only
    line was the arm not applying the rule. Measured on
    ``clawmeets-todo-tickets`` note ``n-8d35c3``, whose derived quote was the
    evidence comment verbatim.

    The rule itself: the quote this produces is searched for in the *rendered*
    DOM (``planAnchor.locate`` over
    ``PlanBody``'s flattened blocks), and ``PlanBody`` renders through the
    ``rehypeHideComments`` plugin, which drops a raw node that is nothing but an
    HTML comment before it can reach the document. A line that is nothing but an
    ``<!-- … -->`` comment therefore renders to nothing, and quoting one produces
    an anchor that provably cannot be found — the note lands in its section's
    loose pane reading *"Couldn't find this line"*, which is a different wrong
    answer rather than a fix. In the plan format this system writes, every
    criterion ends on an ``<!-- evidence: … -->`` line, so that is precisely the
    line an append follows. ``strip_comments`` is the same predicate
    :func:`normalize_spec_text` already uses to call a comment inert.

    That plugin is why this paragraph now names a plugin rather than an absence.
    The rule was originally justified by ``PlanBody`` mounting ``ReactMarkdown``
    with no ``rehype-raw``, on the reasoning that raw HTML then reached the
    document not at all — true of react-markdown 8, which spliced an unhandled
    raw node out of the tree, and false from 9, which converts it to TEXT. The
    comment lines duly appeared on screen and this rule went from principled to
    accidentally-still-correct. The plugin restores the premise deliberately, so
    the two can no longer drift apart unnoticed:
    ``rehypeHideComments.test.tsx`` asserts it.

    Whatever comes back is still a line of ``base`` **verbatim** — this walks
    opcodes to LOCATE, never to build.
    """
    base_lines = base.splitlines()
    anchorable = [ln for ln in base_lines if _is_anchorable(ln)]
    if not anchorable:
        return ""

    def preceding(i1: int) -> str:
        """The line an insertion at ``i1`` follows, or the base's first."""
        before = [ln for ln in base_lines[:i1] if _is_anchorable(ln)]
        return before[-1] if before else anchorable[0]

    # Where a blank-only change range was seen, if one was. Not returned from
    # inside the loop: a real changed line further down is the better anchor and
    # is still reachable, which is what "keep looking" was always for.
    inserted_at: int | None = None

    matcher = difflib.SequenceMatcher(None, base_lines, text.splitlines())
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            return preceding(i1)
        changed = [ln for ln in base_lines[i1:i2] if _is_anchorable(ln)]
        if changed:
            return changed[0]
        # A run of blank or comment-only lines was rewritten. Not something a
        # reader can point at, so keep looking rather than anchoring to
        # whitespace — but do not forget where it was.
        if inserted_at is None:
            inserted_at = i1
    return preceding(inserted_at) if inserted_at is not None else ""
