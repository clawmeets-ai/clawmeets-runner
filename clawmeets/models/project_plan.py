# SPDX-License-Identifier: MIT
"""
clawmeets/models/project_plan.py

The plan sidecar and the whole of the concurrency mechanism (§3, §4).

``PLAN.md`` is the document; ``plan.json`` beside it is everything the document
is not allowed to hold — notes, your staged tray, review rounds, the revision
label and the acceptance record. **The sidecar never contains the body.**

Three properties this module exists to guarantee:

1. **One funnel.** Every byte that reaches PLAN.md goes through
   :func:`apply_edits`. The upload route, the plan routes and the tab are three
   callers of one function, and none of them is a whole-document writer
   (AC-2.8).
2. **A change lands iff its section still says what the writer read.** No merge,
   no base reconstruction, no conflict marker, no retry, and no state in which
   the stored document is text nobody wrote (§4).
3. **A refused write is a note, never a dead end** (§4.3).

**Text surgery is delegated in full to** :mod:`clawmeets.models.plan_markdown`.
Nothing here parses, slices or reasons about headings; if a text operation is
missing, it is a gap in that module, not something to restate here.

**Dependencies are injected, never constructed.** Reads take a
:class:`~clawmeets.models.context.ModelContext`; writes additionally take a
:class:`~clawmeets.sync.runloop.ChangelogRunloop`. The plan layer therefore
never reaches for a server context, and a test drives it with the same runloop
the server uses.

Storage::

    {data_dir}/metadata/projects/<name>-<id>/plan.json

Created **lazily** — first note, row, review or write. A project that predates
this feature has no sidecar, and reading its plan still works, because the body
IS the file (§3.1, AC-2.7).

Lives in ``models/`` for the same reason ``plan_markdown`` does: the
``models/`` subtree is rsync'd wholesale into the runner wheel by
``scripts/build-runner-package.sh``, so a new module here needs no manifest row.
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import re
import secrets
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from clawmeets.models.agent import Agent
from clawmeets.models.chatroom import Chatroom
from clawmeets.models.plan_markdown import (
    PlanLimitError,
    body_sha,
    drops_heading,
    extent_key,
    extract_shorthand,
    first_changed_line,
    heading_line,
    normalize_spec_text,
    parse_sections,
    quote_from_line,
    relevels_heading,
    rename_pair,
    render_diff,
    replace_section,
    section_extent,
    spec_digest,
    split_by_section,
)
from clawmeets.sync.changelog import (
    ChangelogEntryType,
    FilePayload,
    MessagePayload,
    ParticipantAddedPayload,
    ProjectPlanStatePayload,
    RoomCreatedParticipant,
    RoomCreatedPayload,
)
from clawmeets.sync.runloop import BatchEntrySpec
from clawmeets.utils.file_io import FileUtil

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from clawmeets.models.context import ModelContext
    from clawmeets.models.project import Project
    from clawmeets.sync.runloop import ChangelogRunloop


#: One lock for the whole plan layer. Every public coroutine holds it, and the
#: private ``_*_locked`` helpers below assume it is already held so that
#: :func:`submit_review` can compose them into one transaction without
#: re-entering a non-reentrant lock. Precedent: ``models/project_report.py:30``,
#: ``models/desk_sop.py:41``, ``models/desk_todo.py:38``, ``models/brief_tab.py:35``.
_lock = asyncio.Lock()

#: The plan lives in one room, under one name, on every project that has one.
PLAN_ROOM = "shared-context"
PLAN_FILE = "PLAN.md"

#: §5.7's two rooms. A batch whose every addressee is the coordinator or the
#: owner posts where that conversation already lives and **creates nothing** —
#: you are reviewing a document the coordinator keeps, so your review is the
#: reply. Anything else needs a room a specialist can be reached in, and that
#: room is :data:`PLAN_ROOM` — the project's ONE consultation room, which holds
#: ``PLAN.md`` as well as the conversation about it.
#:
#: There used to be a second name here, ``REVIEW_ROOM = "plan-review"``, opened
#: on demand beside ``shared-context``. It bought no isolation: every agent
#: seeded into it was force-added to ``shared-context`` in the same call, so a
#: reviewer could read the document it was being asked about. What it did buy
#: was a room that did not exist when the coordinator was first told to use it,
#: and ~25 lines in :func:`~clawmeets.models.agent.build_state_snapshot`
#: seeding a frozen snapshot to paper over that. ``shared-context`` is in
#: ``project.chatrooms`` from project creation, so the merge deletes the seed
#: and the failure mode together.
USER_ROOM = "user-communication"

#: The owner's identity in this module. ``PlanNote.to`` is documented as
#: *"agent name, or 'user'"* (§3.1), so the plan layer's identity vocabulary is
#: **names**, with the literal ``"user"`` standing for the project owner. Routes
#: map a user JWT to this token; agents pass their own name. Collapsing the two
#: namespaces is the specification's own choice, inherited here rather than
#: re-decided.
OWNER = "user"

# ---------------------------------------------------------------------------
# Limits (§3.4). Exceeding one is ``PlanLimitError`` -> 413, carrying the limit
# and the actual value, which is the shape ``plan_markdown`` already raises.
# ---------------------------------------------------------------------------

MAX_BODY_BYTES = 512 * 1024
MAX_OPEN_NOTES = 200
MAX_DRAFT_ENTRIES = 100
MAX_ROUNDS = 50
MAX_HISTORY = 200
MAX_NOTE_CHARS = 32 * 1024
MAX_EDIT_CHARS = 32 * 1024

#: A note's ``quote`` — the excerpt it is anchored to. **Not a second number**:
#: it is the browser client's ``PLAN_QUOTE_CHARS``, read from
#: ``clawmeets/web/frontend/src/components/project/plan/planAnchor.tsx:52``,
#: so the two surfaces refuse at the same place rather than at two.
#:
#: The client counts UTF-16 code units and Python counts code points, so a
#: quote the client capped is **always** at or under this here too — an astral
#: character is 2 units there and 1 code point here. The server bound therefore
#: never fires on client-shaped input; it fires on input that skipped the
#: client, which is the only input that could ever breach it.
MAX_QUOTE_CHARS = 4000

#: How recently an identical note must have been filed for a repeat to count as
#: a **double submit** rather than a deliberate second note.
#:
#: Short on purpose, and the asymmetry sets the ceiling. A duplicate that gets
#: filed is ugly but recoverable — notes are never deleted, so it sits there
#: forever, but the user can see both and dismiss one. A note that gets
#: **swallowed** cannot be recovered at all: the author believes it was filed
#: and nothing on any surface says otherwise. Eating a note somebody meant is
#: the worse failure, so the window is sized to the smaller of the two risks.
#: Anything at or above a minute starts eating real second notes; ten seconds is
#: unambiguously inside one human gesture and outside *"I meant it"*.
#:
#: There is no time-windowed dedupe in this layer to copy — the house pattern is
#: content-hash dedupe with **no** window (:func:`send_digest`), which works
#: there because re-sending an unchanged note is a no-op and re-filing one is
#: not.
NOTE_DEDUPE_SECONDS = 10

NOTE_STATUSES = ("open", "answered", "applied", "rejected", "dismissed")

#: **DERIVED, NEVER STORED.** A note has no ``kind`` field: it is a comment,
#: optionally carrying a diff (``proposal``), optionally carrying a parent
#: (``reply_to``), optionally carrying nothing but a diff. What a surface
#: PRINTS is computed from those fields on every read by :func:`note_kind`, so
#: the word on the chip can never disagree with the note it labels.
#:
#: The four words, in the precedence :func:`note_kind` applies them:
#:
#: * ``deviation`` — filed by the keeper, to the user, after the plan was FIRST
#:   accepted. The only one that needs a fact from outside the note, and the
#:   fact is a timestamp comparison (:func:`note_is_deviation`).
#: * ``proposal`` — carries replacement text, or carried it before it was
#:   applied, or proposes a delete.
#: * ``reply`` — names a parent. This is the act, and it replaces the stored
#:   ``"question"``, which named the MECHANISM (``submit_review``'s ``ask``
#:   arm) and therefore missed an agent's ``plan note --reply-to`` entirely.
#: * ``note`` — none of the above. It has no content of its own and never did;
#:   it was stored as ``"comment"``.
NOTE_KINDS = ("note", "proposal", "reply", "deviation")
DRAFT_KINDS = ("edit", "accept", "reject", "ask", "dismiss")
RESOLVE_ACTIONS = ("apply", "reject", "answered", "dismiss")

#: Written by the server, read by the agent whose proposal was dropped and never
#: by the user (§2.6). A superseded proposal resolves ``rejected`` — never
#: ``applied``, because its text did not reach the document.
SUPERSEDED_REASON = "superseded by @{by}'s proposal on the same section"

#: The gate section, and the three strings that live in it.
#:
#: ``## Approval`` is not decoration and it is not a status field the coordinator
#: maintains. It is the **target of the go-note** — the bootstrap proposal the
#: server files at project creation, whose whole content is *replace this
#: section's body with "User approves the plan."*. Accepting that proposal is
#: how a plan becomes accepted; there is no second act and no approve route.
#:
#: **A proposal is located by heading**, so the heading is load-bearing in a way
#: no other seed section's is: a coordinator that rewrites the document without
#: it leaves the go-note unappliable and its own project permanently gated.
#: :func:`_prepare_locked` refuses that write while a go-note is open, which is
#: what keeps the guarantee server-enforced rather than prompt-enforced.
APPROVAL_SECTION = "approval"
APPROVAL_HEADING = "## Approval"
APPROVAL_PENDING = "_Not yet approved._"
APPROVAL_GRANTED = "User approves the plan."

#: The go-note's comment. Addressed to the user, so it says what accepting DOES
#: rather than naming the mechanism: the user is not reading about proposals,
#: they are deciding whether work may start.
GO_NOTE_COMMENT = (
    "Work on this project does not start until you accept the plan. "
    "Read it, then Accept this note to approve it — that is what unblocks the "
    "coordinator. Reply here instead if you want something changed first; a "
    "reply leaves this open."
)


def approval_section(text: str) -> str:
    """The ``## Approval`` section's FULL text — heading line included.

    A section edit is the section's whole extent (:func:`replace_section`
    splices over the heading too), so the go-note's proposal and the seed
    template's pending text are built through here rather than by concatenating
    a heading somewhere else and hoping the two agree.
    """
    return f"{APPROVAL_HEADING}\n\n{text}\n"


def approval_state(body: str) -> str:
    """What ``## Approval`` currently SAYS, heading stripped, or ``""``.

    The read half of :func:`approval_section`, and the coordinator's own way of
    checking whether it may start work. The server-side gate is the thing that
    actually refuses ``create_room``; this is what lets the model *know why* —
    a ``NO_OP`` refusal tells it nothing, so a coordinator that cannot see the
    approval state narrates as though a room opened.

    It reads the **document**, not the sidecar, and that is the point: the
    acceptance is a diff the user applied to ``PLAN.md``, so the text they
    approved and the fact of their approval are the same bytes. Nothing has to
    be projected for the coordinator to see it, which is why this works on the
    agent side where ``plan.json`` does not exist.

    ``""`` when the section is absent — a coordinator that rewrote the document
    without the heading. That is refused while a go-note is open
    (:func:`_prepare_locked`), so an empty answer means *not approved* and never
    *approval lost*.
    """
    extent = section_extent(body, APPROVAL_SECTION)
    if extent is None:
        return ""
    text = body[extent[0]:extent[1]]
    # Drop the heading line; the caller asked what the section says, and the
    # heading is what it is called.
    _, _, rest = text.partition("\n")
    return rest.strip()


def _section_prose(body: str, slug: str) -> str:
    """A section's OWN prose: heading dropped, and **stopped at the first
    heading of any depth** that follows.

    Only :func:`_prepare_locked`'s go-note guard uses this, and the truncation
    is the whole reason it is not :func:`approval_state`. Two different things
    shorten the last section's extent without touching a word of its own text,
    and a guard keyed on the raw extent refuses both:

    * a new ``##`` lands after it, ending the extent early — invisible once the
      edges are stripped, so this one is harmless either way;
    * a **create** lands at the end of the document (:func:`_append_section` —
      *"a create lands at the end, where a human can see it and move it"*), and
      a deeper heading like ``### M2`` does not close an ``##`` extent, so it is
      swallowed INTO the last section. ``## Approval`` is last in the seed
      template, so renaming a milestone — a delete plus a create — would read as
      a rewrite of the approval section. Stripping cannot see that; stopping at
      the first heading can.

    What survives the truncation is exactly the text the forgery has to change:
    the sentence directly under the heading, which is what
    :data:`APPROVAL_GRANTED` replaces and what the coordinator is told to read.
    """
    extent = section_extent(body, slug)
    if extent is None:
        return ""
    _, _, rest = body[extent[0]:extent[1]].partition("\n")
    prose: list[str] = []
    for line in rest.split("\n"):
        if line.lstrip().startswith("#"):
            break
        prose.append(line)
    return "\n".join(prose).strip()


#: §5.1's seed template, verbatim. **One template** — the ``full`` variant went
#: with the ``plan_mode`` that selected it. The coordinator may replace every
#: section: this is a starting document, not a schema (§2.4). ``<title>`` is the
#: only substitution.
#:
#: **``## Approval`` is the one section with a rule attached**, and the rule is
#: not "do not edit it" — it is that the *heading* must survive while a go-note
#: is open, because that is what the go-note's proposal is located by.
SEED_TEMPLATE = """# {title}

## Goal

_What is being built and why. Two paragraphs maximum._

## Guardrails

_Constraints, quality bars, scope boundaries._

## Milestones

### M1: <action>

**Deliverable:** `<specific_file.md>`
**Acceptance criteria:**
- **AC-1.1** — <testable statement>

- [ ] **M1** — unassigned

## Approval

_Not yet approved._
"""


# ---------------------------------------------------------------------------
# Errors. One per HTTP status the routes owe (§3.4, §4.1, §6) so that phase 3
# is a mapping table and not a pile of re-derived conditions.
# ---------------------------------------------------------------------------


class PlanNotFoundError(LookupError):
    """No ``PLAN.md`` on this project. Callers surface it as ``404``."""


class PlanForbiddenError(PermissionError):
    """The identity may not do this. Callers surface it as ``403``."""


class PlanSpecLockedError(PlanForbiddenError):
    """M3 — the plan is accepted and this write would move what it **says**.

    **A subclass on purpose.** ``_http``'s five-row table already maps
    :class:`PlanForbiddenError` to ``403``, so this needs no route change and
    no sixth row; a caller that wants the filed note ids catches the narrower
    class.

    **``403`` and not ``409``.** A ``409`` reads *your state is behind, re-read
    and retry* — and there is no retry here. The write is structurally
    forbidden until the user accepts the proposal it just became, which is
    *stop*, which is what ``403`` means. That is also why ``note_ids`` rides in
    the **message text** as well as on the exception: a ``403`` body is a bare
    string (``_conflict_detail`` is :class:`PlanConflictError`'s alone), and an
    agent-facing refusal must carry its own remedy.
    """

    def __init__(
        self,
        message: str,
        *,
        note_ids: Sequence[str] = (),
        revision: int = 0,
    ) -> None:
        super().__init__(message)
        self.note_ids = list(note_ids)
        self.revision = revision


class PlanInputError(ValueError):
    """Malformed request. Callers surface it as ``400``."""


class PlanCorruptError(RuntimeError):
    """The sidecar is on disk and does not validate. **Nothing overwrites it.**

    Deliberately not in the routes' mapping table: it is not one of the five
    statuses §6 owes, it is a ``500``, and a stack in the log beside the project
    id is exactly the loudness this wants. What it buys is the negative — every
    write path in this module reads the sidecar before it writes one, so a plan
    that stops validating stops being written rather than being replaced by a
    fresh default that costs the user every note, every round and the
    acceptance record.
    """


class PlanConflictError(Exception):
    """A precondition failed. Callers surface it as ``409``.

    ``stale`` is populated for a refused write and carries every stale section's
    ``base`` and ``current`` so a client shows your text beside what it says now
    **with no second request and no history lookup** — the server never needs a
    version it is not currently holding (§4.1, AC-2.5).

    ``note_ids`` carries the conflict notes a refused *keeper* write filed
    (§4.3), so the agent can say *"I couldn't land this — it's note n-3c1d for
    you."*
    """

    def __init__(
        self,
        message: str,
        *,
        stale: Sequence["StaleSection"] = (),
        revision: int = 0,
        note_ids: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.stale = list(stale)
        self.revision = revision
        self.note_ids = list(note_ids)


# ---------------------------------------------------------------------------
# The section change (§2.3) — the one unit, in all three of its roles.
# ---------------------------------------------------------------------------


class SectionEdit(BaseModel):
    """Replace ``section`` with ``text``, valid only while it still says ``base``.

    Not a diff, deliberately: a diff carries line positions, so applying one to a
    moved document is guesswork. A section replacement is located by its heading
    and lands correctly at any revision (§2.3).
    """

    section: str
    text: str = ""   # "" deletes the section
    base: str = ""   # "" is the empty base a create carries
    #: **The slug is MEANT not to resolve.** ``base == ""`` cannot carry that
    #: fact, and the collision is a real bug: a brand-new section and a section
    #: that has **vanished** both have no current text, so a proposal filed
    #: against a deleted section read as a create and was appended at the end of
    #: the document — beside the copy it was written to replace. The CLI states
    #: the collision outright (``cli_plan.update``: *""" is a create (§2.3)"*).
    #:
    #: **Three callers set it now, and each knows the fact from a different
    #: place.** ``cli_plan.update`` knows because the owner named a slug that is
    #: not there. ``_absorb_plan`` knows because ``split_by_section`` gave it an
    #: empty ``before``. And the two accept doors — :func:`_row_creates` for the
    #: tray, :func:`resolve_note`'s apply arm for the CLI — know from the NOTE's
    #: ``base_section``, captured when the note was filed, which is the only
    #: value that separates *"this was written to put a missing section back"*
    #: from *"the section was deleted underneath this"*. The row's own ``base``
    #: cannot: it is the section's text as the document reads now, so it is
    #: ``""`` on both.
    #:
    #: Still off by default, so a path that has not thought about it gets the
    #: safe answer: an unresolvable slug is a ``409``.
    create: bool = False


class StaleSection(BaseModel):
    """One row of a refusal (§4.1 step 4).

    ``section``/``base``/``current`` are the three members §6 puts on the wire.
    ``text`` is what the writer wanted and is **not** a wire member — it is here
    because :func:`file_conflict_note` needs the attempted text as the note's
    ``proposal`` while ``base`` becomes its ``base_section`` (§4.3), and the
    refusal is the only place both are known at once.
    """

    section: str
    base: str
    current: str
    text: str = ""


class WriteResult(BaseModel):
    """What a write did — or, when ``ok`` is False, did not do."""

    ok: bool = True
    revision: int = 1
    sections: list[str] = Field(default_factory=list)
    sha: str = ""
    stale: list[StaleSection] = Field(default_factory=list)
    #: M3 — the sections a refused **executing-phase keeper** write would have
    #: moved. Disjoint from ``stale`` and never populated alongside it: staleness
    #: is *"someone else wrote first, re-read and decide"*; this is *"you may not
    #: write this at all until the user accepts it"*. Two refusals, two remedies,
    #: two fields — one field with a flag would put the question inside a body
    #: that has to render one sentence.
    locked: list[StaleSection] = Field(default_factory=list)
    note_ids: list[str] = Field(default_factory=list)
    #: True when the spliced result equalled the stored body — §4.1 step 6, an
    #: idempotent retry. Nothing was appended and no revision moved.
    noop: bool = False


class PlanRevision(BaseModel):
    """One row of ``GET /plan/versions`` — one per write, newest first.

    A changelog read joined against the sidecar's rounds. ``by`` is **who wrote
    the bytes**; ``proposed_by`` is **whose proposals they were**. The two names
    are not interchangeable and a row that conflates them is wrong about the
    specialist whose work it was (§7.1).
    """

    revision: int
    sha: str
    by: str
    at: str
    version: int
    proposed_by: list[str] = Field(default_factory=list)
    note: str = ""


# ---------------------------------------------------------------------------
# The sidecar (§3.1), verbatim, plus the three stamps §6.C/§7.1 add to a round.
# ---------------------------------------------------------------------------


class PlanNote(BaseModel):
    """One note. **Notes are never deleted** — the ladder closes them."""

    id: str
    section: str = ""       # derived slug; ADVISORY. "" = the document as a whole
    to: str = ""            # agent name, or "user"; "" = addressed to nobody
    by: str = ""
    at: str = ""
    status: str = "open"    # open|answered|applied|rejected|dismissed
    #: **THERE IS NO ``kind`` FIELD, AND ITS ABSENCE IS THE POINT.** A note is
    #: a comment: with or without a diff (``proposal``), with or without a
    #: parent (``reply_to``), with or without text (``comment``). Every word a
    #: surface ever printed for a note was a function of those fields plus one
    #: timestamp comparison, so it is computed on read by :func:`note_kind`
    #: and stored nowhere. See :data:`NOTE_KINDS`.
    #:
    #: Historical ``plan.json`` files carry a stored ``kind``; pydantic ignores
    #: the unknown key on load and it is dropped on the next save. Nothing is
    #: lost — :func:`note_kind` reproduces every one of those stored values
    #: from the note's own fields.
    comment: str = ""
    proposal: str = ""      # SectionEdit.text
    base_section: str = ""  # SectionEdit.base
    #: What ``proposal`` held at the moment it was applied. Written by BOTH
    #: apply paths immediately BEFORE they blank ``proposal``, so an applied
    #: note can still render the diff the owner accepted.
    #:
    #: **Why not simply stop blanking ``proposal``.** The blanking is what marks
    #: an applied note re-sendable (:func:`send_digest` hashes ``proposal``), and
    #: retaining it would make ``has_proposal`` true on an applied note and
    #: re-offer Accept for a change that already landed. Splitting the field
    #: leaves every existing predicate exactly as it is.
    #:
    #: Never an input, never mutated after the apply, not part of
    #: :func:`send_digest`, and excluded from the wire on ``PlanNoteView``.
    applied_text: str = ""
    quote: str = ""
    reply_to: str = ""
    round: str = ""
    conflict: bool = False
    resolved_by: str = ""
    resolved_at: str = ""
    resolution: str = ""
    #: The plan's ``revision`` when this note was written. §5.7's warning reads
    #: *"revision 9 → 14"*, and the 9 has no other source: the note records what
    #: its author saw (``base_section``) but not when they saw it. Defaults 0, so
    #: a note filed before this field existed renders the warning without the
    #: numbers rather than with wrong ones.
    revision: int = 0
    #: **The go-note** — the one note the SERVER files, at project creation, and
    #: the only note whose resolution stamps acceptance.
    #:
    #: It is an ordinary ``proposal`` in every other respect: it counts toward
    #: :func:`open_notes_for_you`, it blocks ``create_room`` through the one
    #: gate every other open note blocks through, it renders in the tray like
    #: any other, and the user Accepts it with the control they already know.
    #: This flag buys exactly three exceptions, each written down where it
    #: applies:
    #:
    #: 1. a user **reply** does not close it (:func:`submit_review`'s ``ask``
    #:    arm) — every other note closes ``answered`` on a reply, and on this
    #:    one that would mean typing a question released the gate;
    #: 2. applying it stamps ``accepted_*`` (:func:`_stamp_acceptance_locked`);
    #: 3. while one is open, a write that drops ``## Approval`` is refused
    #:    (:func:`_prepare_locked`).
    #:
    #: Never an input on any route — set by :func:`seed_go_note` alone, at
    #: project creation, and by nothing a caller can reach. There is exactly one
    #: go-note in a plan's life: acceptance is one-way.
    bootstrap: bool = False

    #: **This note proposes REMOVING its section**, and it exists because
    #: ``proposal == ""`` cannot say that on its own.
    #:
    #: An empty proposal already means *"this note has nothing to apply"* — a
    #: plain comment, a question, an already-applied note whose text moved to
    #: ``applied_text``. So a refused DELETE, which is a real proposal whose
    #: replacement text happens to be empty, was indistinguishable from all
    #: three: :func:`_note_view` reported ``has_proposal: false``, the desk
    #: withheld ``Accept`` and ``Reject``, and the user was left holding a note
    #: that could only be dismissed. The gate counts open notes to the user, so
    #: a project could stop on a decision with no control that made it.
    #:
    #: The flag is what makes the delete APPLICABLE rather than merely legible:
    #: ``has_proposal`` becomes ``bool(proposal) or proposes_delete``, the diff
    #: renders the section against nothing, and ``Accept`` stages the row the
    #: tray already builds — ``text=""``, ``base=<current>`` — which
    #: :func:`replace_section` applies as a deletion.
    #:
    #: Set by :func:`_file_spec_lock_locked` alone, on a row whose ``base`` is a
    #: real section and whose ``text`` is empty. **Never an input on any route**,
    #: for the reason ``bootstrap`` is not: a caller that could set it could turn
    #: any note into a one-click section delete.
    proposes_delete: bool = False


class DraftEntry(BaseModel):
    """One row of the "ready to send" tray."""

    id: str
    kind: str               # edit|accept|reject|ask|dismiss
    section: str = ""       # for edit/accept
    text: str = ""          # SectionEdit.text
    base: str = ""          # SectionEdit.base — captured when the row was staged
    note_id: str = ""       # for accept|reject|ask|dismiss
    comment: str = ""       # for ask (the question) and reject (the reason)
    to: str = ""            # derived: the note's author, or the keeper
    at: str = ""


class PlanReviewDraft(BaseModel):
    """At most one open draft per (plan, user). Server-side, so two tabs and two
    devices share one tray and a submit from either empties both."""

    by: str = OWNER
    entries: list[DraftEntry] = Field(default_factory=list)
    batch_comment: str = ""
    room: str = ""
    updated_at: str = ""


class PlanSend(BaseModel):
    """One note, sent once. Re-sent iff ``digest`` changed."""

    note_id: str
    to: str
    digest: str
    sent_at: str
    room: str
    message_id: str | None = None


class PlanReviewRound(BaseModel):
    """One review batch. ``sends`` is keyed by note id and is additive-only."""

    round_id: str
    opened_at: str = ""
    opened_by: str = ""
    room: str = ""
    #: Did this batch move the changelog on the room's account? Historically
    #: *"the room was created"*; since ``shared-context`` absorbed
    #: ``plan-review`` it also covers *"membership was appended"*, because the
    #: consultation room now always exists and only its roster can move. Kept
    #: under the old name so rounds recorded before the merge still read.
    room_created: bool = False
    addressees: list[str] = Field(default_factory=list)
    sends: dict[str, PlanSend] = Field(default_factory=dict)
    revision: int = 0
    closed_at: str = ""

    #: N5 (§7.1) — distinct authors of the proposals this round applied.
    #: Stamped inside :func:`submit_review`'s transaction. Defaults empty, so a
    #: round recorded before this field existed reads as "no proposer recorded".
    applied_from: list[str] = Field(default_factory=list)
    #: §6.C.4 — the tray's batch comment, **carried onto the round rather than
    #: read back from the draft**, because the same transaction empties the tray
    #: the comment lives on. Stamp before the clear or the comment is gone.
    batch_comment: str = ""
    #: The body sha this round produced, "" when it wrote nothing. This is the
    #: join key ``plan_revisions`` needs to attach the two fields above to the
    #: right changelog row — see :func:`plan_revisions`.
    sha: str = ""
    #: Addressees that resolve to nobody in the room this batch posted into.
    #: **Reported and skipped, never fatal** (AC-5.6): one bad name must not
    #: swallow a batch, and a name silently dropped is worse than one printed.
    unresolved: list[str] = Field(default_factory=list)


class PlanHistoryEntry(BaseModel):
    """A display counter and a verb. ``seq`` is **never a precondition**."""

    seq: int
    at: str
    by: str
    verb: str   # create|write|refused|note|apply|resolve|submit|round|approve
    section: str = ""
    detail: str = ""


class ProjectPlan(BaseModel):
    """The sidecar. It never contains the document body."""

    project_id: str
    notes: list[PlanNote] = Field(default_factory=list)
    draft: PlanReviewDraft | None = None
    rounds: list[PlanReviewRound] = Field(default_factory=list)
    history: list[PlanHistoryEntry] = Field(default_factory=list)
    #: STORED. Bumped ONLY inside :func:`submit_review`'s transaction on a
    #: regular project, and on the coordinator's write on a front-desk one
    #: (§2.5, §3.2, §7.2 U3).
    revision: int = 1
    accepted_at: str = ""
    #: **When the owner FIRST put their name to this plan, and it never moves.**
    #: The boundary :func:`note_is_deviation` compares a note's ``at`` against.
    #:
    #: Separate from ``accepted_at`` because that field re-stamps on every
    #: applied deviation — *"when the owner LAST put their name to this text"*,
    #: which is the right meaning for ``changed_since_acceptance`` and exactly
    #: the wrong one here. On a real plan the two have drifted sixteen hours
    #: and four acceptances apart, with eleven deviation notes filed inside the
    #: gap; deriving from ``accepted_at`` would quietly demote all eleven.
    #:
    #: Backfilled on load by :meth:`_backfill_first_acceptance` for every plan
    #: written before this field existed, so there is no migration.
    first_accepted_at: str = ""
    accepted_by: str = ""
    accepted_via: str = ""          # "owner" | "coordinator"
    accepted_revision: int = 0
    accepted_spec_digest: str = ""
    #: When ``init_plan_sidecar`` seeded this plan — and ONLY that call (§3.5
    #: property 3). Empty on a plan a human wrote by hand and on every project
    #: that predates the feature, which is exactly why those read ``executing``
    #: rather than being stopped at the gate. **Not "a sidecar exists"**: a
    #: pre-feature project grows a sidecar the moment someone files a note.
    #: ``plan create`` does not set it either — an owner running it on a
    #: long-running project must not push that project back into ``spec-ing``.
    seeded_at: str = ""
    #: **When the OWNER first opened a review round on this plan, and it never
    #: moves.** The start line of the spec lock before acceptance
    #: (:func:`_spec_is_locked`): once the user has looked at the draft once,
    #: what the plan says is theirs, and the keeper proposes instead of writing.
    #:
    #: Stamped in :func:`submit_review`'s transaction — beside
    #: ``_stamp_acceptance_locked`` and for the identical ordering reason, since
    #: the projection the coordinator's next turn reads is built a few lines
    #: below it — and backfilled on load by
    #: :meth:`_backfill_first_user_review` from ``rounds``, so every plan that
    #: already exists gets the right answer with no migration. That backfill is
    #: also what makes the fact survive the ``MAX_ROUNDS`` trim in :func:`_save`.
    first_user_review_at: str = ""
    created_at: str = ""
    updated_at: str = ""

    @model_validator(mode="after")
    def _backfill_first_user_review(self) -> "ProjectPlan":
        """Recover ``first_user_review_at`` from ``rounds`` (the M3-lock trigger).

        ``PlanReviewRound.opened_by`` has exactly one writer
        (:func:`_open_round_locked`, reached only from :func:`submit_review` and
        :func:`open_round`), and the owner's tray submit is the only caller that
        passes ``by == OWNER``. So *"the earliest round the user opened"* is the
        fact already on disk; it just had no field. Every plan written before
        this shipped — which is every plan this feature is about — reads
        correctly on its next load, and there is no backfill script.

        Deliberately NOT ``any(r.closed_at …)``: the coordinator's own outbound
        batch also closes when the user answers it, and the question here is who
        OPENED a round, i.e. who last spoke about the document.

        ``rounds`` is capped at ``MAX_ROUNDS`` and trims from the front, so a
        very long plan can lose the round this derives from. Latching the value
        into a stored field is what makes that harmless: the fact is recovered
        once and then kept, so the lock can never silently DISENGAGE on a busy
        project. Runs on every load and only ever fills a blank, so it is
        idempotent.
        """
        if self.first_user_review_at:
            return self
        opened = [r.opened_at for r in self.rounds if r.opened_by == OWNER and r.opened_at]
        if opened:
            self.first_user_review_at = min(opened)
        return self

    @model_validator(mode="after")
    def _backfill_first_acceptance(self) -> "ProjectPlan":
        """Recover ``first_accepted_at`` on a plan written before it existed.

        The earliest ``approve`` in ``history`` IS the first acceptance —
        :func:`_stamp_acceptance_locked` is the only writer of that verb and it
        writes one per owner apply. So the fact was already on disk; it just had
        no field.

        ``history`` is capped at 200 and trimmed from the front, so a very long
        plan can have lost its first ``approve``. The fallback is
        ``accepted_at``, which is *later* than the truth — it can only make the
        window narrower, never wider, so the failure mode is a post-acceptance
        note reading as an ordinary one and never the reverse. Silently
        widening the deviation window on old data is the outcome worth ruling
        out; missing a note on a plan with 200+ events is not.

        Runs on every load and only ever fills a blank, so it is idempotent and
        a plan accepted after this shipped is untouched by it.
        """
        if self.first_accepted_at or not self.accepted_at:
            return self
        approvals = [h.at for h in self.history if h.verb == "approve" and h.at]
        self.first_accepted_at = min(approvals) if approvals else self.accepted_at
        return self


# ---------------------------------------------------------------------------
# Derived reads that must have exactly one implementation
# ---------------------------------------------------------------------------


def open_notes_for_you(plan: ProjectPlan) -> int:
    """Open notes on this plan whose ``to == "user"`` — proposal-carrying or not.

    **One number, one definition, four consumers** (§3.3). A proposal addressed
    to another agent does not count; a note addressed to nobody does not count,
    because ``to == ""`` is *"recorded, never sent"* and a note nobody was sent
    must never badge the user, let alone stop work.

    It is read server-side by the execution gate (§7.4), so it is defined here,
    once, rather than restated by each caller — the boolean-and-its-count drift
    §3.3 describes is exactly what a second implementation produces.

    **M5 AC-5.10 — a reply does not release the gate, and AC-5.9 is not
    symmetric.** ``submit_review``'s ``ask`` row now closes its parent
    ``answered``, and this number moves differently depending on which end of
    the two-party channel replied. Written out because the asymmetry is easy to
    read past and both halves are pinned by their own test:

    * **the coordinator answers the user's note** — the parent is ``to ==
      keeper`` and never counted here, so closing it is −0; the response is
      ``to == user`` and open, so +1. AC-5.9's delta is **zero** and the gate
      does not release. It *engages*: the user now has something to decide, and
      they decide it, not the coordinator.
    * **the owner replies to the coordinator's note** — the parent is ``to ==
      user`` and counted, so closing it is **−1**, and the reply is ``to ==
      keeper`` and adds nothing. The gate may release where it did not before.

    The second is correct and is not a leak. This integer means *"waiting on
    the user"*, and a note the user has answered is not waiting on them — the
    ball is with the coordinator. Holding the gate through a user-authored
    reply would narrow its release condition to ``apply``/``reject``/
    ``dismiss`` alone, so a user who replies instead of deciding would jam
    their own project with nothing on any surface saying why.
    """
    return sum(1 for n in plan.notes if n.status == "open" and n.to == OWNER)


def section_changed(body: str, note: PlanNote) -> bool:
    """Does this proposal still make sense? A string comparison, not a merge
    trial (§3.3). False when the section is gone — that is ``section_missing``,
    a different answer.

    False, too, when **no base was ever captured** — and since AC-1.9 that is a
    much narrower set than it was.

    **This paragraph used to say a comment or a question "has nothing to compare
    against", and that is now false.** ``base_section`` was the proposal's base
    and nothing else (:attr:`PlanNote.base_section` = ``SectionEdit.base``), so
    only a write-shaped note carried one; :func:`_capture_base` now fills it in
    for any note that names a section, so a comment anchored to a section
    reports *changed* when its section really has changed. That was the point:
    the notes most likely to go stale were the ones this predicate could not
    answer for. **Nothing in this function changed** to make it true — the
    capture widened, not the test — which is exactly why it needed saying here.

    What has NOT changed is why comparing an EMPTY base is refused. A real
    section's text is never ``""`` (it is at minimum its heading line), so a
    ``""`` base can only mean *nothing was captured*, and comparing anyway made
    every such note report *changed* from the instant it was filed and forever
    after. A banner that fires on every note is one nobody reads. Two shapes
    still land here with no base: a note naming no section at all, and every
    note filed before the capture existed — which is deliberately not backfilled,
    because inventing a base asserts an author saw text they never saw.

    A slug resolving to nothing is :func:`section_missing`, answered above, and
    :func:`_capture_base` is careful not to shadow it.

    :func:`_changed_warning` states the same gate as ``not note.proposal``; it
    is written against ``base_section`` here because the draft-tray probe in
    ``_draft_view`` supplies a row's staged ``base`` and has no proposal to
    offer.

    **Trailing newlines are not an edit**, and the normalization is
    :func:`extent_key`'s rather than this function's — the proof lives beside
    the splice that motivates it. Without it the go-note was unacceptable on
    every project ever created: ``## Approval`` is last in the seed template, so
    the base captured at project creation ends in one newline, and the extent
    ends in two the moment the coordinator appends anything after it.

    The guard below stays on the **raw** ``base_section``. ``""`` still means
    *nothing was captured*; ``extent_key("")`` is ``""`` either way, so
    normalizing first would not change the answer but would make a reader check.
    """
    if not note.base_section:
        return False
    current = _current_section(body, note.section)
    return current is not None and extent_key(current) != extent_key(note.base_section)


def section_missing(body: str, note: PlanNote) -> bool:
    """The advisory slug no longer resolves, so the proposal cannot be applied
    and degrades to a comment with a suggestion attached (§3.3)."""
    return bool(note.section) and _current_section(body, note.section) is None


def _comparison_view(text: str) -> str:
    """Both sides of a quote match, reduced to what they have in common.

    **This is the whole risk of :func:`quote_matches`, so it is its own
    function.** The stored quote came from ``window.getSelection()`` over
    RENDERED DOM text; the body here is RAW MARKDOWN. A naive ``in`` test
    therefore reproduces the bug ``planAnchor.tsx`` already records — every
    quote spanning a bold word, a code span or an emphasis fails to match — only
    this time on the *warning*, which is worse than useless: it would tell the
    owner a passage is gone every time they quoted emphasis.

    So: emphasis and code markers out, links reduced to their text, whitespace
    collapsed. Deliberately incomplete — it does not attempt tables, images or
    reference links, and it does not need to. The bias is toward FALSE
    NEGATIVES, which is the right direction for a warning: staying silent costs
    a hint, crying wolf costs the feature.

    Never mutates the text that gets printed, and is used for nothing but
    counting.
    """
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_`~]+", "", text)
    return " ".join(text.split())


def quote_matches(body: str, note: PlanNote) -> int:
    """How many times this note's quote occurs in its section **as it stands
    now**.

    ``0`` — the passage is gone. ``1`` — anchored. ``>= 2`` — ambiguous, and the
    note is re-attaching to the first match by luck.

    **No ``-1`` sentinel rides anywhere.** *"Was the question askable"* is a
    separate question with its own predicate (:func:`quote_checked`), which is
    the same argument :class:`PlanIndex` already makes for ``accepted_revision``
    — a sentinel on a count asks every caller to know it, and one of them will
    not.

    Counts over :func:`_comparison_view`, never over the raw text.
    """
    if not quote_checked(body, note):
        return 0
    section = _comparison_view(_current_section(body, note.section) or "")
    quote = _comparison_view(note.quote)
    return section.count(quote) if quote else 0


def quote_checked(body: str, note: PlanNote) -> bool:
    """Was the question :func:`quote_matches` answers askable at all?

    False when the note carries no quote and when its ``section`` is empty or
    does not resolve. That last case is :func:`section_missing` — a **different
    answer**, and this must not shadow it: *"the passage is gone"* sends a
    reader looking through a section, and *"the section is gone"* does not.
    """
    if not note.quote or not note.section:
        return False
    return _current_section(body, note.section) is not None


def may_decide(by: str) -> bool:
    """May ``by`` accept or reject a proposal? **The ownership test, and its
    only home.**

    Applying and rejecting are decisions about the document, and the document is
    the owner's. Both doors to those four verbs resolve THIS function —
    :func:`resolve_note`'s guard, :func:`submit_review`'s row guard, and the
    route's ``can_decide`` flag — so none of them restates ``by == OWNER``.

    That is not tidiness: it is the same hazard :func:`add_note` names at its
    *"the one door"* paragraph, *"a rule with two homes is a rule with two
    answers"*, applied to permission instead of addressing. A client that is
    shown an Accept button it will be refused for pressing is exactly the drift
    a second home produces.
    """
    return by == OWNER


def _refuse_non_owner_decision(by: str, verb: str) -> None:
    """Raise the refusal for a decision ``by`` may not make, or return.

    The refusal **sentence**, lifted verbatim from :func:`resolve_note` so the
    two doors to the same four verbs cannot diverge in wording either. This
    function contains no comparison of its own — it asks :func:`may_decide` and
    renders the answer.

    ``verb`` is the caller's own word, because a refusal names what the caller
    tried: :func:`resolve_note` passes its ``action`` (``"apply"``/``"reject"``)
    and :func:`submit_review` passes its row ``kind``
    (``"accept"``/``"reject"``).
    """
    if may_decide(by):
        return
    raise PlanForbiddenError(
        f"@{by} may not {verb} a note — applying and rejecting are "
        f"decisions about the document and the owner makes those. "
        f"File what you want changed as a note to the user "
        f"(`plan note --to user`) and let them decide."
    )


# ---------------------------------------------------------------------------
# Internals — paths, identity, ids
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _plan_path(project: "Project", ctx: "ModelContext") -> Path:
    return ctx.changelog_dir(project.id, project.name) / "plan.json"


def _gen_id(prefix: str, taken: Iterable[str]) -> str:
    """``n-xxxx`` / ``d-xxxx`` / ``r-xxxx``, server-assigned and immutable."""
    seen = set(taken)
    while True:
        candidate = f"{prefix}-{secrets.token_hex(3)}"
        if candidate not in seen:
            return candidate


def keeper(project: "Project") -> str:
    """The agent that may write this document — **the project coordinator**,
    displayed as *"kept by @name"* and not configurable (§4).

    Keeper is not decider: on a regular project the user decides what the bytes
    say by accepting a batch; the keeper is only who may put them there.
    """
    return project.coordinator_name


def may_write(project: "Project", by: str) -> bool:
    """Only the owner and the keeper write the document. Every other agent
    proposes through a note (§4)."""
    return by == OWNER or by == keeper(project)


def _writer_identity(
    project: "Project", ctx: "ModelContext", by: str
) -> tuple[str, str]:
    """``(participant_id, participant_name)`` to attribute a changelog entry to.

    The writing agent, **or the coordinator when the writer is the owner** — the
    owner is not a chatroom participant, so a FILE event authored by them has
    nowhere to sit. Precedent: the owner-upload path, ``files.py:117-146``.

    ``by`` is an agent **name**, because that is the vocabulary ``PlanNote.to``
    and ``keeper()`` speak. ``from_participant_id`` is an **id**, because that is
    what every other writer of a changelog payload puts there
    (``files.py:312``, ``messages.py:320``, ``chatrooms.py:287``) and what any
    reader resolving a participant looks it up by. **Returning the name for both
    is how a FILE event grows an id nothing can resolve** — the avatar renders
    blank, ``Participant.get`` returns ``None``, and nothing errors.

    Resolved through the plan room's participant list, which is the same lookup
    :func:`_members_by_name` already does for a review room. An agent that has
    left the room falls back to its name: an unresolvable id is better than a
    write that fails on attribution.
    """
    if by == OWNER:
        return project.coordinator_id, project.coordinator_name
    if by == project.coordinator_name:
        return project.coordinator_id, by
    room = Chatroom.get(project.id, PLAN_ROOM, ctx)
    if room is not None:
        for participant in room.list_participants():
            if participant is not None and participant.name == by:
                return participant.id, by
    return by, by


# ---------------------------------------------------------------------------
# Internals — sidecar and body I/O
# ---------------------------------------------------------------------------


def _load(project: "Project", ctx: "ModelContext") -> ProjectPlan:
    """Read the sidecar, or an **in-memory** default if there is none.

    Lazy by design: a project that never uses a note never grows a sidecar, and
    this function must not create one — AC-2.7 is asserted on PLAN.md's bytes
    *and* on the sidecar's absence.

    **Absent and unreadable are not the same answer.** Absent is the default;
    unreadable raises :class:`PlanCorruptError`.
    """
    raw = FileUtil.read(_plan_path(project, ctx), "json")
    if isinstance(raw, dict):
        try:
            return ProjectPlan.model_validate(raw)
        except Exception as exc:
            # **Never a fresh default here.** A default is indistinguishable
            # from "this project has no sidecar yet", so the caller writes and
            # the next :func:`_save` lands it on top of the file — every note,
            # every round and the acceptance record gone, with nothing logged
            # and nothing returned non-2xx. Raising is the whole fix: it keeps
            # the bytes on disk for a human to look at.
            logger.error(
                f"plan.json for project {project.id[:8]} does not validate "
                f"({type(exc).__name__}: {exc}) — refusing to read it, and "
                f"therefore refusing to overwrite it"
            )
            raise PlanCorruptError(
                f"Project {project.id!r} has a plan.json that does not "
                f"validate: {exc}"
            ) from exc
    return ProjectPlan(project_id=project.id, created_at=_now())


def _check_sidecar_encodable(payload: dict) -> None:
    """Refuse a sidecar the plan surface would not be able to serve. **The
    backstop, at the one place every write funnels through.**

    :func:`check_note_text` is the guard that produces the *good* message —
    it names the field. This one exists because it is asked by
    :func:`_save`, which is the **only** writer of ``plan.json``: every
    route, every CLI path and every route nobody has written yet reaches disk
    through line ``FileUtil.write(_plan_path(…))`` below. A per-field guard
    closes the fields somebody enumerated; this closes the *class*.

    Two proven holes it exists for, neither of them a note field:
    ``POST /plan/notes/{id}/resolve`` writes ``reason`` straight into
    ``note.resolution``, and ``PUT /plan/draft`` is documented as *"never
    validated … refused only for the 100-row cap"*. Both stored a lone
    surrogate and bricked ``GET /plan`` permanently.

    **The predicate is the crash.** ``FileUtil.write(…, "json")`` serialises
    with ``ensure_ascii=True``, which *accepts* an ill-formed ``str`` happily —
    a surrogate becomes a six-character ASCII escape on disk and reads back as
    the same ill-formed ``str``. Every reader then serialises with
    ``ensure_ascii=False`` and encodes to UTF-8, and *that* is what raises. So
    the check here is ``json.dumps(…, ensure_ascii=False).encode("utf-8")``,
    exactly what Starlette's ``JSONResponse`` does — **note the ``.encode``**:
    ``json.dumps(…, ensure_ascii=False)`` on its own returns a ``str`` and
    never raises, so a guard written without the encode step would pass
    everything and prove nothing.

    ``default=str`` mirrors ``FileUtil.write``'s own call, so this serialises
    the same object graph the writer will and cannot fail on a type the writer
    would have coped with.
    """
    try:
        json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PlanInputError(
            f"this write holds an unpaired UTF-16 surrogate "
            f"(U+{ord(exc.object[exc.start]):04X}); the plan sidecar takes it "
            f"and every later read of the plan fails to encode it, so it is "
            f"refused rather than stored"
        ) from None


def _save(project: "Project", ctx: "ModelContext", plan: ProjectPlan) -> None:
    """Persist the sidecar, applying the two caps that trim from the front.

    **The funnel** — the single ``FileUtil.write`` of ``plan.json`` in this
    module, hence the one place a whole-sidecar guard can cover every path.
    """
    if len(plan.rounds) > MAX_ROUNDS:
        plan.rounds = plan.rounds[-MAX_ROUNDS:]
    if len(plan.history) > MAX_HISTORY:
        plan.history = plan.history[-MAX_HISTORY:]
    plan.updated_at = _now()
    if not plan.created_at:
        plan.created_at = plan.updated_at
    payload = plan.model_dump()
    _check_sidecar_encodable(payload)
    FileUtil.write(_plan_path(project, ctx), payload, "json")


def _note(plan: ProjectPlan, by: str, verb: str, *, section: str = "", detail: str = "") -> None:
    """Append one history row. ``seq`` is a display counter and nothing reads it
    as a precondition (§3.1)."""
    plan.history.append(
        PlanHistoryEntry(
            seq=(plan.history[-1].seq + 1) if plan.history else 1,
            at=_now(),
            by=by,
            verb=verb,
            section=section,
            detail=detail,
        )
    )


def _room(project: "Project", ctx: "ModelContext") -> Chatroom:
    room = Chatroom.get(project.id, PLAN_ROOM, ctx)
    if room is None:
        raise PlanNotFoundError(f"Project {project.id!r} has no {PLAN_ROOM} room")
    return room


def _read_body(project: "Project", ctx: "ModelContext") -> str:
    """The document. ``404`` when there is none — a DM, or a project whose
    PLAN.md was deleted."""
    raw = _room(project, ctx).get_file(PLAN_FILE)
    if raw is None:
        raise PlanNotFoundError(f"Project {project.id!r} has no {PLAN_FILE}")
    return raw.decode("utf-8")


def _body_exists(project: "Project", ctx: "ModelContext") -> bool:
    room = Chatroom.get(project.id, PLAN_ROOM, ctx)
    return room is not None and room.file_exists(PLAN_FILE)


def _current_section(body: str, slug: str) -> str | None:
    """The section's current text, or None when the slug does not resolve.

    The one place this module looks at the document's shape, and it does it by
    asking :mod:`plan_markdown` — there is no heading logic here.
    """
    extent = section_extent(body, slug)
    if extent is None:
        return None
    return body[extent[0]:extent[1]]


def _vanished_refusal(sections: Sequence[str]) -> str:
    """*"Section X no longer exists; the proposal cannot be applied."*

    **One home for the sentence, because two doors owe it and they said
    different things.** :func:`resolve_note`'s apply arm has always said this.
    :func:`submit_review` — the browser's tray accept, which is the door almost
    everyone uses — said *"a staged row is stale"*, which is the one thing that
    had NOT happened: nothing moved under the row, there is nothing to re-read,
    and the reconcile control the browser opens on that message offers to re-base
    onto ``current``, which is ``""``, so it re-sends the same bytes and earns
    the same refusal for as long as the user is willing to keep clicking.

    For a single section this reproduces the older sentence byte for byte, which
    is deliberate: the CLI's ``plan conflicts`` output and two tests match on it.
    """
    names = ", ".join(repr(s) for s in sections)
    one = len(sections) == 1
    return (
        f"Section{'' if one else 's'} {names} no longer "
        f"{'exists' if one else 'exist'}; the proposal cannot be applied"
    )


# ---------------------------------------------------------------------------
# Internals — the write, as data
# ---------------------------------------------------------------------------


def _check_body_size(body: str) -> None:
    size = len(body.encode("utf-8"))
    if size > MAX_BODY_BYTES:
        raise PlanLimitError(
            f"plan body is {size} bytes, limit is {MAX_BODY_BYTES}"
        )


def _splice(body: str, edits: Sequence[SectionEdit]) -> tuple[str, list[StaleSection]]:
    """§4.1 steps 3–5, as a pure function over ``str``.

    Every edit is checked against the body **as it stands**, then — only if none
    is stale — every edit is spliced in document order. Checking and splicing are
    separate passes because the check must not see a body the caller's own
    earlier edit already moved.

    All-or-nothing across the list: applying four of five produces a revision the
    submitter did not intend, and refusing costs them one look at one section.

    **An unresolvable slug is refused unless the edit says ``create``.** This
    used to coerce a missing section to ``""`` before the comparison, which made
    it compare equal to the ``""`` base a create carries — so a proposal filed
    against a section that had since been **deleted** read as *fresh*, raised no
    conflict, and fell through :func:`replace_section` to ``_append_section``,
    landing at the end of the document beside the copy it meant to replace. Two
    different facts wearing one value; the flag separates them, and the default
    is the safe one. It is the same answer ``resolve_note``'s apply path has
    always given (*"Section no longer exists; the proposal cannot be applied"*)
    — that door simply checked first, and the browser's tray did not.

    **THE FLAG IS NOW SET UPSTREAM ON BOTH ACCEPT DOORS, AND THIS FUNCTION DID
    NOT MOVE.** A proposal written against a section that was ALREADY gone is a
    create and always was; the note records that in its ``base_section``, and
    :func:`_row_creates` and :func:`resolve_note`'s apply arm read it there. So
    the two facts are still separated — they are just separated by a caller that
    knows which, rather than by a blanket refusal that could not tell. What
    reaches here is unchanged: a row with ``create`` false and a slug that does
    not resolve is still a ``409``, which is exactly the case the paragraph above
    describes and the one the guard tests pin.

    A **delete** of an already-absent section takes the same 409 rather than a
    silent no-op. One branch, one rule, and the honest answer to *"remove this"*
    when there is nothing there. :func:`delete_section` itself is untouched and
    still no-ops on a missing slug: that function is the surgery, not the policy.
    """
    stale: list[StaleSection] = []
    for edit in edits:
        current = _current_section(body, edit.section)
        if current is None:
            if not edit.create:
                stale.append(
                    StaleSection(
                        section=edit.section, base=edit.base, current="", text=edit.text
                    )
                )
                continue
            current = ""
        # NORMALIZED, AND NOT AS A CONVENIENCE — this is what keeps the tray
        # honest. `_draft_view` computes a row's `stale` through
        # `section_changed`, but the submit that row rides goes through here, so
        # the two must answer alike: normalizing only the first would have a row
        # render `stale: false` and then earn a 409 from `Send review`.
        #
        # It can never lose an update. The bytes it forgives are exactly the
        # trailing newline run `replace_section` discards from the caller and
        # re-takes from the document, so the splice this guards produces the
        # same output either way. `StaleSection` below still reports raw text.
        if extent_key(edit.base) != extent_key(current):
            stale.append(
                StaleSection(
                    section=edit.section, base=edit.base, current=current, text=edit.text
                )
            )
    if stale:
        return body, stale

    out = body
    for edit in edits:
        out = replace_section(out, edit.section, edit.text)
    return out, []


def _file_spec(
    project: "Project", ctx: "ModelContext", body: str, by: str, *, created: bool
) -> BatchEntrySpec:
    """The one changelog entry a plan write appends."""
    content = body.encode("utf-8")
    writer_id, writer_name = _writer_identity(project, ctx, by)
    return BatchEntrySpec(
        entry_type=(
            ChangelogEntryType.FILE_CREATED if created else ChangelogEntryType.FILE_UPDATED
        ),
        payload=FilePayload(
            chatroom_name=PLAN_ROOM,
            filename=PLAN_FILE,
            content_b64=FileUtil.to_base64(content),
            sha256=FileUtil.sha256(content),
            from_participant_id=writer_id,
            from_participant_name=writer_name,
        ),
    )


def _message_spec(
    project: "Project",
    ctx: "ModelContext",
    room_name: str,
    by: str,
    content: str,
    expects: Sequence[str] = (),
) -> tuple[str, BatchEntrySpec]:
    """An **agent's** review message. Never the owner's — see the refusal below.

    A message is not a file, and this is the line where treating them alike cost
    a coordinator six hours of not taking a turn. ``_writer_identity`` stands the
    coordinator in for the owner because a FILE event authored by a
    non-participant has nowhere to sit; a MESSAGE has no such problem
    (``post_user_message`` posts one from the owner on every turn), and the
    substitution produces a message from the coordinator addressed to the
    coordinator, which ``participant_notifier.py:120`` drops as self-authored.

    So the owner does not come through here at all: their messages go out
    through the ordinary user-message door, injected as
    :data:`OwnerPoster`. Refusing rather than falling back is deliberate — a
    silent fallback here is the original bug, reachable again the first time a
    caller forgets the poster.
    """
    if by == OWNER:
        raise PlanInputError(
            "an owner's review message must go out through the user-message "
            "door (submit_review(..., post=...)), not be attributed to the "
            "coordinator — a message the coordinator appears to have sent "
            "itself is dropped by the runner and never answered"
        )
    writer_id, writer_name = _writer_identity(project, ctx, by)
    message_id = str(secrets.token_hex(8))
    return message_id, BatchEntrySpec(
        entry_type=ChangelogEntryType.MESSAGE,
        payload=MessagePayload(
            chatroom_name=room_name,
            id=message_id,
            ts=datetime.now(UTC),
            from_participant_id=writer_id,
            from_participant_name=writer_name,
            content=content,
            expects_response_from=list(expects),
        ),
    )


class _PreparedWrite:
    """A write, computed but **not yet appended**.

    Splitting the computation from the commit is what lets
    :func:`submit_review` put the ``FILE_UPDATED`` and every review message into
    one :meth:`ChangelogRunloop.append_batch` call. Two appends can half-happen;
    one batch cannot, and *"the write and the send are one act or neither"*
    (§4.2) is only true if they share it.
    """

    __slots__ = ("result", "spec", "notes", "moved", "body")

    def __init__(self, result, spec=None, notes=(), moved=(), body=""):
        self.result = result
        self.spec = spec
        #: The shorthand notes this write owes the sidecar, built **here** and
        #: filed unchanged by :func:`_finish_locked`. Carrying the objects
        #: rather than rebuilding them is what lets the caller validate the
        #: exact notes it is about to file, above the append: a reservation of
        #: *n* slots followed by a filing of *m* notes reopens the hole
        #: silently, and nothing would enforce the equality.
        self.notes = list(notes)
        self.moved = list(moved)
        #: The document **as it will read once this write lands** — the text
        #: §5.7 quotes back to a reviewer. On a refusal or a no-op it is the
        #: stored body, which is the same sentence: what the section says now.
        self.body = body

    @property
    def wrote(self) -> bool:
        return self.spec is not None


# ---------------------------------------------------------------------------
# M3 — the executing-phase spec lock.
#
# After acceptance the coordinator still KEEPS the document — it ticks the
# milestone boxes and edits the HTML comments that carry provenance — but it may
# no longer RESPEC it. What the plan says is the user's, and a coordinator write
# that moves it is refused and comes back to the user as a one-click proposal.
#
# The predicate is ``spec_digest`` and not a new one, for the reason §
# Guardrails gives: ``approve --expect-spec`` and ``changed_since_acceptance``
# already ask this exact question of this exact function, and a second
# "did the spec move?" is a second answer waiting to disagree.
# ---------------------------------------------------------------------------


def _user_has_reviewed(plan: ProjectPlan) -> bool:
    """True once the OWNER has closed a review round on this plan.

    Reads the latched :attr:`ProjectPlan.first_user_review_at`, which is stamped
    in :func:`submit_review` and derived from ``rounds`` on load by
    :meth:`ProjectPlan._backfill_first_user_review` — see that validator for why
    the fact is derived rather than migrated, and why it is then kept.

    Named rather than inlined into :func:`_spec_is_locked` because three places
    ask it — the lock, the refusal's wording, and the projection the
    coordinator's prompt reads — and a predicate with three copies has three
    answers.
    """
    return bool(plan.first_user_review_at)


def _spec_is_locked(project: "Project", plan: ProjectPlan, *, by: str) -> bool:
    """Whether ``by``'s write must leave ``spec_digest`` where it found it.

    The sibling of :func:`note_addressee_allowed` — the rule as one named
    predicate, so a caller deriving *"may I write this?"* reads this and not a
    second copy of three conditions.

    Three conjuncts, each load-bearing:

    * **``by == keeper(project)``** — AC-3.5. The owner writes freely in both
      phases. They are the decider, and a lock aimed at the decider is not a
      lock, it is a bug.
    * **``project.surface == "regular"``** — a front-desk plan is ``executing``
      *from creation, forever* (:attr:`Project.phase`, the branch its own
      docstring calls "LOAD-BEARING THREE TIMES") and there the coordinator's
      write **is** the acceptance (:func:`_finish_locked`'s revision bump, §7.2
      U3). Without this conjunct M3 locks every front-desk coordinator out of
      its own document on its second write, and the project's Guardrails forbid
      changing that shape by name.
    * **``plan.accepted_at or _user_has_reviewed(plan)``** — the start line, and
      the only conjunct this change touched.

    **THE START LINE MOVED, AND HERE IS WHY.** The lock used to require
    ``phase == "executing" and plan.accepted_at``: it engaged when the user
    ACCEPTED. That left the whole of ``spec-ing`` ungated, and ``spec-ing`` is
    where the authority actually leaks — a coordinator that has just been given
    a round of user feedback decides, section by section, whether that feedback
    "settles" something, and it decides yes nearly every time because it wrote
    the summary of the feedback itself. Measured on one real project
    (``chuswine-geo-b2b``): after the user's first review round closed the
    coordinator made 26 direct writes to the document — Goal, Guardrails,
    Sequencing and every milestone — against 0 proposals the user could accept
    or reject. The plan was never accepted, so the lock never fired once.

    So the trigger is now *"the user has looked at this"*, not *"the user has
    signed this"*. The coordinator drafts freely until the first user-opened
    review round; from then on what the plan SAYS is the user's, and a keeper
    write that moves ``spec_digest`` is refused and re-filed as a one-click
    proposal.

    **``accepted_at`` is kept as a disjunct, not replaced.** Acceptance remains
    sufficient, so a legacy plan that was accepted without ever recording a
    user-opened round still locks. On any plan that reached acceptance the
    normal way the user reviewed it first, so the disjunct is usually redundant
    — which is exactly the property that makes every existing test of the
    accepted half still describe live behaviour.

    **``phase == "executing"`` had to go**, and its going is not a narrowing of
    AC-3.1 but a deliberate widening past it: the whole point is to fire during
    ``spec-ing``, where that conjunct is false by definition. ``accepted_at``
    implies ``executing``, so nothing that was locked before is unlocked now.

    **What is still legal for the keeper after the lock engages**, because the
    comparison is on ``spec_digest`` and that digest normalizes them away
    (``plan_markdown.spec_digest``): ticking and unticking checkboxes, editing
    ``<!-- … -->`` provenance comments, and reflowing whitespace. Progress
    bookkeeping is untouched, which is what makes moving the start line safe to
    ship without also redesigning how milestones report completion.
    """
    return (
        by == keeper(project)
        and project.surface == "regular"
        and (bool(plan.accepted_at) or _user_has_reviewed(plan))
    )


def _write_refusal(project: "Project", *, by: str) -> str:
    """Why a non-writer's ``plan update`` was refused, **and the route that
    works** (M3 AC-3.6, and § Guardrails on agent-facing 4xx).

    This string used to read *"use `plan note … --edit-file`"*. M2 made that
    exact call refuse that exact caller, so the refusal named a door that also
    refuses you — worse than naming no door, because it costs a round trip to
    learn it. It was the one actively misleading string left in the tree.

    Split by phase, matching :func:`_note_refusal`'s shape and for the same
    reason: ``shared-context`` is the ``spec-ing`` route and the workroom is the
    ``executing`` route, and a refusal that offers both leaves the model to
    guess which of them is open to it right now. The caller here is always a
    specialist — :func:`may_write` admits the owner and the keeper — so phase is
    the only axis.

    Both halves end on *"the coordinator writes"*, which is this refusal's
    equivalent of :func:`_note_refusal`'s *"No --to value makes this note
    legal"*: it closes the search rather than inviting a retry with a different
    ``--section``.
    """
    keeper_name = keeper(project)
    if project.phase == "spec-ing":
        return (
            f"@{by} may not write this plan — answer in `shared-context`, "
            f"where @{keeper_name} relays the user's questions; the "
            f"coordinator writes."
        )
    return (
        f"@{by} may not write this plan — raise it with @{keeper_name} in your "
        f"workroom; the coordinator writes."
    )


def _spec_lock_refusal(
    project: "Project", *, by: str, note_ids: Sequence[str], accepted: bool
) -> str:
    """M3's own refusal, with its remedy in it (AC-3.3).

    Says what still **works** as well as what does not, because a model told
    only *"refused"* retries with a smaller edit of the same kind — and a
    smaller spec edit is still a spec edit, while a checkbox tick of any size
    lands. Names the note ids, because a ``403`` body is a bare string and the
    user's copy of the coordinator's text is the whole point of the refusal.

    **Two reasons behind one flag, because the lock now has two start lines.**
    The single sentence this used to be said *"it is accepted"*, which is FALSE
    on every refusal the pre-acceptance trigger produces — and a ``403`` whose
    stated reason the model can see is untrue is one it argues with rather than
    obeys. ``accepted`` picks the true half; both halves keep the two properties
    the original string was built for.
    """
    ids = ", ".join(note_ids) or "a note"
    why = (
        "it is accepted, and the user decides what it says"
        if accepted
        else "the user has reviewed this plan, and from here they decide what "
        "it says"
    )
    return (
        f"@{by} may not change what this plan says — {why}. Ticking a checkbox "
        f"or editing an HTML comment still applies. Your text is filed as {ids} "
        f"for the user to accept in one click; say why in `user-communication`."
    )


def _prepare_locked(
    project: "Project",
    ctx: "ModelContext",
    plan: ProjectPlan,
    edits: Sequence[SectionEdit],
    *,
    by: str,
) -> _PreparedWrite:
    """§4.1 steps 1–6, plus step 7's shorthand extraction. **Appends nothing.**

    Lock held, and ``plan`` is not mutated: a caller that abandons the write
    after this point leaves the sidecar exactly as it found it. **M3's spec lock
    keeps that promise** — it detects the refusal and reports it in
    ``WriteResult.locked``; the deviation note AC-3.3 owes the user is filed by
    the caller, which is the one holding the transaction that saves it.

    **Two refusals leave here, and they are different questions.** ``stale``
    means *the section moved under you, re-read and decide*. ``locked`` means
    *the plan is accepted and this changes what it says, so it is the user's
    call now*. Neither is ever populated alongside the other.
    """
    if not may_write(project, by):
        raise PlanForbiddenError(_write_refusal(project, by=by))

    for edit in edits:
        if len(edit.text) > MAX_EDIT_CHARS:
            raise PlanLimitError(
                f"section edit is {len(edit.text)} chars, limit is {MAX_EDIT_CHARS}"
            )

    body = _read_body(project, ctx)
    spliced, stale = _splice(body, edits)
    if stale:
        return _PreparedWrite(
            WriteResult(ok=False, revision=plan.revision, sha=body_sha(body), stale=stale),
            body=body,
        )

    # **A replacement must carry the section's OWN heading**, and this is the
    # guard that matters — nothing downstream fires if the heading survives.
    # ``replace_section`` splices over the section's whole extent, heading line
    # included, so a coordinator ticking two checkboxes and sending back only the
    # two bullet lines *deletes* ``### M1``: its criteria are absorbed into the
    # section above, the enclosing ``##`` digest moves, and the correcting write
    # is then refused by the spec lock as an edit to what the plan says. The
    # refusal is right and the damage is already done.
    #
    # **TWO QUESTIONS, NOT ONE, AND THE FIRST ALONE IS NOT THE GRAMMAR.** This
    # said *"nothing downstream fires if the heading never disappears"*, and that
    # was the bug: `drops_heading` asks only whether the text opens with a
    # heading, so the same corruption walked straight through it one level up.
    # A coordinator ran `tail -n +3` over `## Milestones` and sent the rest,
    # which opens on `### M1` — a heading, so the guard passed — and a level-3
    # subtree went over a level-2 extent. `## Milestones` left the document and
    # took M1 through M4 with it. `relevels_heading` is the second question.
    #
    # **Refused, not silently repaired.** Re-attaching the old heading would
    # mask an agent that genuinely meant to restructure, and would make the
    # document disagree with the text its author sent. A ``400`` can name the
    # line it expected; a silent repair cannot.
    #
    # **It sits after the staleness check and before the no-op check**, and the
    # position is deliberate. After staleness, because *"someone else wrote
    # first, re-read and decide"* is the more useful answer and re-reading hands
    # the writer the heading anyway — asking first would turn today's ``409`` on
    # a stale headingless write into a ``400``. Before the write, because this is
    # input validation and not a conflict: there is no current-versus-yours for
    # the caller to reconcile, so it raises rather than returning a
    # ``WriteResult``.
    for edit in edits:
        current = _current_section(body, edit.section)
        if current is None:
            continue
        if drops_heading(current, edit.text):
            raise PlanInputError(
                f"the replacement for {edit.section!r} drops its heading line; "
                f"a section edit is the section's FULL text and must open with "
                f"{heading_line(current)!r} (or a retitled heading of its own)"
            )
        if relevels_heading(current, edit.text):
            raise PlanInputError(
                f"the replacement for {edit.section!r} opens at a different "
                f"heading level; a section edit is spliced over that section's "
                f"own extent, so a heading one level deeper is absorbed by the "
                f"section above it and one level shallower adopts the sections "
                f"below it. Open with {heading_line(current)!r}, or a retitled "
                f"heading at the same level"
            )

    # **THE GO-NOTE'S SECTION SURVIVES WHILE THE GO-NOTE IS OPEN.** A proposal
    # is located by heading, so a coordinator that rewrites the plan without
    # ``## Approval`` — entirely legitimate under *"this is a starting document,
    # not a schema"* — leaves the one note that unblocks its own project with
    # nothing to apply to, and no surface anywhere says why the project stopped.
    #
    # **Server-enforced, deliberately, and this is the whole reason it is here
    # and not in the prompt.** The go-note replaced an Approve button precisely
    # so the guarantee would not depend on an agent remembering a rule; a
    # heading a coordinator can silently delete would hand that dependency
    # straight back.
    #
    # Keyed on the NOTE'S OWN ``section``, not on the constant: a plan whose
    # go-note points somewhere else (one seeded against a retitled section) is
    # protected where it actually lives.
    #
    # It refuses only a write that REMOVES a section that is there now. A plan
    # whose heading is already gone is already stuck, and refusing every
    # subsequent write would take away the one move that could restore it.
    go = open_go_note(plan)
    if go is not None and go.section:
        before = _current_section(body, go.section)
        after = _current_section(spliced, go.section)
        if before is not None and after is None:
            raise PlanInputError(
                f"this write removes the {go.section!r} section, and note "
                f"{go.id} — the one the user accepts to approve this plan — "
                f"proposes a replacement for it. Keep its heading line "
                f"({APPROVAL_HEADING!r} unless you retitled it); rewrite "
                f"anything else you like."
            )
        # **AND NOBODY BUT THE OWNER PUTS WORDS IN IT.** Keeping the heading is
        # not enough on its own: the section's *text* is the sentence the
        # coordinator is told to read as its go signal, and a keeper who can
        # write "User approves the plan." into it is a keeper who can forge the
        # signal it obeys.
        #
        # It would not actually start any work — the gate counts open notes and
        # is indifferent to what the document says — so this is not a hole in
        # the halt. It is a hole in the plan's honesty: the user opens PLAN.md
        # and reads that they approved something they did not, and the desk
        # agrees with the document.
        #
        # **KEYED ON THE EDIT, NOT ON THE SPLICED BODY**, and the difference is
        # a false positive that fires on almost every write. A section's extent
        # runs to the start of the next one, so the LAST section's text includes
        # everything after it — and ``## Approval`` is last in the seed
        # template. Appending any new section therefore shortens its extent
        # without touching a word of it, and a comparison of before-and-after
        # text would refuse a coordinator for adding `## Risks`.
        #
        # Comparing the edit's own replacement text against what the section
        # says asks the question that was meant: *are you writing THIS section?*
        # A whole-document rewrite that carries ``## Approval`` through
        # unaltered still passes, which is the case the text comparison was
        # there to protect.
        if by != OWNER:
            for edit in edits:
                if edit.section != go.section:
                    continue
                current = _current_section(body, edit.section)
                if current is not None and edit.text != current:
                    raise PlanInputError(
                        f"the {go.section!r} section is the user's to write while "
                        f"note {go.id} is open — accepting that note is what "
                        f"fills it in, and it is the signal that work may start. "
                        f"Leave its text exactly as it stands; every other "
                        f"section is yours."
                    )
            # **AND NOT THROUGH THE NEXT SECTION ALONG EITHER.** The loop above
            # asks *"are you writing THIS section?"*, and a single heading line
            # is all it takes for the answer to be no. A replacement for any
            # OTHER section may carry its own ``## Approval`` inside it, and
            # ``section_extent`` resolves a slug to the FIRST match — so the
            # injected copy becomes ``approval``, the real one is demoted to
            # ``approval-2``, and :func:`approval_state` reads the forged text.
            # That is the same forgery the loop above refuses, reached by
            # spelling it into a neighbour, and it lands on the one sentence the
            # coordinator's contract tells it to obey.
            #
            # Asked of the DOCUMENT rather than of the edits, because the edits
            # are what the trick hides in: no edit names ``approval``, and the
            # spliced body is the only place the injected heading is visible.
            # :func:`_section_prose` is what makes that safe to compare — it
            # stops at the first heading that follows, so neither of the two
            # ways an unrelated write shortens the last section's extent (a new
            # ``##`` after it, or a create appended INTO it) reads as a rewrite.
            # A legitimate whole-document rewrite that carries ``## Approval``
            # through unaltered still passes.
            if _section_prose(spliced, go.section) != _section_prose(body, go.section):
                raise PlanInputError(
                    f"this write changes what the {go.section!r} section says "
                    f"while note {go.id} — the one the user accepts to approve "
                    f"this plan — is open. That section is the user's alone, and "
                    f"a second {APPROVAL_HEADING!r} heading inside another "
                    f"section rewrites it just as surely as editing it does. "
                    f"Keep one, and leave its text as it stands."
                )

    if spliced == body:
        # §4.1 step 6 — an idempotent retry, and the reason a review of only
        # rejects, asks and dismissals costs nothing and needed no code: those
        # rows carry no text, so zero edits arrive here.
        return _PreparedWrite(
            WriteResult(ok=True, revision=plan.revision, sha=body_sha(body), noop=True),
            body=body,
        )

    shorthands, cleaned = extract_shorthand(spliced)
    _check_body_size(cleaned)
    changes = split_by_section(body, cleaned)
    moved = [slug for slug, _b, _a in changes]

    if _spec_is_locked(project, plan, by=by) and spec_digest(cleaned) != spec_digest(
        body
    ):
        # **AC-3.1 — splice first, compare after, and compare ``cleaned``.**
        #
        # The order is the criterion's real content. The digest is a property of
        # the resulting *document*, not of the incoming fragment: comparing the
        # fragment would refuse an edit that reverts a section to what the rest
        # of the file already says, and would accept one whose effect depends on
        # where it lands. Everything above — the splice, the staleness check, the
        # no-op check, the shorthand extraction — has already run, so this asks
        # its question of the bytes that were about to be written.
        #
        # ``cleaned`` and not ``spliced``, because shorthand extraction is what
        # actually reaches disk: a ``{@user: …}`` shorthand is a note, not a
        # change to what the plan says, and hashing it would refuse a write whose
        # only "spec move" the extractor was about to remove.
        #
        # **Appends nothing and files nothing.** This function's contract is that
        # ``plan`` is not mutated, so a caller that abandons the write leaves the
        # sidecar as it found it; the deviation note AC-3.3 owes the user is
        # filed by the caller, which is the one holding the transaction that will
        # save it. ``changes`` carries both texts at once — the only place they
        # are both in hand — which is exactly what ``StaleSection`` exists for.
        return _PreparedWrite(
            WriteResult(
                ok=False,
                revision=plan.revision,
                sha=body_sha(body),
                locked=[
                    StaleSection(section=slug, base=before, current=before, text=after)
                    for slug, before, after in _pair_renamed_rows(body, cleaned, changes)
                ],
            ),
            body=body,
        )

    return _PreparedWrite(
        WriteResult(ok=True, revision=plan.revision, sections=moved, sha=body_sha(cleaned)),
        spec=_file_spec(project, ctx, cleaned, by, created=False),
        notes=[
            PlanNote(id="", section=s.section, to=s.owner, by=by, comment=s.text)
            for s in shorthands
        ],
        moved=moved,
        body=cleaned,
    )


def _finish_locked(
    project: "Project", plan: ProjectPlan, prepared: _PreparedWrite, *, by: str, verb: str
) -> None:
    """Everything the sidecar owes a write that is about to land, or has landed.

    Files ``prepared.notes``, which its caller has already validated against
    this same ``plan`` under this same lock hold. It therefore **cannot raise**,
    and that is the property the callers' ordering rests on rather than luck.

    Two callers, two positions, one guarantee. :func:`_apply_locked` calls it
    after its append; :func:`submit_review` calls it *before* one, because the
    ``PROJECT_PLAN_STATE`` projection riding in that append's prelude has to
    count the notes this filer adds. Both are safe for the same reason: this
    function only mutates the in-memory ``plan``, ``_save`` is the module's only
    writer, and it runs after the append on both paths — so an append that
    raises persists nothing either way.
    """
    prepared.result.note_ids += _add_notes_locked(plan, prepared.notes)
    if project.surface == "frontdesk" and by == keeper(project):
        # §7.2 U3 — the coordinator is the sole acceptor on a front-desk plan,
        # so its write IS the acceptance. One extra call site, not a second
        # mechanism.
        plan.revision += 1
        prepared.result.revision = plan.revision
    _note(plan, by, verb, section=",".join(prepared.moved))


async def _apply_locked(
    project: "Project",
    ctx: "ModelContext",
    runloop: "ChangelogRunloop",
    plan: ProjectPlan,
    edits: Sequence[SectionEdit],
    *,
    by: str,
    verb: str = "write",
) -> _PreparedWrite:
    """§4.1, steps 1–7. **The entire concurrency mechanism**, prepared and
    committed in one go — the shape every caller but :func:`submit_review` wants.

    Mutates ``plan`` (history, shorthand notes, the front-desk bump) but does not
    save it: the caller owns the transaction and saves once.

    **Returns the ``_PreparedWrite``, not the bare ``WriteResult``**, and the
    extra member is ``body`` — the document as it reads once this write lands.
    :func:`resolve_note` needs it to stamp ``accepted_spec_digest`` off the text
    the user actually approved, and re-reading the room after the append would
    be a second answer to *"what did this write produce?"* — the mistake
    :func:`submit_review` already avoids by rendering from ``prepared.body``.
    Callers that want only the verdict take ``.result``, which is the same
    object they used to be handed.
    """
    prepared = _prepare_locked(project, ctx, plan, edits, by=by)
    if prepared.result.locked:
        # **M3 AC-3.3, filed here and not in each caller, because it is
        # unconditional.** Staleness has a ``file_conflict`` flag — it is a
        # refusal the caller may want raw. The spec lock has no such flag: a
        # refusal that files nothing loses the coordinator's text with no way to
        # get it back, and *"a refused write is a note, never a dead end"* (§4.3)
        # is the whole reason this path exists.
        #
        # **It does not raise**, and that is load-bearing rather than
        # incidental: ``absorb_plan_upload`` reaches the funnel through here and
        # its docstring's *"it never raises to the route"* is a hard constraint —
        # a raised agent action is swallowed and posted as a stack-shaped line
        # into ``user-communication`` (``api/action_executor.py:136-147``), which
        # loses the write AND shows the user a traceback. :func:`apply_edits`
        # raises on the way out, once the sidecar is saved.
        prepared.result.note_ids += _file_spec_lock_locked(
            project, plan, prepared.result.locked, by=by
        )
        _note(
            plan, by, "refused",
            detail=",".join(s.section for s in prepared.result.locked),
        )
        return prepared
    if not prepared.result.ok:
        _note(plan, by, "refused", detail=",".join(s.section for s in prepared.result.stale))
        return prepared
    if prepared.wrote:
        # The refusal, one statement above the point of no return. This single
        # insertion covers :func:`apply_edits`, :func:`absorb_plan_upload` and
        # :func:`resolve_note` — all three reach the append through here.
        _validate_notes_locked(plan, prepared.notes)
        await runloop.append_batch([prepared.spec])
        _finish_locked(project, plan, prepared, by=by, verb=verb)
    return prepared


def check_note_text(field: str, text: str) -> None:
    """Refuse note text the write path cannot encode. **Public on purpose.**

    A ``str`` holding an unpaired UTF-16 surrogate crosses the wire fine —
    ``json.loads(r'"aaa\\ud83d"')`` is a perfectly ordinary parse, and Python's
    decoder recombines *valid* escaped pairs into one astral code point, so any
    surrogate left standing in the ``str`` was unpaired on the client. It only
    fails later, at encode time, and it fails on **reads**: the sidecar is
    written with ``ensure_ascii=True`` and takes it, then every subsequent
    ``GET /plan`` and the review append serialise with ``ensure_ascii=False``
    and raise ``UnicodeEncodeError: … surrogates not allowed``. One such note
    therefore 500s the plan surface for every participant, permanently, until
    someone edits the file by hand.

    **Refused, not sanitised** — the choice, and why. ``str.toWellFormed()``'s
    Python equivalent (encode with ``errors="replace"`` and decode back) would
    take the note and rewrite the offending code unit to U+FFFD. That is a
    silent edit to *what the user quoted*, and here it is worse than silent: a
    ``quote`` is an **anchor**, matched back against the document text to place
    the highlight, so a quote with a substituted character no longer matches
    anything and the note renders as *"the line this quotes is no longer in
    this section"*. Sanitising would convert a loud 500 into a quiet
    unanchorable note. Refusing costs nothing real, because no well-formed text
    can produce this: a lone surrogate is not a character anyone typed, quoted
    or pasted.

    The test is ``.encode("utf-8")`` itself rather than a hand-rolled
    ``D800–DFFF`` scan, so the guard is **the same operation that crashes**: it
    cannot drift from the failure it exists to prevent, and it reports the
    position the traceback would have.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PlanInputError(
            f"note {field} holds an unpaired UTF-16 surrogate "
            f"(U+{ord(text[exc.start]):04X}) at position {exc.start}; it cannot "
            f"be encoded, and is refused rather than silently rewritten"
        ) from None


def _validate_notes_locked(plan: ProjectPlan, notes: Sequence[PlanNote]) -> None:
    """Raise :class:`PlanLimitError` if filing ``notes`` would breach
    ``MAX_OPEN_NOTES``, ``MAX_NOTE_CHARS`` or ``MAX_QUOTE_CHARS``, and
    :class:`PlanInputError` if any of their text is not encodable.
    **Mutates nothing** — not ``plan``, not ``notes``. Lock held.

    The single owner of the note-filing rules, asked at two different moments:
    by :func:`_add_notes_locked` as it files, and — by every caller that
    appends to the append-only changelog *first* — while the refusal is still
    free. An append cannot be taken back, so a cap tripped below one leaves the
    document written and broadcast to every participant while ``_save`` never
    runs and the caller is told *"nothing was written"*. The rule is not
    copied; only the moment of asking moves.

    The open-note budget is counted **once** and spent across the whole
    sequence: ``n`` notes onto a plan holding ``c`` open ones is refused when
    ``c + n > MAX_OPEN_NOTES``. That is the same arithmetic a per-note loop
    arrives at, and it is answerable before a single one is filed — which is
    the entire difference between a cap and a cap that can be enforced early.
    """
    if not notes:
        return
    open_count = sum(1 for n in plan.notes if n.status == "open")
    incoming = sum(1 for n in notes if n.status == "open")
    if open_count + incoming > MAX_OPEN_NOTES:
        raise PlanLimitError(
            f"plan has {open_count} open notes, limit is {MAX_OPEN_NOTES}"
        )
    for note in notes:
        size = len(note.comment) + len(note.proposal)
        if size > MAX_NOTE_CHARS:
            raise PlanLimitError(f"note is {size} chars, limit is {MAX_NOTE_CHARS}")
        if len(note.quote) > MAX_QUOTE_CHARS:
            raise PlanLimitError(
                f"note quote is {len(note.quote)} chars, limit is "
                f"{MAX_QUOTE_CHARS}"
            )
        # Encoding, on all three text fields rather than on ``quote`` alone.
        # ``quote`` is the field the finding named, but a lone surrogate in
        # ``comment`` or ``proposal`` reaches the identical crash by the
        # identical route — both are serialised by the same ``GET /plan`` and
        # rendered into the same review message — so guarding one and not its
        # two neighbours in this very loop would leave the hole open one flag
        # over. Verified, not assumed: ``--comment $'\ud83d'`` reproduces it.
        for field, text in (
            ("quote", note.quote),
            ("comment", note.comment),
            ("proposal", note.proposal),
        ):
            check_note_text(field, text)


def _add_notes_locked(plan: ProjectPlan, notes: Sequence[PlanNote]) -> list[str]:
    """Validate the whole sequence, **then** stamp ids and timestamps and file.

    All-or-nothing: an over-cap sequence leaves ``plan.notes`` untouched rather
    than half-filed. Filing one at a time and raising mid-loop is the mechanism
    by which *"the first shorthand was filed and then lost"* happens on a caller
    that then abandons the plan; validating first makes that partial state
    unreachable rather than merely harmless. Lock held.
    """
    _validate_notes_locked(plan, notes)
    ids: list[str] = []
    for note in notes:
        note.id = _gen_id("n", (n.id for n in plan.notes))
        note.at = note.at or _now()
        plan.notes.append(note)
        ids.append(note.id)
    return ids


def _find_note(plan: ProjectPlan, nid: str) -> PlanNote:
    for note in plan.notes:
        if note.id == nid:
            return note
    raise PlanNotFoundError(f"Note {nid!r} not found")


def _reply_parent(plan: ProjectPlan, reply_to: str) -> PlanNote | None:
    """The note ``--reply-to`` names, or ``None`` when it names nothing.

    **400, not 404**, for :func:`add_note`'s own stated reason: 403 and 404 read
    *stop*, 400 reads *re-form the call*, and re-forming is the truth here —
    there is a legal note, it is just pointed at an id that is not on this plan.
    A dangling ``reply_to`` is pure garbage: :func:`_thread`, ``_feedback`` and
    ``PlanNoteView.replies`` all match on it and none of them can ever match, so
    the note it produces is an orphan that looks like an answer.
    """
    if not reply_to:
        return None
    parent = next((n for n in plan.notes if n.id == reply_to), None)
    if parent is None:
        # Same rule, and this one had a shell command in it (AC-1.5). A remedy
        # a browser user cannot run is worse than no remedy: it reads as the
        # answer and is not one. The REMEDY survives the rewrite, because §2.6
        # is that a refusal owes one — it is just stated as an act instead of a
        # flag, and it now names the likeliest cause.
        raise PlanInputError(
            "That note is not on this plan — it may have been filed against a "
            "different project. Remove the reply if this is a new note rather "
            "than an answer."
        )
    return parent


def _reply_addressee(parent: PlanNote, by: str) -> str:
    """Who a reply to ``parent`` is *for* — **the other end of the thread**, not
    ``parent.by``.

    On a reply to your OWN note ``parent.by`` is you, so a plain
    ``parent.by`` fallback addresses the answer back at its author and it
    reaches nobody. The case is not exotic: the owner asking a question on their
    own note is the ordinary shape of a plan conversation, and ``PlanEditor``
    already dodges it client-side with ``note.by === 'user' ? note.to : note.by``.

    Stated once, here, so the two doors that file a reply — :func:`add_note`'s
    ``reply_to`` and :func:`submit_review`'s ``ask`` row — derive it identically
    and neither needs a client's help to get it right.

    ``""`` when there is no other end (a note addressed to nobody, replied to by
    its own author). The two doors read that emptiness DIFFERENTLY, on purpose:
    :func:`add_note` keeps it as *recorded, never sent* (``to = []``), because
    omitting ``--to`` on a reply is not a choice anyone makes; while
    :func:`submit_review`'s ask row falls through to ``parent.by``, because a
    tray reply that reaches nobody is the one outcome with no recovery. Neither
    is this function's call to make — it reports the other end or says there is
    none, and the caller decides what "none" costs it.
    """
    return parent.to if parent.by == by else parent.by


def _capture_base(body: str, section: str) -> str:
    """The section as it stands **right now**, captured at file time so a note
    has something honest to compare against later.

    ``base_section`` used to be *the proposal's base* and nothing else, so only
    a write-shaped note carried one. That made :func:`section_changed` useless
    for exactly the notes most likely to go stale: a comment or a question
    anchored to a section is read weeks later against text that has moved, and
    the reader has no way to know it moved. The note is not a write, so there is
    nothing to rebase — but *"the section you were talking about has changed"*
    is worth saying about any note that names one.

    ``""`` when the note names no section, and ``""`` when the slug does not
    resolve. The second case is :func:`section_missing`, a **different answer**,
    and capturing a placeholder here would shadow it — a note whose section was
    deleted must not read as a note whose section was edited.

    **Not retroactive.** Nothing backfills existing notes, and nothing should:
    a captured base asserts *this is the text its author saw*, and inventing one
    for a note filed before the capture existed asserts something false about a
    person.
    """
    if not section:
        return ""
    return _current_section(body, section) or ""


def _derive_quote(base: str, proposal: str) -> str:
    """The quote a sectioned note gets when its author supplied none.

    **The whole of why CLI-filed notes land at the bottom of the page.** The
    editor draws a note inline, against the line it argues about, only when the
    note carries a quote AND a section that still resolves
    (``PlanEditor.tsx:775``: ``!!n.quote && shownIds.has(n.section)``).
    Everything else falls to the page-bottom pane. ``--quote`` is optional and
    no worked example in the ``plan`` skill passes one, so *every* note filed
    from a terminal was unanchored by construction — not by anybody's
    carelessness. The browser composer was always fine, because
    ``planAnchor.tsx`` takes the quote off the user's selection; that asymmetry
    is what made this read like an agent-behaviour problem when it was a
    default.

    Two answers, because a note has two shapes:

    * **A proposal** anchors to :func:`first_changed_line` — the base line it
      actually changes, which is the line its reader wants to be looking at.
    * **Anything else** — a comment, a question, a deviation — anchors to the
      section's own heading. That is the honest answer to *"which line is this
      about"* from an author who named none: the note renders under the heading
      of the section it names, instead of at the bottom of the document.

    ``""`` for the ``_lede`` (no heading to point at) and ``""`` for a section
    that does not resolve — :func:`_capture_base` already returns ``""`` there,
    and both cases have their own answer in :func:`section_missing`, which an
    invented anchor would shadow.

    The result is cut from the stored body, which the server has already
    decoded, so it can never be the lone surrogate :func:`check_note_text`
    exists to refuse.
    """
    if not base:
        return ""
    line = first_changed_line(base, proposal) if proposal else heading_line(base)
    return quote_from_line(line)


def _recent_twin(
    plan: ProjectPlan,
    *,
    by: str,
    to: str,
    section: str,
    quote: str,
    comment: str,
    proposal: str,
    reply_to: str,
    now: str,
) -> PlanNote | None:
    """The twin of a note about to be filed, or ``None``.

    A double-clicked Save, a retried POST and a composer that fires on both
    ``submit`` and ``keydown`` all file the same note twice, and notes are never
    deleted — so the second one is on the plan forever, counts against
    ``MAX_OPEN_NOTES``, and has to be dismissed by hand by somebody who did not
    write it.

    **The key is the seven fields** ``(by, to, section, quote, comment,
    proposal, reply_to)``. ``quote`` is load-bearing rather than decoration:
    filing *"needs a number here"* against two different excerpts of one section
    is ordinary use, and a key without ``quote`` refuses the second at any
    window worth having.

    ``reply_to`` is load-bearing for the same reason and on the axis ``quote``
    cannot see. A reply inherits its parent's ``section`` and ``quote`` when
    those are omitted (:func:`add_note`), so two answers to two DIFFERENT
    questions hanging off one quoted line agree on every other field the moment
    the answer is a short one — *"Yes."*, *"Agreed."*, *"Done."* — and a key
    without ``reply_to`` swallows the second and returns the first note's id.
    The second question then reads unanswered while its author was told 200.
    That is the swallowing this window is sized to avoid, so the thread link is
    on the key; a genuine double submit carries the SAME ``reply_to`` and still
    dedupes.

    Scans **newest first** and stops at the first note outside the window, so
    the cost is the window and not the note list. ``at`` is an ISO-8601 string
    and a note filed before that stamp existed can carry ``""`` — an unparseable
    stamp is treated as **not a match** and skipped, never raised on, because
    refusing to file over a legacy note's bad timestamp is exactly the
    swallowing this window is sized to avoid.
    """
    try:
        cutoff = datetime.fromisoformat(now) - timedelta(seconds=NOTE_DEDUPE_SECONDS)
    except ValueError:      # pragma: no cover — `now` is always our own `_now()`
        return None
    for note in reversed(plan.notes):
        try:
            at = datetime.fromisoformat(note.at)
        except ValueError:
            continue
        if at < cutoff:
            break
        if (
            note.by == by
            and note.to == to
            and note.section == section
            and note.quote == quote
            and note.comment == comment
            and note.proposal == proposal
            and note.reply_to == reply_to
        ):
            return note
    return None


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


async def get_plan(
    project: "Project", ctx: "ModelContext"
) -> tuple[str, ProjectPlan]:
    """``(body, sidecar)``.

    ``404`` when there is no PLAN.md. A project with no sidecar yet gets an
    **in-memory** default — reading a pre-feature project's plan works and
    creates nothing (§3.1, AC-2.7).
    """
    async with _lock:
        return _read_body(project, ctx), _load(project, ctx)


async def get_plan_sidecar(project: "Project", ctx: "ModelContext") -> ProjectPlan:
    """The sidecar alone, for a caller that needs the counts and not the text.

    Same ``404`` contract as :func:`get_plan` — a project with no ``PLAN.md``
    has no plan — reached through :func:`_body_exists` rather than by reading
    and utf-8-decoding the document. On the project-list path that is the whole
    difference between one small JSON read per project and up to
    ``MAX_BODY_BYTES`` per project decoded to learn a boolean.
    """
    async with _lock:
        if not _body_exists(project, ctx):
            raise PlanNotFoundError(f"Project {project.id!r} has no {PLAN_FILE}")
        return _load(project, ctx)


async def plan_revisions(
    project: "Project", ctx: "ModelContext", runloop: "ChangelogRunloop"
) -> list[PlanRevision]:
    """One entry per ``FILE_CREATED``/``FILE_UPDATED`` on PLAN.md, newest first.

    A **changelog read** — no new storage. ``by`` is the changelog author, i.e.
    who wrote the bytes; ``proposed_by`` and ``note`` are joined from the
    sidecar's rounds through one map built once per call.

    **The revision label is walked forward, not stored per row.** A write that
    was not an accepted batch carries the revision that was current when it
    landed (§3.2), so the walk starts at 1 and steps only when it reaches a row a
    round produced — which is why the round stamps the body sha it wrote: it is
    the only key that tells those two kinds of row apart.

    There is no ``changes[]`` and this function does not grow one (§6.C.5).
    """
    async with _lock:
        plan = _load(project, ctx)
        by_sha = {r.sha: r for r in plan.rounds if r.sha}

        try:
            entries = runloop.get_entries_since(0)
        except FileNotFoundError:
            entries = []

        rows: list[PlanRevision] = []
        revision = 1
        for entry in entries:
            if entry.entry_type not in (
                ChangelogEntryType.FILE_CREATED,
                ChangelogEntryType.FILE_UPDATED,
            ):
                continue
            payload = entry.payload
            if getattr(payload, "chatroom_name", "") != PLAN_ROOM:
                continue
            if getattr(payload, "filename", "") != PLAN_FILE:
                continue
            round_ = by_sha.get(payload.sha256)
            if round_ is not None and round_.revision:
                revision = round_.revision
            rows.append(
                PlanRevision(
                    revision=revision,
                    sha=payload.sha256,
                    by=payload.from_participant_name,
                    at=entry.timestamp.isoformat() if entry.timestamp else "",
                    version=entry.version,
                    proposed_by=list(round_.applied_from) if round_ else [],
                    note=round_.batch_comment if round_ else "",
                )
            )
        rows.reverse()
        return rows


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


async def create_plan(
    project: "Project",
    ctx: "ModelContext",
    runloop: "ChangelogRunloop",
    *,
    body: str | None = None,
    by: str,
    title: str = "",
    seeded: bool = False,
) -> WriteResult:
    """Create PLAN.md from ``body`` or from the §5.1 seed template. ``409`` if it
    exists.

    Its ordinary caller is ``POST /projects``; it also exists for the manual case
    and for a deleted PLAN.md. **There is no state in which a plan is unwritten**
    — it is either the coordinator's v1 or the seed, from the moment the project
    exists.

    ``seeded`` stamps :attr:`ProjectPlan.seeded_at`, which is what puts the
    project into ``spec-ing``. **Only ``init_plan_sidecar`` passes it** (§3.5
    property 3): a ``plan create`` run by hand on a project that has been
    executing for weeks must not send it back to the gate.
    """
    async with _lock:
        if not may_write(project, by):
            raise PlanForbiddenError(f"@{by} may not create this plan")
        if _body_exists(project, ctx):
            raise PlanConflictError(f"{PLAN_FILE} already exists")

        text = body if body is not None else SEED_TEMPLATE.format(
            title=title or project.display_name or project.name
        )
        _check_body_size(text)

        plan = _load(project, ctx)
        shorthands, cleaned = extract_shorthand(text)
        note_ids = _add_notes_locked(
            plan,
            [
                PlanNote(id="", section=s.section, to=s.owner, by=by, comment=s.text)
                for s in shorthands
            ],
        )
        await runloop.append_batch([_file_spec(project, ctx, cleaned, by, created=True)])
        if seeded and not plan.seeded_at:
            plan.seeded_at = _now()
        _note(plan, by, "create")
        _save(project, ctx, plan)
        return WriteResult(
            ok=True, revision=plan.revision, sha=body_sha(cleaned), note_ids=note_ids
        )


async def apply_edits(
    project: "Project",
    edits: Sequence[SectionEdit],
    *,
    by: str,
    ctx: "ModelContext",
    runloop: "ChangelogRunloop",
    file_conflict: bool = False,
) -> WriteResult:
    """**The one funnel.** Every byte that reaches PLAN.md comes through here.

    See :func:`_apply_locked` for the seven steps. This wrapper owns the lock and
    the sidecar save, and — when ``file_conflict`` is set, which is what a
    keeper's ``plan update --section`` passes — turns a refusal into notes so it
    is never a dead end (§4.3).

    Does **not** bump ``revision`` on a regular project: that happens once, in
    :func:`submit_review`'s transaction (§2.5).

    Raises :class:`PlanConflictError` (409) on a stale write and
    :class:`PlanSpecLockedError` (403) on one M3 refused — **both after the
    save**, because both refusals have already filed the notes that are their
    only remedy.
    """
    async with _lock:
        plan = _load(project, ctx)
        result = (await _apply_locked(project, ctx, runloop, plan, edits, by=by)).result
        if result.stale and file_conflict:
            # ``+=``, not ``=``. On this path ``note_ids`` is empty, but the
            # spec-lock path above already filled it, and an assignment here
            # would be one refactor away from silently discarding the user's
            # copy of a refused write.
            result.note_ids += _file_conflict_locked(plan, result.stale, by=by)
        _save(project, ctx, plan)
        if result.locked:
            # **After the save.** The notes ``_apply_locked`` filed are the
            # entire content of the refusal, and an exception raised above
            # ``_save`` would take them with it.
            raise PlanSpecLockedError(
                _spec_lock_refusal(
                    project,
                    by=by,
                    note_ids=result.note_ids,
                    accepted=bool(plan.accepted_at),
                ),
                note_ids=result.note_ids,
                revision=result.revision,
            )
        if not result.ok:
            raise PlanConflictError(
                "section changed since this was written",
                stale=result.stale,
                revision=result.revision,
                note_ids=result.note_ids,
            )
        return result


async def absorb_plan_upload(
    project: "Project",
    uploaded: str,
    *,
    by: str,
    ctx: "ModelContext",
    runloop: "ChangelogRunloop",
) -> WriteResult:
    """The interception's model half (§6.1, B11).

    Compares the incoming body to the current one **section by section**:

    * keeper or owner — applies **only the sections that differ**; every other
      section is preserved byte-for-byte, which is protection derived from data
      the server already holds rather than from a client's claim.
    * any other agent — changes **nothing**; files one note ``to=user`` per
      differing section, that section's incoming text as the proposal and the
      current text as its ``base_section``.

    **It never raises to the route.** ``update_file PLAN.md`` must never return
    4xx from any identity, because a failed action is swallowed and posted as a
    raw failure line into ``user-communication``
    (``api/action_executor.py:136-147``) — which loses the write and shows the
    user a stack-shaped message (AC-3.3). A worker that uploads a rewritten
    PLAN.md gets a success, and its work becomes notes on the user's desk instead
    of an overwrite.
    """
    async with _lock:
        plan = _load(project, ctx)
        try:
            current = _read_body(project, ctx)
        except PlanNotFoundError:
            return WriteResult(ok=False, revision=plan.revision)

        # Shorthand is extracted on every path, including the notes path, so an
        # agent's `{@name: …}` is never carried into a proposal as literal text.
        shorthands, incoming = extract_shorthand(uploaded)
        changes = split_by_section(current, incoming)

        # **The cap is checked here, above the commit, and not below it.** Every
        # note this call will file is already constructible — the shorthands the
        # moment :func:`extract_shorthand` returns, the per-section proposals the
        # moment the split does — while on the keeper path they are filed *after*
        # ``_apply_locked`` has appended the ``FILE_UPDATED`` batch. A
        # ``PlanLimitError`` from down there escaped to ``_absorb_plan``, which
        # files a note reading *"The document is unchanged"* over a document that
        # had already changed and been broadcast. This is the only order in which
        # that sentence is true.
        #
        # The **objects** are validated, not a count of them, and they are the
        # same objects filed below. Reserving *n* slots and then filing *m*
        # notes reopens the hole silently; this path was safe only because its
        # arithmetic happened to be right, with nothing enforcing it.
        may_apply = may_write(project, by)
        shorthand_notes = [
            PlanNote(id="", section=s.section, to=s.owner, by=by, comment=s.text)
            for s in shorthands
        ]
        proposal_notes = [] if may_apply else [
            PlanNote(
                id="",
                section=slug,
                to=OWNER,
                by=by,
                comment=f"@{by} uploaded a {PLAN_FILE} that changes this section.",
                proposal=after,
                base_section=before,
            )
            for slug, before, after in changes
        ]
        _validate_notes_locked(plan, proposal_notes + shorthand_notes)

        if may_apply:
            # ``create=not before`` — :func:`split_by_section` returns ``""`` on
            # the side a section is absent from, so an empty ``before`` here is
            # a section the uploaded document ADDS. That is a real create and
            # says so, rather than leaning on the empty base the way the funnel
            # used to; a heading rename arrives as a delete plus a create (§2.3)
            # and needs the flag on its create half.
            edits = [
                SectionEdit(section=slug, text=after, base=before, create=not before)
                for slug, before, after in changes
            ]
            result = (await _apply_locked(
                project, ctx, runloop, plan, edits, by=by, verb="write"
            )).result
            # ``+=``, and keyed on ``stale`` rather than ``not ok``. A keeper
            # whose whole-file upload moves the spec after acceptance is refused
            # by M3, not by staleness: ``_apply_locked`` has already filed that
            # refusal's deviation notes, ``stale`` is empty, and an assignment
            # here would overwrite the ids with ``[]`` — losing, silently and
            # with a ``200`` on the way out, the only copy of what the
            # coordinator wanted to write.
            if result.stale:
                result.note_ids += _file_conflict_locked(plan, result.stale, by=by)
        else:
            note_ids = _add_notes_locked(plan, proposal_notes)
            _note(plan, by, "note", detail=f"upload absorbed as {len(note_ids)} notes")
            result = WriteResult(
                ok=True,
                revision=plan.revision,
                sha=body_sha(current),
                note_ids=note_ids,
                noop=True,
            )

        result.note_ids += _add_notes_locked(plan, shorthand_notes)
        _save(project, ctx, plan)
        return result


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def note_is_deviation(
    project: "Project", plan: ProjectPlan, note: PlanNote
) -> bool:
    """**Derived, on read.** *"The keeper wrote to the user after the user had
    accepted the plan"* — which is what a deviation always was.

    Before acceptance nothing has been agreed, so the same note is an ordinary
    comment, question or proposal. After it the document is a contract, so the
    coordinator writing to the user about it is by construction a report that
    the work and the contract have parted company. There is no second kind of
    message for it to be, which is why this was never independent information
    and never needed a stored bit: it is a timestamp comparison, and storing it
    only froze the comparison at write time.

    **The boundary is ``first_accepted_at``, and the two nearby fields are both
    wrong.** ``accepted_at`` re-stamps on every applied deviation, so it walks
    forward past notes that are definitionally post-acceptance. The project's
    ``phase`` is recomputed live from today's status, so a note's kind would
    change twice in its life without the note changing: five ordinary questions
    asked during ``spec-ing`` would become deviations the instant the plan was
    accepted, and finishing a project (``phase == "complete"``) would silently
    un-deviate its entire history.

    **Not gated on ``status``.** A note IS a deviation forever; the *alarm* is
    about unresolved ones, and ``_open_deviation`` filters ``open`` where that
    belongs. Folding it in here would render every closed deviation as an
    ordinary note and make *"the work went off-contract on the 30th, here is
    what we decided"* unreadable the moment it was decided.

    **Checked against every plan on a real installation**: 139 notes across 12
    projects, 45 stored deviations, and this predicate reproduces exactly those
    45 — no misses, no false positives.

    The one thing it cannot reproduce is a deviation filed by a coordinator who
    has since been replaced: ``keeper`` is today's, where the stored kind froze
    the historical one. That is the price of derivation and it is the right one
    — the alternative is a stored bit that goes stale in the other direction.

    The comparison is a STRING compare and that is deliberate, not a shortcut.
    Both sides come from :func:`_now`, so both are UTC ISO-8601 with the same
    offset spelling and the same field widths — lexicographic order and
    chronological order are the same order. Parsing here would cost a
    ``datetime`` per note per read and buy nothing.
    """
    return (
        bool(plan.first_accepted_at)
        and note.at > plan.first_accepted_at
        and note.to == OWNER
        and note.by == keeper(project)
    )


def note_kind(project: "Project", plan: ProjectPlan, note: PlanNote) -> str:
    """The word a surface prints for a note. One of :data:`NOTE_KINDS`.

    **The only producer.** Every client — the browser chip, ``plan notes``,
    the ``--kind`` filter — reads this off the wire; nothing recomputes it and
    nothing stores it.

    Precedence, and each step is a claim:

    1. ``deviation`` — the contract-level fact outranks the mechanical ones.
    2. ``proposal`` — ``applied_text`` is in the test on purpose. Both apply
       paths blank ``proposal`` into ``applied_text`` so an applied note stops
       re-offering Accept; without this clause an accepted proposal would print
       as a plain ``note`` and the record that anything was ever decided would
       go anonymous.
    3. ``reply`` — from ``reply_to``, which is the better signal than the kind
       it replaces ever was: the stored ``"question"`` was stamped in exactly
       one place (``submit_review``'s ``ask`` arm), so an agent's
       ``plan note --reply-to`` printed as a plain note despite being plainly a
       reply. Ranked BELOW ``proposal`` because the proposal is the actionable
       half and the row already says *"Answering @x's …"* on its own line.
    4. ``note`` — the absence of all three.
    """
    if note_is_deviation(project, plan, note):
        return "deviation"
    if note.proposal or note.applied_text or note.proposes_delete:
        return "proposal"
    if note.reply_to:
        return "reply"
    return "note"


# ---------------------------------------------------------------------------
# M2 — the note channel is a two-party channel: the user and the coordinator.
#
# | ``to``            | ``by`` must be              | phase |
# |-------------------|-----------------------------|-------|
# | ``user``          | the owner, or the keeper    | both  |
# | ``""`` (nobody)   | the owner, or the keeper    | both  |
# | the keeper        | the OWNER only              | both  |
# | any other agent   | — refused                   | both  |
#
# One shape in both phases. A specialist is reached over ``shared-context``
# (:func:`ensure_consultation_roster`), by the coordinator, on the user's behalf —
# never by a note. *"This one is for backend"* written inside a note is prose
# the coordinator reads and routes; it is not an addressing mode.
# ---------------------------------------------------------------------------


def note_addressee_allowed(project: "Project", *, by: str, to: str) -> bool:
    """The table above, as one predicate, for one caller — :func:`add_note`.

    ``to == ""`` is deliberately folded into the ``user`` row rather than left
    unrestricted. *"Recorded, never sent"* still writes a row onto the
    document's sidecar and still reads as a plan-editor artifact; leaving it
    open would hand a specialist back, under a different flag, exactly the
    proposal channel this milestone closes.

    Public because it is the rule, not a detail of the function that enforces
    it: a client deriving *"who may I address?"* must read this and not a
    second copy. The browser composer (M7) is written against it and offers the
    coordinator to the owner and nothing to anyone else.
    """
    keeper_name = keeper(project)
    if to in ("", OWNER):
        return by == OWNER or by == keeper_name
    if to == keeper_name:
        return by == OWNER
    return False


def _note_refusal(project: "Project", *, by: str, to: str) -> str:
    """Why it was refused **and the call that would have worked** (§ M2 AC-5).

    A bare *"forbidden"* aimed at a model is an invitation to bisect: try
    another ``--to``, then another ``--section``, then a shorter ``-m``. Each
    variant therefore ends by closing the search — either *"no ``--to`` value
    makes this note legal"* when the caller has no legal note to file at all,
    or the exact ``--to`` when it does.

    Three shapes, because there are three different remedies:

    * a **specialist** — no ``--to`` is legal; the route is the
      ``shared-context`` room in ``spec-ing`` and its workroom in
      ``executing``;
    * the **owner** naming a non-coordinator agent — the note is legal, the
      addressee is not, so the remedy names ``--to <keeper>``;
    * the **keeper** naming any agent — the remedy is ``plan consult``, not a
      corrected ``--to``, because keeper→keeper is refused too.
    """
    keeper_name = keeper(project)
    if by != OWNER and by != keeper_name:
        if project.phase == "spec-ing":
            return (
                f"@{by} cannot file a plan note. Answer the user in the "
                f"`shared-context` room — @{keeper_name} relays the user's "
                f"questions there and files your answers back to them. "
                f"No --to value makes this note legal."
            )
        return (
            f"@{by} cannot file a plan note. Raise it with @{keeper_name} in "
            f"your workroom; the coordinator files the deviation. "
            f"No --to value makes this note legal."
        )
    if by == OWNER:
        return (
            f"@{to} cannot be the addressee of a plan note. Notes go to "
            f"@{keeper_name}, who relays to the specialists in "
            f"`shared-context`. "
            f"Re-file with `--to {keeper_name}` and name @{to} in the note "
            f"text; naming them there is what gets it routed."
        )
    return (
        f"@{to} cannot be the addressee of a plan note, and neither can "
        f"@{keeper_name}. Ask @{to} in `shared-context` — "
        f"`clawmeets plan consult` seats them — and file what comes back as a "
        f"note to the user. "
        f"No --to value reaches an agent."
    )


async def add_note(
    project: "Project",
    ctx: "ModelContext",
    *,
    by: str,
    section: str = "",
    to: Sequence[str] = (),
    comment: str = "",
    proposal: str = "",
    base_section: str = "",
    quote: str = "",
    reply_to: str = "",
) -> list[str]:
    """File one note per addressee. **Never refused for staleness** — a note
    cannot clobber anything (§4.4).

    ``N`` addressees produce ``N`` **sibling** notes with independent ids and
    statuses, because one note with one status cannot represent *"two agreed, one
    objected"*. ``to`` omitted entirely produces **one** note with ``to == ""``:
    recorded, never sent, and it must not count toward
    :func:`open_notes_for_you`.

    **The one door.** Every note on a regular project comes through here, so
    M2's two-party rule is checked here and nowhere else —
    :func:`review_room_for` and :func:`ensure_consultation_roster` carry no
    permission check of their own, because a rule with two homes is a rule with
    two answers.

    **This claim used to name** :func:`submit_review` **too, and no longer
    does.** ``submit_review`` now carries exactly one permission check — the
    ownership guard on its ``accept``/``reject`` rows (:func:`may_decide`) —
    and it is not a second home for the *addressing* rule this paragraph is
    about. The two rules are different rules with one home each: who may be
    *told* something is decided here, who may *decide* is decided by
    :func:`may_decide`. What has NOT changed is that ``submit_review`` carries
    no two-party check: a batch's ``ask`` rows and ``note_ids`` sends are as
    ungated as they were, so a specialist whose write was refused can still
    speak. The notes
    the *system* files on a caller's behalf — :func:`_file_conflict_locked`'s
    refusal note, :func:`absorb_plan_upload`'s per-section proposals — go
    through :func:`_add_notes_locked` and are correctly not gated: a specialist
    whose write was refused must still be able to tell the user so, and that is
    the plan layer speaking, not the specialist.

    Refused per addressee, and refused as **400** rather than 403 via
    :class:`PlanInputError`. 403 reads *stop*; 400 reads *re-form the call*,
    and re-forming is the truth here — there is a legal act, it is just not
    this command. See :func:`_note_refusal` for the three remedies.

    **``reply_to`` fills in what the caller left out**, the same three fields
    :func:`submit_review`'s ``ask`` row derives: ``section`` and ``quote`` from
    the parent, and ``to`` from the other end of the thread. They are DEFAULTS
    and never overrides — AC-5.8 pins that a response may deliberately land on
    a different section, so anything passed explicitly wins. An unknown
    ``reply_to`` is a **400** (:func:`_reply_parent`), because the note it
    would file is an orphan wearing an answer's clothes.

    **A note that names a section and no quote gets one derived**
    (:func:`_derive_quote`), so it renders against a line instead of at the
    bottom of the page. Same rule as the base it is cut from: a default for an
    omitted field, never an override, and never backfilled onto a stored note.

    **What it does NOT do is close the parent.** ``submit_review`` does
    (AC-5.9) and this door deliberately does not: the plan editor draws only
    OPEN notes, so closing the question on reply erases the top of the exchange
    from the screen and leaves the answer standing alone — the thing anchoring
    the reply exists to prevent. The parent closes when the thread is done, via
    :func:`resolve_note`.
    """
    if not comment and not proposal:
        raise PlanInputError("A note needs a comment or a proposal")
    if proposal and not section:
        # NAMES THE THING, NOT THE FLAG (AC-1.5). This refusal is raised in the
        # model and reaches the browser composer, which has no `--section` and
        # never will — a user reading it went looking for a flag on a form.
        raise PlanInputError("A proposed replacement needs a section to replace")

    async with _lock:
        plan = _load(project, ctx)
        # THE ONE NEW READ (AC-1.9). `add_note` did not touch the document
        # before: a note cannot clobber anything, so it had no reason to. It has
        # one now — `_capture_base` needs the section as it stands to store what
        # this note's author was looking at. Inside the lock that is already
        # held and already ends in `_save`, so nothing is held meaningfully
        # longer; but it is a real change to this function's I/O shape and is
        # written down rather than left to be discovered.
        body = _read_body(project, ctx)

        # ---- the reply derivation, mirroring submit_review's `ask` row ------
        # DEFAULTS FOR OMITTED FIELDS, NEVER OVERRIDES. AC-5.8 pins that a
        # response MAY land on a different section from its parent and that
        # nothing validates `reply_to` against `section`; an explicit value
        # therefore always wins and nothing here can force the parent's.
        #
        # WHY IT IS HERE AT ALL: without it the two doors that file a reply
        # disagree. `submit_review`'s `ask` row derives `section`, `quote` and
        # `to` from the parent; this one took all three verbatim, so an agent
        # replying with `--reply-to` alone filed a note the editor's
        # `!!quote && shownIds.has(section)` partition cannot place — it fell
        # to the bottom-of-page list, addressed to nobody and delivered to
        # nobody. Parity, not a backstop.
        #
        # THE ADDRESSEE CHECK MOVED IN HERE WITH IT, and had to: it must run on
        # the EFFECTIVE addressee. A copy left above the lock would give the
        # two-party rule two homes, and a rule with two homes has two answers.
        parent = _reply_parent(plan, reply_to)
        if parent is not None:
            section = section or parent.section
            # The quote is inherited ONLY onto the parent's own section. A
            # quote is an excerpt OF a section; carried onto a different one it
            # anchors to a line that is not there, and `PlanBody` drops it into
            # the wrong section's loose bucket. `submit_review` needs no such
            # guard only because it always takes the parent's section too — so
            # both doors inherit the quote under exactly the same condition.
            if not quote and section == parent.section:
                quote = parent.quote
            if not to:
                # THE OTHER END OF THE THREAD, not `parent.by`. On a reply to
                # your OWN note `parent.by` is you, and `submit_review`'s
                # `row.to or parent.by` fallback would address the answer back
                # at its author — the case `PlanEditor` dodges client-side with
                # `note.by === 'user' ? note.to : note.by`. Stated once, here,
                # so this door needs no client's help to get it right.
                #
                # `--to ''` is UNTOUCHED and still means recorded-never-sent:
                # it arrives as `[""]`, which is truthy, so this never fires.
                # Only the OMITTED case changes, and on a reply "addressed to
                # nobody" is not a choice anyone makes — somebody asked.
                other_end = _reply_addressee(parent, by)
                to = [other_end] if other_end else []

        addressees = list(to) or [""]
        for addressee in addressees:
            if not note_addressee_allowed(project, by=by, to=addressee):
                raise PlanInputError(_note_refusal(project, by=by, to=addressee))

        # ---- the base, and the quote derived from it ----------------------
        # BOTH COMPUTED ONCE, ABOVE THE LOOP, and the order matters.
        # `_capture_base` used to be called inside the per-addressee loop,
        # where it returned the same value on every pass; it moves out because
        # `_derive_quote` needs it too and neither may be computed per
        # addressee. `_recent_twin` keys on `quote`, so a quote derived AFTER
        # the dedupe loop would not exist when the twin is looked up, and every
        # re-file of an identical note would mint a new id instead of returning
        # the existing one.
        #
        # AN EXPLICIT QUOTE ALWAYS WINS, exactly as an explicit `base_section`
        # does: derivation is the default for an omitted field and never an
        # override. `--section ''` is untouched and still means "about the
        # document as a whole" — `_capture_base` returns "" for it, so nothing
        # is derived and a deliberately unanchored note stays sayable.
        #
        # NOT RETROACTIVE, for `_capture_base`'s reason: an anchor asserts what
        # its author was looking at, and inventing one for a note filed before
        # this existed asserts something false about a person.
        #
        # Gated on `section`, not on `base` alone. `_capture_base` already
        # returns "" without one, but a caller may supply `base_section`
        # explicitly with no section — and a quote on a sectionless note is a
        # claim about a passage nobody can look up.
        base = base_section or _capture_base(body, section)
        if section and not quote:
            quote = _derive_quote(base, proposal)

        # ---- the double-submit net, PER ADDRESSEE --------------------------
        # It runs BELOW the two-party loop above, deliberately: a refused
        # addressee is refused whether or not the note would have deduped, and
        # a dedupe that ran first would turn a permission answer into a silent
        # 200. It runs INSIDE the existing lock, so two simultaneous duplicates
        # serialize for free and neither can read the plan before the other
        # appended.
        #
        # Per addressee, because N addressees produce N sibling notes and each
        # one is deduped against its own twin. `ids` is assembled POSITIONALLY —
        # a reused id lands in the slot its addressee occupies — so the returned
        # `note_ids` still lines up with `to`, which is the whole of the wire
        # contract here.
        #
        # A hit returns 200 with the EXISTING id, not a 400. The outcome the
        # caller wanted is true — there is a note on the plan saying that — and
        # a 400 carrying a bare English string cannot be told apart from a
        # validation failure, so the composer would render an error for an
        # outcome that was fine.
        now = _now()
        ids: list[str] = []
        fresh: list[PlanNote] = []
        slots: list[int] = []
        for addressee in addressees:
            twin = _recent_twin(
                plan,
                by=by,
                to=addressee,
                section=section,
                quote=quote,
                comment=comment,
                proposal=proposal,
                reply_to=reply_to,
                now=now,
            )
            if twin is not None:
                ids.append(twin.id)
                continue
            slots.append(len(ids))
            ids.append("")
            fresh.append(
                PlanNote(
                    id="",
                    section=section,
                    to=addressee,
                    by=by,
                    at=now,
                    comment=comment,
                    proposal=proposal,
                    # An explicitly supplied base still WINS — a proposal
                    # carries the base its author actually read, and capturing
                    # over it would silently rebase the write. Capture is the
                    # DEFAULT for an omitted one, which is every comment and
                    # every question. Resolved once, above the dedupe loop.
                    base_section=base,
                    quote=quote,
                    reply_to=reply_to,
                    revision=plan.revision,
                )
            )

        # A fully-deduped request writes NOTHING — no note, no activity row, no
        # save. An activity row for a note that already existed is a lie, and
        # `history` is the surface a user reads to find out what happened.
        if fresh:
            filed = _add_notes_locked(plan, fresh)
            for slot, nid in zip(slots, filed):
                ids[slot] = nid
            _note(plan, by, "note", section=section, detail=",".join(filed))
            _save(project, ctx, plan)
        return ids


def _file_conflict_locked(
    plan: ProjectPlan, stale: Sequence[StaleSection], *, by: str
) -> list[str]:
    """§4.3 — one ``conflict=true`` note per stale section, ``to=user``.

    A refused write blocks an agent and only the user can clear it, so the note
    carries what the agent wanted (``proposal``) beside what it read
    (``base_section``): everything ``plan conflicts`` needs to print the exact
    ``resolve --apply --force`` that closes it.
    """
    return _add_notes_locked(
        plan,
        [
            PlanNote(
                id="",
                section=s.section,
                to=OWNER,
                by=by,
                conflict=True,
                comment=(
                    f"@{by} could not write `{s.section}` — it changed since this "
                    f"was written."
                ),
                proposal=s.text,
                base_section=s.base,
                revision=plan.revision,
            )
            for s in stale
        ],
    )


def _pair_renamed_rows(
    body: str, cleaned: str, changes: list[tuple[str, str, str]]
) -> list[tuple[str, str, str]]:
    """Collapse a retitle's two rows into the one edit that was actually made.

    :func:`split_by_section` reports a rename as a deletion plus an addition,
    because slugs come from heading text and retitling re-slugs. Left alone that
    reaches the user as two notes — *"remove Goal"* with nothing to accept, and
    *"add Objective"* with no base to diff against — and neither is the decision
    in front of them. One note, one diff, one ``Accept``.

    **Anchored on the OLD slug, and that is what makes it applicable.** The new
    heading does not exist in the stored document — the write was refused — so a
    note naming it would report ``section_missing`` on the desk and, on
    ``Accept``, reach :func:`_splice` with a slug that resolves to nothing and be
    refused as stale. The old slug is the one still there, and
    :func:`replace_section` splices the new full text (its new heading included)
    over that extent, so accepting the note performs the rename with no new
    machinery. The diff the user reads is ``- ## Goal`` / ``+ ## Objective``,
    which is the edit.

    Order is preserved by rebuilding in place rather than by appending the merged
    row, so the notes a refusal files stay in document order.

    Returns ``changes`` untouched when :func:`rename_pair` declines to name a
    pair — the conservative answer there is two rows, which is what this had.
    """
    pair = rename_pair(body, cleaned)
    if pair is None:
        return changes
    old_slug, new_slug = pair
    by_slug = {slug: (before, after) for slug, before, after in changes}
    if old_slug not in by_slug or new_slug not in by_slug:
        # The rename is real but one half moved no text of its own, so the split
        # never emitted a row for it. Nothing to merge, and nothing to lose.
        return changes
    merged = (old_slug, by_slug[old_slug][0], by_slug[new_slug][1])
    return [
        merged if slug == old_slug else (slug, before, after)
        for slug, before, after in changes
        if slug != new_slug
    ]


def _row_deletes(s: StaleSection) -> bool:
    """**A row with a real base and no text is a DELETE, not an absence**, and
    the rule has ONE HOME because two surfaces read it and they came apart.

    :func:`_file_spec_lock_locked` reads it as ``proposes_delete``, which
    :func:`_note_view` folds into ``has_proposal`` — that is what puts ``Accept``
    on the desk. :func:`_spec_lock_comment` reads it to decide what to tell the
    user ``Accept`` will do. For two commits those were different questions
    asked of different values: the flag came from ``base`` and the sentence from
    ``text``, so a refused delete rendered an ``Accept`` button beside a sentence
    reading *"There is no replacement text to accept."* A user read the sentence,
    clicked the button, and removed a 17,705-character section from a plan with
    no confirmation and no undo.

    ``s.base`` is required rather than assumed so a row naming a section that did
    not exist before the write (the create half of a retitle, ``base == ""``) is
    never read as proposing to remove something that was never there.
    """
    return bool(s.base) and not s.text


def _spec_lock_comment(
    *, by: str, section: str, text: str, deletes: bool, accepted: bool
) -> str:
    """What the refused write is, said to the user who has to decide it.

    **It promises ``Accept`` exactly when ``Accept`` is on screen**, and that is
    the whole of this function. It gets that right by asking
    :func:`_row_deletes` — the same predicate that turns the button on — instead
    of asking ``text`` and hoping the two agree. They did not: the string used to
    end *"Accept this to make the change."* unconditionally, which was false when
    ``text`` was empty; the fix at the time made the empty arm say there was
    nothing to accept, and the fix after THAT gave a refused delete its
    ``Accept`` back without revisiting the sentence. Two commits, two half-truths,
    one destroyed section. The flag is now the input, so the pair cannot drift.

    **THREE ARMS, because an empty ``text`` is two different facts.** With a real
    ``base`` it is a DELETE — a proposal whose replacement text happens to be
    empty, which the desk renders as a diff of the section against nothing and
    offers to apply. With no base it is an ABSENCE — nothing to apply, and the
    desk offers neither ``Accept`` nor ``Reject``. The middle arm is the
    destructive one and is the only sentence on this surface that describes an
    irreversible act, so it says so in those words.

    An empty ``text`` is not a rare shape either way. It is what
    :func:`split_by_section` reports for every slug present before the write and
    absent after it: an explicit ``plan update --delete``, and the delete half of
    a **retitle** (a rename re-slugs, so it is a delete plus a create). Both are
    operations this system offers on purpose. A third producer — *a replacement
    that does not re-emit its own heading* — used to reach here too and does not
    any more: :func:`~clawmeets.models.plan_markdown.drops_heading` and
    :func:`~clawmeets.models.plan_markdown.relevels_heading` refuse it at input
    validation, which is where the incident above actually began.

    ``accepted`` splits the lede for the same reason
    :func:`_spec_lock_refusal` splits its own: the lock now also engages on an
    UNACCEPTED plan the user has reviewed once, and this sentence is read by the
    user on their desk. *"The plan is accepted"* on a plan they have not
    accepted is the note contradicting the Accept button sitting next to it.
    """
    state = "the plan is accepted" if accepted else "you have reviewed this plan"
    lede = f"@{by} could not change `{section}` — {state}"
    if text:
        return f"{lede} and this moves what it says. Accept this to make the change."
    if deletes:
        return (
            f"{lede}, and this write REMOVES the section. Accept it and "
            f"`{section}` is DELETED from the plan — there is no replacement "
            f"text, the diff below is the whole section against nothing, and "
            f"nothing puts it back. If you want it kept, reply to @{by} "
            f"instead, or make the change yourself."
        )
    return (
        f"{lede}, and this write REMOVES the section, which moves what it says. "
        f"There is no replacement text to accept: reply to @{by}, or make the "
        f"change yourself."
    )


def _file_spec_lock_locked(
    project: "Project",
    plan: ProjectPlan,
    locked: Sequence[StaleSection],
    *,
    by: str,
) -> list[str]:
    """M3 AC-3.3/AC-3.4 — one **deviation** note per section a refused
    executing-phase keeper write would have moved, ``to=user``, carrying the
    attempted text as its ``proposal`` so the user accepts it in one click.

    **Shaped on :func:`_file_conflict_locked`, deliberately not folded into
    it.** The two differ in ``conflict`` and in the sentence they write, and one
    function behind a flag would put *"was this refused for staleness or for the
    spec lock?"* inside a body whose whole job is to render one sentence. ``base``
    is the section's current text on both, and ``text`` is what the writer
    wanted — which is exactly what :class:`StaleSection` says it is for.

    **Not a surrogate, and not :func:`add_note`.** Two separate reasons, both
    worth having written down:

    * *Not ``add_note``*: this runs with ``_lock`` held, and ``add_note`` takes
      it. M2 already routes the notes the **system** files on a caller's behalf
      through :func:`_add_notes_locked` and says so in ``add_note``'s own
      docstring; this is the third such caller, not a new door.
    * *Not a surrogate like :func:`file_upload_refusal_note`*: that exists
      because the plan layer was speaking on a **worker's** behalf, which M2
      made illegal. Here the note is filed by the coordinator about the
      coordinator's own refused write, and ``by == keeper, to == user`` is legal
      in both phases under M2's AC-2.3. Nothing is being spoken on anyone's
      behalf.

    **Nothing here says "deviation", and that is the point.** The word is
    derived on read by :func:`note_kind`, off :func:`note_is_deviation`, which
    is a timestamp comparison against ``first_accepted_at``. Writing the word
    here would be a second copy of that rule, free to drift the day the rule
    moves — **and it has now moved.** This branch used to imply the word: a
    spec-lock refusal could only happen on an accepted plan, so a note filed
    here was a deviation by construction. Since the lock also engages once the
    user has merely REVIEWED the draft (:func:`_spec_is_locked`), a note filed
    here on an unaccepted plan is an ordinary proposal — correctly, because
    nothing has been agreed for the work to have parted company with. That is
    exactly the drift derivation was chosen to absorb, and it cost this function
    no code.
    """
    # **ONLY THE SECTIONS THAT ACTUALLY MOVE THE SPEC, AND THE ASYMMETRY IS THE
    # POINT.** The refusal upstream is decided on the UNFILTERED list — that
    # decision belongs to :func:`_prepare_locked`, and filtering there could let
    # a write the lock means to refuse fall through and land. What is filtered
    # here is only which sections get put in front of the user as a note.
    #
    # The two halves came apart because they ask different questions. The lock's
    # test is `spec_digest(cleaned) != spec_digest(body)` — normalized, and about
    # the DOCUMENT. The rows come from :func:`split_by_section`, which is a raw
    # byte comparison per section. So one section genuinely moving the spec
    # dragged every other byte-changed section into a `to=user` note with it,
    # including sections normalization folds away completely: an owner was handed
    # a diff whose entire content was `- [ ]` becoming `- [x]` and asked to accept
    # it — on the same document whose prompt promises that ticking a box is not a
    # spec change. :func:`normalize_spec_text` IS the digest's own key, so this
    # asks the lock's question rather than a second copy of it.
    #
    # **AND IF THAT WOULD LEAVE NOTHING, EVERYTHING STAYS.** This function's
    # contract is that a refusal files the coordinator's text somewhere it can be
    # recovered, and :func:`_spec_lock_refusal` names the ids it filed; zero notes
    # is the dead end AC-3.3 exists to prevent. It should be unreachable — a moved
    # digest means some section's normalized text moved — but the fallback costs
    # one comparison and removes the need to prove that.
    moved = [
        s for s in locked
        if normalize_spec_text(s.base) != normalize_spec_text(s.text)
    ]
    locked = moved or locked

    return _add_notes_locked(
        plan,
        [
            PlanNote(
                id="",
                section=s.section,
                to=OWNER,
                by=by,
                comment=(
                    _spec_lock_comment(
                        by=by,
                        section=s.section,
                        text=s.text,
                        # THE SAME PREDICATE THE FLAG BELOW READS, and passing
                        # it rather than letting the sentence re-derive it from
                        # `text` is the entire repair: the button and the
                        # sentence that describes the button now cannot answer
                        # differently. See `_row_deletes`.
                        deletes=_row_deletes(s),
                        accepted=bool(plan.accepted_at),
                    )
                ),
                proposal=s.text,
                base_section=s.base,
                proposes_delete=_row_deletes(s),
                revision=plan.revision,
            )
            for s in locked
        ],
    )


async def file_conflict_note(
    project: "Project",
    ctx: "ModelContext",
    *,
    by: str,
    sections: Sequence[StaleSection],
) -> list[str]:
    """Public entry to §4.3's conflict note, for a caller that refused a write
    outside :func:`apply_edits`."""
    async with _lock:
        plan = _load(project, ctx)
        ids = _file_conflict_locked(plan, sections, by=by)
        _note(plan, by, "refused", detail=",".join(ids))
        _save(project, ctx, plan)
        return ids


async def file_upload_refusal_note(
    project: "Project",
    ctx: "ModelContext",
    *,
    by: str,
    comment: str,
) -> list[str]:
    """**The plan layer telling the user a `PLAN.md` upload could not be
    absorbed**, on behalf of an agent that has no note channel of its own.

    The sibling of :func:`file_conflict_note`, and it exists for the same
    reason. M2 made ``add_note`` two-party, and ``files.py``'s upload fallback
    was filing its *"@x uploaded a PLAN.md that could not be applied"* note
    **as the uploading agent**, ``to=user`` — legal before M2, refused after it,
    and swallowed by that call site's deliberate ``except Exception: pass``. The
    user would simply have stopped being told, with a ``200`` on the way out.

    So the caller moved, not the rule. This is the plan layer speaking, not the
    specialist: a specialist whose write the system refused must still be able
    to have the system say so, exactly as :func:`_file_conflict_locked` already
    does for a stale write. ``by`` is recorded as the note's author because the
    user needs to know **whose** upload it was; it is not a channel that agent
    can reach on its own.
    """
    async with _lock:
        plan = _load(project, ctx)
        ids = _add_notes_locked(
            plan,
            [PlanNote(id="", section="", to=OWNER, by=by,
                      comment=comment, revision=plan.revision)],
        )
        _note(plan, by, "refused", detail=",".join(ids))
        _save(project, ctx, plan)
        return ids


async def resolve_note(
    project: "Project",
    ctx: "ModelContext",
    runloop: "ChangelogRunloop",
    *,
    nid: str,
    action: str,
    by: str,
    reason: str = "",
    force: bool = False,
) -> WriteResult:
    """The status ladder, **server-side, always** (constraint C1).

    ``apply`` is the CLI equivalent of an ``accept`` row: one
    :class:`SectionEdit` — the note's proposal with the section's **current**
    text as ``base`` — through the funnel; it sets ``applied``, **collapses the
    proposal** and keeps the comment and ``base_section`` so the thread reads.
    ``reject`` keeps the proposal so the author can revise.

    ``409`` on a comment-only note, on a section that no longer exists (never a
    fallback placement), and on ``apply`` when the section changed and ``force``
    is unset.

    **``apply`` and ``reject`` are the OWNER's alone** (M2 AC-1) — they are
    decisions about the document, and this project's whole shape is that the
    coordinator drafts and keeps while the user alone decides. It was
    ``may_write`` here, which let the coordinator dispose of the notes it wrote
    itself. The browser closed this route already: every decision stages into
    the owner-only tray and lands through ``submit_review``. This closes the CLI
    one.

    ``answered`` and ``dismiss`` stay reports rather than decisions, with **one
    shape carved out**: the keeper may not close its own note to the user before
    that note has been sent (:func:`_refuse_unsent_self_close`). A proposal the
    user has never seen is not a thread anybody can report as finished. Beyond
    that carve-out they are unchanged, and a note's own addressee may still
    close its own. Under the
    two-party rule the only agent that can ever be an addressee is the
    coordinator, so on any note filed from here on these two verbs are exercised
    by the coordinator and the owner and by nobody else. A note filed **before**
    M2 and addressed to a specialist stays closable by that specialist, which is
    the point of not gating a report.

    Applying after acceptance is never blocked. It makes
    ``changed_since_acceptance`` true and — because it resolves a note — may be
    what *releases* the execution gate (§7.4).
    """
    if action not in RESOLVE_ACTIONS:
        raise PlanInputError(f"Unknown action {action!r}")

    async with _lock:
        plan = _load(project, ctx)
        note = _find_note(plan, nid)

        if action in ("apply", "reject"):
            _refuse_non_owner_decision(by, action)
            # **A DELETE IS A PROPOSAL WHOSE TEXT IS EMPTY**, so the question is
            # "does this note propose anything?" and not "is there a string?".
            # Asking the string alone refused the one decision the user could
            # not make any other way: the note offering `Accept` on the desk
            # would 409 on the click, which is worse than withholding it.
            if not note.proposal and not note.proposes_delete:
                raise PlanConflictError(
                    f"Note {nid} carries no proposal; `{action}` needs one"
                )
        elif note.to and note.to != by and not may_write(project, by):
            raise PlanForbiddenError(f"Note {nid} is not addressed to @{by}")

        if action != "apply":
            _refuse_go_note_close(note, action)
            _refuse_unsent_self_close(plan, note, by=by)

        result = WriteResult(ok=True, revision=plan.revision)

        if action == "apply":
            body = _read_body(project, ctx)
            current = _current_section(body, note.section)
            # **A PROPOSAL FOR A SECTION THAT IS NOT THERE IS A CREATE EXACTLY
            # WHEN THE NOTE WAS WRITTEN THAT WAY**, and the note says which,
            # because `_capture_base` ran when it was FILED: `base_section == ""`
            # means *there was nothing there then*.
            #
            # That is not a corner case; it is the shape the spec lock files
            # whenever a coordinator's refused write RESTORES a deleted section —
            # the write it was not allowed to make, kept verbatim for the user to
            # accept. Refusing it made the one repair the document needed the one
            # thing neither door could do: on the project this was written for,
            # fifteen such notes were filed across four review rounds and every
            # single one of them 409'd on the click.
            #
            # A NON-EMPTY base is still exactly the case this branch was written
            # for — the section existed, it was deleted underneath the proposal,
            # and appending it beside whatever replaced it is the fallback
            # placement §5.5 says there is never one of. That refusal is
            # unchanged, and `_row_creates` carries the same rule for the tray.
            creating = current is None and not note.base_section
            if current is None and not creating:
                raise PlanConflictError(_vanished_refusal([note.section]))
            # **THE MODEL'S ONE STALENESS PREDICATE, NOT A SECOND COPY.** This
            # arm used to compare the strings itself, which is how the CLI and
            # the browser came to disagree: `_draft_view` and `_note_view` ask
            # `section_changed`, this asked raw bytes, and the two answers
            # diverged the instant `extent_key` normalized one of them.
            #
            # The `current is None` branch ABOVE stays where it is and stays
            # explicit. `section_changed` answers False for a vanished section
            # by design — that is `section_missing`, a different question — so
            # folding the two would turn "the proposal cannot be applied" into a
            # silent fallback placement at the end of the document. It answers
            # False on the create path too, for the same reason and harmlessly:
            # an empty `base_section` has nothing to compare.
            #
            # `current` is still read raw, for `SectionEdit.base` below and for
            # the `StaleSection` payload: the refusal has to show the user the
            # bytes the document actually holds, never a normalized key.
            if section_changed(body, note) and not force:
                raise PlanConflictError(
                    f"Section {note.section!r} changed since @{note.by} wrote this",
                    stale=[
                        StaleSection(
                            section=note.section,
                            base=note.base_section,
                            current=current,
                            text=note.proposal,
                        )
                    ],
                    revision=plan.revision,
                )
            prepared = await _apply_locked(
                project,
                ctx,
                runloop,
                plan,
                [SectionEdit(
                    section=note.section,
                    text=note.proposal,
                    base=current or "",
                    create=creating,
                )],
                by=by,
                verb="apply",
            )
            result = prepared.result
            # IMMEDIATELY BEFORE THE CLEAR (AC-1.3), and this ordering is the
            # whole item: set it anywhere else and the field is never populated,
            # `display_diff` is empty forever, and the change ships as a silent
            # no-op that passes a careless review.
            note.applied_text = note.proposal
            note.proposal = ""
            if (note.bootstrap or plan.accepted_at) and result.ok:
                # THE CLI HALF OF ACCEPTANCE. `clawmeets plan resolve <id>
                # --apply --as-user` is what replaced `plan approve`, so this
                # arm and `submit_review`'s accept arm are the only two ways a
                # plan is ever accepted — and both stamp through one helper, off
                # `prepared.body`, so the digest is of the text that was
                # approved and not of the text that preceded it.
                #
                # **`or plan.accepted_at` — RE-STAMPING, and it is the same act.**
                # The bootstrap arm is the FIRST acceptance; this one is every
                # one after it. Applying a proposal is the owner putting their
                # name to a piece of text, and on an already-accepted plan that
                # is precisely a fresh approval of the document it produces.
                # Without it `accepted_revision` would be frozen at whatever
                # revision the go-note closed on, and `changed_since_acceptance`
                # would go true on the owner's own applied deviation and stay
                # true for the life of the project — an amber warning, in the
                # coordinator's prompt every turn, about changes the owner
                # themselves signed off on.
                #
                # **No `by == OWNER` conjunct, and that is deliberate.** The
                # guard at the top of this function is `may_decide`, so `by` is
                # the owner by the time control reaches here; restating the
                # comparison would give the ownership rule the second home
                # :func:`may_decide` exists to prevent.
                _stamp_acceptance_locked(plan, body=prepared.body, by=by)

        _close_note(note, status=_STATUS_FOR[action], by=by, reason=reason)
        _note(plan, by, "resolve", section=note.section, detail=f"{nid} {action}")
        _auto_close_rounds(plan)
        _save(project, ctx, plan)
        return result


_STATUS_FOR = {
    "apply": "applied",
    "reject": "rejected",
    "answered": "answered",
    "dismiss": "dismissed",
}


def _close_note(note: PlanNote, *, status: str, by: str, reason: str = "") -> None:
    note.status = status
    note.resolved_by = by
    note.resolved_at = _now()
    note.resolution = reason


def open_go_note(plan: ProjectPlan) -> PlanNote | None:
    """The open go-note, if this plan has one. At most one ever exists.

    Public because three different questions are answered with it — may this
    write drop ``## Approval``, does this reply close its parent, and is there
    already a go-note to seed — and a predicate with three copies has three
    answers.
    """
    return next((n for n in plan.notes if n.bootstrap and n.status == "open"), None)


def note_was_sent(plan: ProjectPlan, note: PlanNote) -> bool:
    """Has this note ever gone out in a review batch?

    ``rounds[*].sends`` is keyed by note id and is additive-only, so a hit there
    is *"the user's desk has seen this"* and its absence is *"this exists only
    on the server"*. ``note.round`` answers the same question and is cheaper,
    but it is a single slot that the LAST send overwrites; the sends map is the
    record. Both are checked, so a note stamped by a path that predates the map
    still reads as sent.
    """
    if note.round:
        return True
    return any(note.id in r.sends for r in plan.rounds)


def _refuse_unsent_self_close(plan: ProjectPlan, note: PlanNote, *, by: str) -> None:
    """**A keeper may not dispose of its own proposal to the user before the
    user has seen it.** Raise, or return.

    Narrow, and exactly one shape: ``note.by == by``, ``note.to == OWNER``, and
    the note has never appeared in a sent round. ``by`` is the KEEPER whenever
    this fires, but that is derived rather than checked — :func:`add_note` is
    the only door onto ``plan.notes`` and it admits the owner and the keeper
    alone, so a non-owner author of a ``to=user`` note is the keeper by
    construction. Taking a ``project`` here just to restate that would give the
    authorship rule a second home.

    **The hole this closes, measured.** On ``chuswine-geo-b2b`` the coordinator
    filed two proposal notes to the user at 21:37:25 and 21:37:29, dismissed
    both itself at 21:37:44 and 21:37:48, and the review batch that would have
    carried them did not go out until 21:39:57. The user never saw either note.
    Every individual step was legal — ``--dismiss`` is classed as a REPORT
    rather than a decision, and correctly so — but the sequence is the same
    authority leak :func:`_refuse_non_owner_decision` exists to stop, arriving
    through a different door. A proposal the user was never shown is not a
    thread the keeper can report as finished; there is nothing to report.

    **``note_was_sent`` is what keeps this from becoming a decision gate**, and
    the qualifier is the whole design. Once the user HAS seen a note, the keeper
    closing a finished thread is a report again — which is what ``--answered``
    is for, and what the existing rule deliberately leaves open. Nothing else
    moves: the owner closes anything, a specialist closes its own, and the
    keeper closes any note addressed TO it.

    The remedy is named because the refusal is otherwise a dead end: an unsent
    note is still in the tray, so the way to withdraw it is to send it and let
    the user decide, or to file the correction as a further note in the same
    batch.
    """
    if by == OWNER or note.by != by or note.to != OWNER:
        return
    if note_was_sent(plan, note):
        return
    raise PlanForbiddenError(
        f"@{by} may not close {note.id} — you wrote it, it is addressed to the "
        f"user, and it has not been sent yet, so the user has never seen it. "
        f"Closing your own unsent proposal decides it on their behalf. Send the "
        f"batch (`clawmeets plan review`) and let them decide, or file the "
        f"correction as another note in the same batch."
    )


def _refuse_go_note_close(note: PlanNote, verb: str) -> None:
    """**The go-note closes on ``apply`` and on nothing else.**

    The sibling of the reply exemption in :func:`submit_review`, and it exists
    for the identical reason. The execution gate counts notes that are OPEN, so
    *every* way of closing the go-note releases it — and only one of them means
    the user said yes. ``reject`` and ``dismiss`` would leave the plan
    unaccepted **and** unblocked: the coordinator would start executing the very
    plan the user had just turned down, which is the precise outcome the go-note
    exists to make impossible.

    So the refusal is not a policy about what users may feel — it is the gate
    keeping its shape. Rejecting a plan is a real thing to want, and the act
    that expresses it is a *reply*: it reaches the coordinator, it asks for the
    revision, and it leaves the gate up while that happens. What has no
    coherent meaning is closing the gate to signal disapproval.

    Called from the two paths that can close a note by id — :func:`resolve_note`
    and :func:`submit_review` — and from nowhere else, because a predicate with
    two copies has two answers.
    """
    if note.bootstrap:
        raise PlanConflictError(
            f"The go-note is the plan's execution gate, not a proposal to "
            f"{verb} — accepting it is the only act that closes it. Reply to "
            f"it to ask for changes: the coordinator gets the question and the "
            f"gate stays up while the plan is revised."
        )


def _stamp_user_review_locked(plan: ProjectPlan, *, by: str) -> None:
    """Latch :attr:`ProjectPlan.first_user_review_at` — the spec lock's start
    line before acceptance. **Idempotent, and it never moves once set.**

    The sibling of :func:`_stamp_acceptance_locked` and stamped beside it, for
    the identical ordering reason (see the comment at that callsite): the
    projection the coordinator's next turn reads is built a few lines below.

    ``by != OWNER`` is a no-op rather than a refusal, so the coordinator's own
    outbound review batches — which are the majority of them — pass straight
    through. *Who opened the round* is the whole question; a coordinator asking
    the specialists something is not the user reviewing the draft.
    """
    if by != OWNER or plan.first_user_review_at:
        return
    plan.first_user_review_at = _now()


def _stamp_acceptance_locked(plan: ProjectPlan, *, body: str, by: str) -> None:
    """Record the user's acceptance. **The only writer of the five stamps.**

    Called from the two places a proposal can be applied — the tray's Accept
    (:func:`submit_review`) and ``plan resolve --apply`` (:func:`resolve_note`)
    — and from nowhere else. There is no approve route to be a third, and since
    ``revoke_plan`` was deleted there is nothing that clears these fields
    either: they only ever move forward.

    **NOT ONCE PER PLAN — ONCE PER OWNER APPLY.** It used to fire on the
    bootstrap go-note alone, which was right while a revoke could re-file one.
    With acceptance one-way that reading froze ``accepted_revision`` at whatever
    revision the go-note closed on, and the owner's own applied deviation — *the
    designed way an accepted spec changes* — moved the document out from under
    the digest with no way back. ``changed_since_acceptance`` went true on the
    owner's own decision and stayed true forever, in the coordinator's prompt
    every turn.

    So ``accepted_at`` means **when the owner last put their name to this
    text**, and ``changed_since_acceptance`` means what it says: the document
    moved WITHOUT them. On this plan's shapes that leaves exactly one producer
    of a true reading — the owner's own direct ``plan update`` — because a
    keeper's spec move is refused and filed as a deviation rather than landed
    (:func:`_file_spec_lock_locked`), and applying that deviation re-stamps.

    ``body`` is the document **as this write leaves it**, never the stored one,
    and that is the whole reason it is a parameter. The apply is what puts *"User
    approves the plan."* into ``## Approval``, so hashing the pre-write body
    would stamp a digest the document no longer has — and
    ``changed_since_acceptance`` would read true the instant the plan was
    accepted, on the strength of the acceptance itself.

    ``accepted_revision`` is read off ``plan`` rather than passed, so it picks up
    :func:`submit_review`'s increment on that path and correctly does not on
    :func:`resolve_note`'s, which does not increment. Both are *"the revision
    this document had when it was approved"*.

    ``accepted_via`` is ``"owner"`` unconditionally: :func:`may_decide` is the
    door on both paths and it admits nobody else.
    """
    plan.accepted_at = _now()
    # WRITE-ONCE, and the guard is the whole of the difference between this
    # field and the line above it. `accepted_at` re-stamps on every owner apply
    # because that is what "when they LAST put their name to this text" means;
    # `first_accepted_at` is the boundary `note_is_deviation` compares against
    # and moving it would demote every deviation filed since the last apply.
    plan.first_accepted_at = plan.first_accepted_at or plan.accepted_at
    plan.accepted_by = by
    plan.accepted_via = "owner"
    plan.accepted_revision = plan.revision
    plan.accepted_spec_digest = spec_digest(body)
    _note(plan, by, "approve", section=APPROVAL_SECTION)


# ---------------------------------------------------------------------------
# The tray
# ---------------------------------------------------------------------------


async def stage_draft(
    project: "Project",
    ctx: "ModelContext",
    *,
    by: str,
    draft: PlanReviewDraft,
) -> PlanReviewDraft:
    """Replace the whole tray. **Staging is never refused** (§4.4).

    Server-side and one per (plan, user), so two tabs and two devices share one
    tray and a submit from either empties both. **Never validated against the
    document** — staleness is computed on read, never stored, and a row records
    ``base`` and nothing more.

    A second text-carrying row for a section that already has one is *kept*, not
    rejected: §4.2's three-way control is a staging-time affordance in the tab,
    and the server's job is only to never discard a row on its own. §2.6's
    mutual exclusion is resolved at submit, where the superseded proposal
    resolves ``rejected`` with a reason.
    """
    if by != OWNER:
        raise PlanForbiddenError("Agents do not stage; the tray is the owner's")
    if len(draft.entries) > MAX_DRAFT_ENTRIES:
        raise PlanLimitError(
            f"tray has {len(draft.entries)} rows, limit is {MAX_DRAFT_ENTRIES}"
        )
    for entry in draft.entries:
        if entry.kind not in DRAFT_KINDS:
            raise PlanInputError(f"Unknown tray row kind {entry.kind!r}")

    async with _lock:
        plan = _load(project, ctx)
        taken: set[str] = set()
        for entry in draft.entries:
            if not entry.id or entry.id in taken:
                entry.id = _gen_id("d", taken)
            taken.add(entry.id)
            entry.at = entry.at or _now()
        draft.by = by
        draft.updated_at = _now()
        plan.draft = draft
        _save(project, ctx, plan)
        return draft


async def discard_draft(project: "Project", ctx: "ModelContext", *, by: str) -> None:
    """``DELETE /plan/draft``. Discarding your own tray is not a decision about
    the document, so it writes no history and touches no byte of PLAN.md."""
    if by != OWNER:
        raise PlanForbiddenError("Agents do not stage; the tray is the owner's")
    async with _lock:
        plan = _load(project, ctx)
        plan.draft = None
        _save(project, ctx, plan)


# ---------------------------------------------------------------------------
# Rounds
# ---------------------------------------------------------------------------


def _open_round_locked(
    plan: ProjectPlan, *, by: str, room: str, addressees: Sequence[str]
) -> PlanReviewRound:
    round_ = PlanReviewRound(
        round_id=_gen_id("r", (r.round_id for r in plan.rounds)),
        opened_at=_now(),
        opened_by=by,
        room=room,
        addressees=list(addressees),
    )
    plan.rounds.append(round_)
    return round_


async def open_round(
    project: "Project",
    ctx: "ModelContext",
    *,
    by: str,
    room: str,
    addressees: Sequence[str],
) -> PlanReviewRound:
    """Round bookkeeping. ``rounds`` is capped at 50 and trims from the front."""
    async with _lock:
        plan = _load(project, ctx)
        round_ = _open_round_locked(plan, by=by, room=room, addressees=addressees)
        _note(plan, by, "round", detail=round_.round_id)
        _save(project, ctx, plan)
        return round_


def send_digest(note: PlanNote, to: str) -> str:
    """``sha256(to, section, comment, proposal)`` — a note is re-sent **iff** this
    changed, which is what makes editing a note's text re-send that note only.

    **M5 AC-5.3 — the ledger keys on the coordinator now, and that is right; no
    code change, and here is why it needs none.** Under M2's two-party rule
    ``to`` is the keeper on every note the owner sends and ``user`` on every
    note the coordinator sends, so the ledger's question became *"has this
    note's text been sent to the other end of the channel?"* — which is
    precisely the property the digest exists for, and the reason editing a
    note's text re-sends **that note only**.

    *"Already sent to @backend"* does not need replacing; it ceases to exist as
    a **concept**. A specialist is reached by the coordinator's relay into
    ``shared-context``, which is a plain ``reply`` at a turn boundary and
    deliberately rides no ledger at all: the relay is a fresh act each round,
    and a dedupe guard there would be a second write path for a fact this one
    already answers about the only two parties that have one.
    """
    # An explicit POSITIVE list, which is why AC-1.3 needed no change here:
    # `applied_text` is excluded from the digest BY CONSTRUCTION, and adding a
    # field to `PlanNote` cannot silently move a note's send digest. Pinned by
    # assertion in the tests rather than by a guard here.
    material = "\x00".join([to, note.section, note.comment, note.proposal])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


async def record_send(
    project: "Project",
    ctx: "ModelContext",
    *,
    round_id: str,
    send: PlanSend,
) -> None:
    """``sends`` is keyed by note id and is **additive-only**."""
    async with _lock:
        plan = _load(project, ctx)
        for round_ in plan.rounds:
            if round_.round_id == round_id:
                round_.sends[send.note_id] = send
                _save(project, ctx, plan)
                return
        raise PlanNotFoundError(f"Round {round_id!r} not found")


def _auto_close_rounds(plan: ProjectPlan) -> None:
    """A round auto-closes once every note in it is non-open (§5.8)."""
    by_id = {n.id: n for n in plan.notes}
    for round_ in plan.rounds:
        if round_.closed_at or not round_.sends:
            continue
        if all(
            by_id.get(nid) is not None and by_id[nid].status != "open"
            for nid in round_.sends
        ):
            round_.closed_at = _now()


async def close_round(
    project: "Project", ctx: "ModelContext", *, round_id: str, by: str
) -> PlanReviewRound:
    async with _lock:
        plan = _load(project, ctx)
        for round_ in plan.rounds:
            if round_.round_id == round_id:
                round_.closed_at = round_.closed_at or _now()
                _note(plan, by, "round", detail=f"{round_id} closed")
                _save(project, ctx, plan)
                return round_
        raise PlanNotFoundError(f"Round {round_id!r} not found")


# ---------------------------------------------------------------------------
# The submit — one transaction
# ---------------------------------------------------------------------------


#: Renders the message one addressee receives. Injected so that phase 5 can
#: supply §5.7's template without this module knowing anything about it, and so
#: that a test can fault-inject the posting step (AC-2.9's transaction half).
MessageRenderer = Callable[[str, "list[PlanNote]", str], str]


#: Posts an owner's whole submit as ONE act and returns the message ids, in the
#: order the items were given. Injected for the same reason
#: :data:`MessageRenderer` is: this module must keep knowing nothing about
#: WorkTracker, ws_hub or HTTP, and Layer 1 may not import ``server/routes``.
#:
#: The route supplies ``send_addressed_messages_as_user`` — the ordinary
#: user-message machinery, which is what makes a review message wake its
#: addressee. Posting it by hand instead is how this path came to attribute the
#: owner's message to the coordinator (so the runner's self-message guard,
#: ``participant_notifier.py:120``, dropped it), open no ``PendingWork`` (so
#: nothing ever timed out to say so), and broadcast no ``CHANGELOG_UPDATE`` (so
#: no runner fetched it). One door, none of the three.
#:
#: ``prelude_specs`` carry the PLAN.md write, which must take a lower version
#: than any message: §4.2's *"the write and the send are one act or neither"* is
#: preserved by the door putting all of it in one ``append_batch``.
OwnerPoster = Callable[
    ["Sequence[BatchEntrySpec]", "Sequence[tuple[str, str, list[str]]]"],
    Awaitable["list[str]"],
]   # (prelude_specs, [(room, content, expects)]) -> message ids


#: §5.7 quotes the section a note is about, and a plan section can be long. The
#: cap is on the QUOTE, never on the document: the message stays readable and
#: names the one command that prints the rest.
SECTION_QUOTE_CHARS = 4000


def _section_headings(body: str) -> dict[str, str]:
    return {s.id: s.heading for s in parse_sections(body)}


def _last_touched(history: Sequence[PlanHistoryEntry], slug: str) -> str:
    """Who last moved this section, from the sidecar's own history.

    Derived rather than stored: ``PlanHistoryEntry`` already records ``by`` and
    the sections a write moved, so a ``last_touched_by`` field would be a second
    name for a fact the log carries — the drift §3.3 warns about.
    """
    for entry in reversed(list(history)):
        if entry.verb in ("write", "apply", "submit") and slug in entry.section.split(","):
            return entry.by
    return ""


def _quote_warning(body: str, note: PlanNote) -> str:
    """P1-a's one sentence, or ``""``.

    **It exists because AN AGENT HAS NO DOM.** The owner's screen needs none of
    this — the browser already computes the same answer from rendered text as
    ``Hit.count``, and can highlight the passage. A specialist reading a review
    message gets the excerpt as a block quote with no freshness signal at all,
    so it can be handed a passage that is no longer in the plan and reason from
    it in perfect confidence.

    Follows :func:`_changed_warning`'s shape **and its discipline**: silent
    unless there is something to say. ``""`` when the question was not askable,
    and ``""`` when the quote is anchored exactly once — which is the ordinary
    case, so the ordinary message is byte-identical to what it was.
    """
    if not quote_checked(body, note):
        return ""
    hits = quote_matches(body, note)
    if hits == 1:
        return ""
    if hits == 0:
        return (
            "⚠ This passage is no longer in the section. It may have been edited "
            "or removed since the note was written; judge the note against the "
            "section text below."
        )
    return (
        f"⚠ This passage occurs {hits} times in the section, so it does not "
        f"identify one place on its own."
    )


def _changed_warning(
    note: PlanNote, current: str, revision: int, history: Sequence[PlanHistoryEntry]
) -> str:
    """§5.7's *"judge it against what it says now"* banner.

    Rendered **only** when the section actually moved, because a warning that
    appears on every note is one nobody reads. The revision pair is dropped when
    the note predates :attr:`PlanNote.revision` — better silent than wrong about
    a number the reviewer would otherwise trust.
    """
    # `extent_key`, for the same reason `section_changed` uses it: this banner
    # states that predicate's gate in its own words, so a raw compare here put
    # an amber "this section has changed" on every note anchored to the LAST
    # section of the document — into the review message every specialist reads.
    if not note.proposal or extent_key(current) == extent_key(note.base_section):
        return ""
    who = _last_touched(history, note.section)
    span = (
        f"revision {note.revision} → {revision}"
        if note.revision and revision and note.revision != revision
        else "since it was written"
    )
    tail = f", last touched by {who}" if who else ""
    return (
        f"⚠ This section has changed since the proposal was written ({span}{tail}). "
        f"The text above is what it says now.\n"
        f"  Judge the proposal against that, not against what the author saw."
    )


def render_batch_message(
    addressee: str,
    notes: Sequence[PlanNote],
    batch_comment: str = "",
    *,
    body: str = "",
    project_ref: str = "",
    project_title: str = "",
    keeper_name: str = "",
    round_no: int = 0,
    revision: int = 0,
    all_notes: Sequence[PlanNote] = (),
    history: Sequence[PlanHistoryEntry] = (),
) -> str:
    """§5.7's message — **the whole point of the feature**, and the bar it is
    held to is AC-5.3: *the addressee can act without opening anything else.*

    Every argument is plain data — strings and model rows, no ``Project``, no
    ``ModelContext``, no I/O — so ``plan review --dry-run`` renders the identical
    text from what ``GET /plan`` returned. The alternative, a template that lives
    in ``cli_plan.py``, means the tab's submit and the CLI's submit post
    **different messages for the same act**, and the thinner one is whichever
    surface the reader happens to be on.

    Three properties are non-negotiable and each is asserted separately: the
    proposal renders as a unified diff **generated for display and never
    applied**; the section is shown **as it stands now**, with the reviewer told
    when that differs from what the author saw; and the **thread comes with the
    note**, so an addressee joining round 2 need not reconstruct round 1.
    """
    headings = _section_headings(body)
    replies_by_parent: dict[str, list[PlanNote]] = {}
    for note in all_notes:
        if note.reply_to:
            replies_by_parent.setdefault(note.reply_to, []).append(note)

    ref = project_ref or "<project>"
    head = f"@{addressee}" if addressee != OWNER else "For the project owner"
    title = project_title or ref
    lines = [head, "", f"Plan review — round {round_no or 1} — **{title}** (project `{ref}`)"]
    if batch_comment:
        lines.append(batch_comment)

    total = len(notes)
    for i, note in enumerate(notes, 1):
        slug = note.section
        heading = headings.get(slug, "")
        where = f"section `{slug}`" + (f" › {heading}" if heading else "")
        if not slug:
            where = "the document as a whole"
        lines += ["", "---", f"### Note {i} of {total} — {where}", ""]
        lines.append(
            f"Note `{note.id}` by {note.by or 'unknown'} on {(note.at or '')[:10]}:"
        )
        for row in (note.comment or "_(no comment — the proposal is the note)_").split("\n"):
            lines.append(f"> {row}")

        if note.quote:
            lines += ["", "**Quoted excerpt**"]
            lines += [f"> {row}" for row in note.quote.split("\n")]
            warn = _quote_warning(body, note)
            if warn:
                lines.append(warn)

        extent = section_extent(body, slug) if slug else None
        current = body[extent[0]:extent[1]] if extent else ""
        if extent is not None:
            shown = current[:SECTION_QUOTE_CHARS]
            lines += [
                "",
                f"**The section this is about** (`{slug}`, truncated at "
                f"{SECTION_QUOTE_CHARS} chars)",
                "```",
                shown.rstrip("\n"),
                "```",
            ]
            if len(current) > SECTION_QUOTE_CHARS:
                lines.append(
                    f"_Truncated. Full text: `clawmeets plan show {ref} "
                    f"--section {slug}`._"
                )
        elif slug:
            lines += ["", f"**The section this is about** (`{slug}`) no longer exists."]

        if note.proposal:
            lines += ["", "**Proposed change**"]
            warning = _changed_warning(note, current, revision, history)
            if warning:
                lines.append(warning)
            lines += [
                "```diff",
                render_diff(note.base_section, note.proposal, label=slug or "section"),
                "```",
                f"_Full text: `clawmeets plan show {ref} --note {note.id}`_",
            ]

        thread = replies_by_parent.get(note.id, [])
        if thread:
            lines += ["", "**Replies so far**"]
            for reply in sorted(thread, key=lambda n: n.at):
                first = (reply.comment or "").split("\n")[0]
                lines.append(f"> `{reply.id}` @{reply.by}: {first}")

    owner_ref = keeper_name or "<keeper>"
    example = notes[0].id if notes else "n-xxxx"
    example_slug = (notes[0].section if notes else "") or "<slug>"
    lines += [
        "",
        "---",
        "### How to respond",
        f'- Answer or discuss: `clawmeets plan note {ref} --reply-to {example} -m "..."`',
        f"- Suggest a change: `clawmeets plan note {ref} --section {example_slug} "
        f'--to {owner_ref} --edit-file <file> -m "why"`',
        f"- Close it out: `clawmeets plan resolve {ref} {example} --answered|--dismiss`",
        f"- Read the whole plan: `clawmeets plan show {ref} --clean`",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The send ledger and the room — §5.7, and both have exactly one implementation
# ---------------------------------------------------------------------------


def last_send(rounds: Sequence[PlanReviewRound], note_id: str) -> PlanSend | None:
    """The most recent recorded send of this note, or ``None``.

    Takes the rounds rather than the sidecar so the **CLI** can call it with the
    rows ``GET /plan/rounds`` returned: the ledger's shape is the same fact on
    either side of the wire, and a second implementation over there is how a
    terminal and a browser come to disagree about what has been sent.
    """
    for round_ in reversed(list(rounds)):
        send = round_.sends.get(note_id)
        if send is not None:
            return send
    return None


def send_is_current(
    rounds: Sequence[PlanReviewRound], note: PlanNote, to: str
) -> PlanSend | None:
    """The send that makes this note **already out**, or ``None``.

    A note is out iff it was sent to this addressee and its digest has not moved
    since. That is what makes *"editing a note's text re-sends that note only"*
    (AC-5.2) one rule rather than a special case: the edit moves the digest, the
    digest is the ledger's key, and nothing else in the batch moves at all.
    """
    previous = last_send(rounds, note.id)
    if previous is None or previous.digest != send_digest(note, to):
        return None
    return previous


def pending_notes(
    notes: Sequence[PlanNote], rounds: Sequence[PlanReviewRound]
) -> list[PlanNote]:
    """A batch's default set: every **open**, **addressed** note not already out.

    ``to == ""`` is *"recorded, never sent"* (§5.4) and is excluded here for the
    same reason :func:`open_notes_for_you` excludes it — a note nobody was sent
    must not turn up in somebody's inbox because a batch went out.

    Pure, over the two lists, so ``plan review`` picks the same set the server
    would. It is deliberately **not** the default inside :func:`submit_review`:
    the tray submits an explicit ``entries`` list, and a browser submit that
    silently fanned out every open note would be a different act than the one
    the button says it does.

    **The go-note is excluded, and it is the one note that ever needed to be.**
    It is open, addressed to the user and never sent, so it matches every other
    clause here — and it is not the coordinator's to deliver. The server files
    it at project creation, it is in the user's plan tray and on their desk card
    from that moment, and it belongs to no review round. Including it would put
    the server's own gate note inside a batch of the coordinator's questions,
    under a *"Plan review — round 1"* heading, as though the coordinator had
    asked for approval as one item among several. The user has already been
    asked; the note IS the asking.
    """
    return [
        n
        for n in notes
        if n.status == "open"
        and n.to
        and not n.bootstrap
        and send_is_current(rounds, n, n.to) is None
    ]


def review_room_for(addressees: Iterable[str], keeper_name: str) -> str:
    """§5.7's room table, as a pure function of who is being written to.

    Empty when there is nobody to write to — a batch of pure decisions posts
    nothing and therefore needs no room, which is what keeps *"a submit of only
    rejects appends nothing"* from tripping over room creation.

    **M5 AC-5.1 — ``USER_ROOM`` is now the only outcome for a NEW note, and
    this function is not what guarantees it.** The guarantee lives in
    :func:`add_note`, whose :func:`note_addressee_allowed` check makes
    ``{keeper, OWNER}`` the only set an addressee can come from; by the time a
    batch reaches here the addressees are already inside the set the
    ``names <= {keeper_name, OWNER}`` branch tests for. Said out loud because
    the next reader will otherwise believe **this** function enforces the
    two-party rule and will tighten it here — a rule with two homes is a rule
    with two answers, and M2 deliberately gave it one.

    **The ``PLAN_ROOM`` branch is not dead — it is the LEGACY path AND the
    spec-consultation path.** A note filed before M2 may carry
    ``to == <specialist>``, and resolving one must still reach its author; a
    spec consultation relayed by the coordinator lands here too. Its own test
    says so, or the next reader deletes it as unreachable.
    """
    names = {a for a in addressees if a}
    if not names:
        return ""
    if names <= {keeper_name, OWNER}:
        return USER_ROOM
    return PLAN_ROOM


def _is_external_invitee(project: "Project", agent: "Agent") -> bool:
    """Is ``agent`` foreign to ``project`` — registered by a different user?

    The literal predicate the HTTP ``create_room`` door applies
    (``server/routes/chatrooms.py``), reproduced because
    :func:`ensure_consultation_roster` does not go through that door and a
    cross-owner invitee tagged ``False`` would have its runner's notifier
    process a membership it must skip: the tunnel handles delivery for those.

    Only on a ``regular`` project — a DM-shaped project *is* a cross-user
    channel, so further routing there is meaningless — and only for an agent
    whose owner is known. The coordinator is never a candidate: it is added by
    :func:`_consultation_roster` before the roster is walked.

    Free here, where it costs the door a lookup: the roster query already
    handed us the :class:`Agent`, and the door holds only participant ids.
    """
    return (
        project.surface == "regular"
        and bool(project.created_by)
        and bool(agent.registered_by)
        and agent.registered_by != project.created_by
    )


def _consultation_roster(
    project: "Project", ctx: "ModelContext"
) -> list[RoomCreatedParticipant]:
    """Who the consultation room holds: the coordinator, then **the invitable
    roster** — :meth:`Agent.invitable_agents_for_project`.

    **Not** ``project.participating_agents``. That list is ``[coordinator_id]``
    on a fresh regular project (``Project.create``), and the auto-add that ever
    puts anyone else on it runs inside the HTTP ``create_room`` handler, which
    ``_plan_execution_blocked`` refuses while the go-note is open — which is the
    whole of ``spec-ing``, since accepting that note is what ends it. Syncing
    from it therefore produced a room of one, every time, and
    :func:`submit_review` then dropped every specialist addressee into
    ``round.unresolved`` and returned 200 — the review that reached nobody.

    **No allowlist check.** The door raises 403 for an invitee outside
    ``project.matches_invitable``; here the roster *is* that allowlist, so the
    check is satisfied by construction and re-deriving it would put a second
    copy of the rule beside its owner.

    The coordinator is first and ``seen`` de-duplicates, so a roster that
    happens to contain the coordinator does not seat it twice.
    """
    participants = [
        RoomCreatedParticipant(id=project.coordinator_id, name=project.coordinator_name)
    ]
    seen = {project.coordinator_id}
    for agent in Agent.invitable_agents_for_project(
        project, ctx, exclude_ids=frozenset(seen)
    ):
        if agent.id in seen:
            continue
        seen.add(agent.id)
        participants.append(
            RoomCreatedParticipant(
                id=agent.id,
                name=agent.name,
                external=_is_external_invitee(project, agent),
            )
        )
    return participants


async def _sync_roster_into_plan_room(
    project: "Project",
    ctx: "ModelContext",
    runloop: "ChangelogRunloop",
    participants: Sequence[RoomCreatedParticipant],
) -> bool:
    """Put every roster member into ``shared-context``, mirroring the HTTP
    door's auto-add. **Idempotent per participant**, not per room.

    That distinction is the single most important line of this module. Its
    ancestor, ``ensure_review_room``, short-circuited the moment the room
    existed — correct when the room was ``plan-review``, opened on demand, and
    fatal the moment the room became ``shared-context``, which exists on every
    regular project from creation. Pointing the old contract at the new room
    would have returned ``False`` on every call, left the roster admit as dead
    code, and had the coordinator consult an empty room while believing it had
    consulted the team. A pure rename ships that bug; this signature is what
    stops it.

    Agents that cannot open ``PLAN.md`` cannot review it, and ``PLAN.md`` lives
    in this room — that was the original reason for the mirror and it is
    unchanged. What changed is that the room hosting the conversation is now
    the same room, so this is no longer a *second* append beside a
    ``ROOM_CREATED``; it is the whole of what ``consult`` does.

    Tolerates a ``shared-context`` that does not exist (``Chatroom.get`` raises
    ``ValueError`` for an unknown project, returns ``None`` for an unknown
    room) and skips anyone already a member, so a second ``consult`` with an
    unchanged roster appends nothing and does not spam the changelog.
    ``external`` rides through from the roster rather than defaulting to
    ``False``: this membership is the one a foreign agent's runner skips.

    Returns True iff it appended anything — the route broadcasts on that.
    """
    try:
        shared = Chatroom.get(project.id, PLAN_ROOM, ctx)
    except ValueError:
        return False
    if shared is None:
        return False
    existing = set(shared.participants)
    appended = False
    for p in participants:
        if p.id in existing:
            continue
        await runloop.append(
            ChangelogEntryType.PARTICIPANT_ADDED,
            ParticipantAddedPayload(
                chatroom_name=PLAN_ROOM,
                participant_id=p.id,
                participant_name=p.name,
                external=p.external,
            ),
        )
        appended = True
    return appended


async def ensure_consultation_roster(
    project: "Project",
    ctx: "ModelContext",
    runloop: "ChangelogRunloop",
    room_name: str,
) -> bool:
    """**One door.** Make ``room_name`` usable as a batch room, in-process
    through the runloop (B9).

    Lives here rather than in the route because the CLI needs it too, and B9's
    bug was precisely that room creation had one door: an owner holding a user
    JWT could not create the room its own batch needed.

    Three cases, and the middle one is the whole point of the merge:

    ``room_name == ""``
        Nothing to do. A batch of pure decisions posts nothing and needs no
        room, which is what keeps *"a submit of only rejects appends nothing"*
        from tripping over room creation.

    ``room_name == PLAN_ROOM``
        The consultation room, and it **already exists** — it is seeded on every
        project at creation (``server/routes/projects.py``). So there is no room
        to create and the work is entirely
        :func:`_sync_roster_into_plan_room`'s: one ``PARTICIPANT_ADDED`` per
        roster member not already in it. Read that function's docstring before
        reintroducing an early return on "the room exists".

    anything else
        A caller-supplied ``--room``. Legacy and defensive: no in-tree caller
        reaches it any more, since :func:`review_room_for` now returns only
        ``""``, ``USER_ROOM`` (which always exists, so this is a no-op) or
        ``PLAN_ROOM``. Created with the roster if absent, and its members are
        mirrored into ``shared-context`` for the same reason they always were —
        a reviewer that cannot read ``PLAN.md`` is not reviewing.

    Returns True iff it appended anything.
    """
    if not room_name:
        return False
    existing = Chatroom.get(project.id, room_name, ctx)
    if room_name != PLAN_ROOM and existing is not None:
        return False
    participants = _consultation_roster(project, ctx)
    if existing is None:
        await runloop.append(
            ChangelogEntryType.ROOM_CREATED,
            RoomCreatedPayload(chatroom_name=room_name, participants=participants),
        )
        # Created WITH the roster, so for `PLAN_ROOM` there is nothing left to
        # mirror — everyone is already in it.
        if room_name == PLAN_ROOM:
            return True
    mirrored = await _sync_roster_into_plan_room(project, ctx, runloop, participants)
    return mirrored or existing is None


def _row_creates(plan: ProjectPlan, row: DraftEntry) -> bool:
    """Does this tray row mean to CREATE the section it names?

    **Asked of the NOTE, never of the row's own ``base``**, and that is the whole
    of this function. ``acceptRow`` stages ``base`` = the section's text as the
    document reads NOW, so a row whose section has vanished carries ``base == ""``
    in BOTH of the cases :func:`_splice` exists to keep apart: the proposal
    written against a section that has since been DELETED UNDER IT, and the
    proposal written to put back a section that was already gone. One value, two
    facts — the same collision ``SectionEdit.create`` was added to break, one
    layer up. Reading the row's base here would simply move it.

    The NOTE's ``base_section`` is the fact, because it was captured when the
    note was filed (:func:`add_note` → :func:`_capture_base`, which answers ``""``
    for a slug that does not resolve). ``""`` means *there was nothing there when
    this was written*: a create, landing at the end of the document like every
    other create (§15.4). Non-empty means *there was, and it is gone now*: the
    ``409`` that the "vanished section is refused, not appended" guard test pins,
    unchanged.

    **An ``edit`` row is never a create.** It carries no note and was typed
    against text the editor was showing, so a section that vanished under it is
    exactly the staleness the tray already renders as ``missing``. The owner who
    genuinely wants a new section has ``plan update``, which sets the flag
    itself; there is deliberately no way to spell "create" from the tray.

    :func:`resolve_note`'s apply arm carries the identical rule, because
    ``test_both_doors_answer_a_vanished_section_the_same_way`` is right that a
    surface refusing what its sibling accepts is the shape of a bug.
    """
    if row.kind != "accept" or not row.note_id:
        return False
    return not _find_note(plan, row.note_id).base_section


def _collapse_text_rows(entries: Sequence[DraftEntry]) -> tuple[list[DraftEntry], list[DraftEntry]]:
    """§2.6, one layer down: ``apply_edits`` splices one replacement per section,
    so **at most one row per section may carry text**.

    The later row wins — that is the rule the user set. Returns
    ``(applied, superseded)``; nothing is dropped silently, because every
    superseded row's note resolves ``rejected`` with a written reason.
    """
    text_rows = [e for e in entries if e.kind in ("edit", "accept")]
    last_per_section: dict[str, DraftEntry] = {}
    for entry in text_rows:
        last_per_section[entry.section] = entry
    winners = set(id(e) for e in last_per_section.values())
    return (
        [e for e in text_rows if id(e) in winners],
        [e for e in text_rows if id(e) not in winners],
    )


async def submit_review(
    project: "Project",
    ctx: "ModelContext",
    runloop: "ChangelogRunloop",
    *,
    by: str,
    entries: Sequence[DraftEntry] | None = None,
    note_ids: Sequence[str] = (),
    room: str = "",
    resend: bool = False,
    batch_comment: str = "",
    render: MessageRenderer | None = None,
    post: OwnerPoster | None = None,
) -> PlanReviewRound:
    """**One transaction**, and the only place ``revision`` is incremented on a
    regular project (§2.5, §3.2, §4.2).

    In order, under one lock:

    1. ``apply_edits`` with every ``edit`` and ``accept`` row — one write, one
       revision. If any row is stale the whole submit is ``409``, **the tray is
       untouched**, nothing is sent, resolved or written.
    2. Render one message per addressee. Rendering happens **before** anything is
       appended, so a failure here leaves the document unwritten and the tray
       full.
    3. Resolve every decided note **in memory** — nothing is persisted yet.
    4. One :meth:`~ChangelogRunloop.append_batch`: the ``PROJECT_PLAN_STATE``
       projection, the ``FILE_UPDATED`` and every message, atomically. *Send
       review to #room* does one thing that either happens or does not.
    5. Record the round and its sends, clear the tray, ``_save``.

    **Step 3 is above step 4 and that is deliberate.** The projection in step 4's
    prelude has to carry the note count this submit leaves behind, or the
    coordinator turn the messages wake reads the pre-submit count and refuses its
    own work (the comment at the prelude carries the full argument). Moving the
    closes up costs nothing: they are pure in-memory edits to a ``plan`` object
    :func:`_load` re-reads from disk on every call, ``_save`` in step 5 is this
    module's only writer, and so an append that raises still discards all of them
    at once. *"Nothing resolves until the batch lands"* was always a statement
    about persistence, never about statement order.

    Three different ``(append, increment)`` pairs come through here and all three
    are load-bearing: a submit with edits appends one ``FILE_UPDATED`` and
    increments once; a submit of only rejects/asks/dismisses appends nothing and
    leaves ``revision`` unchanged (§4.1 step 6, no code); and a coordinator's
    checkbox tick — which arrives through :func:`apply_edits`, not here — appends
    a ``FILE_UPDATED`` and leaves ``revision`` unchanged (§2.5).
    """
    async with _lock:
        plan = _load(project, ctx)
        draft = plan.draft or PlanReviewDraft(by=by)
        rows = list(entries) if entries is not None else list(draft.entries)

        # **The ownership guard, and it sits HERE for a reason.** Inside the
        # lock, as soon as `rows` is bound and before `_collapse_text_rows`
        # computes anything: a refused batch has computed no write, filed no
        # conflict note, sent no message and touched no sidecar field. A pure
        # raise, matching the staleness branch's *"nothing applied, nothing
        # sent, tray untouched"* rather than the spec-lock branch's deliberate
        # file-and-save asymmetry.
        #
        # It resolves :func:`may_decide` rather than restating ``by == OWNER``,
        # so this door and :func:`resolve_note`'s cannot answer differently —
        # see that function's docstring for why one home is the requirement.
        #
        # ``dismiss`` is deliberately NOT guarded: it writes no plan bytes, it
        # only closes a note, and putting it inside the ownership guard was
        # considered and cut. ``edit``, ``ask`` and ``note_ids`` sends are
        # untouched — a specialist may still propose and still ask.
        for row in rows:
            if row.kind in ("accept", "reject"):
                _refuse_non_owner_decision(by, row.kind)
            # The tray's Reject/Dismiss reach the go-note the same way the CLI's
            # do, and would release the gate on a plan nobody accepted.
            if row.kind in ("reject", "dismiss") and row.note_id:
                _refuse_go_note_close(_find_note(plan, row.note_id), row.kind)
            # BOTH DOORS, one rule. `submit_review` is owner-only for `accept`
            # and `reject` but NOT for `dismiss`/`ask`, so a keeper batch can
            # reach a close from here too — and a guard that lived only in
            # `resolve_note` would be a rule with one home and two answers.
            # `dismiss` alone: it is the only closing kind a tray row carries
            # that is not already owner-gated above (`accept`/`reject`), and
            # `ask` does not close anything.
            if row.kind == "dismiss" and row.note_id:
                _refuse_unsent_self_close(plan, _find_note(plan, row.note_id), by=by)

        comment = batch_comment or draft.batch_comment
        room_name = room or draft.room

        applied_rows, superseded_rows = _collapse_text_rows(rows)
        # **A row that names no section is not a section write.** An ``accept``
        # staged by ``note_id`` alone carries no ``section``/``text``/``base`` —
        # it is a decision, and the bytes it would write are none. It used to
        # become ``SectionEdit(section="", text="", base="")`` and reach the
        # splice, where an unresolvable slug was coerced to ``""`` and the whole
        # thing happened to no-op. That coercion is exactly what let a proposal
        # against a DELETED section read as a create, so it is gone — and the
        # rows that relied on it are filtered here instead, where the fact is
        # actually known. ``_collapse_text_rows`` still sees every row, so what
        # is staged, superseded and resolved is unchanged.
        edits = [
            SectionEdit(
                section=r.section,
                text=r.text,
                base=r.base,
                # **AND THE ROW'S OWN BASE IS NOT THE INPUT.** See
                # `_row_creates`: an accept whose section has vanished carries
                # `base == ""` whether it is a restore or a conflict, so the
                # flag is derived from the NOTE, which knows which. It is
                # consulted by `_splice` only when the slug fails to resolve, so
                # it is inert on every row whose section is still there.
                create=_row_creates(plan, r),
            )
            for r in applied_rows
            if r.section
        ]

        # ---- 1. the write, COMPUTED but not yet appended ------------------
        prepared = _prepare_locked(project, ctx, plan, edits, by=by)
        result = prepared.result
        if result.locked:
            # **M3, and deliberately asymmetric with the staleness branch below.**
            # That one leaves the sidecar exactly as it found it; this one files
            # and saves before it raises.
            #
            # The asymmetry is the milestone, not an oversight, and it is written
            # here so nobody "restores symmetry" later: the document, the sends,
            # the resolutions and the tray are all still untouched — what does
            # NOT evaporate is the coordinator's text, which lands on the user's
            # desk as a deviation they can accept in one click. That is the whole
            # of AC-3.3, and a refusal that dropped it would be a dead end.
            ids = _file_spec_lock_locked(project, plan, result.locked, by=by)
            _note(
                plan, by, "refused",
                detail=",".join(s.section for s in result.locked),
            )
            _save(project, ctx, plan)
            raise PlanSpecLockedError(
                _spec_lock_refusal(
                    project, by=by, note_ids=ids, accepted=bool(plan.accepted_at)
                ),
                note_ids=ids,
                revision=result.revision,
            )
        if not result.ok:
            # Nothing applied, nothing sent, nothing resolved, tray untouched.
            #
            # **AND IT SAYS WHICH REFUSAL IT IS.** "A staged row is stale" is
            # false for a row whose section is GONE: nothing moved under it and
            # there is nothing to re-read. Worse than false, it is a dead end
            # the browser acts on — `PlanEditor` opens `PlanCollide` on this
            # payload, `PlanCollide` offers to re-base the row onto `current`,
            # and `current` for a vanished section is `""`, so the row goes back
            # unchanged and earns the same 409 forever. `resolve_note`'s apply
            # arm has always had the right sentence; `_vanished_refusal` is that
            # sentence, now with one home instead of two.
            gone = [
                s.section for s in result.stale
                if _current_section(prepared.body, s.section) is None
            ]
            raise PlanConflictError(
                f"{_vanished_refusal(gone)}; nothing was applied or sent"
                if gone
                else "a staged row is stale; nothing was applied or sent",
                stale=result.stale,
                revision=result.revision,
            )

        # ---- 2. what each addressee is told -------------------------------
        # A batch carries traffic in **two directions**, and one rule cannot
        # derive both. A decision goes back to whoever WROTE the note — that is
        # what `DraftEntry.to` means, "the note's author, or the keeper". A note
        # sent by id goes to whoever it is ADDRESSED to, which is what makes a
        # `to=user` note reach the user (§5.7).
        decisions = [r for r in rows if r.kind in ("accept", "reject", "ask", "dismiss")]
        by_addressee: dict[str, list[PlanNote]] = {}
        seen: set[tuple[str, str]] = set()

        def _address(addressee: str, note: PlanNote) -> None:
            if not addressee or (addressee, note.id) in seen:
                return
            seen.add((addressee, note.id))
            by_addressee.setdefault(addressee, []).append(note)

        for row in decisions:
            if row.note_id:
                note = _find_note(plan, row.note_id)
                # DISMISSING YOUR OWN NOTE TELLS NOBODY (AC-1.8). The client
                # cannot express "nobody" — it sends `to: ''`, which IS the
                # fallback trigger — so the fallback resolved `note.by`, which
                # on your own note is you, and mailed the user about their own
                # dismissal. Nobody needs telling that they did the thing they
                # just did.
                #
                # A NON-EMPTY `row.to` is still honoured: that is somebody the
                # user chose, and "nobody" is only the answer when nobody was
                # named. Narrow on purpose — dismissing an agent's note still
                # reaches the agent, because there the dismissal IS news.
                if (
                    row.kind == "dismiss"
                    and not row.to
                    and note.by == OWNER
                    and by == OWNER
                ):
                    continue
                _address(row.to or note.by or keeper(project), note)
        for nid in note_ids:
            note = _find_note(plan, nid)
            _address(note.to or keeper(project), note)

        # The send ledger, applied HERE and not after the append. A note whose
        # digest has not moved is already out: naming it explicitly is §5.7's
        # one refusal, and reaching it through the default set is simply a
        # no-op. Filtering after the messages were posted — which is what the
        # first implementation did — re-sent the note and recorded nothing,
        # making AC-5.2's "re-sending sends nothing" false in the loudest
        # possible way: a duplicate message with no ledger row to explain it.
        named = set(note_ids)
        for addressee, addressed in list(by_addressee.items()):
            kept = []
            for note in addressed:
                previous = send_is_current(plan.rounds, note, addressee)
                if previous is None or resend:
                    kept.append(note)
                    continue
                if note.id in named:
                    raise PlanConflictError(
                        f"Note {note.id} is already out in round "
                        f"{_round_of(plan, note.id)} — pass --resend to send it again",
                        revision=plan.revision,
                    )
            if kept:
                by_addressee[addressee] = kept
            else:
                del by_addressee[addressee]

        # §5.7's room table. Resolved here rather than by each caller so the
        # terminal and the browser agree, and CREATED here so an owner holding a
        # user JWT can send a batch that needs one (B9).
        room_name = room_name or review_room_for(by_addressee, keeper(project))
        if by_addressee and not room_name:
            raise PlanInputError("A review with addressees needs a room to post into")
        room_created = await ensure_consultation_roster(
            project, ctx, runloop, room_name
        )
        members = _members_by_name(project, ctx, room_name) if by_addressee else {}

        # AC-5.6 — an addressee nobody in the room answers to is reported and
        # skipped. Done HERE, before anything is rendered or appended, so a
        # typo'd name costs the batch nothing at all; the alternative, letting
        # it through with an empty `expects`, posts a message addressed to
        # somebody who does not exist and calls that a send.
        unresolved = sorted(
            a for a in by_addressee if a != OWNER and a not in members
        )
        for name in unresolved:
            del by_addressee[name]
        if render is None:
            render = functools.partial(
                render_batch_message,
                body=prepared.body,
                project_ref=project.name,
                project_title=project.display_name or project.name,
                keeper_name=keeper(project),
                round_no=len(plan.rounds) + 1,
                revision=plan.revision + (1 if prepared.wrote and project.surface == "regular" else 0),
                all_notes=plan.notes,
                history=plan.history,
            )
        # Two senders, because these are two different acts. An OWNER's review
        # is a message FROM THE USER and goes out through the ordinary
        # user-message door (`post`); an agent's review is a message from that
        # agent and is built here. The owner case used to be built here too,
        # through `_writer_identity`, which substitutes the coordinator for the
        # owner — right for a FILE event (the owner is not a room participant,
        # so one authored by them has nowhere to sit) and fatal for a MESSAGE,
        # because the coordinator addressed as itself is dropped by the runner's
        # self-message guard and the turn never starts.
        owner_send = by == OWNER and post is not None
        message_specs: list[BatchEntrySpec] = []
        message_ids: dict[str, str] = {}
        outbound: list[tuple[str, str, list[str]]] = []
        addressed_order: list[str] = []
        for addressee, notes in by_addressee.items():
            content = render(addressee, notes, comment)
            expects = [members[addressee]] if addressee in members else []
            if owner_send:
                outbound.append((room_name, content, expects))
                addressed_order.append(addressee)
                continue
            message_id, spec = _message_spec(
                project, ctx, room_name, by, content, expects
            )
            message_ids[addressee] = message_id
            message_specs.append(spec)

        # The `ask` rows' question notes, built HERE and filed unchanged in
        # step 4. Building them above the append is what lets **one** validation
        # cover both of this transaction's note filers — the shorthands
        # `_finish_locked` owes and the questions the loop below owes — because
        # the cap is a budget over the whole transaction. Asking about them
        # separately would let each pass and the pair fail, which is the
        # original bug with extra steps. And a submit of only `ask` rows never
        # reaches `_finish_locked` at all, so reordering that one alone leaves
        # this filer on the wrong side of the append.
        #
        # Two of the fields are what turns an `ask` row into a general REPLY
        # rather than only a question back at a note's author:
        #
        # `to` honours the ROW before falling back to the parent's author. The
        # row already carries the tray's own derivation of "who is on the other
        # end of this thread" (`DraftEntry.to`), and the decision loop eleven
        # lines above (`row.to or note.by or keeper(project)`) already trusts it.
        # Hard-coding `parent.by` mis-addresses the one case the fallback cannot
        # express: a reply to your OWN note, where `parent.by == "user"` and the
        # reply would be addressed straight back at the user.
        #
        # **The fallback is THREE-TERM, and the third term is load-bearing**
        # (AC-1.7, as amended). `_reply_addressee` — beside `_reply_parent`, and
        # what `add_note` uses — answers "the other end of the thread" rather
        # than "the parent's author". Alone it is right on a reply to your own
        # ADDRESSED note and a no-op on someone else's, but it REGRESSES the
        # third case: a reply to your own note addressed to NOBODY, where it
        # returns `""` and the reply goes nowhere, while `parent.by` delivers it
        # to the owner — who is exempt from the unresolved sweep, so that is a
        # real send and a real badge.
        # `test_ac_b1_a_reply_to_your_own_un_addressed_note_resolves_to_the_user`
        # shipped that trade on the stated grounds that a note landing in its
        # author's own queue is recoverable and a note that goes nowhere is not.
        # Trailing `or parent.by` keeps that case exactly as it was and changes
        # only the case AC-1.7 exists for, so the amendment inverts nothing —
        # which is why it is written as three terms and not two. Do not shorten
        # it to `row.to or _reply_addressee(parent, by)`: that is the two-term
        # form the amendment exists to reject, and its cost is a note with no
        # addressee, which nobody can recover.
        #
        # `quote` is inherited from the parent so the reply hangs under the same
        # line the argument is about instead of falling into the side pane.
        # Already-stored text, so the client-side quote cap still covers it —
        # this opens no new uncapped path.
        ask_notes: dict[int, PlanNote] = {}
        for idx, row in enumerate(decisions):
            if row.kind != "ask" or not row.note_id:
                continue
            parent = _find_note(plan, row.note_id)
            ask_notes[idx] = PlanNote(
                id="",
                section=parent.section,
                to=row.to or _reply_addressee(parent, by) or parent.by,
                by=by,
                comment=row.comment,
                # From `prepared.body` — the document as it will read once THIS
                # write lands — and NOT the stored body. The reply is filed in
                # the same transaction as the write, so the text its author saw
                # is the text this batch produces, and §5.7 quotes that same
                # body back to them.
                base_section=_capture_base(prepared.body, parent.section),
                quote=parent.quote,
                reply_to=parent.id,
            )
        _validate_notes_locked(plan, prepared.notes + list(ask_notes.values()))

        # ---- 3. the sidecar, IN MEMORY --------------------------------------
        # Applied here, above the append, because the projection built just
        # below has to count the notes this transaction closes — and only
        # `_close_note` knows which those are. `_save` stays below the append,
        # so the rollback property is unchanged: `_load` reads fresh from disk
        # on every call and `_save` is this module's only writer, so an append
        # that raises discards every mutation below in one go. The invariant
        # was never statement order; it was that nothing PERSISTS until the
        # batch lands.
        #
        # `_validate_notes_locked` still runs ABOVE all of it, so an `ask` row
        # cannot loosen its own transaction's open-note budget.
        wrote = prepared.wrote
        if wrote:
            _finish_locked(project, plan, prepared, by=by, verb="submit")
            if project.surface == "regular":
                # On front-desk the coordinator's write IS the accepted batch and
                # `_finish_locked` already counted it (§7.2 U3). Incrementing here
                # too would make one act two revisions.
                plan.revision += 1

        # Set by an `accept` row below and consumed once, after the loop. See
        # that row for why a batch must not stamp per decision.
        stamp_acceptance = False
        for idx, row in enumerate(decisions):
            if not row.note_id:
                continue
            note = _find_note(plan, row.note_id)
            if row.kind == "accept":
                # Immediately before the clear — see `resolve_note`. This is the
                # path the browser uses, so it is the one that matters most.
                note.applied_text = note.proposal
                note.proposal = ""
                _close_note(note, status="applied", by=by)
                if note.bootstrap or plan.accepted_at:
                    # **APPLYING A PROPOSAL IS ACCEPTING THE PLAN**, on the
                    # go-note the first time and on every owner-applied note
                    # after it. There is no approve route and no second act;
                    # this row is it, and `resolve_note`'s twin arm carries the
                    # argument for the `or` at length.
                    #
                    # **A FLAG AND NOT A CALL, because this is a LOOP.** A batch
                    # may accept several notes, and stamping per row would put
                    # one `approve` row in the history for each — the same act
                    # recorded N times. The single call is below the loop.
                    stamp_acceptance = True
            elif row.kind == "reject":
                _close_note(note, status="rejected", by=by, reason=row.comment)
            elif row.kind == "dismiss":
                _close_note(note, status="dismissed", by=by, reason=row.comment)
            elif row.kind == "ask":
                # **M5 AC-5.9 — a reply CLOSES the parent ``answered``**, in
                # the same transaction that files the outgoing question, and
                # this reverses a decision that stood here as deliberate: *"the
                # parent stays open: a question is not a resolution."*
                #
                # It is a resolution. `answered` is the ladder's own word for
                # *"the addressee has replied"*, and that is exactly what an
                # `ask` row is — the addressee replying. Leaving the parent
                # open made a reply the one act on this channel that resolved
                # nothing, so a note the user had answered kept badging the
                # user's own desk card and kept its round from auto-closing.
                # The three siblings above it (`accept`, `reject`, `dismiss`)
                # all close what they act on; this one now matches them.
                #
                # The close is IN MEMORY with the other three, and `_save` is
                # below the append, so a submit that trips a cap or a stale row
                # still persists no close — the rollback tests in
                # `test_plan_routes.py` pin that, and they pin it on the sidecar,
                # not on where the statement sits. And `_validate_notes_locked`
                # counted open notes above every one of these closes, so an `ask`
                # row still never loosens its own transaction's budget, only the
                # next one's.
                #
                # The gate consequence is not symmetric; see
                # :func:`open_notes_for_you` for which direction moves and why
                # that direction is correct.
                # **EXCEPT THE GO-NOTE, which a reply leaves open.** Every
                # sentence above is right about an ordinary note and wrong about
                # this one, because the go-note is not a question — it is the
                # gate. Closing it on reply would mean that typing *"hmm, what
                # about the auth milestone?"* dropped the note count to zero,
                # released the execution gate and started the work the user was
                # in the middle of questioning. Accepting it is the only thing
                # that means yes, so accepting it is the only thing that closes
                # it.
                #
                # The outgoing question is filed either way: a reply to the
                # go-note is a real message to the coordinator and must reach
                # it. Only the parent's status differs.
                if not note.bootstrap:
                    _close_note(note, status="answered", by=by)
                # The note object is the one validated above the append, not a
                # new one built to match it.
                _add_notes_locked(plan, [ask_notes[idx]])

        for row in superseded_rows:
            if row.kind != "accept" or not row.note_id:
                continue
            note = _find_note(plan, row.note_id)
            winner = next(
                (r for r in applied_rows if r.section == row.section), None
            )
            winner_note = (
                _find_note(plan, winner.note_id)
                if winner is not None and winner.note_id
                else None
            )
            _close_note(
                note,
                status="rejected",
                by=by,
                reason=SUPERSEDED_REASON.format(by=(winner_note.by if winner_note else by)),
            )

        # ---- 4. ONE act ----------------------------------------------------
        # The PROJECT_PLAN_STATE, the FILE_UPDATED and every message share one
        # append_batch. Everything that can fail — a stale row, an unreachable
        # room, a renderer that throws — has already failed above, with the
        # document unwritten and the tray still on disk.
        #
        # True on BOTH senders: the owner path hands the write to the door as
        # `prelude_specs` rather than appending it separately, so §4.2 holds
        # there too and the file still takes the lower version (it wakes nobody;
        # the messages must find it on disk).
        #
        # **The projection goes in the prelude, and the order is the whole
        # correctness argument** — the same one commit `1a2f4c5e` made for the
        # acceptance marker, now for the note count. The messages below start a
        # coordinator turn; that turn reads `blocked_notes` exactly once, at turn
        # start (`build_state_snapshot`), out of a runner `meta.json` that only
        # PROJECT_PLAN_STATE can move — and it cannot re-read it, because it
        # holds the project runloop lock the entry needs in order to land. A
        # projection appended AFTER the messages is therefore invisible to the
        # very turn the user's decision dispatched: §7.4's execution gate NO_OPs
        # that turn's first `create_room` and tells the user to resolve a note
        # they just resolved, with nothing left on any surface to wake the
        # coordinator again.
        #
        # Unlike the approve path this could not be fixed by swapping two
        # statements in the route: the fact does not predate the turn. The note
        # closes that lower the count are this transaction's own, which is why
        # they moved above the append (step 3) and the projection is built from
        # the in-memory plan rather than re-read from the sidecar.
        #
        # Prelude and not a same-batch afterthought: entries that wake nobody
        # take the lower versions, so ordering also covers what membership in
        # one batch would miss — a runner whose changelog fetch happens to end
        # between the two versions.
        # **THE ONE STAMP, AND THIS IS THE ONLY PLACE IT CAN GO.** Below every
        # `_close_note` and below `plan.revision += 1`, so `accepted_revision`
        # is the revision this batch produced rather than the one it started
        # from; above `plan_state_spec`, which reads the stamps into the prelude
        # the coordinator's next turn projects off. Move it under the append and
        # the turn these messages wake reads a `meta.json` that still calls the
        # plan unaccepted — the same ordering argument the block below makes for
        # the note count, and it is not a second argument, it is that one.
        if stamp_acceptance:
            _stamp_acceptance_locked(plan, body=prepared.body, by=by)

        # **THE SPEC LOCK'S START LINE, AND IT IS STAMPED HERE FOR THE ARGUMENT
        # DIRECTLY ABOVE.** This is the user's own review batch, so it is the
        # act that engages :func:`_spec_is_locked` on an unaccepted plan. The
        # round that records it is not opened until step 5, BELOW the append —
        # so a lock derived only from `rounds` would be invisible to the very
        # coordinator turn these messages wake, and that turn is the first one
        # that must not rewrite the document. Same ordering, same reason, same
        # bug class as the acceptance marker and the note count.
        _stamp_user_review_locked(plan, by=by)

        prelude = [prepared.spec] if prepared.wrote else []
        state_spec = plan_state_spec(project, plan)
        if state_spec is not None:
            prelude.append(state_spec)
        if owner_send:
            if outbound:
                posted = await post(prelude, outbound)
                for addressee, message_id in zip(addressed_order, posted):
                    message_ids[addressee] = message_id
            elif prelude:
                await runloop.append_batch(prelude)
        else:
            specs = prelude + message_specs
            if specs:
                await runloop.append_batch(specs)

        # ---- 5. the round, and the one write of the sidecar ----------------
        # `_save` below is what makes every in-memory mutation since step 3
        # durable, and it is deliberately the last thing: reaching it means the
        # batch landed.
        round_ = _open_round_locked(
            plan, by=by, room=room_name, addressees=sorted(by_addressee)
        )
        round_.room_created = room_created
        round_.unresolved = unresolved
        round_.revision = plan.revision if wrote else 0
        round_.sha = result.sha if wrote else ""
        round_.batch_comment = comment
        round_.applied_from = sorted(
            {
                _find_note(plan, r.note_id).by
                for r in applied_rows
                if r.kind == "accept" and r.note_id
            }
        )

        now = _now()
        for addressee, notes in by_addressee.items():
            for note in notes:
                digest = send_digest(note, addressee)
                note.round = round_.round_id
                round_.sends[note.id] = PlanSend(
                    note_id=note.id,
                    to=addressee,
                    digest=digest,
                    sent_at=now,
                    room=room_name,
                    message_id=message_ids.get(addressee),
                )

        plan.draft = None
        _note(plan, by, "submit", detail=round_.round_id)
        _auto_close_rounds(plan)
        _save(project, ctx, plan)
        return round_


def _last_send(plan: ProjectPlan, note_id: str) -> PlanSend | None:
    return last_send(plan.rounds, note_id)


def _round_of(plan: ProjectPlan, note_id: str) -> str:
    """The round a note went out in — what §5.7's ``409`` has to name, because
    *"already sent"* without *"where"* is not something a caller can act on."""
    for round_ in reversed(plan.rounds):
        if note_id in round_.sends:
            return round_.round_id
    return "?"


def _members_by_name(
    project: "Project", ctx: "ModelContext", room_name: str
) -> dict[str, str]:
    """``{participant name: id}`` for the room a review posts into, so a message
    can name who it expects an answer from."""
    if not room_name:
        return {}
    room = Chatroom.get(project.id, room_name, ctx)
    if room is None:
        raise PlanNotFoundError(f"Chatroom {room_name!r} not found")
    return {p.name: p.id for p in room.list_participants() if p is not None}


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


def ensure_approval_section(body: str) -> str:
    """``body`` with a ``## Approval`` section, appended if it has none.

    The seed template already carries one, so this fires on exactly one input:
    the coordinator-supplied v1 that ``POST /projects`` may pass instead of the
    template (:func:`~clawmeets.server.routes.projects.generate_plan_file`).
    Without it, such a project gets a go-note proposing a replacement for a
    section that is not there — unappliable, so the gate never releases and the
    project is stopped from its first turn with no surface saying why.

    Appended rather than refused: a caller handing over a plan body is not
    making a claim about approval mechanics, and failing project creation over a
    missing heading would be a strange thing to do to them.
    """
    return body if section_extent(body, APPROVAL_SECTION) else (
        body.rstrip("\n") + "\n\n" + approval_section(APPROVAL_PENDING)
    )


def _seed_go_note_locked(project: "Project", plan: ProjectPlan, body: str) -> str:
    """File the go-note. Returns its id, or ``""`` when one is already open.

    **An ordinary proposal, by the coordinator, to the user.** It is filed by
    hand rather than through :func:`add_note` for one reason — ``bootstrap`` is
    not a parameter of that door and must not become one — and it deliberately
    reproduces nothing else: addressing, the base capture and the quote
    derivation are all the same helpers ``add_note`` uses, so the note the user
    sees is shaped exactly like every other note in the tray — a comment
    carrying a diff, which is all a proposal ever was.

    Idempotent on the open go-note, so a retried project creation files one
    note, not two.

    **Regular projects only, and this is the same guard the gate carries.** On a
    front-desk project there is nobody in the accepting role — that is what
    ``surface == "regular"`` means in :func:`_plan_execution_blocked`. A
    go-note on such a project would be a
    proposal addressed to a ``user`` who is not reading it: never acceptable,
    open forever, and counted in ``plan_open_notes`` on the desk card of a
    project that is not waiting on anybody. Filing no note is the honest answer,
    and it keeps *"which shapes have an acceptance step"* to one predicate
    rather than two that have to agree.
    """
    if project.surface != "regular":
        return ""
    if open_go_note(plan) is not None:
        return ""
    base = _capture_base(body, APPROVAL_SECTION)
    proposal = approval_section(APPROVAL_GRANTED)
    keeper_name = keeper(project)
    ids = _add_notes_locked(plan, [
        PlanNote(
            id="",
            section=APPROVAL_SECTION,
            to=OWNER,
            by=keeper_name,
            at=_now(),
            comment=GO_NOTE_COMMENT,
            proposal=proposal,
            base_section=base,
            quote=_derive_quote(base, proposal),
            revision=plan.revision,
            bootstrap=True,
        )
    ])
    _note(plan, keeper_name, "note", section=APPROVAL_SECTION, detail=",".join(ids))
    return ids[0]


async def seed_go_note(project: "Project", ctx: "ModelContext") -> str:
    """File the go-note at project creation. **The gate, and the whole of it.**

    Called by ``init_plan_sidecar`` in the same request that seeds ``PLAN.md``,
    and **before** that route publishes ``PROJECT_PLAN_STATE`` — the ordering is
    load-bearing and its argument lives at the call site.

    **Seeded by the SERVER, not asked of the coordinator**, and that is the
    difference between a gate and a convention. Under *"the coordinator files a
    note asking for the go"* nothing blocks until the coordinator chooses to
    file one, so a coordinator that drafts a plan and immediately opens three
    workrooms would be doing something the server used to make impossible. One
    call at creation keeps the guarantee where it was.

    It replaces ``approve_plan`` and ``POST /plan/approve`` outright. Acceptance
    is now what happens when the user applies this note — one mechanism, the
    same one every other proposal uses, on a surface they are already reading.
    """
    async with _lock:
        plan = _load(project, ctx)
        body = _read_body(project, ctx)
        nid = _seed_go_note_locked(project, plan, body)
        if nid:
            _save(project, ctx, plan)
        return nid


async def publish_plan_state(
    project: "Project", ctx: "ModelContext", runloop: "ChangelogRunloop"
) -> bool:
    """Fan the sidecar's lifecycle facts onto the changelog (D13, §7.2).

    **The one publisher.** Every mutating plan surface calls it — the plan
    routes' ``_sync``, the upload interception's two exits, and
    ``init_plan_sidecar`` — and nothing else writes those five project fields.

    **Why a changelog append and not a ``meta.json`` write.** §7.2 says the
    acceptance fields are *"denormalized like ``report_published_at``"*, and that
    precedent is a **server-only** field: nothing replays it and a runner never
    learns it. But the two readers here are the **execution gate**
    (``build_state_snapshot``, ``models/agent.py``) and the coordinator's
    steady-state prompt block — both of which run in the **agent process**, off
    its own synced ``meta.json``, with no access to the server-side ``plan.json``
    this projects. A direct write reaches neither, so a gate built on one is
    inert in production **and green in any test whose server and runner share a
    ``ModelContext``**. The changelog is the only channel that carries server
    state to a runner; ``PROJECT_ALLOWLIST_UPDATED`` and ``DISPLAY_NAME_CHANGED``
    are the same shape.

    Idempotent, and that matters: it is called after every mutation, and a plan
    edit that moves no lifecycle fact must not cost a changelog version. Returns
    True iff it appended.

    **The append is one half; :func:`plan_state_spec` is the other.** A caller
    whose wake-up and whose projection have to land in ONE ``append_batch``
    (:func:`submit_review`) cannot use this function — it owns its own append —
    so the diff and the payload live in the builder and this is the thin
    ``append`` around it. One definition, two shapes; never a second diff.
    """
    from .project import Project

    # **Re-read the project, do not trust the caller's copy.** The diff below is
    # the entire idempotency guarantee, and it compares the sidecar against
    # ``Project``'s stored fields — which a caller that loaded its ``project``
    # before mutating the plan is holding a pre-mutation snapshot of. A route
    # that publishes twice in one request (``post_plan_review``: once in
    # ``submit_review``'s prelude, once through ``_sync`` on the way out) would
    # then append the identical projection twice, the second time only because
    # its own first append is invisible to a stale object.
    fresh = Project.get(project.id, ctx) or project
    spec = plan_state_spec(fresh, _load(project, ctx))
    if spec is None:
        return False
    await runloop.append(spec.entry_type, spec.payload)
    return True


def plan_state_spec(
    project: "Project", plan: ProjectPlan
) -> BatchEntrySpec | None:
    """The lifecycle projection as a batch spec, or ``None`` if nothing moved.

    Split out of :func:`publish_plan_state` so a caller that must place the
    projection **ahead of the message that wakes on it, inside one atomic
    batch**, can do so without restating the diff.

    Takes the ``plan`` rather than loading it, and that is the whole point:
    :func:`submit_review` calls this with its own in-memory plan *after* the
    transaction's note closes have been applied and *before* the append that
    persists nothing yet. A version that re-read the sidecar would project the
    state the submit is about to leave behind, which is exactly the bug.
    """
    incoming = (
        plan.seeded_at or None,
        plan.accepted_at or None,
        plan.accepted_revision,
        plan.accepted_spec_digest,
        open_notes_for_you(plan),
        # **MONOTONE, and the ``or`` is the guarantee.** The coordinator's
        # steady-state prompt is the only consumer, and what it must never do is
        # tell a coordinator the lock has LIFTED. ``first_user_review_at`` is
        # latched on the sidecar already, so this is belt and braces against the
        # one path that could clear it — a sidecar restored from an older copy —
        # and it costs a boolean or.
        plan.first_user_review_at
        or (
            project.plan_user_reviewed_at.isoformat()
            if project.plan_user_reviewed_at
            else None
        ),
    )
    current = (
        project.plan_seeded_at.isoformat() if project.plan_seeded_at else None,
        project.plan_accepted_at.isoformat() if project.plan_accepted_at else None,
        project.plan_accepted_revision,
        project.plan_accepted_spec_digest,
        project.plan_open_notes,
        (
            project.plan_user_reviewed_at.isoformat()
            if project.plan_user_reviewed_at
            else None
        ),
    )
    # Timestamps round-trip through ``datetime`` on the Project side, so compare
    # on the parsed value rather than the string — otherwise an unchanged plan
    # appends an entry on every mutation forever.
    if _same_plan_state(incoming, current):
        return None
    return BatchEntrySpec(
        entry_type=ChangelogEntryType.PROJECT_PLAN_STATE,
        payload=ProjectPlanStatePayload(
            project_id=project.id,
            plan_seeded_at=incoming[0],
            plan_accepted_at=incoming[1],
            plan_accepted_revision=incoming[2],
            plan_accepted_spec_digest=incoming[3],
            plan_open_notes=incoming[4],
            plan_user_reviewed_at=incoming[5],
        ),
    )


def _same_plan_state(a: tuple, b: tuple) -> bool:
    """Compare two projections, treating equal instants written differently as
    equal (``+00:00`` vs ``Z``, microsecond formatting)."""
    for x, y in zip(a, b):
        if isinstance(x, str) and isinstance(y, str) and x != y:
            try:
                if datetime.fromisoformat(x) == datetime.fromisoformat(y):
                    continue
            except ValueError:
                pass
            return False
        if x != y:
            return False
    return True


def changed_since_acceptance(body: str, plan: ProjectPlan) -> bool:
    """Accepted, and the spec digest has moved since (§3.3).

    Never reaches the execution gate: §7.4 decision 1 is explicit that the gate's
    trigger is :func:`open_notes_for_you` and **explicitly not** this.
    """
    return bool(plan.accepted_at) and spec_digest(body) != plan.accepted_spec_digest
