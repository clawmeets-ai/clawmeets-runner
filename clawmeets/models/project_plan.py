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
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from clawmeets.models.agent import Agent
from clawmeets.models.chatroom import Chatroom
from clawmeets.models.plan_markdown import (
    SPEC,
    PlanLimitError,
    body_sha,
    boxes_labelled,
    find_boxes,
    flip_boxes,
    drops_heading,
    duplicated_heading,
    extent_key,
    extract_shorthand,
    first_changed_line,
    heading_line,
    heading_slug,
    legacy_spec_digest,
    named_criterion,
    normalize_spec_text,
    parse_criteria,
    parse_sections,
    section_holding_quote,
    section_layers,
    quote_from_line,
    relevels_heading,
    rename_pair,
    render_diff,
    replace_section,
    section_extent,
    spec_digest,
    split_by_section,
    unclaimed_criteria,
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
#: ``models/desk_sop.py:41``, ``models/desk_todo.py:38``, ``models/brief_tab.py:45``.
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

#: A keeper write's ``--why`` — one line saying what the write changes and
#: why. On a write the spec lock refuses it leads the proposal the user is
#: handed (:func:`_file_spec_lock_locked`), so it is capped like a changelog
#: line: the rationale belongs in ``user-communication``.
MAX_WHY_CHARS = 600

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

#: The note ladder. **ADVISORY, NEVER A VALIDATION SET**, and this is a
#: commitment rather than an observation:
#:
#: * It is never narrowed into a ``Literal``, an ``Enum`` or a pydantic
#:   validator. :attr:`PlanNote.status` is a plain ``str`` and stays one, so a
#:   sidecar written by a newer build is readable by an older one — an unknown
#:   word deserializes, renders as itself, and raises nothing.
#: * A reader treats **anything that is not ``open`` as terminal** and
#:   enumerates nothing. Every note-state partition in this module and every
#:   caller of it tests ``== "open"`` or ``!= "open"``; not one lists the
#:   terminal words. That is what makes adding a word to this tuple a local
#:   change rather than an audit of every surface.
#: * ``folded`` is the sixth, and it answers the question the first five could
#:   not: a proposal the user chose to *keep alongside* another one on the same
#:   section. ``applied`` means "its text IS the section"; ``rejected`` means
#:   "its text is NOT in the document"; ``folded`` means "its text is in the
#:   section that landed, but the section is not its text". Closing such a note
#:   ``rejected`` — which is what this module did before ``folded`` existed —
#:   tells its author their work was dropped when it is sitting in the plan.
NOTE_STATUSES = ("open", "answered", "applied", "rejected", "dismissed", "folded")

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

#: Written by the server onto a note its own author replaced before the user
#: ever saw it (:func:`_supersede_prior_locked`). **Not** a variant of
#: ``SUPERSEDED_REASON`` above, and the split is deliberate: that one is
#: ``submit_review`` telling an agent the user chose someone else's text over
#: theirs, which is a DECISION about two proposals that both reached the desk.
#: This one is a filing-time collapse of one author's own drafts, and nobody
#: decided anything — there was only ever going to be one row.
REPLACED_UNSENT_REASON = (
    "replaced by {id} — a newer proposal on the same section from the same "
    "author, filed before this one was sent"
)

#: Written by the server on a row the user chose to **fold**, and read by the
#: agent whose proposal was folded away. The counterpart of
#: :data:`SUPERSEDED_REASON`, and the two are mutually exclusive on any one row:
#: *superseded* means the text did NOT reach the document, *folded* means it
#: DID. Neither sentence may say the other's word — an agent told "superseded"
#: about text that landed may re-file work that is already in the plan, which is
#: the failure this reason exists to remove.
FOLDED_REASON = (
    "folded into @{by}'s change on the same section "
    "— your text is in the section that landed"
)

#: The confirm note's comment. Addressed to the user, so it says what
#: dismissing DOES rather than naming the mechanism: the user is not reading
#: about proposals, they are deciding whether work may start.
#:
#: **It carries no diff and names no section**, and both are the same decision.
#: There used to be a ``## Approval`` section for the acceptance to land in, and
#: this note used to be a proposal to replace its body — which meant approving
#: was a trip to a card holding a one-line diff, in a grammar unlike every other
#: row in the list. The section is gone (see :data:`SEED_TEMPLATE`), so there is
#: nothing to travel to, and the note is anchored to the document as a whole
#: (``section == ""``) the way :class:`PlanNote` has always allowed.
#:
#: **The wording has to say that dismissing starts work and that replying does
#: not**, because those are the only two acts this row offers and they are not
#: symmetric: one releases the coordinator, the other leaves the gate up.
GO_NOTE_COMMENT = (
    "Nothing in this plan is waiting on an answer from you. This is the start "
    "gate: dismiss it to start work on the plan as written. Reply instead if "
    "you want changes; that keeps the gate closed."
)

#: **The confirm note's OTHER wording, and the one a project is born with.**
#: :data:`GO_NOTE_COMMENT` says *"Nothing in this plan is waiting on an
#: answer from you"*, and at
#: ``POST /projects`` that sentence is false: :func:`seed_go_note` files the row
#: in the same request that seeds ``PLAN.md``, before the coordinator has taken
#: a turn, so the document it invites a read of is the bare
#: :data:`SEED_TEMPLATE`. The coordinator's spec-stage contract then spends its
#: whole first turn drafting and hands the draft to the roster
#: (``clawmeets plan consult``) *before the user sees any of it* — and the
#: ``BATCH_COMPLETE`` that brings the answers back is a worker turn away. Every
#: regular project with a roster therefore had a window, minutes wide, in which
#: the row on the user's desk invited them to confirm a draft that was still out
#: for review. A dismiss inside it stamps ``first_accepted_at``, and the STEP 4
#: fold-in of the consult the user never knew was running lands on the far side
#: of that line — refused as a write, filed as a deviation. The user's first
#: sight of the plan is then an off-contract alarm about a contract they were
#: never shown.
#:
#: **The row does not move and the gate does not move; only these bytes do.**
#: The seed stays where it is for the reason :func:`_sync_go_note_locked`
#: records — a confirm note born when the coordinator sends a review is a gate
#: the coordinator can skip — so this is one open, spec-layer, user-addressed
#: note either way and ``_plan_execution_blocked`` cannot tell the two apart.
#: :func:`_sync_go_note_locked` swaps it for :data:`GO_NOTE_COMMENT` on the
#: keeper's first quiet round, which is the moment the plan is actually put in
#: front of the user.
#:
#: **It still offers the dismiss, and saying so is the point.** The owner is the
#: decider; a row that withheld the verb would be a lock aimed at the decider
#: (:func:`_spec_is_locked`). What changes is that taking it now reads as
#: *"start on the draft as it stands"* — which is what it always did — instead
#: of as an answer to a question nobody had asked yet. It also leaves a
#: coordinator that never reaches its review send unable to strand the user:
#: the wording names the way out.
#:
#: **"ONLY YOUR OWN HAND" IS ADDRESSED TO THE ONE READER WHO IS NOT THE USER.**
#: The previous wording ended *"dismiss now only if you want work to start"*,
#: which is a sentence for a human at a desk — and it is also read by an agent
#: holding the owner's credential, to whom it looks like an offered option. On
#: ``onboard-angel-investor`` that is exactly how it was read. The bytes are not
#: the guard (:func:`_refuse_presigned_go_note` is), but a row whose text
#: contradicts the refusal is a row that invites the retry.
#:
#: ``{keeper}`` is interpolated at the filing site, which already has the name.
GO_NOTE_DRAFTING = (
    "@{keeper} is drafting this and has it out for review with the agents. "
    "Nothing to do yet — it will come back to you for a confirm. Only your own "
    "hand may dismiss this row, and dismissing it starts work on the draft as "
    "it stands."
)

#: Why the confirm note went away without being approved: the round the
#: coordinator is sending carries real questions, so the answers to THOSE are
#: the gate and a confirm row beside them would be false — the note says
#: *"Nothing in this plan is waiting on an answer from you"*. Written as the note's ``resolution`` so the closed
#: row still reads in the history. See :func:`_sync_go_note_locked`.
GO_NOTE_RETIRED = (
    "this round carries questions — answering them is what the plan is waiting "
    "on, and the confirm comes back when nothing is open"
)


#: The section that says what the user's approval does **not** buy, and the
#: heading it replaces.
#:
#: **This is not ``## Guardrails`` renamed for taste.** That heading invited
#: every constraint anyone could think of and duly got them: across the 151
#: seeded plans carrying one, the median ran 108 words, the 90th percentile 615
#: and the longest 5,029 — and not one was ever left empty. Most of that content
#: had a better home and was sitting there because the heading was open-ended.
#: A scope boundary (*"exactly these six items, no more"*) restates the Goal, so
#: building a seventh thing already violates something observable and already
#: produces a deviation. A quality bar (*"every recommendation cites a source
#: the reader can open"*) is an acceptance criterion by definition. Sequencing
#: the keeper chose is milestone content and costs the user no decision when it
#: changes.
#:
#: **What is left is the one class the deviation channel cannot see, and that
#: asymmetry — not importance, not frequency — is what earns it a heading.** A
#: deviation fires when the spec MOVES: someone notices Goal or Acceptance
#: Criteria no longer describe the work and files a note to the user. An agent
#: that submits the order, sends the email or ships the deploy has not moved the
#: spec — it has OVER-satisfied it. There is no criterion for it to notice, so
#: there is nothing for it to file, and by the time anyone could read a note the
#: act cannot be taken back.
#:
#: So the test for a line belonging here is: **violating it cannot be undone by
#: more work.** Extra code is deletable and extra research is discardable; a
#: placed order is neither. Expect most plans to say *"None."*
NOT_AUTHORIZED_SECTION = "not-authorized"
NOT_AUTHORIZED_HEADING = "## Not Authorized"

#: The pre-2026-09-09 heading, read as a **fallback and never written**. The
#: plans already on disk carry their denials under it, and a seed template
#: changing is not a migration — those documents are never rewritten. Reading
#: only the new slug would mean the rule binds on no project that exists today,
#: which is exactly the failure the change is meant to fix.
LEGACY_NOT_AUTHORIZED_SECTION = "guardrails"
LEGACY_NOT_AUTHORIZED_HEADING = "## Guardrails"

#: Cap on what reaches a prompt. Generous against the new shape — five lines of
#: denial is around 300 characters, so the cap never fires on a section written
#: to the hint — and deliberately brutal against the legacy one, where a
#: 5,029-word section would otherwise displace the turn's actual instructions.
NOT_AUTHORIZED_MAX_CHARS = 1200
NOT_AUTHORIZED_MAX_LINES = 12

#: Truncation is **announced, never silent.** A model shown four denials and
#: told there are more will open the document; one shown four and told nothing
#: will act as though four is all there are — which is the same failure as not
#: injecting the section at all, only harder to notice.
NOT_AUTHORIZED_TRUNCATED = (
    "... (truncated - read `{heading}` in PLAN.md in full before any outward act)"
)

#: What an unfilled section says. Either answer means there is nothing to
#: enforce, so nothing is injected: a prompt carrying the seed's own italic hint
#: would teach the model to write more hint.
_NOT_AUTHORIZED_EMPTY = {"none", "none.", "n/a", "na", "-", "\u2014"}


def _looks_like_placeholder(text: str) -> bool:
    """Is this still the seed's italic hint rather than an answer?

    Fully wrapped in underscores, which is how every slot in
    :data:`SEED_TEMPLATE` marks itself unfilled (``_Not yet approved._`` is the
    same device). A real denial written entirely in italics would be missed and
    that is the right trade: the cost is one un-injected line, where the reverse
    error injects instructional prose into every worker turn on the project.
    """
    return text.startswith("_") and text.endswith("_")


def _cap_not_authorized(text: str, heading: str) -> str:
    """Trim to :data:`NOT_AUTHORIZED_MAX_LINES` / ``_MAX_CHARS``, announcing it."""
    capped = "\n".join(text.splitlines()[:NOT_AUTHORIZED_MAX_LINES])
    if len(capped) > NOT_AUTHORIZED_MAX_CHARS:
        capped = capped[:NOT_AUTHORIZED_MAX_CHARS].rstrip()
    if capped == text:
        return capped
    return capped + "\n" + NOT_AUTHORIZED_TRUNCATED.format(heading=heading)


def not_authorized_state(body: str) -> str:
    """What this plan forbids outright, capped and ready to paste into a prompt.

    ``""`` when there is nothing to say, and **nothing to say is the expected
    answer** — most projects touch nothing irreversible. Three cases collapse to
    it: no such section, a section still holding the seed's italic hint, and a
    section that says *"None."* A coordinator that leaves the hint in place has
    not authorized anything unusual, which is the same state as writing "None"
    and is treated as such rather than as an omission to complain about.

    **Both slugs are read, new one first.** ``## Not Authorized`` wins where
    both exist, which is the only case in which someone deliberately wrote under
    both headings. An empty new section falls through to the legacy one, so a
    coordinator that added the new heading without moving the content still gets
    the content enforced.

    **This is not merely informational**: it is the
    only route by which the section reaches a model at all, on either the
    coordinator or the worker side. It runs in the **agent** process against the
    synced document, so it must stay a pure function of the bytes — no sidecar,
    no network.
    """
    for slug, heading in (
        (NOT_AUTHORIZED_SECTION, NOT_AUTHORIZED_HEADING),
        (LEGACY_NOT_AUTHORIZED_SECTION, LEGACY_NOT_AUTHORIZED_HEADING),
    ):
        extent = section_extent(body, slug)
        if extent is None:
            continue
        # Drop the heading line; the caller asked what the section FORBIDS.
        _, _, rest = body[extent[0]:extent[1]].partition("\n")
        text = rest.strip()
        if not text or _looks_like_placeholder(text):
            continue
        if text.lower() in _NOT_AUTHORIZED_EMPTY:
            continue
        return _cap_not_authorized(text, heading)
    return ""


#: §5.1's seed template, verbatim. **One template** — the ``full`` variant went
#: with the ``plan_mode`` that selected it. The coordinator may replace every
#: section: this is a starting document, not a schema (§2.4). ``<title>`` is the
#: only substitution.
#:
#: **``## Approval`` IS GONE, AND ITS GOING IS THE CHANGE RATHER THAN A
#: TIDY-UP.** It existed to be the *target of the go-note* — a section whose
#: body the user's acceptance replaced with one fixed sentence — and being a
#: diff target is what made approving unlike every other act on this surface:
#: the user travelled to a card, read a one-line diff they had no say in, and
#: accepted it. The confirm note now carries no diff at all
#: (:data:`GO_NOTE_COMMENT`) and is dismissed in the list, in the same grammar
#: as every other row.
#:
#: **Three server guards went with it, and each was load-bearing only because
#: the section was.** A write that dropped the heading left the go-note
#: unappliable and the project silently gated, so it was refused; a second
#: ``## Approval`` buried in another section took the ``approval`` slug and
#: demoted the real one, so that was refused too; and an accept that closed the
#: note without writing the line marked a plan approved that said it was not.
#: None of the three has anything left to protect. The GENERAL duplicate-heading
#: refusal in :func:`_prepare_locked` stays exactly where it is — it guards
#: every section, and the approval incident was its occasion, not its subject.
#:
#: **And the digest is now what it always should have been.**
#: :func:`spec_digest` hashes the spec-layer top-level sections, ``## Approval``
#: among them — so the act of approving MOVED the hash, which is why the stamp
#: has to be taken from the post-write body. What is left is Goal, Not
#: Authorized and Acceptance Criteria: *"the plan moved without you"* now means
#: exactly that.
#:
#: **The layered shape** (:func:`plan_markdown.section_layers`): Goal,
#: Not Authorized and Acceptance Criteria are the user's — unmarked, therefore
#: spec, therefore locked once the user has looked at the plan. ``## Milestones``
#: carries ``<!-- layer: detail -->`` and is the keeper's to re-cut freely, and
#: every ``### M<n>`` inside it inherits that.
#:
#: **Criteria are hoisted out of the milestones**, and that hoist is the
#: precondition for everything else: "milestones are detail, acceptance criteria
#: are spec" is self-contradictory while the criteria live INSIDE a milestone,
#: because freeing the milestone frees whatever is nested in it. ``<m>`` in
#: ``AC-<m>.<n>`` now addresses a criteria GROUP rather than a milestone number,
#: which cost no parser change — nothing ever read it as a milestone index. A
#: milestone cites the ids it advances with ``<!-- advances: … -->`` instead of
#: owning them, and dropping such a claim is refused
#: (:func:`_coverage_regressed`).
#:
#: **``**Deliverable:** `<specific_file.md>`​`` is gone, and its going is the
#: point rather than a tidy-up.** That one line seeded two biases into every
#: project this system has ever created: that a plan is about files (a coding
#: bias, on a template that also has to serve research, design and operations),
#: and that a criterion names an OUTPUT rather than a behaviour. What a
#: criterion will be shown by now lives in an ``<!-- evidence: … -->`` comment,
#: which is inert to the digest and therefore free for the keeper to change
#: without spending one of the user's decisions.
#:
#: **The slot no longer asks for a falsifier, and the reason is what the rule
#: was always for.** Criteria used to come back unit-test-shaped — naming a
#: function, a selector, a file path — so any ordinary implementation change
#: moved what the plan *said* and dragged the user into a decision they should
#: never have been asked for. The wording chosen to lift them out of that was
#: "an invariant with a falsifier", and the device ate the intent: every
#: criterion arrived as ``<assertion>; falsified by <negation>``, unreadable,
#: usually a restatement, and still pinned to a CSS class. A slot is a pattern
#: to copy, not a principle to apply, so the slot now carries the property
#: itself — observable behaviour, at an altitude a rename cannot disturb — and
#: the counter-case construction is taught nowhere.
SEED_TEMPLATE = """# {title}

## Goal

_What outcome this produces and why it is worth producing. Two paragraphs maximum._

## Not Authorized

_What approving this plan does NOT buy. One line per irreversible outward act
the agents must not take — sending, publishing, ordering, deploying, deleting,
spending. The test is that violating it cannot be undone by more work, which
is why a scope boundary belongs in Goal and a quality bar in Acceptance
Criteria: those are recoverable. Most plans say "None."_

## Acceptance Criteria

_Observable outcomes, one plain sentence each — what must be true for whoever
receives this, not how it gets made. If an ordinary change in method or
implementation would break one, it is too fine-grained. A quality bar a
reader can check IS a criterion; a scope boundary belongs in Goal. Grouped;
`<m>` is the group._

### G1: <the promise this group makes>

- [ ] **AC-1.1** — <who or what> <observable outcome>.
      <!-- evidence: a test, a file, or whatever a reader could check it against -->

## Milestones <!-- layer: detail -->

### M1: <action>  <!-- advances: AC-1.1 -->

- [ ] **M1** — unassigned
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
    #: **A THIRD READER JOINED THE TWO ACCEPT DOORS**, and it is the one a user
    #: actually sees: :func:`section_new`. The two above act on the distinction
    #: and were always right; the LABEL on the note never asked, so sixteen
    #: proposals to ADD a section were shown reading *"the section is no longer
    #: in the document"*. Same predicate, same field, one more caller.
    #:
    #: Still off by default, so a path that has not thought about it gets the
    #: safe answer: an unresolvable slug is a ``409``.
    create: bool = False


class PlanTick(BaseModel):
    """*"Tick the box labelled M2."* — a write that carries no section text.

    **The only input to this module that names WHAT to change without saying what
    the result should say**, and the asymmetry is the whole reason it exists. A
    :class:`SectionEdit` is a full-section replacement, so ticking a checkbox
    through one meant resupplying every byte of ``## Milestones`` — which is what
    :mod:`clawmeets.llm.prompt_builder` told coordinators to do, and it is a trap.
    The writer's copy of a section is its own idea of that section; on an accepted
    plan any drift from the document promotes a one-character edit into a spec
    move, the whole write is refused atomically, and **every** tick in it is lost
    — including the ones in sections that had not drifted at all. On
    ``ig-incumbent-teardown`` that cost thirteen ticks, produced four deviation
    notes on a closed project, and ended with the coordinator telling its owner in
    a published report that nothing but the boxes had changed. It was wrong, and
    the refusal it read gave it no way to know.

    So a tick names a box and nothing else. The section text is materialized in
    :func:`_prepare_locked` from the body it already holds under ``_lock``, which
    is the only authoritative copy there is — so a tick **cannot be stale**, and
    because :func:`normalize_spec_text` folds marker state it **cannot move the
    spec digest** either. Two of the three ways a write is refused are closed by
    construction rather than by a check; the third, ``may_write``, still applies.

    ``label`` addresses the box by the token that opens its text — ``M2``,
    ``AC-3.1`` (:func:`_box_label`). It resolves against the live document or it
    is a ``400`` naming what it saw: a tick must never fall through to creating a
    section, which is exactly what the old whole-section path did with a retitled
    milestone's stale slug.
    """

    label: str
    #: ``False`` unticks. Present so the verb is symmetric — a milestone reopened
    #: is as ordinary as one finished, and an agent that can only tick would
    #: reach for a section replace to undo one.
    checked: bool = True


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
    #: **WHY the write in ``locked`` was refused, when the answer is a layer
    #: flip.** ``[(slug, before, after)]`` for every top-level section whose
    #: ``<!-- layer: … -->`` resolution changed.
    #:
    #: It exists because the rows in ``locked`` can fail to name the cause. The
    #: layer manifest is hashed, so dropping ``<!-- layer: detail -->`` from a
    #: heading IS a spec move — but ``_rows_worth_showing`` filters rows against
    #: the STORED body, where that section is still ``detail``, so the one row
    #: that explains the refusal is dropped and what survives are the derived
    #: child rows. On ``onboard-angel-investor`` a coordinator rewrote
    #: ``## Milestones`` without its marker and was told, four times, that it
    #: *"could not add the section ``m1-…``"* — four slugs the document has
    #: never had, and no mention of the marker. It retried, filed 17 notes, and
    #: concluded it had a stale local copy.
    #:
    #: Empty whenever no layer moved, which is the ordinary spec edit.
    relayered: list[tuple[str, str, str]] = Field(default_factory=list)
    #: **WHAT moved, for the ordinary refusal — the case ``relayered`` does not
    #: cover.** Slugs whose normalized text differs, and slugs the stored
    #: document does not have.
    #:
    #: ``relayered`` was the first answer to *"a refusal that does not name its
    #: cause is one a model bisects"*, and it only ever covered the layer flip.
    #: The plain arm stayed silent, and it fails the same way: on
    #: ``ig-incumbent-teardown`` a coordinator re-sent ``## Acceptance Criteria``
    #: and ``## Milestones`` built from its pre-review draft, was handed two note
    #: ids and the sentence *"ticking a checkbox still applies"*, and told its
    #: owner — in ``user-communication`` and then in a published report — that
    #: nothing but the boxes had changed. Two sections had lost three acceptance
    #: criteria between them. Nothing it was told contradicted its conclusion.
    #:
    #: ``added`` is its own field rather than part of ``moved`` because the two
    #: have different remedies and the note's own wording conflates them: a slug
    #: the plan has never had reads as *"could not add the section …"*, which
    #: describes a deliberate addition, when the cause is almost always a
    #: heading that was retitled and re-slugged underneath the writer.
    moved: list[str] = Field(default_factory=list)
    added: list[str] = Field(default_factory=list)
    #: **How many per-section rows the refusal dropped without filing a note.**
    #: ``_rows_worth_showing`` is right to drop them — nobody should be asked to
    #: accept a checkbox tick or a re-cut milestone — but the WRITER has to know
    #: its other sections did not land either, because the write is
    #: all-or-nothing. Thirteen sections, four notes and no number is how *"only
    #: the boxes"* became a conclusion a model could reach honestly.
    dropped: int = 0
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
    #: **The layer ``section`` was in WHEN THIS NOTE WAS FILED** — stamped once
    #: by :func:`add_note`, never recomputed, never an input on any route.
    #:
    #: It exists so :func:`open_notes_for_you` can stop counting a note about a
    #: section the keeper was free to rewrite without asking. A note in the
    #: detail layer is a remark about work the plan already delegated; halting
    #: the whole project over it spends the owner's attention on a decision the
    #: document says is not theirs.
    #:
    #: **Stamped, not derived, and that is the whole design.** The layer could
    #: be looked up at count time from ``section_layers(body)``, and must not
    #: be, for three reasons that compound:
    #:
    #: 1. **The counter has no document.** The gate runs in the agent process
    #:    off ``project.plan_open_notes``, a projection over the wire; that
    #:    process has ``PLAN.md`` but not the sidecar, and the server-side
    #:    producers include :func:`plan_summary_for`, which documents at length
    #:    that it never reads a body. A derived answer drags up to 512 KB of
    #:    markdown into the one path built to avoid it, per project, per list.
    #: 2. **A stale anchor has no derivable answer.** Sections get renamed and
    #:    deleted under notes that outlive them, and a lookup that misses has to
    #:    invent a layer — defaulting the misses to detail silently releases the
    #:    gate on live questions. Filing time never faces the question: the
    #:    section is right there.
    #: 3. **It would be re-gameable.** A keeper cannot mark a section ``detail``
    #:    to escape the lock — the manifest is hashed, so the flip comes back as
    #:    a proposal — but a marker flip applied later must not retroactively
    #:    un-block notes already filed against the section as spec.
    #:
    #: Defaults to ``SPEC`` and every producer other than :func:`add_note`
    #: leaves it there, which is the conservative direction and is deliberate on
    #: two of them: :func:`_file_conflict_locked` and the spec-lock filer report
    #: refusals the caller could not work around, and ``_coverage_regressed``'s
    #: note is anchored to the milestone it is *about* — the one alarm the
    #: layered lock added, which a detail-layer exemption would delete. Every
    #: note already on disk loads without the key and blocks exactly as it did.
    layer: str = SPEC
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
    #: It is an ordinary **comment** in every other respect: it counts toward
    #: :func:`open_notes_for_you`, it blocks ``create_room`` through the one
    #: gate every other open note blocks through, it renders in the comment list
    #: like any other, and the user closes it with the ``Dismiss`` they already
    #: know. It carries no ``proposal`` and no ``section`` — it is anchored to
    #: the document as a whole — which is what removed the trip to a card
    #: holding a one-line diff nobody had a say in.
    #:
    #: This flag buys exactly three exceptions, each written down where it
    #: applies:
    #:
    #: 1. a user **reply** does not close it (:func:`_reply_closes_parent`) —
    #:    every other note closes ``answered`` on a reply, and on this one that
    #:    would mean typing a question released the gate;
    #: 2. **``dismiss`` is the ONLY verb that closes it**, and it stamps
    #:    ``accepted_*`` (:func:`_refuse_go_note_close`,
    #:    :func:`_stamp_acceptance_locked`). Every other close — reject,
    #:    answered, a supersede, a fold — releases the gate without meaning yes,
    #:    so every other close is refused;
    #: 3. the coordinator's own review send retires it when the round carries
    #:    questions and re-files it when it does not
    #:    (:func:`_sync_go_note_locked`).
    #:
    #: Never an input on any route — set by :func:`_seed_go_note_locked` alone,
    #: and by nothing a caller can reach. **It may be filed more than once**,
    #: which is the one thing that changed about its lifecycle: a round the
    #: coordinator sends with nothing to ask re-files it, so a plan that came
    #: back quiet after a round of questions still comes back for a confirm.
    #: Acceptance itself stays one-way — once ``accepted_at`` is set it is never
    #: filed again.
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
    #: **The user chose to keep this row's text alongside the winner's.**
    #: Optional, opt-in, and inert everywhere except one branch of
    #: ``submit_review``'s superseded loop: absent or ``False`` reproduces the
    #: pre-fold behaviour character for character, which is what makes the field
    #: additive rather than a change of meaning.
    #:
    #: Set by the client on the row(s) being folded **away**, never on the
    #: winner, and it carries no status, no author name and no id. The winner's
    #: identity is entirely server-derived — see ``_close_spec_for_superseded``
    #: — because the winner is not decided until submit, by
    #: :func:`_collapse_text_rows` over the whole tray, so any client-side
    #: snapshot of it is stale by construction.
    #:
    #: Honoured only where the loop already acts: an ``accept`` row carrying a
    #: ``note_id`` that ``_collapse_text_rows`` classified superseded. On every
    #: other row shape it is read by nobody, so it needs no validation arm.
    folded: bool = False


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
    #: Notes the sender named that are addressed to the SENDER — its own inbox,
    #: not a send. Skipped and reported (:func:`submit_review`): mailing them
    #: back to their addressee opens a round nobody will ever answer.
    own_inbox: list[str] = Field(default_factory=list)


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
    #: **WHICH HAND PUT THE OWNER'S NAME TO IT.** ``accepted_by`` is who the
    #: decision belongs to and is always ``"user"`` on a regular project; this is
    #: how their credential reached the server.
    #:
    #: ``"owner"`` — a real user session (the browser, or ``--token``).
    #: ``"assistant"`` — the owner's ``{username}-assistant`` bearer, which
    #: ``resolve_credential`` accepts AS the owner (``plan resolve --as-user``).
    #: ``"coordinator"`` — a front-desk plan, which is final without the
    #: requester's approval and never carried an owner in the accepting role.
    #:
    #: It was ``"owner"`` unconditionally, which made the record unable to say
    #: what actually happened on ``onboard-angel-investor``: the coordinator ran
    #: ``--as-user`` against the confirm note and every field in the acceptance
    #: said the user. See :func:`_stamp_acceptance_locked`.
    accepted_via: str = ""          # "owner" | "assistant" | "coordinator"
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
    #: **The plan's ``revision`` at the last review round the OWNER was on the
    #: receiving end of.** Unlike ``first_user_review_at`` it moves every time,
    #: because the question it answers is *"has this document changed since YOU
    #: last looked at it?"* and that has a new answer every round.
    #:
    #: It is not :attr:`PlanNote.revision`'s job and cannot be derived from it.
    #: That field records what a NOTE's author saw; the fact needed here is what
    #: the READER saw, and the two coincide only for a note the reader wrote.
    #: Deriving it from the confirm note's copy is worse than merely wrong —
    #: that note is filed by the server, not by a reader, so its number is the
    #: revision at whatever moment the server filed it and never the revision
    #: the owner was shown.
    #:
    #: Stamped in :func:`submit_review` by :func:`_stamp_owner_seen_locked`,
    #: **after** the batch has rendered, so the message the owner is reading
    #: reports the span it closed rather than a span of zero.
    #:
    #: Defaults ``0``, and ``0`` means *"never presented"* — the banner and the
    #: receipt changelog are both omitted on it, so a plan written before this
    #: field existed renders exactly as it did and the owner's first round is
    #: not handed a changelog of a draft they have never read.
    owner_last_seen_revision: int = 0
    #: When the coordinator first PUT THIS PLAN IN FRONT OF the user — the
    #: instant its first review round opened. The spec lock's other half
    #: (:func:`_plan_was_offered`): ``accepted_at`` says they signed,
    #: this says there was something to sign, and a plan can carry the first
    #: without the second because the confirm row is dismissible from the desk
    #: card while it still reads :data:`GO_NOTE_DRAFTING`.
    #:
    #: **LATCHED IN :func:`submit_review`, NOT DERIVED FROM ``rounds``**, and
    #: the reason is the ordering note on :func:`_stamp_user_review_locked`'s
    #: callsite: the round is not opened until step 5, BELOW the append that
    #: carries this batch's messages — so a predicate reading ``rounds`` is
    #: False for the projection those same messages ride with, and the very turn
    #: they wake is told it may still rewrite the document. The latch is stamped
    #: above the projection instead, so the batch that offers the plan is the
    #: batch that says so.
    #:
    #: Author-agnostic on purpose: an owner-opened round means the user was IN
    #: the document, which is the same fact from the other side.
    first_offered_at: str = ""
    created_at: str = ""
    updated_at: str = ""

    @model_validator(mode="after")
    def _backfill_first_offered(self) -> "ProjectPlan":
        """Recover ``first_offered_at`` from ``rounds`` — the same trick its
        sibling below uses, and for the same reason: the fact was already on
        disk and had no field.

        Every plan written before this shipped reads correctly on its next load
        and there is no migration script. That matters more here than it looks:
        on the server this shipped against, 11 of 17 accepted plans carry a
        blank ``first_user_review_at``, and keying the lock on THAT would have
        silently unlocked ten ratified plans. All eleven have rounds, so this
        backfill is what keeps them locked.

        Fills a blank only, so the latch always wins; and because the trim takes
        rounds from the FRONT, the recovered instant can be later than the true
        first offer on a very busy plan. That is harmless — every reader asks
        whether it is set, never what it says.
        """
        if not self.first_offered_at:
            opened = [r.opened_at for r in self.rounds if r.opened_at]
            if opened:
                self.first_offered_at = min(opened)
        return self

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

    **AND ONLY IF THE NOTE IS IN THE SPEC LAYER** (``PlanNote.layer``). A
    detail-layer section is one the keeper may rewrite with ``plan update`` and
    no refusal, so a note filed against one is a remark about work the document
    has already delegated — and stopping the entire project over it asks the
    owner for a decision the plan says is not theirs to make. The precedent is
    the write receipt, which is *visible, never blocking*, on the same reasoning:
    a receipt that counted "would turn every pre-acceptance keeper write into a
    project-wide stop, and the coordinator would be halted by its own
    bookkeeping."

    **The honest cost, stated because it is a behaviour change and not only a
    count change.** A detail-layer note is where a keeper asks *"I want your
    taste on this before I build it"*, and the halt is what converted that
    question into an answer. Non-blocking means the keeper asks and then builds
    anyway. That is what the layer already says — detail is the keeper's — so
    the completion is upstream, in the keeper deciding rather than asking; but
    the note is still SENT, still lands in the plan editor's comment list, and
    still stands in the note history — so nothing is hidden, only un-halted.
    What it loses is the two surfaces that read this integer: the desk card's
    *"N for you"* pill and the gate itself.

    The layer is READ OFF THE NOTE and never off the document: this function has
    no body, by design (it is called from paths that document at length that
    they never read one), and a note's layer is a fact about when it was filed.
    See ``PlanNote.layer``.
    """
    return sum(
        1
        for n in plan.notes
        if n.status == "open" and n.to == OWNER and n.layer == SPEC
    )


def execution_blocked(surface: str, open_notes: int) -> int:
    """§7.4's gate — **the rule, once** — as the count that motivates it.

    ::

        blocked == surface == "regular" and open_notes_for_you > 0

    **Two conjuncts, and they now have two callers in two processes.** The rule
    used to live entirely inside ``_plan_execution_blocked`` in the AGENT
    process, which reads its count off the runner's synced ``meta.json``. That
    is still a caller and still correct — the whole argument for each conjunct
    is written out there and is not repeated here. What changed is that
    ``create_chatroom`` now asks the same question SERVER-side, off the sidecar,
    because the agent-side answer is structurally one turn stale:

    * the snapshot the validator reads is frozen once per turn, deliberately, so
      the retry loop has a fixed target to terminate against; and
    * un-freezing it would not help, because the count travels to the runner as
      a ``PROJECT_PLAN_STATE`` changelog entry applied through the runloop — and
      a coordinator turn is holding the runloop lock while it files the very
      notes that should stop it.

    So on the one shape the gate exists for — a coordinator that files blocking
    notes and emits ``create_room`` inside a single turn — the agent-side check
    is blind by construction. Measured on ``onboard-angel-investor``: the turn
    ran 12:00:08 → 12:06:02, two spec notes to the user were filed at 12:04:42
    and 12:04:43, and ``create_room milestone-2-field-research`` passed at
    12:06:02 against a snapshot that had counted zero six minutes earlier.

    **The count is the parameter and not the plan**, because the two callers
    hold different sources for it and neither can hold the other's: the agent
    has ``Project.plan_open_notes`` (the projection) and the server has
    ``open_notes_for_you(plan)`` (the sidecar it is derived from). Passing the
    number in is what lets one rule serve both without either process learning
    about the other's storage. The two sources are reconciled by test.
    """
    if surface != "regular":
        return 0
    return max(open_notes, 0)


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
    the splice that motivates it. The incident that bought it: while the confirm
    note was a proposal against ``## Approval`` — last in the seed template — the
    base captured at project creation ended in one newline and the extent ended
    in two the moment the coordinator appended anything after it, so the note
    that unblocks a project reported itself stale on every project ever created.
    That note carries no base now, but the rule it forced is general and every
    last-section proposal still depends on it.

    The guard below stays on the **raw** ``base_section``. ``""`` still means
    *nothing was captured*; ``extent_key("")`` is ``""`` either way, so
    normalizing first would not change the answer but would make a reader check.
    """
    if not note.base_section:
        return False
    current = _current_section(body, note.section)
    return current is not None and extent_key(current) != extent_key(note.base_section)


def section_changed_since_answer(plan: ProjectPlan, body: str, note: PlanNote) -> bool:
    """:func:`section_changed` as a READER of the note should see it: measured
    from the newest reply by the person the note was addressed to, when there
    is one, rather than from the moment the note was filed.

    **The incident.** The user replied *"skip BTG for now"* on a section; the
    coordinator rewrote the section to do exactly that 32 seconds later; the
    user's own note then read *"This section changed since @user wrote this"* —
    a warning about the fix it asked for. The edit was the answer, and the
    compare had no way to know it.

    **The reply chain already carries what this needs.** Every reply captures
    its own base at file time (:func:`_capture_base` via :func:`add_note`), so
    an addressee reply filed AFTER the edit holds the section as the addressee
    left it. Comparing against that says *"changed since this was dealt
    with"*, which is the question the banner is really asking: an edit by
    anyone after the answer still fires it. Nothing stored changes — the note
    keeps the base its author saw — and the answer is derived on every read.

    **Comments only.** A proposal's ``base_section`` is not a reading
    position, it is the base :func:`resolve_note` rebases against, and the
    editor's collision flow keys off this same flag; quieting it there would
    make ``Accept`` look clean and then 409. So a note carrying a proposal (or
    a delete flag) gets :func:`section_changed` unchanged.

    **Order matters, and the safe way.** A reply filed BEFORE the edit captured
    the old text, so the banner still fires — which is why the ``plan`` skill
    says *edit, then reply*.
    """
    if note.proposal or note.proposes_delete:
        return section_changed(body, note)
    answers = [
        n for n in plan.notes
        if n.reply_to == note.id
        and n.base_section
        and n.section == note.section
        and n.by != note.by
        and (not note.to or n.by == note.to)
    ]
    if not answers:
        return section_changed(body, note)
    latest = max(answers, key=lambda n: n.at)
    return section_changed(
        body, PlanNote(id=note.id, section=note.section, base_section=latest.base_section)
    )


def section_new(body: str, note: PlanNote) -> bool:
    """Does this note PROPOSE the section it names? Then the slug not resolving
    is the point of the note rather than a problem with it.

    **THE MODEL HAS ALWAYS KNOWN THIS AND THE LABEL NEVER ASKED.** The rule is
    :func:`resolve_note`'s apply arm verbatim — ``current is None and not
    note.base_section`` — and :func:`_row_creates` carries the same one for the
    tray. Both of them go on to CREATE the section and say so at length. The
    field a reader actually sees, :func:`section_missing`, asked only whether
    the slug resolved, so every one of those notes was labelled *"the section
    `x` is no longer in the document"* — reporting a deletion that never
    happened, on a note whose whole purpose is to write that section. That is
    not a rare shape: it is what the spec lock files whenever a coordinator's
    refused write ADDS a section, and there were sixteen such notes on one
    server when this was written.

    ``PlanNote.base_section`` names the two readers of the create-vs-vanished
    rule. This is the third, and it is the one that was missing: the predicate
    now has one home and three callers rather than two callers and a field that
    guessed.

    **``proposal`` is part of the test, and** :func:`_row_creates` **not asking
    it is not a disagreement.** That function is handed a row that has already
    chosen a note to accept, so a note with nothing to apply never reaches it;
    this one is asked of every note on the plan, including the plain comment
    that names a section somebody later deleted — which is genuinely dangling
    and must keep saying so. A refused DELETE is excluded by the same clause and
    correctly: ``proposes_delete`` against an absent section removes nothing, so
    it creates nothing either.
    """
    if not note.section or not note.proposal or note.base_section:
        return False
    return _current_section(body, note.section) is None


def section_missing(body: str, note: PlanNote) -> bool:
    """The advisory slug no longer resolves, so the proposal cannot be applied
    and degrades to a comment with a suggestion attached (§3.3).

    **A note that PROPOSES the section is not missing it** —
    :func:`section_new` is that case and answers it, and this must not shadow
    it for the same reason :func:`section_changed` must not shadow this one:
    *"there is nothing there any more"* and *"there is nothing there yet"* send
    a reader to two different places, and only one of them is somewhere to go.

    The narrowing runs everywhere the flag is read, which is the point of doing
    it here rather than at each surface. ``plan list-notes --dangling`` stops
    returning notes that are creates, ``plan show-note``'s dangling warning
    stops firing on them, and ``PlanNoteRow`` prints the sentence that is true.
    """
    if not note.section or section_new(body, note):
        return False
    return _current_section(body, note.section) is None


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

    __slots__ = ("result", "spec", "notes", "moved", "body", "why")

    def __init__(self, result, spec=None, notes=(), moved=(), body="", why=""):
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
        #: The writer's one-line reason, carried onto the proposals a refused
        #: spec write files (:func:`_file_spec_lock_locked`). Length-checked in
        #: :func:`_prepare_locked`, above the append.
        self.why = why

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


def _plan_was_offered(plan: ProjectPlan) -> bool:
    """Has the coordinator ever put this plan **in front of** the user?

    ``rounds`` is appended by :func:`submit_review` and by nothing else, so a
    non-empty list means at least one review batch reached somebody's plate.
    That is the whole of *"this document has been handed over"*.

    **NOT :func:`_user_has_reviewed`, and the difference is the point.** That
    one asks whether the user has ANSWERED; this one asks whether they were
    ASKED. A plan may be accepted without either — the confirm row is
    dismissible from the desk card and, until the coordinator sends its first
    round, that row reads :data:`GO_NOTE_DRAFTING`, so taking it says *"start on
    the draft as it stands"* rather than *"I have read this"*.

    **Why this fact and not the review stamp**, on a server holding 17 accepted
    plans: 11 of them carry a blank ``first_user_review_at`` — legacy stamps
    predating the field, which
    :meth:`ProjectPlan._backfill_first_user_review` cannot recover because their
    rounds were opened by the keeper. Keying the lock on the review stamp would
    have silently unlocked ten ratified plans mid-execution. Every one of those
    11 has between one and nine recorded rounds, so ``rounds`` separates them
    from the drafting case with nothing to migrate and no new field.

    **Monotone, which is what makes it safe to gate a lock on.** ``_save``
    trims ``rounds`` to ``MAX_ROUNDS`` from the FRONT, so a list that has been
    non-empty never returns to empty and this predicate never goes back to
    False. A lock that could disengage on a busy project is the failure
    :func:`_stamp_user_review_locked` latches against, reached from the other
    side.

    Named rather than inlined for the reason :func:`_user_has_reviewed` is:
    three callers — the lock, ``changed_since_acceptance``, and
    :func:`resolve_note`'s review stamp — and a predicate with three copies has
    three answers.
    """
    return bool(plan.first_offered_at)


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
    * **the start line** — ``accepted_at`` (with :func:`_plan_was_offered`)
      OR :func:`_user_has_reviewed`, whichever comes first. See below.

    **THE START LINE IS THE USER'S FIRST REVIEW, AND IT MOVED HERE ON PURPOSE —
    FOR THE SECOND TIME.** It read ``accepted_at or _user_has_reviewed(plan)``
    once before (``39a88eac``), answering ``chuswine-geo-b2b``: after the user's
    first review the coordinator made 26 direct writes and 0 proposals. It was
    then moved back to acceptance (``a7008e30``) with write *receipts* in the
    gap, on the argument that before a signature there is no baseline for a
    diff to be a delta against.

    That argument was wrong about the baseline. Once the user has reviewed the
    draft there IS one — the document they read and commented on — and the
    receipts design failed in its own way on ``agent-home-folder-view``: the
    coordinator folded the user's three answers straight into the text, closed
    only its own questions, and left the user's three notes open on sections
    that had moved under them, each now warning *"this section changed since
    you wrote this"* about the very edit it asked for. The owner's stated
    preference settles it: after their first review they read diffs and
    comments, never a document that moved unnoticed. So a keeper's spec move
    after that point is refused and filed as a proposal, exactly as it is after
    acceptance.

    **The never-shown case stays unlocked**, and that is what the
    ``_plan_was_offered`` conjunct on the acceptance arm is for. A user who
    dismissed the confirm row while it still read :data:`GO_NOTE_DRAFTING`
    released the work and reviewed nothing — on ``onboard-angel-investor``
    locking there produced 17 proposals against a draft nobody had opened.
    :func:`_user_has_reviewed` cannot be true in that state: it needs a review
    round the OWNER closed.

    The coherence risk a7008e30 named — rejecting one hunk of five leaves a
    plan nobody designed — is handled where it arises: a proposal is a whole
    section, never a hunk, and the keeper files related proposals as one batch
    naming each other (``system_skills/plan/SKILL.md``).

    **ACCEPTANCE ALONE STOPPED MEANING RATIFIED, WHICH IS WHY THE ACCEPTANCE ARM
    CARRIES :func:`_plan_was_offered`.** A lock on an accepted plan rests on
    *"there IS a baseline the user saw"*, and that implication held while ``accepted_at`` could only be
    stamped one way: the user dismissing a row that said *"Nothing in this plan
    is waiting on an answer from you. This is the start gate"*. The confirm row now has a second
    wording (:data:`GO_NOTE_DRAFTING`, seeded before the coordinator has taken a
    turn), and taking THAT one says *"start on the draft as it stands"*. It
    releases the gate, it is the owner's call to make, and it ratifies nothing —
    ``rounds`` is empty, ``first_user_review_at`` is blank and
    ``owner_last_seen_revision`` is zero, because the document was never put in
    front of anybody.

    Refusing there costs exactly what the paragraph above says refusing costs
    when no baseline exists, and it was measured: on ``onboard-angel-investor``
    the user dismissed the drafting row 2m12s in, the coordinator's next two
    turns of ordinary drafting were refused section by section, and what reached
    the user was **17 open proposals** — hunk-by-hunk adjudication of a document
    they had never opened, against a baseline they had never agreed to. The
    coordinator then abandoned the draft and told them it "had a stale copy
    locally".

    So the acceptance arm is split in two: ``accepted_at``, and
    :func:`_plan_was_offered` — *"of a document they were shown"*. A legacy plan
    has rounds and locks exactly as it did. The review arm needs no such guard:
    a review the owner closed is itself proof they were shown the document.

    **``phase == "executing"`` is still gone**, and its going is not a narrowing
    of AC-3.1: ``accepted_at`` implies ``executing``, so the conjunct was
    redundant on the only inputs that reach it.

    :attr:`ProjectPlan.first_user_review_at` is latched and backfilled from
    ``rounds`` on load, so every legacy plan answers the review arm correctly,
    and the coordinator's prompt reads the same projection it always did.

    **What is still legal for the keeper after the lock engages**, because the
    comparison is on ``spec_digest`` and that digest normalizes them away
    (``plan_markdown.spec_digest``): ticking and unticking checkboxes, editing
    ``<!-- … -->`` provenance comments, and reflowing whitespace. Progress
    bookkeeping is untouched, which is what makes moving the start line safe to
    ship without also redesigning how milestones report completion.

    **AND THE WHOLE DETAIL LAYER, WHICH IS THE OTHER HALF OF THE SAME
    SENTENCE.** ``spec_digest`` no longer hashes the text of a section marked
    ``<!-- layer: detail -->``, so re-cutting, splitting, merging, reordering
    and re-assigning milestones all leave the digest where they found it and
    land silently. **This predicate did not change to make that true** — the
    material the digest reads did. What this function decides is still only
    *whose document is it*, never *which part of it*.

    Two acts inside a detail section are still refused, and both are refused by
    the arm above rather than by anything here: changing a section's ``layer:``
    marker (the manifest is hashed, so the flip moves the digest) and dropping
    an ``<!-- advances: … -->`` claim so that a criterion is left unclaimed
    (:func:`_coverage_regressed`).
    """
    return (
        by == keeper(project)
        and project.surface == "regular"
        and (
            (bool(plan.accepted_at) and _plan_was_offered(plan))
            or _user_has_reviewed(plan)
        )
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
    project: "Project",
    *,
    by: str,
    note_ids: Sequence[str],
    relayered: Sequence[tuple[str, str, str]] = (),
    moved: Sequence[str] = (),
    added: Sequence[str] = (),
    dropped: int = 0,
) -> str:
    """M3's own refusal, with its remedy in it (AC-3.3).

    Says what still **works** as well as what does not, because a model told
    only *"refused"* retries with a smaller edit of the same kind — and a
    smaller spec edit is still a spec edit, while a checkbox tick of any size
    lands. Names the note ids, because a ``403`` body is a bare string and the
    user's copy of the coordinator's text is the whole point of the refusal.

    **The ``accepted`` flag is gone, and its going is the point.** It existed
    while the lock had two start lines and one of them was *"the user has
    reviewed this"*, on which *"it is accepted"* would have been a stated reason
    the model can see is untrue — and a ``403`` a model can disprove is one it
    argues with rather than obeys. :func:`_spec_is_locked` engages at the
    user's first review OR at acceptance, and the user having reviewed the plan
    is true in both, so the notes name the review (:func:`_spec_lock_comment`)
    and this refusal names neither.

    **``relayered`` LEADS when there is one, because it is the one cause this
    refusal could not otherwise name.** Every other spec move is visible in the
    notes: the rows carry the section and the diff, and a coordinator reading
    them knows what it changed. A layer flip is not — the row that moved is
    filtered out (:func:`_relayered`), so the model is handed note ids for
    sections it did not think it was touching and a sentence about *"what this
    plan says"* it cannot reconcile with a rewrite it believes was detail-layer
    and free.

    What it does next is bisect. On ``onboard-angel-investor`` that cost two
    turns, 41 shell calls, $3.15 and 17 notes the user never asked for, and
    ended with the coordinator telling them it *"had a stale copy locally"* — a
    wrong diagnosis it had no evidence against. The remedy is one marker, so
    the sentence names it.
    """
    ids = ", ".join(note_ids) or "a note"
    cause = ""
    if relayered:
        # A parent's flip drags every child it contains, so the list can run
        # long on a big section. The first rows are the ones carrying the
        # remedy — `_relayered` reports in document order, so the parent leads —
        # and a sentence naming twelve slugs is one a model skims.
        shown, rest = relayered[:3], len(relayered) - 3
        moved = "; ".join(
            f"`{slug}` from `{before}` to `{after}`" for slug, before, after in shown
        )
        if rest > 0:
            moved += f" (and {rest} nested under it)"
        cause = (
            f" THE CAUSE IS A LAYER CHANGE, not the prose: {moved}. The "
            f"`<!-- layer: ... -->` marker lives in the heading and is hashed, "
            f"so a rewrite that drops it moves the spec however little else "
            f"changed. Put the marker back on that heading and re-run the same "
            f"write — it will land, and these notes become unnecessary."
        )
    # **The ordinary refusal names what moved, for `relayered`'s own reason.**
    # `cause` above leads when a layer flipped because that is the one cause the
    # rows cannot show. This is the other silence: the rows exist, but they reach
    # the model as note IDS, and an id says nothing about what is in it. A
    # coordinator told only *"filed as n-85511d, n-377b28"* has to open the notes
    # to learn anything, and the sentence it was told instead — that ticking a
    # checkbox still applies — is an invitation to conclude the boxes were the
    # whole of it. One did, in a report its owner read.
    #
    # Subordinate to `relayered`: when a layer moved, the marker is the remedy
    # and the moved prose is downstream of it, so naming both would offer two
    # fixes for one cause.
    if not relayered and (moved or added):
        parts = []
        if moved:
            shown = ", ".join(f"`{s}`" for s in list(moved)[:4])
            rest = len(moved) - 4
            parts.append(
                f"{shown}{f' (and {rest} more)' if rest > 0 else ''} — the text "
                f"you sent for {'these' if len(moved) > 1 else 'this'} is not "
                f"what the plan says now"
            )
        if added:
            shown = ", ".join(f"`{s}`" for s in list(added)[:4])
            rest = len(added) - 4
            parts.append(
                f"{shown}{f' (and {rest} more)' if rest > 0 else ''} — "
                f"{'these are' if len(added) > 1 else 'this is'} not "
                f"{'sections' if len(added) > 1 else 'a section'} this plan "
                f"has, so the write would ADD "
                f"{'them' if len(added) > 1 else 'it'}; a slug comes from "
                f"heading text, so a heading retitled since you last read it "
                f"re-slugged and your old slug now names nothing"
            )
        cause = (
            f" WHAT MOVED: {'; '.join(parts)}. Re-read with `clawmeets plan "
            f"show <project> --section <slug>` and send that text back, "
            f"changed only where you mean to change it."
        )
    # **The dropped count, because the write is ALL-OR-NOTHING.** The filtered
    # rows are correctly not the user's business — nobody accepts a checkbox
    # tick — but their sections did not land either, and a coordinator counting
    # note ids to find out what happened undercounts by exactly this number.
    if dropped:
        cause += (
            f" AND {dropped} further section(s) in the same write were refused "
            f"with it and filed nothing, because their only change was one the "
            f"user does not decide (a checkbox, an HTML comment, a `layer: "
            f"detail` section). This write was all-or-nothing: NONE of it "
            f"landed. Re-send those separately — and tick boxes with `clawmeets "
            f"plan tick <project> <label>`, which is never refused."
        )
    return (
        f"@{by} may not change what this plan says — the user has reviewed "
        f"it, and decides what it says. Ticking a checkbox or editing an HTML "
        f"comment still applies. Your text is filed as {ids} for the user to "
        f"accept in one click; say why in `user-communication`.{cause}"
    )


def _relayered(before: str, after: str) -> list[tuple[str, str, str]]:
    """Sections whose resolved layer MOVED — ``[(slug, before, after)]``.

    The one refusal cause that is invisible in the rows it produces.
    ``_layer_manifest`` is hashed into ``spec_digest``, so flipping a heading's
    ``<!-- layer: … -->`` marker is a spec move however little prose changed;
    but :func:`_rows_worth_showing` decides what the user sees by looking the
    row's slug up in the **stored** body, where the section still carries its
    old layer. A detail section that just became spec is therefore dropped from
    the notes, and the surviving rows are whatever its children looked like
    after the rewrite — frequently *"could not add the section …"* naming slugs
    the document has never had.

    Reported rather than fixed, and the split is deliberate: the filter's own
    argument is still right (reading layers out of the incoming text would let a
    keeper suppress its own note by marking the section ``detail`` in the very
    write being filtered), and the AC-3.3 invariant that a refusal always files
    *something* is untouched. What changes is only that the refusal now says the
    word ``layer``.

    Restricted to slugs present on BOTH sides: a section that appeared or
    vanished has no layer to have moved, and its row already names it.
    """
    a, b = section_layers(before), section_layers(after)
    return [
        (slug, a[slug], b[slug]) for slug in a if slug in b and a[slug] != b[slug]
    ]


def _coverage_regressed(before: str, after: str) -> bool:
    """Would this write leave a criterion unclaimed that WAS claimed before?

    **The one thing that makes "milestones are implementation detail" safe
    rather than merely convenient.** If the keeper may re-cut milestones without
    asking, it may also delete the milestone that carried the work behind
    ``AC-2.1`` — and the user finds out at completion, which is the moment this
    whole design exists to stop being the moment things surface.

    So re-cutting, splitting, merging, reordering and re-owning milestones stay
    free, and exactly one milestone edit is not: one that drops coverage. That
    is a spec change, because it changes what the user said yes to.

    **A REGRESSION test, not a validity test, and the distinction is the entire
    migration story.** It never asks *"is every criterion claimed?"* — a plan
    that carries no ``<!-- advances: … -->`` markers has the identical unclaimed
    set before and after every write, so this cannot fire on any document that
    exists today. Nothing has to be backfilled and no plan has to adopt the
    convention to keep working.
    """
    return bool(unclaimed_criteria(after) - unclaimed_criteria(before))


def _tick_edits(body: str, ticks: Sequence[PlanTick]) -> list[SectionEdit]:
    """Resolve each :class:`PlanTick` against ``body`` into an ordinary
    :class:`SectionEdit`, so that below this line there is one kind of write.

    **The materialization is the feature.** ``body`` is the document as it reads
    under ``_lock``, so the ``base`` this hands ``_splice`` is by definition the
    section's current text and the staleness check cannot fire; and the only
    difference between ``base`` and ``text`` is marker state, which
    :func:`normalize_spec_text` folds — so ``spec_digest`` cannot move and the
    spec lock cannot fire either. Nothing downstream needs to know a tick
    happened, which is why this returns edits rather than growing a second path.

    **One edit per SECTION, not per tick.** Two boxes in one ``### G3`` are two
    ticks and one splice: ``_splice`` checks every edit against the body as it
    stands *before* applying any, so two edits naming the same slug would both
    carry the pre-flip base and the second would overwrite the first's flip.
    Flipping both into one section text is the only shape that composes.

    A label that resolves to nothing, or to more than one box, is a ``400`` that
    NAMES WHAT IT SAW. The alternatives are both worse than refusing: falling
    through to a create is how a retitled milestone's stale slug got appended to
    the end of a plan as a phantom duplicate, and picking the first of two
    matches would tick whichever box happens to come first in the file.
    """
    if not ticks:
        return []

    flipped = body
    #: The slugs to splice, in the order their first tick named them — a set
    #: would make the edit list's order depend on the hash of a slug.
    touched: dict[str, None] = {}
    for tick in ticks:
        found = boxes_labelled(body, tick.label)
        if not found:
            labels = sorted({b.label for b in find_boxes(body) if b.label})
            raise PlanInputError(
                f"no checkbox labelled {tick.label!r} in this plan. A tick "
                f"addresses a box by the label that OPENS its line — "
                f"`- [ ] **M2** — …` is `M2`. This plan has: "
                f"{', '.join(labels) or '(no labelled checkboxes)'}"
            )
        if len(found) > 1:
            where = ", ".join(f"`{b.section}`" for b in found)
            raise PlanInputError(
                f"{len(found)} checkboxes are labelled {tick.label!r}, in "
                f"{where}. Give the label one home — a duplicate label is an "
                f"ambiguous address, and ticking whichever came first in the "
                f"file would be a guess"
            )
        box = found[0]
        flipped = flip_boxes(flipped, [box], tick.checked)
        touched[box.section] = None

    out: list[SectionEdit] = []
    for slug in touched:
        current = _current_section(body, slug)
        after = _current_section(flipped, slug)
        if current is None or after is None:
            # **Raised, never skipped.** A box whose enclosing section cannot be
            # addressed has no splice to ride, and dropping it here would be the
            # very failure this whole path exists to remove: a tick the caller
            # was told nothing about. `_enclosing_section` gives preamble boxes
            # the `_lede` slug, so there is no known document shape that reaches
            # this — which is the reason to make it loud rather than tolerated.
            raise PlanInputError(
                f"the checkbox is in a part of the plan that cannot be "
                f"addressed as a section (slug {slug!r}). Give it a heading and "
                f"tick it again"
            )
        out.append(SectionEdit(section=slug, text=after, base=current))
    return out


def _prepare_locked(
    project: "Project",
    ctx: "ModelContext",
    plan: ProjectPlan,
    edits: Sequence[SectionEdit],
    *,
    by: str,
    why: str = "",
    ticks: Sequence[PlanTick] = (),
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

    ``why`` is the keeper's one-line reason. It is never required; on a write
    the spec lock refuses it leads each proposal filed for the user. Capped
    here, at input validation, so a note built from it cannot breach
    ``MAX_NOTE_CHARS`` below.
    """
    if not may_write(project, by):
        raise PlanForbiddenError(_write_refusal(project, by=by))
    if len(why) > MAX_WHY_CHARS:
        raise PlanInputError(
            f"--why is {len(why)} chars, limit is {MAX_WHY_CHARS}. It is one "
            f"line, not the rationale — the rationale goes in "
            f"`user-communication`, where the user can answer it."
        )

    for edit in edits:
        if len(edit.text) > MAX_EDIT_CHARS:
            raise PlanLimitError(
                f"section edit is {len(edit.text)} chars, limit is {MAX_EDIT_CHARS}"
            )

    body = _read_body(project, ctx)
    edits = list(edits) + _tick_edits(body, ticks)
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

    # **AND NO WRITE PUTS TWO HEADINGS UNDER ONE SLUG** — asked of every
    # section, at every phase.
    #
    # **IT SURVIVED THE REMOVAL OF ``## Approval`` AND IS NOT A LEFTOVER OF IT.**
    # The go-note used to be a proposal against that section, which gave this
    # module two further arms here: keep the heading alive while the note is
    # open, and refuse a second ``## Approval`` smuggled inside another
    # section's replacement. Both were about ONE section's role as a diff
    # target, and the confirm note carries no diff now, so both are gone. This
    # one is about the id space every section shares — the approval incident
    # below was its OCCASION, not its subject — and deleting it because its
    # famous example retired would leave `## Goal` shadowable by exactly the
    # copy-paste that started this.
    #
    # The two guards on each edit ask about its FIRST line: does it carry a
    # heading (`drops_heading`), and is that heading at the section's own level
    # (`relevels_heading`). Nothing looks at the rest of the text, and a
    # replacement may legitimately carry further headings — re-cutting `### M1`
    # into M1 + M2 is how a milestone list grows, and is pinned by
    # `test_the_keeper_rewrites_a_detail_section_after_the_users_round_and_files_nothing`.
    # So the question is not whether a replacement adds a heading. It is
    # whether the heading it adds is one the plan ALREADY HAS.
    #
    # The incident: a coordinator regenerated `## Milestones` into a file,
    # copied one section too far, and the file ended with the document's own
    # `## Approval` block. Ids are assigned in source order, so the injected
    # copy took `approval` and the user's real, accepted section was demoted to
    # `approval-2` — then reported to them as a brand-new section to approve, by
    # a spec lock doing exactly its job. The user read a note saying their
    # keeper "could not change `approval-2`" about a slug nobody had ever typed.
    #
    # **This is not a new rule; it is an existing one reaching its second
    # door.** `add_note` already refuses to create the `duplicate id` state
    # (`_refuse_unknown_section`, and
    # `test_the_typo_that_used_to_append_a_second_heading_under_one_slug` calls
    # it *"the only one of these with real damage"*). A surface refusing what
    # its sibling accepts is the shape of a bug, and this was the open half.
    #
    # **Asked of the SPLICED document, and of the transition rather than the
    # state.** Of the document, because the duplicate is only visible once the
    # text is in place — no single edit names the section it collides with. Of
    # the transition, because a plan that already holds two `### Notes` must
    # stay writable; refusing on the state would leave a document that exists
    # with no way to edit it and no way back.
    #
    # Refused, not silently repaired, for the reason the two guards above give:
    # a `400` can name the heading it saw, and a rename chosen by the server
    # would make the stored document disagree with the text its author sent.
    dup = duplicated_heading(body, spliced)
    if dup:
        raise PlanInputError(
            f"this write gives the plan a second {dup!r} heading. Two headings "
            f"slug to one id, so the copy that comes FIRST in the document takes "
            f"it and the other is renamed — every note, every quote and every "
            f"later edit addressed to that section then lands on whichever came "
            f"first. If you meant to rewrite that section, edit it by its own "
            f"slug; if you copied more text than you meant to, send only the "
            f"section you are writing."
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

    if _spec_is_locked(project, plan, by=by) and (
        spec_digest(cleaned) != spec_digest(body)
        or _coverage_regressed(body, cleaned)
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
        # **THE SECOND ARM IS NOT A SECOND LOCK.** ``spec_digest`` no longer
        # hashes the detail layer, which is what buys the keeper its freedom —
        # and that freedom has exactly one way to be abused, which
        # :func:`_coverage_regressed` closes: a milestone edit that drops the
        # work behind a criterion. Same refusal, same rows, same filed note,
        # because it is the same kind of act — it changes what the user said yes
        # to. Both arms ask their question of ``cleaned`` against ``body``, both
        # sides computed here in one call, so neither needs a migration.
        #
        # **Appends nothing and files nothing.** This function's contract is that
        # ``plan`` is not mutated, so a caller that abandons the write leaves the
        # sidecar as it found it; the deviation note AC-3.3 owes the user is
        # filed by the caller, which is the one holding the transaction that will
        # save it. ``changes`` carries both texts at once — the only place they
        # are both in hand — which is exactly what ``StaleSection`` exists for.
        rows = [
            StaleSection(section=slug, base=before, current=before, text=after)
            for slug, before, after in _pair_renamed_rows(body, cleaned, changes)
        ]
        # **Computed here, off the same rows and the same stored body the filer
        # uses.** `_rows_worth_showing` is what decides which rows become notes,
        # so asking it here is how the refusal's own count of what it dropped
        # cannot disagree with the number of notes the user actually receives.
        # Recomputing it in the filer would be a second answer to one question.
        shown = _rows_worth_showing(rows, body)
        present = section_layers(body)
        return _PreparedWrite(
            WriteResult(
                ok=False,
                revision=plan.revision,
                sha=body_sha(body),
                locked=rows,
                relayered=_relayered(body, cleaned),
                moved=[s.section for s in shown if s.section in present],
                added=[s.section for s in shown if s.section not in present],
                dropped=len(rows) - len(shown),
            ),
            body=body,
            why=why,
        )

    return _PreparedWrite(
        WriteResult(ok=True, revision=plan.revision, sections=moved, sha=body_sha(cleaned)),
        spec=_file_spec(project, ctx, cleaned, by, created=False),
        notes=[
            PlanNote(id="", section=s.section, quote=s.quote, to=s.owner, by=by, comment=s.text)
            for s in shorthands
        ],
        moved=moved,
        body=cleaned,
        why=why,
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
    why: str = "",
    ticks: Sequence[PlanTick] = (),
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
    prepared = _prepare_locked(project, ctx, plan, edits, by=by, why=why, ticks=ticks)
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
            project, plan, prepared.result.locked, by=by, body=prepared.body,
            why=prepared.why,
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


def _supersede_prior_locked(plan: ProjectPlan, note: PlanNote) -> None:
    """**One author gets one unsent proposal per section.** Close the earlier
    ones as ``dismissed``, naming the note that replaced them. Lock held; the
    incoming ``note`` must already carry its id and must NOT yet be in
    ``plan.notes``.

    **Why it is here and not in** :func:`add_note`. ``add_note``'s
    :func:`_recent_twin` net catches the *identical* re-file and returns the
    existing id. This is its sibling for the *changed* re-file, and it has to
    sit one level down because the door that produced the incident does not go
    through ``add_note`` at all: :func:`_file_spec_lock_locked` files the
    server's own deviation note directly onto :func:`_add_notes_locked` when a
    write is refused, and so do :func:`_file_conflict_locked` and
    :func:`absorb_plan_upload`. On ``clawmeets-plan-fold-status`` that left the
    user four rows on ``## Milestones`` where one was meant, two of them
    server-written delete-rows the coordinator never chose to file, and the
    coordinator hand-wrote *"apply this and dismiss the other three"* because
    it had no way to reconcile them itself.

    **It is not a hole in** :func:`_refuse_unsent_self_close`, and the
    ``note_was_sent`` clause is the whole reason. That guard exists because a
    keeper that files a question and then closes it before the batch goes out
    has decided it on the user's behalf — the ``chuswine-geo-b2b`` sequence its
    docstring records. Nothing is decided here: the close is atomic with filing
    a REPLACEMENT on the same section addressed to the same person, so what
    reaches the desk is the same question in its current wording, never
    silence. A bare ``resolve --dismiss`` on your own unsent note stays a 403,
    because that one really does end with nothing on the desk.

    **Proposals only.** Comments and questions stack legitimately on one
    section — §4.4's *"a note with no proposal never collides"* — and two of
    them are two things to say, not one restated. Two unsent PROPOSALS on one
    section are already unrepresentable: a proposal is the section's whole
    replacement text, the document can hold one, and the skill's own remedy for
    the collision was to file a third note superseding both by hand. This does
    that in the one place every filing door passes through, so the coordinator
    never has to notice.

    ``bool(proposal) or proposes_delete`` is the desk's ``has_proposal``
    (``routes/project_plans.py``) minus its ``status == "open"`` clause on the
    delete arm, which is redundant here — the loop only looks at open notes.

    The confirm note is exempt (``bootstrap``): it closes on ``dismiss`` and on
    nothing else (:func:`_refuse_go_note_close`), it carries no proposal so the
    guard above has already returned, and at most one is ever open at a time, so
    nothing can legitimately replace it.
    """
    if not (note.proposal or note.proposes_delete):
        return
    if not note.section or note.bootstrap:
        return
    for prior in plan.notes:
        if prior.id == note.id or prior.status != "open" or prior.bootstrap:
            continue
        if prior.by != note.by or prior.to != note.to:
            continue
        if prior.section != note.section:
            continue
        if not (prior.proposal or prior.proposes_delete):
            continue
        if note_was_sent(plan, prior):
            # The user has seen it. From here only they may close it, which is
            # exactly what `_refuse_unsent_self_close` leaves open and what
            # this must not take away: two rows on the desk is a worse outcome
            # than one, but a row vanishing from under the reader is worse than
            # both.
            continue
        _close_note(
            prior,
            status="dismissed",
            by=note.by,
            reason=REPLACED_UNSENT_REASON.format(id=note.id),
        )


def _add_notes_locked(plan: ProjectPlan, notes: Sequence[PlanNote]) -> list[str]:
    """Validate the whole sequence, **then** stamp ids and timestamps and file.

    All-or-nothing: an over-cap sequence leaves ``plan.notes`` untouched rather
    than half-filed. Filing one at a time and raising mid-loop is the mechanism
    by which *"the first shorthand was filed and then lost"* happens on a caller
    that then abandons the plan; validating first makes that partial state
    unreachable rather than merely harmless. Lock held.

    **THE QUOTE IS DERIVED HERE, not in each caller, and that placement is the
    whole repair.** :func:`add_note` — the public door — has derived one since
    the anchor work landed, but it is not the door most notes come through:
    :func:`_file_spec_lock_locked`, :func:`_file_conflict_locked` and
    :func:`absorb_plan_upload` all build a :class:`PlanNote` by hand and hand it
    straight to this function, and every one of them omitted ``quote``. So the
    notes the SERVER files about a refused write — the ones a user is most
    likely to be reading, because they arrive unasked — were exactly the notes
    that could never render beside the text they are refusing. On
    ``clawmeets-todo-tickets`` that was both open notes on the plan.

    Fixing it at the call sites would have been four places to remember and a
    fifth to forget. Here it is structural: this is the one function every
    filing door already passes through, which is the argument
    :func:`_supersede_prior_locked` makes at length for living at this level.

    ``add_note``'s own derivation stays where it is and is not made redundant by
    this one: :func:`_recent_twin` keys on ``quote``, so the quote must exist
    BEFORE the dedupe scan, which is upstream of here. A note that arrives
    carrying a quote is untouched by the guard below — including every note that
    came through that door.

    **``base_section`` is the whole input**, so no document read is needed and
    none is taken; this function has never held a body and does not start now. A
    caller that captured no base derives no quote and its note settles at the
    end of its section, which is the middle rung of :func:`_derive_quote`'s
    ladder and the correct fall.
    """
    _validate_notes_locked(plan, notes)
    ids: list[str] = []
    for note in notes:
        if note.section and not note.quote:
            note.quote = _derive_quote(note.base_section, note.proposal)
        note.id = _gen_id("n", (n.id for n in plan.notes))
        note.at = note.at or _now()
        # BEFORE THE APPEND, and that ordering is load-bearing twice over. The
        # id is stamped, so the reason can name the note that replaced them;
        # `plan.notes` still holds only PRIOR notes, so the incoming one cannot
        # supersede itself. A batch carrying two rows on one section collapses
        # the same way — the first is appended by the time the second is
        # processed — which is the answer a per-note rule has to give.
        #
        # The open-note budget above is counted before any of this, so a
        # sequence that would fit only AFTER superseding is still refused. That
        # is the conservative direction and it costs nothing real: superseding
        # is what stops repeated re-files from accumulating toward the cap in
        # the first place.
        _supersede_prior_locked(plan, note)
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


def _reply_closes_parent(
    project: "Project", plan: ProjectPlan, parent: PlanNote, *, by: str
) -> bool:
    """Does a reply to ``parent``, filed by ``by``, close it ``answered``?

    **The rule AC-5.9 states, with one home instead of two.** It lived inside
    :func:`submit_review`'s ``ask`` arm, which is the OWNER's door — the tray is
    owner-only — so the half of AC-5.9 that runs the other way never ran at all.
    :func:`open_notes_for_you` has documented that half since it was written
    (*"the coordinator answers the user's note … closing it is −0"*), and
    :func:`add_note`, the door the coordinator actually uses, left the parent
    open on the stated grounds that *"the plan editor draws only OPEN notes"*.

    The consequence was one-sided and the user saw it: every note the user wrote
    was answered and stayed open, so it kept a ``Dismiss`` button and kept
    asking the person who had already been answered to close their own question.
    The cost the old comment was avoiding — the question leaving the screen when
    it closes — is paid where it belongs, on the reply: ``PlanNoteRow``'s
    *"Answering your question"* line now carries the parent's own first line, so
    the answer still reads as an answer with the question gone.

    ``answered`` is the ladder's own word for *the addressee has replied*, so
    every clause below is that sentence made checkable:

    * **an already-closed parent stays as it closed.** A reply to a note someone
      resolved underneath you must not overwrite ``applied`` with ``answered``:
      the second is a weaker fact and it would erase which decision was made.
    * **never the go-note.** :func:`_refuse_go_note_close` carries the argument
      at length — every way of closing it releases the execution gate and only
      ``dismiss`` means yes, so typing *"what about the auth milestone?"* must not
      start the work being questioned. The outgoing reply is filed either way;
      only the parent's status differs.
    * **only someone the note is addressed to**, which is :func:`resolve_note`'s
      own report guard verbatim. Under the two-party rule the addressee is the
      user or the keeper, so in practice this admits exactly the two ends of the
      thread and refuses nothing anybody can reach.
    * **not your own unsent proposal to the user**, which is
      :func:`_refuse_unsent_self_close` — the same authority leak, arriving
      through the reply door instead of the ``--dismiss`` one.

    It **returns a bool where those two RAISE**, and that is the difference that
    matters: filing a reply is always legal. A shape that may not close the
    parent still files the note and simply leaves the parent open, because the
    caller asked to say something and not to resolve anything.
    """
    if parent.status != "open" or parent.bootstrap:
        return False
    if parent.to and parent.to != by and not may_write(project, by):
        return False
    if by != OWNER and parent.by == by and parent.to == OWNER:
        return note_was_sent(plan, parent)
    return True


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


def _derive_section(body: str, quote: str) -> str:
    """The section a QUOTED note belongs to when its author named none.

    The other direction of :func:`_derive_quote`, and the pair is what makes the
    editor's placement a ladder rather than a cliff — see that function for the
    three rungs.

    **A quote is what makes this legal.** ``--section ''`` on its own is a
    deliberate choice with its own meaning — *"about the document as a
    whole"* — and nothing here may overrule it. But an author who supplied an
    EXCERPT has already said the note is about a particular passage; a passage
    lives in a section, and refusing to look up which one is not respect for
    their choice, it is a shrug. So this fires on ``quote and not section`` and
    on nothing else, which leaves the whole-document note exactly as sayable as
    it was.

    Reachable from the terminal, where ``--quote`` and ``--section`` are
    independent options and nothing pairs them; the browser composer always
    sends both, because ``planAnchor`` takes them off one selection.

    ``""`` when the excerpt is in no section or in two;
    :func:`section_holding_quote` carries why an ambiguous match must not be
    guessed at.
    """
    return section_holding_quote(body, quote) if quote else ""


def _salvage_section(body: str, section: str, quote: str) -> str:
    """A section slug that does not resolve, replaced by the one the note's own
    EXCERPT lives in — or handed back untouched when there is nothing to go on.

    **Nothing in this system checks that a section slug is real**, at either
    door, and a slug is the one address a caller types by hand. ``plan note
    --section m2`` when the document says ``m2-api-and-ingest`` is accepted,
    filed, stamped SPEC — so it blocks execution — and dropped at the bottom of
    the page under a sentence about a section that was never deleted.

    It is also the answer to a race nobody has reported yet and everybody can
    reach: a slug is derived from heading TEXT, so retitling a section re-slugs
    it. Select a block in the browser, have somebody retitle the section
    underneath you, and the note you send names a slug that stopped existing
    between the selection and the Save. The quote still resolves, because the
    passage did not move — only its address did.

    **The quote is what makes the lookup legal**, and it is the same bargain
    :func:`section_holding_quote` is written for: an author who supplied an
    excerpt has already said which passage they mean, so reading the section off
    it is a lookup rather than an override. With no excerpt there is nothing to
    look anything up FROM, and the slug stands as typed for
    :func:`_refuse_unknown_section` to answer.

    Returns ``section`` UNCHANGED on every path that is not a salvage —
    including a resolvable slug, which is the overwhelmingly common one and is
    answered by the first test without a scan. Nothing here ever overrules a
    slug that works.
    """
    if not section or not quote or _current_section(body, section) is not None:
        return section
    return section_holding_quote(body, quote) or section


def _refuse_unknown_section(body: str, section: str, proposal: str) -> None:
    """Refuse a note whose typed section is neither in the document nor being
    written by the note itself. **400, and the only one of these that had teeth.**

    A typo'd slug on a plain comment costs a misplaced note. A typo'd slug with
    a proposal corrupts the document, and does it without a ``--force``: the CLI
    reads the section's text to fill ``base_section``, gets nothing back for a
    slug that does not resolve, and sends ``""`` — which the model reads as
    *"this proposal CREATES a section"* (:func:`_row_creates`,
    :func:`resolve_note`'s apply arm). One ``resolve --apply`` later the document
    holds a SECOND ``## M2 — API and ingest``, one slug over two headings, which
    is the ``duplicate id`` state ``plan show --sections`` warns about and
    nothing repairs.

    **Narrow, because proposing a section that does not exist yet is legal and
    is how half this system works** — the spec lock files a restore of a deleted
    section that way, and so does the upload absorber. The test that separates a
    create from a typo is the proposal's OWN first heading: a replacement whose
    heading slugs to the section the note names means to write that section, and
    one whose heading slugs to anything else (or that has no heading at all)
    named a section it cannot be talking about. Neither filer reaches here in any
    case — both build their notes by hand for :func:`_add_notes_locked` — so this
    guard sits on exactly the door a human or an agent types a slug at.

    Asked only of a slug the caller TYPED, which is :func:`add_note`'s gate
    rather than this function's and is argued there: what the exemption lets
    through is a comment on a section that is gone, which writes nothing.

    **NAMES THE THING, NOT THE FLAG** (AC-1.5), like the ``PlanInputError`` above
    it: this refusal reaches the browser composer too, which has no
    ``--section`` and never will.
    """
    if not section or _current_section(body, section) is not None:
        return
    if proposal and heading_slug(proposal) == section:
        return
    raise PlanInputError(
        f"There is no section {section!r} in this plan. Name one the document "
        f"has, quote the passage you mean, or write a proposal whose own "
        f"heading creates it."
    )


def _criterion_quote(base: str, comment: str) -> str:
    """The line of ``base`` defining the criterion this note's PROSE names.

    **The address was typed — into the message body instead of into ``--ac``.**
    ``plan note --ac AC-2.3`` resolves to exactly the ``(section, quote)`` pair a
    note wants and is documented in the plan skill in three places; agents still
    open with *"AC-1.3: ``--as-user`` is specified as suppressing…"* and pass
    ``--section m1-…`` alone. On one server that was 24 notes — the largest
    single class of sectioned-but-unanchored note there is — and after two
    rounds of documentation changes it is not a documentation problem. The
    author knew which line they meant and wrote it down; this reads what they
    wrote.

    **Scoped to ``base``, which IS the section the note names**, and the scope is
    the guard rather than a detail of it. An id in prose is as often a
    cross-reference — *"this conflicts with AC-2.3"* on a note about a different
    milestone — as it is an anchor, and a cross-reference must not move the
    note. A criterion defined elsewhere is simply not in this text, so it finds
    nothing and the note keeps the anchor it would have had.

    **Exactly one occurrence, or nothing.** ``parse_criteria`` reports
    occurrences rather than definitions, so an id both defined and referred back
    to inside one section yields two rows and no answer. First-in-document-order
    would be the definition and would usually be right; *usually* is the wrong
    standard for an anchor, and the fall is one rung to the section heading
    rather than to the bottom of the page.

    Fence-aware on both sides — ``named_criterion`` over the comment,
    ``parse_criteria`` over the base — so a note quoting a code sample of a
    criterion anchors to nothing, like every other scanner in this system.
    """
    wanted = named_criterion(comment)
    if not wanted:
        return ""
    hits = [c for c in parse_criteria(base) if c.id.upper() == wanted]
    return hits[0].quote if len(hits) == 1 else ""


def _derive_quote(base: str, proposal: str, *, comment: str = "") -> str:
    """The quote a sectioned note gets when its author supplied none.

    **The whole of why CLI-filed notes land at the bottom of the page.**

    THE PLACEMENT LADDER, which this function and :func:`_derive_section` exist
    to climb. Where a note is drawn is decided by what it can name, in three
    rungs, and the fall between them is one rung at a time:

    * **section + quote** — inline in the body, under the block holding that
      line. The answer everybody wants.
    * **section, no quote** — at the END OF THAT SECTION, in ``PlanBody``'s
      loose pane, labelled with why it could not be placed more precisely. The
      note is still next to the text it argues about.
    * **neither** — the page-bottom ``NOTES`` list. Correct, and the only
      correct answer: a note that names no part of the document has no part of
      the document to sit beside.

    Derivation is how a note that LOOKS like the bottom rung is recognised as a
    higher one. It never invents an anchor to climb a rung it has not earned:
    every function here returns ``""`` rather than a guess, which is a fall of
    exactly one rung and never a wrong placement.

    ``--quote`` is optional and no worked example in the ``plan`` skill passes
    one, so *every* note filed from a terminal was unanchored by construction — not by anybody's
    carelessness. The browser composer was always fine, because
    ``planAnchor.tsx`` takes the quote off the user's selection; that asymmetry
    is what made this read like an agent-behaviour problem when it was a
    default.

    Two answers, because a note has two shapes:

    * **A proposal** anchors to :func:`first_changed_line` — the base line it
      actually changes, which is the line its reader wants to be looking at.
    * **Anything else** — a comment, a question, a deviation — anchors to the
      criterion its own prose names (:func:`_criterion_quote`), and failing that
      to the section's own heading. That is the honest answer to *"which line is
      this about"* from an author who named none: the note renders under the
      line it is arguing about if it said which, under the heading of the
      section it names if it did not, and at the bottom of the document in
      neither case.

    The criterion arm is BELOW the proposal arm and not merged with it, because
    where the two disagree the proposal is right: a rewrite of ``AC-2.3`` that
    changes the line under it should anchor to the line it changes, and
    ``first_changed_line`` is the finer answer. Prose is consulted only where
    there is no diff to read.

    ``""`` for the ``_lede`` (no heading to point at) and ``""`` for a section
    that does not resolve — :func:`_capture_base` already returns ``""`` there,
    and both cases have their own answer in :func:`section_missing`, which an
    invented anchor would shadow. Those two are no longer the same outcome as
    each other, which is the ladder earning its keep: the ``_lede`` note names a
    section that resolves, so it settles at the END of the preamble; the note
    whose section is gone names nothing that resolves and goes to the bottom of
    the page, still carrying ``section_missing``'s sentence.

    The result is cut from the stored body, which the server has already
    decoded, so it can never be the lone surrogate :func:`check_note_text`
    exists to refuse.
    """
    if not base:
        return ""
    if proposal:
        return quote_from_line(first_changed_line(base, proposal))
    return _criterion_quote(base, comment) or quote_from_line(heading_line(base))


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
                PlanNote(
                    id="", section=s.section, quote=s.quote,
                    to=s.owner, by=by, comment=s.text,
                )
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
    why: str = "",
    ticks: Sequence[PlanTick] = (),
) -> WriteResult:
    """**The one funnel.** Every byte that reaches PLAN.md comes through here.

    See :func:`_apply_locked` for the seven steps. This wrapper owns the lock and
    the sidecar save, and — when ``file_conflict`` is set, which is what a
    keeper's ``plan update --section`` passes — turns a refusal into notes so it
    is never a dead end (§4.3).

    Does **not** bump ``revision`` on a regular project: that happens once, in
    :func:`submit_review`'s transaction (§2.5).

    ``why`` is the keeper's optional one-line reason. On a write the spec lock
    refuses it leads each proposal filed for the user
    (:func:`_file_spec_lock_locked`); a write that lands ignores it.

    ``ticks`` are :class:`PlanTick` rows — *"tick the box labelled M2"* — which
    :func:`_prepare_locked` materializes into ordinary edits against the body it
    reads under this lock. They carry no section text, so they cannot be stale
    and cannot move the spec digest; a tick therefore never earns a 409, a 403 or
    a ``--why``. They compose with ``edits`` in one transaction, and a label that
    does not resolve to exactly one box is a 400 before anything is written.

    Raises :class:`PlanConflictError` (409) on a stale write and
    :class:`PlanSpecLockedError` (403) on one M3 refused — **both after the
    save**, because both refusals have already filed the notes that are their
    only remedy — and :class:`PlanInputError` (400), before anything is written,
    on a ``why`` over :data:`MAX_WHY_CHARS`.
    """
    async with _lock:
        plan = _load(project, ctx)
        result = (
            await _apply_locked(
                project, ctx, runloop, plan, edits, by=by, why=why, ticks=ticks
            )
        ).result
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
                    relayered=result.relayered,
                    moved=result.moved,
                    added=result.added,
                    dropped=result.dropped,
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
            PlanNote(id="", section=s.section, quote=s.quote, to=s.owner, by=by, comment=s.text)
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

    **``first_user_review_at`` IS A CONJUNCT, BECAUSE A DEPARTURE FROM A
    CONTRACT NEEDS A CONTRACT THE USER ACTUALLY SAW.** Acceptance alone does not
    establish one: the confirm note is dismissible from the desk card, so a user
    may release the gate — legitimately, it is their call — without ever opening
    the document. On ``onboard-angel-investor`` that is exactly what happened,
    and the coordinator's very next write, the fold-in of the roster consult its
    own contract told it to run, came back to the user as a red *"off-contract
    — your call"* on a plan they had never read. The alarm was true about the
    timestamps and false about everything it meant.

    **It narrows the WORD, never the LOCK.** :func:`_spec_is_locked` keys on
    ``accepted_at`` and is untouched, so the keeper still may not write the spec
    and :func:`_file_spec_lock_locked` still files its text with the proposal
    and the base the user accepts in one click. What changes is only that
    :func:`note_kind` falls through to ``"proposal"`` — which is what such a row
    is — and that :func:`_open_deviation` stops reddening a card about a
    contract that was never handed over.

    **The masking this could have caused is closed at the stamp, not here.**
    ``first_user_review_at`` used to move only inside :func:`submit_review`, so
    a user who only ever used the single-note resolve door would never set it
    and would never see a deviation again for the life of the project. Both
    acceptance doors now stamp it when the confirm note they are closing had
    actually been offered (``plan.rounds`` non-empty) — see
    :func:`_stamp_user_review_locked`.
    """
    return (
        bool(plan.first_accepted_at)
        and bool(plan.first_user_review_at)
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
    keep_open: bool = False,
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
    A note whose prose opens on ``AC-2.3`` and passes no ``--ac`` is anchored to
    that criterion's line, provided the criterion is in the section the note
    names (:func:`_criterion_quote`).

    **A SECTION SLUG IS CHECKED, WHICH IT NEVER WAS.** In order: a slug that
    does not resolve is repaired from the note's own quote where the excerpt
    names one section and one only (:func:`_salvage_section`), and a slug the
    CALLER TYPED that still does not resolve is a **400**
    (:func:`_refuse_unknown_section`) unless the note's proposal is the thing
    that creates it. The refusal exists because the same typo carrying an
    ``--edit-file`` used to append a SECOND heading under the same slug on a
    single ``resolve --apply``: the CLI fills ``base_section`` by reading the
    section it cannot find, sends ``""``, and ``""`` is how this model spells
    *"create"*. An INHERITED slug — the one ``reply_to`` fills in above — is
    never refused: answering a note about a section somebody has since deleted
    is a conversation that has to stay possible, and it is only ever a comment,
    because the ``proposal and not section`` refusal at the top of this function
    means a reply carrying a proposal arrives with its slug typed out.

    **A reply CLOSES the parent** ``answered`` (AC-5.9), in the same save that
    files it, under :func:`_reply_closes_parent` — the same predicate
    ``submit_review``'s ``ask`` row asks. ``keep_open`` files the reply and
    leaves the parent open: the replier's own *"not done yet"* — a partial
    answer, or a question back — which the predicate cannot infer.

    **This reverses what stood here, and the reversal is the point.** The old
    rule was that ``submit_review`` closed and this door deliberately did not,
    *"because the plan editor draws only OPEN notes, so closing the question on
    reply erases the top of the exchange from the screen"*, with the parent left
    to close *"when the thread is done, via* :func:`resolve_note` *"*. Nothing
    ever did that. And because the tray is OWNER-ONLY, the door that closed was
    the user's and the door that did not was the coordinator's — so the
    asymmetry ran in exactly one direction: a user's note, once answered, stayed
    open and kept asking the user to dismiss their own answered question.
    :func:`open_notes_for_you` has spelled out the coordinator half of AC-5.9
    since it was written; this makes that paragraph true.

    The cost the old rule was avoiding is real and is paid rather than
    reinstated: ``PlanNoteRow``'s *"Answering your question"* line carries the
    parent's own first line, so a closed question is still legible from the
    answer. The parent is resolved off the UNFILTERED note list precisely so it
    survives being closed.
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
        # WHAT THE CALLER TYPED, kept because the refusal below turns on it.
        # The reply derivation immediately after this can fill `section` in from
        # a parent whose own section has since been deleted, and answering a
        # note about a section somebody removed is exactly the conversation that
        # must stay possible. A slug a caller SUPPLIED is theirs to get wrong; a
        # slug the system supplied on their behalf is not.
        typed_section = section
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

        # ---- the section checked, then the base, then the quote ------------
        # THE ORDER IS THE ONLY ONE THAT WORKS, and each step feeds the next:
        # the section is settled first (derived, then salvaged, then checked),
        # because it decides which text the base is captured from, and the base
        # is what the quote is cut out of. Deriving the quote first would cut it
        # from a section this call had not settled on yet.
        #
        # THE TWO DERIVATIONS NEVER BOTH FIRE — each is gated on the other
        # being present — so a note carrying neither is left carrying neither.
        # That is not a failure to derive: it is the bottom rung of
        # `_derive_quote`'s ladder, and a note about the document as a whole
        # belongs to no section by definition.
        #
        # ALL COMPUTED ONCE, ABOVE THE LOOP, and the order matters there too.
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
        # override. `--section ''` WITH NO QUOTE is untouched and still means
        # "about the document as a whole" — `_capture_base` returns "" for it,
        # so nothing is derived and a deliberately unanchored note stays
        # sayable. `--section ''` WITH a quote is a different sentence and is
        # read as one: the excerpt already names a passage, so the section it
        # lives in is looked up rather than treated as a refusal to name one.
        #
        # NOT RETROACTIVE, for `_capture_base`'s reason: an anchor asserts what
        # its author was looking at, and inventing one for a note filed before
        # this existed asserts something false about a person.
        #
        # The quote step is gated on `section`, not on `base` alone.
        # `_capture_base` already returns "" without one, but a caller may
        # supply `base_section` explicitly with no section — and a quote CUT
        # FROM a section nobody named is a claim about a passage nobody can look
        # up. Note which direction that argues: it refuses to invent an excerpt
        # for an unnamed section, and says nothing against reading the section
        # off an excerpt the author supplied, which is the line below.
        section = section or _derive_section(body, quote)
        # Gated on `section`, not on `base` alone. `_capture_base` already
        # returns "" without one, but a caller may supply `base_section`
        # explicitly with no section — and a quote on a sectionless note is a
        # claim about a passage nobody can look up.
        #
        # SALVAGE, THEN REFUSE, THEN CAPTURE, THEN DERIVE, and each step feeds
        # the next.
        # The salvage may CHANGE which section this note lands on, so it runs
        # above the base capture (which reads that section's text), above the
        # layer resolution below (which is keyed on the slug), and above the
        # refusal, which must not fire on a slug that was just repaired.
        #
        # The salvage is asked of the EFFECTIVE section and the refusal only of
        # a TYPED one, and the asymmetry is deliberate: repairing an address is
        # always worth doing, and being told you got one wrong is only useful if
        # you wrote it.
        #
        # THE EXEMPTION COSTS NOTHING, and it is worth saying why rather than
        # trusting it. What it lets through is a note on a section the system
        # named, which is only ever a REPLY inheriting a parent whose section has
        # since been deleted — and a reply like that can never carry a proposal,
        # because `proposal and not section` is refused above the lock, so the
        # CLI resolves the parent's section itself and it arrives TYPED. A
        # comment is all that reaches here unchecked, and a comment on a section
        # that is gone writes nothing and corrupts nothing. The damaging shape —
        # a proposal against an absent section, which `resolve_note` reads as a
        # create and appends under whatever heading it opens with — always
        # carries a typed slug and is always checked.
        section = _salvage_section(body, section, quote)
        if typed_section:
            _refuse_unknown_section(body, section, proposal)
        base = base_section or _capture_base(body, section)
        if section and not quote:
            quote = _derive_quote(base, proposal, comment=comment)

        # ---- the layer, resolved ONCE, off the same read ------------------
        # Free here and nowhere else. `body` is already in hand for the base,
        # and the section named above is the one this note actually lands on —
        # `reply_to` has already filled it in — so the lookup is a dict get over
        # a walk that is happening anyway.
        #
        # BELOW the reply derivation on purpose: a reply inherits its parent's
        # section, and stamping before that would classify the note by a section
        # it does not have. Resolved against the CURRENT document, which is the
        # one its author is looking at; `PlanNote.layer` carries why this is
        # stamped rather than looked up when it is counted.
        #
        # AN UNKNOWN SLUG STAMPS `SPEC`, and so does `--section ''`. Both are
        # the blocking direction, and both are right for the same reason: a note
        # nobody can place is not a note the plan has delegated.
        layer = section_layers(body).get(section, SPEC) if section else SPEC

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
                    layer=layer,
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
            # **AC-5.9's OTHER HALF, and the reason it is here rather than in a
            # second copy of the rule.** BELOW the append, because a reply that
            # trips the open-note cap must close nothing —
            # `_add_notes_locked` validates and raises before anything is
            # filed, so an over-cap request leaves the parent exactly as it
            # was. Above `_save`, so the close and the note it answers persist
            # in one write or neither.
            #
            # INSIDE `if fresh`, with the activity row and the save. A fully
            # deduped request wrote nothing and must close nothing: the
            # identical reply that WAS filed already closed this parent, and
            # re-closing would restamp `resolved_at` on a thread nothing
            # happened to.
            #
            # `_auto_close_rounds` follows for `resolve_note`'s reason — a
            # round closes once every note in it is non-open, and this is now a
            # path that can make that true. Without it the coordinator's answer
            # would close the note and leave the round it arrived in open
            # forever, which is the bookkeeping half of the same asymmetry.
            # `keep_open` is the replier saying *not done yet* — a partial
            # answer, or a question back. The reply is filed either way; only
            # the parent's status differs, exactly as for the shapes
            # `_reply_closes_parent` refuses.
            if (
                parent is not None
                and not keep_open
                and _reply_closes_parent(project, plan, parent, by=by)
            ):
                _close_note(parent, status="answered", by=by)
                _note(plan, by, "resolve", section=parent.section,
                      detail=f"{parent.id} answered")
                _auto_close_rounds(plan)
                # **THE THIRD DOOR ONTO THE EXECUTION GATE.** Filing a note only
                # ever RAISES `open_notes_for_you`, so this is the one branch of
                # `add_note` that can lower it: the owner answering with
                # `--reply-to` closes a `to == user` parent, and if it was the
                # last one the gate goes down on an unaccepted plan. Hence the
                # call is in here with the close and not at the function's end,
                # where it would file a confirm note on plans no close touched.
                _hold_gate_locked(project, plan)
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


def _rows_worth_showing(
    rows: Sequence[StaleSection], body: str
) -> list[StaleSection]:
    """Which of a write's per-section rows are **worth putting in front of the
    user** — the ones whose change the plan's own rules call a spec change.

    **ONE HOME, TWO CALLERS, AND THAT IS WHY IT IS A FUNCTION.**
    :func:`_file_spec_lock_locked` asks it to decide which sections become
    proposals, and :func:`_prepare_locked` asks it to count what the refusal
    dropped. A second copy would be a second answer, and this filter has
    already drifted from its caller once — see below.

    **The decision to refuse is NOT taken here, and must not be.** That belongs
    to :func:`_prepare_locked`, which asks it of the UNFILTERED document;
    filtering there could let a write the lock means to refuse fall through and
    land. What is filtered here is only which sections are shown.

    The two halves came apart because they ask different questions. The lock's
    test is ``spec_digest(cleaned) != spec_digest(body)`` — normalized, and about
    the DOCUMENT. The rows come from :func:`split_by_section`, which is a raw
    byte comparison per section. So one section genuinely moving the spec
    dragged every other byte-changed section into a ``to=user`` note with it,
    including sections normalization folds away completely: an owner was handed
    a diff whose entire content was ``- [ ]`` becoming ``- [x]`` and asked to
    accept it — on the same document whose prompt promises that ticking a box is
    not a spec change. :func:`normalize_spec_text` IS the digest's own key, so
    this asks the lock's question rather than a second copy of it.

    **AND THE SAME DEFECT REACHED THROUGH A SECOND DOOR.** Normalization was the
    first filter; the layer is the second, and it exists for the identical
    reason. A write that legitimately moves the spec in one section — or drops a
    criterion's claim — also carries every milestone byte the keeper touched in
    the same call, and those rows are, by construction, sections the lock does
    not protect. Showing them would hand the user re-cut milestones to decide on
    a document whose whole promise is that re-cutting milestones is not their
    decision.

    ``body`` is the **stored** document — what the plan said BEFORE this write —
    on both call paths, and that is load-bearing rather than incidental: reading
    the layers out of the incoming text would let a keeper suppress its own note
    by marking the section ``<!-- layer: detail -->`` in the very write being
    filtered.

    **AND IF THAT WOULD LEAVE NOTHING, EVERYTHING STAYS.** The refusal path's
    contract is that the coordinator's text lands somewhere recoverable and
    :func:`_spec_lock_refusal` names the ids it filed; zero notes is the dead end
    AC-3.3 exists to prevent. It should be unreachable — a moved digest means
    some section's normalized text moved — but the fallback costs one comparison
    and removes the need to prove that.
    """
    layers = section_layers(body) if body else {}
    kept = [s for s in rows if layers.get(s.section, SPEC) == SPEC]
    kept = [
        s for s in kept
        if normalize_spec_text(s.base) != normalize_spec_text(s.text)
    ]
    return kept or list(rows)


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


def _row_adds_section(s: StaleSection) -> bool:
    """**A row with text and no base ADDS a section; it does not change one.**

    The mirror of :func:`_row_deletes`, added for the same reason and after the
    same kind of report: :func:`_spec_lock_comment` opened *"could not change
    ``X``"* on every row, so a refused ADD named a section the document does not
    have and told the user the keeper had failed to change it. A user read that
    as the keeper disputing what the plan already said.

    The rows that reach here with an empty base are not rare and are not
    mistakes: :func:`~clawmeets.models.plan_markdown.split_by_section` reports
    ``base == ""`` for every slug absent before the write and present after —
    an explicit create, the create half of a **retitle**, and a heading that
    appeared inside another section's replacement.

    ``s.text`` is required rather than assumed so the two predicates stay
    disjoint: a row with neither base nor text is the ABSENCE
    :func:`_spec_lock_comment`'s third arm already speaks for, and must not
    acquire a second sentence claiming something is being added.

    **Not** :func:`_row_creates`, which asks the same question of a tray
    ``DraftEntry`` and answers it from the NOTE's ``base_section`` because a
    row's own base cannot tell a create from a section deleted underneath it.
    Here there is no note yet — this row is what one is about to be filed FROM —
    and ``StaleSection.base`` is the write's own before-text, so the collision
    that function exists to break cannot arise. Two names, because they read
    different fields of different objects.
    """
    return bool(s.text) and not s.base


def _spec_lock_comment(
    *, by: str, section: str, text: str, deletes: bool, creates: bool = False
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

    **A FOURTH ARM, because "could not change" was false on an ADD.** A row
    with text and no base is a section the document does not have yet
    (:func:`_row_adds_section`), and the sentence named it as something the
    keeper had failed to *change* — so a user was handed a note about
    ``approval-2``, a slug nobody typed, reading as though the keeper disputed
    what their plan already said. The verb now follows the row. The predicate is
    passed in for the reason the paragraph above gives about ``deletes``: a
    sentence that re-derives a fact its own caller already settled is a second
    answer waiting to disagree with the first.

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

    **The lede names the user's review, not acceptance**, because the lock
    engages at whichever comes first (:func:`_spec_is_locked`) and *"the plan is
    accepted"* on a plan they have only reviewed is the note contradicting the
    Accept button beside it. A review is true in both states, so one sentence
    serves both without a flag.
    """
    verb = "add the section" if creates else "change"
    lede = (
        f"@{by} could not {verb} `{section}` directly — you have reviewed this "
        f"plan, so what it says changes only when you accept it"
    )
    if text:
        moves = "this adds to what it says" if creates else "this moves what it says"
        return f"{lede}; {moves}. Accept this to make the change."
    if deletes:
        return (
            f"{lede}; this write REMOVES the section. Accept it and "
            f"`{section}` is DELETED from the plan — there is no replacement "
            f"text, the diff below is the whole section against nothing, and "
            f"nothing puts it back. If you want it kept, reply to @{by} "
            f"instead, or make the change yourself."
        )
    return (
        f"{lede}; this write REMOVES the section, which moves what it says. "
        f"There is no replacement text to accept: reply to @{by}, or make the "
        f"change yourself."
    )


def _lead_with_why(why: str, comment: str) -> str:
    """Put the writer's one-line reason above the system's sentence.

    The reason is what the user scans for; the sentence beneath it says what
    Accept will do.
    """
    why = why.strip()
    return f"{why}\n\n{comment}" if why else comment


def _file_spec_lock_locked(
    project: "Project",
    plan: ProjectPlan,
    locked: Sequence[StaleSection],
    *,
    by: str,
    body: str = "",
    why: str = "",
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
    locked = _rows_worth_showing(locked, body)

    return _add_notes_locked(
        plan,
        [
            PlanNote(
                id="",
                section=s.section,
                to=OWNER,
                by=by,
                comment=_lead_with_why(
                    why,
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
                        # SAME RULE, SAME PLACE, for the reason the `deletes`
                        # comment above gives: the sentence never re-derives a
                        # fact the note's own fields already settle.
                        creates=_row_adds_section(s),
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


#: A receipt's ``resolution``, and **the discriminator** — the one field that
#: separates a receipt from every other closed note on the plan.
#:
#: It is a string in ``resolution`` and not a ``kind`` field for the reason
#: :class:`PlanNote` states in its own words: *"THERE IS NO ``kind`` FIELD, AND
#: ITS ABSENCE IS THE POINT."* Every word a surface prints for a note is
#: computed from the note's existing fields, and a receipt is nothing more
#: exotic than a comment the system filed and closed in the same breath. A
#: fifth :data:`NOTE_KINDS` member would be a second place to keep that rule.
#:
#: **Legacy only.** Receipts were filed for pre-acceptance keeper writes until
#: the spec lock moved to the user's first review (:func:`_spec_is_locked`);
#: nothing files one any more, and this survives so the ones already on disk
#: still render in the owner's changelog.
RECEIPT_RESOLUTION = "recorded — a keeper write before acceptance, nothing pending"

def _is_write_receipt(note: PlanNote) -> bool:
    """Is this note a pre-acceptance write receipt?

    Named because two surfaces ask it — the render that coalesces receipts into
    the owner's changelog, and every test that asserts a receipt never reaches
    the desk as a decision — and asking it by hand is how the ``resolution``
    string acquires a second, subtly different spelling.
    """
    return note.resolution == RECEIPT_RESOLUTION


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
    acting_agent: str = "",
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

    ``acting_agent`` is **how the owner's credential got here** and is empty for
    a real user JWT. It is not a second ``by``: ``by`` stays ``OWNER`` when it is
    set, so every ownership rule above reads exactly as it did. Two things need
    it and nothing else does — :func:`_refuse_presigned_go_note`, which refuses
    the one act a borrowed hand must not make, and ``accepted_via``, which said
    ``"owner"`` about an assistant's command. See
    :class:`~clawmeets.server.routes.project_plans.PlanIdentity`.
    """
    if action not in RESOLVE_ACTIONS:
        raise PlanInputError(f"Unknown action {action!r}")

    async with _lock:
        plan = _load(project, ctx)
        note = _find_note(plan, nid)

        # **FIRST, ABOVE EVERY OTHER REFUSAL ON THIS ROUTE.** The confirm note
        # carries no proposal, so an `apply` on it would otherwise come back as
        # *"carries no proposal"* — true, and useless: it names a missing field
        # where the caller needs to be told which verb this note takes and what
        # that verb does. One list, one permitted verb.
        _refuse_go_note_close(note, action, by=by)
        # **AND WHOSE HAND, which `by` alone cannot say.** The guard above asks
        # whether the caller may decide; this one asks whether the caller IS the
        # decider or is borrowing their credential. Kept separate because the
        # two questions have different inputs and only this door carries the
        # second one — `submit_review` calls it from its own pre-pass.
        _refuse_presigned_go_note(
            project, plan, note, action, acting_agent=acting_agent
        )

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
            if plan.accepted_at and result.ok:
                # **RE-STAMPING, AND IT IS THE SAME ACT.** Applying a proposal
                # is the owner putting their name to a piece of text, and on an
                # already-accepted plan that is precisely a fresh approval of
                # the document it produces. Without it `accepted_revision` would
                # be frozen at whatever revision the confirm note closed on, and
                # `changed_since_acceptance` would go true on the owner's own
                # applied deviation and stay true for the life of the project —
                # an amber warning, in the coordinator's prompt every turn,
                # about changes the owner themselves signed off on.
                #
                # **THE `note.bootstrap` CONJUNCT IS GONE BECAUSE IT IS NOW
                # UNREACHABLE, not because first acceptance stopped mattering.**
                # `apply` used to be the confirm note's one permitted verb and
                # therefore the CLI half of FIRST acceptance; `dismiss` is that
                # verb now, and it stamps at the bottom of this function.
                # Leaving the conjunct here would be a second acceptance trigger
                # behind a guard that always refuses first.
                #
                # **No `by == OWNER` conjunct, and that is deliberate.** The
                # guard at the top of this function is `may_decide`, so `by` is
                # the owner by the time control reaches here; restating the
                # comparison would give the ownership rule the second home
                # :func:`may_decide` exists to prevent.
                _stamp_acceptance_locked(
                    plan, body=prepared.body, by=by, acting_agent=acting_agent
                )

        # **THE CLI HALF OF ACCEPTANCE, AND IT IS A DISMISS NOW.** The only
        # close `_refuse_go_note_close` permits on the confirm note is this one,
        # so reaching here with `note.bootstrap` set means the owner said yes.
        # Off the STORED body, unlike `submit_review`'s twin: this path writes
        # no bytes, so the document as it stands is the document that was
        # approved.
        #
        # `may_decide` does not guard `dismiss` — a note's own addressee may
        # close its own — but the confirm note is addressed to the user and to
        # nobody else, and `may_write` cannot reach it either: the branch above
        # admits a writer only when `note.to != by`, and here it is `by`. So the
        # owner is the only identity that can arrive at this line.
        # **`status == "open"` IS THE WRITE-ONCE GUARD, and it is not
        # defensive.** This route does not refuse a close on an
        # already-closed note — `_close_note` just rewrites the same three
        # fields — so a second `--dismiss` on the confirm note reached this
        # stamp and moved `accepted_at`, `accepted_revision` and
        # `accepted_spec_digest` forward onto whatever the document said by
        # then. That silently clears a live *"the plan changed since
        # approval"* warning, which is the one thing acceptance stamps exist
        # to raise. `first_accepted_at` was already safe; these three were not.
        if action == "dismiss" and note.bootstrap and note.status == "open":
            _stamp_acceptance_locked(
                plan, body=_read_body(project, ctx), by=by,
                acting_agent=acting_agent,
            )
            # **AND WHETHER THAT DISMISS WAS ALSO A REVIEW, WHICH IT IS NOT
            # ALWAYS.** `submit_review`'s twin stamps unconditionally and is
            # right to: an owner composing a batch is an owner working in the
            # document. This door is the single-row dismiss on the desk card,
            # which reaches the confirm note without opening the plan at all —
            # so the fact it records has to be the one the row was offering.
            #
            # `plan.rounds` is that fact, and it is the same one the wording
            # flip keys on: empty means the coordinator has never put this plan
            # in front of the user, so what was dismissed said `GO_NOTE_DRAFTING`
            # — *"start on the draft as it stands"*, an instruction, not a
            # verdict on a document they read. Non-empty means a round went out
            # and the row had swapped to `GO_NOTE_COMMENT`, which is the confirm
            # and IS the review.
            #
            # Without this the acceptance and the review would latch together on
            # the one path where they genuinely come apart, and
            # `note_is_deviation`'s new conjunct would buy nothing.
            if _plan_was_offered(plan):
                _stamp_user_review_locked(plan, by=by)
        _close_note(note, status=_STATUS_FOR[action], by=by, reason=reason)
        _note(plan, by, "resolve", section=note.section, detail=f"{nid} {action}")
        _auto_close_rounds(plan)
        # **THE SECOND DOOR ONTO THE EXECUTION GATE, and it never asked.** This
        # route closes exactly one note, so it is the shortest path there is to
        # an unaccepted plan with nothing open for the user: retire the confirm
        # with a question round, then `plan note --dismiss` that question, and
        # `open_notes_for_you` — the whole of `_plan_execution_blocked` — reads
        # zero on a plan nobody confirmed. Only `submit_review` used to restore
        # the invariant, and only on a keeper's send.
        #
        # Below the close, so the count it reads is the one this call leaves
        # behind, and below the acceptance stamp above, so the owner's `dismiss`
        # of the confirm note is not answered with a fresh confirm note.
        _hold_gate_locked(project, plan)
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
    """The open confirm note, if this plan has one. **At most one is ever open**
    — which is a weaker claim than the *"exactly one per plan"* this used to
    make, and the weakening is :func:`_sync_go_note_locked`: a round that
    carries questions retires the note and a later quiet round files a fresh
    one, so a plan's history may hold several. Only the open one is ever asked
    for, and acceptance is still one-way.

    Public because three different questions are answered with it — does this
    reply close its parent, is there already one to seed, and does this round
    need one — and a predicate with three copies has three answers.
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


def _refuse_go_note_close(note: PlanNote, verb: str, *, by: str = OWNER) -> None:
    """**The go-note closes on ``dismiss`` and on nothing else.**

    The sibling of the reply exemption in :func:`_reply_closes_parent`, and it
    exists for the identical reason. The execution gate counts notes that are
    OPEN, so *every* way of closing this note releases it — and only one of them
    means the user said yes. A ``reject`` would leave the plan unapproved **and**
    unblocked: the coordinator would start executing the very plan the user had
    just turned down, which is the precise outcome this note exists to make
    impossible. So would ``answered``, a supersede and a fold.

    So the refusal is not a policy about what users may feel — it is the gate
    keeping its shape. Rejecting a plan is a real thing to want, and the act
    that expresses it is a *reply*: it reaches the coordinator, it asks for the
    revision, and it leaves the gate up while that happens. What has no
    coherent meaning is closing the gate to signal disapproval.

    **THE PERMITTED VERB USED TO BE ``apply``, AND THE INVERSION IS THE WHOLE
    CHANGE.** While the note was a proposal against ``## Approval``, saying yes
    meant applying a diff, and ``dismiss`` was refused here on the grounds that
    it released the gate without writing the approval. There is no section and
    no diff now, so there is nothing for an apply to land and ``dismiss`` — the
    verb already on every other row, meaning *"I'm not acting on this"* — is the
    only close the row offers. What keeps *"no"* expressible is unchanged and is
    the reason this guard still has a list to refuse: replying is how you turn a
    plan down, and a reply leaves the gate up.

    ``accept`` is on the refused list even though it can no longer write
    anything: a proposal-less accept would close the note ``applied`` and
    release the gate while :func:`_stamp_acceptance_locked` — which now hangs
    off the dismiss — never ran. That is the *"approved plan that says it is
    not"* failure one door over, and the old
    ``_refuse_unapprovable_go_note_accept`` existed for exactly it. Naming the
    verb here retires that function rather than the rule it carried.

    ``by`` defaults to :data:`OWNER` so a caller that has already established
    the identity — or is asking purely about the verb — reads unchanged. Both
    live call sites pass it.

    Called from the two paths that can close a note by id —
    :func:`resolve_note` and :func:`submit_review` — and from nowhere else,
    because a predicate with two copies has two answers.
    """
    if note.bootstrap and verb == "dismiss" and not may_decide(by):
        # **AND IT IS THE OWNER'S ALONE**, which `dismiss` does not otherwise
        # guarantee: `dismiss` is a REPORT rather than a decision everywhere
        # else, so `resolve_note` lets a note's addressee — or any writer —
        # close one, and `submit_review` deliberately leaves it outside the
        # ownership pass. On this note the report IS the decision: it stamps the
        # acceptance and releases the coordinator.
        #
        # It was covered by accident before this line existed. The confirm note
        # is never carried in a review batch (`pending_notes` excludes it), so
        # `note_was_sent` is always false for it and `_refuse_unsent_self_close`
        # caught the keeper on the "you wrote it and it has not been sent" arm.
        # That is a true sentence about the wrong rule, and it would stop being
        # true the day anything sent the note.
        raise PlanForbiddenError(
            f"@{by} may not dismiss the confirm note — dismissing it is how the "
            f"user approves this plan, and approving it is theirs alone. Use "
            f"`clawmeets plan note --to user` if you want to tell them "
            f"something about it."
        )
    if note.bootstrap and verb != "dismiss":
        raise PlanConflictError(
            f"The confirm note is the plan's execution gate, not a proposal to "
            f"{verb} — dismissing it is the only act that closes it, and that "
            f"is what starts the work. Reply to it to ask for changes: the "
            f"coordinator gets the question and the gate stays up while the "
            f"plan is revised."
        )


def _refuse_presigned_go_note(
    project: "Project",
    plan: ProjectPlan,
    note: PlanNote,
    verb: str,
    *,
    acting_agent: str,
) -> None:
    """**The keeper may not approve a plan the user has never been shown.**
    Raise, or return.

    The credential-layer sibling of :func:`_refuse_go_note_close`'s ownership
    arm. That one asks *may this identity decide?* and answers it with
    :func:`may_decide`. This one asks a question ``by`` cannot carry — *is this
    the decider, or the decider's assistant holding their credential?* — and it
    is a separate function because it takes a separate input: the acting agent,
    which only the plan route resolves and only two doors pass on.

    **THE INCIDENT, and why the two existing refusals did not fire.** On
    ``onboard-angel-investor`` the coordinator ran ``clawmeets plan resolve
    --dismiss --as-user`` against the confirm note at 11:56:32, two seconds
    after its own plan write, inside its own turn — and eight minutes before it
    first offered the plan. Both of the guards that exist for this are keyed on
    the caller's NAME, and ``resolve_user_from_credential`` had already rewritten
    that to ``"user"``:

    * :func:`_refuse_go_note_close` — *"dismissing it is how the user approves
      this plan, and approving it is theirs alone"*; and
    * :func:`_refuse_unsent_self_close` — *"you wrote it, it is addressed to the
      user, and it has not been sent yet"*, which covers the confirm note by
      construction because :func:`pending_notes` never carries it.

    So this is not a new policy. It is the rule those two already state, asked
    of the credential instead of the name.

    **THREE CONJUNCTS, AND THE THIRD IS WHAT KEEPS THE DOOR OPEN.**

    ``acting_agent == keeper(project)`` — not *any* assistant. The conflict this
    refuses is precisely the keeper approving its own draft; an owner's
    assistant relaying a decision on a project it does not coordinate is doing
    the job ``--as-user`` exists for.

    ``not _plan_was_offered(plan)`` — **the narrow version, deliberately.**
    Before the first round the row reads :data:`GO_NOTE_DRAFTING` and taking it
    means *"start on the draft as it stands"*; a plan that has never been put in
    front of the user cannot be a plan they meaningfully approved. Once the
    coordinator has sent its review, the document is on the user's desk and the
    assistant relaying *"they said approve it"* is a relay again — so this
    refusal lifts by itself and nothing legitimate is permanently lost. The
    wider reading (refuse the coordinating assistant always) would block that
    relay, which is a real thing owners ask for.

    ``verb == "dismiss"`` — the only close the confirm note takes at all
    (:func:`_refuse_go_note_close` refuses the rest by name), and the one that
    stamps the acceptance.

    **What the keeper loses is nothing it needs.** Before the plan is offered
    the spec lock keeps the pen with the keeper — :func:`_spec_is_locked` is
    False for it — so it can still write, restructure and re-draft the document
    freely. The single act taken away is putting the owner's name to it.

    The remedy is named because a refusal that is a dead end is how an agent
    retries forever: send the round, and let the user take the row.
    """
    if not note.bootstrap or verb != "dismiss" or not acting_agent:
        return
    if acting_agent != keeper(project) or _plan_was_offered(plan):
        return
    raise PlanForbiddenError(
        f"@{acting_agent} may not dismiss the confirm note on a plan it "
        f"drafted and has never sent. Dismissing it is how the user approves "
        f"this plan, and this plan has not been in front of them — you are the "
        f"keeper holding their credential, which is the one hand that cannot "
        f"make this decision. Send the round (`clawmeets plan review`) and let "
        f"them take the row; until then the document is still yours to write."
    )


def _stamp_offered_locked(plan: ProjectPlan) -> None:
    """Latch :attr:`ProjectPlan.first_offered_at` — the spec lock's other half.
    **Idempotent, and it never moves once set.**

    Takes no ``by``: any round means this document has been in front of the
    user, whoever opened it (:func:`_plan_was_offered`).

    **Called ABOVE the projection, and that placement is the whole of it.**
    :func:`_open_round_locked` runs in step 5, below the append that carries
    this batch's messages, so a predicate derived from ``rounds`` reads False
    for the ``plan_state_spec`` those messages ride with — and the coordinator
    turn they wake would be told it is still the document's writer during the
    very batch that hands the document over. Same hazard, same fix and same
    argument as :func:`_stamp_user_review_locked` two functions down; that one
    states it from the other side.
    """
    if plan.first_offered_at:
        return
    plan.first_offered_at = _now()


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


def _stamp_owner_seen_locked(
    plan: ProjectPlan, *, by: str, addressees: Iterable[str]
) -> None:
    """Latch :attr:`ProjectPlan.owner_last_seen_revision` — *"the owner has now
    seen the document at this revision"*. **It moves every round**, unlike its
    two neighbours, because the question it answers has a new answer each time.

    **Two arms, one fact.** The owner has seen this revision if a batch was
    addressed TO them, or if the batch was theirs — a review the owner submits
    is a review they wrote against the document in front of them. Stamping only
    the first arm would leave the banner claiming the plan moved since a round
    the owner themselves closed.

    **Called BELOW the render**, and the ordering is the whole of its
    correctness: the batch the owner is about to read reports the span *from*
    this value, so stamping first would collapse every banner to
    ``revision N → N`` and silently delete the signal. Same argument as
    :func:`_stamp_acceptance_locked`'s placement, reached from the other side —
    that one must be stamped before the projection reads it, this one after the
    message reads it.
    """
    if by != OWNER and OWNER not in addressees:
        return
    plan.owner_last_seen_revision = plan.revision


def _stamp_acceptance_locked(
    plan: ProjectPlan, *, body: str, by: str, acting_agent: str = ""
) -> None:
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
    and that is the whole reason it is a parameter. It used to be load-bearing
    for acceptance ITSELF — the apply wrote *"User approves the plan."* into
    ``## Approval``, a spec-layer section, so hashing the pre-write body stamped
    a digest the document no longer had and ``changed_since_acceptance`` read
    true the instant the plan was accepted. With ``## Approval`` gone, approving
    moves no byte and the two bodies agree on the dismiss path. It stays a
    parameter because the OTHER caller still needs it: a batch that accepts a
    proposal and re-stamps in the same transaction must hash the text that
    batch produced.

    ``accepted_revision`` is read off ``plan`` rather than passed, so it picks up
    :func:`submit_review`'s increment on that path and correctly does not on
    :func:`resolve_note`'s, which does not increment. Both are *"the revision
    this document had when it was approved"*.

    ``accepted_via`` names **which hand**, and it used to say ``"owner"``
    unconditionally on the grounds that :func:`may_decide` is the door on both
    paths and admits nobody else. That sentence is true about the NAME and false
    about the credential: the owner's ``{username}-assistant`` bearer resolves to
    the owner, so an assistant's ``--as-user`` command wrote ``accepted_by:
    user``, ``accepted_via: owner``, ``history: by user, verb approve`` — three
    fields, all of them the owner, none of them the actor. ``by`` still says
    ``user`` and should: the assistant really was acting with the owner's
    authority, on their instruction. This field is where the difference goes.
    """
    plan.accepted_at = _now()
    # WRITE-ONCE, and the guard is the whole of the difference between this
    # field and the line above it. `accepted_at` re-stamps on every owner apply
    # because that is what "when they LAST put their name to this text" means;
    # `first_accepted_at` is the boundary `note_is_deviation` compares against
    # and moving it would demote every deviation filed since the last apply.
    plan.first_accepted_at = plan.first_accepted_at or plan.accepted_at
    plan.accepted_by = by
    plan.accepted_via = "assistant" if acting_agent else "owner"
    plan.accepted_revision = plan.revision
    plan.accepted_spec_digest = spec_digest(body)
    # Section-less, like the note it records: approval is a fact about the
    # document rather than about one heading in it.
    _note(plan, by, "approve")


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


class BatchDecision(BaseModel):
    """What **this submit** decided about ONE note, for ONE addressee.

    A model rather than a tuple so a renderer cannot mistake ``kind`` for
    ``reason`` positionally, and so a later field lands here rather than
    widening every call site.

    ``kind`` is the DECISION, not the row's kind: the two agree on ``accept``,
    ``reject``, ``ask`` and ``dismiss``, and diverge on the one case the row
    alone cannot answer — an ``accept`` that ``_collapse_text_rows`` classified
    superseded decided ``fold`` or ``superseded``, never ``accept``. Deriving it
    from ``row.kind`` alone would print *"accepted"* to an author whose text is
    not the section, which is the exact lie this whole change exists to remove.
    """

    kind: str          # accept|fold|superseded|reject|ask|dismiss
    reason: str = ""


#: The sentence per decision, and the ``{reason}`` half is appended **only** when
#: the row carries one — because two kinds structurally never do and the line
#: must not print a dangling dash:
#:
#: * ``accept`` — the decisions loop passes no reason at all, so it is
#:   decision-only and this module does not invent one for it.
#: * a BROWSER ``dismiss`` — the tray's dismiss row sets no comment, so it prints
#:   a bare decision. Honest but thin; inventing a reason here would be the
#:   renderer making up a fact the user never typed.
#:
#: ``fold`` and ``superseded`` map to the EMPTY string on purpose: their reasons
#: (:data:`FOLDED_REASON`, :data:`SUPERSEDED_REASON`) already name their own verb
#: and are whole sentences, so a prefix would say the word twice.
DECISION_TEXT: dict[str, str] = {
    "accept": "accepted",
    "reject": "rejected",
    "ask": "replied",
    "dismiss": "dismissed",
    "fold": "",
    "superseded": "",
}
DECISION_LINE = "**Decided this round:** {text}"
DECISION_WITH_REASON = "{text} — {reason}"


def _decision_of(row: DraftEntry, *, whole_sentence: str = "") -> BatchDecision:
    """The decision one submitted ROW made, built **from the row**.

    From the row and never from the note, because of the ordering this
    transaction depends on: rendering is step 2 and the closes are step 3, so at
    render time the note is still ``open``, its ``resolution`` is still empty,
    and the ``accept`` arm has not yet moved ``proposal`` into ``applied_text``.
    A line derived from note state would print *"open"* for every decision in
    the batch. See :func:`submit_review`'s step 3 for why hoisting the closes
    above the render is not the fix.

    ``whole_sentence`` is the caller's answer for the two cases the row cannot
    answer alone — a superseded row, folded or not — and it is the SAME string
    the close writes to ``note.resolution``, computed once by
    :func:`_close_spec_for_superseded` and passed here, so the message and the
    stored record can never word the same fact differently.
    """
    if whole_sentence:
        return BatchDecision(
            kind="fold" if row.folded else "superseded", reason=whole_sentence
        )
    return BatchDecision(kind=row.kind, reason=row.comment)


def _decision_line(decision: BatchDecision | None) -> str:
    """One rendered decision line, or ``""`` when there is nothing to say.

    **``""`` is the byte-identity contract.** The caller appends nothing for an
    empty string, so a render with no decisions passed is character-for-character
    what it was before this existed. That is the single most load-bearing
    default in this change: the quote and diff tests that render a note directly
    pass no decisions, and neither does ``plan review --dry-run`` — which is not
    a stub, because ``cli_plan.review`` POSTs only ``note_ids`` and never
    ``entries``, so a CLI review batch carries zero decisions by construction and
    the dry run's *"the preview IS the message"* assertion stays true
    permanently rather than accidentally.
    """
    if decision is None:
        return ""
    text = DECISION_TEXT.get(decision.kind, decision.kind)
    if not text:
        text = decision.reason
    elif decision.reason:
        text = DECISION_WITH_REASON.format(text=text, reason=decision.reason)
    return DECISION_LINE.format(text=text) if text else ""


#: How many receipt lines the owner's changelog block shows before it stops
#: listing and starts counting. High enough that an ordinary round of feedback
#: fits whole; low enough that a runaway keeper cannot bury the notes under it.
MAX_CHANGELOG_LINES = 15


def _owner_changelog(
    addressee: str,
    all_notes: Sequence[PlanNote],
    *,
    revision: int,
    last_seen: int,
) -> list[str]:
    """The owner's *"here is what moved since you last read this"* block.

    **THE OTHER HALF OF LETTING THE KEEPER WRITE.** Before acceptance a keeper's
    spec change lands instead of arriving as a proposal, which buys the user a
    document that is coherent every time they open it and costs them the diff
    they used to accept hunk by hunk. This block is what they get instead: the
    span the document moved over, and a named line per change. Without it the
    trade is not a trade — it is the ``chuswine-geo-b2b`` failure with better
    manners, 26 silent writes and nothing on any surface naming one.

    So the lines are **named changes, not a summary**, and that is a property of
    the ``--why`` the write path requires rather than of this render: the user
    answers by quoting one line back, and a line they cannot name is a line that
    forces them to re-read the whole document to object to one thing.

    Receipts are selected by ``revision >= last_seen`` rather than by timestamp
    because ``plan.revision`` only moves inside :func:`submit_review` — so every
    receipt filed between two rounds carries the revision that round opened at,
    and the comparison is exact rather than approximately-ordered. They are
    identified by :func:`_is_write_receipt`, which reads ``resolution``: there is
    no ``kind`` field to read and there is deliberately not going to be one.

    Returns ``[]`` — not a heading with nothing under it — for every case that
    is not *the owner, on a document they have seen before, that has since
    moved*. An empty section a reader learns to skip is worse than no section.
    """
    if addressee != OWNER or not last_seen or last_seen == revision:
        return []
    receipts = [
        n for n in all_notes
        if _is_write_receipt(n) and n.revision >= last_seen
    ]
    if not receipts:
        return []
    lines = [
        "",
        f"**This plan moved — revision {last_seen} → {revision} — since your "
        f"last review round.** These changes landed directly, so read the "
        f"document end to end rather than diffing it: plan sections are "
        f"entangled by design, and narrowing one milestone changes what the "
        f"next one depends on.",
        "",
        "**What moved, and why**",
    ]
    for note in receipts[:MAX_CHANGELOG_LINES]:
        lines.append(f"- {note.comment}")
    if len(receipts) > MAX_CHANGELOG_LINES:
        lines.append(
            f"- _…and {len(receipts) - MAX_CHANGELOG_LINES} more — "
            f"`clawmeets plan list-notes <project>` lists them all._"
        )
    lines.append(
        "Accept the plan if it reads right, or reply naming the line you want "
        "changed."
    )
    return lines


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
    last_seen: int = 0,
    all_notes: Sequence[PlanNote] = (),
    history: Sequence[PlanHistoryEntry] = (),
    decisions: Mapping[tuple[str, str], BatchDecision] | None = None,
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

    ``decisions`` is what THIS submit decided, and it is keyed by
    **(ADDRESSEE, NOTE ID)** — never by note id alone. Three reasons from this
    module, and the third is the one with teeth:

    1. ``submit_review``'s ``by_addressee`` is filled by TWO loops — the
       decisions loop and the plain-send loop over ``note_ids`` — and this
       function is handed the merged list. One message can therefore carry a
       decided note beside a freshly-sent undecided one.
    2. The same note can reach two addressees in one batch (the ``seen`` set is
       itself keyed on the pair), so a note-keyed map has no room for the case.
    3. Dismissing your OWN note tells nobody, and that skip is an ADDRESSING
       decision: it ``continue``s before the row reaches ``by_addressee``. A
       note-keyed map would still hold that dismissal and would attach it to the
       same note arriving by the SEND path — telling the user *"dismissed"*
       about a note they were merely shown. The pair key drops it, because the
       addressee it was skipped for is never a key.

    ``last_seen`` is the revision this addressee last had this document put in
    front of them (:attr:`ProjectPlan.owner_last_seen_revision`), and it drives
    **the owner's changelog block** — the banner naming the span the plan moved
    over, plus the receipts filed inside it. That block is what makes
    *"re-read the document"* a reasonable instruction rather than an unbounded
    one: the user is told which sections moved and why before they open it.

    Rendered for the OWNER alone, and only once they have seen the plan before
    (``last_seen`` truthy and different from ``revision``). A specialist reviews
    against the current text and has no baseline of their own, and an owner's
    FIRST round has nothing to be a changelog *since* — handing them the draft's
    whole construction history at the moment they are asked to read the draft is
    noise, not evidence.

    Left at their defaults, ``last_seen`` and ``decisions`` both make the render
    byte-identical to what it was before either parameter existed — see
    :func:`_decision_line`.
    **Placement is deliberate:** the decision sits immediately under the note it
    decided and above the evidence, so an addressee reading top-down learns the
    outcome before the diff. And it reaches whoever the ROW addresses, which is
    not necessarily the note's author — a ``to``-overridden reject reaches the
    person the user named, and the line goes with it.
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
    lines += _owner_changelog(addressee, all_notes, revision=revision, last_seen=last_seen)

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

        decided = _decision_line((decisions or {}).get((addressee, note.id)))
        if decided:
            lines += ["", decided]

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

    example = notes[0].id if notes else "n-xxxx"
    lines += [
        "",
        "---",
        "### How to respond",
        f'- Answer or discuss: `clawmeets plan note {ref} --reply-to {example} -m "..."`',
        f"- Answer with a change: `clawmeets plan note {ref} --reply-to {example} "
        f'--edit-file <the section\'s whole new text> -m "why"` — a diff they '
        f"accept or reject",
        "- Either reply closes the note you answer. Do not leave a note open "
        "after acting on it.",
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


def _edit_for_row(
    plan: ProjectPlan, body: str, row: DraftEntry
) -> SectionEdit | None:
    """**The one answer to "what does this row write", or ``None`` for nothing.**

    Both questions in this transaction are downstream of it: what
    :func:`apply_edits` splices, and — through :func:`_collapse_text_rows` —
    which rows are competing to splice the same section. They were two
    predicates before, and the pair is what let a row be superseded for a
    section it was never going to write.

    **A ROW THAT NAMES ITS SECTION IS TAKEN AT ITS WORD.** ``section``, ``text``
    and ``base`` are what the user staged against text they were looking at, and
    ``base`` is §4.4's whole staleness contract. Nothing here second-guesses it;
    only ``create`` is derived, for the reason :func:`_row_creates` gives.

    **AN ``accept`` THAT NAMES NO SECTION IS ANSWERED FROM ITS NOTE**, which is
    where the answer has always been. ``{"kind": "accept", "note_id": ...}`` is
    the canonical accept body — the shape ``can_decide`` advertises and the one
    ``DraftEntry`` permits, since ``section`` is optional — and it used to write
    no bytes at all while the accept arm closed the note ``applied`` with
    ``applied_text`` set. The note then claimed to be in a document it was not
    in, on every surface at once: the tray drew its diff out of ``applied_text``
    beside a section that still read the old way, its author was told
    *"accepted"* and stopped waiting, and ``revision`` never moved, so
    ``plan show --versions`` had nothing for the user to find the loss by. The
    text was unrecoverable by then — an accept empties ``proposal`` into
    ``applied_text``, and :func:`resolve_note` refuses ``--apply`` on a note
    that carries no proposal.

    So the three fields come off the NOTE, exactly as :func:`resolve_note`'s
    apply arm has always taken them, and the two doors onto one decision stop
    disagreeing about what that decision writes.

    **``base`` IS THE NOTE'S, EXCEPT WHEN THERE ISN'T ONE.** ``base_section`` is
    the section's bytes captured when the note was FILED, so handing it to
    :func:`_splice` makes the staleness question here identical to
    :func:`section_changed`'s — down to the ``409`` and the
    :class:`StaleSection` payload ``PlanCollide`` already knows how to re-base.
    The ``or`` carries ``section_changed``'s own exemption rather than a second
    opinion about it: ``""`` means *nothing was captured* — a note filed before
    :func:`_capture_base` widened, which is deliberately never backfilled — and
    comparing an empty base against real text would refuse every one of those
    notes forever. The current extent is what *"there is nothing to compare"*
    looks like written as a base, and it is what ``resolve_note`` passes on
    every apply.

    ``None`` is still a real answer, for the rows that genuinely write nothing:
    a row kind that is not a write, an accept on a note that only asked a
    question, and an accept on a note naming no section. A
    :attr:`PlanNote.proposes_delete` note IS a write — its empty text is the
    delete — which is why the test below is ``proposal or proposes_delete`` and
    not the string alone.

    **The one thing that got louder.** A call that used to ``200`` and write
    nothing can now ``409`` when the section moved or vanished under the note.
    That is the correct refusal and the one ``resolve --apply`` has always
    given; it is a refusal appearing on a route that never refused, because the
    route never did anything.
    """
    if row.section:
        return SectionEdit(
            section=row.section,
            text=row.text,
            base=row.base,
            # **AND THE ROW'S OWN BASE IS NOT THE INPUT.** See `_row_creates`: an
            # accept whose section has vanished carries `base == ""` whether it is
            # a restore or a conflict, so the flag is derived from the NOTE, which
            # knows which. It is consulted by `_splice` only when the slug fails to
            # resolve, so it is inert on every row whose section is still there.
            create=_row_creates(plan, row),
        )
    if row.kind != "accept" or not row.note_id:
        return None
    note = _find_note(plan, row.note_id)
    if not note.section or not (note.proposal or note.proposes_delete):
        return None
    return SectionEdit(
        section=note.section,
        text=note.proposal,
        base=note.base_section or (_current_section(body, note.section) or ""),
        # The same rule the row-carried branch uses and the same one
        # `resolve_note` computes as `creating`: an empty capture means there was
        # nothing there when this was written, which is a create.
        create=_row_creates(plan, row),
    )


def _collapse_text_rows(
    plan: ProjectPlan, body: str, entries: Sequence[DraftEntry]
) -> tuple[list[DraftEntry], list[DraftEntry]]:
    """§2.6, one layer down: ``apply_edits`` splices one replacement per section,
    so **at most one row per section may carry text**.

    The later row wins — that is the rule the user set. Returns
    ``(applied, superseded)``; nothing is dropped silently, because every
    superseded row's note resolves ``rejected`` with a written reason.

    **ONLY THE ROWS THAT WRITE ARE RACED, and the question is put to
    :func:`_edit_for_row` rather than answered here.** That is the same
    function the splice reads, so what competes to write a section and what
    actually writes it cannot drift apart — and they had. This keyed on the raw
    ``row.section``, which an ``accept`` staged by ``note_id`` alone leaves
    empty: every such row landed in one bucket under ``""``, so a batch that
    decided two of them closed all but the last ``rejected``, carrying
    :data:`SUPERSEDED_REASON` — *"superseded by @X's proposal on the same
    section"*, said about rows that name no section and propose no text, and
    said to the author of a decision the user had actually made.

    ``test_d14_the_tray_re_signs_too_and_a_batch_stamps_exactly_once`` is that
    batch, in this repo's own suite: two proposals, two different sections, one
    submit, and the first of them silently thrown away.

    This function's contract is *a superseded row lost a race to write a
    section*. It can only hold if the race is run over what is written, which
    is why the predicate moved rather than merely growing an exception.
    """
    text_rows = [e for e in entries if e.kind in ("edit", "accept")]
    writes = {id(e): _edit_for_row(plan, body, e) for e in text_rows}
    last_per_section: dict[str, DraftEntry] = {}
    for entry in text_rows:
        edit = writes[id(entry)]
        if edit is not None:
            last_per_section[edit.section] = entry
    losers = {
        id(e) for e in text_rows if writes[id(e)] is not None
    } - {id(e) for e in last_per_section.values()}
    return (
        [e for e in text_rows if id(e) not in losers],
        [e for e in text_rows if id(e) in losers],
    )


def _winner_note_for(
    plan: ProjectPlan,
    body: str,
    row: DraftEntry,
    applied_rows: Sequence[DraftEntry],
) -> PlanNote | None:
    """The note behind the row that WON ``row``'s section, or ``None``.

    **Not a guess, and not taken from the wire.** It is the exact inverse of
    :func:`_collapse_text_rows`' own keying: that function builds
    ``last_per_section[...]``, one winner per distinct section, so for any
    superseded row writing section ``S`` there is exactly one applied row
    writing ``S``. The lookup is total and single-valued by construction, which
    is why the fold field carries no winner identity — there is nothing for the
    client to tell the server that the server does not already know, and a
    client-side snapshot of the winner would be stale by construction anyway.

    **THROUGH :func:`_edit_for_row`, BECAUSE THAT IS THE SECTION THAT WAS
    RACED.** ``row.section`` is empty on an accept staged by ``note_id`` alone,
    and matching on the raw field would look for a winner under ``""`` — a key
    the collapse never used — and name whichever unrelated row happened to
    carry it, or nobody. The inverse of a keying has to invert the same key.

    ``None`` is a real answer, not a failure: the winner may be the user's own
    ``edit`` row, which carries no ``note_id``. :func:`_close_spec_for_superseded`
    is what turns that into a name.
    """
    target = _edit_for_row(plan, body, row)
    if target is None:
        return None
    winner = next(
        (
            r for r in applied_rows
            if (edit := _edit_for_row(plan, body, r)) is not None
            and edit.section == target.section
        ),
        None,
    )
    return (
        _find_note(plan, winner.note_id)
        if winner is not None and winner.note_id
        else None
    )


def _close_spec_for_superseded(
    row: DraftEntry, winner_note: PlanNote | None, *, by: str
) -> tuple[str, str]:
    """``(status, reason)`` for ONE row in the superseded set.

    ``folded`` is honoured only on a row the loop already acts on — an
    ``accept`` carrying a ``note_id`` — so the flag is inert on every other row
    shape and needs no validation arm of its own. An absent or false flag returns
    exactly ``("rejected", SUPERSEDED_REASON.format(...))``, which is this
    module's behaviour before the field existed, character for character. That
    identity is the whole proof the field is additive.

    **THE WINNING AUTHOR, and why it is never empty on a fold.** The reason names
    the winner, and a name is what makes the sentence say anything at all — with
    it empty the surface can only say *"folded"* with no object. ``winner_note``
    is ``None`` when the winner is the user's own ``edit`` row, and
    :attr:`PlanNote.by` carries no non-empty guarantee of its own, so the author
    resolves through **two** fallbacks — ``(winner_note.by if winner_note else "")
    or by`` — landing on the submitter, who is never empty. Drop the second
    ``or`` and the guarantee becomes a hope.

    The ``rejected`` arm keeps its ONE fallback deliberately: changing it would
    change a sentence this project is not here to change.
    """
    if row.folded:
        return (
            "folded",
            FOLDED_REASON.format(by=(winner_note.by if winner_note else "") or by),
        )
    return (
        "rejected",
        SUPERSEDED_REASON.format(by=(winner_note.by if winner_note else by)),
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
    why: str = "",
    render: MessageRenderer | None = None,
    post: OwnerPoster | None = None,
    acting_agent: str = "",
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

    ``why`` is :func:`apply_edits`'s parameter, taken here for the same write and
    forwarded unchanged, so a keeper ``edit`` row the spec lock refuses files
    its proposal with the same lead line a ``PUT …/plan`` would. It is **not**
    ``batch_comment`` — that is the sentence above the
    notes in the rendered message, addressed to whoever the batch is sent to;
    this is the line filed on the document, addressed to the owner.
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
            # **EVERY CLOSING KIND, and `dismiss` is the one that passes.**
            # These reach the confirm note the same way the CLI's do, and all
            # but one of them would release the gate on a plan nobody approved.
            # `accept` is on the list for the reason `_refuse_go_note_close`
            # spells out: the note carries no proposal, so an accept writes
            # nothing, closes it `applied`, and never reaches the stamp — the
            # failure the retired `_refuse_unapprovable_go_note_accept` guarded.
            # `ask` is absent because a reply closes nothing here
            # (:func:`_reply_closes_parent`).
            if row.kind in ("accept", "reject", "dismiss") and row.note_id:
                _refuse_go_note_close(
                    _find_note(plan, row.note_id), row.kind, by=by
                )
            # BOTH DOORS, one rule. `submit_review` is owner-only for `accept`
            # and `reject` but NOT for `dismiss`/`ask`, so a keeper batch can
            # reach a close from here too — and a guard that lived only in
            # `resolve_note` would be a rule with one home and two answers.
            # `dismiss` alone: it is the only closing kind a tray row carries
            # that is not already owner-gated above (`accept`/`reject`), and
            # `ask` does not close anything.
            if row.kind == "dismiss" and row.note_id:
                note = _find_note(plan, row.note_id)
                _refuse_unsent_self_close(plan, note, by=by)
                # **BOTH DOORS, one rule** — the paragraph above, applied to
                # the credential guard as well. `by` is `OWNER` on an
                # assistant bearer, so a rule keyed on the name alone is
                # disarmed here in exactly the way it was on `resolve_note`.
                _refuse_presigned_go_note(
                    project, plan, note, row.kind, acting_agent=acting_agent
                )

        comment = batch_comment or draft.batch_comment
        room_name = room or draft.room

        # **THE DOCUMENT, READ ONCE AND ABOVE EVERYTHING THAT ASKS ABOUT IT.**
        # `_edit_for_row` needs it to answer an `accept` staged by `note_id`
        # alone — the note's `base_section` is the staleness input, and the
        # section's CURRENT extent is the fallback for a note filed before that
        # capture existed. `_prepare_locked` re-reads the same bytes under the
        # same lock, so the collapse races the rows over exactly the body the
        # splice will check them against.
        body = _read_body(project, ctx)
        applied_rows, superseded_rows = _collapse_text_rows(plan, body, rows)
        # **WHAT EACH ROW WRITES IS ASKED ONCE, OF ONE FUNCTION.**
        # `_collapse_text_rows` just raced these rows over the very sections
        # `_edit_for_row` names, so the set that reaches the splice and the set
        # that competed to reach it cannot disagree — which they did while this
        # was a second, hand-written predicate here.
        #
        # `None` is *this row writes nothing*, and it is a real answer for two
        # shapes: accepting a note that only asked a question, and any row kind
        # that is not a write. Both are still staged, decided and resolved
        # below; they simply splice no bytes.
        edits = [
            edit for edit in (_edit_for_row(plan, body, r) for r in applied_rows)
            if edit is not None
        ]

        # ---- 1. the write, COMPUTED but not yet appended ------------------
        # **``why`` reaches the funnel from HERE too**, so a keeper spec move
        # carried inside a review batch and refused by the lock files its
        # proposal with the same lead line as a `plan update` would.
        prepared = _prepare_locked(project, ctx, plan, edits, by=by, why=why)
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
            ids = _file_spec_lock_locked(
                project, plan, result.locked, by=by, body=prepared.body,
                why=prepared.why,
            )
            _note(
                plan, by, "refused",
                detail=",".join(s.section for s in result.locked),
            )
            _save(project, ctx, plan)
            raise PlanSpecLockedError(
                _spec_lock_refusal(
                    project,
                    by=by,
                    note_ids=ids,
                    relayered=result.relayered,
                    moved=result.moved,
                    added=result.added,
                    dropped=result.dropped,
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
        # **What this submit decided, told to whoever it decided it AT.** Built
        # here, in the addressing loop, because the key is the pair — see
        # `render_batch_message` for the three reasons a note-keyed map is
        # wrong, and `_decision_of` for why it is built from the ROW and not
        # from the note it is about.
        #
        # The superseded set is already in hand (`_collapse_text_rows` ran well
        # above), so the two rows this loop cannot classify on its own — an
        # `accept` that lost its section, folded or not — are classified without
        # a lookahead, by the same function step 3 closes them with. One string,
        # written once, sent and stored.
        decided: dict[tuple[str, str], BatchDecision] = {}
        superseded_ids = {id(r) for r in superseded_rows}

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
                addressee = row.to or note.by or keeper(project)
                whole = ""
                if id(row) in superseded_ids:
                    _status, whole = _close_spec_for_superseded(
                        row, _winner_note_for(plan, body, row, applied_rows), by=by
                    )
                if addressee:
                    decided[(addressee, note.id)] = _decision_of(
                        row, whole_sentence=whole
                    )
                _address(addressee, note)
        # A NOTE ADDRESSED TO THE SENDER IS ITS INBOX, NOT A SEND. On
        # `agent-home-folder-view` the coordinator ran `plan review` after
        # folding the user's answers in; the user's three replies — addressed to
        # it and still open — were its default set, and went out as a round
        # addressed to itself that nothing would ever wake it for. They are
        # answered with `plan note --reply-to`, never re-sent.
        #
        # AGENTS ONLY. The owner sending a keeper's `to=user` notes lands a
        # message on their own plate and wakes nobody — the supported way an
        # owner's terminal pushes a batch out — so it stays a send.
        own_inbox: list[str] = []
        for nid in note_ids:
            note = _find_note(plan, nid)
            addressee = note.to or keeper(project)
            if addressee == by and by != OWNER:
                own_inbox.append(nid)
                continue
            _address(addressee, note)
        if own_inbox and not by_addressee and not rows:
            raise PlanInputError(
                f"{', '.join(own_inbox)} "
                f"{'is' if len(own_inbox) == 1 else 'are'} addressed to you — "
                f"they are yours to answer, not to send. Answer each with "
                f"`clawmeets plan note <project> --reply-to <id> -m \"...\"` "
                f"(add `--edit-file <section.md>` to propose the change it asks "
                f"for); the reply closes the note and reaches its author."
            )

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
                # READ HERE, STAMPED LATER. This is the revision the owner saw
                # at their PREVIOUS round, and the banner's whole content is the
                # span between it and the one above;
                # `_stamp_owner_seen_locked` moves it forward only after every
                # message has rendered.
                last_seen=plan.owner_last_seen_revision,
                all_notes=plan.notes,
                history=plan.history,
                # Bound HERE, after the decisions loop filled it, and passed as
                # a keyword on the partial rather than through
                # `MessageRenderer` — which is unchanged, and that is the point.
                # The two callers that render without a decision keep working BY
                # OMISSION rather than by edit: an injected renderer and
                # `plan review --dry-run` both call `render_batch_message`
                # directly and neither has a decision to pass.
                decisions=decided,
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
                # INHERITED FROM THE PARENT, not re-resolved off `prepared.body`.
                # This row takes the parent's `section` unconditionally, so the
                # two fields must agree, and the parent is the one that already
                # answered the question at the moment it could still be asked.
                # Re-resolving would also make a reply's blocking-ness depend on
                # a marker flip that landed between the question and the answer —
                # the retroactive case `PlanNote.layer` exists to rule out.
                layer=parent.layer,
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

        # Set by an `accept` row below, or by a `dismiss` of the confirm note,
        # and consumed once after the loop. See either row for why a batch must
        # not stamp per decision.
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
                # **THE BROWSER HALF OF ACCEPTANCE.** `_refuse_go_note_close`
                # ran in the pre-pass above and permits exactly this verb on the
                # confirm note, so reaching here with `note.bootstrap` set means
                # the owner said yes. Same flag, same one call below the loop,
                # for the same reason the `accept` arm sets it rather than
                # stamping: a batch may carry several closes and the history
                # must record ONE approve.
                #
                # `status == "open"` is the same write-once guard `resolve_note`
                # carries, for the same reason: a row naming an
                # already-dismissed confirm note would re-stamp the acceptance
                # onto a later document and clear a live *"changed since
                # approval"* warning.
                if note.bootstrap and note.status == "open":
                    stamp_acceptance = True
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
                # in the middle of questioning. Dismissing it is the only
                # thing that means yes, so dismissing it is the only thing that
                # closes it.
                #
                # The outgoing question is filed either way: a reply to the
                # go-note is a real message to the coordinator and must reach
                # it. Only the parent's status differs.
                #
                # **THE GO-NOTE TEST MOVED INTO `_reply_closes_parent` AND TOOK
                # THE REST OF THE RULE WITH IT.** This arm was `not
                # note.bootstrap` and was the ONLY place AC-5.9 was implemented
                # — and this arm is reachable by the owner alone, because the
                # tray is owner-only. So the half of AC-5.9 that runs when the
                # COORDINATOR answers never ran: `add_note` left the parent
                # open, and every note the user wrote stayed open after it had
                # been answered. Both doors ask the predicate now, which is the
                # same one-home argument `_reply_addressee` makes for the
                # addressee three fields over.
                #
                # Nothing about the owner's path changes shape: `may_write` is
                # true for the owner, so only the go-note and an
                # already-resolved parent are withheld here, and the second was
                # never something a live row could reach.
                if _reply_closes_parent(project, plan, note, by=by):
                    _close_note(note, status="answered", by=by)
                # The note object is the one validated above the append, not a
                # new one built to match it.
                _add_notes_locked(plan, [ask_notes[idx]])

        for row in superseded_rows:
            if row.kind != "accept" or not row.note_id:
                continue
            note = _find_note(plan, row.note_id)
            status, reason = _close_spec_for_superseded(
                row, _winner_note_for(plan, body, row, applied_rows), by=by
            )
            # **EVERY way of closing the confirm note releases the execution
            # gate, and only `dismiss` means the user said yes** — that is
            # `_refuse_go_note_close`'s entire argument. This loop is the one
            # closing path in the module that calls `_close_note` directly: the
            # ownership pass above guards `reject`/`dismiss` rows and
            # `resolve_note` guards its own, so the guard belongs to THE LOOP,
            # not to either of its two branches.
            #
            # It was briefly on the `folded` branch alone, and the branch it
            # skipped was reachable by simply not setting the flag: stage an
            # accept on the go-note, stage any later row on `## Approval` behind
            # it, and `_collapse_text_rows` hands the go-note's row here as a
            # loser. It closed `rejected` with `SUPERSEDED_REASON`, and because
            # the acceptance stamp keys off the row being submitted rather than
            # its text landing, the plan came out marked ACCEPTED with
            # `## Approval` still reading whatever the winning row wrote. Gate
            # open, plan unapproved, document self-contradicting.
            #
            # The verb tracks the act so the refusal names what was attempted;
            # the sentence it builds is identical in shape either way.
            #
            # The raise unwinds the whole submit before step 4's append: no
            # bytes written, no message sent, no note closed, tray untouched.
            _refuse_go_note_close(note, "fold" if row.folded else "supersede")
            _close_note(note, status=status, by=by, reason=reason)

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
            _stamp_acceptance_locked(
                plan, body=prepared.body, by=by, acting_agent=acting_agent
            )

        # **WHETHER THIS ROUND LEAVES A CONFIRM ON THE USER'S PLATE**, decided
        # here — above `plan_state_spec`, which counts the open notes this batch
        # leaves behind, and BELOW the stamp, which is what makes *"already
        # approved"* true for it. See the function for the three rules.
        #
        # It ran ABOVE the stamp while the owner's own batch returned early out
        # of it, and that order is now the wrong one: this batch may be the
        # acceptance itself — the `dismiss` arm in step 3 closed the confirm note
        # and set the flag above — and a rule that re-files a confirm whenever an
        # unaccepted plan has nothing open for the user would, read one statement
        # earlier, file a fresh gate note in the very write where the user said
        # yes. The stamp is what tells it not to.
        _sync_go_note_locked(project, plan, by=by, addressed=by_addressee)

        # **THE SPEC LOCK'S START LINE, AND IT IS STAMPED HERE FOR THE ARGUMENT
        # DIRECTLY ABOVE.** This is the user's own review batch, so it is the
        # act that engages :func:`_spec_is_locked` on an unaccepted plan. The
        # round that records it is not opened until step 5, BELOW the append —
        # so a lock derived only from `rounds` would be invisible to the very
        # coordinator turn these messages wake, and that turn is the first one
        # that must not rewrite the document. Same ordering, same reason, same
        # bug class as the acceptance marker and the note count.
        _stamp_user_review_locked(plan, by=by)

        # **AND WHAT THE OWNER HAS NOW SEEN.** Beside the stamp above because
        # both record something about a round that is closing, but on a
        # different clock: that one latches once and never moves, this one moves
        # every round. Below the render for the reason its docstring gives — the
        # messages just built report the span FROM the old value.
        _stamp_owner_seen_locked(plan, by=by, addressees=by_addressee)

        # **THE OFFER, LATCHED ABOVE THE PROJECTION IT HAS TO REACH.** This
        # batch is the plan going in front of somebody; the round that records
        # it is not opened until step 5, below the append. See
        # `_stamp_offered_locked`.
        _stamp_offered_locked(plan)
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
        round_.own_inbox = sorted(own_inbox)
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


def _seed_go_note_locked(
    project: "Project", plan: ProjectPlan, *, comment: str = GO_NOTE_COMMENT
) -> str:
    """File the confirm note. Returns its id, or ``""`` when one is already open.

    **An ordinary comment, by the coordinator, to the user.** It is filed by
    hand rather than through :func:`add_note` for one reason — ``bootstrap`` is
    not a parameter of that door and must not become one — and it deliberately
    reproduces nothing else: it is a plain note with a comment, addressed to the
    owner, anchored to the document rather than to a section, so the row the
    user sees is shaped exactly like every other row in the list.

    **It takes no ``body`` and captures no base, which is the change.** It used
    to be a proposal against ``## Approval``: a base capture, a derived quote,
    and a one-line diff the user travelled to a card to read. The section is
    gone (:data:`SEED_TEMPLATE`) and so is the trip.

    Idempotent on the open confirm note, so a retried project creation files one
    note, not two.

    **Regular projects only, and this is the same guard the gate carries.** On a
    front-desk project there is nobody in the accepting role — that is what
    ``surface == "regular"`` means in ``_plan_execution_blocked``. A confirm note
    on such a project would be addressed to a ``user`` who is not reading it:
    never closeable, open forever, and counted in ``plan_open_notes`` on the desk
    card of a project that is not waiting on anybody. Filing no note is the
    honest answer, and it keeps *"which shapes have an acceptance step"* to one
    predicate rather than two that have to agree.

    **``comment`` is a parameter and the default is the confirm wording**, so
    the two callers read as what they are. :func:`seed_go_note` — the one at
    ``POST /projects``, before the coordinator has drafted anything — passes
    :data:`GO_NOTE_DRAFTING`; :func:`_hold_gate_locked` takes the default,
    because it only ever RESTORES a gate on a plan that already had one, which
    by construction is a plan the coordinator has already offered. It is
    interpolated rather than concatenated so a wording may name the keeper
    without this function growing a second formatting rule.
    """
    if project.surface != "regular":
        return ""
    if open_go_note(plan) is not None:
        return ""
    keeper_name = keeper(project)
    ids = _add_notes_locked(plan, [
        PlanNote(
            id="",
            to=OWNER,
            by=keeper_name,
            at=_now(),
            comment=comment.format(keeper=keeper_name),
            revision=plan.revision,
            bootstrap=True,
        )
    ])
    _note(plan, keeper_name, "note", detail=",".join(ids))
    return ids[0]


def _hold_gate_locked(project: "Project", plan: ProjectPlan) -> None:
    """**AN UNACCEPTED REGULAR PLAN ALWAYS HAS AT LEAST ONE OPEN NOTE FOR THE
    USER.** The execution gate's whole content, asserted as an invariant rather
    than reconstructed by each door that could break it. Lock held; mutates
    ``plan`` only.

    ``_plan_execution_blocked`` is ``open_notes_for_you`` and nothing else —
    there is no *"has the user accepted"* conjunct anywhere in it — so *"the
    count never reaches zero before acceptance"* is not a nice property of the
    note flow, it IS the gate. :func:`seed_go_note` establishes the invariant in
    the same request that seeds ``PLAN.md``; this restores it.

    **THE INVARIANT HAS THREE DOORS, AND IT USED TO BE ASKED AT ONE.**
    :func:`_sync_go_note_locked` ran only on a ``submit_review``, and only for a
    keeper's send, so every other way of closing the user's last note left the
    gate down on a plan nobody had confirmed:

    * ``submit_review`` **by the owner** — the user answers the question round
      that retired the confirm note. Their reply closes the parent (AC-5.9),
      nothing is addressed back to them, and the count goes to zero *on the very
      event that wakes the coordinator*. That turn could open workrooms.
    * :func:`resolve_note` — ``plan note --dismiss``/``--reject`` on that last
      question. Never called the sync at all.
    * :func:`add_note` — the owner answering with ``--reply-to``, which closes
      the parent through :func:`_reply_closes_parent`. Same hole, third door.

    All three end with an unaccepted plan and an open gate, and the coordinator's
    prompt line *"Plan confirmed by the user: NOT YET"* was the only thing left
    standing in the way — a convention, which is precisely what
    :func:`seed_go_note` argues a gate must not be. Worse, the re-file was
    reachable only from a later keeper review round, so a coordinator that
    answered in chat and never sent another round left the gate down for good.

    **Asked as a count, not as a round shape.** *"Did this round carry
    questions"* is a fact about one door's arguments; *"is anything open for the
    user"* is a fact about the plan, which is what the gate reads. The count
    includes the confirm note itself (``to == OWNER``, spec layer), so this is
    idempotent on a plan that already has one open and it never files a second.

    It also covers a case the round-shape rule got wrong on its own terms: a
    round carrying only DETAIL-layer questions retired the confirm note, and
    :func:`open_notes_for_you` does not count detail-layer notes — so the gate
    dropped to zero inside a single write, with the retirement reason claiming
    the user's answers were holding it.

    **Acceptance is one-way and stops this**, or every close after it would
    re-gate a project mid-execution. Front-desk projects have nobody in the
    accepting role; :func:`_seed_go_note_locked` carries that argument.

    **IT RESTORES A GATE AND NEVER INTRODUCES ONE**, which is the whole of the
    ``any(n.bootstrap ...)`` guard and it is not defensive. :func:`seed_go_note`
    establishes the invariant at project creation and is the ONLY thing entitled
    to decide a plan has an acceptance step — a plan with no bootstrap note in
    its history never had one, and filing the first one from a note-close would
    gate a project on a rule that was not in force when it started. That set is
    not empty: the creation route documents that *"a project whose seed fails has
    no go-note, so it is unblocked as well as planless"*, and every project that
    predates the seed is in it too.

    With the guard, *"a closed bootstrap note on an unaccepted plan"* has exactly
    one cause — :func:`_sync_go_note_locked` retired it, which is the one path in
    the module that closes it without stamping acceptance
    (:func:`_refuse_go_note_close` permits no other close, and ``dismiss`` by the
    owner stamps). So this fires precisely when a retirement is outstanding, and
    that is the obligation the retirement incurs.
    """
    if project.surface != "regular" or plan.accepted_at:
        return
    if not any(n.bootstrap for n in plan.notes):
        return
    if open_notes_for_you(plan) > 0:
        return
    # **NOT THROUGH `_validate_notes_locked`, like the creation seed it
    # duplicates.** This is a server-filed note on a plan with nothing open for
    # the user, so it lands at its quietest; refusing the caller's whole write
    # over the open-note cap would refuse it for adding the one row that says
    # the plan is waiting on a confirm.
    _seed_go_note_locked(project, plan)


def _sync_go_note_locked(
    project: "Project",
    plan: ProjectPlan,
    *,
    by: str,
    addressed: dict[str, list[PlanNote]],
) -> None:
    """**Does this round leave a confirm note on the user's plate, and what does
    it say?** Four rules, in one place, asked once per review send. Lock held;
    mutates ``plan`` only.

    * **The round carries at least one question for the user** → the open
      confirm note is retired in the same write. The user's answers ARE the
      gate, the open-note count never dips to zero, and they never read *"Nothing in
      this plan is waiting on an answer from you"* directly above four things
      that are open.
    * **The round carries none, it is the KEEPER's, and the row still holds the
      drafting wording** → it is swapped for :data:`GO_NOTE_COMMENT`. This is
      the same sentence as the rule above, asked of the other wording: a row
      seeded at ``POST /projects`` says the coordinator is still drafting
      (:data:`GO_NOTE_DRAFTING`), and leaving it there after the plan has been
      handed over would state the mirror-image falsehood — *"nothing to do
      yet"* on a plan whose only remaining act is the user's. The offer is the
      send, so the send is where the bytes change.
    * **The round carries none, and the plan is not approved** → the note is
      filed, or left alone if it is already open. This is the quiet plan, and it
      is also the case a once-per-project note misses entirely: round 1 asks
      four questions, the user answers, the coordinator rewrites half the
      document, and round 2 has nothing left to ask. Under a single seeded note
      that round reaches the user silently and with the gate already released.
      Under this rule it comes back for a confirm.
    * **The plan is already approved** → never filed again. Acceptance is
      one-way, and re-filing here would re-block a project mid-execution on
      every quiet round the coordinator sent.

    **THE CREATION SEED STAYS, AND IT IS WHAT MAKES THIS SAFE.** The execution
    gate is *"a regular project with at least one open note addressed to the
    user"* and nothing else — there is no *"has the user seen the plan"*
    condition anywhere. So a confirm note born only when the coordinator sends a
    review is a gate the coordinator can skip: draft a plan, send no review,
    zero open notes, free to open workrooms on turn one. :func:`seed_go_note`
    files it in the same request that seeds ``PLAN.md``, before the coordinator
    has had a turn, and this function only ever moves it afterwards.

    **``by == OWNER`` USED TO RETURN EARLY HERE, and that was the gate's hole.**
    The reasoning was that the question is about what the COORDINATOR is handing
    over, and the retire half is indeed the coordinator's alone — the owner's
    replies are addressed to the keeper, so ``addressed`` holds nothing for them
    and the retirement cannot fire on their batch anyway. But the FILE half is
    not about who sent; it is about what the plan looks like afterwards, and the
    owner's batch is exactly the one that empties their own plate: they answer
    the question round that retired the confirm, their reply closes the parent,
    and the count goes to zero on an unaccepted plan. The early return is gone
    and :func:`_hold_gate_locked` — which is reached from this door and from the
    two others that can close the last note — decides the file half. A confirm
    the owner has deliberately left open is still left alone, by the count
    rather than by the sender: an open one is one open note for them.

    **CALLED BELOW ``submit_review``'S ACCEPTANCE STAMP, and it has to be.** The
    call sat above it, which was harmless only while the owner returned early:
    the acceptance batch dismisses the confirm note in step 3 and stamps
    ``accepted_at`` after, so an unconditional file half above the stamp reads an
    unaccepted plan with nothing open and re-files a confirm note during the very
    write in which the user said yes. It is still above ``plan_state_spec``,
    which is the ordering the projection argument needs.

    ``addressed`` is ``submit_review``'s ``by_addressee`` **after** the send
    ledger and the unresolved sweep, so it is what this round actually puts in
    front of the user rather than what it was asked to. A re-send that the
    ledger filtered to nothing is a quiet round, correctly.

    The confirm note itself is excluded from the count, or a round that carried
    only the confirm would read as a round of questions and retire it.
    """
    if project.surface != "regular" or plan.accepted_at:
        return
    questions = [n for n in addressed.get(OWNER, []) if not n.bootstrap]
    go = open_go_note(plan)
    if questions and go is not None:
        _close_note(
            go, status="dismissed", by=keeper(project), reason=GO_NOTE_RETIRED
        )
    elif (
        go is not None
        and by == keeper(project)
        and go.comment != GO_NOTE_COMMENT
    ):
        # **THE OFFER, AND IT IS ONE ASSIGNMENT.** The row has been standing
        # since `POST /projects` in the drafting wording (`GO_NOTE_DRAFTING`),
        # because until this send the coordinator had not put the plan in front
        # of anybody: it was drafting, then waiting on the roster. A keeper's
        # quiet round IS that hand-over — nothing is open, and the only thing
        # left for the user to do is confirm — so the row now says so.
        #
        # `by == keeper(project)` because the owner's own batch is not the
        # coordinator offering the plan. It is reachable here: an owner who
        # replies to the drafting note sends a round addressed to the keeper,
        # which leaves `questions` empty, and flipping on it would claim an
        # offer the coordinator has not made.
        #
        # In the `elif`, not beside it: a round that carries questions retires
        # the row instead, and `_hold_gate_locked` re-files it at the
        # `GO_NOTE_COMMENT` default once those questions close. Both post-round
        # paths therefore land on the confirm wording, and neither can leave a
        # plan that has been reviewed still calling itself a draft.
        go.comment = GO_NOTE_COMMENT
    # **THE FILE HALF IS THE INVARIANT AND IS ASKED AS ONE**
    # (:func:`_hold_gate_locked`), unconditionally, including right after the
    # retirement above: a round whose questions are all DETAIL-layer retires the
    # confirm note and adds nothing `open_notes_for_you` counts, so the arm that
    # used to `return` here dropped the gate to zero inside this write. When the
    # questions do count, the call is a no-op.
    _hold_gate_locked(project, plan)


async def seed_go_note(project: "Project", ctx: "ModelContext") -> str:
    """File the confirm note at project creation. **The gate, and the whole of
    it.**

    Called by ``init_plan_sidecar`` in the same request that seeds ``PLAN.md``,
    and **before** that route publishes ``PROJECT_PLAN_STATE`` — the ordering is
    load-bearing and its argument lives at the call site.

    **Seeded by the SERVER, not asked of the coordinator**, and that is the
    difference between a gate and a convention. Under *"the coordinator files a
    note asking for the go"* nothing blocks until the coordinator chooses to
    file one, so a coordinator that drafts a plan and immediately opens three
    workrooms would be doing something the server used to make impossible. One
    call at creation keeps the guarantee where it was —
    :func:`_sync_go_note_locked` only ever moves the note afterwards, and never
    creates the window this closes.

    It replaces ``approve_plan`` and ``POST /plan/approve`` outright. Acceptance
    is now what happens when the user dismisses this note — one mechanism, the
    same verb every other note offers, on a surface they are already reading.

    **It files :data:`GO_NOTE_DRAFTING`, not the confirm wording, and that is
    the only thing about this function that is not the gate.** At this instant
    the coordinator has not taken a turn and the document is the bare
    :data:`SEED_TEMPLATE`, so *"Nothing in this plan is waiting on an
    answer from you"* would be false for as long as the spec-stage contract takes — a
    drafting turn plus the roster consult it ends on. The row, its addressee,
    its layer and therefore ``_plan_execution_blocked`` are identical either
    way; :func:`_sync_go_note_locked` swaps the bytes for
    :data:`GO_NOTE_COMMENT` when the coordinator actually puts the plan in
    front of the user.
    """
    async with _lock:
        plan = _load(project, ctx)
        nid = _seed_go_note_locked(project, plan, comment=GO_NOTE_DRAFTING)
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
        # Monotone for the same reason and by the same `or`: the latch never
        # clears, so this only ever goes None -> set and the prompt can never be
        # told the lock lifted.
        plan.first_offered_at
        or (
            project.plan_offered_at.isoformat()
            if project.plan_offered_at
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
        project.plan_offered_at.isoformat() if project.plan_offered_at else None,
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
            plan_offered_at=incoming[6],
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

    **THE LEGACY ARM IS THE WHOLE MIGRATION, AND THERE IS NO OTHER HALF.**
    Narrowing what :func:`spec_digest` hashes necessarily changed the number it
    returns, and every plan already accepted stores an
    ``accepted_spec_digest`` computed under the old formula. Left alone, every
    accepted plan in the system would announce *"the plan changed since
    approval"* on its next load, having changed nothing — and a warning that
    fires on every project at once is a warning nobody reads again.

    Fixed at **read time**, not by a backfill: a plan whose stored stamp matches
    the old formula over the current body has not moved, and says so. No
    migration script, no stored-state rewrite, nothing to run. The next
    acceptance re-stamps with the new formula (``submit_review``'s
    ``plan.accepted_spec_digest = spec_digest(body)``) and the arm goes
    vestigial for that plan.

    **It cannot hide a real change.** It only ever makes the answer *more*
    forgiving, and only when nothing spec-relevant moved under the old rules
    either — the old formula hashed a superset of the new one's material, so
    matching it means every section the new formula reads is also untouched.

    The lock path needs no equivalent and gets none: :func:`_prepare_locked`
    compares ``spec_digest(cleaned)`` against ``spec_digest(body)``, both sides
    new formula, both computed in the same call. Same for ``approve
    --expect-spec``, which is a handshake inside one session.

    **:func:`_plan_was_offered` guards the whole thing, for the same reason
    :func:`_spec_is_locked` carries it.** *"The spec changed since the user
    accepted it"* presumes the user has a mental model of what they accepted,
    and on a plan dismissed out of the drafting row they have none: the document
    was never put in front of them, and the keeper is still writing it because
    that same fact leaves the lock open. Without this guard the coordinator is
    told every turn that the spec has drifted — from a baseline the user never
    saw, by writes the server just told it were fine. That is the third copy of
    one falsehood; all three contract predicates now agree on what a contract
    is, and they agree by naming the same two facts.
    """
    if not plan.accepted_at or not _plan_was_offered(plan):
        return False
    return (
        spec_digest(body) != plan.accepted_spec_digest
        and legacy_spec_digest(body) != plan.accepted_spec_digest
    )
