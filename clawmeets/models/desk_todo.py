# SPDX-License-Identifier: MIT
"""
clawmeets/models/desk_todo.py

Desk to-do store — the My Desk right-rail "plate": tasks that still sit
with the manager. Every field on a to-do is one the owner can edit in the
web UI: the text, labels, a draft prompt and its recipient, and stored
attachments. Items arrive two ways — the owner captures one in the rail, or
an agent publishes one (``clawmeets todo publish``) carrying a suggested
prompt and recipient, written into the same ``draft_*`` fields the owner's
recipient picker writes. Nothing an agent publishes is shown in a place the
owner cannot edit it.

Clicking a to-do opens the Task take-over (a guided dispatch surface); the
plate is an *ordered* list the manager drags to reorder, so — unlike the
one-file-per-artifact brief-tab registry — every user's plate is a single
ordered JSON document. Reorder is then a plain array rewrite under one
lock, and agent-publish is a prepend.

A brand-new owner does not open an empty plate: ``SEED`` below is the
onboarding item every account starts with, materialized in memory on the first
read and written to disk by the first mutation. It is never re-asserted
afterwards — see the comment on the constant.

Storage::

    {data_dir}/desk-todos/
      <owner_user_id>.json     # ordered list[DeskTodo], newest capture first
                               # ABSENT => the SEED, not an empty plate
    {data_dir}/desk-seed-state/
      <owner_user_id>.json     # {"given": [seed id, ...]} — every starter row
                               # this account was EVER given; only ever grows

Mutations are broadcast to the owner via ``DESK_TODO_SYNC`` (see
``server/routes/desk_todos.py``) so the desk refetches ``GET /me/desk/todos``.
"""
from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
import secrets
import shutil
import unicodedata
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from clawmeets.models.desk_sop import CONVENE_PROCEDURE
from clawmeets.models.provider_agents import MODEL_AGENT_DEFINITION, provider_agent_names
from clawmeets.utils.file_io import FileUtil

logger = logging.getLogger("clawmeets.models.desk_todo")

_lock = asyncio.Lock()

TODOS_DIR = "desk-todos"

# The seeded-items record: which starter rows this account has EVER been given,
# kept apart from the plate so that deleting a row cannot erase the fact that
# it was given. It is what lets ``backfill_starter_todos`` add a NEW starter row
# to old plates without ever re-adding one the owner already threw away.
# Written by the first plate save and by the backfill; nothing removes an entry.
SEED_STATE_DIR = "desk-seed-state"

# Real attachment bytes, one directory per to-do::
#
#     {data_dir}/desk-todo-files/<owner_user_id>/<todo_id>/<attachment_id>
#
# Deliberately NOT the message-attachment layout in ``server/routes/files.py``.
# That one keys on ``(project_id, chatroom_name, filename)`` and writes through
# a changelog every project participant's runner replays into its own sandbox —
# so reusing it would fan a user's PRIVATE to-do attachments out to agents
# before the user ever dispatched them. Three further blockers, independent of
# the privacy one: that path has no delete route at all (append-only changelog),
# filename is its identity so two to-dos each holding ``report.pdf`` collide,
# and 20 MB of private bytes would live permanently in a replay stream every
# runner reads on join.
#
# This is a sibling of the desk documents already here (``desk-labels/``,
# ``desk-sops/``, ``desk-read-state/``, ``brief-tabs/``), on the
# ``knowledge_pack.py`` precedent: raw bytes on disk, base64 on the wire, no
# changelog. Both path segments and the filename are SERVER-MINTED ids, so path
# traversal is structurally impossible here rather than merely checked for —
# no client-supplied string ever reaches a path segment.
TODO_FILES_DIR = "desk-todo-files"

# The display ``name`` is not a path segment, but it IS echoed back in a
# Content-Disposition header by the download route, so it is bounded and
# scrubbed on the way in.
MAX_ATTACHMENT_NAME_LEN = 200

# Labels. Deliberately looser than the rail's chip-overflow threshold — the
# backend limit should never be the thing that makes a layout decision for the
# UI.
MAX_LABELS_PER_TODO = 8
MAX_SLUG_LEN = 32

# ---------------------------------------------------------------------------
# The starter plate
# ---------------------------------------------------------------------------
#
# What a brand-new owner finds on the plate. Ordinary to-dos — editable,
# archivable, deletable, no ``locked`` field and no guard anywhere in this
# module protecting one. A helpful start, not a fixture.
#
# THE SEED IS A FIRST-WRITE MATERIALIZATION AND IS NEVER RE-ASSERTED. It decides
# only what an owner who has never written a plate sees; the moment anything is
# written — a capture, an archive toggle, a label edit, an agent publish — the
# file is authoritative forever after. An implementation that re-adds missing
# rows on read is wrong, and it is wrong in the way that silently resurrects the
# onboarding item every time the owner ticks it off. Mirrors
# ``models/desk_sop.py`` and ``models/desk_label.py``, which carry the long-form
# version of this argument; all three make the same promise and must keep
# making it.
#
# The decision is made on ``path.exists()`` and on nothing else, because
# ``FileUtil.read`` answers the same ``None`` for a missing file and for a
# JSONDecodeError: deciding "seed?" on the parsed value would re-seed a CORRUPT
# plate, which is the one case where the owner's rows are still on disk and most
# need not to be written over.
#
# ``by_agent_id`` is null: no agent published this. ``drafted`` IS true,
# because the row genuinely arrives carrying a ready prompt — that is the field
# the plate reads to show a preview of it and the toast reads to say "edit it,
# then command" rather than telling the owner to write a draft that is already
# there.
#
# The recipient pair is left null on purpose. A null recipient already resolves
# to ``{username}-assistant`` (``utils/todoDraft.ts``), which is exactly who
# this item is addressed to — and naming it here instead would bake one user's
# assistant name into a constant. It also has to be null for a further reason
# specific to this item: at account creation the assistant agent does not exist
# yet (it is registered later, from the user's own machine), so there is no id
# to store.
SEED_TIMESTAMP = "1970-01-01T00:00:00+00:00"

# The step 2 template lives between two marker lines so tests (and anyone
# editing it) can address it without parsing prose. It is shown to the owner
# filled in as its own to-do on their plate, and THEY fire it — the assistant
# never runs it on its own.
STEP2_TEMPLATE_START = "--- STEP 2 PROMPT TEMPLATE ---"
STEP2_TEMPLATE_END = "--- END TEMPLATE ---"

_ONBOARDING_STEP2_TEMPLATE = """\
Staff a team for my priorities. Run it as ONE project that you coordinate,
so registering, onboarding and handing off all happen under one plan.
You staff my to-dos; you never start them. I fire each one myself.

About me: <2-3 lines from USER.md: role, company, industry, how I work>

My priorities, in order (to-do id · title · success · status · next step):
1. <id> · <title> · <success> · <status> · <next step>
2. ...

Max new agents: 3.
(Edit the list or this setting before firing this to-do if you like.)

1. Design the team. Reuse agents I already have whenever one fits; propose
   a new one only when nothing on my roster can do the job well, and no more
   than the max above. Priorities beyond that stay with you; say why. Each
   new agent gets a name, an industry matched to mine, expertise specific to
   the priorities it owns ("B2B SaaS pricing analyst", not "Researcher"),
   and mentors: you alone by default, since you now hold what every model
   agent told you about me. Add one of my existing agents only when
   the role genuinely overlaps its domain, and give each mentor a distinct
   part of the brain dump. One agent can own several priorities. Prefer
   fewer, sharper agents.

2. Create the project (your create-project skill) with --agent-pool owned,
   one --agent per EXISTING agent you're reusing, and --spawned-from <the id
   of this conversation>. The new agents don't exist yet; you add them to
   the roster as you register them. Its plan must contain:
   - A team table: agent | new or existing | mentors | to-dos it owns (id +
     title) | the agent's one-line job.
   - Milestones:
     Register: register each approved new agent, with a role description
       built from its industry and expertise. Make sure it is running, and
       add it to this project (clawmeets project allowlist <project> --agent
       <name>). Register the top-priority agent first.
     Onboard <agent>: one per new agent, each in its own workroom, in
       priority order, run in parallel:
       a. Brain dump. Each mentor writes its part: everything proprietary,
          hard-won or non-obvious that touches this agent's industry and
          expertise. That covers my profile above, the priorities it will own,
          our conventions and stack, decisions already made and why, what we
          tried that failed, the people and systems involved, and anything a
          competent outsider would get wrong about how we work.
       b. Deep research. The new agent researches its industry and expertise
          to practitioner depth: the state of the art, the standard tools and
          their trade-offs, common failure modes, and where the field is
          heading. It then reconciles that against the brain dump and flags
          every place our practice contradicts it.
       c. Memorize. The new agent reflects and commits the brain dump and
          its research to memory. Done when it posts a short inventory of
          what it now knows, plus every conflict or gap it flagged.
     Hand off: for each priority, update its to-do in place (never publish
       a duplicate):
         clawmeets todo update <id> --recipient <agent> \\
           --draft-prompt "<first assignment>"
       The first assignment states the goal, the current status, and the
       very next deliverable and what makes it good enough, written so I
       could fire it without editing. If the to-do already has
       a draft, keep it and put your first assignment above it. Priorities
       that stay with you get a draft addressed to you. Skip any to-do that's
       no longer open. Hand off each to-do as soon as ITS agent finishes
       onboarding; don't wait for the others. Do not fire any to-do.
   - Acceptance criteria I can check: every priority on my desk has an agent
     and a first assignment I could fire without editing; every new agent
     can say what it knows about me and its domain, and what it flagged.
   - Not Authorized: register no agent beyond what the accepted plan names.
     Fire no to-do, ever. I start each one myself.

3. Wait for me to accept the plan. Register nothing before that. If I edit
   the team, update the plan to match.

4. Run the milestones. When they're done, complete the project with a
   report: the team (name, one-line job, what it now knows, what it
   flagged), and my staffed to-dos in priority order (title, agent, the
   first deliverable it will produce), each ready for me to fire. Don't
   archive this to-do; it follows the project.
"""

ONBOARDING_SEED_ID = "t-seed-onboarding-know-me"
THINKING_PARTNER_SEED_ID = "t-seed-thinking-partner"
CONVENE_SEED_ID = "t-seed-convene-models"

# The second starter row's prompt, inline and word for word as the product plan
# wrote it. Deliberately NOT a pointer to the "Register new agent" SOP in
# ``models/desk_sop.py``: that SOP is the owner's to edit or delete, and this
# to-do has to keep working after they do. ``thought_partner`` is load-bearing —
# the thinking template's starter messages are tied to that exact agent name.
THINKING_PARTNER_PROMPT = (
    'Register a new agent for me and bring it to full working standard before it takes on any real work.\n'
    '\n'
    '  Name:      thought_partner  (keep this exact name; its starter messages are tied to it)\n'
    '  Industry:  decision-making and strategic thinking for my work and life\n'
    '  Expertise: consultant-style interviewing that finds the real question behind a request; structured brainstorming (analogies, constraint flips, assumption reversal, pre-mortems, perspective shifts) followed by narrowing to 2–3 options with cheap tests\n'
    '  Mentors:   my assistant (already briefed by my model agents)\n'
    '\n'
    # Knowledge flows model agents -> assistant -> everyone else, and the
    # owner can fire this row before onboarding. Without this line
    # thought_partner would inherit a brain dump the model agents never fed.
    "If my model agents haven't briefed you yet (step 1 of my onboarding "
    "to-do), do that briefing first, then come back to this.\n"
    '\n'
    'Run these five steps in order and do not skip one:\n'
    '\n'
    '1. Register the agent, with a role description built from the industry and expertise above, and install the `grill-me` and `broaden-options` skills on it.\n'
    "2. Brain dump. My assistant writes down what a thinking partner needs to know about me: decisions I've made and why, bets that held up or didn't, assumptions I keep making, and the questions I tend to avoid. Hand it to the new agent.\n"
    '3. Deep research. Have the new agent research interviewing and brainstorming practice to practitioner depth (how good consultants and coaches question people, which ideation methods work and when each fails), then reconcile that against the brain dump.\n'
    '4. Memorize. Have the new agent reflect and commit both to memory.\n'
    "5. Confirm it's in my sidebar.\n"
    '\n'
    'Report back with what it now knows about how I decide, and every gap it flagged.'
)

# The third starter row's prompt: a model-agent panel, the demo of what several
# models together give that one model alone does not. Inline, like the row
# above, rather than a pointer to the "SYSTEM:Convene panel" SOP the owner may
# delete; the moderation steps are that SOP's own ``CONVENE_PROCEDURE``, so the
# two cannot drift. Capped at 2 rounds so the demo stays cheap (about 9 agent
# turns with three model agents).
CONVENE_PROMPT = (
    f"{MODEL_AGENT_DEFINITION}\n"
    "\n"
    "Show me what my model agents can do together. From my USER.md and my "
    "plate, propose 3 judgment questions I really face (not lookups), one "
    "sentence each. I pick one or write my own.\n"
    "\n"
    "Then convene a panel on it:\n"
    "  Panel:      every model agent on my roster\n"
    "  Max rounds: 2\n"
    "  Consensus:  majority\n"
    # The assistant moderates and never votes (CONVENE_PROCEDURE), so the
    # fallback panel is made of other agents, never the assistant itself.
    "If I have fewer than 2 model agents, tell me, and offer the model agent I "
    "do have plus thought_partner (or another of my agents) as the panel "
    "instead, saying plainly that they may run on the same model, so the "
    "panel is less diverse.\n"
    "Don't create the project until I've picked the question.\n"
    "\n"
    f"{CONVENE_PROCEDURE}\n"
    "\n"
    "Also report the one claim a single model alone would have got wrong, and "
    "which model agent caught it."
)

SEED: tuple[dict[str, object], ...] = (
    {
        "id": ONBOARDING_SEED_ID,
        # Phrased for the OWNER reading their own plate: the row names the
        # thing that needs doing, not the party doing it. Clicking it loads
        # ``draft_prompt`` into the composer already addressed to
        # ``{username}-assistant``, which is where the second person belongs.
        "text": "Get to know me & staff my priorities",
        "drafted": True,
        # Self-contained rather than a pointer to an SOP: step 2 is a template
        # the assistant fills from what step 1 produced, so the two have to
        # travel together. Step 1 is the one place a new account's USER.md gets
        # written; there is no separate personalize item any more.
        #
        # PRODUCT RULE, pinned in tests/test_desk_todo_seed.py: onboarding
        # staffs to-dos and never starts them. Nothing here may tell the
        # assistant to run ``todo trigger`` — the owner's own first click on a
        # staffed to-do is the activation event.
        "draft_prompt": (
            "Help me get set up. There are two steps. Do step 1 now. For step "
            "2 you only prepare a to-do; it starts when I fire it.\n"
            "\n"
            "Step 1. Get to know me.\n"
            # The model agents (models/provider_agents.py) are plain agents in
            # code; mentoring the assistant is written down here and nowhere
            # else. Knowledge flows one way: model agents -> assistant ->
            # thought_partner and every agent "Staff my priorities" creates.
            f"{MODEL_AGENT_DEFINITION}\n"
            "First, have my model agents mentor you. Their names come from "
            "this fixed list, and only the ones on my roster exist: "
            f"{', '.join(provider_agent_names())}. Send each one on my roster "
            "the same request with your direct-message skill: \"Brief my "
            "assistant on me as a mentor would. Cover what your model has "
            "seen of my work on this computer: instruction files and saved "
            "memories; the projects and repos I've worked in; my conventions "
            "and stack; decisions I made and why; what I tried that failed; "
            "and the topics I keep coming back to. Summaries only; leave out "
            "secrets.\" While they answer, start the interview below. Before "
            "your next question, read their replies (`clawmeets dm history "
            "<agent>`), use them to pre-fill USER.md, and skip any question "
            "they already answered. Commit everything else they told you to "
            "your own memory, noting which model agent said what. Then show "
            "me a short inventory of what you learned, which facts came from "
            "which agent, and every place two model agents disagree, so I can "
            "correct them.\n"
            "Write my USER.md with your personalize skill: ask for the easiest "
            "input first (a resume, a pasted bio, a link), then ask only what "
            "is still missing and important. Keep it quick: never ask for due "
            "dates, and don't ask what success looks like; propose it and let "
            "me correct it. If USER.md already exists, read it and fill only "
            "the gaps. Make sure it covers my role, company, industry, how I "
            "like to work, and my top priorities, ranked.\n"
            "Then put each priority on my plate. Run `clawmeets todo list "
            "--no-archived` first and reuse a to-do that already covers it; "
            "publish one only when none does, with its success outcome, "
            "current status and very next step in the draft (`clawmeets todo "
            "publish --text \"<title>\" --draft-prompt \"Success: ... "
            "Status: ... Next step: ...\"`).\n"
            "Show me what you wrote and the priority list, and let me correct "
            "both before you go on.\n"
            "\n"
            "Step 2. Staff my priorities.\n"
            "Once I confirm, fill in the About me line and my priorities in the "
            "template below from my USER.md and my plate. Leave the rest as "
            "written, including `<the id of this conversation>`: that is "
            "filled in when the to-do runs. Put the result on my plate as ONE "
            "new to-do titled \"Staff my priorities\", with the filled-in "
            "template as its draft prompt (`clawmeets todo publish --text "
            "\"Staff my priorities\" --draft-prompt \"<filled template>\"`). "
            "If an open to-do with that title is already on my plate, update "
            "its draft instead (`clawmeets todo update <id> --draft-prompt "
            "\"<filled template>\"`). Then tell me step 1 is done and that "
            "\"Staff my priorities\" is waiting on my plate: I can edit it "
            "there and fire it next. Do not fire it yourself.\n"
            "\n"
            f"{STEP2_TEMPLATE_START}\n"
            f"{_ONBOARDING_STEP2_TEMPLATE}"
            f"{STEP2_TEMPLATE_END}"
        ),
    },
    {
        # Right after onboarding, which is what writes the USER.md the brain
        # dump in step 2 of this prompt draws on. Same shape as the row above:
        # a ready prompt, and a null recipient that resolves to the owner's own
        # assistant — the one agent that can register another.
        "id": THINKING_PARTNER_SEED_ID,
        "text": "Add a thinking partner for big decisions",
        "drafted": True,
        "draft_prompt": THINKING_PARTNER_PROMPT,
    },
    {
        # Third: it proposes questions from USER.md and the plate, which
        # onboarding writes. Same shape as the rows above. Like them it never
        # fires anything; it creates the panel project only after the owner
        # picks the question.
        "id": CONVENE_SEED_ID,
        "text": "Get the best answer from all my models combined",
        "drafted": True,
        "draft_prompt": CONVENE_PROMPT,
    },
)

# Distinguishes "key absent from a PATCH body" (leave the stored value
# untouched) from an explicit JSON ``null`` (clear the field). Used only by
# ``patch_todo`` for the three-way ``draft_recipient_*`` merge; the route
# layer detects key presence off the raw request body (see
# ``server/routes/desk_todos.py``) since a non-serializable sentinel can't be
# a FastAPI ``Body`` default.
_UNSET: object = object()


class LabelError(ValueError):
    """A coded label failure. ``str(e)`` is EXACTLY the wire format the label
    data contract §9 froze::

        "<code>: <human sentence>"
        "labels.cap_exceeded: at most 8 labels per to-do (got 9)"

    Subclasses ``ValueError`` on purpose: the three ``except ValueError`` handlers
    already in ``server/routes/desk_todos.py`` map it to a 400 with ``str(e)`` as
    ``detail``, which is byte-for-byte what §9 asks for — so the to-do write paths
    need no new except clause. ``status`` is only read by
    ``server/routes/desk_labels.py``, where 404 and 409 are also reachable.

    Every §9 row gets a constructor here, so a code string is written once in the
    codebase and cannot be mistyped at one call site and right at another. The
    registry-only codes live here too even though only ``desk_label`` raises
    them: one home for the codes beats a second, near-identical error class, and
    ``desk_label`` may import from this module (never the reverse).

    The code is frozen from first release; the sentence after it is free to
    change forever, and nothing may parse it.
    """

    def __init__(self, code: str, sentence: str, status: int = 400) -> None:
        super().__init__(f"{code}: {sentence}")
        self.code = code
        self.sentence = sentence
        self.status = status

    # --- to-do write paths ---------------------------------------------------
    @classmethod
    def cap_exceeded(cls, got: int) -> "LabelError":
        return cls(
            "labels.cap_exceeded",
            f"at most {MAX_LABELS_PER_TODO} labels per to-do (got {got})",
        )

    @classmethod
    def invalid_slug(cls, raw: str) -> "LabelError":
        return cls(
            "labels.invalid_slug",
            f"{raw!r} is not a valid label — use letters, numbers and hyphens",
        )

    @classmethod
    def slug_too_long(cls, slug: str) -> "LabelError":
        return cls(
            "labels.slug_too_long", f"{slug!r} exceeds {MAX_SLUG_LEN} characters"
        )

    @classmethod
    def invalid_type(cls) -> "LabelError":
        return cls("labels.invalid_type", "labels must be a list of strings")

    # --- registry paths (raised by models/desk_label.py) ---------------------
    @classmethod
    def name_too_long(cls, limit: int) -> "LabelError":
        return cls("labels.name_too_long", f"name exceeds {limit} characters")

    @classmethod
    def registry_full(cls, limit: int) -> "LabelError":
        return cls("labels.registry_full", f"at most {limit} labels in your list")

    @classmethod
    def unknown_color(cls, token: str, palette: tuple[str, ...]) -> "LabelError":
        return cls(
            "labels.unknown_color",
            f"{token!r} is not a known palette token — use one of: "
            + ", ".join(palette),
        )

    @classmethod
    def invalid_kind(cls) -> "LabelError":
        # THE CODE STRING IS FROZEN. It is published in the to-do data contract
        # and the browser branches on it at ``utils/labelErrors.ts`` to write
        # its own copy; changing it here is a silent wire break. Only the
        # SENTENCE moves, and it moves because reciting a one-member enum
        # ("kind must be 'context'") answers a question nobody asked. What the
        # caller actually did was ask for a state, and what they need to hear is
        # that states are not a label any more — the lifecycle they want is
        # derived on ``state`` and cannot be set by hand.
        return cls(
            "labels.invalid_kind",
            "every label is a context now — the 'state' kind is retired, and "
            "whether the work is running is derived on the to-do's `state`",
        )

    @classmethod
    def kind_immutable(cls, slug: str, stored: str, wanted: str) -> "LabelError":
        return cls(
            "labels.kind_immutable",
            f"{slug!r} is a {stored} and cannot become a {wanted} — "
            f"delete it and add it again",
        )

    @classmethod
    def kind_conflict(cls, slug: str, other_kind: str, wanted: str) -> "LabelError":
        """One symmetric condition, one code, either direction — the sentence is
        generated from the OTHER row's kind."""
        return cls(
            "labels.kind_conflict",
            f"{slug!r} is a {other_kind}, not a {wanted}",
            status=409,
        )

    @classmethod
    def kind_mismatch(
        cls, slug: str, slug_kind: str, into: str, into_kind: str
    ) -> "LabelError":
        return cls(
            "labels.kind_mismatch",
            f"{slug!r} is a {slug_kind} and {into!r} is a {into_kind} — "
            f"labels can only be merged into the same kind",
        )

    @classmethod
    def unknown_slug(cls, slug: str) -> "LabelError":
        return cls(
            "labels.unknown_slug",
            f"label {slug!r} is not in your label list",
            status=404,
        )

    @classmethod
    def unknown_merge_target(cls, slug: str) -> "LabelError":
        return cls(
            "labels.unknown_merge_target",
            f"label {slug!r} is not in your label list",
            status=404,
        )

    @classmethod
    def duplicate(cls, slug: str) -> "LabelError":
        return cls("labels.duplicate", f"label {slug!r} already exists", status=409)

    @classmethod
    def merge_self(cls, slug: str) -> "LabelError":
        return cls("labels.merge_self", f"cannot merge {slug!r} into itself")

    @classmethod
    def registry_unreadable(cls) -> "LabelError":
        """The registry file exists and cannot be read, so a write that starts
        from "zero rows" would replace rows nobody has seen.

        409, not 400 and not 500: the request is well-formed and the server is
        healthy — it is the CURRENT STATE of the document that makes the write
        unsafe, which is exactly what 409 means. A 400 would blame the caller
        for a file they did not corrupt, and a 500 would invite a retry that is
        guaranteed to fail the same way.

        Only DIRECT registry mutations raise this. ``auto_register`` skips
        instead: refusing to save someone's to-do because a label INDEX is
        unreadable is a worse outcome than the corruption.
        """
        return cls(
            "labels.registry_unreadable",
            "your label list is on disk but cannot be read — it has not been "
            "changed, and it will not be overwritten until it is repaired or "
            "removed",
            status=409,
        )


# Runs of anything outside the stored charset collapse to a single "-". That is
# a REPLACE, not a drop, and it is what makes the contract's own worked example
# true ("Someday/Maybe" -> "someday-maybe", not "somedaymaybe"); it also matches
# the mock's `NORM` character-for-character, which is the other implementation
# of this grammar and the one @frontend ships against.
_NON_SLUG_RUN = re.compile(r"[^a-z0-9-]+")
_DASH_RUN = re.compile(r"-{2,}")
_VALID_SLUG = re.compile(r"[a-z0-9][a-z0-9-]*")


def normalize_label(raw: str) -> str:
    """One label through the contract's normalization, in order: NFKC + strip;
    drop ONE leading '@'; lowercase; collapse every run of non-``[a-z0-9-]`` to
    a single '-'; collapse '-' runs and trim them; then validate.

    Raises ``LabelError('labels.invalid_slug')`` when the result is empty or
    fails ``^[a-z0-9][a-z0-9-]*$``, and ``labels.slug_too_long`` past 32 chars.
    Length is checked AFTER normalization — a 40-char display name that
    normalizes to 20 is fine.

    ``'@Office'`` -> ``office`` · ``'Wait For'`` -> ``wait-for`` ·
    ``'Someday/Maybe'`` -> ``someday-maybe`` · ``'@ '`` -> invalid_slug.
    """
    if not isinstance(raw, str):
        raise LabelError.invalid_type()
    text = unicodedata.normalize("NFKC", raw).strip()
    if text.startswith("@"):
        text = text[1:]
    text = _NON_SLUG_RUN.sub("-", text.lower())
    text = _DASH_RUN.sub("-", text).strip("-")
    if not text or not _VALID_SLUG.fullmatch(text):
        raise LabelError.invalid_slug(raw)
    if len(text) > MAX_SLUG_LEN:
        raise LabelError.slug_too_long(text)
    return text


def normalize_labels(raw: object) -> list[str]:
    """A whole label list, normalized and de-duplicated preserving FIRST-SEEN
    order (chip order is meaningful; nothing reads it yet, but nothing may
    scramble it either).

    ``None`` -> ``[]``. A non-list, or a list holding a non-string, raises
    ``labels.invalid_type`` — which is why every route reads the label keys
    untyped: FastAPI's own 422 carries a LIST as ``detail`` and would break the
    contract's client-side ``^labels\\.[a-z0-9_]+: `` parse rule.

    Does NOT check the cap. The cap is a property of the RESULT of a write, not
    of an input, and checking it here would reject the add-delta that swaps a
    label at 8. See ``merge_label_keys``.
    """
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise LabelError.invalid_type()
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise LabelError.invalid_type()
        slug = normalize_label(item)
        if slug not in out:
            out.append(slug)
    return out


def merge_label_keys(
    current: list[str],
    *,
    replace: object = _UNSET,
    add: object = None,
    remove: object = None,
) -> list[str]:
    """Apply the three label keys in their ruled precedence — replace, then add,
    then remove — and enforce the cap ONCE, on the final set.

    Three behaviours an implementer cannot infer from the endpoint list, and all
    three have a named test:

      * The cap is checked AFTER dedupe, so re-adding a label an 8-label item
        already carries is a no-op, not a 400. The assistant retries commands,
        and a rule that fails only on the second run is the worst kind.
      * The cap is checked AFTER ``remove`` is applied, so
        ``{add_labels:[x], remove_labels:[y]}`` on an 8-label item succeeds.
        Checking between add and remove would refuse a legal swap.
      * ``got`` in the error sentence is the size of the WOULD-BE result, not
        the length of the payload. The sentence describes the to-do, which is
        what the owner is looking at.

    Raises ``labels.cap_exceeded`` and applies nothing — the caller has not saved
    yet, so a rejected write leaves the plate byte-identical.
    """
    result = list(current) if replace is _UNSET else normalize_labels(replace)
    for slug in normalize_labels(add):
        if slug not in result:
            result.append(slug)
    removing = set(normalize_labels(remove))
    if removing:
        result = [s for s in result if s not in removing]
    if len(result) > MAX_LABELS_PER_TODO:
        raise LabelError.cap_exceeded(len(result))
    return result


class AttachmentError(ValueError):
    """An attachment failure that already knows its HTTP status.

    Subclasses ``ValueError`` on purpose, exactly as ``LabelError`` does: it
    rides the ``except ValueError`` handlers already in
    ``server/routes/desk_todos.py`` if a caller forgets the narrower clause,
    degrading to a 400 with the right sentence rather than a 500.

    ``str(e)`` is EXACTLY the wire ``detail``: a plain string, never a list,
    never a code prefix. Unlike ``LabelError`` there is no machine-readable
    code, because nothing parses these — the contract froze the SENTENCES, and
    the client renders them verbatim.

    This module may not raise ``HTTPException``: ``models/`` ships in the runner
    wheel and fastapi is not a runner dependency. Every limit it enforces
    therefore arrives as a parameter rather than being read from the
    environment here.
    """

    def __init__(self, status: int, sentence: str) -> None:
        super().__init__(sentence)
        self.status = status
        self.sentence = sentence


class DeskTodoAttachment(BaseModel):
    """One real, stored file on a to-do. The bytes live on disk at
    ``{data_dir}/desk-todo-files/<owner>/<todo_id>/<id>``; this row is the ONLY
    thing that names them, and a blob the plate does not name is garbage by
    definition.

    Written by the owner, on save, and nobody else — ``clawmeets todo
    publish`` carries no attachments, so an agent can never park bytes on its
    owner's plate.

    ``name`` is the original filename, DISPLAY ONLY. It never becomes a path
    segment: the file on disk is named by ``id``, which is why traversal is
    structurally impossible here rather than merely checked for.
    """

    id: str
    name: str
    size: int
    content_type: str
    sha256: str
    created_at: str


class DeskTodo(BaseModel):
    """A single item on the manager's plate.

    Only fields the owner can edit in the web UI live here (plus ids,
    timestamps and the hidden ``by_agent_id``). Rows written by older servers
    may carry keys that have since been removed — ``origin``, ``source``,
    ``due``, ``by_agent_name``, ``suggest_agent_*``, ``context``, ``files``,
    ``done_steps``, ``available``, ``linked``. They are not declared, so
    pydantic's default ``extra="ignore"`` drops them on load and the next
    ``_save`` persists the slim shape — the same strategy
    ``_archived_read_shim`` describes for ``status``. No migration pass.
    """

    id: str
    owner_user_id: str
    text: str
    # The owner filed it away. A DISPOSAL, never a lifecycle state — nothing
    # project-driven writes it and nothing reads it as "the work finished".
    # A bool rather than a renamed two-value string on purpose: a string is
    # what lets a third value get added later and re-invent lifecycle state,
    # which is the exact confusion this field exists to end.
    archived: bool = False
    created_at: str
    updated_at: str

    # Ids of the projects and DM threads this to-do spawned. ONE id space, so a
    # DM-thread id and a regular project id are indistinguishable here — which
    # is required, because the desk's **command** button spawns a DM thread and
    # that is the feature's most common creation path.
    #
    # A REFERENCE, not a foreign key: an id whose project was deleted stays in
    # the list, is legal, and simply stops contributing to the derived state.
    # This module stores the ids and knows NOTHING about projects — exactly as
    # it stores label slugs and knows nothing about the registry. The
    # derivation lives in the third module, ``models/desk_todo_link.py``, which
    # imports this one and is never imported by it.
    project_ids: list[str] = Field(default_factory=list)

    # Bare normalized slugs — the owner's GTD contexts and states. Presentation
    # (display name, colour, which axis) lives in the SEPARATE registry document
    # (``models/desk_label.py``); this list is NOT a foreign key into it. A slug
    # with no registry row is valid and renders unregistered, which is the whole
    # reason the plate stays readable with the registry gone. Default-empty, so
    # every plate written before labels existed loads unchanged and there is no
    # migration script.
    labels: list[str] = Field(default_factory=list)

    draft_prompt: str | None = None
    # Recipient of the draft — picked in the take-over composer, or resolved
    # from an agent publish's ``--suggest``. Both id and name are persisted so
    # the plate pill still labels correctly after a rename/removal; the
    # frontend re-resolves against the live roster and falls back if the agent
    # is gone. A suggested name that resolved to nobody stores the name with a
    # null id. Null until set.
    draft_recipient_id: str | None = None
    draft_recipient_name: str | None = None
    # Real stored bytes the OWNER parked on this item. Always a list, never
    # null — ``default_factory`` means every plate row written before
    # attachments existed validates unchanged, so there is no migration.
    attachments: list[DeskTodoAttachment] = Field(default_factory=list)

    # True once the item carries a ready draft: set when the manager saves one
    # in the take-over, or when an agent publishes with a draft prompt.
    drafted: bool = False

    # Hidden provenance, never displayed: the agent that published this item,
    # kept ONLY so ``DELETE /me/desk/todos/{id}`` can let the publishing agent
    # retract its own item. Null for anything the owner captured.
    by_agent_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _archived_read_shim(cls, data: Any) -> Any:
        """Read a plate written in the old ``status: "open" | "done"`` shape.

        NOT compat-nostalgia, and not optional. Every plate on disk today
        carries ``status``; without this line every to-do the owner ever filed
        away springs back onto their open plate on first load, which is exactly
        what AC-4.3 forbids::

            legacy `status` present AND `archived` absent -> archived = (status == "done")
            `archived` present                           -> it wins, always
            neither                                      -> False (the field default)

        ``status`` is deliberately NOT redeclared as a field, so pydantic's
        default ``extra="ignore"`` drops it once this validator has read it, and
        the owner's next ORDINARY save — a rename, a label edit, an archive
        toggle — persists the new shape as a side effect of a write they asked
        for. There is no rewrite pass, no bulk write over anyone's plate, and no
        schema-version bump: the plate is one JSON document per owner and
        ``_save`` already rewrites the whole array, so the re-shape costs
        nothing and touches exactly the one owner who wrote.

        The shim is also what makes the change loadable in both directions
        during a rollout: a row already carrying ``archived`` is untouched here.
        """
        if not isinstance(data, dict):
            return data
        if "archived" in data:
            return data
        if "status" in data:
            data = dict(data)
            data["archived"] = data.get("status") == "done"
        return data


def gen_id() -> str:
    return "t-" + secrets.token_hex(6)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _path(data_dir: Path, owner_user_id: str) -> Path:
    return Path(data_dir) / TODOS_DIR / f"{owner_user_id}.json"


def _seed_rows(owner_user_id: str) -> list[DeskTodo]:
    """The starter plate, materialized in memory for one owner.

    The timestamps are the fixed epoch constant rather than the clock. The seed
    is materialized on every read of an unwritten plate, so minting ``now()``
    here would make two consecutive GETs return different ``created_at`` values
    for the same row — churn a client can legitimately notice, in a field
    nothing needs. It is also simply more truthful: the owner did not capture
    this. Rows keep the constant when the first write materializes them to
    disk; anything captured or published afterwards is stamped with the real
    clock and, being prepended, sorts ahead of them.
    """
    return [
        DeskTodo(
            owner_user_id=owner_user_id,
            created_at=SEED_TIMESTAMP,
            updated_at=SEED_TIMESTAMP,
            **row,
        )
        for row in SEED
    ]


def _load(data_dir: Path, owner_user_id: str) -> list[DeskTodo]:
    """Every read and every write starts here, so "missing" vs "empty" vs
    "unreadable" is decided in exactly one place.

    NO FILE -> the SEED, in memory. A file that exists but holds no usable rows
    -> ZERO rows, never the seed as a repair: an owner who cleared their last
    to-do gets an empty plate and keeps it, and a corrupt document is not
    written over with starter content.
    """
    path = _path(data_dir, owner_user_id)
    if not path.exists():
        return _seed_rows(owner_user_id)
    raw = FileUtil.read(path, "json")
    if not isinstance(raw, list):
        return []
    out: list[DeskTodo] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            out.append(DeskTodo.model_validate(row))
        except Exception:
            continue
    return out


def _save(data_dir: Path, owner_user_id: str, todos: list[DeskTodo]) -> None:
    """Every write ends here. The FIRST one — no plate file yet — is the moment
    the in-memory seed becomes real, so it also records every ``SEED`` id as
    given. All of them, not just the ones still in ``todos``: the owner was
    shown the whole seed, and a first write that deletes a starter row must
    still count it as given or the backfill would hand it straight back."""
    path = _path(data_dir, owner_user_id)
    first_save = not path.exists()
    FileUtil.write(path, [t.model_dump() for t in todos], "json")
    if first_save:
        record_given_seed_ids(data_dir, owner_user_id, (row["id"] for row in SEED))


# --- the seeded-items record ------------------------------------------------


def _seed_state_path(data_dir: Path, owner_user_id: str) -> Path:
    return Path(data_dir) / SEED_STATE_DIR / f"{owner_user_id}.json"


def given_seed_ids(data_dir: Path, owner_user_id: str) -> set[str]:
    """Seed ids this account has ever been given. A missing or unreadable
    record reads as the empty set."""
    raw = FileUtil.read(_seed_state_path(data_dir, owner_user_id), "json")
    given = raw.get("given") if isinstance(raw, dict) else None
    if not isinstance(given, list):
        return set()
    return {g for g in given if isinstance(g, str)}


def record_given_seed_ids(
    data_dir: Path, owner_user_id: str, ids: Iterable[str],
) -> None:
    """Union ``ids`` into the record. Never removes an entry, and skips the
    write when nothing is new."""
    have = given_seed_ids(data_dir, owner_user_id)
    merged = have | set(ids)
    if merged == have:
        return
    FileUtil.write(
        _seed_state_path(data_dir, owner_user_id),
        {"given": sorted(merged)},
        "json",
    )


# --- the one-time starter backfill ------------------------------------------


class BackfillReport(BaseModel):
    """Per-account outcome of ``backfill_starter_todos``, by user id.

    ``would_add`` and ``added`` name the same accounts; ``added`` is filled
    only when the run applied. ``skipped_unreadable`` is a plate file that
    exists but is not a JSON list — it is left alone rather than written over,
    and unlike a missing plate the new-account seed will NOT reach it."""

    would_add: list[str] = Field(default_factory=list)
    added: list[str] = Field(default_factory=list)
    skipped_already_given: list[str] = Field(default_factory=list)
    skipped_no_plate: list[str] = Field(default_factory=list)
    skipped_unreadable: list[str] = Field(default_factory=list)

    @property
    def skipped(self) -> int:
        return (
            len(self.skipped_already_given)
            + len(self.skipped_no_plate)
            + len(self.skipped_unreadable)
        )


# Starter rows added after launch, each given once to every existing plate by
# ``backfill_starter_todos``. Each is decided on its own: an owner who deleted
# one still gets the next.
BACKFILL_SEED_IDS: tuple[str, ...] = (THINKING_PARTNER_SEED_ID, CONVENE_SEED_ID)


def _seed_row(seed_id: str, owner_user_id: str) -> dict[str, Any]:
    (row,) = (r for r in SEED if r["id"] == seed_id)
    return DeskTodo(
        owner_user_id=owner_user_id,
        created_at=SEED_TIMESTAMP,
        updated_at=SEED_TIMESTAMP,
        **row,
    ).model_dump()


def backfill_starter_todos(
    data_dir: Path, user_ids: Iterable[str], *, apply: bool,
) -> BackfillReport:
    """Give every existing saved plate each ``BACKFILL_SEED_IDS`` row, once.

    OFFLINE ONLY: plate writes are serialized by the in-process ``_lock``, which
    this cannot share with a running server — the CLI wrapper refuses
    ``apply=True`` while the server is up. Per account:

    - no plate file          -> skipped_no_plate (the new seed covers it)
    - plate not a JSON list  -> skipped_unreadable (never written over)

    and then per starter row, each decided on its own:

    - id in the given record -> left out (deleted stays deleted)
    - id already on the plate-> left out (and recorded, on apply)
    - otherwise              -> inserted right after the nearest earlier
                                ``SEED`` row still on the plate, or at the top
                                if none is; on apply the record gains it and
                                the onboarding id (a pre-feature plate was
                                given onboarding at its first save)

    The account lands in would_add (and added, on apply) if any row went in,
    otherwise in skipped_already_given. The plate is edited as raw JSON rows so
    every other row is written back exactly as it was read, including rows the
    model would reject.
    """
    seed_order = [str(r["id"]) for r in SEED]
    report = BackfillReport()
    for uid in user_ids:
        path = _path(data_dir, uid)
        if not path.exists():
            report.skipped_no_plate.append(uid)
            continue
        raw = FileUtil.read(path, "json")
        if not isinstance(raw, list):
            report.skipped_unreadable.append(uid)
            continue
        given = given_seed_ids(data_dir, uid)
        inserted: list[str] = []
        already_there: list[str] = []
        for seed_id in BACKFILL_SEED_IDS:
            if seed_id in given:
                continue
            ids = [r.get("id") if isinstance(r, dict) else None for r in raw]
            if seed_id in ids:
                already_there.append(seed_id)
                continue
            earlier = seed_order[: seed_order.index(seed_id)]
            at = 0
            for prev in reversed(earlier):
                if prev in ids:
                    at = ids.index(prev) + 1
                    break
            raw.insert(at, _seed_row(seed_id, uid))
            inserted.append(seed_id)
        if apply and already_there:
            record_given_seed_ids(data_dir, uid, already_there)
        if not inserted:
            report.skipped_already_given.append(uid)
            continue
        report.would_add.append(uid)
        if apply:
            FileUtil.write(path, raw, "json")
            record_given_seed_ids(data_dir, uid, [ONBOARDING_SEED_ID, *inserted])
            report.added.append(uid)
    return report


# --- the attachment blob store ----------------------------------------------
# The plate JSON is the ONLY source of truth. A blob it does not name is
# garbage by definition — which is what makes the orphan sweep below one
# ``listdir`` and a set difference, and what makes the ordering rule in
# ``patch_todo`` sufficient rather than merely hopeful.


def gen_attachment_id() -> str:
    """``"a-" + token_hex(6)``. Sibling of ``gen_id``'s ``"t-"`` prefix, so an
    id is self-describing in a log line and a to-do id can never be mistaken
    for an attachment id in a URL."""
    return "a-" + secrets.token_hex(6)


def normalize_attachment_name(raw: str) -> str:
    """The stored display ``name``: strip control characters and path
    separators, drop leading dots, collapse whitespace, bound the length, and
    fall back to ``"file"`` if nothing survives.

    Belt and braces over the route's ``_validate_filename``, which rejects
    traversal outright. This one exists for a different reason: the name is
    echoed back in a ``Content-Disposition`` header by the download route, so a
    name carrying a quote, a CR or a NUL could inject a header parameter. It is
    scrubbed once, on the way in, so every later reader of ``name`` — header,
    UI, log — gets an already-safe string.
    """
    text = raw or ""
    # Path separators first, so "a/b" collapses to two words rather than "ab".
    for sep in ("/", "\\"):
        text = text.replace(sep, " ")
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = re.sub(r"\s+", " ", text).strip().lstrip(".").strip()
    text = text[:MAX_ATTACHMENT_NAME_LEN].strip()
    return text or "file"


def guess_content_type(name: str) -> str:
    """The stored ``content_type``, derived from the NORMALIZED display name.

    Derived server-side rather than trusted from the client — the inbound entry
    is exactly ``{filename, content_b64}`` and carries no type — so a caller
    cannot mislabel bytes into a browser-rendered type. Same
    ``mimetypes.guess_type`` + ``application/octet-stream`` fallback
    ``server/routes/files.py`` already applies on download. The type is
    DESCRIBED, never gated: there is no allowlist on this path, because a to-do
    exists to be dispatched and the message path that receives it has none
    either — a file you can send in a DM must not be one you cannot park.
    """
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def _owner_files_dir(data_dir: Path, owner_user_id: str) -> Path:
    """``{data_dir}/desk-todo-files/<owner_user_id>/`` — the sweep's unit."""
    return Path(data_dir) / TODO_FILES_DIR / owner_user_id


def todo_files_dir(data_dir: Path, owner_user_id: str, todo_id: str) -> Path:
    """``…/<todo_id>/`` — the delete's unit. Both segments are server-minted."""
    return _owner_files_dir(data_dir, owner_user_id) / todo_id


def read_attachment_blob(
    data_dir: Path, owner_user_id: str, todo_id: str, attachment_id: str
) -> bytes | None:
    """The stored bytes, or ``None`` when the file is missing.

    ``None`` is not an error here. A plate row naming vanished bytes is the
    deliberately-surfaced case: the route turns it into a 404 that renders as a
    broken chip, rather than a 500 or a silently-empty download.
    """
    return FileUtil.read(
        todo_files_dir(data_dir, owner_user_id, todo_id) / attachment_id, "bytes"
    )


def write_attachment_blob(
    data_dir: Path, owner_user_id: str, todo_id: str, attachment_id: str, raw: bytes
) -> None:
    """Write one blob, through ``FileUtil.write(..., "bytes")`` so it gets the
    same atomic write + parent mkdir every other document on this desk gets.

    Deliberately a module-level function rather than an inline ``FileUtil``
    call: this is the seam a test patches to prove that a failed blob write
    leaves the plate byte-identical. Monkeypatching ``FileUtil.write`` globally
    would break the plate write too and prove nothing.
    """
    FileUtil.write(
        todo_files_dir(data_dir, owner_user_id, todo_id) / attachment_id, raw, "bytes"
    )


def _remove_dir_if_empty(path: Path) -> None:
    try:
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    except OSError as e:  # pragma: no cover - filesystem edge
        logger.warning(f"Could not remove empty attachment dir {path}: {e}")


def _write_new_blobs(
    data_dir: Path,
    owner_user_id: str,
    todo_id: str,
    minted: list[tuple[str, bytes]],
) -> None:
    """Write every newly-minted blob, BEFORE the plate row that will name them.

    On ANY exception: unlink every blob THIS call already wrote, remove the
    to-do's directory if it is now empty, and re-raise unchanged. Without the
    rollback the plate would still be byte-identical (we never reached
    ``_save``) but a directory would exist on disk. The sweep would eventually
    take it; "eventually" is not the guarantee worth having when the whole
    point of writing bytes first is that a failure leaves nothing behind.
    """
    written: list[Path] = []
    try:
        for attachment_id, raw in minted:
            write_attachment_blob(data_dir, owner_user_id, todo_id, attachment_id, raw)
            written.append(
                todo_files_dir(data_dir, owner_user_id, todo_id) / attachment_id
            )
    except BaseException:
        for path in written:
            FileUtil.delete(path)
        _remove_dir_if_empty(todo_files_dir(data_dir, owner_user_id, todo_id))
        raise


# --- the two-document escape hatch ------------------------------------------
# ``desk_label.delete_label`` / ``merge_label`` mutate BOTH the plate and the
# registry and must hold the plate lock across both writes, so they cannot call
# ``patch_todo`` / ``reorder_todos`` — those take this same non-reentrant lock
# and would deadlock. They get these three instead.
#
# LOCK ORDER, and it is the only rule preventing a deadlock here:
#     plate_lock (OUTER)  ->  desk_label._registry_lock (INNER).  NEVER reversed.
# Nothing in the type system enforces that, which is why it is written on both
# modules.
plate_lock = _lock


def load_plate(data_dir: Path, owner_user_id: str) -> list[DeskTodo]:
    """The plate, unlocked. Callers MUST already hold ``plate_lock``."""
    return _load(data_dir, owner_user_id)


def save_plate(data_dir: Path, owner_user_id: str, todos: list[DeskTodo]) -> None:
    """Write the plate, unlocked. Callers MUST already hold ``plate_lock``."""
    _save(data_dir, owner_user_id, todos)


def sweep_orphan_todo_files(data_dir: Path, owner_user_id: str) -> int:
    """Delete every blob the plate does not name, for one owner. Returns the
    number of paths removed.

    Two levels, because there are two ways to orphan bytes:

      * an entire ``<todo_id>/`` directory whose to-do is gone from the plate,
        and
      * a single ``<attachment_id>`` file inside a LIVE to-do whose descriptor
        was reconciled away — which is also what a crash between the blob write
        and the plate write leaves behind, and is the real orphan risk that
        deleting a to-do never exercises.

    CALLERS MUST ALREADY HOLD ``plate_lock`` — the same contract, for the same
    reason, that ``load_plate`` / ``save_plate`` carry. Both callers
    (``delete_todo``, ``patch_todo``) are already inside it, and taking the
    non-reentrant lock here would deadlock. It reads through ``load_plate``,
    never ``_load``-plus-lock.

    Deliberately NOT scheduled. There is no cron and no background job, because
    a sweep that runs only from a scheduler is a sweep that silently is not
    running in someone's deployment. It can afford to be synchronous precisely
    because the plate is the only source of truth: a blob it does not name is
    garbage by definition, so this is one ``listdir`` of a small per-owner
    directory diffed against a list already in memory.

    Best-effort by contract: a filesystem error is logged, never raised. The
    caller's write has already succeeded, and failing it on a bookkeeping side
    effect would leave the user with an item they cannot remove.
    """
    root = _owner_files_dir(data_dir, owner_user_id)
    if not root.is_dir():
        return 0
    referenced = {
        t.id: {a.id for a in t.attachments}
        for t in load_plate(data_dir, owner_user_id)
    }
    removed = 0
    try:
        entries = list(root.iterdir())
    except OSError as e:  # pragma: no cover - filesystem edge
        logger.warning(f"Orphan sweep could not list {root}: {e}")
        return 0
    for entry in entries:
        try:
            if entry.name not in referenced:
                # The to-do is gone from the plate; everything under it is
                # unreachable by definition.
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink()
                removed += 1
                continue
            if not entry.is_dir():
                continue
            keep = referenced[entry.name]
            for blob in entry.iterdir():
                if blob.name not in keep:
                    if blob.is_dir():
                        shutil.rmtree(blob, ignore_errors=True)
                    else:
                        blob.unlink()
                    removed += 1
            _remove_dir_if_empty(entry)
        except OSError as e:  # pragma: no cover - filesystem edge
            logger.warning(f"Orphan sweep could not remove {entry}: {e}")
    return removed


def list_todos(data_dir: Path, owner_user_id: str) -> list[DeskTodo]:
    """Return the owner's plate in stored (manager-controlled) order."""
    return _load(data_dir, owner_user_id)


def get_todo(data_dir: Path, owner_user_id: str, todo_id: str) -> DeskTodo | None:
    for t in _load(data_dir, owner_user_id):
        if t.id == todo_id:
            return t
    return None


async def add_todo(
    data_dir: Path,
    owner_user_id: str,
    text: str,
    *,
    labels: object = None,
) -> DeskTodo:
    """Capture an owner task; prepended to the plate (newest first).

    ``labels`` is normalized and capped BEFORE the row is built, so a rejected
    capture never reaches the document."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Task text cannot be empty")
    slugs = merge_label_keys([], replace=labels)
    async with _lock:
        todos = _load(data_dir, owner_user_id)
        now = _now()
        todo = DeskTodo(
            id=gen_id(),
            owner_user_id=owner_user_id,
            text=text,
            created_at=now,
            updated_at=now,
            labels=slugs,
        )
        todos.insert(0, todo)
        _save(data_dir, owner_user_id, todos)
        return todo


async def publish_agent_todo(
    data_dir: Path,
    owner_user_id: str,
    *,
    by_agent_id: str,
    text: str,
    draft_prompt: str | None = None,
    draft_recipient_id: str | None = None,
    draft_recipient_name: str | None = None,
    labels: object = None,
) -> DeskTodo:
    """Push an agent-published task onto the owner's plate (prepended).

    The suggested prompt and recipient land in the same ``draft_*`` fields the
    owner's take-over writes, so what the agent suggests is exactly what the
    owner sees and edits. The caller (the route) resolves the recipient.

    ``labels`` is validated identically to ``add_todo``. Note what does NOT
    happen here: no auto-registration. The item is the agent's to write; the
    vocabulary is the owner's to curate."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Task text cannot be empty")
    slugs = merge_label_keys([], replace=labels)
    async with _lock:
        todos = _load(data_dir, owner_user_id)
        now = _now()
        todo = DeskTodo(
            id=gen_id(),
            owner_user_id=owner_user_id,
            text=text,
            created_at=now,
            updated_at=now,
            by_agent_id=by_agent_id,
            draft_prompt=draft_prompt or None,
            draft_recipient_id=draft_recipient_id,
            draft_recipient_name=draft_recipient_name,
            drafted=bool(draft_prompt),
            labels=slugs,
        )
        todos.insert(0, todo)
        _save(data_dir, owner_user_id, todos)
        return todo


def _reconcile_attachments(
    data_dir: Path,
    owner_user_id: str,
    target: DeskTodo,
    desired: list[dict],
    total_limit: int | None,
) -> list[DeskTodoAttachment]:
    """Turn the desired-state array into the to-do's new attachment list.

    Pure planning plus blob WRITES. It does not save the plate and it does not
    delete anything — deletion is the caller's step 3, strictly after the plate
    stops naming the bytes. The caller holds ``plate_lock``.

    ``desired`` is the route's already-decoded form: ``{"keep": id}`` entries
    pass through, ``{"name", "raw", "content_type"}`` entries are new bytes.
    Order is preserved exactly as sent — the tray is an ordered thing on
    screen, and nothing here may scramble it.

    The array is the COMPLETE desired state, which is what makes it
    structurally duplicate-proof: a client that re-sends bytes for an
    already-stored file instead of ``{keep: id}`` gets a new id and the old id
    is absent, so the worst case is churn (new id, same bytes) and never two
    rows for one file.

    Three cases worth stating because they are not inferable from the shape:

      * A ``{keep}`` naming an id that is not on THIS to-do raises
        ``AttachmentError(400, ...)``. Loud, not silent: absence-means-delete
        is only safe because the client holds the complete desired state, so a
        client that has demonstrably lost track of what is stored — a stale
        second tab — is told in the same request rather than having its
        confusion committed. It is also the only option consistent with the
        save being atomic; silently dropping the entry would half-honour it.
      * A duplicate ``{keep}`` of the same id collapses to one row.
      * The per-to-do total is checked HERE, on the post-reconcile result
        (kept + new), not on the input — the same discipline
        ``merge_label_keys`` applies to the label cap, and for the same reason:
        the sentence has to describe the to-do, which is what the owner is
        looking at. It raises having applied nothing, so the plate stays
        byte-identical and no byte reaches disk.
    """
    stored = {a.id: a for a in target.attachments}
    now = _now()

    result: list[DeskTodoAttachment] = []
    seen_keeps: set[str] = set()
    minted: list[tuple[str, bytes]] = []

    for entry in desired:
        keep_id = entry.get("keep")
        if keep_id is not None:
            existing = stored.get(keep_id)
            if existing is None:
                raise AttachmentError(
                    400, f"attachment {keep_id!r} is not on this to-do"
                )
            if keep_id in seen_keeps:
                continue
            seen_keeps.add(keep_id)
            result.append(existing)
            continue

        raw: bytes = entry["raw"]
        attachment_id = gen_attachment_id()
        result.append(
            DeskTodoAttachment(
                id=attachment_id,
                name=entry["name"],
                size=len(raw),
                content_type=entry["content_type"],
                sha256=FileUtil.sha256(raw),
                created_at=now,
            )
        )
        minted.append((attachment_id, raw))

    if total_limit is not None:
        total = sum(a.size for a in result)
        if total > total_limit:
            raise AttachmentError(
                413,
                f"attachments total {total} bytes, exceeds the {total_limit}-byte "
                "per-to-do limit (raise CLAWMEETS_MAX_MESSAGE_ATTACHMENTS_BYTES)",
            )

    # Only now, with every check passed, do bytes reach the disk — so a 413 or
    # a bad {keep} never leaves one behind.
    if minted:
        _write_new_blobs(data_dir, owner_user_id, target.id, minted)
    return result


async def patch_todo(
    data_dir: Path,
    owner_user_id: str,
    todo_id: str,
    *,
    archived: bool | None = None,
    text: str | None = None,
    draft_prompt: str | None = None,
    draft_recipient_id: str | None = _UNSET,  # _UNSET → leave untouched
    draft_recipient_name: str | None = _UNSET,  # None → clear, str → set
    drafted: bool | None = None,
    labels: object = _UNSET,  # _UNSET → leave untouched, None/[] → clear
    add_labels: object = None,
    remove_labels: object = None,
    attachments: object = _UNSET,  # _UNSET → untouched, None/[] → remove all
    total_limit: int | None = None,  # post-reconcile cap, supplied by the route
) -> DeskTodo | None:
    """Patch a task in place. Returns the updated task, or None if missing.

    ``draft_recipient_id`` / ``draft_recipient_name`` are three-way: ``_UNSET``
    (the default, when the caller omits them) leaves the stored value alone,
    an explicit ``None`` clears it, and a string overwrites it. The older
    scalar fields collapse absent+None into "untouched" (a ``None`` never
    clears them).

    ``labels`` is three-way in exactly the same way. ``add_labels`` /
    ``remove_labels`` need no three-way handling — absent and empty both mean
    no-op — and are idempotent in both directions: adding a label the item
    already carries and removing one it does not are 200s, not errors. All three
    are delegated to ``merge_label_keys``, which is where the cap is checked.

    ``attachments`` is three-way in exactly the same way ``labels`` is:
    ``_UNSET`` (the caller omitted the key) leaves the stored list alone,
    ``None`` or ``[]`` removes all, and a list reconciles to exactly that
    desired state. The ABSENT case is the one that matters most — a done
    toggle, a rename or a label edit must never nuke a to-do's files, and every
    one of those goes through this function. ``total_limit`` is the per-to-do
    byte cap; the route supplies it, because this module ships in the runner
    wheel and may not read the server's environment for itself.

    ORDER INSIDE THE LOCK, and it is the only rule that keeps a crash
    survivable::

        1. plan + write NEW blobs        (_reconcile_attachments)
        2. write the plate               (_save)
        3. delete now-unreferenced blobs (sweep_orphan_todo_files)

    New bytes go down before the row that names them, so a failed blob write
    leaves nothing in the JSON. Deletions happen strictly AFTER the row stops
    naming them, because delete-then-failed-save would destroy bytes the plate
    still points at — the one failure the "plate is the only source of truth"
    invariant does not cover. Both halves stay inside the same lock, and the
    worst case (a crash in the gap) leaves an unreferenced directory that the
    next attachment-bearing save sweeps.
    """
    async with _lock:
        todos = _load(data_dir, owner_user_id)
        target: DeskTodo | None = None
        for t in todos:
            if t.id == todo_id:
                target = t
                break
        if target is None:
            return None
        if archived is not None:
            # THE ONE WRITER (AC-1.6). This assignment is the single place in
            # the product that records "the owner filed this away". Its three
            # callers are the desk checkbox, the CLI archive/unarchive verbs,
            # and the trigger's --consume option. Nothing project-driven writes
            # it and nothing reads it as "the work finished".
            #
            # A bool has no value to validate, so the old
            # `status must be 'open' or 'done'` branch is gone outright. None
            # still means "untouched", exactly as it did for `status`.
            target.archived = archived
        if text is not None:
            text = text.strip()
            if text:
                target.text = text
        if draft_prompt is not None:
            target.draft_prompt = draft_prompt
        if draft_recipient_id is not _UNSET:
            target.draft_recipient_id = draft_recipient_id
        if draft_recipient_name is not _UNSET:
            target.draft_recipient_name = draft_recipient_name
        if drafted is not None:
            target.drafted = drafted
        if labels is not _UNSET or add_labels is not None or remove_labels is not None:
            target.labels = merge_label_keys(
                target.labels,
                replace=labels,
                add=add_labels,
                remove=remove_labels,
            )
        if attachments is not _UNSET:
            desired = list(attachments) if isinstance(attachments, list) else []
            # Step 1: plan, then write the new bytes — before the row names them.
            target.attachments = _reconcile_attachments(
                data_dir, owner_user_id, target, desired, total_limit
            )
        target.updated_at = _now()
        # Step 2: the plate. From here on the new blobs are reachable and the
        # ones the desired-state array dropped are, by definition, garbage.
        _save(data_dir, owner_user_id, todos)
        if attachments is not _UNSET:
            # Step 3: collect that garbage — strictly after the row stopped
            # naming it, never before. This is the SAME primitive delete_todo
            # uses; the reconcile has no delete branch of its own to get wrong.
            sweep_orphan_todo_files(data_dir, owner_user_id)
        return target


async def associate_todo(
    data_dir: Path, owner_user_id: str, todo_id: str, project_id: str
) -> DeskTodo | None:
    """Add one project (or DM-thread) id to a to-do. Returns the updated to-do,
    or None if the to-do is gone.

    IDEMPOTENT: associating a project the to-do already carries is a no-op that
    still returns the to-do, not a 409. That matches ``add_labels`` /
    ``remove_labels`` and it matches them for the same reason — the owner's
    assistant retries commands, and a rule that succeeds on the first run and
    fails on the second is the worst kind of rule to hand an agent.

    Stores the id VERBATIM. It does not resolve the project, does not validate
    it, and does not check who owns it: this module knows nothing about
    projects. Authorization is the ROUTE's job and lives there, because the
    visibility predicate it needs (``server/routes/_batch.py``) is not shipped
    in the runner wheel while this module is.
    """
    project_id = (project_id or "").strip()
    if not project_id:
        raise ValueError("project_id cannot be empty")
    async with _lock:
        todos = _load(data_dir, owner_user_id)
        for t in todos:
            if t.id == todo_id:
                if project_id not in t.project_ids:
                    t.project_ids.append(project_id)
                    t.updated_at = _now()
                    _save(data_dir, owner_user_id, todos)
                return t
        return None


async def dissociate_todo(
    data_dir: Path, owner_user_id: str, todo_id: str, project_id: str
) -> DeskTodo | None:
    """Remove one project id from a to-do. Returns the updated to-do, or None
    if the to-do is gone.

    Idempotent in the other direction: removing an id the to-do does not carry
    is a no-op that still returns the to-do.

    This is also the owner's ONLY way to clear a DANGLING id — one whose
    project has been deleted — which is why the route above it takes no
    ownership check on the way out. A deleted project resolves to nothing, so
    an ownership check would make exactly the rows that most need removing the
    ones that cannot be removed.
    """
    async with _lock:
        todos = _load(data_dir, owner_user_id)
        for t in todos:
            if t.id == todo_id:
                if project_id in t.project_ids:
                    t.project_ids = [p for p in t.project_ids if p != project_id]
                    t.updated_at = _now()
                    _save(data_dir, owner_user_id, todos)
                return t
        return None


async def reorder_todos(
    data_dir: Path, owner_user_id: str, ordered_ids: list[str]
) -> list[DeskTodo]:
    """Reorder the plate to match ``ordered_ids``. Ids not present are ignored;
    any todos omitted from ``ordered_ids`` are appended in their prior order so
    a partial list (e.g. only the open items) never drops the rest."""
    async with _lock:
        todos = _load(data_dir, owner_user_id)
        by_id = {t.id: t for t in todos}
        seen: set[str] = set()
        ordered: list[DeskTodo] = []
        for tid in ordered_ids:
            t = by_id.get(tid)
            if t is not None and tid not in seen:
                ordered.append(t)
                seen.add(tid)
        for t in todos:
            if t.id not in seen:
                ordered.append(t)
        _save(data_dir, owner_user_id, ordered)
        return ordered


async def delete_todo(data_dir: Path, owner_user_id: str, todo_id: str) -> bool:
    """Remove a task, and its stored bytes with it. Returns True if it existed.

    The plate write goes FIRST here, for the same reason blob deletions come
    last in ``patch_todo``: the row is what makes the bytes reachable, so it
    goes first and the bytes follow. A crash in the gap leaves an unreferenced
    directory, which is garbage by definition and which the sweep — run here,
    and on the next attachment-bearing save — takes.

    ``rmtree`` is best-effort by contract: a filesystem error is logged, not
    raised. The to-do is already gone from the plate, and failing the delete
    would leave the user with an item they cannot remove; the sweep is the
    backstop. It runs over the whole OWNER directory rather than just this
    to-do's, because a delete is the cheapest moment to notice that some
    earlier crash left bytes behind.
    """
    async with _lock:
        todos = _load(data_dir, owner_user_id)
        kept = [t for t in todos if t.id != todo_id]
        if len(kept) == len(todos):
            return False
        _save(data_dir, owner_user_id, kept)
        try:
            shutil.rmtree(todo_files_dir(data_dir, owner_user_id, todo_id))
        except FileNotFoundError:
            pass
        except OSError as e:  # pragma: no cover - filesystem edge
            logger.warning(
                f"Could not remove attachment dir for to-do {todo_id!r}: {e}"
            )
        sweep_orphan_todo_files(data_dir, owner_user_id)
        return True
