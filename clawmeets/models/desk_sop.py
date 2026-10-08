# SPDX-License-Identifier: MIT
"""
clawmeets/models/desk_sop.py

Desk SOP store — the My Desk right-rail "SOP" library: stored, reusable
prompts the manager hands to an agent over and over.

An SOP body carries typed blanks written ``{{Label|kind:config}}``. The grammar
has two implementations, kept in lockstep by the shared corpus at
``tests/fixtures/sop_templates.json``: ``models/sop_template.py`` (server + CLI)
and ``web/frontend/src/components/composer/placeholders.ts`` (browser). Clicking
one in the rail loads it into the desk composer as fillable chips, already
addressed to ``agent_id``; ``clawmeets sop trigger`` fills the same blanks from
``--set`` pairs and DMs the result to that agent.

Unlike the to-do plate this list is *searched*, not dragged — so there is no
reorder verb and no ordering contract beyond "newest first". It is still one
ordered JSON document per user, because that keeps a write a single
whole-file rewrite under one lock.

A brand-new owner does not open an empty rail: ``SEED`` below is a small
starter library, materialized in memory on the first read and written to disk
by the first mutation. It is never re-asserted afterwards — see the comment on
the constant.

Titles of the form ``Prefix:Name`` group the rail into sections (the frontend's
``utils/sopSection.ts`` derives the heading from the prefix). That is purely a
title convention: nothing here parses it, stores it, or validates it, so a
rename is the only thing needed to move an SOP between sections.

Storage::

    {data_dir}/desk-sops/
      <owner_user_id>.json     # list[DeskSop], newest first
                               # ABSENT => the SEED, not an empty library

Mutations are broadcast to every live session the owner holds via
``DESK_SOP_SYNC`` (see ``server/routes/desk_sops.py``), so a library edited
in one browser tab updates in all the others without a reload.
"""
from __future__ import annotations

import asyncio
import secrets
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from clawmeets.utils.file_io import FileUtil

_lock = asyncio.Lock()

SOPS_DIR = "desk-sops"

# Longest body we will store. Generous for a prompt template, small enough
# that a runaway paste can't bloat the owner's single JSON document.
MAX_BODY_CHARS = 16_384

# ---------------------------------------------------------------------------
# The starter library
# ---------------------------------------------------------------------------
#
# What a brand-new owner finds in the rail. These are ordinary SOPs — editable,
# deletable, schedulable, no ``locked`` field and no guard anywhere in this
# module protecting one. A helpful start, not a fixture.
#
# THE SEED IS A FIRST-WRITE MATERIALIZATION AND IS NEVER RE-ASSERTED. It decides
# only what an owner who has never written a library sees; the moment anything
# is written, the file is authoritative forever after. An implementation that
# re-adds missing rows on read is wrong, and it is wrong in the way that
# silently resurrects a seed SOP every time the owner deletes it. Mirrors
# ``models/desk_label.py``, which carries the long-form version of this
# argument — the two modules make the same promise and must keep making it.
#
# The decision is made on ``path.exists()`` and on nothing else, because
# ``FileUtil.read(..., default=...)`` returns the default on a JSONDecodeError
# as well as on a missing file: deciding "seed?" on the parsed value would
# re-seed a CORRUPT library, which is the one case where the owner's rows are
# still on disk and most need not to be written over.
#
# The ``SYSTEM:`` prefix is not decoration and not an enum. Titles of the form
# ``Prefix:Name`` group the rail into sections (``utils/sopSection.ts`` in the
# frontend derives the section from the prefix), so the four platform SOPs land
# under one SYSTEM heading and stay out of the owner's own list. The eight
# ``SOLOPRENEUR:`` rows after them are a numbered playbook (register an investor,
# validate the idea, ... build the product) and file under their own heading;
# the rail orders a section by display name with numeric collation, so the
# ``1.``..``8.`` in each title is what keeps them in playbook order. An owner who
# renames one
# to ``Ops:Register new agent`` moves it to an Ops section; that is the whole
# mechanism, and there is no server-side registry of section names to keep in
# sync with it.
#
# Every seed leaves ``agent_id``/``agent_name`` null: every one is a job only
# the owner's assistant can do (the SOLOPRENEUR ones create projects and
# register agents on the owner's behalf), and a null recipient already resolves to
# ``{username}-assistant`` (``utils/sopRecipient.ts``). Naming it here instead
# would bake one user's assistant name into a constant.
SEED_TIMESTAMP = "1970-01-01T00:00:00+00:00"

# The moderation steps of the convene panel: the blind round, the claim ledger,
# the reconciliation loop and the report. Shared with the convene starter to-do
# in ``models/desk_todo.py``, which carries it inline (the owner may delete this
# SOP; the to-do has to keep working) but must not drift from it. The SOP body
# below is byte-identical to the text it had before this was split out, which
# ``tests/test_desk_sops.py`` pins.
CONVENE_PROCEDURE = (
    "Create a project with the panel. You moderate: you run the loop "
    "and keep the ledger, but you do not vote and do not add your own "
    "view. Keep the ledger and each round's tracker as files in the "
    "project.\n"
    "\n"
    "A. Blind round. Open one room per panelist and send every "
    "panelist the identical brief at the same time, so they all work "
    "in parallel. Tell each panelist to work only in its own room and "
    "not to open another panelist's room until you publish the claim "
    "ledger. Each answers with: a conclusion, a confidence level, the "
    "key claims with the evidence or source behind each one, its "
    "assumptions stated explicitly, and what would change its mind.\n"
    "\n"
    "B. Claim ledger. Once every answer is in, merge them into one "
    "claim ledger: one row per claim, listing which panelists support "
    "it, which dispute it, the evidence each side gave, and a status "
    "(agreed or open). Publish it to the whole panel.\n"
    "\n"
    "C. Reconciliation loop. Repeat the following until a stop "
    "condition is met.\n"
    "  1. Each panelist, in its own room and in parallel with the "
    "others, goes through every open ledger item and answers each one "
    "with:\n"
    "       - hold: rebut with evidence, a source, a comparable, or "
    "first-principles reasoning;\n"
    "       - concede: name the specific fact or reasoning step that "
    "changed its view (\"the others agree\" is not a reason); or\n"
    "       - refine: restate the claim more narrowly so it can be "
    "agreed.\n"
    "     A new claim is allowed only with evidence, and enters the "
    "ledger as open.\n"
    "  2. Record every answer in this round's tracker: one row per "
    "open ledger item, one column per panelist.\n"
    "  3. Close the round yourself by reconciling the tracker into the "
    "ledger:\n"
    "       - every panelist accepts the item: mark it agreed;\n"
    "       - a factual dispute: check the source, settle it, and "
    "record how;\n"
    "       - the split comes from a hidden assumption: name the "
    "assumption and split the item into one branch per assumption, "
    "with the conclusion under each;\n"
    "       - the facts are agreed but weighed differently: mark it a "
    "judgment call, write both sides down, and take it out of the "
    "loop.\n"
    "     Publish the updated ledger to the panel.\n"
    "  Stop when any of these is true:\n"
    "    - the consensus bar is met on the conclusion;\n"
    "    - the max rounds have been run;\n"
    "    - no substantial progress: a round ends with no ledger item "
    "changing status and no panelist changing its position.\n"
    "\n"
    "D. Report back with: the consensus conclusion and its confidence "
    "(or \"no consensus\"), and which stop condition ended the loop; "
    "the points everyone agreed on; the disputes that were settled and "
    "how; the open disagreements side by side, each with the evidence "
    "that would resolve it; and the final position of each panelist. "
    "Never average the positions together, and never hide a dissent."
)


SEED: tuple[dict[str, str], ...] = (
    {
        "id": "sop-seed-personalize-assistant",
        "title": "SYSTEM:Personalize assistant",
        "body": (
            "Write an onboarding briefing for a brand-new assistant who knows "
            "nothing about me, drawing on everything you already know — our "
            "history together, your memory files, and my knowledge packs.\n"
            "\n"
            "Cover exactly these five sections, in this order:\n"
            "\n"
            "1. Who I am — my role, my company, and what I am accountable for.\n"
            "2. What I am working on right now — each live project, its current "
            "state, and who else is involved.\n"
            "3. My priorities for {{Horizon|select:this quarter,this month,the "
            "next six months}} — ranked, each with the outcome that would make "
            "it a success.\n"
            "4. How to work with me — how I like to be communicated with, what "
            "I want decided without me, what I always want to be asked about, "
            "and the mistakes a new assistant is most likely to make with me.\n"
            "5. My business — model, customers, offering, and the numbers that "
            "actually matter.\n"
            "\n"
            "Be specific: real names, real numbers, real dates. Never "
            "\"various clients\" or \"several projects\". Where you do not "
            "actually know something, write \"UNKNOWN — ask\" rather than "
            "guessing, and collect every UNKNOWN at the end as a numbered list "
            "of questions for me.\n"
            "\n"
            "Then stop. Show me the briefing and ask me to confirm or correct "
            "it — memorize nothing yet. Once I confirm, reflect and commit the "
            "confirmed briefing to memory so it carries into future sessions."
        ),
    },
    {
        "id": "sop-seed-register-agent",
        "title": "SYSTEM:Register new agent",
        "body": (
            "Register a new agent for me and bring it to full working standard "
            "before it takes on any real work.\n"
            "\n"
            "  Name:      {{Name|text:CTO}}\n"
            "  Industry:  {{Industry|text:AI and blockchain}}\n"
            "  Expertise: {{Expertise|text:software architecture, Python, "
            "TypeScript, React, distributed systems}}\n"
            "  Mentors:   {{Mentors|agents}}\n"
            "\n"
            "Run these four steps in order and do not skip one:\n"
            "\n"
            "1. Register the agent, with a role description built from the "
            "industry and expertise above.\n"
            "2. Brain dump. Every mentor named above writes down everything "
            "proprietary, hard-won, or non-obvious it knows that touches this "
            "agent's industry and expertise — our conventions and stack, "
            "decisions already made and the reasoning behind them, what we "
            "tried that failed, the people and systems involved, and anything "
            "a competent outsider would get wrong about how we work. Default "
            "to my assistant alone; add a peer only when this role genuinely "
            "straddles someone else's domain, and give each mentor a distinct "
            "slice so two of them don't write the same brief twice. Hand every "
            "brief to the new agent.\n"
            "3. Deep research. Have the new agent research its own industry and "
            "expertise to practitioner depth — state of the art, the standard "
            "tools and their trade-offs, the common failure modes, and where "
            "the field is heading — then reconcile that against the brain dump "
            "and flag anything in our practice that contradicts it.\n"
            "4. Memorize. Have the new agent reflect and commit both the brain "
            "dump and its research to memory, so it opens every future session "
            "already expert instead of reading in.\n"
            "\n"
            "Report back with the agent's name, a short inventory of what it "
            "now knows, and every conflict or gap it flagged."
        ),
    },
    {
        "id": "sop-seed-create-project",
        "title": "SYSTEM:Create new project",
        "body": (
            "Start a new multi-agent project.\n"
            "\n"
            "  Team:      {{Team|text:brand and product}}\n"
            "  Objective: {{Objective|text:produce our tagline, mission "
            "statement, and vision statement}}\n"
            "\n"
            "Before you create anything, come back to me with three things: "
            "the objective restated as a concrete deliverable (what artifact "
            "exists at the end, and what makes it good enough to ship), the "
            "roster you would actually staff it with — say so and explain if it "
            "differs from the team I named, rather than substituting quietly — "
            "and the milestones, each ending in something I can review.\n"
            "\n"
            "Wait for my go-ahead. Creating the project is also the kickoff, so "
            "there is no undo. Once I approve, create it with that roster and "
            "those milestones, and tell me anything you need from me that would "
            "otherwise block the work."
        ),
    },
    {
        "id": "sop-seed-convene-panel",
        "title": "SYSTEM:Convene panel",
        "body": (
            "Convene a panel: have several agents answer the same question "
            "independently, then reconcile their findings until they agree or "
            "until the remaining disagreement is clear enough for me to "
            "decide.\n"
            "\n"
            "  Question:   {{Question|text:Is NVDA a buy at today's price on a "
            "3-year horizon? Give bull, base and bear cases with a price "
            "target for each}}\n"
            "  Panel:      {{Panel|agents}}\n"
            "  Max rounds: {{Max rounds|number:3}}\n"
            "  Consensus:  {{Consensus bar|select:unanimous,supermajority,"
            "majority}}\n"
            "\n"
            f"{CONVENE_PROCEDURE}"
        ),
    },
    {
        "id": "sop-seed-solopreneur-1-register-angel-investor",
        "title": "SOLOPRENEUR:1. Register angel_investor",
        "body": (
            "Please use the SOP \"SYSTEM:Register new agent\" to register an "
            "angel_investor agent specialized in investing in pre-revenue "
            "companies in the {{industry|text:AI application}} industry. The "
            "agent's expertise should focus on evaluating the quality of the "
            "idea and its product market fit rather than anything else "
            "including sourcing deals, legal, compliance, corporate formation "
            "which we will have some other experts to cover the ground. While "
            "many other important signals exist in fundraising, such as deal "
            "source channel and founder background, the agent should evaluate "
            "an investment opportunity purely on the quality of the idea and "
            "its product market fit. The investor agent understands the risk "
            "of early stage investing. Rather than looking for a bulletproof "
            "product, it seeks a strong contrarian view that, once validated, "
            "would generate far more value than incumbents, along with early "
            "indicators or evidence supporting that contrarian view."
        ),
    },
    {
        "id": "sop-seed-solopreneur-2-validate-idea",
        "title": "SOLOPRENEUR:2. Validate idea",
        "body": (
            "Please create a project with angel_investor to iterate on the "
            "following:\n"
            "\n"
            "- Problem: {{pain point|text:many amateur golfers who find golfing "
            "cool but don't know where to start yet finding and paying for a "
            "coach seems to be a big commitment}}\n"
            "- Proposed Solution: {{proposed solution|text:golf app built for "
            "amateur that teaches users with a pre-defined curriculum and can "
            "watch user's swing live, analyze the swing, suggest suitable "
            "drills, and essentially act as an AI coach who is always "
            "available, constructive, and never embarrassing for shy users}}\n"
            "- Unique Insight that something only I know: {{unique insight|"
            "text:existing golf coaching apps have been focusing on golf "
            "enthusiasts while I want to focus on the people just started golf "
            "and youtube shorts have been a popular source for knowledge and "
            "drills but very unreliable quality and often wrong}}\n"
            "\n"
            "In the project, you should act as an entrepreneur to iterate with "
            "angel_investor. In each iteration, you should make a revised pitch "
            "to the angel_investor. Based on the feedback, either push back or "
            "adapt and revise your pitch. The iterations should stop only when "
            "no substantial progress can be made, max 10 back-and-forth "
            "iterations, or the angel_investor accepts the pitch (willing to "
            "invest). The acceptance can be conditional yet the conditions "
            "should be as plausible as possible.\n"
            "\n"
            "Please also help name the product while researching existing "
            "trademarks, patented names, and DNS availability. Lastly, in the "
            "case that an acceptable pitch can be identified, please create an "
            "interactive HTML pitch deck as well."
        ),
    },
    {
        "id": "sop-seed-solopreneur-3-register-product-marketing",
        "title": "SOLOPRENEUR:3. Register product and marketing agents",
        "body": (
            "Please use the \"SYSTEM:Register new agent\" SOP to create the "
            "product and marketing agents about the {{product name or idea "
            "description|text:the golf app for beginner}} with relevant agents "
            "as mentors."
        ),
    },
    {
        "id": "sop-seed-solopreneur-4-icp-go-to-market",
        "title": "SOLOPRENEUR:4. Strategize ICP and Go-To-Market",
        "body": (
            "Please create a project with product and marketing agents to "
            "design the go-to-market strategy. For the output, I would like to "
            "see two tables:\n"
            "\n"
            "- ICP table listing persona, user demand, product supply, and how "
            "product satisfies user demand\n"
            "- Channel table listing marketing channel, persona, messaging, "
            "estimated investment, estimated return, priority, how the message "
            "through the channel can touch the persona effectively. The "
            "estimated investment, estimated return, desired outcome; priority "
            "can be 1-5 or high, low, mid.\n"
            "\n"
            "In the project, you should act as an entrepreneur and coordinate "
            "with marketing, product and angel_investor agents to iterate on "
            "the go-to-market strategy. In each iteration, you should ask "
            "marketing agent to revise the go-to-market strategy, and ask "
            "product and angel_investor agents for feedback. The iterations "
            "should stop only when no substantial progress can be made, after "
            "a maximum of 8 back-and-forth iterations, or when you, as an "
            "entrepreneur, are comfortable that at least one marketing channel "
            "can effectively reach one of our ICP personas and the "
            "corresponding messaging can effectively convert them to use our "
            "product."
        ),
    },
    {
        "id": "sop-seed-solopreneur-5-register-design-eng",
        "title": "SOLOPRENEUR:5. Register designer and eng team",
        "body": (
            "Please create and set up the shared git mono repo at "
            "~/clawmeets_demo for the agents to collaborate on. Its git repo url "
            "is file:// followed by that directory's absolute path (git does not "
            "expand ~ inside a url, so file:///~/clawmeets_demo would not "
            "resolve). In the repo, there should be a README.md describing your understanding of "
            "the product ({{product name or idea description|text:the golf app "
            "for beginner}}) and ios, android, web, db, backend folders empty "
            "initially.\n"
            "\n"
            "Then, use the \"SYSTEM:Register new agent\" SOP to create the "
            "designer, fullstack_engineer, android_engineer, ios_engineer "
            "agents with relevant agents as mentors.\n"
            "\n"
            "- The designer agent should be experienced in web, mobile and "
            "graphic UI/UX design, information architecture, color palette and "
            "font selection\n"
            "- The fullstack_engineer agent should be experienced in python, "
            "sqlite, javascript, html css react tailwind and set its git repo "
            "url to that repo url\n"
            "- The android_engineer agent should be experienced in android "
            "development stack and set its git repo url to that repo url\n"
            "- The ios_engineer agent should be experienced in iOS development "
            "stack (Swift/SwiftUI/Xcode) and set its git repo url to that repo "
            "url"
        ),
    },
    {
        "id": "sop-seed-solopreneur-6-spec-product",
        "title": "SOLOPRENEUR:6. Spec product",
        "body": (
            "Please create a project with product agent to create a product PRD "
            "that describes the necessary features to make an MVP for "
            "{{product name or idea description|text:the golf app for "
            "beginner}} that provides substantial value to the ICP.\n"
            "\n"
            "In the project, you should act as an entrepreneur and coordinate "
            "with product agent to iterate on the user stories for each "
            "necessary feature and then order the feature in a way that's easy "
            "to be built incrementally. In each iteration, the product agent "
            "should write or revise each feature along with its user story "
            "while you as an entrepreneur should evaluate if each feature "
            "should be included in the MVP. The iterations should stop only "
            "when no substantial progress can be made, after a maximum of 8 "
            "back-and-forth iterations, or when you, as an entrepreneur, are "
            "comfortable that the implemented features can provide substantial "
            "value to the ICP for them to engage.\n"
            "\n"
            "When the project is done, please save the PRD to the "
            "~/clawmeets_demo git repo and commit to the main branch."
        ),
    },
    {
        "id": "sop-seed-solopreneur-7-design-product",
        "title": "SOLOPRENEUR:7. Design product",
        "body": (
            "Please create a project with designer agent to create the product "
            "mock up based on the PRD for {{product name or idea description|"
            "text:the golf app for beginner}} MVP saved under the "
            "~/clawmeets_demo git repo.\n"
            "\n"
            "In the project, you should act as an entrepreneur and coordinate "
            "with designer agent to iterate on the design mock up. In each "
            "iteration, the designer agent should create or revise its mock up "
            "for better user experience while you as an entrepreneur should "
            "evaluate how the ICP user would feel about the mock up and give "
            "feedback on how the design can be improved. The iterations should "
            "stop only when no substantial progress can be made, after a "
            "maximum of 8 back-and-forth iterations, or when you, as an "
            "entrepreneur, are comfortable that the design once implemented, "
            "will be easy enough for the ICP to engage with.\n"
            "\n"
            "When the project is done, please save the design mockup to the "
            "~/clawmeets_demo git repo and commit to the main branch."
        ),
    },
    {
        "id": "sop-seed-solopreneur-8-build-product",
        "title": "SOLOPRENEUR:8. Build product",
        "body": (
            "Please create a project with product and relevant engineer agents "
            "to build the product based on the PRD and design mockup saved "
            "under the ~/clawmeets_demo git repo. The engineer agents "
            "should break the milestones down based on the features/user "
            "stories (build incrementally one feature at a time) rather than "
            "horizontally (storage layer, backend service layer, frontend "
            "layer). In each milestone, you should coordinate for the engineer "
            "to implement the feature and have the product agent review the "
            "change to accept the milestone or push back for the engineer to "
            "iterate further. If there are multiple products say ios, android, "
            "website need to be built, we should prioritize the milestones to "
            "build website first, ios second, and android last."
        ),
    },
)

# Distinguishes "key absent from a PATCH body" (leave the stored value
# untouched) from an explicit JSON ``null`` (clear the field). Used only for
# the ``agent_*`` pair: a user with no agents saves an SOP with no recipient,
# and later assigning/unassigning one needs absent != null. The route layer
# detects key presence off the raw request body (see
# ``server/routes/desk_sops.py``) since a non-serializable sentinel can't be
# a FastAPI ``Body`` default. Mirrors ``desk_todo._UNSET``.
_UNSET: object = object()


class DeskSop(BaseModel):
    """One stored prompt in the owner's SOP library."""

    id: str
    owner_user_id: str
    title: str
    # Default recipient. Both id and name are persisted so the rail still
    # labels the card correctly after a rename or removal; the frontend
    # re-resolves against the live roster (``utils/sopRecipient.ts``) and
    # falls back to the owner's assistant. Same precedent as
    # ``DeskTodo.draft_recipient_{id,name}``.
    agent_id: str | None = None
    agent_name: str | None = None
    # The prompt itself, with ``{{blanks}}`` left un-substituted.
    body: str
    created_at: str
    updated_at: str


def gen_id() -> str:
    return "sop-" + secrets.token_hex(6)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _path(data_dir: Path, owner_user_id: str) -> Path:
    return Path(data_dir) / SOPS_DIR / f"{owner_user_id}.json"


def _seed_rows(owner_user_id: str) -> list[DeskSop]:
    """The starter library, materialized in memory for one owner.

    The timestamps are the fixed epoch constant rather than the clock. The seed
    is materialized on every read of an unwritten library, so minting ``now()``
    here would make two consecutive GETs return different ``created_at`` values
    for the same row — churn a client can legitimately notice, in a field
    nothing needs. It is also simply more truthful: the owner did not write
    these. Rows keep the constant when the first write materializes them to
    disk; anything the owner creates afterwards is stamped with the real clock
    and sorts ahead of them.
    """
    return [
        DeskSop(
            owner_user_id=owner_user_id,
            created_at=SEED_TIMESTAMP,
            updated_at=SEED_TIMESTAMP,
            **row,
        )
        for row in SEED
    ]


def _load(data_dir: Path, owner_user_id: str) -> list[DeskSop]:
    """Every read and every write starts here, so "missing" vs "empty" vs
    "unreadable" is decided in exactly one place.

    NO FILE -> the SEED, in memory. A file that exists but holds no usable rows
    -> ZERO rows, never the seed as a repair: an owner who deleted their last
    SOP gets an empty rail and keeps it, and a corrupt document is not written
    over with starter content. That is why the branch is ``path.exists()`` and
    not the parsed value — ``FileUtil.read(..., default=...)`` answers the same
    ``None`` for a missing file and for a JSONDecodeError.
    """
    path = _path(data_dir, owner_user_id)
    if not path.exists():
        return _seed_rows(owner_user_id)
    raw = FileUtil.read(path, "json", default=None)
    if not isinstance(raw, list):
        return []
    out: list[DeskSop] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            out.append(DeskSop.model_validate(row))
        except Exception:
            continue
    return out


def _save(data_dir: Path, owner_user_id: str, sops: list[DeskSop]) -> None:
    FileUtil.write(
        _path(data_dir, owner_user_id),
        [s.model_dump() for s in sops],
        "json",
    )


def list_sops(data_dir: Path, owner_user_id: str) -> list[DeskSop]:
    """Return the owner's library, newest first."""
    return _load(data_dir, owner_user_id)


def get_sop(data_dir: Path, owner_user_id: str, sop_id: str) -> DeskSop | None:
    for s in _load(data_dir, owner_user_id):
        if s.id == sop_id:
            return s
    return None


async def create_sop(
    data_dir: Path,
    owner_user_id: str,
    *,
    title: str,
    body: str,
    agent_id: str | None = None,
    agent_name: str | None = None,
) -> DeskSop:
    """Store a new SOP; prepended so the newest is first."""
    title = (title or "").strip()
    if not title:
        raise ValueError("SOP title cannot be empty")
    body = (body or "").strip()
    if not body:
        raise ValueError("SOP body cannot be empty")
    if len(body) > MAX_BODY_CHARS:
        raise ValueError(f"SOP body exceeds {MAX_BODY_CHARS} characters")
    async with _lock:
        sops = _load(data_dir, owner_user_id)
        now = _now()
        sop = DeskSop(
            id=gen_id(),
            owner_user_id=owner_user_id,
            title=title,
            agent_id=agent_id,
            agent_name=agent_name,
            body=body,
            created_at=now,
            updated_at=now,
        )
        sops.insert(0, sop)
        _save(data_dir, owner_user_id, sops)
        return sop


async def patch_sop(
    data_dir: Path,
    owner_user_id: str,
    sop_id: str,
    *,
    title: str | None = None,
    body: str | None = None,
    agent_id: str | None = _UNSET,  # _UNSET → leave untouched
    agent_name: str | None = _UNSET,  # None → clear, str → set
) -> DeskSop | None:
    """Patch an SOP in place. Returns the updated SOP, or None if missing.

    ``title`` / ``body`` never clear: a blank value is ignored, because both
    are required and the dialog already blocks Save on either being empty.
    ``agent_id`` / ``agent_name`` are three-way (see ``_UNSET``)."""
    if body is not None and len(body) > MAX_BODY_CHARS:
        raise ValueError(f"SOP body exceeds {MAX_BODY_CHARS} characters")
    async with _lock:
        sops = _load(data_dir, owner_user_id)
        target: DeskSop | None = None
        for s in sops:
            if s.id == sop_id:
                target = s
                break
        if target is None:
            return None
        if title is not None:
            title = title.strip()
            if title:
                target.title = title
        if body is not None:
            body = body.strip()
            if body:
                target.body = body
        if agent_id is not _UNSET:
            target.agent_id = agent_id
        if agent_name is not _UNSET:
            target.agent_name = agent_name
        target.updated_at = _now()
        _save(data_dir, owner_user_id, sops)
        return target


async def delete_sop(data_dir: Path, owner_user_id: str, sop_id: str) -> bool:
    """Remove an SOP. Returns True if it existed."""
    async with _lock:
        sops = _load(data_dir, owner_user_id)
        kept = [s for s in sops if s.id != sop_id]
        if len(kept) == len(sops):
            return False
        _save(data_dir, owner_user_id, kept)
        return True
