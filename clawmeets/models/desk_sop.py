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
# frontend derives the section from the prefix), so these three land under one
# SYSTEM heading and stay out of the owner's own list. An owner who renames one
# to ``Ops:Register new agent`` moves it to an Ops section; that is the whole
# mechanism, and there is no server-side registry of section names to keep in
# sync with it.
#
# Every seed leaves ``agent_id``/``agent_name`` null: all three are jobs only
# the owner's assistant can do, and a null recipient already resolves to
# ``{username}-assistant`` (``utils/sopRecipient.ts``). Naming it here instead
# would bake one user's assistant name into a constant.
SEED_TIMESTAMP = "1970-01-01T00:00:00+00:00"

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
