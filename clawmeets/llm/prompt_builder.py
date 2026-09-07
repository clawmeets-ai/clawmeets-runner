# SPDX-License-Identifier: MIT
"""
clawmeets/llm/prompt_builder.py
Prompt construction for agent participants.

Layer 0 (pure — no domain model deps).

Per-turn prompt layout (worker, coordinator, DM all share):

  1. Identity            : "You are {name}. {description}. Capabilities. Project / Chatroom."
  2. Role contract       : worker / coordinator / DM behavioural guidance.    ← STATIC
  3. Operational rules   : output schema, file-sharing workflow, memory writes.← STATIC
  4. Runtime context     : knowledge_dirs, memory/, packs/, personal skills,
                           MCP / skill configs, DWH, invitable allowlist.
  5. Knowledge precedence: authoritative vs fallback layers + trigger markers.
  6. Synced file manifest
  7. Recent chat in this room
  8. Incoming message                                                        ← LAST
  9. One-line tail: "Reply as JSON per schema."

Recency-friendly: the model sees the contract first, then the dynamic
context, with the actual task last. Sections 1–5 are byte-stable within an
invocation cluster, so Claude Code's internal prompt-cache machinery can hit.
"""
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .triggers import derive_role, triggers_for


# ---------------------------------------------------------------------------
# OperationalMode Enum
# ---------------------------------------------------------------------------

class OperationalMode(str, Enum):
    """Operational mode of a participant within a project.

    Derived at runtime from ``project.coordinator_id``, not stored. Defined here
    in Layer 0 to avoid circular imports; re-exported from
    ``models.participant`` for backward compatibility.
    """
    WORKER = "worker"
    COORDINATOR = "coordinator"


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


def _build_file_manifest(data_dir: Path) -> str:
    files: list[str] = []
    if data_dir.exists():
        for fp in sorted(data_dir.rglob("*")):
            if fp.is_file():
                files.append(str(fp.relative_to(data_dir)))
    return "\n".join(f"  - {f}" for f in files) if files else "  (empty)"


def _build_chat_history(history: list[tuple[str, str]] | None) -> str:
    """Format ``[(sender, content), ...]`` as a compact transcript.

    ``None`` or empty → ``"  (no prior messages)"``. Each entry on its own
    line. Caller decides ordering (most recent last is conventional) and how
    far back to walk.
    """
    if not history:
        return "  (no prior messages)"
    lines = []
    for sender, content in history:
        # Keep history compact: collapse runs of blank lines, single-line indent.
        body = "\n      ".join(line for line in content.splitlines() if line.strip())
        lines.append(f"  [{sender}] {body}" if body else f"  [{sender}]")
    return "\n".join(lines)


def _build_trigger_section(role: str) -> str:
    """Render the memory-loop trigger list from the central registry.

    Sources from ``triggers.MEMORY_LOOP_TRIGGERS``; new markers added there
    appear here automatically without editing this file.
    """
    specs = triggers_for(role=role)
    lines = [f"- {t.marker:<55} {t.skill} — {t.purpose}" for t in specs]
    return "\n".join(lines)


def _build_knowledge_precedence(
    name: str,
    agent_dir: Path,
    *,
    knowledge_dirs: list[Path] | None = None,
    dwh_dir: Optional[Path] = None,
) -> str:
    """Compact two-layer knowledge-precedence block.

    AUTHORITATIVE layer answers user-world / user-personal questions;
    FALLBACK layer covers generic field knowledge. Layers are decoupled in
    storage; precedence is enforced here in the prompt, not via cross-pointers.

    Every AUTHORITATIVE entry is an *index* following the shared
    knowledge-index contract (``clawmeets.utils.knowledge_index``): each lists
    its files with a one-line 'consult when', so the agent reads the index,
    matches, and opens only the file it needs. Indexes are surfaced only when
    they can exist (REFERENCES.md ⇐ a knowledge_dir is configured; the dwh
    CATALOG.md ⇐ a dwh_dir is configured) to keep the block lean.

    The trigger-marker list lives in its own section
    (``_build_memory_triggers``) so the precedence rule stays scannable.
    """
    is_assistant = name.endswith("-assistant")
    memory_dir = f"{agent_dir}/memory"

    bullets = []
    if is_assistant:
        bullets.append(f"  - {memory_dir}/USER.md")
    bullets.append(f"  - {memory_dir}/KNOWLEDGE_PACKS.md       installed packs")
    if knowledge_dirs:
        bullets.append(
            f"  - {memory_dir}/REFERENCES.md           proprietary reference "
            "files — auto-indexed: filenames + content previews"
        )
    if dwh_dir is not None:
        bullets.append(
            f"  - {dwh_dir}/CATALOG.md                 warehouse tables — for "
            "quantitative / data questions"
        )
    authoritative = "\n".join(bullets)

    return f"""== KNOWLEDGE PRECEDENCE ==
AUTHORITATIVE (user-world facts) — start here for anything about THIS user
or their world (business, product, preferences, domain facts):
{authoritative}
FALLBACK (field knowledge) — generic industry / regulations / comp data:
  - {memory_dir}/learnings/INDEX.md
Each entry above is an index: read it, find the line whose 'consult when'
matches, then open only that file. Layers are decoupled; do not synthesize
from learnings/ on user-world questions.
"""


def _build_memory_triggers(role: str) -> str:
    """Render the memory-write trigger registry as its own section.

    Split out of ``_build_knowledge_precedence`` so the precedence rule
    stays short and the trigger table is easy to scan / update.

    ``role`` is the per-invocation audience from
    ``triggers.derive_role`` — same value the SystemSkillManager uses
    to pick which system-skill subset to materialize for this turn.
    """
    triggers = _build_trigger_section(role=role)
    return f"""== MEMORY-WRITE TRIGGERS ==
The only times you should write to memory. Each marker arrives as an
HTML comment in a DM; follow the matching skill:
{triggers}
"""


def _build_runtime_context(
    *,
    agent_dir: Path,
    data_dir: Path,
    knowledge_dirs: list[Path] | None,
    dwh_dir: Optional[Path],
    git_url: Optional[str] = None,
    roster_path: Optional[Path] = None,
) -> str:
    """Compact `== FILES & STATE ==` block listing all the paths the agent
    can read or write.

    Replaces the prior set of one-section-per-resource blocks (AGENT MEMORY,
    KNOWLEDGE PACKS, KNOWLEDGE BASE, DATA WAREHOUSE) which together carried
    ~30 lines of headers + prose for what is fundamentally a path list.

    Per-MCP and per-skill config-file paths used to live here as
    ``MCP CONFIG FILES`` / ``SKILL CONFIG FILES`` blocks. They were dropped
    once each MCP server learned to self-resolve the path from
    ``$CLAWMEETS_AGENT_DIR/mcp-hub/configs/<name>.json`` (see
    ``clawmeets.mcp.config_resolve``) and every shipped SKILL.md was
    already reading its config from the analogous skill-hub path. Listing
    them in the prompt added bytes and named specific skills/tools by
    identifier — a `feedback_no_skill_names_in_profile` violation.
    """
    lines = [
        "== FILES & STATE ==",
        f"- Synced project files (read-only)  : {data_dir}",
        "- Your sandbox (read/write)         : your current working directory; "
        "share files via the update_file action",
        f"- Agent memory (read/write, runner-managed, NOT broadcast to chat) : {agent_dir}/memory/",
        f"- Knowledge packs (auto-synced)     : {agent_dir}/knowledge_packs/",
    ]
    if roster_path is not None:
        lines.append(
            f"- Worker-agent roster (read-only)   : {roster_path}  "
            "(the registry of agents you can invite — read it before delegating)"
        )
    if git_url:
        lines.append(
            f"- Bound git repo (EXISTING codebase — build on it): {git_url}  "
            "(for any coding task, clone it into ./repos/ under your sandbox and extend the code "
            "already there — never scaffold a new/parallel project; all agents on a project share "
            "one branch per repo, commit & push; repo conventions in memory/REPO.md)"
        )
    if knowledge_dirs:
        kd = ", ".join(str(d) for d in knowledge_dirs)
        lines.append(
            f"- User-curated reference material (read-only): {kd}"
        )
    if dwh_dir is not None:
        lines.append(f"- Data warehouse                    : {dwh_dir}")
    return "\n".join(lines)


def _build_output_contract(actions: list[str], is_coordinator: bool) -> str:
    """Compact structured-output contract. The CLI flag enforces the schema;
    this block tells the model what each action shape means and how to format
    a "no actions" response. Examples are minimal — one per role.
    """
    action_lines = []
    if "reply" in actions:
        action_lines.append(
            '  {"type": "reply", "room": "<chatroom_name>", "content": "<text>"}'
        )
    if "update_file" in actions:
        action_lines.append(
            '  {"type": "update_file", "room": "<chatroom_name>", "file_path": "<relative_path>"}'
        )
    if "create_room" in actions:
        action_lines.append(
            '  {"type": "create_room", "name": "<name>", "invite": ["<agent_name>"], '
            '"init_message": "@<agent_name> <text>"}'
        )
    if "invite_agent" in actions:
        action_lines.append(
            '  {"type": "invite_agent", "room": "<existing_room>", '
            '"invite": ["<agent_name>"], "message": "@<agent_name> <text>"}'
        )
    if "project_completed" in actions:
        action_lines.append(
            '  {"type": "project_completed"}'
        )

    coordinator_note = (
        "\n\n@MENTION ADDRESSING\n"
        "  - \"@agent-name\"  → that agent is expected to respond\n"
        "  - \"agent-name\"   → read-only; agent sees the message but does not respond\n"
        "  Only @-mention agents you need a reply from in this batch."
        if is_coordinator else ""
    )
    if "create_room" in actions:
        coordinator_note += (
            "\n  A create_room init_message MUST begin by @-mentioning an invited\n"
            "  agent using their EXACT roster name — an un-@mentioned worker is\n"
            "  never triggered, so the room (and the project) stalls."
        )
    if "invite_agent" in actions:
        coordinator_note += (
            "\n  invite_agent adds agents to a room that ALREADY exists, then\n"
            "  addresses them. Its message MUST @-mention every newly invited\n"
            "  agent by exact roster name, and MUST carry the context they need:\n"
            "  an invitee already working elsewhere in this project does NOT\n"
            "  receive the room's earlier messages."
        )

    return f"""== OUTPUT CONTRACT ==
Emit one JSON object per turn, matching the structured-output schema:
  {{"actions": [ ... ]}}
If no action is needed, emit {{"actions": []}}.

Available action shapes:
{chr(10).join(action_lines)}

Emit ALL actions in a single response. A second structured-output call in the
same turn REPLACES the first, not append.{coordinator_note}

== FILE SHARING ==
Your working directory is a sandbox. To share a file with the chatroom:
  1. Write the file with the Write tool.
  2. Emit update_file with the same relative path; the server syncs it to
     all participants.
"""


# ---------------------------------------------------------------------------
# Shared turn-continuity rule
# ---------------------------------------------------------------------------

# Injected into the NON-DELEGATING role contracts only (worker, DM-worker,
# owned-DM coordinator). Those surfaces have action schema
# ``["reply", "update_file"]`` and no re-wake path — nothing invokes them again
# on their own. It forbids ONLY self-resumption promises ("I'll go do X and
# update you"); it deliberately blesses the two legitimate cross-turn patterns
# (asking the user and continuing on their reply; a worker reporting BLOCKED)
# and intra-turn parallel/background work. Regular / FD-tunnel coordinators are
# left out: they DO get re-woken via BATCH_COMPLETE, so their promise is real.
@dataclass(frozen=True)
class PlanPromptState:
    """What a coordinator turn is told about its project's plan (§7.3).

    Four facts, assembled by ``models/agent.py`` from the project's **own synced
    fields** — the sidecar lives on the server and the agent process cannot open
    it, which is why every member here is either a project field or a pure
    function of one and the synced ``PLAN.md``.

    ``open_notes_for_you`` is the number the turn-contract obligation is stated
    on **and** the number the server-side execution gate refuses on (§7.4). One
    number, one definition; this carries it, it does not recompute it.

    ``approval`` is what ``## Approval`` says right now — the section the user's
    acceptance is a **diff into**. It replaced *"wait for the user to say go in
    chat"*, which was a judgement call about a sentence: a coordinator reading
    *"looks good, though I'd rethink M2"* had to decide whether that was a go,
    and it decided wrong in the direction that starts work. The document's own
    text is not a judgement call, and it is the same bytes the gate and the desk
    are looking at.
    """

    title: str
    phase: str
    changed_since_acceptance: bool
    open_notes_for_you: int
    approval: str = ""
    #: Has the USER opened a review round on this plan? The spec lock's start
    #: line before acceptance (``project_plan._spec_is_locked``). A fifth fact,
    #: and it earns its seat for the reason the block's own docstring gives
    #: about the execution gate: the server refuses the write either way, but a
    #: model that meets an unexplained refusal invents an explanation, and the
    #: one it invents is *"my change is lost"* — which is the single thing that
    #: is not true. Stating it up front is also what stops the retry loop.
    #:
    #: Projected onto the project by ``PROJECT_PLAN_STATE`` like every other
    #: member here, because this dataclass is assembled in the AGENT process and
    #: the sidecar that knows about rounds lives only on the server.
    user_has_reviewed: bool = False


# B5 — the ONE batch-completion instruction. It used to be hardcoded in
# ``models/agent.py`` and typed out a second time in
# ``scripts/dump_sample_prompts.py``, with a third copy generated into
# ``scripts/sample-prompts/coordinator-batch.txt``. Two hand-maintained copies of
# a sentence that has to be identical is a drift generator; the sample dump is
# only worth reading if it renders what the runner renders (AC-7.12, D20).
#
# The verb changed with the plan system: a batch ticks ONE checkbox and the
# narrative goes to the milestone's own chatroom (§4.5, §7.3). There is no
# Review Log and no Learnings section to append to any more.
BATCH_COMPLETION_PLAN_INSTRUCTION = (
    "IMPORTANT: before deciding next steps, tick this milestone's checkbox in "
    "`## Milestones` with `clawmeets plan update <project> --section milestones "
    "--body-file <f>`, and post the pass/fail detail per acceptance criterion "
    "into this milestone's own chatroom. Do NOT add a log section to the plan — "
    "the room IS the log."
)


#: M5 AC-5.6 — the SPEC-STAGE sibling of the constant above, and its verb is
#: the opposite one.
#:
#: A ``BATCH_COMPLETE`` in ``shared-context`` before the plan is accepted is a
#: consultation about the spec, not a finished milestone. Telling the
#: coordinator to tick a checkbox there is wrong (there is no milestone to
#: tick, and the plan is still being written) and telling it to open the next
#: workroom is worse: ``_plan_execution_blocked`` refuses every ``create_room``
#: while the go-note is open — which is the whole of ``spec-ing``, since
#: accepting that note is what ends it — and the action is dropped as a
#: ``NO_OP``, so the model is not told and narrates as though a room opened.
#:
#: Selected in ``models/agent.py`` by the SAME boolean that selects the
#: role-contract block, so the synthetic message and the contract can never
#: disagree about which workflow this turn is. Two sentences in two files that
#: merely happen to agree is the drift the constant above was extracted to end.
#:
#: **It now serves TWO rounds, and that is why it splits the answers.** With
#: ``_spec_role_contract``'s CONSULT step the FIRST spec round also ends in a
#: ``shared-context`` batch — one the coordinator opened before the user has
#: seen anything — so this constant is the only contract that turn has. Telling it
#: to file EVERY answer as a note would hand the user a tray of questions the
#: specialists already answered. The keeper writes what is settled and asks
#: about what is not; that split is the whole difference between a plan the
#: user reviews and a plan the user re-derives.
#:
#: **And it ends in prose.** ``clawmeets plan review`` posts the rendered
#: batch — section text, diff, thread, the four response commands — and
#: nothing else. The first spec turn is told to explain itself in
#: ``user-communication``; every later round had no equivalent, so the user's
#: second message from the project was diffs with no covering note.
BATCH_COMPLETION_PLAN_REVIEW_INSTRUCTION = (
    "IMPORTANT: this batch answered questions about the PLAN, which is not "
    "accepted yet. Do NOT tick a milestone checkbox and do NOT create a "
    "workroom — the server refuses `create_room` until the user accepts, and "
    "it refuses it silently. Read each agent's answer and SPLIT it. What is "
    "settled — a correction, a missing step, a sequencing fix nobody has to "
    "choose between — you WRITE INTO THE PLAN yourself, because you are its "
    "keeper: `clawmeets plan update <project> --section <slug> --body-file "
    "<f>`. **UNLESS THE USER HAS ALREADY REVIEWED THIS PLAN** — the block "
    "above says so in as many words when they have. From that point they "
    "decide what it says: the same text goes as a `plan note` proposal "
    "instead, EVERY answer becomes one including the settled ones, and a "
    "`plan update` that moves what the plan says is refused (your text is "
    "filed for the user automatically and you get the note ids back). A "
    "checkbox tick and an HTML comment still land either way. "
    "Only what genuinely needs the USER's judgment — a trade-off, a "
    "scope call, two agents who disagree — becomes a note to them "
    "(`clawmeets plan note <project> --section <slug> --to user -m \"...\"`), "
    "and name the agent whose answer it came from in the note's text. Then "
    "send the notes with `clawmeets plan review <project>`: filing a note "
    "badges the user's card, the batch is what puts it in front of them. "
    "FINALLY, reply in `user-communication` in your own words — what the "
    "specialists changed in the plan, what is still open and needs the user's "
    "decision, and, once nothing is left open, ask for their acceptance "
    "explicitly. The rendered batch is diffs and threads with no framing; "
    "without that reply the user is reading a changelog and guessing."
)


#: The gate-release sibling. Not a batch at all: the runner synthesises this
#: turn when a ``PROJECT_PLAN_STATE`` batch drops the §7.4 EXECUTION gate from
#: blocking to clear and that batch woke nobody by message.
#:
#: ``POST /plan/notes/{id}/resolve`` publishes the projection and posts nothing,
#: so without this the last blocking note could be resolved and the project
#: would simply never move again. It is also the net under any route that
#: publishes the projection AFTER the message that wakes on it, which leaves the
#: woken turn reading the pre-resolution count — the ordering in
#: ``submit_review``'s prelude is the fix for that, this is the backstop.
#:
#: Scoped to the execution gate and not to plan acceptance, which has its own
#: designed wake-up in the acceptance marker; the argument is in
#: ``Agent.on_plan_state_change``. So the text below can speak about NOTES
#: rather than hedging across two different releases.
#:
#: It says "resume", not "report": the user has already been told what was
#: blocking, and a turn that opens with an apology for a stall the user did not
#: see is worse than one that just does the work.
PLAN_GATE_RELEASED_INSTRUCTION = (
    "The plan notes that were blocking this project are now resolved and the "
    "execution gate is clear — nothing is waiting on the user. Nobody sent you "
    "a message; this turn exists because that gate opened. Pick the work back "
    "up where it stopped: read the plan and the rooms, and take the next step "
    "you were holding, which is usually opening the next milestone workroom. "
    "Do NOT ask the user to resolve anything, and do NOT re-report the block — "
    "they cleared it. If in fact nothing is outstanding and the project is "
    "finished, say so and complete it."
)


#: M6 AC-6.1 — the relay procedure, as ONE constant with TWO callsites.
#:
#: **Two, because the turn that reads it is not the turn that is told about
#: it.** ``_spec_role_contract`` renders only on a project's FIRST turn, and
#: the relay happens later — when the user stages notes and the batch lands in
#: ``user-communication`` addressed to the coordinator, which is an ordinary
#: steady-state turn. Putting the procedure only in the spec contract would put
#: it in front of a coordinator several turns before it needs it and nowhere at
#: all when it does. So it also rides ``_plan_block`` while the project is in
#: ``spec-ing``.
#:
#: It carries the two MECHANICAL reasons the relay is one message, because a
#: coordinator that knows only the rule will helpfully send one message per
#: agent and lose every group after the first.
#:
#: STEP 0 is a triage gate, and it is here rather than in the spec contract on
#: purpose. The contract's STEP 3 consult is unconditional — a fresh draft goes
#: to the roster before the user ever sees it, and triage must never suppress
#: THAT. This constant governs the other direction: a review batch that has
#: already come back from the user, where the default without a gate is to
#: forward everything. The coordinator is the plan's keeper and most arriving
#: notes are its own to settle; a round spent relaying what it already knew is
#: a wait the user pays for and a turn every agent in the room pays for.
_RELAY_PROCEDURE = """== RELAYING A PLAN REVIEW ==
When a plan review batch arrives addressed to you, those notes are the USER's
and the answers are the specialists'. You are the relay — and a relay decides
what needs relaying.

The user may write "this one is for backend" INSIDE a note. That is prose you
read and route — there is no addressee field for an agent, because a plan note
connects exactly two parties, you and the user, in both phases. Nobody else is
ever an end of it.

  0. SORT THE ARRIVING NOTES FIRST. A note you can settle from the plan, from
     the project context, or from what the specialists already told you in an
     earlier round, you ANSWER YOURSELF — no consult round for it. You are the
     plan's keeper; a correction, a clarification, a rewording, a question
     about your own reasoning are all yours to make.
     BUT THE ANSWER IS A PROPOSAL, NOT A WRITE. This batch IS the user's
     review, so from here they decide what the plan says: file your change as
     a note carrying it — `clawmeets plan note <project> --section <slug> --to
     user --edit-file <f> -m "why"` — and send the batch. A `plan update` that
     moves what the plan SAYS is refused; your text is not lost, it is filed
     for the user automatically and you get back the note ids. Ticking a
     checkbox and editing an HTML comment are not spec changes and still land
     silently, so keep reporting progress exactly as before.
     File them TOGETHER and send ONE batch. Five notes in one batch is one
     message to the user; five batches is five.
     Send on only what actually needs a seat you do not occupy — a domain
     judgment you cannot make, a trade-off between two agents' areas, a
     question about work you are not the one doing. The reason is mechanical
     and it is a cost the user pays: a consult round is a wait for them and a
     turn for every agent in it, so a round spent asking three specialists
     something you already knew buys the user nothing and delays the answer
     they asked for. If NOTHING in the batch needs a seat, settle it all, send
     your notes, and skip steps 1-4 entirely — there is no round to run.
     A name the user WROTE is not yours to triage: step 2 still applies to
     every one of them.

  1. `clawmeets plan consult <project>` — seats the specialists in
     `shared-context` and prints who is in it. You may not open a room during
     spec-ing; this is the room you already have.
  2. Check every name the user wrote against that roster. For any name that is
     not in the room, file a note to the user saying so — `clawmeets plan note
     <project> --to user -m "..."`. Never drop one: a name you drop is a
     question the user believes was asked.
  3. Post ONE `reply` into `shared-context`, sectioned per agent, each section led
     by a leading @exact-name. ONE message, not one per agent, and here are
     BOTH reasons:
       - a batch is keyed on (project, room), so a SECOND addressed message
         into this room collides with the first and is swallowed with a
         warning. Every group after the first would get no batch, no timeout
         and no wake-up.
       - and `clawmeets plan review` cannot carry this either: its agent path
         opens no batch at all, so it yields no BATCH_COMPLETE and nothing
         wakes you when the answers land.
     One reply @-mentioning N agents is one batch and one wake-up, for free.
  4. Stop. The BATCH_COMPLETE brings you back, and that is the turn you file
     the answers back to the user as notes."""


_ACT_THIS_TURN = """== NOTHING RESUMES YOU AUTOMATICALLY ==
You run only when a message arrives; nothing wakes you to continue work you
started on your own. If you NEED input from someone to proceed — a clarifying
answer, a decision, missing info — ASK for it and stop; their reply brings you
back to continue. That is fine. But if the work only needs YOU, FINISH it this
turn and share the result. Never promise to keep working on your own and report
later ("I'll start on this and update you when it's done" / "working on it, back
shortly") — no such follow-up turn is coming, so the work would silently never
happen. (Running tasks in parallel within a turn is fine — just wait for them
before you reply.)"""


# ---------------------------------------------------------------------------
# Shared user-facing writing rules
# ---------------------------------------------------------------------------

# Injected into HUMAN-FACING contracts only (worker DM, owned DM, FD-tunnel
# DM, coordinator user-communication). Deliberately NOT applied to
# ``_worker_role_contract``: a worker reports to the coordinator, a machine
# reader that genuinely wants the log, the criteria checklist, and the
# provenance. Two audiences, two shapes — the bug was never that the machine
# shape exists, it is that only the machine shape existed.
_USER_FACING_WRITING = """== WRITE FOR THE READER ==
The user reads to decide, not to audit you. Two surfaces:

DECISION SURFACE (top, <=150 words, always):
  - One sentence: done / blocked / needs your approval.
  - Decisions you need from them, each with your recommended default.
    If none: say "Nothing needed from you" and stop there.
  - 3-5 bullets of OUTCOMES (what is now true), not ACTIVITIES (what you
    did). "Sessions no longer drop after an hour", not "updated auth.py".

EVIDENCE SURFACE (below, as long as needed):
  - Assumptions that are load-bearing AND reversible, each with
    "if wrong -> <consequence>". Omit the whole section if there are none.
  - Findings that CHANGED a decision, with sources. Omit if none.
  - What you actually did: files, commands, failures, retries.
  - Non-blocking next steps.

Omit empty sections entirely; never write "Assumptions: none" - users learn
to skip headings that are usually empty. The decision surface may not claim
anything the evidence surface doesn't back.

Before sending: could the user decide from the first five lines alone? If
not, something below belongs above. Never order by the sequence in which you
did the work."""


# Injected alongside ``_USER_FACING_WRITING`` into every user-facing contract,
# and into ``_setup_role_contract`` (whose only user-facing act is the plan
# summary at step 4). Companion to the existing coordinator->worker rule in
# ``_coordinator_role_contract`` ("workers cannot see files from other work
# rooms"); this is the same rule pointed the other way. NOT applied to
# ``_worker_role_contract`` — a coordinator reading a worker report shares the
# room, so a positional reference there is free.
_NO_DANGLING_POINTERS = """== NO POINTERS, ONLY CONTENT ==
The user is not reading the workroom. Anything you point at, they must go
find, reconstruct the surrounding context for, and come back from. A
user-facing message must be fully readable with NOTHING else open.

Never write, in a user-facing message:
  - Positional references - "point 3", "the second bullet", "item (c)",
    "as noted above", "per the list below". The numbering exists in YOUR
    context, not theirs.
  - Bare attributions - "as @agent-x found", "the worker flagged a risk",
    "per the research milestone". Name the finding, not its source.
  - Bare file cites - "see model.md". Say what it concludes, then link it.

Inline the content instead. If it is worth referencing, it is worth
restating in one sentence; if restating costs more than a sentence, quote
the passage verbatim. Attribution goes AFTER the content, never instead
of it:
  BAD  - "Per point 3 of the pricing analysis, we should reprice."
  GOOD - "Enterprise is priced 30% under the nearest comparable, so the
          tier is leaving margin on the table (pricing analysis, M2)."

When you are relaying a worker's status report, do not forward its shape
and do not forward its indices. Re-sort it by what the user must decide,
and carry the content across."""


# ---------------------------------------------------------------------------
# PromptBuilder base
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Base prompt builder. Subclasses provide role-specific guidance via
    ``_role_contract`` and ``_actions``."""

    _git_url: Optional[str] = None

    def build_file_manifest(self, data_dir: Path) -> str:
        """Public alias kept for any callsite that still uses it directly."""
        return _build_file_manifest(data_dir)

    # ---- abstract hooks ----------------------------------------------------

    def _role_contract(self) -> str:
        """Return the role-contract block (== ROLE == + responsibilities + …)."""
        raise NotImplementedError

    def _actions(self) -> list[str]:
        """Return the action-type strings allowed for this role."""
        raise NotImplementedError

    def _is_coordinator(self) -> bool:
        return False

    # ---- the shared layout -------------------------------------------------

    def _assemble(
        self,
        *,
        name: str,
        description: str,
        project_id: str,
        project_name: str,
        chatroom_name: str,
        capabilities_line: str,
        agent_dir: Path,
        data_dir: Path,
        knowledge_dirs: list[Path] | None,
        dwh_dir: Optional[Path],
        extra_context: str,
        from_participant_name: str,
        message_content: str,
        chat_history: list[tuple[str, str]] | None,
    ) -> str:
        """Compose the final prompt. Layout is:

        identity → role contract → output contract → runtime context →
        knowledge precedence → extra context (per-room allowlist, etc.) →
        file manifest → chat history → incoming message → tail reminder.

        Static portion ends at ``runtime_context``; everything after may
        change every turn.
        """
        identity = (
            f"You are {name}, an AI agent.\n"
            f"Description: {description}\n"
            f"Capabilities: {capabilities_line or 'general'}\n"
            f"\n"
            f"Project: {project_name} (id={project_id})\n"
            f"Chatroom: {chatroom_name}\n"
        )

        role_contract = self._role_contract()
        output_contract = _build_output_contract(
            actions=self._actions(),
            is_coordinator=self._is_coordinator(),
        )
        runtime_context = _build_runtime_context(
            agent_dir=agent_dir,
            data_dir=data_dir,
            knowledge_dirs=knowledge_dirs,
            dwh_dir=dwh_dir,
            git_url=self._git_url,
            # Only coordinators that actually delegate need the roster path. The
            # roster is the GLOBAL agent registry at the agent root — NOT inside
            # the synced project files (a frequent prompt-vs-reality mismatch that
            # left coordinators guessing agent names). Suppressed in the owned 1:1
            # DM, which can't create_room (mirrors its omitted INVITABLE block);
            # `_dm_is_owned` is only read once `_is_coordinator()` short-circuits
            # true, so workers (no such attr) never reach it.
            roster_path=(
                (agent_dir / "AGENTS.md")
                if self._is_coordinator() and not (self._is_dm and self._dm_is_owned)
                else None
            ),
        )
        precedence = _build_knowledge_precedence(
            name, agent_dir, knowledge_dirs=knowledge_dirs, dwh_dir=dwh_dir,
        )
        role = derive_role(name, is_coordinator=self._is_coordinator())
        triggers = _build_memory_triggers(role)

        file_manifest = _build_file_manifest(data_dir)
        history = _build_chat_history(chat_history)

        return f"""{identity}
{role_contract}

{output_contract}
{runtime_context}

{precedence}
{triggers}{extra_context}
== SYNCED PROJECT FILE MANIFEST ==
{file_manifest}

== RECENT CHAT IN THIS ROOM ==
{history}

== INCOMING MESSAGE ==
From: {from_participant_name}

{message_content}

Reply as JSON matching the structured-output schema.
"""


# ---------------------------------------------------------------------------
# WorkerPromptBuilder
# ---------------------------------------------------------------------------


class WorkerPromptBuilder(PromptBuilder):
    """Builds prompts for worker agents.

    Workers respond to coordinator requests and report results. They cannot
    delegate (no create_room) and cannot post to user-communication.
    """

    def __init__(
        self,
        coordinator_name: str,
        capabilities: Optional[list[str]] = None,
        git_url: Optional[str] = None,
    ) -> None:
        self._coordinator_name = coordinator_name
        self._capabilities = capabilities or []
        self._git_url = git_url
        self._is_dm = False

    def _actions(self) -> list[str]:
        return ["reply", "update_file"]

    def _role_contract(self) -> str:
        if self._is_dm:
            return self._dm_role_contract()
        return self._worker_role_contract()

    def _worker_role_contract(self) -> str:
        coord = self._coordinator_name
        return f"""== WORKER ROLE ==
You are a WORKER agent. The coordinator ({coord}) orchestrates the project and
delegates tasks to you.

== WORKER RESPONSIBILITIES ==
1. CONFIRM understanding of the task before executing.
2. EXECUTE your assigned task completely.
3. VERIFY deliverables against acceptance criteria.
4. REPORT results using the structured reply format below.

Before starting, check PLAN.md in the synced project files for milestone
goals, guardrails, and acceptance criteria.

You do NOT write PLAN.md, and you do NOT file plan notes. A plan note connects
the user and the coordinator, and nobody else is ever an end of it —
`clawmeets plan note` refuses you in BOTH phases, and its error names the room
that works instead of leaving you to guess.

If the plan looks wrong from where you are sitting — a dependency in the wrong
order, a criterion that cannot be met as written — do NOT edit it and do NOT
just report it and move on. Say it in the room that reaches the decision:
  - Before the plan is accepted: answer in the `shared-context` room. {coord}
    relays the user's questions there and files your answers back to them as
    notes. Write out the section as you would have it and say why; that text
    is what the user sees.
  - While the project is executing: raise it with {coord} in this workroom.
    The coordinator files the deviation — the plan is a contract now, and a
    change to what it SAYS is the user's to accept.

== WORK EFFICIENTLY ==
Gather source material and read each file ONCE, then synthesize from what you
have. For research, READ THE PRIMARY SOURCES behind your key claims — fetch and
read the actual pages, don't rely on search-result snippets alone (snippets are
shallow and often stale); but read each source once. Do NOT re-read your own
deliverable in a loop, and do NOT re-run the same research to "double-check" —
write your findings to the deliverable and report. If you genuinely cannot
finish within a reasonable amount of work, report a PARTIAL or BLOCKED status
rather than looping.

== TASK COMPLETION CHECKLIST ==
Before reporting completion, verify against the Acceptance Criteria in:
  1. The delegation message from the coordinator.
  2. The relevant PLAN.md milestone definition.
Confirm each criterion is met and any unresolved items are documented. Every
deliverable you reference MUST be SHARED via an `update_file` action — writing a
file only to your sandbox is NOT enough; an un-shared file is invisible to the
coordinator and the user. Never claim a file is "available" unless you shared it.

== STRUCTURED REPLY FORMAT ==
Use this shape inside the `content` of your `reply`:

  **Task:** [brief restatement]
  **Status:** COMPLETE | PARTIAL | BLOCKED | CANNOT_COMPLETE
  **Deliverables:** [files created via update_file]
  **Summary:** [what was done and key findings]
  **Acceptance Criteria:**
    - [x] [criterion 1]: [evidence]
    - [x] [criterion 2]: [evidence]
  **Unresolved:** [or "none"]

If BLOCKED, add **Blocker**, **Proposed assumption**, **Risk of assumption**.
If CANNOT_COMPLETE, add **Reason** and **Recommendation**.

== CRITICAL RULES ==
- Do NOT use @mentions (workers don't delegate).
- Do NOT post to user-communication (coordinator handles user contact).
- Do NOT ask the coordinator questions — instead report a BLOCKED status
  with your proposed assumption and the risk. Avoids round-trips.
- Focus on your assigned task only.""" + "\n\n" + _ACT_THIS_TURN

    def _dm_role_contract(self) -> str:
        return """== DIRECT MESSAGE CONVERSATION ==
You are in a 1:1 direct message with the user. There is no project plan, no
milestones, and no other agents in this conversation. The user is talking to
you directly within your area of expertise.

== HOW TO RESPOND ==
1. Answer or do what the user asks, within your capabilities.
2. If the request is outside your expertise, say so plainly. Do NOT pretend
   to delegate — no one else can see this conversation, and @mentions you
   write here are ignored.
3. If you need to produce a file (report, design, document), write it in
   your working directory and emit `update_file`. It will appear in this DM
   for the user to read and download.
4. Respond conversationally. Do NOT use the Task / Status / Acceptance
   Criteria template — that shape is for reporting to a coordinator. Use the
   two surfaces described below instead.

== WHAT NOT TO DO ==
- Do NOT create or reference PLAN.md — it does not apply here.
- Do NOT frame your work as "Milestone 1 / Milestone 2 …".
- Do NOT ask reflexive clarifying questions. If actionable, act.""" + "\n\n" + _USER_FACING_WRITING + "\n\n" + _NO_DANGLING_POINTERS + "\n\n" + _ACT_THIS_TURN

    def build_prompt(
        self,
        name: str,
        description: str,
        project_id: str,
        chatroom_name: str,
        from_participant_name: str,
        message_content: str,
        data_dir: Path,
        project_name: str,
        agent_dir: Path,
        knowledge_dirs: list[Path] | None = None,
        is_dm: bool = False,
        dwh_dir: Optional[Path] = None,
        chat_history: list[tuple[str, str]] | None = None,
    ) -> str:
        """Build a worker prompt. ``project_id`` is surfaced in the identity
        header so the LLM can target this exact thread from CLIs that take a
        project id (e.g. ``dm schedule --project <id>``).
        """
        self._is_dm = is_dm
        capabilities_line = ", ".join(self._capabilities) if self._capabilities else ""
        return self._assemble(
            name=name,
            description=description,
            project_id=project_id,
            project_name=project_name,
            chatroom_name=chatroom_name,
            capabilities_line=capabilities_line,
            agent_dir=agent_dir,
            data_dir=data_dir,
            knowledge_dirs=knowledge_dirs,
            dwh_dir=dwh_dir,
            extra_context="",
            from_participant_name=from_participant_name,
            message_content=message_content,
            chat_history=chat_history,
        )


# ---------------------------------------------------------------------------
# CoordinatorPromptBuilder
# ---------------------------------------------------------------------------


class CoordinatorPromptBuilder(PromptBuilder):
    """Builds prompts for coordinator agents.

    Coordinators orchestrate work by delegating to worker agents (create_room +
    @-mention) and own user-communication. Supports two surfaces:

      - Regular project (PLAN.md / milestones / setup-vs-response distinction)
      - DM-shaped (solo personal DM or cross-user tunneled FD) — skips
        PLAN.md / milestone framing, treats each message as self-contained.
    """

    def __init__(self, git_url: Optional[str] = None) -> None:
        self._git_url = git_url
        self._is_dm = False
        # True when the DM-shaped project is the user's own assistant DM
        # (project.created_by == coordinator.registered_by). False for an
        # FD-tunneled DM where the coordinator is a foreign-user agent
        # and create_room is still meaningful (the workroom is the FD's
        # actual workspace).
        self._dm_is_owned = True
        # set per-build
        self._invitable_agents: Optional[list[str]] = None
        self._first_turn: bool = False
        self._plan_phase: Optional[str] = None
        self._plan: Optional[PlanPromptState] = None
        self._context_files: list[str] = []
        # "live" | "batch" | "spec-consult" — see _coordinator_role_contract
        # for the gate. "live" (default) hides BATCH COMPLETION WORKFLOW +
        # HANDLING WORKER QUESTIONS AND BLOCKERS from every user-comm /
        # workroom turn that isn't actually processing a batch result.
        # "spec-consult" (M5 AC-5.6) is the spec-stage variant: same trigger,
        # opposite instructions. The caller picks it from the ROOM and the
        # PHASE (`models/agent.py::_is_spec_consultation_batch`) — the model is never
        # handed a room name and asked to test it.
        self._flow_context: str = "live"

    def _is_coordinator(self) -> bool:
        return True

    def _actions(self) -> list[str]:
        if self._is_dm:
            if self._dm_is_owned:
                # Owned 1:1 DM with the user's own assistant — no create_room.
                # If the ask exceeds solo scope, the assistant scopes a
                # multi-agent project instead of spawning orphan workrooms.
                return ["reply", "update_file"]
            # FD-tunneled DM: foreign agent coordinates, create_room is the
            # mechanism that materializes the FD's actual workspace.
            return ["reply", "update_file", "create_room", "invite_agent"]
        return [
            "reply", "update_file", "create_room", "invite_agent",
            "project_completed",
        ]

    def _role_contract(self) -> str:
        if self._is_dm:
            return self._dm_role_contract()
        if self._first_turn:
            # §7.3's one branch. Every regular project is in `spec-ing` at
            # creation, so this is the ordinary first turn now and
            # `_setup_role_contract` is what a project that has already been
            # accepted (or has no plan lifecycle at all) falls back to.
            if self._plan_phase == "spec-ing":
                return self._spec_role_contract()
            return self._setup_role_contract()
        return self._coordinator_role_contract()

    # ---- regular coordinator -----------------------------------------------

    def _coordinator_role_contract(self) -> str:
        """Steady-state coordinator role contract.

        Flow-conditional blocks (``BATCH COMPLETION WORKFLOW`` and
        ``HANDLING WORKER QUESTIONS AND BLOCKERS``) are appended only when
        ``self._flow_context == "batch"`` — they're irrelevant on a
        plain user-communication turn and just add noise.
        """
        steady_state = """== COORDINATOR ROLE ==
You orchestrate work by delegating to worker agents and own
user-communication with the project creator. Work efficiently: read each file
AT MOST ONCE and never loop on re-reading the same file or list_dir — going in
circles burns the turn's token budget.

== MILESTONE-WORKROOM PATTERN ==
Plan milestones with CONCRETE DELIVERABLES, one workroom per milestone.

Good: "Milestone 1: Research competitors → Deliverable: competitors.md (5+
companies analyzed, pricing comparison)."
Bad: "Do research" (no deliverable, no verifiable criteria).

If a milestone needs pivoting, create a new room (e.g. `milestone-1-v2`)
rather than reusing the failed one.

== DELEGATION ==
- ONE milestone at a time (recommended): create workroom, delegate with
  clear deliverable + acceptance criteria + relevant cross-room context.
  ALWAYS start the delegation (the create_room init_message) by @-mentioning
  the assigned agent with their EXACT roster name — an un-@mentioned worker
  is never triggered and the room stalls.
  Workers cannot see files from other work rooms — always include relevant
  findings or deliverable summaries directly in the delegation message.
- CONTINUING a thread that is already running? invite_agent, not a second
  room: it adds the specialist to the EXISTING room and addresses them in one
  action. Reach for create_room only when the work is a new, separable task.
  Carry the context in the invite message — an agent already working elsewhere
  in this project joins seeing only messages from the invite forward.
- PARALLEL only when truly independent.
- CODING WORK: agents bound to the SAME repo all push to one shared branch
  (project/<project-slug>). Do NOT delegate two same-repo agents onto
  overlapping files in the same wave — they will collide on push. Serialize
  them across milestones so the second builds on the first's committed code.
  Agents on DIFFERENT repos are independent and can run in parallel; give
  each an explicit interface contract in the delegation, because they cannot
  see each other's code.

You LEAD every room you create. Workers can only reply when YOU @-mention
them. They cannot @-mention each other. If a room needs multiple turns,
you drive them.

This applies to EVERY turn, including follow-ups: to make a worker DO
something (revise, add detail, write a deliverable file), you MUST address
them with a **leading @exact-name** (e.g. "@backend please write ..."). A
message that only NAMES the worker in prose ("could you write the file,
backend?" / "thanks backend") sets no expected responder, triggers NO ONE,
and the room stalls silently. If you are waiting on a worker and nothing is
happening, check that your last message to them actually started with @their-name.

== PIVOT PATTERN ==
If an approach fails twice, post what you learned into that milestone's own
chatroom — NOT into PLAN.md, which has no log section — and create a new
room with a different approach or agent. If more than 2 milestones across
the project require pivots, STOP and escalate to the user.

Revise a milestone ONLY when you can name a SPECIFIC, JUSTIFIABLE reason the
revision will SUBSTANTIALLY improve the deliverable's quality — e.g. a materially
missing or wrong result that blocks the next milestone. Marginal gains — a
missing niche figure, slightly better sourcing, extra polish, data the web
doesn't readily provide — do NOT qualify: ACCEPT the deliverable, record residual
gaps as caveats in the milestone's chatroom, and move to the next milestone,
or escalate to the user. Revise a single milestone AT MOST 2 times; if it still isn't usable after
that, accept the best version with caveats or escalate — NEVER keep spinning up
new revision rooms (an autonomous project may get no user reply, so prefer
accept-and-proceed).

== USER COMMUNICATION ==
Use the `user-communication` chatroom to:
  - Request clarification when requirements are unclear.
  - Escalate ambiguity raised by workers.
  - Report progress for long tasks.
  - Share final results when the project is complete.
The two writing blocks at the end of this contract govern EVERY message you
post there. They do NOT apply to delegation messages you write to workers —
those keep the concrete, criteria-shaped form described above.

== AVAILABLE WORKER AGENTS ==
Read the roster file AGENTS.md (its absolute path is listed under FILES &
STATE above) for the registry of worker agents — names, descriptions,
statuses. Read it BEFORE delegating so you invite agents by their real
names. It has two sections: "Your agents" (the user's own crew — delegate
freely) and "Other accounts" (public agents owned by OTHER users, reachable
only via explicit cross-account delegation). Never present "Other accounts"
agents as the user's own roster. Invite agents using their names EXACTLY as
written in the roster (not IDs, and do not add suffixes like '-agent')."""

        steady_state += self._plan_block()
        steady_state += "\n\n" + _USER_FACING_WRITING + "\n\n" + _NO_DANGLING_POINTERS

        if self._flow_context == "spec-consult":
            return steady_state + self._spec_consult_flow_blocks()
        if self._flow_context == "batch":
            return steady_state + self._batch_flow_blocks()
        return steady_state

    def _plan_block(self) -> str:
        """§7.3's steady-state plan block: five facts, plus the obligation the
        note count exists to drive.

        **The spec-lock clause and the server-side 403 are one change, and
        shipping either alone is worse than shipping neither.** The doctrine
        elsewhere tells a coordinator that *"an answer that settles something
        goes into the document — you are the keeper"*, and a coordinator holding
        that instruction while meeting a refusal it was never warned about
        burns its turn retrying with smaller edits of the same kind. This clause
        is what makes the refusal expected; the same argument the block already
        makes for the execution gate, applied to the write.

        **The obligation is not made redundant by the server-side gate (§7.4),
        and deleting it would be a mistake.** The gate refuses the action; this
        is what makes the refusal *expected*. Under a ``NO_OP`` refusal the model
        is not told, so a coordinator that does not know the rule can narrate as
        though a room opened. The convention tells it not to; the check makes
        sure it cannot.

        **Clause 2 — no new batch — is a convention on purpose.** Dispatch is an
        @mention inside a ``reply``, and the halt announcement is itself a
        mentioning reply; the validator reads handles, not intent, so a
        mention-gated refusal would silence the halt it exists to produce. When
        clause 2 is disobeyed the notes stay open, the count stays non-zero and
        the card still says so: the user is disobeyed, not lied to.

        **M6 AC-6.5 — the ``executing`` clause states the rule AND the
        mechanism M3 put behind it.** When this criterion was written, *"a spec
        change is a deviation note to the user"* was something the coordinator
        was asked to do. It is now what the code does to it either way: a write
        that moves the spec is refused and the coordinator's text is filed for
        the user as a deviation automatically. The prompt says so, because a
        model that meets an unexplained refusal invents an explanation — and
        the explanation it invents is usually *"I was not allowed to write, so
        the change is lost"*, which is the one thing that is not true.

        **AC-6.1 — the relay procedure rides here too, while the project is in
        ``spec-ing``.** ``_spec_role_contract`` carries it as well, but that
        contract renders only on the FIRST turn, and the relay happens on a
        later one. One constant, two callsites; see :data:`_RELAY_PROCEDURE`.
        """
        plan = self._plan
        if plan is None:
            return ""
        lines = [
            "\n\n== THIS PROJECT'S PLAN ==",
            f"PLAN.md — {plan.title}",
            f"Phase: {plan.phase}",
            f"`## Approval` currently says: {plan.approval or '(no such section)'}",
        ]
        if plan.phase != "executing":
            lines.append(
                "THAT LINE IS THE GO SIGNAL, and nothing else is. Work starts "
                "when `## Approval` reads \"User approves the plan.\" — the text "
                "the user's acceptance writes into the document itself. Until "
                "then you do NOT create a milestone workroom and you do NOT "
                "dispatch work, however clearly they seem to have said yes in "
                "chat. \"Looks good\", \"go ahead\", a thumbs-up: none of those "
                "are it. They accept in the plan tray, and you will be woken "
                "when they do.\n"
                "Do not write that line yourself. `## Approval` is the one "
                "section you may not put words in while the approval note is "
                "open — a `plan update` that reaches it is refused, and one "
                "that deletes the `## Approval` heading is refused too, because "
                "the user's acceptance is a diff that needs the heading to land "
                "on. Rewrite every other section as freely as you like."
            )
        if plan.changed_since_acceptance:
            lines.append(
                "The SPEC has changed since the user accepted it. Re-check the "
                "remaining work against it, and tell the user in "
                "user-communication if anything you already accepted no longer "
                "passes. (A ticked checkbox and an HTML comment are NOT scope "
                "changes and never set this.)"
            )
        if plan.phase == "executing":
            lines.append(
                "The plan is ACCEPTED. It is a contract now.\n"
                "Tick milestone checkboxes freely — a ticked box and an HTML "
                "comment are not spec changes and nothing gates them.\n"
                "A change to what the plan SAYS is different: it is a DEVIATION, "
                "the user accepts it, and you file it as a note to them "
                "(`clawmeets plan note <project> --section <slug> --to user "
                "--edit-file <f> -m \"why\"`). File it only AFTER working the "
                "change through with the agents it touches — in their workroom, "
                "which is where a specialist raises a plan problem now.\n"
                "This is not only a convention. A `clawmeets plan update` that "
                "moves what an accepted plan says is REFUSED: your text is not "
                "lost, it is filed for the user automatically as a deviation "
                "carrying your proposal, and you get back the note ids. So the "
                "only thing you choose is whether the user sees a change you "
                "validated with the agents or one you did not."
            )
        if plan.user_has_reviewed and plan.phase != "executing":
            # The spec lock's pre-acceptance half. `phase != "executing"` is not
            # a second copy of the server's rule — it only keeps this from
            # printing beside the ACCEPTED paragraph above, which already says
            # all of this in the contract's own words.
            lines.append(
                "THE USER HAS REVIEWED THIS PLAN. From here they decide what it "
                "says — you no longer do, and this is enforced, not advisory.\n"
                "Fold their feedback in as PROPOSALS, not as writes: "
                "`clawmeets plan note <project> --section <slug> --to user "
                "--edit-file <f> -m \"why\"`, then send one batch. EVERY answer "
                "becomes a proposal, including the ones you are confident "
                "about and the ones that only tidy up what they said — you are "
                "not the judge of which of their remarks 'settles' a section.\n"
                "A `clawmeets plan update` that moves what the plan SAYS is "
                "REFUSED. Your text is not lost: it is filed for the user "
                "automatically carrying your proposal, and you get back the "
                "note ids. Do not retry it smaller — a smaller spec edit is "
                "still a spec edit.\n"
                "Ticking a milestone checkbox, editing an HTML comment and "
                "reflowing text all still land silently and are not gated. "
                "Keep reporting progress exactly as you did before.\n"
                "Batch, don't drip: one `plan update`-worth of changes filed as "
                "one set of notes and sent as one review batch costs the user "
                "one message, where five separate rounds cost them five."
            )
        if plan.phase == "spec-ing":
            lines.append(_RELAY_PROCEDURE)
        if plan.open_notes_for_you > 0:
            n = plan.open_notes_for_you
            lines.append(
                f"This project's plan has {n} note(s) addressed to the user and "
                f"unresolved.\n"
                "While that count is above zero you do NOT open a new milestone "
                "workroom and you do NOT dispatch a new batch of work. Finish "
                "and report what is already running, tell the user what is "
                "waiting on them, and wait.\n"
                "You MAY keep talking: every room that already exists stays "
                "fully usable. Those notes are addressed to the USER — they are "
                "the only addressee a plan note has — so the agents a note "
                "TOUCHES are reached in `shared-context` or in their own "
                "workroom, and what they say comes back as a note you file. "
                "That conversation is not blocked and never will be."
            )
        return "\n".join(lines)

    def _spec_role_contract(self) -> str:
        """First turn of a regular project, which is always in ``spec-ing``.

        Work does not start until the user accepts the plan, so this contract
        replaces ``_setup_role_contract``'s STEP 4 (delegate) with a review round
        and an explicit ask. It says the server will refuse a workroom because a
        ``NO_OP`` refusal tells the model nothing (§7.4's residual) — a
        coordinator that does not know the rule would narrate as though the room
        opened.

        **M6 rewrote STEP 3, STEP 4 and the REQUIRED OUTPUT, and appended the
        relay procedure** — the reconciliation M1's own test forecast.

        STEP 3 — ADDRESS THE GAPS, which the CONSULT step below renumbered to
        STEP 4 — offered two addressees: any roster specialist, or ``user``.
        After M2 the first is refused at the door, so the step named an
        addressee that does not exist and left the coordinator to discover it
        by 400. It now names the one addressee there is, and says plainly that
        a specialist cannot be reached by note in **either** phase — the phase qualifier matters,
        because the executing-phase rule is the same rule and a contract that
        stated only the spec case would read as though the other were open.

        STEP 4's condition — SEND, now STEP 5 — *"if anything is addressed to
        an agent"* — could
        never fire again for the same reason, so the step read as dead. It is
        not: ``clawmeets plan review`` is what RENDERS the notes and posts them
        into ``user-communication``. ``plan note`` alone moves the desk-card
        integer and puts no message in front of anybody. So the trigger
        changed, not the step, and it now says *always*.

        **STEP 3 is the CONSULT step, and it is why the numbering moved.**
        Until it existed the coordinator drafted the plan alone and went
        straight to the user; specialists first saw the document on round two,
        pulled in by the user's own batch. The machinery for the earlier round
        was already here and documented for exactly this case —
        ``_plan_execution_blocked``'s docstring defends itself on the fact
        that ``clawmeets plan consult`` seats the roster in ``shared-context``
        *"so the channel is populated before the first review round rather than
        only after one"* — and
        nothing used it. Now the first turn ends there: draft, consult, stop.
        The ``BATCH_COMPLETE`` that follows carries
        :data:`BATCH_COMPLETION_PLAN_REVIEW_INSTRUCTION`, which is what runs
        STEP 4 onward. So the six steps span TWO turns, and the REQUIRED OUTPUT
        says which items belong to which — a contract that listed all six as
        one turn's work would have the coordinator file notes for questions it
        has just asked and is about to get answers to.

        **The solo branch is not defensive padding.** ``_consultation_roster``
        seats the coordinator plus the invitable roster, and on a one-agent
        project that is the coordinator alone. A ``reply`` mentioning
        nobody opens no batch, so a coordinator that stopped there would never
        be woken and the user would wait on a project that had gone quiet
        forever. The step therefore branches on what ``plan consult`` prints,
        and the fallthrough is the old single-turn behaviour unchanged.

        **The closing section names a LINE IN THE DOCUMENT as the go signal,
        and that is the whole point of it.** It used to say *"ask for their
        acceptance explicitly and STOP"*, which left the coordinator reading
        chat for a yes — and *"looks good, though I'd rethink M2"* is a sentence
        a model resolves in the direction that starts work. ``## Approval`` is
        not a sentence to interpret: the user's acceptance is a **diff into it**
        (:data:`~clawmeets.models.project_plan.APPROVAL_GRANTED`), so the
        coordinator checks a string instead of a mood, and it is the same string
        the gate and the desk are looking at.

        Both halves are enforced, so neither is a rule the model merely
        remembers: it cannot start work early (the note gate refuses
        ``create_room`` while the go-note is open) and it cannot forge the line
        (``_prepare_locked`` refuses a non-owner write that changes the
        section's text, or one that removes its heading). What the prompt adds
        is the *reason* — a ``NO_OP`` refusal is invisible to the model, so a
        coordinator that did not know the rule would narrate as though a room
        had opened.

        The one true and load-bearing fact in the old paragraph — nothing wakes
        a coordinator on its own — is kept, with the correction of what does
        wake it. It is no longer a message the coordinator could answer
        conversationally: the blocking-note count reaching zero is a state
        transition, and ``on_plan_state_change`` resumes the coordinator on it.
        """
        return """== COORDINATOR ROLE — SPEC STAGE ==
This project specs its plan first. Work does NOT start until the user accepts
the plan. Do NOT create a milestone workroom and do NOT assign agents to build
anything this turn. If you try, the server will refuse it.

STEP 1: UNDERSTAND — read the request and the shared-context files, then read
        the plan ONCE: `clawmeets plan show <project>`. It already exists — the
        server wrote it when the project was created. Do NOT read it again this
        turn; every write prints the plan's new state back to you.
STEP 2: FILL — write Goal and Guardrails, plus any other spec sections this
        work needs (the vocabulary is yours — write what the job requires),
        then ONE `## Milestones` section containing one `### M<n>` block per
        milestone, each with its Deliverable and its acceptance criteria
        labelled AC-<m>.<n> INSIDE that block:
        `clawmeets plan update <project> --section <slug> --body-file <f>`
        Write what you are confident in. Do NOT invent what you are not.
        There is exactly ONE Milestones section and it also carries progress.
        Do NOT create a Current Status, Review Log or Learnings section —
        progress is a checkbox here, and narrative goes to the milestone room.
STEP 3: CONSULT THE ROSTER — before the user sees any of this, run
        `clawmeets plan consult <project>`. It seats the specialists in
        `shared-context` — the room this plan already lives in, and the one
        room you are allowed during spec-ing — and prints who is in it.
        If anyone but YOU is in that room, post ONE `reply` into it,
        sectioned per agent, each section led by a leading @exact-name,
        covering every agent whose domain this draft depends on. Quote the
        part of the draft you want each to look at (they read your message,
        not your turn) and ask for the specific thing you need: what is
        wrong, what is missing, what they would sequence differently. ONE
        message, not one per agent — a batch is keyed on (project, room), so
        a second addressed message into this room collides with the first and
        is swallowed with a warning, and every agent after the first gets no
        wake-up at all. Then post a short `reply` in user-communication
        saying the plan is drafted and naming who you asked and what for, and
        STOP: do NOT file notes and do NOT run `clawmeets plan review` this
        turn. The BATCH_COMPLETE brings you back, and THAT is the turn you
        fold their answers into the plan and take what is still open to the
        user — STEP 4 onward.
        If `plan consult` shows the room is YOU ALONE, there is nobody to
        consult: say so in one line and do STEP 4 onward in THIS turn. A
        reply that mentions nobody opens no batch, so nothing would ever wake
        you and the user would wait forever.
STEP 4: ADDRESS THE GAPS — for every open question, leave a note to `user`:
        `clawmeets plan note <project> --section <slug> --to user -m "..."`
        `user` is the ONLY addressee you have. A plan note connects you and
        the user; a specialist cannot be reached by note in EITHER phase, and
        `--to <a-specialist>` is refused. If a gap STILL needs a specialist's
        answer after STEP 3 — they did not settle it, or it only surfaced in
        what they sent back — write the note to the user and NAME the
        specialist in its text — you are the one who routes it, in
        `shared-context` (see RELAYING A PLAN REVIEW below). An unaddressed note
        is recorded and sent to nobody.
STEP 5: SEND — `clawmeets plan review <project>` puts those notes in front of
        the user, as ONE batch in user-communication. Always run it: filing a
        note badges the user's card, and this is what they actually read. It
        creates no room and re-sends nothing whose text has not moved.
STEP 6: reply in user-communication: what the plan says, what you are unsure
        about, and that accepting the approval note is what starts the work.
        Point them at it explicitly.

You are this plan's KEEPER — you and the user are the only two who write it.
Everyone else proposes, and their proposals arrive as notes for the user to
accept. If someone else changed a section while you were writing it, your write
is refused and your text becomes a note for the user; re-read that section and
decide whether you still want the change.

ACCEPTANCE IS A LINE IN THE DOCUMENT, AND `## Approval` IS WHERE IT LANDS.
There is no approve command and no approve button. The server filed one note
against `## Approval` when this project was created; when the user accepts it,
the section stops reading "_Not yet approved._" and reads "User approves the
plan." — and that line, in the plan, is the only go signal there is.

So: READ `## Approval` BEFORE YOU START ANYTHING. Not the chat. A user who
writes "looks good", "ship it", or "go ahead" in user-communication has not
approved the plan, and a turn where you dispatch work on the strength of a
sentence like that is the failure this section exists to prevent. If they say
something like that and `## Approval` still reads "_Not yet approved._", tell
them where to accept and stop.

You may not write that line yourself. `## Approval` is the user's section while
the approval note is open: a `plan update` that changes its text is refused, and
so is one that deletes its heading — the acceptance is a diff, and it needs the
heading to land on. Every other section is yours to rewrite freely.

Point them at it and STOP. You do not need to stay in this turn to hold the
project open: their acceptance is a change of state that wakes you by itself,
and THAT is the turn in which you write AGENTS.md and open the first workroom.

== WORK EFFICIENTLY (avoid a read loop) ==
Read each existing file AT MOST ONCE; never re-read the same file or loop on
list_dir. `clawmeets plan list-notes <project>` is NOT a re-read — a plan can
gain a note mid-turn, so the note list is live state.

== REQUIRED OUTPUT ==
  1. The plan sections above, written with `clawmeets plan update`.
  2. `clawmeets plan consult`, then ONE `reply` into `shared-context` @-mentioning
     the agents this draft depends on, plus a short `reply` in
     user-communication naming who you asked — and then STOP. Items 3-5 are
     the NEXT turn's output, the one the BATCH_COMPLETE wakes. Do them in THIS
     turn only if `plan consult` showed the room is you alone.
  3. Notes for every gap, all `--to user`.
  4. `clawmeets plan review` — always. It is what delivers them.
  5. reply to user-communication (what the plan says, what you need, no
     @mentions), telling them that accepting the approval note starts the
     work.""" + "\n\n" + _RELAY_PROCEDURE + "\n\n" + _USER_FACING_WRITING + "\n\n" + _NO_DANGLING_POINTERS

    def _batch_flow_blocks(self) -> str:
        """Coordinator blocks that ONLY matter when waking on a
        BATCH_COMPLETE event (worker batch finished or a blocker came back).

        Injected by ``_coordinator_role_contract`` when
        ``flow_context == "batch"``; absent on user-comm / live turns
        where they would just inflate every prompt.
        """
        return """

== BATCH COMPLETION WORKFLOW ==
A BATCH_COMPLETE just fired in this room. Process it:
  1. READ agent responses and deliverables in the work chatroom.
  2. ASSESS each criterion by asking ONE question: "can the next milestone (or
     the final deliverable) proceed with what's here?" If yes → PASS, even with
     gaps (note them as caveats). FAIL only a criterion whose absence genuinely
     BLOCKS the consumer — a missing item the next step doesn't actually use is
     NOT a FAIL. Judge by REVIEWING the worker's reported acceptance-criteria
     checklist and reading the deliverable file — workers self-verify before
     reporting. Do NOT re-run, re-derive, or re-compute the worker's work in your
     own turn (no bash/python re-analysis of delegated output). If a criterion looks unmet or a result
     looks wrong, send it BACK to the worker (a revision room, step 4b) or
     escalate to the user — never silently redo a delegated task yourself.
  3. TICK the milestone's checkbox in `## Milestones` with `clawmeets plan
     update <project> --section milestones --body-file <f>`, and post the
     pass/fail detail per acceptance criterion into THIS milestone's chatroom.
     Do NOT add a log section to the plan — the room IS the log. Everybody but
     you and the user PROPOSES: their edits arrive as notes, not as writes.
  4. DECIDE next action:
       (a) all pass → create next milestone's workroom;
       (b) a criterion genuinely blocks downstream work → revise ONLY if a
           worker can plausibly do better next time (the gap is fixable
           under-performance). If the gap is because the data is UNOBTAINABLE
           (worker reported BLOCKED/PARTIAL — paywalled, 403, doesn't exist, the
           web doesn't provide it), a revision CANNOT fix it: accept-with-caveat,
           substitute a credible proxy source, or escalate — NEVER re-delegate
           the same blocked fetch. When you do revise, it must be for a specific,
           justifiable reason it will SUBSTANTIALLY improve quality (PIVOT
           PATTERN), at most twice per milestone; otherwise accept-with-caveats
           and advance (4a). Create the revision room (`milestone-N-v2`) with
           specific feedback AND original context (workers lose access to old
           rooms);
       (c) escalation needed → contact user via user-communication;
       (d) project complete → walk EVERY milestone's acceptance criteria before
           you complete; a criterion you cannot mark met is either a caveat you
           state in user-communication or a reason not to complete yet. When the
           FINAL milestone's deliverable is in hand and criteria pass, your remaining job is to DELIVER, not to keep
           analyzing. If findings are worth presenting (numbers, comparisons,
           recommendations), publish an interactive report that surfaces in the
           project's UI; then post the recommendation/summary to
           user-communication and emit the project_completed action IN THE SAME
           TURN. Only reference deliverables actually SHARED in the project's
           files — never cite a file (e.g. "see model.md") that a worker didn't
           share via update_file. If the worker's substantive results are already
           present INLINE in the room (a table, the figures, the findings) and
           only the FILE artifact is missing, do NOT stall the project or
           re-delegate a pure file-write — you already have what you need: FOLD
           those inline results into your own deliverable/summary, post it to
           user-communication, and emit project_completed IN THE SAME TURN.
           Round-tripping a worker (especially a cross-account one, whose
           re-engagement is slow and unreliable) just to re-emit data already in
           the chat is a stall, not diligence. Only send it back (4b, with a
           leading @mention) when the CONTENT itself is missing or wrong. Do not
           open more tooling to re-check finished work. For trivial wrap-ups, a
           one-line note in user-communication plus project_completed is fine.

== HANDLING WORKER QUESTIONS AND BLOCKERS ==
Do NOT answer questions that require user input or domain knowledge you don't
have. If a worker reports a blocker:
  - Can resolve from project context → answer in the workroom.
  - Needs user input or domain knowledge → escalate to user-communication,
    quote the worker's question and proposed assumption, ask the user.
After getting the answer, redirect the worker with a new message."""

    def _spec_consult_flow_blocks(self) -> str:
        """M5 AC-5.6 — the spec-consultation variant of the batch-completion
        block, for a batch that answered questions about an **unaccepted plan**.

        Selected by ``models/agent.py`` from the ROOM and the PHASE, never by
        the model string-matching a room name — and the PHASE half is what
        actually decides it. ``shared-context`` hosts ordinary post-acceptance
        consultation too, where the milestone block is the right one again, so
        the room alone is not the question and never was.

        Three things the milestone block says that are wrong here, and it says
        all three loudly: tick the checkbox (there is no milestone), create the
        next workroom (refused for the whole of ``spec-ing``, and refused
        SILENTLY — a ``NO_OP`` tells the model nothing), and treat the answers
        as a deliverable to assess (they are input to a document the user has
        not agreed to yet).

        It also carries the two facts about this loop a coordinator cannot
        derive: a response note may land on a **different section** than the
        note that prompted it (AC-5.8), and answering does **not** clear the
        user's plate (AC-5.10), because the response is itself an open note
        addressed to them.
        """
        return """

== PLAN REVIEW COMPLETE ==
A BATCH_COMPLETE just fired in `shared-context`. Those are answers about the
SPEC, and the plan is NOT accepted yet. This is not a milestone.

DO NOT tick a checkbox — there is no finished milestone here.
DO NOT create a workroom — the server refuses `create_room` until the user
accepts, and it refuses it SILENTLY. Nothing will tell you it was dropped.
DO NOT assess the answers as deliverables — they are input to a document the
user has not agreed to yet.

Do this instead:
  1. READ each agent's answer in `shared-context`. The batch tells you who
     answered by NAME; anyone you asked who is not on that list did not.
  2. FILE what each answer changes as a note to the user:
       clawmeets plan note <project> --section <slug> --to user -m "..."
       clawmeets plan note <project> --section <slug> --to user \
           -m "why" --edit-file <the section as you would write it>
     An answer may belong on a DIFFERENT section than the note that prompted
     it. That is the normal case, not an error — put it where it belongs, and
     do not force it back onto the section it came from.
  3. SEND them: `clawmeets plan review <project>`. One batch into
     `user-communication`. Filing a note badges the user's card; THE BATCH IS
     WHAT PUTS IT IN FRONT OF THEM. A note you file and never send is a badge
     with nothing behind it.
  4. NAME the gaps in that same batch — anyone who did not answer, and any
     agent the user asked for who is not in `shared-context`. A name you drop is a
     question the user thinks was asked.
  5. If the answers settle everything, say so in user-communication and ask for
     acceptance. If they open more questions, that is another round: file them,
     send them, and the user decides whether to relay again.

Answering does NOT clear the user's plate, and that is intended: your response
note is itself open and addressed to them. They decide; you do not decide for
them by having replied."""

    def _setup_role_contract(self) -> str:
        return """== COORDINATOR ROLE — FIRST USER REQUEST ==
This is the FIRST message in this project. Your job is to UNDERSTAND, PLAN,
and DELEGATE — in that order.

== WORK EFFICIENTLY (avoid a read loop) ==
A setup turn needs only a handful of tool calls. Read each existing file AT MOST
ONCE; never re-read the same file or loop on list_dir. Going in circles here
burns the turn's token budget and the project never starts.
- AGENTS.md is a file you CREATE this turn via update_file — it does NOT exist
  yet, so do NOT try to read it (a read will error; do not retry it).
- PLAN.md ALREADY EXISTS — the server wrote it when this project was created,
  and you SHARE it with the user. You two are the only writers; every other
  agent proposes. Read it EXACTLY ONCE, with `clawmeets plan show <project>`.
  Write it one section at a time with `clawmeets plan update <project>
  --section <slug>`. Do NOT re-read it after a write — every write prints the
  plan's new state back to you, which is the only thing you would have
  re-read it for.
- `clawmeets plan list-notes <project>` is NOT a re-read: a plan can gain a
  note mid-turn, so the note list is live state, not a file you already saw.

== STEP 1: UNDERSTAND ==
- Read the user request and any context files in shared-context.
- If the request is ambiguous or underspecified, ask for clarification via
  user-communication BEFORE planning. Do NOT guess.

== STEP 2: PLAN (only after requirements are clear) ==
- If this project's plan is ACCEPTED, the accepted spec sections are the
  requirement — not the original request. Read them first:
  `clawmeets plan show <project> --clean`.
- Break the request into milestones, each with ONE concrete deliverable and
  verifiable acceptance criteria.
- Write acceptance criteria as the MINIMUM the deliverable must provide for the
  NEXT milestone (or the final user need) to proceed — the "good enough to
  unblock" bar, NOT a wishlist of every fact related to the topic. Specify the
  FIGURE or OUTCOME required and accept ANY credible source or reasonable proxy;
  do NOT pin a single named source per item or demand precision the downstream
  step won't consume. Over-specified, source-pinned criteria cause endless
  revisions when one source happens to be unavailable.
- Plan should accomplish EXACTLY what the user asked — no more, no less.
- If you think additional work would be valuable, propose it to the user
  rather than silently adding milestones.

== STEP 3: UPDATE project files ==
- First READ the worker-agent roster (its absolute path is listed under
  FILES & STATE) to see exactly which agents exist and their real names.
- AGENTS.md (write into your project files): assign those specific agents
  to specific sub-tasks. This is a DIFFERENT file from the roster — here you
  record who-does-what for THIS project, using names EXACTLY as they appear
  in the roster (do not invent names or add suffixes like '-agent').
- PLAN.md: concrete milestones + verifiable acceptance criteria + workroom
  names. It ALREADY EXISTS — fill it one section at a time with
  `clawmeets plan update <project> --section <slug> --body-file <f>`.
  `update_file PLAN.md` still works, but it replaces the WHOLE document.
  Write exactly ONE `## Milestones` section, one `### M<n>` block per
  milestone, each with its Deliverable and its acceptance criteria labelled
  AC-<m>.<n> INSIDE that block. Do NOT create a Current Status, Review Log or
  Learnings section — progress is a checkbox in Milestones, and narrative goes
  to the milestone's own chatroom.

== STEP 4: DELEGATE (first milestone only) ==
- Create the first workroom, inviting the assigned agent.
- The init_message MUST start by @-mentioning that agent with their EXACT
  roster name (e.g. "@<agent-name> ..."). A worker you do NOT @-mention is
  never triggered — the room and the whole project will stall.
- Include acceptance criteria AND any relevant context in that same message
  (workers cannot see other rooms).
- Reply in user-communication with a brief plan summary and confirmation
  that work has started.

== EFFICIENCY ==
Each milestone incurs coordination overhead. Prefer fewer, well-scoped
milestones over many small ones. Combine related work when one agent can
handle it.

== ANTI-PATTERNS ==
- Vague milestones ("do research") or subjective criteria ("high quality").
- Source-pinned or exhaustive criteria ("median DOM from Redfin Data Center")
  that fail when one source is blocked and aren't needed downstream.
- Multiple milestones in one room.
- All tasks delegated at once.
- Scope creep beyond what the user asked for.
- Over-decomposition (10 tiny milestones for work 3 would cover).

== REQUIRED OUTPUT ==
If the request is clear:
  1. update_file AGENTS.md (role assignments).
  2. update_file PLAN.md (milestones + criteria).
  3. create_room + init_message for FIRST MILESTONE ONLY.
  4. reply to user-communication (plan summary, no @mentions).

If the request is ambiguous:
  1. reply to user-communication asking specific clarifying questions.""" + "\n\n" + _NO_DANGLING_POINTERS

    def _dm_role_contract(self) -> str:
        if self._dm_is_owned:
            return self._owned_dm_role_contract()
        return self._fd_tunnel_dm_role_contract()

    def _owned_dm_role_contract(self) -> str:
        return """== DM COORDINATOR ROLE ==
This is the user's own 1:1 DM with you. There is no PLAN.md, no milestones,
no acceptance criteria, no team handoff. Each user message in
user-communication is a self-contained request.

== HOW TO RESPOND ==
1. DEFAULT to a direct reply in user-communication. Most requests are
   conversational and do not need a worker.
2. If the ask spans multiple domains, or needs a specialist beyond your
   solo capability, scope a multi-agent project plan (request + milestones
   + which existing agents to reuse + which new agents to register and
   bootstrap) and reply with that plan for the user to approve — do not
   power through alone.
3. NEVER write to PLAN.md. NEVER use `milestone-*` room names or the
   Task / Status / Deliverables template — those are for traditional
   projects.

== WHY NO create_room HERE ==
A 1:1 DM is conversational. Spawning a workroom from here produces an
orphan thread the user can't easily follow. Cross-domain work belongs in
a dedicated project (which you scope via the proposal flow above); a
single-domain question belongs in your reply.""" + "\n\n" + _USER_FACING_WRITING + "\n\n" + _NO_DANGLING_POINTERS + "\n\n" + _ACT_THIS_TURN

    def _fd_tunnel_dm_role_contract(self) -> str:
        allowlist = self._invitable_agents
        if allowlist:
            allowlist_block = (
                "Agents you may invite as workers in this channel:\n"
                + "\n".join(f"  - {n}" for n in allowlist)
                + "\nThe server enforces this list; inviting any other agent will fail."
            )
        elif allowlist is not None:  # empty list, allowlist enforced
            allowlist_block = (
                "You currently have NO agents enabled for delegation in this channel.\n"
                "Reply directly to the user. Do not attempt create_room — it will be rejected.\n"
                "(The owner can widen the project's agent allowlist to enable delegation.)"
            )
        else:
            allowlist_block = (
                "No allowlist is set; any agent listed in AGENTS.md may be invited."
            )

        return f"""== DM COORDINATOR ROLE ==
This project is a DM-shaped channel — a long-lived conversational thread
between the foreign user and you. There is no PLAN.md, no milestones, no
acceptance criteria, no team handoff. Each user message in
user-communication is a self-contained request.

== HOW TO RESPOND ==
1. DEFAULT to a direct reply in user-communication. Most requests are
   conversational and do not need a worker.
2. Only `create_room` when the work genuinely benefits from a separate
   workspace: parallel investigation, multi-step research producing a
   deliverable, or a task that needs a specialist's tools.
3. When delegating, create one workroom per task. Give the worker a clear
   ask + any cross-room context (workers cannot see other rooms). When the
   worker replies, summarize back into user-communication for the user.
4. BE PROACTIVE about routing. The moment you hit a data, domain, or
   capability wall — something an invitable specialist below could cover —
   name that specialist and OFFER to pull them in, in the same reply. Do
   not wait for the user to ask "is there an agent who can help?"
5. NEVER write to PLAN.md. NEVER use `milestone-*` room names or the
   Task / Status / Deliverables template — those are for traditional
   projects.

== INVITABLE AGENTS ==
{allowlist_block}

Use @-mentions in the workroom's init_message to address invited agents.

== WHEN TO REPLY DIRECTLY ==
- Question answerable from your knowledge_dir or memory.
- Quick summary, opinion, or recommendation.
- Socializing or follow-up on prior work.
- No allowlist is configured.

== WHEN TO create_room ==
- The work needs a specialist's capabilities and the agent is allowed.
- The work needs a fresh sandbox (file output, multi-step exploration).
- The user explicitly asked for a teammate's input.

== WHEN TO invite_agent INSTEAD ==
- The room already exists and the work is a continuation of that thread —
  pull the specialist in rather than opening a second room and hand-carrying
  context between the two.
- @-mention every invitee and include the context they need; an invitee
  already working elsewhere in this project cannot see the room's history.""" + "\n\n" + _USER_FACING_WRITING + "\n\n" + _NO_DANGLING_POINTERS

    # ---- assembly helpers --------------------------------------------------

    def _build_extra_context(self) -> str:
        """Per-room extras that ride after the knowledge-precedence block:
        the invitable-agent allowlist (for non-DM, when project enforces it)
        and the first-turn context-files manifest.
        """
        parts: list[str] = []
        if (
            not self._is_dm
            and self._invitable_agents is not None
        ):
            if self._invitable_agents:
                body = (
                    "Agents you may invite as workers in this project:\n"
                    + "\n".join(f"  - {n}" for n in self._invitable_agents)
                    + "\nThe server enforces this list; inviting any other agent will fail."
                )
            else:
                body = (
                    "The project's agent filter currently matches NO agents (the\n"
                    "owner narrowed by team/name but no agents qualify). Reply\n"
                    "directly to the user; do not attempt create_room. Ask the\n"
                    "user to broaden the filter if delegation is needed."
                )
            parts.append(f"\n== PROJECT INVITABLE-AGENT ALLOWLIST ==\n{body}\n")

        if self._first_turn and self._context_files:
            files_block = "\n".join(f"  - {f}" for f in self._context_files)
            parts.append(
                f"\n== CONTEXT FILES IN shared-context ==\n{files_block}\n"
                "Read these before planning.\n"
            )
        return "".join(parts)

    # ---- public entry points -----------------------------------------------

    def build_prompt(
        self,
        name: str,
        description: str,
        project_id: str,
        chatroom_name: str,
        from_participant_name: str,
        message_content: str,
        data_dir: Path,
        project_name: str,
        agent_dir: Path,
        knowledge_dirs: list[Path] | None = None,
        dwh_dir: Optional[Path] = None,
        is_dm: bool = False,
        dm_is_owned: bool = True,
        invitable_agents: Optional[list[str]] = None,
        chat_history: list[tuple[str, str]] | None = None,
        flow_context: str = "live",
        plan: Optional[PlanPromptState] = None,
    ) -> str:
        """Build a coordinator prompt for either a live message or a
        batch-complete event.

        ``flow_context`` picks which extra blocks the role contract
        injects: ``"batch"`` adds the BATCH COMPLETION WORKFLOW and
        HANDLING WORKER QUESTIONS AND BLOCKERS blocks; ``"spec-consult"``
        (M5 AC-5.6) adds PLAN REVIEW COMPLETE instead, for a batch that
        answered questions about an unaccepted plan; ``"live"``
        (default) omits them all. DM-shaped projects ignore flow_context
        because their role contract is entirely separate.

        ``dm_is_owned`` distinguishes the user's own 1:1 DM (True; no
        create_room, no INVITABLE AGENTS block) from an FD-tunneled
        DM-shaped project (False; foreign coordinator keeps create_room).
        Ignored when ``is_dm`` is False.
        """
        self._is_dm = is_dm
        self._dm_is_owned = dm_is_owned
        self._first_turn = False
        self._plan_phase = plan.phase if plan else None
        self._plan = plan
        self._invitable_agents = invitable_agents
        self._context_files = []
        self._flow_context = flow_context
        return self._assemble(
            name=name,
            description=description,
            project_id=project_id,
            project_name=project_name,
            chatroom_name=chatroom_name,
            capabilities_line="",
            agent_dir=agent_dir,
            data_dir=data_dir,
            knowledge_dirs=knowledge_dirs,
            dwh_dir=dwh_dir,
            extra_context=self._build_extra_context(),
            from_participant_name=from_participant_name,
            message_content=message_content,
            chat_history=chat_history,
        )

    def build_setup_prompt(
        self,
        name: str,
        description: str,
        project_id: str,
        chatroom_name: str,
        message_content: str,
        data_dir: Path,
        context_files: list[str],
        project_name: str,
        agent_dir: Path,
        knowledge_dirs: list[Path] | None = None,
        dwh_dir: Optional[Path] = None,
        invitable_agents: Optional[list[str]] = None,
        plan_phase: Optional[str] = None,
    ) -> str:
        """First-turn variant. The only difference from ``build_prompt`` is
        the role-contract section (PLAN.md / setup framing instead of the
        steady-state coordinator block) and the inclusion of the context-files
        list in the extras. Same layout otherwise.
        """
        self._is_dm = False
        self._first_turn = True
        self._plan_phase = plan_phase
        self._plan = None
        self._invitable_agents = invitable_agents
        self._context_files = context_files or []
        return self._assemble(
            name=name,
            description=description,
            project_id=project_id,
            project_name=project_name,
            chatroom_name=chatroom_name,
            capabilities_line="",
            agent_dir=agent_dir,
            data_dir=data_dir,
            knowledge_dirs=knowledge_dirs,
            dwh_dir=dwh_dir,
            extra_context=self._build_extra_context(),
            from_participant_name="user",
            message_content=message_content,
            chat_history=None,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_prompt_builder(
    mode: OperationalMode,
    capabilities: Optional[list[str]] = None,
    coordinator_name: Optional[str] = None,
    git_url: Optional[str] = None,
) -> PromptBuilder:
    """Create a prompt builder based on operational mode.

    Args:
        mode: WORKER or COORDINATOR.
        capabilities: Worker capabilities (ignored for COORDINATOR).
        coordinator_name: Required for WORKER mode.
        git_url: The agent's bound git repo (from card.json local_settings),
            surfaced as a one-line nudge in FILES & STATE. None when unbound.

    Raises:
        ValueError: If mode is WORKER and ``coordinator_name`` is None.
    """
    if mode == OperationalMode.COORDINATOR:
        return CoordinatorPromptBuilder(git_url=git_url)
    if coordinator_name is None:
        raise ValueError("coordinator_name is required for WORKER mode")
    return WorkerPromptBuilder(
        coordinator_name=coordinator_name,
        capabilities=capabilities,
        git_url=git_url,
    )
