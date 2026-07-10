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
            f"- Bound git repo (your code goes here): {git_url}  "
            "(clone it into ./repos/ under your sandbox, branch per request, commit & push; "
            "repo conventions in memory/REPO.md)"
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
            f"Project: {project_name}\n"
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
goals, guardrails, acceptance criteria, and prior learnings.

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
- Focus on your assigned task only."""

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
4. Respond conversationally. Do NOT use a Task / Status / Acceptance Criteria
   template — that shape is for coordinated project work.

== WHAT NOT TO DO ==
- Do NOT create or reference PLAN.md — it does not apply here.
- Do NOT frame your work as "Milestone 1 / Milestone 2 …".
- Do NOT ask reflexive clarifying questions. If actionable, act."""

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
        """Build a worker prompt. ``project_id`` is unused at present; kept
        for callsite compatibility.
        """
        del project_id  # reserved for future use
        self._is_dm = is_dm
        capabilities_line = ", ".join(self._capabilities) if self._capabilities else ""
        return self._assemble(
            name=name,
            description=description,
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
        self._context_files: list[str] = []
        # "live" | "batch" — see _coordinator_role_contract for the gate.
        # "live" (default) hides BATCH COMPLETION WORKFLOW + HANDLING
        # WORKER QUESTIONS AND BLOCKERS from every user-comm / workroom
        # turn that isn't actually processing a batch result.
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
            return ["reply", "update_file", "create_room"]
        return ["reply", "update_file", "create_room", "project_completed"]

    def _role_contract(self) -> str:
        if self._is_dm:
            return self._dm_role_contract()
        if self._first_turn:
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
- PARALLEL only when truly independent.

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
If an approach fails twice, document in PLAN.md Learnings and create a new
room with a different approach or agent. If more than 2 milestones across
the project require pivots, STOP and escalate to the user.

Revise a milestone ONLY when you can name a SPECIFIC, JUSTIFIABLE reason the
revision will SUBSTANTIALLY improve the deliverable's quality — e.g. a materially
missing or wrong result that blocks the next milestone. Marginal gains — a
missing niche figure, slightly better sourcing, extra polish, data the web
doesn't readily provide — do NOT qualify: ACCEPT the deliverable, record residual
gaps as caveats in PLAN.md, and move to the next milestone, or escalate to the
user. Revise a single milestone AT MOST 2 times; if it still isn't usable after
that, accept the best version with caveats or escalate — NEVER keep spinning up
new revision rooms (an autonomous project may get no user reply, so prefer
accept-and-proceed).

== USER COMMUNICATION ==
Use the `user-communication` chatroom to:
  - Request clarification when requirements are unclear.
  - Escalate ambiguity raised by workers.
  - Report progress for long tasks.
  - Share final results when the project is complete.

== AVAILABLE WORKER AGENTS ==
Read the roster file AGENTS.md (its absolute path is listed under FILES &
STATE above) for the registry of worker agents — names, descriptions,
statuses. Read it BEFORE delegating so you invite agents by their real
names. It has two sections: "Your agents" (the user's own crew — delegate
freely) and "Other accounts" (public agents owned by OTHER users, reachable
only via explicit cross-account delegation). Never present "Other accounts"
agents as the user's own roster. Invite agents using their names EXACTLY as
written in the roster (not IDs, and do not add suffixes like '-agent')."""

        if self._flow_context == "batch":
            return steady_state + self._batch_flow_blocks()
        return steady_state

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
  3. UPDATE PLAN.md BEFORE deciding next steps (mark milestone, append to
     Review Log + Learnings sections).
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
       (d) project complete → when the FINAL milestone's deliverable is in hand
           and criteria pass, your remaining job is to DELIVER, not to keep
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

    def _setup_role_contract(self) -> str:
        return """== COORDINATOR ROLE — FIRST USER REQUEST ==
This is the FIRST message in this project. Your job is to UNDERSTAND, PLAN,
and DELEGATE — in that order.

== WORK EFFICIENTLY (avoid a read loop) ==
A setup turn needs only a handful of tool calls. Read each existing file AT MOST
ONCE; never re-read the same file or loop on list_dir. PLAN.md and AGENTS.md are
files you CREATE this turn via update_file — they do NOT exist yet, so do NOT
try to read them (a read will error; do not retry it). Going in circles here
burns the turn's token budget and the project never starts.

== STEP 1: UNDERSTAND ==
- Read the user request and any context files in shared-context.
- If the request is ambiguous or underspecified, ask for clarification via
  user-communication BEFORE planning. Do NOT guess.

== STEP 2: PLAN (only after requirements are clear) ==
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
  names.

PLAN.md template:
```markdown
# Project Plan: <name>

## Goal
[clear statement of what the user wants]

## Guardrails
- [constraints, quality requirements, scope boundaries]

## Milestones
- [ ] Milestone 1: <action>
      Deliverable: <specific_file.md>
      Acceptance Criteria:   # the MINIMUM the next step needs — any credible source ok
        - [ ] <outcome/figure needed for the next milestone>
        - [ ] <verifiable condition 2>
      Workroom: milestone-1-<slug>, Agent: <agent_name>
- [ ] Milestone 2: ...

## Current Status
Planning complete. Starting Milestone 1.

## Learnings
(updated after each milestone)

## Review Log
(pass/fail per acceptance criterion, after each batch)
```

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
  1. reply to user-communication asking specific clarifying questions."""

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
single-domain question belongs in your reply."""

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
- The user explicitly asked for a teammate's input."""

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
    ) -> str:
        """Build a coordinator prompt for either a live message or a
        batch-complete event.

        ``flow_context`` picks which extra blocks the role contract
        injects: ``"batch"`` adds the BATCH COMPLETION WORKFLOW and
        HANDLING WORKER QUESTIONS AND BLOCKERS blocks; ``"live"``
        (default) omits them. DM-shaped projects ignore flow_context
        because their role contract is entirely separate.

        ``dm_is_owned`` distinguishes the user's own 1:1 DM (True; no
        create_room, no INVITABLE AGENTS block) from an FD-tunneled
        DM-shaped project (False; foreign coordinator keeps create_room).
        Ignored when ``is_dm`` is False.
        """
        del project_id
        self._is_dm = is_dm
        self._dm_is_owned = dm_is_owned
        self._first_turn = False
        self._invitable_agents = invitable_agents
        self._context_files = []
        self._flow_context = flow_context
        return self._assemble(
            name=name,
            description=description,
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
    ) -> str:
        """First-turn variant. The only difference from ``build_prompt`` is
        the role-contract section (PLAN.md / setup framing instead of the
        steady-state coordinator block) and the inclusion of the context-files
        list in the extras. Same layout otherwise.
        """
        del project_id
        self._is_dm = False
        self._first_turn = True
        self._invitable_agents = invitable_agents
        self._context_files = context_files or []
        return self._assemble(
            name=name,
            description=description,
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
