# SPDX-License-Identifier: MIT
"""
clawmeets/models/agent.py
Public worker agent implementation using composition.

Agents are registered by admins and discoverable in the registry.
They execute specific tasks when addressed by a coordinator.

All state is read from the filesystem (card.json) - no in-memory state.
Extends PersistableParticipant for Active Record persistence.
"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional

from .participant import ParticipantRole, OperationalMode
from .persistable import PersistableParticipant
from ..api.actions import ActionBlock, COORDINATOR_ACTION_SCHEMA, WORKER_ACTION_SCHEMA
from ..api.action_validator import ActionValidator, StateSnapshot
from ..api.responses import AgentStatus
from ..llm.base import LLMInvocationError, LLMRateLimitError, LLMTimeoutError
from . import model_config as _model_config
from ..llm.prompt_builder import (
    BATCH_COMPLETION_PLAN_INSTRUCTION,
    BATCH_COMPLETION_PLAN_REVIEW_INSTRUCTION,
    PLAN_GATE_RELEASED_INSTRUCTION,
    CoordinatorPromptBuilder,
    create_prompt_builder,
)
from ..llm.triggers import derive_role
from ..runner.invocation_registry import invoke_with_registry as _invoke_with_registry
from ..sync.changelog import ProjectStatus
from ..utils.file_io import FileUtil

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..api.client import ClawMeetsClient
    from ..llm.prompt_builder import PlanPromptState
    from .context import ModelContext
    from .chat_message import ChatMessage
    from .project import Project

logger = logging.getLogger(__name__)

# Retry configuration for transient LLM CLI failures
_MAX_RETRIES = 2  # Total attempts: 3 (1 original + 2 retries)
_INITIAL_RETRY_DELAY = 30  # seconds
_TRANSIENT_INDICATORS = (
    "overloaded",
    "rate_limit",
    "529",
    "503",
    "too many requests",
    # Dropped connections carry no status code, so they can only be caught
    # here. Both phrasings are real: "Connection closed mid-response" and
    # "The socket connection was closed unexpectedly".
    "connection closed",
    "connection was closed",
)

# Semantic-validation retry budget (see action_validator + _invoke_validated).
# Hard integer cap -> at most 1 + _MAX_VALIDATION_RETRIES = 3 model invocations
# per triggered turn, 0 extra on the common valid path. This is distinct from
# _MAX_RETRIES above (transient CLI failures); the two never compound because a
# validation retry re-enters the transient loop only after a *successful* invoke.
_MAX_VALIDATION_RETRIES = 2


def _is_transient_error(error: LLMInvocationError) -> bool:
    """Check if an LLMInvocationError is likely transient (retryable).

    A timeout is NOT transient. The runner's kill window and the server's
    ``PendingWork.timeout_seconds`` are the same bound by design (see
    ``server/batch_timeout.resolve_batch_timeout``), so an invocation that
    burns its full window dies at the moment the batch expires. Retrying
    starts a fresh full-length attempt into a batch whose pending_work is
    already gone: the reply is never credited, the CANCEL_LLM the checker
    sends lands during the backoff sleep with nothing registered, and the
    UI has already marked the recipient terminal. Fall through to the
    terminal path instead — ``_post_error_notification`` posts a non-ack
    reply, which usually closes the batch before the checker even ticks.
    """
    if isinstance(error, LLMRateLimitError):
        return False  # Rate limits should not be retried with short backoff
    if isinstance(error, LLMTimeoutError):
        return False

    # Structural signal wins where the provider reported one: >= 500 is the
    # backend's problem and worth the backoff, any 4xx is ours and will fail
    # again identically. Same policy as the OpenRouter wire path
    # (llm/_openai_wire.py). A present-but-terminal status short-circuits
    # rather than falling through — deferring to the string match would let
    # a stray "503" in a 400's body resurrect it.
    status = getattr(error, "status_code", None)
    if status is not None:
        return status >= 500

    msg = str(error).lower()
    return any(indicator in msg for indicator in _TRANSIENT_INDICATORS)


_TRIGGER_MARKER_RE = re.compile(r"<!--\s*clawmeets:[a-z0-9-]+-trigger\s*-->")


def _resume_marker_for_dm(chatroom, agent_name: str) -> Optional[str]:
    """Return the trigger marker carried by the agent's most recent reply.

    Marker-driven skills (e.g. /clawmeets:personalize) activate on
    description match against the inbound message. On a
    follow-up turn the user's reply carries no marker, so the skill would
    stop firing. The SKILL.md convention asks the agent to echo the marker
    on every interim reply; we surface that signal back into the next
    turn's inbound by prepending it, restoring routing across turns.
    Returns None if no prior agent reply carries a marker.
    """
    for msg in reversed(chatroom.get_messages()):
        if msg.from_participant_name == agent_name:
            m = _TRIGGER_MARKER_RE.search(msg.content)
            return m.group(0) if m else None
    return None


def _schema_allows_agent_invites(action_schema: dict) -> bool:
    """True iff ``action_schema`` permits an action that names invitees.

    Precise discriminator for "does this turn need the agent scans" — only a
    turn that can emit ``create_room`` or ``invite_agent`` needs
    ``invitable_agents`` / ``resolvable_agents``. Reads the ``oneOf`` variants
    by structure (not by identity), so it stays correct for the full
    COORDINATOR schema, the restricted coordinator-DM schema, and
    WORKER_ACTION_SCHEMA alike — an assistant acting as coordinator is covered,
    a coordinator on an owned DM (worker schema) correctly skips the scan.
    """
    variants = (
        action_schema.get("properties", {})
        .get("actions", {})
        .get("items", {})
        .get("oneOf", [])
    )
    return any(
        v.get("properties", {}).get("type", {}).get("const")
        in ("create_room", "invite_agent")
        for v in variants
    )


def _agent_name_variants(full_name: str) -> set[str]:
    """Full registry name plus its short (post-``{owner}-`` prefix) form.

    Used only to build ``resolvable_agents`` for the feedback-message branch
    (§1a-B), so a known-but-not-invitable agent is recognized whether the model
    referenced it by full name (cross-owner, e.g. ``clawmeets-nyc_dining``) or
    by short name. Never affects the PASS/REJECT decision.
    """
    variants = {full_name}
    if "-" in full_name:
        variants.add(full_name.split("-", 1)[1])
    return variants


def build_state_snapshot(
    model_ctx: "ModelContext",
    project_id: str,
    *,
    include_agents: bool,
) -> StateSnapshot:
    """Read local (no-HTTP) server state ONCE per turn for validation.

    Sources every referent set from the SAME ``Project.get(project_id, ctx)``
    the turn already loaded (its rooms, status), so the validator's valid-set is
    built from the identical local synced snapshot the prompt is built from
    (the stale-local-snapshot invariant: the model can only reference rooms it
    can see, and the validator sees exactly those, so a valid reference cannot
    false-reject; the executor's 4xx handler backstops any TOCTOU race).

    ``existing_rooms`` is **exactly** that read — no seeded member, no widened
    invariant. It used to carry one: ``plan-review`` while the project was
    awaiting approval. That room was opened mid-turn by
    ``clawmeets plan consult`` through the SERVER's context and the runner
    could not adopt it until the turn released the runloop lock, so the spec
    contract told a coordinator to open a room and then reply into it in the
    same turn, and the reply was rejected as nonexistent until the retry budget
    ran out. Consultation now happens in ``shared-context``, which is in
    ``project.chatrooms`` before the coordinator's first turn, so the hazard
    does not arise and the seed that patched it is gone.

    Keyed on the single ``project_id`` in scope, it automatically reads the
    correct side of a two-sided Front-Desk tunnel (§1b) — the side whose turn is
    running — with no cross-tunnel read and no extra project fetch.

    Agent scans (``invitable_agents`` + ``resolvable_agents``) run ONLY when
    ``include_agents`` (a turn that can emit ``create_room`` or
    ``invite_agent``). Worker turns and owned-DM turns skip both scans entirely
    (reviewer M1 refinement).

    Called ONCE per turn, before the retry loop, and never from inside it. The
    one field the loop may move afterwards — ``invitable_agents`` — is refreshed
    through :func:`refresh_invitable_agents`, deliberately a separate, narrower
    function rather than a second call to this one: re-running the whole builder
    would also re-read ``blocked_notes``, which is non-monotone and whose live
    value would make the verdict oscillate (see ``StateSnapshot``'s docstring).

    **There used to be a SECOND plan field here**, ``awaiting_plan_approval``,
    computed by a ``_plan_preapproval_blocked`` that fired for the whole of
    ``spec-ing``. It is gone, and nothing lost a guarantee: the block a project
    starts life under is now an ordinary open plan note — the **go-note**,
    seeded by ``POST /projects`` in the same transaction as ``PLAN.md`` — which
    the one remaining field already counts. Two gates and two snapshot members
    collapsed into one of each because they were always the same rule read
    twice.
    """
    from .project import Project

    project = Project.get(project_id, model_ctx)
    project_active = project.status == ProjectStatus.ACTIVE
    existing_rooms = frozenset(project.chatrooms)

    if include_agents:
        invitable_agents = frozenset(
            Agent.invitable_short_names_for_project(project, model_ctx)
        )
        resolvable_agents = frozenset(
            variant
            for other in Agent.list_all(model_ctx, viewer_is_admin=True)
            for variant in _agent_name_variants(other.name)
        )
    else:
        invitable_agents = frozenset()
        resolvable_agents = frozenset()

    return StateSnapshot(
        existing_rooms=existing_rooms,
        invitable_agents=invitable_agents,
        resolvable_agents=resolvable_agents,
        project_active=project_active,
        # Same gate as the agent scans above and for the same reason: only a
        # turn that can emit `create_room` / `invite_agent` has a reader for
        # this, so a worker turn skips the shared-context lookup entirely.
        blocked_notes=(
            _plan_execution_blocked(project) if include_agents else 0
        ),
    )


def refresh_invitable_agents(
    model_ctx: "ModelContext",
    project_id: str,
) -> frozenset[str]:
    """Re-read ONLY the project's invitable-agent set from local state.

    The monotone half of the once-per-turn snapshot. Deliberately narrow: it
    re-derives the single field a coordinator can legitimately change *during*
    its own turn (via ``clawmeets project allowlist``, whose
    ``PROJECT_ALLOWLIST_UPDATED`` entry replays into the local ``meta.json``)
    and touches nothing else. Everything the retry loop must see hold still —
    rooms, project status, the plan gate — stays with the frozen snapshot.

    Mirrors ``build_state_snapshot``'s call exactly, ``exclude_ids`` included
    (i.e. omitted), so the union it feeds is homogeneous with the set it widens;
    a differently-scoped scan here would silently add or drop the agent's own
    short name relative to the frozen half.

    Returns an EMPTY set on any read failure rather than raising. The caller
    unions the result, so an empty set is a no-op — a project that vanished
    mid-turn degrades to today's frozen behaviour instead of killing a turn
    whose real work may already be valid.
    """
    from .project import Project

    try:
        project = Project.get(project_id, model_ctx)
        return frozenset(Agent.invitable_short_names_for_project(project, model_ctx))
    except Exception:
        logger.debug(
            "invitable refresh failed for project %s; keeping the frozen set",
            project_id,
            exc_info=True,
        )
        return frozenset()


def _plan_execution_blocked(project: "Project") -> int:
    """§7.4's gate — **the only one** — as the count that motivates it.

    ::

        blocked == surface == "regular" and open_notes_for_you > 0

    **Two conjuncts, where there used to be two functions and six.** The old
    pair — a pre-approval gate on ``phase == "spec-ing"`` and an execution gate
    on ``phase == "executing" and open_notes > 0 and shared_context_has_peer``
    — were the same rule stated twice, and the redesign that removed the Approve
    button collapsed them. Every conjunct that went, went for a reason worth
    keeping written down, because each looks like something a reader might
    restore:

    ``phase == "executing"`` — **deleted, and deleting it is what makes the
    go-note work at all.** The block a project starts life under is now an
    ordinary open plan note, seeded by ``POST /projects`` before the coordinator
    has taken a turn. A gate that only fires once the plan is accepted would be
    clear for exactly the window the go-note exists to close, and the note would
    block nothing at the one moment it has to block everything.

    ``shared_context_has_peer`` — **deleted, and its argument went with the
    button.** ``Q12`` added it because a coordinator refused ``create_room``
    while consulting an empty ``shared-context`` was a livelock silent on both
    ends: the room is seeded with ``[coordinator]`` alone, so on a fresh project
    nothing said why the project had stopped. That cannot arise now. The release
    condition is *"the user accepts the go-note"*, the note is in the plan tray
    on the user's own desk with the coordinator's ask in it, and the coordinator
    is told about it in its prompt. Keeping the clause would make the gate
    return 0 on every brand-new project — which is every project the go-note is
    for.

    **The consultation half of that argument survives the deletion, and it is
    what keeps this a gate rather than a deadlock.** A blocked coordinator still
    has to be able to reach a specialist, and it can: this rests on
    ``shared-context`` alone, because no note can reach a specialist in either
    phase — ``user`` is the only addressee a plan note has. So the room IS the
    channel,
    it exists from project creation, and ``clawmeets plan consult`` seats the
    invitable roster on demand. The mechanism is the load-bearing part:
    ``ensure_consultation_roster`` appends ``PARTICIPANT_ADDED`` straight through
    the runloop and never passes through the ``create_room`` action this gate
    refuses. Were it an action, the one channel a gated coordinator has would be
    closed by the gate itself.

    Note what this does NOT become: *"the room has a peer in it"*. That was the
    deleted conjunct, and it is a different claim from *"the room exists and can
    be populated"*. A reader who collapses the two reintroduces the hole.

    ``phase`` is not read at all any more, so this opens no file the pass has
    not already opened; it was already a pure function of stored fields, and now
    it is a shorter one.

    **What did NOT change, and why each of the two survivors is still here:**

    ``surface == "regular"`` — on a **front-desk** project there is no user in
    the accepting role (§7.2), so the halt's release condition does not exist
    and the halt would be a deadlock, stopping every front-desk engagement at
    its first agent proposal. On a **DM** it is inert rather than wrong: a DM
    has no plan, so the count is zero either way. The conjunct is kept as the
    explicit shape guard it has always been.

    ``open_notes_for_you > 0`` — explicitly **NOT** ``changed_since_acceptance``.
    The digest comparison is the tempting reading and it is backwards: it fires
    on **the user's own accepted edit**, because it does not care who moved the
    bytes. Gating on it would halt a project *because the user edited it*.

    **The two plan facts are read from ``meta.json``, not from the sidecar,**
    and that is not an optimisation. This function runs in the **agent process**;
    ``plan.json`` lives in the server's ``metadata/`` tree and is not synced.
    They arrive here over ``PROJECT_PLAN_STATE`` (D13). If that entry ever stops
    being published the gate silently reads zeros and never fires, which is why
    the projection is reconciled against its source by test rather than trusted.
    """
    if project.surface != "regular":
        return 0
    return max(project.plan_open_notes, 0)


def plan_prompt_state(
    project: "Project", ctx: "ModelContext"
) -> "PlanPromptState | None":
    """§7.3's five steady-state facts, assembled from **synced** state only.

    ``None`` when the project has no plan lifecycle to report — a DM, or any
    project that was never seeded (which is every project predating the feature,
    §3.5 property 3). A coordinator on one of those gets exactly the prompt it
    gets today.

    Every member is a project field or a pure function of one and the synced
    ``PLAN.md``. That is a constraint, not a coincidence: this runs in the agent
    process, which cannot open the server-side sidecar, so ``open_notes_for_you``
    arrives over ``PROJECT_PLAN_STATE`` (D13) and *"the spec moved"* is
    ``spec_digest(synced body) != plan_accepted_spec_digest`` — the sidecar's own
    definition, computed from the two halves that are local.

    The title is the project's display label rather than the document's ``# ``
    heading: it is the name the user sees on the card, and deriving it would put
    a heading parser in the prompt builder for a string the project already owns.
    """
    from ..llm.prompt_builder import PlanPromptState
    from .chatroom import Chatroom
    from .plan_markdown import spec_digest
    from .project_plan import approval_state

    if project.surface != "regular" or project.plan_seeded_at is None:
        return None
    # One read of the synced document serves both facts. It used to be opened
    # only when there was an acceptance digest to compare against — but the
    # approval text matters most in exactly the case that had no digest, so the
    # read moved out of the branch rather than being done a second time inside
    # a new one.
    try:
        room = Chatroom.get(project.id, "shared-context", ctx)
        raw = room.get_file("PLAN.md") if room is not None else None
    except ValueError:
        raw = None
    body = raw.decode("utf-8", errors="replace") if raw is not None else None
    changed = False
    if (
        body is not None
        and project.plan_accepted_at is not None
        and project.plan_accepted_spec_digest
    ):
        changed = spec_digest(body) != project.plan_accepted_spec_digest
    return PlanPromptState(
        title=project.display_name or project.name,
        phase=project.phase,
        changed_since_acceptance=changed,
        open_notes_for_you=project.plan_open_notes,
        approval=approval_state(body) if body is not None else "",
        user_has_reviewed=project.plan_user_reviewed_at is not None,
    )


def _is_spec_consultation_batch(project: "Project", chatroom_name: str) -> bool:
    """M5 AC-5.6 — is this ``BATCH_COMPLETE`` a **spec consultation**?

    ::

        spec_consult == chatroom_name == PLAN_ROOM
                        and surface == "regular"
                        and phase == "spec-ing"

    The one predicate behind both halves of the variant — the trailing
    instruction on the synthetic message and the role-contract block — so the
    two cannot come apart.

    **Selection is by room and phase, in Python, against the constant.** The
    room name is compared to
    :data:`~clawmeets.models.project_plan.PLAN_ROOM` rather than to a string
    literal, and the model is never handed a room name and asked to test it:
    a rule the prompt asks the model to evaluate is a rule the model can
    decline to evaluate.

    **Every conjunct is a decision.**

    ``chatroom_name == PLAN_ROOM`` — the project's one consultation room, where
    the relay posts and where ``PLAN.md`` itself lives.

    ``surface == "regular"`` — belt and braces, in the shape
    :func:`_plan_execution_blocked` already uses and for the same reason.
    :attr:`Project.phase` derives ``executing`` for every non-regular shape, so
    this is redundant *right now*; it stays because that derivation branch is
    named load-bearing three times and a reviewer trimming it would delete
    exactly it.

    ``phase == "spec-ing"`` — **this is what makes the milestone workflow
    wrong, and it now carries the whole of the room half's old weight.** When
    the consultation room was ``plan-review``, opened on demand, the room name
    alone was nearly a proxy for the stage. ``PLAN_ROOM`` is not: it exists on
    every regular project from creation and hosts ordinary post-acceptance
    consultation too, where ``create_room`` is no longer refused and ticking a
    milestone's checkbox is the correct instruction again. Selecting on the
    room alone was merely imprecise before the merge; it is plainly wrong
    after it. Do not trim this conjunct.

    A pure function of a project and a room name, so a test states the truth
    table directly instead of driving a whole batch to observe which prompt
    came out.
    """
    from .project_plan import PLAN_ROOM

    return (
        chatroom_name == PLAN_ROOM
        and project.surface == "regular"
        and project.phase == "spec-ing"
    )


def _participant_names(ids: "Sequence[str]", ctx: "ModelContext") -> list[str]:
    """M5 AC-5.7 — batch participants as **names**, for a prompt that must group
    by agent.

    ``BatchCompletePayload.responded_participants`` carries participant **ids**
    (``PendingWork`` fills them from the resolved ``expects`` of the message
    that opened the batch), and they were joined raw into the synthetic message
    the coordinator reads. A coordinator asked to *"group the answers by
    agent"* was being handed a list of UUIDs — it cannot match one to a name in
    the room, so it either guesses or ignores the list.

    Resolved through :meth:`Participant.get`, the typed factory
    :meth:`Chatroom.list_participants` already uses. Deliberately not resolved
    through the room: an id that has since left the room is still a fact about
    the batch, and going through the membership list would silently drop it.

    **Falls back to the raw id** for anything unresolvable rather than omitting
    it. A shorter list is how *"who did NOT answer"* goes missing, and an
    unresolvable id printed as itself is at least a question the reader can
    ask; a name that is simply absent is not.
    """
    from .participant import Participant

    names: list[str] = []
    for pid in ids:
        try:
            participant = Participant.get(pid, ctx)
        except Exception:
            participant = None
        names.append(getattr(participant, "name", None) or pid)
    return names


# ─────────────────────────────────────────────────────────────────────────────
# DM-thread auto-title (M4c) — generator constants + pure helpers
# ─────────────────────────────────────────────────────────────────────────────

# Instruction handed to LLMProvider.generate_text. Ends with a blank line then
# the message so the provider-agnostic deterministic default
# (llm.base.deterministic_text_snippet) isolates the message as its payload.
_DM_TITLE_INSTRUCTION = (
    "Generate a concise, specific 3-6 word title (no surrounding quotes, no "
    "trailing period) that captures the topic of a conversation that opens with "
    "the message below. Reply with the title text only.\n\n{message}"
)

# Low-signal defer (R5/MF3): a triggering message at/under this length OR matching
# the greeting-only pattern carries no topic, so titling is deferred to a later
# substantive message — up to the defer budget, after which a terse thread is
# force-titled deterministically so it still converges off "New chat".
_DM_TITLE_MIN_SIGNAL_CHARS = 12
_DM_TITLE_DEFER_BUDGET = 3
_DM_GREETING_RE = re.compile(
    r"^(hi|hii+|hey|hello|yo|sup|hiya|howdy|gm|gn|good\s+(morning|evening|afternoon)|"
    r"hey\s+there|hi\s+there|hello\s+there|you\s+there|yt|test+|ping|ok|okay|k|"
    r"thanks|thank\s+you|ty|cool|nice)[\s!.?,\-]*$",
    re.IGNORECASE,
)


def _dm_title_is_too_terse(content: str) -> bool:
    """True when a triggering message is too low-signal to title from (R5/MF3)."""
    stripped = content.strip()
    return len(stripped) <= _DM_TITLE_MIN_SIGNAL_CHARS or bool(_DM_GREETING_RE.match(stripped))


def _clean_dm_title(text: str, *, max_words: int = 6, max_chars: int = 60) -> str:
    """Normalize raw generator output into a short, punctuation-clean title.

    Every DM-thread title flows through here — model-composed output, the live
    trigger's deterministic-truncation fallback, and the offline backfill — so
    this is the single choke point that capitalizes the leading character. That
    keeps a deterministic-truncation title (which arrives fully lowercase from
    the model-free fallback, e.g. ``react state bug``) from rendering lowercase
    in the sidebar, while a first-char-only bump leaves the casing of every
    NON-leading word intact (``debug reactDOM`` → ``Debug reactDOM``) and is a
    no-op on an already-capitalized model title.
    """
    if not text:
        return ""
    cleaned = " ".join(text.split())  # collapse all whitespace/newlines
    cleaned = cleaned.strip().strip("\"'`").strip()  # drop wrapping quotes
    cleaned = re.sub(r"^(chat\s+)?title\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE).strip()
    words = cleaned.split(" ")
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words])
    cleaned = cleaned[:max_chars]
    cleaned = cleaned.rstrip(" .,;:!?-—–").strip()  # drop trailing punctuation
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]  # capitalize leading char only
    return cleaned


class Agent(PersistableParticipant):
    """
    Public worker agent - executes specific tasks using composition.
    Registered by admin, discoverable in registry.

    Agents only respond when explicitly addressed via expects_response_from.
    They execute tasks and report results back to the coordinator.

    All state is read from the filesystem (card.json) on each property access.
    This ensures the model always reflects the current state on disk.

    Extends PersistableParticipant for Active Record methods:
        - Agent.get(), Agent.get_by_name(), Agent.list_all()
        - Agent.register(), Agent.verify_token()
        - agent.save(), agent.update_status(), agent.heartbeat()
        - agent.to_response()

    Composition:
        - ClaudeCLI: Direct Claude CLI invocation
        - ActionBlockExecutor: Processes Claude output and executes actions via HTTP
        - Prompt builders created on-demand via create_prompt_builder()
    """

    # Active Record: directory subdirectory for agents
    _role_subdir: ClassVar[str] = "agents"

    @classmethod
    def resolve_with_namespace(
        cls,
        name: str,
        project: "Project",
        ctx: "ModelContext",
    ) -> Optional["Agent"]:
        """Resolve a name to an Agent in the project owner's namespace.

        Exact match on the full registry name wins. If no exact match, try
        the project owner's namespace by prefixing ``{owner_username}-``.
        """
        from .user import User

        agent = cls.get_by_name(name, ctx)
        if agent is not None:
            return agent

        if not project.created_by:
            return None
        owner = User.get(project.created_by, ctx)
        if owner is None:
            return None
        return cls.get_by_name(f"{owner.username}-{name}", ctx)

    @classmethod
    def get_by_name_for_owner(
        cls,
        name: str,
        owner_user_id: Optional[str],
        ctx: "ModelContext",
    ) -> Optional["Agent"]:
        """Resolve an agent by the short or full name a human would type,
        inside one owner's namespace.

        Exact match on the full registry name first, then
        ``{owner.username}-{name}``. Returns None when neither hits — callers
        treat that as "keep the raw string as a display label" rather than an
        error, because a to-do or SOP naming a since-deleted agent should still
        save with the name visible.

        Sibling of ``resolve_with_namespace``, which does the same two tiers
        keyed off a *project's* owner. This one takes the owner id directly,
        for the desk routes, where the namespace comes from the calling agent's
        ``registered_by``.
        """
        from .user import User

        agent = cls.get_by_name(name, ctx)
        if agent is not None:
            return agent

        if not owner_user_id:
            return None
        owner = User.get(owner_user_id, ctx)
        if owner is None:
            return None
        return cls.get_by_name(f"{owner.username}-{name}", ctx)

    @staticmethod
    def short_name(full_name: str, owner_username: Optional[str]) -> str:
        """Strip ``{owner_username}-`` prefix from a full agent name."""
        if not owner_username:
            return full_name
        prefix = f"{owner_username}-"
        if full_name.startswith(prefix):
            return full_name[len(prefix):]
        return full_name

    def short_name_for(self, owner_username: Optional[str]) -> str:
        """This agent's short name in the given owner's namespace."""
        return Agent.short_name(self.name, owner_username)

    @classmethod
    async def sync_from_server(
        cls,
        ctx: "ModelContext",
        exclude_ids: set[str],
        owner_username: Optional[str] = None,
    ) -> int:
        """Sync all worker agents from server to local filesystem.

        Fetches the agent registry from server and persists each agent's
        card.json locally. Called on startup to populate local agents/
        directory with current registry state.

        Also generates a global AGENTS.md file listing all available agents,
        which coordinators reference during prompts.

        Args:
            ctx: ModelContext for filesystem access (must have client configured)
            exclude_ids: Set of agent IDs to skip (pass empty set if none)
            owner_username: Username of the runner's owner. When set, agents
                owned by this user render with short names in AGENTS.md.

        Returns:
            Number of agents synced

        Raises:
            ValueError: If ctx.client is not configured
        """
        if ctx.client is None:
            raise ValueError("ModelContext.client must be configured for sync_from_server")
        agents = await ctx.client.list_agents()
        server_ids: set[str] = set()
        synced_count = 0
        owner_user_id: Optional[str] = None
        for agent_data in agents:
            agent_id = agent_data.id
            server_ids.add(agent_id)
            if agent_id in exclude_ids:
                continue

            # Sync card.json for this worker agent
            agent = cls.get_or_create(agent_id, ctx)
            agent.update_card(**agent_data.model_dump(mode="json"))
            synced_count += 1

            # Locate the owner's user id by matching username against any
            # agent registered by that user (agent name starts with
            # ``{owner_username}-``). Avoids an extra server round-trip.
            if (
                owner_user_id is None
                and owner_username
                and agent_data.registered_by
                and agent_data.name.startswith(f"{owner_username}-")
            ):
                owner_user_id = agent_data.registered_by

        logger.info(f"Synced {synced_count} worker agents from server")

        # Self-heal: prune local peer cards whose id is no longer in the
        # registry. Covers deletions missed while the runner was offline
        # (envelope never delivered) and retroactively cleans any stale
        # cards from before AGENT_REGISTRY_CHANGE(delete) was wired up.
        pruned_count = 0
        for peer in cls.list_all(ctx, viewer_is_admin=True):
            if peer.id in server_ids or peer.id in exclude_ids:
                continue
            if cls.prune_peer_card(peer.id, ctx):
                pruned_count += 1
        if pruned_count:
            logger.info(f"Pruned {pruned_count} stale peer card(s) missing from registry")

        # Generate global AGENTS.md file after syncing all agents.
        cls.regenerate_agents_md(
            ctx, owner_username=owner_username, owner_user_id=owner_user_id
        )

        return synced_count

    @classmethod
    def regenerate_agents_md(
        cls,
        ctx: "ModelContext",
        owner_username: Optional[str] = None,
        owner_user_id: Optional[str] = None,
    ) -> None:
        """Re-render the global AGENTS.md roster from local peer cards.

        Reads the synced peer cards on disk and rewrites AGENTS.md in place.
        Called by ``sync_from_server`` after a full sync, and by the reactive
        loop on AGENT_STATUS_CHANGE so the roster's status column stays fresh
        between full syncs (the status flip is written to each peer card.json,
        but AGENTS.md must be re-rendered to reflect it).

        Pass the resolved owner id as the viewer so the runner-owner's OWN
        private (non-discoverable) crew is included — agents are private by
        default, so without this the coordinator's roster is empty and it has
        nobody to delegate to. When ``owner_user_id`` is None (unknown owner)
        this is a no-op for private agents (falls back to discoverable-only),
        and it only ever adds the owner's own private agents — never another
        account's.

        When ``owner_user_id`` is not supplied but ``owner_username`` is, the
        id is resolved from local peer cards (an agent whose name starts with
        ``{owner_username}-``) — so callers that only know the username (e.g.
        the reactive loop's status-change handler) still include the private
        crew without a server round-trip.
        """
        if owner_user_id is None and owner_username:
            prefix = f"{owner_username}-"
            for peer in cls.list_all(ctx, viewer_is_admin=True):
                if peer.registered_by and peer.name.startswith(prefix):
                    owner_user_id = peer.registered_by
                    break
        all_agents = cls.list_all(
            ctx, discoverable_only=True, viewer_user_id=owner_user_id
        )
        cls._generate_agents_md(
            all_agents,
            ctx.participants_dir / "AGENTS.md",
            owner_username=owner_username,
            owner_user_id=owner_user_id,
        )

    @classmethod
    def prune_peer_card(cls, agent_id: str, ctx: "ModelContext") -> bool:
        """Hard-delete a peer agent's local card directory.

        Called by:
          - the runner's ``AGENT_REGISTRY_CHANGE(action="delete")`` handler,
            for surgical pruning the moment a peer is deleted server-side;
          - ``sync_from_server``'s diff-and-prune step, to self-heal at
            startup / on every full sync.

        Idempotent: a missing directory returns False without raising. The
        only file-system side effect is ``shutil.rmtree`` on every matching
        peer dir (there should be exactly one in practice).

        Args:
            agent_id: ID of the peer agent whose local card should be removed
            ctx: ModelContext for filesystem access

        Returns:
            True if any directory was removed, False if no match existed.
        """
        role_dir = ctx.participants_dir / cls._role_subdir
        if not role_dir.exists():
            return False
        matches = [d for d in role_dir.glob(f"*-{agent_id}") if d.is_dir()]
        if not matches:
            return False
        for d in matches:
            shutil.rmtree(d)
            logger.info(f"Pruned peer card: {d.name}")
        return True

    @classmethod
    def _generate_agents_md(
        cls,
        agents: list["Agent"],
        output_path: "Path",
        owner_username: Optional[str] = None,
        owner_user_id: Optional[str] = None,
    ) -> None:
        """Generate AGENTS.md file listing all available agents.

        This file is referenced by coordinators in their prompts to see
        which agents are available for delegation.

        When ``owner_username``/``owner_user_id`` are provided, agents owned by
        that user are listed by their short name (without the ``{owner}-``
        prefix) so the coordinator can address them concisely. Other agents
        keep their fully-qualified name.

        Args:
            agents: List of Agent objects
            output_path: Path to write AGENTS.md
            owner_username: Runner owner's username (used to strip prefixes)
            owner_user_id: Runner owner's user id (used to match registered_by)
        """
        header = "| Agent | Role | Status | Notes |\n|-------|------|--------|-------|"

        def _row(a: "Agent") -> str:
            display = a.name
            if owner_user_id and a.registered_by == owner_user_id:
                display = a.short_name_for(owner_username)
            if display != a.name:
                return f"| {display} | {a.description} | {a.status.value} | (full name: `{a.name}`) |"
            return f"| {display} | {a.description} | {a.status.value} | |"

        def _table(group: list["Agent"]) -> str:
            if not group:
                return "| (none) | - | - | |"
            return "\n".join(_row(a) for a in group)

        # Partition by ownership so the user's own crew is never conflated with
        # other accounts' public (discoverable) agents. Only meaningful when we
        # know the owner; otherwise fall back to a single flat table.
        if owner_user_id:
            owned = [a for a in agents if a.registered_by == owner_user_id]
            external = [a for a in agents if a.registered_by != owner_user_id]
            sections = (
                f"## Your agents\n"
                f"Your own crew — delegate to these directly.\n\n"
                f"{header}\n{_table(owned)}\n\n"
                f"## Other accounts (public — reach via cross-account delegation)\n"
                f"Public agents owned by **other users**, NOT part of your roster. "
                f"They appear here only so you can delegate across accounts when the "
                f"user explicitly asks; never present them as the user's own agents.\n\n"
                f"{header}\n{_table(external)}"
            )
        elif agents:
            sections = f"{header}\n{_table(agents)}"
        else:
            sections = f"{header}\n| (no agents registered) | - | - | |"

        namespace_note = ""
        if owner_username:
            namespace_note = (
                f"- Agents you own render by **short name**; address them with "
                f"``@short-name`` (e.g. ``@researcher`` instead of "
                f"``@{owner_username}-researcher``). Fully-qualified names also work.\n"
            )

        content = f"""# Available Agents

{sections}

## Notes
- Use @mentions to delegate work: "@agent-name please do X"
- Check agent status before delegating (online agents respond faster)
- Agent names (not IDs) should be used in chatroom invites
{namespace_note}"""
        FileUtil.write(output_path, content, "text")
        logger.debug(f"Generated AGENTS.md at {output_path}")

    def __init__(
        self,
        id: str,
        model_ctx: "ModelContext",
    ) -> None:
        """Initialize an Agent.

        For Active Record operations (server-side), only id and model_ctx are needed.
        For task execution (runner-side), model_ctx should have cli and client configured.

        Runtime dependencies are accessed via model_ctx:
        - model_ctx.cli: Claude CLI for LLM invocation
        - model_ctx.knowledge_dirs: Additional directories for Claude access
        - model_ctx.client: ClawMeetsClient for HTTP operations
        - model_ctx.action_executor: ActionBlockExecutor for action execution

        Args:
            id: The agent's unique identifier
            model_ctx: ModelContext for filesystem access (may include cli/knowledge_dirs/client)
        """
        super().__init__(id, model_ctx)
        # Per-project memo of the EXECUTION gate's last observed value, read
        # only by :meth:`on_plan_state_change` and seeded by
        # :meth:`on_plan_gate_prime`. In-memory on purpose: it answers "did THIS
        # entry release the gate", a question about two adjacent observations,
        # not durable state — and the prime is what keeps a restart from
        # counting as the first of them.
        self._plan_gate_blocked: dict[str, bool] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Role Property (from Participant ABC)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def role(self) -> ParticipantRole:
        """Return AGENT role."""
        return ParticipantRole.AGENT

    def _resolve_invitable_agents_for_prompt(
        self,
        project: "Project",
    ) -> Optional[list[str]]:
        """Resolve the per-turn invitable-agent list for the coordinator prompt.

        Reads the project-side allowlist (``project.agent_{names,teams}``):

          - Non-empty filter → matches every locally-known agent against
            ``matches_invitable`` (which handles full-name, id,
            owner-relative short-name, and team match uniformly).
          - Empty filter → returns ``None`` (the prompt falls back to
            AGENTS.md as the discovery surface).

        The candidate pool is NOT owner-scoped: cross-owner agents the user
        explicitly listed in ``agent_names`` (e.g. ``clawmeets-nyc_dining``
        on a chengtao-owned project) must reach ``matches_invitable`` to be
        accepted — matching what the server's create_room enforcement does.
        """
        if not project.enforces_invitable_allowlist:
            return None
        return Agent.invitable_short_names_for_project(
            project, self._model_ctx, exclude_ids=frozenset({self.id})
        )

    @classmethod
    def invitable_agents_for_project(
        cls,
        project: "Project",
        model_ctx: "ModelContext",
        *,
        exclude_ids: frozenset[str] = frozenset(),
    ) -> list["Agent"]:
        """**The invitable roster**: the agents a coordinator may put into ``project``.

        With an explicit allowlist set (``project.agent_{names,teams}``) →
        the resolved allowlist; with no allowlist → the owner's own crew
        (matching what AGENTS.md lists under "Your agents").

        The candidate pool is admin-scoped (skips the discoverable filter so
        non-discoverable owned agents are visible); ``resolve_invitable_agents``
        drops anything the project hasn't allowed.

        Returns the ``Agent`` objects rather than their names because two
        callers want two different projections of one query and a second query
        would drift from this one. :func:`invitable_short_names_for_project`
        wants the display names, for the coordinator prompt and for the
        *"valid agents: …"* hint on a failed invite;
        :func:`~clawmeets.models.project_plan.ensure_consultation_roster` wants
        ``id``, ``name`` and ``registered_by``, because it is appending room
        membership and tagging cross-owner invitees ``external``.

        **This set is also the allowlist**, which is why the roster sync
        needs no separate allowlist check: everything here has already passed
        :meth:`Project.matches_invitable`, the one owner of that rule.
        """
        viewer_owner_id = project._invitable_viewer_owner_id(model_ctx)
        candidates = [
            a
            for a in cls.list_all(model_ctx, viewer_is_admin=True)
            if a.id not in exclude_ids
        ]
        if project.enforces_invitable_allowlist:
            return project.resolve_invitable_agents(candidates, model_ctx)
        # No allowlist: the invitable set is the owner's own crew, the
        # same "Your agents" AGENTS.md shows.
        return [
            a
            for a in candidates
            if viewer_owner_id and a.registered_by == viewer_owner_id
        ]

    @classmethod
    def invitable_short_names_for_project(
        cls,
        project: "Project",
        model_ctx: "ModelContext",
        *,
        exclude_ids: frozenset[str] = frozenset(),
    ) -> list[str]:
        """:meth:`invitable_agents_for_project`, as the names a human types.

        Used to surface the invitable list in the coordinator prompt and to tell
        a coordinator which names are valid when a ``create_room`` invite 404s.
        """
        viewer_owner_id = project._invitable_viewer_owner_id(model_ctx)
        matched = cls.invitable_agents_for_project(
            project, model_ctx, exclude_ids=exclude_ids
        )

        names: list[str] = []
        for other in matched:
            # Strip ``{owner}-`` prefix when the agent is owned by the same
            # user (display name); cross-owner agents keep the full registry
            # name.
            if (
                viewer_owner_id
                and other.registered_by == viewer_owner_id
                and "-" in other.name
            ):
                names.append(other.name.split("-", 1)[1])
            else:
                names.append(cls.short_name(other.name, None))
        return names

    @property
    def linked_user_id(self) -> Optional[str]:
        """User this agent is linked to (alias for registered_by).

        Returns the user ID of whoever registered this agent. Every agent is
        owned by exactly one user; public agents are visible to all, private
        agents (``discoverable_through_registry=false``) are visible only to
        their owner's coordinator via AGENTS.md.
        """
        return self.registered_by

    def _coordinator_dm_action_schema(
        self, project: "Project"
    ) -> tuple[dict, bool]:
        """Pick the action schema + ``dm_is_owned`` flag for a coordinator turn.

        Owned DM (the user's own 1:1 with their assistant) is restricted to
        ``reply`` + ``update_file`` — cross-domain work moves into a scoped
        project via the proposal flow, not orphan workrooms. FD-tunneled
        DM-shaped projects (foreign coordinator) keep the full coordinator
        schema since ``create_room`` is how the FD's actual workspace
        materializes. Non-DM regular projects always use the full schema.

        ``created_by == registered_by`` alone can't tell an owned DM from an
        FD channel the agent hosts for a foreign requester: FD channels are
        stamped to the host (the agent's own owner), so that equality is true
        for *every* FD channel this agent hosts. Gate on
        ``fd_requester_name is None`` as well — set for ``-fd-`` tunnel names
        and None for genuine ``-dm-`` owned DMs — so FD channels fall through
        to the full coordinator schema (``create_room`` + roster restored).
        """
        if (
            project.is_dm_project
            and project.fd_requester_name is None
            and project.created_by is not None
            and project.created_by == self.registered_by
        ):
            return WORKER_ACTION_SCHEMA, True
        return COORDINATOR_ACTION_SCHEMA, False

    async def on_message(
        self,
        project_id: str,
        chatroom_name: str,
        message: "ChatMessage",
        addressed_to_me: bool,
        trigger_version: int,
    ) -> None:
        """Route a message based on the agent's role in this project.

        If the agent is the project's coordinator, handle as coordinator
        (user-communication → user-request handler; other rooms → coordinate).
        Otherwise handle as worker (only when addressed).
        """
        from .project import Project
        from .chatroom import Chatroom

        project = Project.get(project_id, self._model_ctx)
        if project is None:
            logger.warning(
                f"Agent {self.name}: project {project_id[:8]} not found locally; skipping message"
            )
            return

        if self.is_coordinator_for(project):
            chatroom = Chatroom.get(project_id, chatroom_name, self._model_ctx)
            if chatroom is None:
                logger.warning(
                    f"Agent {self.name} (coordinator): chatroom {chatroom_name!r} not found "
                    f"for project {project_id[:8]}; skipping"
                )
                return
            if chatroom.is_user_communication_room:
                # Fire-once DM auto-title (M4c), AWAITED within the turn (SF3) so
                # msg-2 already sees the flipped "New chat" latch. Best-effort: it
                # never blocks or aborts the substantive reply below.
                await self._maybe_autotitle_dm_thread(project, chatroom, message)
                await self._handle_user_request(project_id, chatroom_name, message, trigger_version)
            elif addressed_to_me:
                await self._coordinate(project_id, chatroom_name, message, trigger_version)
            return

        # Worker mode
        if not addressed_to_me:
            return
        await self._execute_task(project_id, chatroom_name, message, trigger_version)

    async def _maybe_autotitle_dm_thread(
        self,
        project: "Project",
        chatroom: "Chatroom",
        message: "ChatMessage",
    ) -> None:
        """Fire-once DM-thread auto-title, driven from ``on_message`` (R1).

        Overwrites the seeded ``"New chat"`` placeholder from the FIRST
        principal message alone (ChatGPT-style). Fire-once holds via three
        independent guards (R2): the in-process ``"New chat"`` latch (this method
        no-ops the moment the placeholder is gone), runner per-project
        serialization (this is awaited within the turn — SF3), and the
        replay-idempotent overwrite applied server-side. Runs on the side that
        has a running coordinator:

        - **owned DM** — principal = ``created_by`` (the human owner), same as
          before.
        - **host Front Desk end** — the foreign agent's runloop composes the
          title from the REQUESTER's mirrored first message (principal = that
          requesting human). The requester (local) end has no running
          coordinator, so its row re-titles via the reverse tunnel mirror of the
          resulting ``DISPLAY_NAME_CHANGED`` (see ``TunnelSubscriber``).

        Best-effort throughout: any failure is swallowed/logged so a transient
        error never blocks or aborts the agent's reply.
        """
        from .project import NEW_CHAT_PLACEHOLDER

        # guard 1 — dm-shaped only: owned DM OR Front Desk host end (R3).
        if not project.is_dm_shaped:
            return
        # guard 2 — placeholder latch (R2-i): only title while still "New chat"
        if (project.display_name or "") != NEW_CHAT_PLACEHOLDER:
            return
        # guard 3 — TITLING-PRINCIPAL-authored, non-ack, real content only.
        #   owned DM  -> principal = created_by (the human owner).
        #   host FD   -> principal = the human REQUESTER who authored the
        #               mirrored first message (from_participant_id is the author
        #               unless it's an agent — see _dm_title_principal_id's
        #               agent-negative resolution). None-safe: if the author is
        #               an agent/unknown we skip (never mis-attribute or crash).
        principal_id = self._dm_title_principal_id(project, message)
        if (
            message.is_ack
            or principal_id is None
            or message.from_participant_id != principal_id
        ):
            return
        content = (message.content or "").strip()
        if not content:
            return
        if self._model_ctx.client is None:
            return

        # guard 4 — low-signal defer with budget + terminal fallback (R5/MF3).
        # Filter on the titling principal (not created_by) so the host FD end
        # accumulates the REQUESTER's messages, not the host owner's.
        user_msgs = [
            m for m in chatroom.get_messages()
            if not m.is_ack and m.from_participant_id == principal_id
        ]
        if _dm_title_is_too_terse(content):
            if len(user_msgs) < _DM_TITLE_DEFER_BUDGET:
                return  # DEFER: latch stays "New chat"; a later message titles it
            # Budget exhausted on a persistently terse thread: converge off
            # "New chat" using the accumulated user content (terminal fallback).
            source_text = " ".join(m.content for m in user_msgs).strip() or content
        else:
            # Substantive message: title from the later triggering message (MF3),
            # i.e. THIS message's content alone.
            source_text = content

        try:
            title = await self._generate_dm_thread_title(source_text)
            if not title or title == NEW_CHAT_PLACEHOLDER:
                return
            # expected_current = literal "New chat" sentinel (MF2), never the
            # agent's last-observed title — awaited within the turn (SF3).
            await self._model_ctx.client.set_project_display_name(
                project.id, title, expected_current=NEW_CHAT_PLACEHOLDER,
            )
        except Exception:
            logger.warning(
                f"Agent {self.name}: DM auto-title failed for project "
                f"{project.id[:8]} (non-fatal); leaving 'New chat'",
                exc_info=True,
            )

    async def _maybe_autotitle_project(self, project: "Project") -> None:
        """Fire-once auto-title for a REGULAR project, driven from
        ``on_first_user_request`` (the coordinator's first turn).

        Regular projects are created with a caller-chosen kebab slug (``name``)
        and no ``display_name``, so every label falls back to the raw slug. This
        gives them the same human-readable title DM threads get — generated from
        the project's ``request`` (its canonical goal statement) via the shared
        one-shot title path, then persisted through ``DISPLAY_NAME_CHANGED``.

        Fire-once holds on the ``display_name is None`` latch (a titled project
        has a non-null ``display_name``, so re-runs/replays no-op). Best-effort:
        the caller swallows/logs any failure so titling never blocks or aborts
        coordinator setup. On failure the meaningful slug simply remains — no
        placeholder is seeded, unlike the DM path's ``"New chat"``.
        """
        # dm/frontdesk are handled by _maybe_autotitle_dm_thread — never double-title.
        if project.is_dm_shaped:
            return
        # Fire-once latch: only title while display_name is still unset.
        if project.display_name is not None:
            return
        if self._model_ctx.client is None:
            return
        source_text = (project.request or "").strip()
        if not source_text:
            return
        title = await self._generate_dm_thread_title(source_text)
        if not title:
            return
        # Regular projects carry no "New chat" sentinel; the None latch above is
        # the guard, so expected_current is None (skips the server's defensive
        # compare).
        await self._model_ctx.client.set_project_display_name(
            project.id, title, expected_current=None,
        )

    def _dm_title_principal_id(
        self, project: "Project", message: "ChatMessage"
    ) -> Optional[str]:
        """Who must have authored the message for it to seed the auto-title.

        - **owned DM** (``surface == "dm"``): the project owner ``created_by``.
        - **host Front Desk end** (``surface == "frontdesk"``): the human
          REQUESTER who authored the mirrored first message —
          ``message.from_participant_id`` UNLESS that id is an agent.

        Resolution is **agent-NEGATIVE**, and deliberately so. This method runs
        on the host coordinator's OWN runner, whose ``participants_dir`` carries
        neither the server's ``passwd`` nor the coordinator's own card. A
        positive ``User.get(from_participant_id)`` could therefore NEVER resolve
        the requester off-server — it silently returned ``None`` and the host
        never emitted ``DISPLAY_NAME_CHANGED`` (the DD-title path stayed dead in
        production). So we invert the test: the author IS the human requester
        unless it is an agent — the coordinator itself (``self._id``, which the
        runner never syncs into its own ``agents/``) or any synced peer ``Agent``
        card. None-safe: a blank/unknown author -> ``None`` (SF2 — skip titling,
        never mis-attribute to the host owner or crash).
        """
        if project.is_frontdesk_project:
            sender = message.from_participant_id
            if (
                not sender
                or sender == self._id
                or Agent.get(sender, self._model_ctx) is not None
            ):
                return None
            return sender
        return project.created_by

    async def _generate_dm_thread_title(self, source_text: str) -> str:
        """Generate a short thread title from user content alone (first-message
        design). Uses the provider's one-shot ``generate_text`` — which carries a
        deterministic-truncation default on every provider (MF5), so this never
        strands a thread even if the provider hasn't implemented real generation
        or a live call errors — then normalizes the result. Falls back to a
        deterministic clean of ``source_text`` if the generation is empty.
        """
        cli = self._model_ctx.cli
        raw = ""
        if cli is not None:
            try:
                raw = await cli.generate_text(
                    _DM_TITLE_INSTRUCTION.format(message=source_text)
                )
            except Exception:
                logger.warning(
                    f"Agent {self.name}: generate_text raised during DM auto-title; "
                    "using deterministic fallback",
                    exc_info=True,
                )
                raw = ""
        title = _clean_dm_title(raw)
        if not title:
            title = _clean_dm_title(source_text)
        return title

    async def _emit_acknowledgment(
        self,
        project_id: str,
        chatroom_name: str,
        trigger_version: int,
    ) -> None:
        """Emit an acknowledgment message before processing.

        Best-effort: the ack is non-essential UI sugar (it feeds
        ``clawmeets user listen``). A transient server error here must NOT abort
        the substantive turn that follows — otherwise a single 503 on this POST
        propagates up through ``on_batch_complete`` → ``runloop.sync`` and wedges
        the project (the runloop cursor only advances on full success).
        """
        if not self._model_ctx.client:
            return

        try:
            await self._model_ctx.client.post_message(
                project_id=project_id,
                chatroom_name=chatroom_name,
                content="Message received, processing...",
                source_version=trigger_version,
                is_ack=True,
            )
        except Exception:
            logger.warning(
                f"Agent {self.name}: acknowledgment post failed (non-fatal); continuing",
                exc_info=True,
            )

    async def _execute_task(
        self,
        project_id: str,
        chatroom_name: str,
        message: "ChatMessage",
        trigger_version: int,
    ) -> None:
        """
        Execute the task requested in the message using Claude.

        Uses composition objects for prompt building, execution, and action execution.
        Requires cli and action_executor to be configured.
        Prompt builder is created on-demand based on operational mode.
        """
        agent_name = self.name  # Use property to get current name

        if not self._model_ctx.cli:
            logger.warning(f"Agent {agent_name}: CLI not configured, cannot execute task")
            return

        action_executor = self._model_ctx.action_executor
        if not action_executor:
            logger.warning(f"Agent {agent_name}: Client not configured, cannot execute task")
            return

        # Emit acknowledgment before processing
        await self._emit_acknowledgment(project_id, chatroom_name, trigger_version)

        # Get project for context
        from .project import Project
        project = Project.get(project_id, self._model_ctx)

        # Compute project-aware paths from ModelContext
        data_dir = self._model_ctx.project_dir(project_id, project.name)
        sandbox_dir = self._model_ctx.sandbox_dir(project_id, project.name)
        log_dir = self._model_ctx.llm_log_dir(project_id, project.name)

        additional_dirs = self._extra_dirs(data_dir, sandbox_dir)

        # Create prompt builder on-demand for worker mode
        # Use project.coordinator_name to avoid lookup (worker may not have coordinator's card)
        prompt_builder = create_prompt_builder(
            OperationalMode.WORKER,
            coordinator_name=project.coordinator_name,
            capabilities=self.capabilities,
            git_url=self._model_ctx.git_url,
        )

        is_dm = project.is_dm_shaped  # owned DM OR Front Desk end -> DM prompt variant
        inbound_content = message.content

        from .chatroom import Chatroom
        chatroom = Chatroom.get(project_id, chatroom_name, self._model_ctx)
        if is_dm and chatroom is not None:
            resume_marker = _resume_marker_for_dm(chatroom, self.name)
            if resume_marker and resume_marker not in inbound_content:
                inbound_content = f"{resume_marker}\n{inbound_content}"

        chat_history = (
            chatroom.recent_history_for_prompt(exclude_message_id=message.id)
            if chatroom is not None
            else None
        )

        # Build prompt - extract message fields for Layer 0 compatibility
        prompt = prompt_builder.build_prompt(
            name=self.name,
            description=self.description,
            project_id=project_id,
            chatroom_name=chatroom_name,
            from_participant_name=message.from_participant_name or message.from_participant_id,
            message_content=inbound_content,
            data_dir=data_dir,
            project_name=project.name,
            agent_dir=self._model_ctx.base_dir,
            knowledge_dirs=self._model_ctx.knowledge_dirs,
            dwh_dir=self._model_ctx.dwh_dir,
            is_dm=is_dm,
            chat_history=chat_history,
        )

        # Execute using ClaudeCLI with retry for transient failures
        action_block, usage, validation_notes = await self._invoke_validated(
            project_id=project_id,
            chatroom_name=chatroom_name,
            prompt=prompt,
            sandbox_dir=sandbox_dir,
            log_dir=log_dir,
            additional_dirs=additional_dirs,
            action_schema=WORKER_ACTION_SCHEMA,
            trigger_version=trigger_version,
            role=derive_role(self.name, is_coordinator=False),
            model_config_name=message.model_config_name,
        )

        logger.info(
            f"Agent {agent_name}: Claude invocation complete "
            f"(cost=${usage.cost_usd:.4f}, tokens in={usage.input_tokens} out={usage.output_tokens})"
        )

        # Process using ActionBlockExecutor - executes actions via HTTP
        replied_chatrooms = await action_executor.process(
            action_block=action_block,
            project_id=project_id,
            sandbox_dir=sandbox_dir,
        )
        await self._post_validation_notes(project_id, validation_notes, trigger_version)

        # If the LLM didn't reply in the triggering chatroom, post a closure
        # so the server marks this worker as responded and clears PendingWork.
        # Without this, the typing indicator would persist until batch timeout.
        if chatroom_name not in replied_chatrooms:
            await self._emit_no_action_message(
                project_id, chatroom_name, trigger_version, replied_chatrooms
            )

    def _runner_self_card(self) -> dict:
        """Read this runner's own top-level card.json.

        The runner's self-card lives at ``participants_dir/card.json`` (NOT under
        ``agents/`` where ``_load_card`` looks — that is the synced-peer layout).
        Carries the named ``model_configs`` mirrored from the server, which the
        per-request override resolves against.
        """
        path = self._model_ctx.participants_dir / "card.json"
        return FileUtil.read(path, "json") or {}

    def _build_override_cli(self, model_config_name: "str | None"):
        """Build a one-turn LLM provider for a per-request model override.

        Returns an ``LLMProvider`` built from the named config, or ``None`` to
        fall back to the agent's default ``model_ctx.cli``. Returns ``None``
        silently only when there is nothing to honor: no name given, no factory
        attached (off-runner / tests), or no config resolves (the card has no
        matching config and no default). An unknown/stale name still degrades to
        the default config via ``resolve()`` and builds normally (spec #3).

        An explicit per-request selection is authoritative: whenever it resolves
        to a config we build from that config, even when it equals the current
        default. We deliberately do NOT short-circuit to ``model_ctx.cli`` on an
        equals-default match — ``model_ctx.cli`` only reflects the default if the
        AGENT_SETTINGS_CHANGE hot-swap fired and succeeded in *this* process
        since the default last moved, which is not guaranteed (a runner that
        started before the config existed keeps a stale CLI). Building the
        provider here is construction-only (no subprocess; the binary runs at
        invoke time), so honoring the selection every turn is cheap.

        Raises ``LLMInvocationError`` when a resolved config's provider CANNOT be
        built (e.g. the selected CLI binary is missing/broken). The caller turns
        this into a user-visible error and aborts the turn — silently answering
        on a different model than the one explicitly selected is wrong.
        """
        if not model_config_name:
            return None
        factory = self._model_ctx.cli_factory
        if factory is None:
            return None
        card = self._runner_self_card()
        # resolve_raw() falls back to the default config for an unknown/stale
        # name, so cfg is None only when the card has no default at all — nothing
        # to honor, so fall back to model_ctx.cli. We use resolve_raw (not the
        # redacted resolve) because we need the config's own RAW api_key below.
        cfg = _model_config.resolve_raw(card, model_config_name)
        if cfg is None:
            return None
        settings = dict(card.get("local_settings") or {})
        settings["llm_provider"] = cfg.get("provider")
        settings["llm_model"] = cfg.get("model")
        settings["llm_base_url"] = cfg.get("base_url")
        # Feed the SELECTED config's own BYO key to the provider builder. This
        # overrides any default-config key already mirrored onto local_settings
        # so a NON-default selection uses ITS key, not the default's. None for a
        # keyless CLI config, which the factory treats as "no BYO key" (env-var
        # fallback for -api providers). The raw key never leaves the runner.
        settings["llm_api_key"] = cfg.get("api_key")
        try:
            provider = factory(settings)
        except Exception as e:  # noqa: BLE001 — surface, don't silently downgrade
            logger.warning(
                f"Agent {self.name}: selected model "
                f"'{model_config_name}' (provider={cfg.get('provider')!r}) "
                f"failed to build: {e}",
                exc_info=True,
            )
            raise LLMInvocationError(
                f"Selected model {model_config_name!r} is unavailable: {e}"
            ) from e
        logger.info(
            f"Agent {self.name}: per-request model override "
            f"'{model_config_name}' → provider={cfg.get('provider')!r} "
            f"model={cfg.get('model')!r}"
        )
        return provider

    def _extra_dirs(self, data_dir: Path, sandbox_dir: Path) -> list[Path]:
        """Extra read/write roots beyond the sandbox cwd, shared by every
        invoke site.

        Order: the synced project data dir (only when distinct from the
        sandbox cwd), the knowledge dirs, then the agent ``memory_dir``. The
        memory dir is appended so every provider that gates file access on the
        allow-list (gemini ``--include-directories``, codex/claude
        ``--add-dir``) can reach ``memory/`` — the reflect/personalize
        read+write target that previously sat outside every allowed root.
        Centralizing the four formerly-identical blocks here keeps that wiring
        from drifting between call sites again.
        """
        dirs: list[Path] = []
        if data_dir != sandbox_dir:
            dirs.append(data_dir)
        dirs.extend(self._model_ctx.knowledge_dirs)
        dirs.append(self._model_ctx.memory_dir)
        return dirs

    async def _invoke_with_transient_retry(
        self,
        *,
        project_id: str,
        chatroom_name: str,
        prompt: str,
        sandbox_dir: Path,
        log_dir: Path,
        additional_dirs: list[Path],
        action_schema: dict,
        trigger_version: int,
        role: str,
        correction: "str | None" = None,
        override_cli=None,
    ) -> "tuple[ActionBlock, LLMUsage]":
        """One model invocation, with the existing transient-failure retry.

        Extracts the per-call-site transient-retry loop (rate-limit → surface &
        raise; transient CLI failure → bounded exponential backoff) so all four
        turn methods share one copy. ``correction`` (the validation feedback) is
        threaded to ``invoke_with_registry`` and appended to the prompt on a
        semantic-validation retry.
        """
        retry_delay = _INITIAL_RETRY_DELAY
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return await _invoke_with_registry(
                    self._model_ctx,
                    project_id,
                    chatroom_name,
                    prompt,
                    sandbox_dir,
                    log_dir,
                    additional_dirs,
                    action_schema=action_schema,
                    trigger_version=trigger_version,
                    role=role,
                    correction=correction,
                    override_cli=override_cli,
                )
            except LLMRateLimitError as e:
                logger.warning(
                    f"Agent {self.name}: rate limited "
                    f"(type={e.rate_limit_type}, resets={e.resets_at_human})"
                )
                await self._post_error_notification(
                    project_id, chatroom_name, self.name, e, trigger_version
                )
                raise
            except LLMInvocationError as e:
                if attempt < _MAX_RETRIES and _is_transient_error(e):
                    logger.warning(
                        f"Agent {self.name}: transient failure "
                        f"(attempt {attempt + 1}/{_MAX_RETRIES + 1}), "
                        f"retrying in {retry_delay}s: {e}"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    await self._post_error_notification(
                        project_id, chatroom_name, self.name, e, trigger_version
                    )
                    raise
        # Unreachable: the loop either returns or raises on every path.
        raise AssertionError("transient-retry loop exited without a result")

    async def _invoke_validated(
        self,
        *,
        project_id: str,
        chatroom_name: str,
        prompt: str,
        sandbox_dir: Path,
        log_dir: Path,
        additional_dirs: list[Path],
        action_schema: dict,
        trigger_version: int,
        role: str,
        model_config_name: "str | None" = None,
    ) -> "tuple[ActionBlock, LLMUsage, list[str]]":
        """Invoke → validate → maybe retry, up to ``_MAX_VALIDATION_RETRIES``.

        Provably terminating (plan §4): the state snapshot is taken ONCE before
        the loop (a closed target), the budget is a hard integer counter, and
        only a REJECT_RETRY classification re-invokes — NO_OP / PASS never do,
        so idempotent actions (create-existing-room, re-complete) cannot spin.
        Never keys on free-form content, so valid-but-non-deterministic prose
        triggers zero retries.

        **The invitee set is refreshed between attempts, and only ever widened.**
        A coordinator's remedy for a non-invitable invitee — ``clawmeets project
        allowlist`` — lands *inside* the turn that discovers the problem, so a
        set frozen at turn start could never see it: the coordinator widens the
        roster, is rejected by the stale set, assumes it has the name wrong, and
        burns the whole budget on spelling variants while the action is dropped.
        Between attempts the loop therefore re-derives just that one field and
        UNIONS it in (``StateSnapshot.with_invitable``), then RE-VALIDATES the
        block it already has. If the widening rescued it, the turn breaks out
        with no extra invocation at all — the one-shot case (widen the roster and
        open the room in the same turn) costs zero retries. Nothing else in the
        snapshot moves, so the target stays closed; it is now monotone rather
        than motionless, which still forbids oscillation, and termination was
        always the integer budget's job anyway.

        At budget exhaustion the terminal fallback drops the still-offending
        actions, keeps the valid ones, and surfaces a note (identical to today's
        4xx behavior — no regression). Returns the (possibly filtered) block,
        its usage, and the notes to post to user-communication.
        """
        include_agents = _schema_allows_agent_invites(action_schema)
        snapshot = build_state_snapshot(
            self._model_ctx, project_id, include_agents=include_agents
        )

        # Resolve a per-request model override once, before the retry loop, so
        # every attempt (incl. validation corrections) runs on the same
        # one-turn provider. None ⇒ the agent's default model_ctx.cli. If an
        # explicitly-selected model can't be built (broken/missing CLI), surface
        # a user-visible error and abort the turn — never silently answer on the
        # default model. Mirrors the rate-limit post-then-raise pattern.
        try:
            override_cli = self._build_override_cli(model_config_name)
        except LLMInvocationError as e:
            await self._post_error_notification(
                project_id, chatroom_name, self.name, e, trigger_version
            )
            raise

        correction: "str | None" = None
        block: "ActionBlock"
        usage: "LLMUsage"
        result = None
        for _attempt in range(_MAX_VALIDATION_RETRIES + 1):
            block, usage = await self._invoke_with_transient_retry(
                project_id=project_id,
                chatroom_name=chatroom_name,
                prompt=prompt,
                sandbox_dir=sandbox_dir,
                log_dir=log_dir,
                additional_dirs=additional_dirs,
                action_schema=action_schema,
                trigger_version=trigger_version,
                role=role,
                correction=correction,
                override_cli=override_cli,
            )
            result = ActionValidator().validate(block, snapshot)
            if not result.retryable:
                break
            # Did the roster widen under us (the coordinator's own allowlist
            # edit landing mid-turn)? Re-validating the block we already have is
            # pure and free — the validator does no I/O — so a rescued block
            # costs zero further invocations.
            if include_agents:
                widened = snapshot.with_invitable(
                    refresh_invitable_agents(self._model_ctx, project_id)
                )
                if widened is not snapshot:
                    snapshot = widened
                    result = ActionValidator().validate(block, snapshot)
                    if not result.retryable:
                        break
            correction = result.feedback()

        # ``result.retryable`` is True here only if the budget was exhausted with
        # a hard-invalid referent still present -> terminal fallback (drop them).
        drop = result.retryable
        block.actions = result.surviving_actions(drop_retryable=drop)
        notes = result.notes()
        if drop:
            notes = notes + result.dropped_notes()
        return block, usage, notes

    async def _post_validation_notes(
        self,
        project_id: str,
        notes: list[str],
        trigger_version: int,
    ) -> None:
        """Post no-op / dropped-action notes to user-communication.

        Same surfacing channel as the executor's 4xx failure note, so a skipped
        or dropped action is visible to the coordinator/user on the next
        dispatch. Silent-fails per note so a missing user-communication room
        can't mask the turn's real work.
        """
        if not notes:
            return
        client = self._model_ctx.client
        if client is None:
            return
        for note in notes:
            try:
                await client.post_message(
                    project_id=project_id,
                    chatroom_name="user-communication",
                    content=note,
                    source_version=trigger_version,
                )
            except Exception:
                logger.exception("Failed to post validation note to user-communication")

    async def _post_error_notification(
        self,
        project_id: str,
        chatroom_name: str,
        agent_name: str,
        error: Exception,
        trigger_version: int,
    ) -> None:
        """Post an error notification to the chatroom when a task fails.

        Posts with is_ack=False: the error is the worker's terminal report,
        not a transient ack. Counting it as a batch response frees the
        typing chip immediately and lets the coordinator pivot in seconds
        instead of waiting for batch timeout.
        """
        if not self._model_ctx.client:
            return

        try:
            if isinstance(error, LLMRateLimitError):
                self.update_status(AgentStatus.RATE_LIMITED)
                reset_info = f" Resets at {error.resets_at_human}." if error.resets_at_human else ""
                content = f"I've hit a rate limit and cannot continue.{reset_info}"
            else:
                detail = str(error).strip()
                max_len = 500
                if len(detail) > max_len:
                    detail = detail[:max_len] + "…"
                content = f"I encountered an error and couldn't complete this task:\n{detail}"

            await self._model_ctx.client.post_message(
                project_id=project_id,
                chatroom_name=chatroom_name,
                content=content,
                source_version=trigger_version,
                is_ack=False,
            )
        except Exception:
            logger.error(f"Agent {agent_name}: failed to post error notification", exc_info=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Coordinator Callbacks (when Agent is project coordinator)
    # ─────────────────────────────────────────────────────────────────────────

    async def on_batch_complete(
        self,
        project_id: str,
        chatroom_name: str,
        message_id: str,
        responded_participants: list[str],
        trigger_version: int,
    ) -> None:
        """Handle batch completion when acting as coordinator.

        Only triggers if this agent is the project's coordinator
        (determined by project.coordinator_id).
        """
        from .project import Project
        project = Project.get(project_id, self._model_ctx)
        if not project or not self.is_coordinator_for(project):
            return  # Not the coordinator for this project

        logger.info(
            f"Agent {self.name} (as coordinator): Batch complete in {chatroom_name}, "
            f"responded: {responded_participants}"
        )
        await self._process_batch_results(
            project_id, chatroom_name, message_id, responded_participants, timed_out=[],
            trigger_version=trigger_version,
        )

    async def on_plan_gate_prime(self, project_id: str) -> None:
        """Seed the gate memo from state that predates every entry to come.

        Fired by the notifier out of ``ChangelogRunloop.load_state`` — the one
        moment at which the synced ``meta.json`` is guaranteed to hold the value
        the gate had *before* anything this process will apply. That is a value
        :meth:`on_plan_state_change` structurally cannot read for itself:
        ``ModelContext`` runs at priority 0 and has already written the entry by
        the time the notifier sees it.

        **Without this, the fix has a hole shaped exactly like the bug.** The
        memo is per-process and the runloop resumes from a *persisted* cursor,
        so a restart never replays the ``PROJECT_PLAN_STATE`` entries that would
        re-prime it. Project stalls on an open note → runner restarts → the user
        resolves the last note through ``POST /plan/notes/{id}/resolve``, which
        posts no message → the release is recorded as a first observation and
        swallowed → the project stays stopped with nothing on any surface left
        to wake it. Priming makes a restart indistinguishable from a live
        observation.

        Silent on a project this agent does not coordinate, and on one whose
        ``meta.json`` is not on disk yet: neither has a gate this agent reads.
        """
        from .project import Project
        project = Project.get(project_id, self._model_ctx)
        if not project or not self.is_coordinator_for(project):
            return
        self._plan_gate_blocked[project_id] = bool(
            _plan_execution_blocked(project)
        )

    async def on_plan_state_change(
        self,
        project_id: str,
        trigger_version: int,
    ) -> None:
        """A ``PROJECT_PLAN_STATE`` batch landed. Resume if it OPENED the gate.

        **The one re-dispatch the runner mints itself, and it is not
        self-continuation.** The signal is the USER's act — resolving the last
        note addressed to them — arriving as a changelog entry like any other.
        What is unusual is only that the entry carries no message, so before
        this arm existed nothing read it.

        Two holes it closes, and the first is not a race at all:

        * ``POST /plan/notes/{id}/resolve`` publishes the projection and posts
          nothing. Resolve the last blocking note that way and the project
          simply never moved again — no stale read required.
        * A route that publishes the projection AFTER the message that wakes on
          it leaves that turn reading the pre-resolution count, because the turn
          holds the runloop lock this entry needs and so cannot re-read it.
          §7.4's gate then NO_OPs the turn's ``create_room`` and it stops; the
          entry drains into silence behind it. ``submit_review``'s prelude
          ordering is the fix for that one, and this is the net under it —
          including under the next route to get the ordering wrong. The
          notifier's version comparison is what keeps the net from ALSO firing
          when the ordering is right and the woken turn is already correct.

        **THIS ARM NOW CARRIES ACCEPTANCE TOO, and that is a merge rather
        than a widening.** There used to be a second gate and a second wake-up:
        ``POST /plan/approve`` posted an acceptance marker as the user, and this
        method was deliberately narrowed to the execution gate so an approval
        would not fire both. Both of those are gone. Acceptance is the user
        applying the **go-note**, which is an ordinary open plan note, so it
        moves this one gate and nothing else — the double-dispatch the old
        narrowing existed to prevent cannot be constructed any more.

        The two doors that apply it land differently and both are covered:
        the tray's Accept goes through ``submit_review``, which posts a real
        message to the coordinator with the projection in the **same batch's
        prelude**, so the notifier's version comparison sees the woken turn
        already reading post-acceptance state and this arm stays quiet;
        ``plan resolve --apply`` posts nothing at all, and this arm is the only
        thing that starts the work. The old marker's best-effort failure mode —
        a project whose ``user-communication`` room is missing never getting its
        wake-up — went with the marker.

        **On the TRANSITION, never on the level.** An entry exists only because
        a lifecycle field moved (``publish_plan_state`` is idempotent), but
        plenty of those moves are a note being FILED, which must wake nobody.
        Resuming on ``blocked == 0`` alone would re-run the coordinator on every
        unrelated plan edit for the rest of the project's life.

        A project with no memo at all is one whose ``meta.json`` did not exist
        when :meth:`on_plan_gate_prime` ran — a project created after this
        process started, which has no stalled turn to rescue.
        """
        from .project import Project
        project = Project.get(project_id, self._model_ctx)
        if not project or not self.is_coordinator_for(project):
            return

        # ModelContext (priority 0) applied the batch before the notifier's
        # `on_sync_complete` ran, so this reads the POST-batch gate; the memo —
        # primed at load, then carried forward here — supplies the other half.
        blocked = bool(_plan_execution_blocked(project))
        was_blocked = self._plan_gate_blocked.get(project_id)
        self._plan_gate_blocked[project_id] = blocked
        if was_blocked is not True or blocked:
            return

        logger.info(
            f"Agent {self.name} (as coordinator): plan gate released in "
            f"project {project_id[:8]}, resuming"
        )
        await self._process_batch_results(
            project_id,
            "user-communication",
            "",
            [],
            timed_out=[],
            trigger_version=trigger_version,
            gate_released=True,
        )

    async def on_batch_timeout(
        self,
        project_id: str,
        chatroom_name: str,
        message_id: str,
        responded_participants: list[str],
        timed_out_participants: list[str],
        trigger_version: int,
    ) -> None:
        """Handle batch timeout when acting as coordinator.

        Only triggers if this agent is the project's coordinator.
        """
        from .project import Project
        project = Project.get(project_id, self._model_ctx)
        if not project or not self.is_coordinator_for(project):
            return  # Not the coordinator for this project

        logger.info(
            f"Agent {self.name} (as coordinator): Batch timeout in {chatroom_name}, "
            f"responded: {responded_participants}, timed out: {timed_out_participants}"
        )
        await self._process_batch_results(
            project_id, chatroom_name, message_id, responded_participants,
            timed_out=timed_out_participants,
            trigger_version=trigger_version,
        )

    async def on_first_user_request(
        self,
        project_id: str,
        chatroom_name: str,
        message: "ChatMessage",
        context_files: list[str],
        trigger_version: int,
    ) -> None:
        """Handle first user request when acting as coordinator.

        Uses coordinator setup prompt to analyze context, create plan,
        and delegate initial tasks.

        Only triggers if this agent is the project's coordinator.
        Falls back to normal on_message() for worker mode.
        """
        from .project import Project
        project = Project.get(project_id, self._model_ctx)
        if not project or not self.is_coordinator_for(project):
            # Not the coordinator - fall back to normal message handling
            await self.on_message(
                project_id=project_id,
                chatroom_name=chatroom_name,
                message=message,
                addressed_to_me=True,
                trigger_version=trigger_version,
            )
            return

        logger.info(
            f"Agent {self.name} (as coordinator): Handling first user request "
            f"in project {project_id[:8]}"
        )
        # Fire-once auto-title from the request. Best-effort: never blocks or
        # aborts setup (mirrors the DM auto-title contract).
        try:
            await self._maybe_autotitle_project(project)
        except Exception:
            logger.warning(
                f"Agent {self.name}: project auto-title failed for "
                f"{project_id[:8]} (non-fatal); leaving slug label",
                exc_info=True,
            )
        await self._invoke_coordinator_setup(
            project_id=project_id,
            chatroom_name=chatroom_name,
            message=message,
            context_files=context_files,
            trigger_version=trigger_version,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Coordinator Implementation Methods
    # ─────────────────────────────────────────────────────────────────────────

    async def _process_batch_results(
        self,
        project_id: str,
        chatroom_name: str,
        message_id: str,
        responded_participants: list[str],
        timed_out: list[str],
        trigger_version: int,
        gate_released: bool = False,
    ) -> None:
        """Process batch results as coordinator.

        Invokes Claude with coordinator context to decide next steps:
        - Delegate more work
        - Summarize results
        - Complete project

        ``gate_released`` re-uses this whole path for a turn that is NOT a
        batch: :meth:`on_plan_state_change`. Everything a coordinator turn
        needs is identical — same prompt builder, same snapshot, same executor —
        and only the synthetic incoming message differs, so the alternative was
        sixty duplicated lines that would drift.
        """
        action_executor = self._model_ctx.action_executor
        if not self._model_ctx.cli or not action_executor:
            logger.warning(f"Agent {self.name}: Dependencies not configured for coordinator mode")
            return

        from .project import Project
        from .chatroom import Chatroom

        project = Project.get(project_id, self._model_ctx)
        chatroom = Chatroom.get(project_id, chatroom_name, self._model_ctx)

        # Compute project-aware paths from ModelContext
        data_dir = self._model_ctx.project_dir(project_id, project.name)
        sandbox_dir = self._model_ctx.sandbox_dir(project_id, project.name)
        log_dir = self._model_ctx.llm_log_dir(project_id, project.name)

        additional_dirs = self._extra_dirs(data_dir, sandbox_dir)

        # Build batch status — the incoming "message" the coordinator sees is
        # synthetic: "batch complete in <room>; responded X, timed out Y".
        # Recent conversation rides through the prompt's RECENT CHAT block;
        # deliverable files ride through the SYNCED PROJECT FILE MANIFEST.
        #
        # AC-5.7 — as NAMES, not ids. Both lists, because "who timed out" is
        # the half a coordinator acts on and it was equally unreadable.
        status_parts = [f"Batch complete in '{chatroom_name}'"]
        if responded_participants:
            responded_names = _participant_names(
                responded_participants, self._model_ctx
            )
            status_parts.append(f"Responded: {', '.join(responded_names)}")
        if timed_out:
            timed_out_names = _participant_names(timed_out, self._model_ctx)
            status_parts.append(f"Timed out: {', '.join(timed_out_names)}")
        batch_status = ". ".join(status_parts)

        # M5 AC-5.6 — ONE boolean picks BOTH the trailing instruction and the
        # role-contract block, so the synthetic message and the contract can
        # never tell this turn two different stories about what it is.
        spec_consult = not gate_released and _is_spec_consultation_batch(
            project, chatroom_name
        )

        # B5 — one constant, rendered identically here and by
        # ``scripts/dump_sample_prompts.py`` (AC-7.12). Three of them now: the
        # milestone one, its spec-stage sibling, and the gate-release turn that
        # is not a batch at all.
        if gate_released:
            batch_content = PLAN_GATE_RELEASED_INSTRUCTION
        else:
            instruction = (
                BATCH_COMPLETION_PLAN_REVIEW_INSTRUCTION
                if spec_consult
                else BATCH_COMPLETION_PLAN_INSTRUCTION
            )
            batch_content = f"{batch_status}.\n\n{instruction}"

        chat_history = (
            chatroom.recent_history_for_prompt()
            if chatroom is not None
            else None
        )

        # Create coordinator prompt builder for this task
        coordinator_builder = create_prompt_builder(
            OperationalMode.COORDINATOR,
            git_url=self._model_ctx.git_url,
        )
        assert isinstance(coordinator_builder, CoordinatorPromptBuilder)

        # Build prompt - agents are referenced via AGENTS.md file.
        # flow_context="batch" injects the BATCH COMPLETION WORKFLOW and
        # HANDLING WORKER QUESTIONS AND BLOCKERS blocks into the role
        # contract; otherwise they sit dormant on every user-comm turn
        # for no benefit. "spec-consult" (AC-5.6) swaps in the spec-stage
        # variant, off the same boolean as the instruction above.
        action_schema, dm_is_owned = self._coordinator_dm_action_schema(project)
        prompt = coordinator_builder.build_prompt(
            name=self.name,
            description=self.description,
            project_id=project_id,
            chatroom_name=chatroom_name,
            from_participant_name="System",
            message_content=batch_content,
            data_dir=data_dir,
            project_name=project.name,
            agent_dir=self._model_ctx.base_dir,
            knowledge_dirs=self._model_ctx.knowledge_dirs,
            dwh_dir=self._model_ctx.dwh_dir,
            is_dm=project.is_dm_shaped,
            dm_is_owned=dm_is_owned,
            invitable_agents=self._resolve_invitable_agents_for_prompt(project),
            chat_history=chat_history,
            flow_context="spec-consult" if spec_consult else "batch",
            plan=plan_prompt_state(project, self._model_ctx),
        )

        if not gate_released:
            # No ack on the gate-release turn: the ack answers a message, and
            # nobody sent one. Posting "Message received, processing..." into
            # `user-communication` unprompted reads as the coordinator talking
            # to itself.
            await self._emit_acknowledgment(
                project_id, chatroom_name, trigger_version
            )

        action_block, usage, validation_notes = await self._invoke_validated(
            project_id=project_id,
            chatroom_name=chatroom_name,
            prompt=prompt,
            sandbox_dir=sandbox_dir,
            log_dir=log_dir,
            additional_dirs=additional_dirs,
            action_schema=action_schema,
            trigger_version=trigger_version,
            role=derive_role(self.name, is_coordinator=True),
        )

        logger.info(
            f"Agent {self.name} (coordinator): Batch processing complete "
            f"(cost=${usage.cost_usd:.4f})"
        )

        replied_chatrooms = await action_executor.process(
            action_block=action_block,
            project_id=project_id,
            sandbox_dir=sandbox_dir,
        )
        await self._post_validation_notes(project_id, validation_notes, trigger_version)

        # If the coordinator moved on (created next milestone room, updated
        # PLAN.md, etc.) without replying in this room, post a closure so
        # the self-batch pending work clears and the typing chip goes away.
        if chatroom_name not in replied_chatrooms:
            await self._emit_no_action_message(project_id, chatroom_name, trigger_version)

    async def _invoke_coordinator_setup(
        self,
        project_id: str,
        chatroom_name: str,
        message: "ChatMessage",
        context_files: list[str],
        trigger_version: int,
    ) -> None:
        """Invoke Claude with coordinator setup prompt.

        Uses CoordinatorPromptBuilder.build_setup_prompt() for first request handling.
        """
        action_executor = self._model_ctx.action_executor
        if not self._model_ctx.cli or not action_executor:
            raise ValueError(f"Agent {self.name}: Dependencies not configured for coordinator setup")

        from .project import Project

        project = Project.get(project_id, self._model_ctx)

        # Compute project-aware paths from ModelContext
        data_dir = self._model_ctx.project_dir(project_id, project.name)
        sandbox_dir = self._model_ctx.sandbox_dir(project_id, project.name)
        log_dir = self._model_ctx.llm_log_dir(project_id, project.name)

        additional_dirs = self._extra_dirs(data_dir, sandbox_dir)

        # Create coordinator prompt builder for this task
        coordinator_builder = create_prompt_builder(
            OperationalMode.COORDINATOR,
            git_url=self._model_ctx.git_url,
        )
        assert isinstance(coordinator_builder, CoordinatorPromptBuilder)

        await self._emit_acknowledgment(project_id, chatroom_name, trigger_version)

        # DM-shaped projects (owned DM + Front Desk host end) skip the PLAN.md /
        # milestones setup prompt — wrong shape for a casual conversational
        # channel. Use the soft live coordinator prompt instead so the first
        # message is treated like any other.
        action_schema, dm_is_owned = self._coordinator_dm_action_schema(project)
        if project.is_dm_shaped:
            from .chatroom import Chatroom
            chatroom = Chatroom.get(project_id, chatroom_name, self._model_ctx)
            chat_history = (
                chatroom.recent_history_for_prompt(exclude_message_id=message.id)
                if chatroom is not None
                else None
            )
            prompt = coordinator_builder.build_prompt(
                name=self.name,
                description=self.description,
                project_id=project_id,
                chatroom_name=chatroom_name,
                from_participant_name=message.from_participant_name or message.from_participant_id,
                message_content=message.content,
                data_dir=data_dir,
                project_name=project.name,
                agent_dir=self._model_ctx.base_dir,
                knowledge_dirs=self._model_ctx.knowledge_dirs,
                dwh_dir=self._model_ctx.dwh_dir,
                is_dm=True,
                dm_is_owned=dm_is_owned,
                invitable_agents=self._resolve_invitable_agents_for_prompt(project),
                chat_history=chat_history,
            )
        else:
            # Build setup prompt - agents are referenced via AGENTS.md file.
            # Pass the same allowlist resolver so the FIRST delegation respects
            # any project agent_teams/agent_names filter (the server-side
            # chatroom-create gate enforces it either way; passing it here
            # avoids the coordinator picking unreachable invitees on turn 1).
            prompt = coordinator_builder.build_setup_prompt(
                name=self.name,
                description=self.description,
                project_id=project_id,
                chatroom_name=chatroom_name,
                message_content=message.content,
                data_dir=data_dir,
                context_files=context_files,
                project_name=project.name,
                agent_dir=self._model_ctx.base_dir,
                knowledge_dirs=self._model_ctx.knowledge_dirs,
                dwh_dir=self._model_ctx.dwh_dir,
                invitable_agents=self._resolve_invitable_agents_for_prompt(project),
                # §7.3's one branch: `spec-ing` gets the spec contract, anything
                # else (a pre-feature project, an already-accepted plan) keeps
                # today's setup contract.
                plan_phase=project.phase,
            )

        action_block, usage, validation_notes = await self._invoke_validated(
            project_id=project_id,
            chatroom_name=chatroom_name,
            prompt=prompt,
            sandbox_dir=sandbox_dir,
            log_dir=log_dir,
            additional_dirs=additional_dirs,
            action_schema=action_schema,
            trigger_version=trigger_version,
            role=derive_role(self.name, is_coordinator=True),
            model_config_name=message.model_config_name,
        )

        logger.info(
            f"Agent {self.name} (coordinator): Setup invocation complete "
            f"(cost=${usage.cost_usd:.4f})"
        )

        replied_chatrooms = await action_executor.process(
            action_block=action_block,
            project_id=project_id,
            sandbox_dir=sandbox_dir,
        )
        await self._post_validation_notes(project_id, validation_notes, trigger_version)

        # Close out the triggering room if the LLM didn't reply there —
        # keeps the self-batch pending work from getting stuck and the
        # typing indicator from pinning on the coordinator until timeout.
        if chatroom_name not in replied_chatrooms:
            await self._emit_no_action_message(project_id, chatroom_name, trigger_version)

    # ─────────────────────────────────────────────────────────────────────────
    # Coordinator Live-Message Methods (when Agent is project coordinator)
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_user_request(
        self,
        project_id: str,
        chatroom_name: str,
        message: "ChatMessage",
        trigger_version: int,
    ) -> None:
        """Handle a live message in user-communication (non-first-user-request)."""
        await self._invoke_coordinator_response(project_id, chatroom_name, message, trigger_version)

    async def _coordinate(
        self,
        project_id: str,
        chatroom_name: str,
        message: "ChatMessage",
        trigger_version: int,
    ) -> None:
        """Handle coordination messages in work chatrooms (addressed to coordinator)."""
        await self._invoke_coordinator_response(project_id, chatroom_name, message, trigger_version)

    async def _invoke_coordinator_response(
        self,
        project_id: str,
        chatroom_name: str,
        message: "ChatMessage",
        trigger_version: int,
    ) -> None:
        """Invoke Claude with the coordinator prompt for a live message.

        Used for both user-communication responses and in-room coordination
        after first-user-request / batch-complete paths have been handled
        elsewhere.
        """
        name = self.name

        if not self._model_ctx.cli:
            raise RuntimeError(f"Agent {name} (coordinator): CLI not configured")

        action_executor = self._model_ctx.action_executor
        if not action_executor:
            raise RuntimeError(f"Agent {name} (coordinator): Action executor not configured")

        await self._emit_acknowledgment(project_id, chatroom_name, trigger_version)

        from .project import Project
        project = Project.get(project_id, self._model_ctx)

        data_dir = self._model_ctx.project_dir(project_id, project.name)
        sandbox_dir = self._model_ctx.sandbox_dir(project_id, project.name)
        log_dir = self._model_ctx.llm_log_dir(project_id, project.name)

        additional_dirs = self._extra_dirs(data_dir, sandbox_dir)

        prompt_builder = create_prompt_builder(
            OperationalMode.COORDINATOR,
            git_url=self._model_ctx.git_url,
        )
        assert isinstance(prompt_builder, CoordinatorPromptBuilder)

        from .chatroom import Chatroom
        chatroom = Chatroom.get(project_id, chatroom_name, self._model_ctx)
        chat_history = (
            chatroom.recent_history_for_prompt(exclude_message_id=message.id)
            if chatroom is not None
            else None
        )

        action_schema, dm_is_owned = self._coordinator_dm_action_schema(project)
        prompt = prompt_builder.build_prompt(
            name=self.name,
            description=self.description,
            project_id=project_id,
            chatroom_name=chatroom_name,
            from_participant_name=message.from_participant_name or message.from_participant_id,
            message_content=message.content,
            data_dir=data_dir,
            project_name=project.name,
            agent_dir=self._model_ctx.base_dir,
            knowledge_dirs=self._model_ctx.knowledge_dirs,
            dwh_dir=self._model_ctx.dwh_dir,
            is_dm=project.is_dm_shaped,
            dm_is_owned=dm_is_owned,
            invitable_agents=self._resolve_invitable_agents_for_prompt(project),
            chat_history=chat_history,
            plan=plan_prompt_state(project, self._model_ctx),
        )

        action_block, usage, validation_notes = await self._invoke_validated(
            project_id=project_id,
            chatroom_name=chatroom_name,
            prompt=prompt,
            sandbox_dir=sandbox_dir,
            log_dir=log_dir,
            additional_dirs=additional_dirs,
            action_schema=action_schema,
            trigger_version=trigger_version,
            role=derive_role(self.name, is_coordinator=True),
            model_config_name=message.model_config_name,
        )

        logger.info(
            f"Agent {name} (coordinator): Claude invocation complete "
            f"(cost=${usage.cost_usd:.4f}, tokens in={usage.input_tokens} out={usage.output_tokens})"
        )

        replied_chatrooms = await action_executor.process(
            action_block=action_block,
            project_id=project_id,
            sandbox_dir=sandbox_dir,
        )
        await self._post_validation_notes(project_id, validation_notes, trigger_version)

        if chatroom_name not in replied_chatrooms:
            await self._emit_no_action_message(
                project_id, chatroom_name, trigger_version, replied_chatrooms
            )

    async def _emit_no_action_message(
        self,
        project_id: str,
        chatroom_name: str,
        source_version: int,
        replied_chatrooms: set[str] | None = None,
    ) -> None:
        """Post a closure reply when the LLM produced no reply action for this room.

        Posted non-ack so the server's record_response marks this participant
        as responded, clears PendingWork, and fires BATCH_COMPLETE. When the
        LLM did reply but misrouted to a different chatroom, surface that
        instead of the upbeat zero-action filler so the user has a pointer
        to the stranded reply.
        """
        if not self._model_ctx.client:
            raise RuntimeError(f"Agent {self.name}: Client not configured, cannot emit no-action message")

        if replied_chatrooms:
            rooms = ", ".join(sorted(replied_chatrooms))
            content = f"(My reply landed in: {rooms}. See there for the actual answer.)"
        else:
            content = "Message processed, no further action needed at the moment!"
        await self._model_ctx.client.post_message(
            project_id=project_id,
            chatroom_name=chatroom_name,
            content=content,
            source_version=source_version,
        )
