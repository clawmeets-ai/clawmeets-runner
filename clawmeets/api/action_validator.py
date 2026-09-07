# SPDX-License-Identifier: MIT
"""
clawmeets/api/action_validator.py
Deterministic, semantic pre-execution gate for action blocks.

This module is part of Layer 0 (pure - no domain model dependencies) and is
deliberately I/O-free: it imports only the standard library and the Layer-0
action types. It holds NO model/LLM client and performs NO network or
filesystem access — the whole classifier is a pure function of

    classify(structured_referent, snapshot) -> {PASS, NO_OP, REJECT_RETRY}

so the same (actions, snapshot) always yields a bit-identical verdict. The
only model call in the validation *layer* is the corrective retry, which lives
one level up in ``models/agent.py`` (``_invoke_validated``), never here.

Load-bearing invariant (see plan §0 — AC-INV-1/2/3):
- AC-INV-1: this module imports no model/LLM/provider/network module.
- AC-INV-2: ``ActionValidator.validate`` reads ONLY structured referents
  (``room``/``name``/``invite`` and project status) — never ``content``,
  ``init_message``, ``message``, or ``file_path`` bytes — so valid-but-non-deterministic
  output (differently-worded prose) can never trip a retry.
- AC-INV-3: the gate performs zero model invocations.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .actions import ActionBlock


class ViolationKind(str, Enum):
    """How a failed precondition is resolved."""
    REJECT_RETRY = "reject_retry"   # hard-invalid referent -> feed back & retry
    NO_OP = "no_op"                 # idempotent already-satisfied -> skip + note


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable, once-per-turn view of local server state used for validation.

    Built by ``build_state_snapshot`` in ``models/agent.py`` (the only
    fs-touching piece); the validator itself does no I/O. Because the snapshot
    is taken ONCE before the retry loop and reused for every retry, the valid
    sets the model is asked to choose from form a closed target — the
    corrective feedback cannot oscillate, which is what makes the loop provably
    terminating (plan §4).

    **Closed and MONOTONE, not motionless.** Exactly one field —
    :attr:`invitable_agents` — may move between retries, only through
    :meth:`with_invitable`, and only ever by union, so it can only GROW. Every
    other field is frozen for the life of the turn. Monotonicity is what keeps
    the feedback non-oscillating: a name the validator has already accepted can
    never become invalid again, so two successive corrections can never
    contradict each other. Termination itself was never carried by the target
    holding still — it rests on the hard integer budget
    (``_MAX_VALIDATION_RETRIES``, ``models/agent.py``).

    Why that one field has to move: the coordinator's own remedy for a
    non-invitable invitee — ``clawmeets project allowlist`` — takes effect
    *during* the turn that discovers the problem. With the invitee set frozen
    across the whole turn the remedy was unobservable, so a coordinator could
    widen the roster and still have its ``create_room`` dropped by a set
    captured before the widening, having spent the whole budget hunting for a
    spelling error that was never there.
    """
    existing_rooms: frozenset[str]      # Project.chatrooms
    invitable_agents: frozenset[str]    # Agent.invitable_short_names_for_project (the valid invite set)
    resolvable_agents: frozenset[str]   # Agent.list_all short-names (global registry). FEEDBACK-BRANCH
                                        # ONLY — never affects PASS/REJECT, only picks the
                                        # ghost vs. known-but-not-invitable message. Empty on
                                        # non-create_room turns (workers/owned DMs emit no create_room).
    project_active: bool                # Project.status == ACTIVE
    # §7.4's gate — THE gate, singular — as ONE field: the count of plan notes
    # addressed to the user and unresolved on a regular project, and 0 otherwise.
    # Zero IS "not blocked" and is the whole predicate — the classifier must
    # never re-derive `surface`, which is exactly what a second field would
    # invite.
    #
    # There WAS a second field, `awaiting_plan_approval`, for a pre-approval
    # gate that fired for the whole of `spec-ing`. It is gone with the Approve
    # button: the block a project starts life under is now an ordinary open plan
    # note — the go-note, seeded by `POST /projects` — so this count already
    # carries it, from the coordinator's very first turn.
    #
    # A count and not a bool because the note this drives has to say <N>, and a
    # bool cannot. Sourced ONCE per turn by ``build_state_snapshot``
    # (``models/agent.py``) and NEVER read live. ``invitable_agents``' monotone
    # refresh does NOT generalize to here, and the difference is the whole
    # reason it is safe there: a widened roster only ever ADDS valid names,
    # whereas this count moves in BOTH directions as notes are filed and
    # resolved. A live read here would let the verdict flip back and forth on
    # wall-clock state — precisely the oscillation the closed target forbids.
    blocked_notes: int = 0

    def with_invitable(self, more: frozenset[str]) -> "StateSnapshot":
        """Return this snapshot with ``more`` unioned onto ``invitable_agents``.

        The ONLY sanctioned mutation of a snapshot, and a union rather than a
        replacement so the set can only grow (see the class docstring: closed
        and monotone, not motionless). A shrink would be unsound — the model may
        already have been told a name is valid, and taking that back mid-turn is
        exactly the oscillation the fixed target existed to prevent.

        Returns ``self`` unchanged when ``more`` adds nothing, so the caller can
        use identity (``new is not old``) as a cheap "did the roster widen?"
        test rather than re-comparing sets.
        """
        if more <= self.invitable_agents:
            return self
        return replace(self, invitable_agents=self.invitable_agents | more)


@dataclass(frozen=True)
class Violation:
    """One failed precondition, with everything needed to feed back or note."""
    action_index: int
    action_type: str
    kind: ViolationKind
    feedback_message: str   # actionable correction string (REJECT_RETRY only)
    note_message: str       # user-communication note (NO_OP skip, or dropped-after-retries)


@dataclass
class ValidationResult:
    """Outcome of validating one ActionBlock against a snapshot.

    Holds a reference to the original action dicts so ``surviving_actions`` can
    return the filtered list without the caller re-threading it.
    """
    actions: list[dict[str, Any]]
    violations: list[Violation] = field(default_factory=list)

    @property
    def retryable(self) -> bool:
        """True iff any violation is REJECT_RETRY (drives the retry decision)."""
        return any(v.kind is ViolationKind.REJECT_RETRY for v in self.violations)

    def feedback(self) -> str:
        """Concatenated correction block for the next invocation (retryables only).

        Deterministically derived from the snapshot: names each bad referent,
        the full finite valid set, and the concrete remediation.
        """
        parts = [
            v.feedback_message
            for v in self.violations
            if v.kind is ViolationKind.REJECT_RETRY
        ]
        header = (
            "CORRECTION: your previous action block contained actions that were "
            "rejected before execution because they reference server state that "
            "does not exist. Fix ONLY the issues below and re-emit the complete "
            "action block (keep the valid actions):\n\n"
        )
        return header + "\n\n".join(parts)

    def notes(self) -> list[str]:
        """No-op skip notes to post to user-communication (idempotent actions)."""
        return [
            v.note_message
            for v in self.violations
            if v.kind is ViolationKind.NO_OP
        ]

    def dropped_notes(self) -> list[str]:
        """Notes for REJECT_RETRY actions dropped at budget exhaustion (terminal fallback)."""
        return [
            v.note_message
            for v in self.violations
            if v.kind is ViolationKind.REJECT_RETRY
        ]

    def surviving_actions(self, drop_retryable: bool) -> list[dict[str, Any]]:
        """Actions to actually execute.

        Always drops NO_OP actions (idempotent — already satisfied); also drops
        REJECT_RETRY actions when ``drop_retryable`` (the terminal fallback,
        after the retry budget is exhausted). PASS actions are always kept, so a
        turn's valid work survives even when one action is unfixable.
        """
        drop = {
            v.action_index
            for v in self.violations
            if v.kind is ViolationKind.NO_OP
            or (drop_retryable and v.kind is ViolationKind.REJECT_RETRY)
        }
        return [a for i, a in enumerate(self.actions) if i not in drop]


class ActionValidator:
    """Pure, deterministic semantic gate.

    Inspects ONLY structured referents (room/agent names, project status) —
    never free-form ``content``/``init_message``/``message``/``file_path`` — so
    it can never trip on valid-but-non-deterministic output. Classifies every
    action per the plan §1 full-sweep table across all five action types.
    """

    def validate(self, action_block: "ActionBlock", snap: StateSnapshot) -> ValidationResult:
        """Classify every action in ``action_block`` against ``snap``.

        Tracks an effective room set = ``snap.existing_rooms`` plus rooms created
        by earlier PASSing ``create_room`` actions in THIS block, so an in-block
        ``create_room -> reply`` (or ``create_room -> invite_agent``) pair
        validates correctly. Unknown action types
        are ignored (``ActionBlock.typed_actions`` already drops them). No I/O.
        """
        result = ValidationResult(actions=action_block.actions)
        effective_rooms: set[str] = set(snap.existing_rooms)

        for index, action in enumerate(action_block.actions):
            atype = action.get("type")

            if atype in ("reply", "update_file"):
                room = action.get("room")
                if room not in effective_rooms:
                    result.violations.append(
                        self._unknown_room(index, atype, room, snap)
                    )

            elif atype == "create_room":
                violation = self._check_create_room(index, action, snap, effective_rooms)
                if violation is not None:
                    result.violations.append(violation)
                else:
                    # Passed: the new room is visible to later in-block references.
                    name = action.get("name")
                    if name:
                        effective_rooms.add(name)

            elif atype == "invite_agent":
                violation = self._check_invite_agent(index, action, snap, effective_rooms)
                if violation is not None:
                    result.violations.append(violation)

            elif atype == "project_completed":
                if not snap.project_active:
                    result.violations.append(self._already_complete(index))

            # Any other (unknown/future) type: silently pass — not our concern.

        return result

    # -- per-action classifiers -------------------------------------------------

    def _check_create_room(
        self,
        index: int,
        action: dict[str, Any],
        snap: StateSnapshot,
        effective_rooms: set[str],
    ) -> Violation | None:
        """create_room precedence: invitees, then existing-name, then the plan gate.

        An invalid invitee is the canonical hard-invalid case (REJECT_RETRY). If
        every invitee is valid but the room name already exists, the create is
        idempotently satisfied (NO_OP). Then §7.4's gate, and PASS if it is
        clear.

        **The order is load-bearing and the gate is last on purpose.** The
        existing-name check MUST stay ahead of it: if the room already exists,
        nothing new opens, so there is nothing to prevent. Placing the gate third
        is what makes *"the gate stops a NEW room opening"* exactly true rather
        than approximately.

        **One rule, ONE trigger point.** There used to be two in this slot — a
        pre-approval gate and an execution gate, tested in order because their
        notes said different things. They are one gate now: a project awaiting
        approval is a project with the go-note open, which is a project with a
        plan note addressed to the user. One ready-made snapshot field, nothing
        re-derived.

        **``create_room`` is the ONLY action this gate refuses.** ``invite_agent``
        used to be refused alongside it and no longer is — see
        :meth:`_check_invite_agent` for the whole of that decision.
        """
        invite = action.get("invite") or []
        bad = [name for name in invite if name not in snap.invitable_agents]
        if bad:
            return self._bad_invitees(index, action.get("name"), bad, snap)

        name = action.get("name")
        if name in effective_rooms:
            return self._room_exists(index, name)

        if snap.blocked_notes:
            return self._plan_blocked(index, name, snap.blocked_notes)

        return None

    def _check_invite_agent(
        self,
        index: int,
        action: dict[str, Any],
        snap: StateSnapshot,
        effective_rooms: set[str],
    ) -> Violation | None:
        """invite_agent precedence: room, then invitees, then the plan gate.

        Mirrors :meth:`_check_create_room` with the one structural difference
        the two actions actually have: ``create_room`` names a room that must
        NOT exist yet, ``invite_agent`` names one that must. So the room check
        leads here (a room the coordinator can't see is a hard-invalid referent,
        exactly like ``reply``'s) and the invitee check follows, reusing the same
        ghost-vs-known feedback split.

        **There is deliberately no "already a member" NO_OP.** Membership is
        idempotent server-side and the *message* is the point of the action — a
        NO_OP would silently swallow a coordinator re-engaging a member that
        went quiet, which is a supported use of this action rather than a
        mistake.

        **THE PLAN GATE DOES NOT APPLY HERE, and that is a decision the user
        made rather than an omission.** ``invite_agent`` used to carry both plan
        refusals in a third slot, mirroring ``create_room``'s. It does not any
        more: a user whose plan is waiting on them very often wants another
        agent's opinion *in order to* decide, and a gate that stops them pulling
        one in blocks the thing that would release it.

        The consequence, stated once and plainly because it is real: the gate
        stops a new **room**, not new **people**. ``shared-context`` always
        exists and its roster can be filled while the gate is shut, so a blocked
        coordinator can seat specialists there and hold a real working
        conversation in it. That is the point of the room — it is the
        consultation channel, and consultation is exactly what a blocked project
        should be doing — but it also means the gate is a stop on *dispatch*,
        not a freeze on the project.

        Reads only structured referents (``room`` / ``invite``), never the
        ``message`` bytes — AC-INV-2 holds.
        """
        room = action.get("room")
        if room not in effective_rooms:
            return self._unknown_room(index, "invite_agent", room, snap)

        invite = action.get("invite") or []
        bad = [name for name in invite if name not in snap.invitable_agents]
        if bad:
            return self._bad_invitees(index, room, bad, snap, action_type="invite_agent")

        return None

    def _unknown_room(
        self, index: int, action_type: str, room: Any, snap: StateSnapshot
    ) -> Violation:
        """reply/update_file to a room absent from the effective set -> REJECT_RETRY."""
        feedback = (
            f"{action_type} rejected: chatroom `{room}` does not exist in this "
            f"project. Existing chatrooms are: {_fmt_set(snap.existing_rooms)}. "
            f"Re-emit targeting an existing room, or create it first with a "
            f"create_room action."
        )
        note = (
            f"Note: {action_type} to `{room}` dropped — no such chatroom "
            f"(unresolved after {_RETRY_WORD} retries)."
        )
        return Violation(index, action_type, ViolationKind.REJECT_RETRY, feedback, note)

    def _bad_invitees(
        self,
        index: int,
        room_name: Any,
        bad: list[str],
        snap: StateSnapshot,
        action_type: str = "create_room",
    ) -> Violation:
        """create_room / invite_agent with non-invitable invitees -> REJECT_RETRY.

        ``action_type`` defaults to ``"create_room"`` so the original call site
        (and the tests that reach these private builders directly) is untouched;
        ``invite_agent`` passes its own, since telling the model to "re-emit
        create_room" when it emitted an invite is a wasted retry.

        Splits the feedback text (plan §1a-B) — a name absent from the global
        registry is an unknown/ghost (server 404 class); a real agent outside
        this project's allowlist is 'known but not invitable' (server 403 class)
        and MUST NOT be described as a ghost, so the model does not waste a retry
        chasing a spelling fix for a correctly-spelled name. Both remain
        REJECT_RETRY, so §4 termination is untouched.

        **The 'known' branch names the remedy, and that is the point of it.**
        Telling a coordinator only that a name is un-invitable and that
        re-spelling will not help leaves it with no move except to drop the
        invite — which is what happened in the field: the coordinator widened
        the roster with ``clawmeets project allowlist``, was rejected again by a
        set captured before the widening, concluded it must have the name wrong,
        and spent its entire budget on spelling variants. The refresh in
        ``_invoke_validated`` makes the widening take effect; this sentence is
        what tells the model the widening is available at all. The guardrail
        travels with it: the allowlist is the OWNER's scope, so the text says to
        widen only on a request or approval the user has actually given.
        """
        ghosts = [n for n in bad if n not in snap.resolvable_agents]
        known = [n for n in bad if n in snap.resolvable_agents]

        lines = [f"{action_type} rejected:"]
        if ghosts:
            lines.append(
                f"  {_fmt_names(ghosts)} is not a registered agent for this project."
                if len(ghosts) == 1
                else f"  {_fmt_names(ghosts)} are not registered agents for this project."
            )
        if known:
            lines.append(
                f"  {_fmt_names(known)} is a real agent but is NOT invitable in this "
                f"project (outside its invitable allowlist); re-spelling will not "
                f"help — do NOT re-emit the same name."
                if len(known) == 1
                else f"  {_fmt_names(known)} are real agents but are NOT invitable in "
                f"this project (outside its invitable allowlist); re-spelling will "
                f"not help — do NOT re-emit the same names."
            )
            it_them = "it" if len(known) == 1 else "them"
            flags = " ".join(f"--agent {n}" for n in known)
            lines.append(
                f"  You can FIX this instead of routing around it. Widening the "
                f"roster is a supported coordinator action: "
                f"`clawmeets project allowlist <project_id> {flags}` — your runner "
                f"token authorizes it, no --token needed. The allowlist is the "
                f"OWNER's guardrail, so only run it when the user has asked for "
                f"{it_them} or approved {it_them} joining; if they have not, say so "
                f"in user-communication and re-emit this {action_type} on a later "
                f"turn. If you DO run it now, the widening is picked up on your "
                f"NEXT attempt in this same turn — re-emit the SAME {action_type} "
                f"unchanged."
            )
        options = f"Re-emit {action_type} with an invitable invitee (exact spelling)"
        if known:
            options += ", widen the allowlist as described above"
        options += ", or drop the invite."
        lines.append(
            f"  Invitable agents here are: {_fmt_set(snap.invitable_agents)}. "
            f"{options}"
        )
        feedback = "\n".join(lines)
        note = (
            f"Note: {action_type} `{room_name}` dropped — invalid invitee(s) "
            f"{_fmt_names(bad)} (unresolved after {_RETRY_WORD} retries)."
        )
        return Violation(index, action_type, ViolationKind.REJECT_RETRY, feedback, note)

    def _room_exists(self, index: int, name: Any) -> Violation:
        """create_room whose name already exists -> NO_OP (idempotent create)."""
        note = (
            f"Note: create_room `{name}` skipped — a chatroom with that name "
            f"already exists."
        )
        return Violation(index, "create_room", ViolationKind.NO_OP, "", note)

    def _plan_blocked(
        self, index: int, name: Any, notes: int, action_type: str = "create_room"
    ) -> Violation:
        """create_room while the plan has open notes -> NO_OP (§7.4).

        Modelled on :meth:`_room_exists`, and the note is **addressed to the
        user** because they are the only party who can clear the condition.

        **It is now the ONLY plan refusal, so it carries what the deleted one
        said.** ``_plan_unapproved`` used to answer the pre-approval case with
        its own sentence, because the open-note count was zero there and *"N
        notes are addressed to you"* would have been false. That is no longer
        possible: a project awaiting approval has the go-note open, so the count
        is at least one and the sentence is true. What the old note carried that
        this one did not — *the act you are being asked for is the one that
        starts the work*, and *nothing else is blocked* — is folded in below,
        because on a brand-new project this text is the only thing the user
        reads about why nothing is happening.

        ``action_type`` still defaults to ``create_room`` and no caller passes
        anything else: ``invite_agent`` is ungated now. The parameter stays
        because the tests reach this builder directly.

        **The kind is NO_OP and not REJECT_RETRY, and that is load-bearing.** The
        ``StateSnapshot`` is taken once before the retry loop and this field
        deliberately cannot change within it (``invitable_agents``' monotone
        refresh is the one carve-out and does not reach here), so a REJECT_RETRY
        would fail identically three times, burn two invocations and drop the
        action anyway. The
        argument is already in :meth:`_already_complete`'s docstring: *"There is
        no referent the model could fix, so a retry could never converge."*

        The residual, stated rather than sold around: under NO_OP the model is
        not told (``feedback_message`` is REJECT_RETRY-only), so the exposure is
        one turn of possible over-claiming. The coordinator's turn-contract
        obligation (§7.3) is what makes the refusal *expected*; this is the
        backstop under it, not a security boundary.
        """
        one = notes == 1
        note = (
            f"Note: {action_type} `{name}` skipped — {notes} plan "
            f"note{'' if one else 's'} {'is' if one else 'are'} addressed to you "
            f"and unresolved. Open the project's Plan tab and decide them — "
            f"Accept, Reject or Dismiss — and work continues. On a project that "
            f"has not started yet, the note waiting for you IS the approval: "
            f"accepting it is what approves the plan. Nothing else is held up "
            f"meanwhile — you can keep talking to the coordinator, and it can "
            f"still bring other agents in."
        )
        return Violation(index, action_type, ViolationKind.NO_OP, "", note)

    def _already_complete(self, index: int) -> Violation:
        """project_completed on a non-ACTIVE project -> NO_OP (idempotent complete).

        There is no referent the model could fix, so a retry could never
        converge; classify NO_OP and never re-invoke (plan §1, §4).
        """
        note = (
            "Note: project_completed skipped — project is already marked "
            "COMPLETED (or FAILED)."
        )
        return Violation(index, "project_completed", ViolationKind.NO_OP, "", note)


# The retry budget, spelled for the note strings. Kept as a word so the note
# text does not silently drift if the numeric cap (models/agent.py
# _MAX_VALIDATION_RETRIES) is tuned; the number itself is not load-bearing here.
_RETRY_WORD = "the allotted"


def _fmt_set(items: frozenset[str]) -> str:
    """Render a valid set as a stable, sorted, bracketed list for feedback."""
    return "[" + ", ".join(sorted(items)) + "]"


def _fmt_names(names: list[str]) -> str:
    """Render referent names as backticked, comma-separated tokens."""
    return ", ".join(f"`{n}`" for n in names)
