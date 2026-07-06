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
  ``init_message``, or ``file_path`` bytes — so valid-but-non-deterministic
  output (differently-worded prose) can never trip a retry.
- AC-INV-3: the gate performs zero model invocations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    sets the model is asked to choose from form a closed, fixed target — the
    corrective feedback cannot oscillate, which is what makes the loop provably
    terminating (plan §4).
    """
    existing_rooms: frozenset[str]      # Project.chatrooms
    invitable_agents: frozenset[str]    # Agent.invitable_short_names_for_project (the valid invite set)
    resolvable_agents: frozenset[str]   # Agent.list_all short-names (global registry). FEEDBACK-BRANCH
                                        # ONLY — never affects PASS/REJECT, only picks the
                                        # ghost vs. known-but-not-invitable message. Empty on
                                        # non-create_room turns (workers/owned DMs emit no create_room).
    project_active: bool                # Project.status == ACTIVE


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
    never free-form ``content``/``init_message``/``file_path`` — so it can
    never trip on valid-but-non-deterministic output. Classifies every action
    per the plan §1 full-sweep table across all four action types.
    """

    def validate(self, action_block: "ActionBlock", snap: StateSnapshot) -> ValidationResult:
        """Classify every action in ``action_block`` against ``snap``.

        Tracks an effective room set = ``snap.existing_rooms`` plus rooms created
        by earlier PASSing ``create_room`` actions in THIS block, so an in-block
        ``create_room -> reply`` pair validates correctly. Unknown action types
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
        """create_room precedence (plan §1): invitee check FIRST, then existing-name.

        An invalid invitee is the canonical hard-invalid case (REJECT_RETRY). If
        every invitee is valid but the room name already exists, the create is
        idempotently satisfied (NO_OP). Otherwise PASS.
        """
        invite = action.get("invite") or []
        bad = [name for name in invite if name not in snap.invitable_agents]
        if bad:
            return self._bad_invitees(index, action.get("name"), bad, snap)

        name = action.get("name")
        if name in effective_rooms:
            return self._room_exists(index, name)

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
        self, index: int, room_name: Any, bad: list[str], snap: StateSnapshot
    ) -> Violation:
        """create_room with non-invitable invitees -> REJECT_RETRY.

        Splits the feedback text (plan §1a-B) — a name absent from the global
        registry is an unknown/ghost (server 404 class); a real agent outside
        this project's allowlist is 'known but not invitable' (server 403 class)
        and MUST NOT be described as a ghost, so the model does not waste a retry
        chasing a spelling fix for a correctly-spelled name. Both remain
        REJECT_RETRY, so §4 termination is untouched.
        """
        ghosts = [n for n in bad if n not in snap.resolvable_agents]
        known = [n for n in bad if n in snap.resolvable_agents]

        lines = ["create_room rejected:"]
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
        lines.append(
            f"  Invitable agents here are: {_fmt_set(snap.invitable_agents)}. "
            f"Re-emit create_room with an invitable invitee (exact spelling), or "
            f"drop the invite."
        )
        feedback = "\n".join(lines)
        note = (
            f"Note: create_room `{room_name}` dropped — invalid invitee(s) "
            f"{_fmt_names(bad)} (unresolved after {_RETRY_WORD} retries)."
        )
        return Violation(index, "create_room", ViolationKind.REJECT_RETRY, feedback, note)

    def _room_exists(self, index: int, name: Any) -> Violation:
        """create_room whose name already exists -> NO_OP (idempotent create)."""
        note = (
            f"Note: create_room `{name}` skipped — a chatroom with that name "
            f"already exists."
        )
        return Violation(index, "create_room", ViolationKind.NO_OP, "", note)

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
