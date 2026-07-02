# SPDX-License-Identifier: MIT
"""
clawmeets/llm/triggers.py
Memory-loop trigger registry — single source of truth for the
``<!-- clawmeets:*-trigger -->`` markers the prompt builder surfaces and the
runtime regex-matches.

Layer 0 (no domain model deps). Consumed by:
- ``prompt_builder._build_trigger_section`` — renders the visible list per role
- ``agent._TRIGGER_MARKER_RE`` — regex for resume-marker detection in DMs
"""
from __future__ import annotations

from dataclasses import dataclass

# Role enum shared with SystemSkillManager and Agent.invoke audience pick.
ROLE_WORKER = "worker"
ROLE_COORDINATOR = "coordinator"
ROLE_ASSISTANT = "assistant"
ROLES: tuple[str, ...] = (ROLE_WORKER, ROLE_COORDINATOR, ROLE_ASSISTANT)


@dataclass(frozen=True)
class TriggerSpec:
    marker: str                       # HTML-comment marker as it appears in chat messages
    skill: str                        # Slash-command form the agent should follow
    purpose: str                      # One-line description shown to the LLM
    audiences: tuple[str, ...]        # Subset of ROLES — which roles see this trigger


# Order matters: rendered in this order in the prompt.
MEMORY_LOOP_TRIGGERS: list[TriggerSpec] = [
    TriggerSpec(
        marker="<!-- clawmeets:reflect-trigger -->",
        skill="/clawmeets:reflect",
        purpose="distill recent activity into learnings/ and audit the wiki",
        audiences=ROLES,
    ),
    TriggerSpec(
        marker="<!-- clawmeets:personalize-trigger -->",
        skill="/clawmeets:personalize",
        purpose="personalize yourself on first run — assistants interview the user for USER.md; workers/coordinators study recent activity and deliver a tailored field-knowledge dump",
        audiences=ROLES,
    ),
    TriggerSpec(
        marker="<!-- clawmeets:rerun-{slug} -->",
        skill="/clawmeets:rerun-project",
        purpose="re-run a previously saved project skill against new inputs",
        audiences=(ROLE_ASSISTANT,),
    ),
]


def triggers_for(*, role: str) -> list[TriggerSpec]:
    """Return the subset of triggers visible to a participant role."""
    return [t for t in MEMORY_LOOP_TRIGGERS if role in t.audiences]


def derive_role(name: str, *, is_coordinator: bool) -> str:
    """The agent's role for an invocation: assistant > coordinator > worker.

    Assistant predicate is the ``-assistant`` name suffix (see
    ``project_user_assistant_naming_convention`` memory). Coordinator vs.
    worker is per-project.
    """
    if name.endswith("-assistant"):
        return ROLE_ASSISTANT
    return ROLE_COORDINATOR if is_coordinator else ROLE_WORKER
