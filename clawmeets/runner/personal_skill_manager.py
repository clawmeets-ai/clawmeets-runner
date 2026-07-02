# SPDX-License-Identifier: MIT
"""
clawmeets/runner/personal_skill_manager.py

Manages the personal-skill-hub directory for an agent.

Personal skills are agent-private SKILL.md files authored during scheduled
reflection (Promote / Correct modes in /clawmeets:reflect). They are never
synced to the server — same privacy boundary as ``learnings/``.

Each LLM provider's ``_prepare_invocation`` flattens this hub together
with ``skill-hub/skills/`` into the CLI's native cwd skill-discovery path
(via ``materialize_skill_tree``), so a personal skill authored mid-session
is visible on the next turn.
"""
from __future__ import annotations

from pathlib import Path


class PersonalSkillManager:
    """Owns ``{agent_dir}/personal-skill-hub/`` on the runner.

    Layout::

        {agent_dir}/personal-skill-hub/
        └── skills/
            └── <skill_name>/
                └── SKILL.md       (agent-authored during reflection)

    The runner does not write SKILL.md files here — that is the agent's
    job during reflection. It only guarantees the directory exists so
    ``materialize_skill_tree`` finds a (possibly empty) source dir.
    """

    def __init__(self, agent_dir: Path) -> None:
        self.hub_dir = agent_dir / "personal-skill-hub"
        (self.hub_dir / "skills").mkdir(parents=True, exist_ok=True)
