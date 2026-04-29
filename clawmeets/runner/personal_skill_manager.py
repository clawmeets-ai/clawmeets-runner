# SPDX-License-Identifier: MIT
"""
clawmeets/runner/personal_skill_manager.py

Manages the personal-skill-hub plugin directory for an agent.

Personal skills are agent-private SKILL.md files authored during scheduled
reflection (Promote / Correct modes in /clawmeets:reflect). They are never
synced to the server — same privacy boundary as `learnings/`. Claude Code
discovers them via the `--plugin-dir` mechanism and registers each as a
`/personal:<name>` slash-command, with the body loaded just-in-time on
invocation.
"""
from __future__ import annotations

import json
from pathlib import Path

PLUGIN_JSON = {
    "name": "personal",
    "version": "1.0.0",
    "description": "Personal skills authored by the agent during reflection",
}


class PersonalSkillManager:
    """
    Manages a local personal-skill-hub plugin directory for an agent.

    Directory layout:
        {agent_dir}/personal-skill-hub/
        ├── .claude-plugin/
        │   └── plugin.json
        └── skills/
            └── <skill_name>/
                └── SKILL.md
    """

    def __init__(self, agent_dir: Path) -> None:
        self.hub_dir = agent_dir / "personal-skill-hub"
        self._ensure_plugin_structure()

    def _ensure_plugin_structure(self) -> None:
        """Create the plugin directory structure if it doesn't exist (idempotent)."""
        plugin_dir = self.hub_dir / ".claude-plugin"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        skills_dir = self.hub_dir / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        plugin_json = plugin_dir / "plugin.json"
        if not plugin_json.exists():
            plugin_json.write_text(json.dumps(PLUGIN_JSON, indent=2))

    @property
    def plugin_dir(self) -> Path:
        """Return the personal-skill-hub plugin directory path for --plugin-dir."""
        return self.hub_dir
