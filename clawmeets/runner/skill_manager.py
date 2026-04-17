# SPDX-License-Identifier: MIT
"""
clawmeets/runner/skill_manager.py

Manages the skill-hub plugin directory for an agent.
Downloads and caches SKILL.md files installed via the ClawMeets Skill Hub.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clawmeets.api.client import ClawMeetsClient

logger = logging.getLogger("clawmeets.runner")

PLUGIN_JSON = {
    "name": "skill-hub",
    "version": "1.0.0",
    "description": "Skills installed via ClawMeets Skill Hub",
}


class SkillManager:
    """
    Manages a local skill-hub plugin directory for an agent.

    Directory layout:
        {agent_dir}/skill-hub/
        ├── .claude-plugin/
        │   └── plugin.json
        └── skills/
            ├── pdf/
            │   └── SKILL.md
            └── web-artifacts/
                └── SKILL.md
    """

    def __init__(self, agent_dir: Path) -> None:
        self.skill_hub_dir = agent_dir / "skill-hub"
        self._ensure_plugin_structure()

    def _ensure_plugin_structure(self) -> None:
        """Create the skill-hub plugin directory structure if it doesn't exist."""
        plugin_dir = self.skill_hub_dir / ".claude-plugin"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        skills_dir = self.skill_hub_dir / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        plugin_json = plugin_dir / "plugin.json"
        if not plugin_json.exists():
            plugin_json.write_text(json.dumps(PLUGIN_JSON, indent=2))

    async def sync_from_server(self, client: "ClawMeetsClient", agent_id: str) -> None:
        """Catch-up: fetch installed skills from server, download missing ones, remove extras."""
        try:
            resp = await client._http.get(f"/agents/{agent_id}/skills")
            resp.raise_for_status()
            data = resp.json()
            server_skills = set(data.get("installed_skills", []))
        except Exception as e:
            logger.warning(f"Failed to fetch installed skills from server: {e}")
            return

        local_skills = set(self.installed_skills())

        # Download missing skills
        for skill_name in server_skills - local_skills:
            try:
                resp = await client._http.get(f"/skills/{skill_name}")
                resp.raise_for_status()
                skill_data = resp.json()
                content = skill_data.get("content")
                if content:
                    self.install_skill(skill_name, content)
                    logger.info(f"Synced skill: {skill_name}")
            except Exception as e:
                logger.warning(f"Failed to sync skill {skill_name}: {e}")

        # Remove uninstalled skills
        for skill_name in local_skills - server_skills:
            self.uninstall_skill(skill_name)
            logger.info(f"Removed uninstalled skill: {skill_name}")

    def install_skill(self, skill_name: str, skill_md: str) -> None:
        """Write a SKILL.md file to the skill-hub plugin directory."""
        skill_dir = self.skill_hub_dir / "skills" / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(skill_md)
        logger.info(f"Installed skill: {skill_name}")

    def uninstall_skill(self, skill_name: str) -> None:
        """Remove a skill directory from the skill-hub plugin."""
        skill_dir = self.skill_hub_dir / "skills" / skill_name
        if skill_dir.exists():
            shutil.rmtree(skill_dir)
            logger.info(f"Uninstalled skill: {skill_name}")

    def installed_skills(self) -> list[str]:
        """List installed skill names."""
        skills_dir = self.skill_hub_dir / "skills"
        if not skills_dir.exists():
            return []
        return sorted(
            d.name for d in skills_dir.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()
        )

    @property
    def plugin_dir(self) -> Path:
        """Return the skill-hub plugin directory path for --plugin-dir."""
        return self.skill_hub_dir
