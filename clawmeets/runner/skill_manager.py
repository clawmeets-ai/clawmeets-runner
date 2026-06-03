# SPDX-License-Identifier: MIT
"""
clawmeets/runner/skill_manager.py

Manages the skill-hub plugin directory for an agent.
Downloads and caches SKILL.md files installed via the ClawMeets Skill Hub.
"""
from __future__ import annotations

import base64
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from clawmeets.models.knowledge_pack import validate_filepath

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
                files = _decode_skill_files(skill_data.get("files") or {})
                if content:
                    self.install_skill(skill_name, content, files=files)
                    logger.info(f"Synced skill: {skill_name}")
            except Exception as e:
                logger.warning(f"Failed to sync skill {skill_name}: {e}")

        # Remove uninstalled skills
        for skill_name in local_skills - server_skills:
            self.uninstall_skill(skill_name)
            logger.info(f"Removed uninstalled skill: {skill_name}")

    def install_skill(
        self,
        skill_name: str,
        skill_md: str,
        files: dict[str, bytes] | None = None,
    ) -> None:
        """Write SKILL.md plus optional sibling files (template.html, render.py, …)
        to ``{agent_dir}/skill-hub/skills/{skill_name}/`` so SKILL.md procedures
        can reference siblings via ``$CLAWMEETS_AGENT_DIR``.

        Wipes the skill directory before rewriting so a re-install (e.g. after a
        skill author renames a sibling) does not leave stale files behind.
        """
        skill_dir = self.skill_hub_dir / "skills" / skill_name
        if skill_dir.exists():
            shutil.rmtree(skill_dir)
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(skill_md)
        for relpath, body in (files or {}).items():
            normalized = validate_filepath(relpath)
            dest = skill_dir / normalized
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
        logger.info(
            f"Installed skill: {skill_name} ({len(files or {})} sibling files)"
        )

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


def _decode_skill_files(files: dict[str, dict]) -> dict[str, bytes]:
    """Decode the ``{relpath: {"content_b64": <ascii>}}`` wire shape used by
    ``SkillSyncPayload.skill_files`` and ``GET /skills/{name}.files``."""
    decoded: dict[str, bytes] = {}
    for relpath, entry in (files or {}).items():
        b64 = (entry or {}).get("content_b64") or ""
        if b64:
            decoded[relpath] = base64.b64decode(b64, validate=True)
    return decoded
