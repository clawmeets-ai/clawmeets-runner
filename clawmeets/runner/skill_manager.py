# SPDX-License-Identifier: MIT
"""
clawmeets/runner/skill_manager.py

Manages the skill-hub directory for an agent.

Downloads and caches SKILL.md files installed via the ClawMeets Skill Hub.
Each LLM provider's ``_prepare_invocation`` runs ``materialize_skill_tree``
to flatten ``skill-hub/skills/`` + ``personal-skill-hub/skills/`` into the
CLI's native cwd skill-discovery path (Claude: ``.claude/skills/``;
Codex + Gemini: ``.agents/skills/``), so each CLI's own auto-loader
surfaces the skills — no prompt-side INDEX needed.
"""
from __future__ import annotations

import base64
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from clawmeets.models.knowledge_pack import validate_filepath

if TYPE_CHECKING:
    from clawmeets.api.client import ClawMeetsClient

logger = logging.getLogger("clawmeets.runner")


class SkillManager:
    """
    Manages a local skill-hub directory for an agent.

    Directory layout::

        {agent_dir}/skill-hub/
        └── skills/
            ├── pdf/
            │   └── SKILL.md
            └── web-artifacts/
                └── SKILL.md
    """

    def __init__(self, agent_dir: Path) -> None:
        self.skill_hub_dir = agent_dir / "skill-hub"
        (self.skill_hub_dir / "skills").mkdir(parents=True, exist_ok=True)

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

        # Remove uninstalled skills. uninstall_skill now performs a bounded
        # (5s) blocking revoke HTTP POST, so offload it to a thread to avoid
        # stalling the runner's event loop during catch-up sync.
        import asyncio

        for skill_name in local_skills - server_skills:
            await asyncio.to_thread(self.uninstall_skill, skill_name)
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

    def revoke_skill_token(self, skill_name: str, token_file: str = "token.json") -> bool:
        """Best-effort: revoke the skill's Google grant and delete its token.

        Resolves ``skill-hub/state/<skill>/<token_file>`` under this agent_dir
        and calls :func:`revoke_token`. Swallows all errors — revocation must
        never block uninstall/disconnect. Returns whether a token file was
        present (i.e. there was something to revoke).
        """
        token_path = self.skill_hub_dir / "state" / skill_name / token_file
        try:
            from clawmeets.integrations.auth.google_oauth import revoke_token

            return revoke_token(token_path)
        except Exception as e:  # never let revoke failure escape
            logger.warning(f"Revoke of skill token for {skill_name} failed: {e}")
            return False

    def uninstall_skill(self, skill_name: str) -> None:
        """Revoke the skill's Google grant + delete its token, then remove the
        skill code directory.

        Revocation runs first so uninstall also withdraws consent on Google's
        side (previously the token was left live on disk). Revoke failure is
        logged inside ``revoke_skill_token`` and never raised — code removal
        always proceeds.
        """
        self.revoke_skill_token(skill_name)
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


def _decode_skill_files(files: dict[str, dict]) -> dict[str, bytes]:
    """Decode the ``{relpath: {"content_b64": <ascii>}}`` wire shape used by
    ``SkillSyncPayload.skill_files`` and ``GET /skills/{name}.files``."""
    decoded: dict[str, bytes] = {}
    for relpath, entry in (files or {}).items():
        b64 = (entry or {}).get("content_b64") or ""
        if b64:
            decoded[relpath] = base64.b64decode(b64, validate=True)
    return decoded
