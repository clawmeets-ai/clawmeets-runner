# SPDX-License-Identifier: MIT
"""
clawmeets/runner/system_skill_manager.py

Manages the system-skill-hub directory for an agent.

System skills are bundled with the runner package (``clawmeets/system_skills/``)
and ship a ``manifest.json`` sibling declaring ``audiences`` (worker /
coordinator / assistant) and ``surfaces`` (agent_runtime / user_cli). The
runner pre-materializes per-audience symlink trees at startup; each LLM
provider's ``_prepare_invocation`` picks the dir for the current audience
and feeds it into ``materialize_skill_tree`` as the BASE layer (below
``skill-hub/`` and ``personal-skill-hub/``).

Layout::

    {agent_dir}/system-skill-hub/
    ├── skills-worker/
    │   ├── reflect -> {pkg}/system_skills/reflect
    │   └── …
    ├── skills-coordinator/
    │   └── …
    └── skills-assistant/
        ├── reflect -> …
        ├── interview -> …
        └── …
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("clawmeets.runner")

AUDIENCES: tuple[str, ...] = ("worker", "coordinator", "assistant")
SURFACE_AGENT_RUNTIME = "agent_runtime"


class SystemSkillManager:
    """Owns ``{agent_dir}/system-skill-hub/`` on the runner.

    Source-of-truth is the bundled ``clawmeets/system_skills/`` tree.
    Per-audience filtered views are pre-materialized at runner startup.
    """

    def __init__(self, agent_dir: Path) -> None:
        self.hub_dir = agent_dir / "system-skill-hub"
        self.hub_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def source_dir(cls) -> Path:
        """Resolve the bundled ``clawmeets/system_skills/`` dir.

        Works for both source-tree dev (this file lives at
        ``clawmeets/runner/system_skill_manager.py``) and installed pip
        wheel (same relative layout under ``site-packages/clawmeets/``).
        """
        return Path(__file__).resolve().parent.parent / "system_skills"

    def skills_dir_for(self, audience: str) -> Path:
        """The per-audience symlink-tree dir. May not exist if
        ``materialize_audiences`` has not run yet."""
        return self.hub_dir / f"skills-{audience}"

    def materialize_audiences(self) -> None:
        """Build ``skills-worker/``, ``skills-coordinator/``, and
        ``skills-assistant/`` as symlink trees rooted at the bundled
        source. Idempotent — wipes and rebuilds each per-audience dir
        on every call (cheap; symlinks only)."""
        source = self.source_dir()
        if not source.is_dir():
            logger.warning(
                "SystemSkillManager: source dir not found at %s; "
                "system-skill-hub will be empty",
                source,
            )
            entries: dict[str, tuple[Path, set[str]]] = {}
        else:
            entries = _read_runtime_manifests(source)

        for audience in AUDIENCES:
            target = self.skills_dir_for(audience)
            _replace_dir(target)
            for name, (skill_dir, audiences) in entries.items():
                if audience in audiences:
                    (target / name).symlink_to(skill_dir.resolve())


def _read_runtime_manifests(source: Path) -> dict[str, tuple[Path, set[str]]]:
    """Walk ``source`` for ``<name>/{SKILL.md,manifest.json}`` pairs.

    Returns ``{name: (skill_dir, audiences_set)}`` for skills whose
    manifest declares ``surfaces`` containing ``agent_runtime``. Skills
    without ``agent_runtime`` (e.g. signup / bootstrap / start) are
    skipped — they only exist for the Claude plugin tree.
    """
    out: dict[str, tuple[Path, set[str]]] = {}
    for skill_dir in sorted(source.iterdir()):
        if not skill_dir.is_dir():
            continue
        if not (skill_dir / "SKILL.md").exists():
            continue
        manifest_path = skill_dir / "manifest.json"
        if not manifest_path.exists():
            logger.warning(
                "SystemSkillManager: %s missing manifest.json — skipping",
                skill_dir.name,
            )
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as e:
            logger.warning(
                "SystemSkillManager: failed to parse %s: %s — skipping",
                manifest_path,
                e,
            )
            continue
        surfaces = set(manifest.get("surfaces") or [])
        if SURFACE_AGENT_RUNTIME not in surfaces:
            continue
        audiences = set(manifest.get("audiences") or [])
        if not audiences:
            # agent_runtime surface but no audience — nothing to materialize
            continue
        out[skill_dir.name] = (skill_dir, audiences)
    return out


def _replace_dir(target: Path) -> None:
    """Wipe and recreate ``target`` as a fresh empty dir."""
    if target.is_symlink() or target.exists():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.mkdir(parents=True, exist_ok=True)
