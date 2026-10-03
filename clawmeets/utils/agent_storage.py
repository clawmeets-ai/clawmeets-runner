# SPDX-License-Identifier: MIT
"""
clawmeets/utils/agent_storage.py
The agent's two persistent storage folders — one place that decides where
they live, creates them, and names the env vars that carry them.

- **Local storage** (``local_settings.local_storage_dir``, default
  ``{AGENT_DIR}/storage``): this agent's own files that outlive any one
  project — caches, downloads, datasets, working state. Unlike the per-project
  sandbox, nothing here is scoped to a chat thread.
- **Shared storage** (``local_settings.shared_storage_dir``, default
  ``{CLAWMEETS_DIR}/shared_storage``): one folder every agent on this computer
  can read and write, for handing files between agents without going through
  a chatroom. ``{AGENT_DIR}/shared_storage`` is a symlink to it so the folder
  shows up in the agent's home.

Both are exposed to every LLM invocation as ``$AGENT_LOCAL_STORAGE_DIR`` /
``$AGENT_SHARED_STORAGE_DIR`` and listed in the prompt's FILES & STATE block.
Paths in local_settings resolve exactly like ``knowledge_dir``
(``FileUtil.resolve_local_dir``): absolute and ``~`` verbatim, relative
against the agent's own home (AGENT_DIR).

Called at agent registration (so a new agent's home already has them), at
every runner start (so agents registered before this existed — or created
from the web app — get them too), and on a settings change.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .file_io import FileUtil

logger = logging.getLogger(__name__)

LOCAL_STORAGE_ENV = "AGENT_LOCAL_STORAGE_DIR"
SHARED_STORAGE_ENV = "AGENT_SHARED_STORAGE_DIR"

LOCAL_STORAGE_KEY = "local_storage_dir"
SHARED_STORAGE_KEY = "shared_storage_dir"

LOCAL_STORAGE_DIRNAME = "storage"
SHARED_STORAGE_DIRNAME = "shared_storage"


@dataclass(frozen=True)
class AgentStorage:
    """Resolved absolute storage folders for one agent."""

    local: Path
    shared: Path

    def env(self) -> dict[str, str]:
        """The env vars handed to every LLM subprocess."""
        return {
            LOCAL_STORAGE_ENV: str(self.local),
            SHARED_STORAGE_ENV: str(self.shared),
        }


def clawmeets_dir_for(agent_dir: Path) -> Path:
    """The ClawMeets data dir (``~/.clawmeets`` by default) an agent lives in.

    Agents live at ``{CLAWMEETS_DIR}/agents/<name>-<id>/``, so the layout
    answers it directly; the env/default fallback only covers an agent dir
    placed somewhere else by hand (``--save``).
    """
    if agent_dir.parent.name == "agents":
        return agent_dir.parent.parent
    return Path(os.environ.get("CLAWMEETS_DATA_DIR", str(Path.home() / ".clawmeets")))


def default_shared_storage_dir(clawmeets_dir: Path) -> Path:
    return clawmeets_dir / SHARED_STORAGE_DIRNAME


def resolve_storage(agent_dir: Path, local_settings: dict) -> AgentStorage:
    """Settings → absolute folders. An empty setting means the default."""
    local = FileUtil.resolve_local_dir(
        local_settings.get(LOCAL_STORAGE_KEY) or "", agent_dir
    )
    shared = FileUtil.resolve_local_dir(
        local_settings.get(SHARED_STORAGE_KEY) or "", agent_dir
    )
    return AgentStorage(
        local=(local or agent_dir / LOCAL_STORAGE_DIRNAME).absolute(),
        shared=(shared or default_shared_storage_dir(clawmeets_dir_for(agent_dir))).absolute(),
    )


def ensure_storage(agent_dir: Path, storage: AgentStorage) -> None:
    """Create both folders and point ``{agent_dir}/shared_storage`` at the
    shared one. Idempotent; never raises — a storage folder that can't be
    created must not stop the runner, the agent just finds it missing.

    The link is only ever created or re-pointed when it is a symlink (or
    absent). A real folder already sitting at that name is left alone.
    """
    for d in (storage.local, storage.shared):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"Could not create storage folder {d}: {e}")

    link = agent_dir / SHARED_STORAGE_DIRNAME
    if link.absolute() == storage.shared:
        return  # the shared folder IS the link location; nothing to link
    try:
        if link.is_symlink():
            if Path(os.readlink(link)) == storage.shared:
                return
            link.unlink()
        elif link.exists():
            logger.warning(
                f"{link} exists and is not a symlink; leaving it alone "
                f"(shared storage is {storage.shared})"
            )
            return
        link.symlink_to(storage.shared, target_is_directory=True)
    except OSError as e:
        # Windows without symlink privilege lands here. The env var and the
        # prompt still carry the real path, so the agent can reach it.
        logger.warning(f"Could not link {link} -> {storage.shared}: {e}")
