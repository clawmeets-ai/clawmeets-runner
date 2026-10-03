# SPDX-License-Identifier: MIT
"""
clawmeets/runner/agent_dir_readmes.py
README.md files for the folders the runner creates and manages inside an
agent's home folder.

The home-folder browser lets the owner read an agent's folder directly, so
each system folder explains itself in place: what it holds, who writes it,
and whether hand edits survive. The text lives here — next to nothing else —
so a folder's README changes in the same commit as the code that changes the
folder.

Written at every runner start (and when the storage settings change).
System-folder READMEs are owned by the runner and rewritten whenever their
text differs; the README in a storage folder is written only when missing,
because those folders belong to the agent and the owner.

Not covered on purpose: ``agents/`` (synced peer cards; a stray file there is
read as a peer) and the per-project subfolders of ``projects/`` / ``metadata/``
/ ``sandbox/`` (one README per project would be noise).
"""
from __future__ import annotations

import logging
from pathlib import Path

from clawmeets.utils.agent_storage import (
    LOCAL_STORAGE_DIRNAME,
    LOCAL_STORAGE_ENV,
    SHARED_STORAGE_DIRNAME,
    SHARED_STORAGE_ENV,
    AgentStorage,
    clawmeets_dir_for,
    default_shared_storage_dir,
)
from clawmeets.utils.file_io import FileUtil

logger = logging.getLogger(__name__)

README_FILENAME = "README.md"

_MANAGED_NOTE = (
    "_Written by the ClawMeets runner on every start — edits to this file are "
    "overwritten._"
)


def _root_readme(storage: AgentStorage) -> str:
    return f"""# Agent home folder

Everything this agent keeps on this computer. Only chats and files shared in a
chatroom are synced to the ClawMeets server; the rest of this folder stays here.

| Path | What it is | Who writes it |
|------|------------|---------------|
| `card.json` | The agent's identity and local settings (model, git repo, knowledge folder, storage folders, skill/MCP configs). Holds secrets. | Runner, from Agent Settings |
| `credential.json`, `env.json` | The agent's server token and its local env vars (`clawmeets env set`). Secrets. | Runner / CLI |
| `AGENTS.md` | Roster of agents this agent can delegate to. | Runner |
| `agents/` | Synced cards of the other agents it can see. Secrets. | Runner |
| `memory/` | What the agent has learned about you and its field. | The agent |
| `knowledge_packs/` | Knowledge packs you installed on this agent. | Runner |
| `skill-hub/` | Skills installed from the catalog, plus their configs and login state. | Runner |
| `personal-skill-hub/` | Skills this agent wrote for itself. | The agent |
| `system-skill-hub/` | Built-in skills that ship with ClawMeets. | Runner |
| `mcp-hub/` | Installed MCP servers (integrations) and their configs. | Runner |
| `projects/` | Synced copy of every project's chatrooms and shared files. | Runner (sync) |
| `metadata/` | Per-project bookkeeping: changelog, costs, model logs. | Runner |
| `sandbox/` | One working folder per project — where the agent actually works. | The agent |
| `{LOCAL_STORAGE_DIRNAME}/` | Local storage: files the agent keeps across projects (`${LOCAL_STORAGE_ENV}`). | The agent |
| `{SHARED_STORAGE_DIRNAME}/` | Link to shared storage, one folder all agents on this computer share (`${SHARED_STORAGE_ENV}`). | Any agent |
| `stdout.log`, `stderr.log`, `agent.pid` | The runner process's logs and process id. | Runner |

This agent's storage folders right now:

- Local storage: `{storage.local}`
- Shared storage: `{storage.shared}`

Each folder above has its own README.md with details.

{_MANAGED_NOTE}
"""


_FOLDER_READMES: dict[str, str] = {
    "memory": f"""# memory/

The agent's long-term memory — what it has learned about you, your world and
its field. The agent writes here itself, mainly during scheduled reflection and
the first-run personalize step; you can read, correct or delete anything.

- `USER.md` — (assistant only) who you are and how you like to work.
- `KNOWLEDGE_PACKS.md` — index of the knowledge packs installed on this agent.
- `REFERENCES.md` — auto-built index of your knowledge folder (Agent Settings →
  Knowledge). Rebuilt by the runner; don't hand-edit.
- `REPO.md` — conventions the agent has learned about its bound git repo.
- `learnings/` — field knowledge, one file per topic, indexed by
  `learnings/INDEX.md`.

Not synced to the server and never shared in chat.

{_MANAGED_NOTE}
""",
    "knowledge_packs": f"""# knowledge_packs/

The knowledge packs you installed on this agent (Agent Settings → Knowledge),
one folder per pack. The runner mirrors them from the server: installing or
uninstalling a pack adds or removes its folder here, so edit a pack in the web
app, not here. `_meta.json` records what is installed; the agent finds packs
through `memory/KNOWLEDGE_PACKS.md`.

{_MANAGED_NOTE}
""",
    "skill-hub": f"""# skill-hub/

Skills installed on this agent from the ClawMeets catalog (Agent Settings →
Skills).

- `skills/<name>/SKILL.md` — each skill's instructions, downloaded from the
  server. Replaced on reinstall; don't edit.
- `configs/<name>.json` — each skill's settings (sources, API keys, schedules).
  Change them through Agent Settings; direct edits are overwritten.
- `state/<name>/` — login tokens and browser state a skill keeps between runs.

`configs/` and `state/` hold secrets.

{_MANAGED_NOTE}
""",
    "personal-skill-hub": f"""# personal-skill-hub/

Skills this agent wrote for itself — procedures it found itself repeating,
saved so it can run them again. Each lives at `skills/<name>/SKILL.md` and is
invoked as `/personal:<name>`.

In the home-folder browser you can read, edit or delete them; your assistant
can read them but not change them.

{_MANAGED_NOTE}
""",
    "system-skill-hub": f"""# system-skill-hub/

Built-in skills that ship with ClawMeets (git workflow, reflection, plans,
reports, …). `skills-<role>/` holds links to the subset each role sees —
`assistant`, `coordinator` or `worker`. Rebuilt from the installed ClawMeets
package on every start; upgrade ClawMeets to change them.

{_MANAGED_NOTE}
""",
    "mcp-hub": f"""# mcp-hub/

MCP servers (integrations) installed on this agent (Agent Settings →
Integrations).

- `manifests/` — how to launch each server, downloaded from ClawMeets.
- `servers/<name>/` — each server's local state, including OAuth tokens.
- `configs/<name>.json` — each server's settings. Change them through Agent
  Settings; direct edits are overwritten.
- `dist/` — bundled server code.

`servers/` and `configs/` hold secrets.

{_MANAGED_NOTE}
""",
    "projects": f"""# projects/

A synced copy of every project this agent is in, one folder per project
(`<project-name>-<project-id>/`). Each `chatrooms/<room>/` holds the room's
messages (`CHATS.ndjson`), members, and the files shared in it.

Kept in sync with the server — edit nothing here; a change is lost on the next
sync. The agent reads these as read-only project files and shares files back
through the chat.

{_MANAGED_NOTE}
""",
    "metadata": f"""# metadata/

The runner's per-project bookkeeping, under `projects/<project-name>-<project-id>/`:

- `meta.json` — the project's settings as last synced.
- `changelog.ndjson`, `runloop_state.json` — the sync log and how far the agent
  has processed it.
- `cost.ndjson` — model usage and cost per turn.
- `cli-stdout.log`, `cli-stderr.log` — raw output of each model run; the first
  place to look when a turn failed.

{_MANAGED_NOTE}
""",
    "sandbox": f"""# sandbox/

The agent's working folders, one per project at
`projects/<project-name>-<project-id>/`. Each model turn runs with that folder
as its working directory: drafts, scratch files, and git clones (`repos/`) live
there. A file only reaches the chat when the agent shares it.

Use local storage (`{LOCAL_STORAGE_DIRNAME}/`) for files that should outlive
a project.

{_MANAGED_NOTE}
""",
}


def _local_storage_readme() -> str:
    return f"""# storage/

This agent's local storage — files it keeps across projects: caches,
downloads, datasets, working state. The agent finds it as
`${LOCAL_STORAGE_ENV}`; the location is set in Agent Settings → Storage.

Private to this agent and this computer: nothing here is synced or shared in
chat. Organize it however you like — this README is written once and never
overwritten.
"""


def _shared_storage_readme() -> str:
    return f"""# shared_storage/

Shared storage for every agent on this computer. Each agent finds it as
`${SHARED_STORAGE_ENV}`, and it is linked into each agent's home folder as
`{SHARED_STORAGE_DIRNAME}/`.

Use it to hand files between agents without posting them in a chat. Nothing
here is synced to the ClawMeets server. Every agent can change or delete what
is here, so give each agent or workflow its own subfolder.

This README is written once and never overwritten.
"""


def _write_if_changed(path: Path, text: str) -> None:
    if FileUtil.read(path, "text") != text:
        FileUtil.write(path, text, "text")


def write_agent_dir_readmes(agent_dir: Path, storage: AgentStorage) -> None:
    """Write the README.md files for the agent's system folders and its
    default storage folders. Never raises — a README is documentation, and a
    failure to write one must not stop the runner."""
    try:
        _write_if_changed(agent_dir / README_FILENAME, _root_readme(storage))
        for folder, text in _FOLDER_READMES.items():
            d = agent_dir / folder
            d.mkdir(parents=True, exist_ok=True)
            _write_if_changed(d / README_FILENAME, text)

        # Storage folders get a README only where ClawMeets chose the
        # location; a folder the owner pointed us at is theirs.
        defaults = (
            (agent_dir / LOCAL_STORAGE_DIRNAME, storage.local, _local_storage_readme()),
            (
                default_shared_storage_dir(clawmeets_dir_for(agent_dir)),
                storage.shared,
                _shared_storage_readme(),
            ),
        )
        for default, actual, text in defaults:
            readme = actual / README_FILENAME
            if actual == default.absolute() and actual.is_dir() and not readme.exists():
                FileUtil.write(readme, text, "text")
    except OSError as e:
        logger.warning(f"Could not write agent-folder READMEs in {agent_dir}: {e}")
