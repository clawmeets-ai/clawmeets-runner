# SPDX-License-Identifier: MIT
"""
clawmeets/runner/turn_scope.py

Classify a turn and compute what it may see (plan section 4.3).

Three turn kinds:

- ``requester``: the project carries a ``requester_key``. The turn serves
  someone other than the owner (a hub conversation, a Front Desk host project,
  a channel conversation). It sees this conversation, the agent's other
  conversations with the SAME requester (read only), and what the owner
  publishes: knowledge dirs, knowledge packs, ``KNOWLEDGE_PACKS.md``,
  ``REFERENCES.md``, ``memory/learnings/`` and skill bodies, all read only.
- ``project``: the project is ``remote_content``. The turn sees this one project
  and the owner's published knowledge, minus ``REFERENCES.md`` and
  ``learnings/``.
- ``full``: everything else; unchanged behaviour. A full turn still lists the
  memory-excluded projects' dirs in ``deny_roots`` so a sandboxed full turn
  (reflection) cannot read them from disk.

Writes in both scoped kinds go to the project sandbox only. Everything the
visibility table marks "Denied" or "Not visible" is listed in ``deny_roots`` as a
real path, so the sandbox profile can deny it explicitly even when an ancestor
directory is allowed.

The function is pure over (project, the local project index, the agent's dirs):
it reads no file contents and makes no network call. The project attributes it
needs (``requester_key``, ``remote_content``, ``memory_excluded``) are read with
``getattr`` so an older project model simply yields a full turn.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, Optional

from ..utils import env_store, nofollow_fs

if TYPE_CHECKING:
    from ..models.context import ModelContext
    from ..models.project import Project

logger = logging.getLogger(__name__)

TurnKind = Literal["full", "project", "requester"]
SCOPED_KINDS: frozenset[str] = frozenset({"project", "requester"})

# Provider and user-session locations that no scoped turn may see. Relative to
# the user's home; resolved at compute time.
USER_SECRET_DIRS: tuple[str, ...] = (
    ".claude",
    ".claude.json",
    ".codex",
    ".gemini",
    ".config/opencode",
    ".local/share/opencode",
    ".ssh",
    ".aws",
    ".config/gcloud",
    ".netrc",
    ".gitconfig",
    ".git-credentials",
    "Library/Keychains",
)


def _real(p: Path) -> Path:
    """Real path of ``p`` (symlinks resolved, even when ``p`` does not exist)."""
    return Path(os.path.realpath(Path(p).expanduser()))


def _uniq(paths: Iterable[Path]) -> tuple[Path, ...]:
    out: list[Path] = []
    for p in paths:
        rp = _real(p)
        if rp not in out:
            out.append(rp)
    return tuple(out)


@dataclass(frozen=True)
class TurnScope:
    """What one turn may read, write and must never see.

    ``read_roots`` are readable (not writable); ``write_roots`` are readable and
    writable; ``deny_roots`` are denied even when an allowed root contains them.
    ``inject_env_keys`` are the env.json keys marked ``--scoped-turns``;
    ``mcp_servers`` the MCP servers opted in with ``clawmeets mcp scoped-turns``.
    ``token`` / ``grant_id`` are filled after the turn token is minted.
    ``write_pins`` holds each write root's ``(st_dev, st_ino)`` taken here, so
    the runner's later writes and uploads can tell the root was swapped
    (``utils.nofollow_fs``) without resolving its path again.
    """

    kind: TurnKind
    read_roots: tuple[Path, ...] = ()
    write_roots: tuple[Path, ...] = ()
    write_pins: tuple[tuple[int, int], ...] = ()
    deny_roots: tuple[Path, ...] = ()
    inject_env_keys: frozenset[str] = frozenset()
    mcp_servers: frozenset[str] = frozenset()
    token: Optional[str] = None
    grant_id: Optional[str] = None
    project_id: str = ""
    requester_key: Optional[str] = None
    # Directories whose top-level instruction files (CLAUDE.md, AGENTS.md,
    # GEMINI.md) the provider may load: the knowledge dirs only (plan item 5).
    instruction_roots: tuple[Path, ...] = ()
    # Project ids the turn may act on; the server computes the authoritative
    # list on the token, this one is the runner's local view.
    project_ids: tuple[str, ...] = ()

    @property
    def scoped(self) -> bool:
        return self.kind in SCOPED_KINDS

    @property
    def server_scope(self) -> str:
        """The ``scope`` value the turn-token route expects."""
        return "requester" if self.kind == "requester" else "project"

    def pin_for(self, root: Path) -> Optional[tuple[int, int]]:
        """The pin of write root ``root``, or ``None`` if it is not one."""
        return dict(zip(self.write_roots, self.write_pins)).get(Path(root))

    def with_grant(self, token: str, grant_id: str) -> "TurnScope":
        return dataclasses.replace(self, token=token, grant_id=grant_id)

    def additional_dirs(self, cwd: Path) -> list[Path]:
        """Every root but the cwd, for the provider's ``--add-dir`` list."""
        cwd_r = _real(cwd)
        return [p for p in (*self.write_roots, *self.read_roots) if p != cwd_r]


def classify_turn(project: "Project") -> TurnKind:
    """``requester`` when the project serves someone else, else ``full``.

    A hub thread's ``remote_content`` does not scope a turn (plan-owner-isolation
    section 3): the requester side is protected by who coordinates. A project a
    scoped turn created is scoped by ``scope_key`` (its migrated key, or the
    one the migration would give it)."""
    from ..models.project import scope_key

    return "requester" if scope_key(project) else "full"


def _real_dir(path: Path) -> Optional[Path]:
    """Real path of ``path`` when it is a real directory, ``None`` when it is
    missing, a symlink or anything else. Its ancestors may be links (macOS
    ``/var``); the last component, which a turn may have written, may not."""
    try:
        st = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return _real(path.parent) / path.name if stat.S_ISDIR(st.st_mode) else None


def ensure_sandbox(sandbox: Path) -> Path:
    """Return the real path of ``sandbox``, made a real directory first.

    A scoped turn can write its sandbox's contents, and once could replace the
    sandbox itself with a symlink (to the agent dir, the user's home), which
    every later realpath of it would follow. Whatever is there that is not a
    real directory is unlinked (never followed) and a fresh directory made.
    """
    sandbox.parent.mkdir(parents=True, exist_ok=True)
    if _real_dir(sandbox) is None and os.path.lexists(sandbox):
        logger.warning(f"turn scope: {sandbox} is not a real directory; rebuilding it")
        os.unlink(sandbox)
    sandbox.mkdir(exist_ok=True)
    real = _real_dir(sandbox)
    assert real is not None
    return real


def _project_dirs(ctx: "ModelContext", project_id: str, project_name: str) -> list[Path]:
    """Every on-disk dir a project owns under the agent home."""
    return [
        ctx.project_dir(project_id, project_name),
        ctx.sandbox_dir(project_id, project_name),
        ctx.changelog_dir(project_id, project_name),
    ]


def _list_projects(ctx: "ModelContext") -> list["Project"]:
    from ..models.project import Project

    try:
        return Project.list_all(ctx)
    except Exception:  # noqa: BLE001 - an unreadable index must not widen scope
        return []


def _agent_secret_paths(agent_dir: Path) -> list[Path]:
    """The agent home's credential-bearing paths (visibility table rows 9-10)."""
    return [
        agent_dir / "credential.json",
        env_store.store_path(agent_dir),
        env_store.scoped_store_path(agent_dir),
        agent_dir / "card.json",
        agent_dir / "skill-hub" / "configs",
        agent_dir / "mcp-hub",
        agent_dir / "AGENTS.md",
        agent_dir / "memory" / "USER.md",
    ]


def _storage_paths(ctx: "ModelContext") -> list[Path]:
    out: list[Path] = []
    storage = ctx.storage
    if storage is not None:
        out.extend([storage.local, storage.shared])
    out.extend(ctx.extra_dirs)
    return out


def compute_turn_scope(
    agent: object,
    project: "Project",
    chatroom: str,
    ctx: "ModelContext",
    *,
    user_home: Optional[Path] = None,
    scoped_mcp_servers: Iterable[str] = (),
) -> TurnScope:
    """Compute the turn's scope. Pure over the project, the local index and dirs.

    ``agent`` is accepted for the plan's signature and future per-agent policy;
    nothing here reads it. ``user_home`` defaults to ``Path.home()`` (tests pass
    a temp dir). ``scoped_mcp_servers`` is the opted-in MCP set.
    """
    del agent, chatroom  # the scope is per project, not per room
    kind = classify_turn(project)
    agent_dir = _real(ctx.base_dir)
    home = _real(user_home or Path.home())
    all_projects = _list_projects(ctx)
    sandbox = ctx.sandbox_dir(project.id, project.name)
    if kind == "full":
        # A sandbox a scoped turn swapped for a link must not be the cwd the
        # runner prepares for the owner's next (unsandboxed) full turn.
        if os.path.lexists(sandbox):
            ensure_sandbox(sandbox)
        excluded: list[Path] = []
        for p in all_projects:
            if p.id != project.id and getattr(p, "memory_excluded", False):
                excluded.extend(_project_dirs(ctx, p.id, p.name))
        return TurnScope(
            kind="full",
            deny_roots=_uniq(excluded),
            project_id=project.id,
            project_ids=(project.id,),
        )

    from ..models.project import scope_key

    requester_key = scope_key(project) if kind == "requester" else None
    sandbox = ensure_sandbox(sandbox)

    # This conversation, and (requester turns) the agent's other conversations
    # with the same requester, read only.
    own_ids = [project.id]
    read: list[Path] = [ctx.project_dir(project.id, project.name)]
    if kind == "requester":
        for p in all_projects:
            if p.id != project.id and scope_key(p) == requester_key:
                own_ids.append(p.id)
                read.append(ctx.project_dir(p.id, p.name))
                # Another turn wrote that sandbox: never follow it if a link.
                other_sb = _real_dir(ctx.sandbox_dir(p.id, p.name))
                if other_sb is not None:
                    read.append(other_sb)

    # What the owner publishes: knowledge, packs, skill bodies.
    knowledge = list(ctx.knowledge_dirs)
    read.extend(knowledge)
    read.append(ctx.packs_dir)
    read.append(ctx.memory_dir / "KNOWLEDGE_PACKS.md")
    read.extend(ctx.skill_source_dirs(role=None))
    read.extend((agent_dir / "system-skill-hub").glob("skills-*"))
    if kind == "requester":
        read.append(ctx.memory_dir / "REFERENCES.md")
        read.append(ctx.memory_dir / "learnings")

    # Everything denied, listed explicitly so an ancestor allow cannot reopen it.
    deny: list[Path] = _agent_secret_paths(agent_dir)
    deny.extend(_storage_paths(ctx))
    deny.append(agent_dir.parent.parent / "turns")  # every turn home, incl. others'
    deny.extend(home / rel for rel in USER_SECRET_DIRS)
    if kind == "project":
        deny.append(ctx.memory_dir / "REFERENCES.md")
        deny.append(ctx.memory_dir / "learnings")
    allowed_ids = set(own_ids)
    for p in all_projects:
        if p.id not in allowed_ids:
            deny.extend(_project_dirs(ctx, p.id, p.name))

    read_roots = _uniq(r for r in read if _real(r) != sandbox)
    write_roots = (sandbox,)
    # A deny never overrides one of the turn's own roots.
    own = set(read_roots) | set(write_roots)
    deny_roots = tuple(d for d in _uniq(deny) if d not in own)

    return TurnScope(
        kind=kind,
        read_roots=read_roots,
        write_roots=write_roots,
        write_pins=(nofollow_fs.pin_dir(sandbox),),
        deny_roots=deny_roots,
        inject_env_keys=frozenset(env_store.scoped_keys(agent_dir)),
        mcp_servers=frozenset(scoped_mcp_servers),
        project_id=project.id,
        requester_key=requester_key,
        instruction_roots=_uniq(knowledge),
        project_ids=tuple(own_ids),
    )
