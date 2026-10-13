# SPDX-License-Identifier: MIT
"""Runner-mediated git for project turns: the bound remote is never mounted.

A project turn (a hub DM thread, a project created from a restricted turn)
runs sandboxed and holds no git credential. Its sandbox may hold a clone of
the agent's bound repo (``sandbox/projects/<project>/repos/<name>/``); the
bound remote itself can be the user's own working copy
(``file:///Users/…/clawmeets``), and mounting it would let a steered turn
write ``refs/heads/main`` or plant ``.git/hooks/*`` / ``core.fsmonitor`` that
runs with the owner's full powers the next time anyone runs git there.

So the runner does the git that crosses the boundary:

* **Before the turn** ``refresh_mirror`` updates a runner-owned bare staging
  mirror (``~/.clawmeets/repo-mirrors/<agent>/<project>.git``) with only the
  project branch and the base branch from the configured URL. The turn sees
  it read-only (as ``CLAWMEETS_AGENT_GIT_URL``) and can fetch and rebase.
* **After the turn** ``push_project_branch`` fetches ``refs/heads/project/<slug>``
  from the turn's clone into the mirror (the clone is only ever the REMOTE of
  that fetch, so its config and hooks are never loaded as the runner's own),
  then pushes that one ref, never forced, to the URL from the agent's config
  (never the clone's ``origin``). A non-fast-forward is refused.

Every git command here runs with hooks and fsmonitor off, no system config,
and transport protocols limited to the configured URL's scheme. ``main`` and
every other branch of the remote are unreachable whatever the remote's own
branch protection.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_BRANCH_PREFIX = "project/"
_SAFE_SLUG = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
#: Flags on every runner git command near turn-written content.
GIT_SAFETY = (
    "-c", "core.hooksPath=/dev/null",
    "-c", "core.fsmonitor=false",
    "-c", "protocol.ext.allow=never",
    "-c", "uploadpack.allowAnySHA1InWant=false",
)


@dataclass(frozen=True)
class PushResult:
    pushed: bool
    reason: Optional[str] = None      # "unchanged" | "non_fast_forward" | "no_branch" | "error: …"
    branch: Optional[str] = None


def project_branch(project_name: str) -> str:
    if not _SAFE_SLUG.match(project_name) or project_name.startswith("."):
        raise ValueError(f"unsafe project slug {project_name!r}")
    return f"{PROJECT_BRANCH_PREFIX}{project_name}"


def _scheme(url: str) -> str:
    if "://" in url:
        return url.split("://", 1)[0].lower()
    if re.match(r"^[\w.-]+@[\w.-]+:", url):
        return "ssh"
    return "file"


#: Files that make git read objects or refs from another repository on disk.
_FOREIGN_REPO_POINTERS = ("commondir", "objects/info/alternates", "objects/info/http-alternates")


def unsafe_clone_reason(clone: Path) -> Optional[str]:
    """Why the turn's clone must not be fetched from, or None when it is safe.

    The runner's fetch runs outside the sandbox, so anything in the clone that
    redirects git to another repository on disk (a gitfile ``.git``, a
    symlinked ``.git`` or anything under it, alternates, ``commondir``) would
    let the turn copy that repository's objects into the mirror and push them.
    """
    if clone.is_symlink() or clone.parent.is_symlink():
        return "clone path is a symlink"
    git_dir = clone / ".git"
    if git_dir.is_symlink() or not git_dir.is_dir():
        return ".git is not a real directory"
    for rel in _FOREIGN_REPO_POINTERS:
        if os.path.lexists(git_dir / rel):
            return f".git/{rel} present"
    for root, dirs, files in os.walk(git_dir):
        for name in (*dirs, *files):
            if os.path.islink(os.path.join(root, name)):
                return f"symlink inside .git: {os.path.relpath(os.path.join(root, name), git_dir)}"
    return None


class RepoPusher:
    """Owns the staging mirrors and the one push a project turn may cause."""

    def __init__(self, agent_dir: Path, git_url: str, *, base_branch: str = "main",
                 mirrors_root: Optional[Path] = None, git: str = "git") -> None:
        self._agent_dir = Path(agent_dir)
        self._url = git_url
        self._base = base_branch
        root = mirrors_root or (self._agent_dir.parent.parent / "repo-mirrors" / self._agent_dir.name)
        self._root = Path(root)
        self._git = git

    # ── plumbing ─────────────────────────────────────────────────────────

    def _env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update({
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ALLOW_PROTOCOL": ":".join(sorted({_scheme(self._url), "file"})),
        })
        return env

    def _run(self, *args: str, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
        cmd = [self._git, *GIT_SAFETY, *args]
        return subprocess.run(cmd, cwd=cwd, env=self._env(), capture_output=True, text=True,
                              timeout=300, check=check)

    def mirror_path(self, project_id: str, project_name: str) -> Path:
        return self._root / f"{project_name}-{project_id}.git"

    # ── before the turn ──────────────────────────────────────────────────

    def refresh_mirror(self, project_id: str, project_name: str) -> Path:
        """Create or update the bare mirror with the project and base branches
        only. Returns its path (bind it read-only into the turn)."""
        branch = project_branch(project_name)
        mirror = self.mirror_path(project_id, project_name)
        if not (mirror / "HEAD").exists():
            mirror.mkdir(parents=True, exist_ok=True)
            self._run("init", "--bare", "--quiet", str(mirror))
        for ref in (self._base, branch):
            self._run("--git-dir", str(mirror), "fetch", "--quiet", "--no-tags", self._url,
                      f"+refs/heads/{ref}:refs/heads/{ref}", check=False)
        return mirror

    # ── after the turn ───────────────────────────────────────────────────

    def push_project_branch(self, project_id: str, project_name: str, clone: Path) -> PushResult:
        """Fetch the project branch from the turn's clone into the mirror and
        push that one ref to the configured URL, fast-forward only."""
        branch = project_branch(project_name)
        refusal = unsafe_clone_reason(Path(clone))
        if refusal is not None:
            logger.warning("refusing to fetch from turn clone %s: %s", clone, refusal)
            return PushResult(False, f"unsafe_clone: {refusal}", branch)
        ref = f"refs/heads/{branch}"
        mirror = self.refresh_mirror(project_id, project_name)
        before = self._rev(mirror, ref)
        staged = f"refs/turn/{branch}"
        got = self._run("--git-dir", str(mirror), "fetch", "--quiet", "--no-tags",
                        str(Path(clone).resolve()), f"+{ref}:{staged}", check=False)
        if got.returncode != 0:
            return PushResult(False, "no_branch", branch)
        after = self._rev(mirror, staged)
        if after is None or after == before:
            return PushResult(False, "unchanged", branch)
        pushed = self._run("--git-dir", str(mirror), "push", "--quiet", "--no-verify",
                           self._url, f"{staged}:{ref}", check=False)
        if pushed.returncode != 0:
            text = (pushed.stderr or "") + (pushed.stdout or "")
            if "non-fast-forward" in text or "rejected" in text or "fetch first" in text:
                return PushResult(False, "non_fast_forward", branch)
            return PushResult(False, f"error: {text.strip()[:200]}", branch)
        self._run("--git-dir", str(mirror), "update-ref", ref, after, check=False)
        return PushResult(True, None, branch)

    def _rev(self, git_dir: Path, ref: str) -> Optional[str]:
        out = self._run("--git-dir", str(git_dir), "rev-parse", "--verify", "--quiet", ref, check=False)
        return out.stdout.strip() or None if out.returncode == 0 else None

    def clones_in(self, sandbox: Path) -> list[Path]:
        """Candidate clones of the bound repo in a project sandbox:
        ``repos/<name>/`` with a ``.git``. Their content is untrusted; they are
        only ever fetched FROM."""
        repos = Path(sandbox) / "repos"
        if not repos.is_dir():
            return []
        return sorted(p for p in repos.iterdir() if p.is_dir() and (p / ".git").exists()
                      and not p.is_symlink())
