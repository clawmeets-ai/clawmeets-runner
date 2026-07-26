# SPDX-License-Identifier: MIT
"""
clawmeets/integrations/openmontage/_lib.py

Deterministic stage-handoff harness for OpenMontage pipelines — paired
skills: om-stage (worker) / om-produce (coordinator).

OpenMontage (an agent-first video production system) carries the stage
know-how itself: pipeline manifests in ``pipeline_defs/<pipeline>.yaml``,
per-stage director skills in ``skills/<skill>.md``, and Python tools run
from the repo root. This module owns everything around a stage that must
not be left to LLM prose: the state-fork git mechanics (clone/fetch,
``pull --ff-only``, data-only commit scope, push), the single-writer
lease, the source-pin verification, and the binary-artifact mode.

State channel: one shared git branch (the "state-fork") where every
handoff is a data commit under ``projects/<p>/`` — the commit IS the
receipt. Binary artifacts ride one of two modes, declared by a committed
``.om-state.json`` marker at the fork root (a property of the fork, not
of any agent, so a mixed fleet fails loudly instead of diverging):

  - ``"binaries": "git"`` (default) — assets/renders are committed too.
    Universal (works with just a reachable git remote) but accretes; the
    sandbox cost is mitigated with blob-filtered partial clones.
  - ``"binaries": "media_root"`` — assets/renders live on a shared
    directory (NFS across machines); the working tree holds gitignored
    symlinks and checkpoints reference absolute paths + sha256.

Per-agent config (``skill-hub/configs/om-stage.json``, resolved via
``integrations/_config_resolve``):

  {
    "repo_dir": "/abs/path/to/OpenMontage",   // required: local install
    "state_remote": "",   // default: file://{repo_dir}
    "state_branch": "",   // default: state-fork/openmontage
    "media_root": ""      // required only when the fork declares media_root
  }

Single-writer enforcement is layered: an atomic lease file (media_root
when shared, else ``{repo_dir}/.om-leases/``) stops same-host races
up-front, and ``pull --ff-only`` / push-rejection catches anything the
lease can't see (e.g. git-mode fleets spread across machines). Neither
layer ever auto-merges — a violation is surfaced to the coordinator.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from clawmeets.integrations._config_resolve import resolve_skill_config_path
from clawmeets.utils.file_io import FileUtil

SKILL_NAME = "om-stage"
DEFAULT_STATE_BRANCH = "state-fork/openmontage"
DEFAULT_WORKDIR = "repos/openmontage-state"
STATE_MARKER = ".om-state.json"
LEASE_TTL_SECONDS = 3600  # one LLM kill window past the 1800s invocation cap

_PROJECT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(explicit_config: str = "") -> dict[str, str]:
    """Resolve and validate the om-stage config. Raises RuntimeError with an
    actionable message when missing/invalid — the SKILL.md relays it verbatim.
    """
    path = resolve_skill_config_path(SKILL_NAME, explicit_config)
    if not path:
        raise RuntimeError(
            "om-stage is not configured. Open Agent Settings → Skills → "
            "om-stage → Configure and set repo_dir to your OpenMontage "
            "install (or pass --config)."
        )
    data = FileUtil.read(Path(path), "json")
    if not isinstance(data, dict):
        raise RuntimeError(f"om-stage config at {path} is not valid JSON.")
    repo_dir = str(data.get("repo_dir") or "").strip()
    if not repo_dir:
        raise RuntimeError(f"om-stage config at {path} is missing 'repo_dir'.")
    repo = Path(repo_dir).expanduser()
    if not repo.is_dir():
        raise RuntimeError(
            f"repo_dir {repo} does not exist. Install OpenMontage there "
            f"(git clone + `make setup`) or fix the config."
        )
    return {
        "repo_dir": str(repo),
        "state_remote": str(data.get("state_remote") or "").strip()
        or f"file://{repo}",
        "state_branch": str(data.get("state_branch") or "").strip()
        or DEFAULT_STATE_BRANCH,
        "media_root": str(data.get("media_root") or "").strip(),
    }


def _validate_project(project: str) -> str:
    if not _PROJECT_RE.match(project):
        raise ValueError(
            f"Invalid project slug {project!r}: lowercase letters, digits, "
            f"'.', '_', '-' only."
        )
    return project


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(workdir: Path, *args: str) -> str:
    """Run git in ``workdir``; raise RuntimeError with stderr on failure."""
    proc = subprocess.run(
        ["git", "-C", str(workdir), *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def _clone_or_update(cfg: dict[str, str], workdir: Path) -> None:
    branch = cfg["state_branch"]
    if (workdir / ".git").is_dir():
        _git(workdir, "fetch", "origin")
        _git(workdir, "checkout", branch)
        try:
            _git(workdir, "pull", "--ff-only", "origin", branch)
        except RuntimeError as exc:
            raise RuntimeError(
                f"State branch {branch} cannot fast-forward — another writer "
                f"violated single-writer. Do NOT merge; report to the "
                f"coordinator. ({exc})"
            ) from exc
        return
    workdir.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "git", "clone",
            "--filter=blob:none",  # skip historical blobs (old takes/renders)
            "--branch", branch,
            cfg["state_remote"], str(workdir),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Cannot clone state fork {cfg['state_remote']} (branch {branch}): "
            f"{proc.stderr.strip()}"
        )


def _read_state_marker(workdir: Path) -> tuple[dict[str, Any], list[str]]:
    """Read ``.om-state.json`` from the fork. Returns (marker, warnings)."""
    warnings: list[str] = []
    marker = FileUtil.read(workdir / STATE_MARKER, "json")
    if not isinstance(marker, dict):
        warnings.append(
            f"{STATE_MARKER} missing on the state branch — assuming "
            f'{{"binaries": "git"}}. Commit the marker to make the mode explicit.'
        )
        marker = {"binaries": "git"}
    mode = marker.get("binaries", "git")
    if mode not in ("git", "media_root"):
        raise RuntimeError(f'{STATE_MARKER} has unknown binaries mode {mode!r}.')
    return marker, warnings


def _check_trackable(workdir: Path, project: str, mode: str) -> None:
    """The fork's .gitignore must let pipeline state (and, in git mode,
    binaries) be staged by a plain ``git add`` — an ignore hit here means
    ``.om-state.json`` and ``.gitignore`` diverged on the fork."""
    probes = [f"projects/{project}/pipeline/checkpoint_probe.json"]
    if mode == "git":
        probes.append(f"projects/{project}/assets/probe.bin")
    ignored = [
        p for p in probes
        if subprocess.run(
            ["git", "-C", str(workdir), "check-ignore", "-q", p],
            capture_output=True,
        ).returncode == 0
    ]
    if ignored:
        raise RuntimeError(
            f"Fork .gitignore ignores {ignored} while {STATE_MARKER} declares "
            f"binaries={mode} — the fork's marker and .gitignore diverged. "
            f"Fix the fork (docs/STATE_PROTOCOL.md §2) before running stages."
        )


def _verify_source_pin(workdir: Path, marker: dict[str, Any]) -> list[str]:
    """Every path changed since the pinned upstream commit must be project
    data or explicitly sanctioned. Returns warnings; raises on violation."""
    pin = str(marker.get("source_pin") or "").strip()
    if not pin:
        return [
            f"{STATE_MARKER} carries no source_pin — skipping source-identity "
            f"verification."
        ]
    sanctioned = tuple(marker.get("sanctioned_paths") or [])
    out = _git(workdir, "diff", "--name-only", f"{pin}..HEAD")
    bad = [
        p for p in out.splitlines()
        if p and not p.startswith("projects/") and not p.startswith(sanctioned)
    ]
    if bad:
        raise RuntimeError(
            f"Source-identity violation: non-sanctioned paths changed since "
            f"the pin {pin[:12]}: {bad[:10]}. Stop and report to the coordinator."
        )
    return []


# ---------------------------------------------------------------------------
# Lease (single-writer, same-host layer)
# ---------------------------------------------------------------------------

def _lease_dir(cfg: dict[str, str]) -> Path:
    if cfg["media_root"]:
        return Path(cfg["media_root"]).expanduser() / ".om-leases"
    return Path(cfg["repo_dir"]) / ".om-leases"


def _lease_path(cfg: dict[str, str], project: str) -> Path:
    return _lease_dir(cfg) / f"{project}.lease"


def _agent_identity() -> str:
    return os.environ.get("CLAWMEETS_AGENT_ID", "") or f"pid-{os.getpid()}"


def acquire_lease(cfg: dict[str, str], project: str, stage: str) -> dict[str, Any]:
    """Atomically create the per-project lease. Stale leases (older than
    LEASE_TTL_SECONDS — a crashed run; live invocations are killed at 1800s)
    are broken with a note in the returned lease dict."""
    path = _lease_path(cfg, project)
    path.parent.mkdir(parents=True, exist_ok=True)
    lease = {
        "agent_id": _agent_identity(),
        "stage": stage,
        "acquired_at": time.time(),
    }
    payload = json.dumps(lease).encode()
    for attempt in (1, 2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
            return lease
        except FileExistsError:
            holder = FileUtil.read(path, "json") or {}
            age = time.time() - float(holder.get("acquired_at") or 0)
            if attempt == 1 and age > LEASE_TTL_SECONDS:
                path.unlink(missing_ok=True)  # stale: crashed run, break it
                lease["broke_stale_lease"] = holder
                continue
            raise RuntimeError(
                f"Project {project!r} is being written by "
                f"{holder.get('agent_id', 'unknown')} (stage "
                f"{holder.get('stage', '?')}, {int(age)}s ago). One writer per "
                f"project — wait for its receipt or ask the coordinator to "
                f"run stage-abort."
            )
    raise RuntimeError(f"Could not acquire lease for {project!r}.")  # unreachable


def release_lease(
    cfg: dict[str, str], project: str, *, force: bool = False
) -> Optional[dict[str, Any]]:
    """Release the lease if held by this agent (or unconditionally with
    ``force``). Returns the released lease dict, or None if no lease."""
    path = _lease_path(cfg, project)
    holder = FileUtil.read(path, "json")
    if holder is None:
        return None
    if not force and holder.get("agent_id") != _agent_identity():
        raise RuntimeError(
            f"Lease on {project!r} is held by {holder.get('agent_id')!r}, not "
            f"you. Use --force only on the coordinator's instruction."
        )
    path.unlink(missing_ok=True)
    return holder


# ---------------------------------------------------------------------------
# Media root (binaries=media_root mode)
# ---------------------------------------------------------------------------

_MEDIA_SUBDIRS = ("assets", "renders")


def _wire_media_symlinks(
    cfg: dict[str, str], workdir: Path, project: str
) -> dict[str, str]:
    """Point the clone's ``projects/<p>/{assets,renders}`` at the shared
    media root. Tools keep writing where they always write; bytes land on
    shared storage; git sees gitignored symlinks."""
    if not cfg["media_root"]:
        raise RuntimeError(
            "The state fork declares binaries=media_root but this agent has "
            "no media_root configured. Set it in Agent Settings → Skills → "
            "om-stage → Configure (must point at the shared media directory)."
        )
    root = Path(cfg["media_root"]).expanduser()
    if not root.is_dir():
        raise RuntimeError(
            f"media_root {root} does not exist or is not mounted."
        )
    proj_dir = workdir / "projects" / project
    proj_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for sub in _MEDIA_SUBDIRS:
        target = root / "projects" / project / sub
        target.mkdir(parents=True, exist_ok=True)
        link = proj_dir / sub
        if link.is_symlink():
            if link.resolve() != target.resolve():
                link.unlink()
                link.symlink_to(target)
        elif link.exists():
            if any(link.iterdir()):
                raise RuntimeError(
                    f"{link} exists with content but should be a symlink to "
                    f"{target}. Resolve manually (move the content to the "
                    f"media root) before continuing."
                )
            link.rmdir()
            link.symlink_to(target)
        else:
            link.symlink_to(target)
        out[sub] = str(target)
    return out


def _media_manifest(
    cfg: dict[str, str], project: str, since: float
) -> list[dict[str, Any]]:
    """Hash media files touched since ``since`` (lease acquisition) so the
    receipt carries verifiable pointers."""
    if not cfg["media_root"]:
        return []
    manifest = []
    base = Path(cfg["media_root"]).expanduser() / "projects" / project
    for sub in _MEDIA_SUBDIRS:
        d = base / sub
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.stat().st_mtime >= since:
                manifest.append({
                    "path": str(f),
                    "size": f.stat().st_size,
                    "sha256": FileUtil.sha256(f.read_bytes()),
                })
    return manifest


# ---------------------------------------------------------------------------
# Pipeline manifest
# ---------------------------------------------------------------------------

def _load_stage_entry(
    workdir: Path, pipeline: str, stage: str
) -> tuple[dict[str, Any], Path]:
    """Return (stage entry from the manifest, absolute director-skill path)."""
    import yaml

    manifest_path = workdir / "pipeline_defs" / f"{pipeline}.yaml"
    if not manifest_path.is_file():
        available = sorted(
            p.stem for p in (workdir / "pipeline_defs").glob("*.yaml")
        )
        raise RuntimeError(
            f"Unknown pipeline {pipeline!r}. Available: {available}"
        )
    manifest = yaml.safe_load(manifest_path.read_text())
    stages = manifest.get("stages") or []
    entry = next((s for s in stages if s.get("name") == stage), None)
    if entry is None:
        raise RuntimeError(
            f"Pipeline {pipeline!r} has no stage {stage!r}. Stages: "
            f"{[s.get('name') for s in stages]}"
        )
    skill_rel = entry.get("skill") or ""
    director = workdir / "skills" / f"{skill_rel}.md"
    if skill_rel and not director.is_file():
        raise RuntimeError(
            f"Director skill {director} not found on the state branch."
        )
    entry = dict(entry)
    entry["orchestration"] = manifest.get("orchestration") or {}
    return entry, director


# ---------------------------------------------------------------------------
# Protocol path binding (STATE_PROTOCOL.md §2)
# ---------------------------------------------------------------------------

def _pipeline_paths(workdir: Path, project: str) -> dict[str, str]:
    """The pinned write_checkpoint binding: pipeline_dir=projects/<p>,
    project_id="pipeline" → checkpoints at projects/<p>/pipeline/…"""
    pdir = workdir / "projects" / project
    return {
        "pipeline_dir": str(pdir),           # pass as write_checkpoint(pipeline_dir=…)
        "project_id": "pipeline",            # pass as write_checkpoint(project_id=…)
        "checkpoint_dir": str(pdir / "pipeline"),
        "decision_log": str(pdir / "pipeline" / "decision_log.json"),
        "cost_log": str(pdir / "pipeline" / "cost_log.json"),
        "creative_dir": str(pdir / "pipeline" / "creative"),
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def stage_begin(
    project: str,
    stage: str,
    pipeline: str,
    *,
    workdir: str = DEFAULT_WORKDIR,
    explicit_config: str = "",
) -> dict[str, Any]:
    """Prepare a stage: sync the fork, verify identity, acquire the lease,
    wire media symlinks (media_root mode), and emit everything the agent
    needs to execute the stage."""
    _validate_project(project)
    cfg = load_config(explicit_config)
    warnings: list[str] = []

    venv_python = Path(cfg["repo_dir"]) / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        raise RuntimeError(
            f"OpenMontage install at {cfg['repo_dir']} has no .venv — run "
            f"`make setup` there first."
        )
    import shutil as _shutil
    if not _shutil.which("ffmpeg"):
        warnings.append("ffmpeg not on PATH — compose/edit stages will fail.")

    wd = Path(workdir).expanduser()
    _clone_or_update(cfg, wd)
    marker, w = _read_state_marker(wd)
    warnings += w
    _check_trackable(wd, project, marker["binaries"])
    warnings += _verify_source_pin(wd, marker)

    lease = acquire_lease(cfg, project, stage)
    try:
        media_paths: dict[str, str] = {}
        if marker["binaries"] == "media_root":
            media_paths = _wire_media_symlinks(cfg, wd, project)
        stage_entry, director = _load_stage_entry(wd, pipeline, stage)
    except Exception:
        release_lease(cfg, project)
        raise

    paths = _pipeline_paths(wd, project)
    ckpt_dir = Path(paths["checkpoint_dir"])
    prior = sorted(str(p) for p in ckpt_dir.glob("checkpoint_*.json"))
    creative = sorted(str(p) for p in Path(paths["creative_dir"]).glob("*.json"))

    return {
        "status": "ok",
        "workdir": str(wd),
        "binaries_mode": marker["binaries"],
        "python": str(venv_python),
        "remotion_composer": str(Path(cfg["repo_dir"]) / "remotion-composer"),
        "paths": paths,
        "stage": stage_entry,
        "director_skill": str(director),
        "prior_checkpoints": prior,
        "creative_files": creative,
        "media_paths": media_paths,
        "lease": lease,
        "warnings": warnings,
    }


def stage_commit(
    project: str,
    stage: str,
    *,
    workdir: str = DEFAULT_WORKDIR,
    explicit_config: str = "",
) -> dict[str, Any]:
    """Validate, commit (data-only), push, release the lease, and emit the
    handoff receipt. The pushed commit is the durable receipt; the returned
    JSON is what the agent posts to the room (pointers, never bytes)."""
    _validate_project(project)
    cfg = load_config(explicit_config)
    wd = Path(workdir).expanduser()
    if not (wd / ".git").is_dir():
        raise RuntimeError(f"No state clone at {wd} — run stage-begin first.")

    ckpt_path = wd / "projects" / project / "pipeline" / f"checkpoint_{stage}.json"
    ckpt = FileUtil.read(ckpt_path, "json")
    if not isinstance(ckpt, dict):
        raise RuntimeError(
            f"Checkpoint {ckpt_path} missing or unparseable — call "
            f"write_checkpoint before stage-commit."
        )

    # Data-only commit scope: nothing outside projects/<p>/ may be dirty.
    # -uall expands untracked dirs so paths compare against the scope prefix.
    porcelain = _git(wd, "status", "--porcelain", "-uall")
    scope = f"projects/{project}/"
    outside = [
        line for line in porcelain.splitlines()
        if line[3:].split(" -> ")[0] and not line[3:].split(" -> ")[0].startswith(scope)
    ]
    if outside:
        raise RuntimeError(
            f"Refusing to commit: changes outside {scope}: "
            f"{[l.strip() for l in outside[:10]]}. Source paths are frozen — "
            f"revert them (stage-abort resets the tree) or report to the "
            f"coordinator."
        )

    _git(wd, "add", scope)
    staged = [p for p in _git(wd, "diff", "--cached", "--name-only").splitlines() if p]
    if not staged:
        raise RuntimeError("Nothing to commit — did the stage produce no changes?")

    status = str(ckpt.get("status", "completed"))
    artifacts = sorted((ckpt.get("artifacts") or {}).keys()) if isinstance(
        ckpt.get("artifacts"), dict
    ) else []
    cost_log = FileUtil.read(Path(_pipeline_paths(wd, project)["cost_log"]), "json")
    cost = ""
    if isinstance(cost_log, dict):
        cost = f"{cost_log.get('total_spent_usd', '?')}/{cost_log.get('budget_usd', '?')} USD"

    message = (
        f"data({project}/{stage}): checkpoint {status}\n\n"
        f"artifacts: {', '.join(artifacts) or 'none'}\n"
        f"checkpoint: projects/{project}/pipeline/checkpoint_{stage}.json\n"
        + (f"cost_snapshot: {cost}\n" if cost else "")
    )
    _git(wd, "commit", "-m", message)

    lease_before = FileUtil.read(_lease_path(cfg, project), "json") or {}
    try:
        _git(wd, "push", "origin", cfg["state_branch"])
    except RuntimeError as exc:
        # Keep the lease: state is torn and nothing else should proceed.
        raise RuntimeError(
            f"Push rejected — another writer raced you (single-writer "
            f"violated upstream). Do NOT pull-merge-retry; report to the "
            f"coordinator. ({exc})"
        ) from exc

    commit_sha = _git(wd, "rev-parse", "HEAD").strip()
    release_lease(cfg, project)

    return {
        "status": "ok",
        "receipt": {
            "project": project,
            "stage": stage,
            "checkpoint_status": status,
            "checkpoint": f"projects/{project}/pipeline/checkpoint_{stage}.json",
            "commit": commit_sha,
            "artifacts": artifacts,
            "cost_snapshot": cost or None,
            "media": _media_manifest(
                cfg, project, float(lease_before.get("acquired_at") or 0)
            ),
        },
    }


def stage_abort(
    project: str,
    *,
    workdir: str = DEFAULT_WORKDIR,
    explicit_config: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """Coordinator-directed recovery: release the lease and hard-reset the
    clone's project tree to the pushed state."""
    _validate_project(project)
    cfg = load_config(explicit_config)
    released = release_lease(cfg, project, force=force)
    wd = Path(workdir).expanduser()
    reset = False
    if (wd / ".git").is_dir():
        _git(wd, "fetch", "origin")
        _git(wd, "reset", "--hard", f"origin/{cfg['state_branch']}")
        _git(wd, "clean", "-fd", "--", f"projects/{project}/")
        reset = True
    return {"status": "ok", "released_lease": released, "tree_reset": reset}
