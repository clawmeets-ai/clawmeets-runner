# SPDX-License-Identifier: MIT
"""
clawmeets/sync/git_sandbox.py
Git-based sandbox management via changelog subscriber.

This module is part of Layer 0 (pure - no domain model dependencies).
It manages git clone/init, branch creation, and commit/push operations
for code-aware sandbox projects.

Branching model:
  - project/{project_name}: accumulates merged work from all milestones
  - chatroom/{project_name}/{room}: per-chatroom work branch

Git operations by role:
  - Workers: commit sandbox changes on sync_complete, push if git_url set
  - Coordinator (git_url set): fetch + merge chatroom branch on batch_complete
  - Coordinator (no git_url): no git operations (files are artifacts in project_dir)
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from .changelog import (
    BatchCompletePayload,
    ChangelogEntry,
    ChangelogEntryType,
    RoomCreatedPayload,
)
from .subscriber import ChangelogSubscriber

logger = logging.getLogger(__name__)


def _run_git(cwd: Path, *args: str, check: bool = True) -> str:
    """Run a git command synchronously and return stdout.

    Args:
        cwd: Working directory for git command
        *args: Git command arguments (e.g. "clone", "--branch", "main")
        check: If True, raise on non-zero exit code

    Returns:
        stdout as string
    """
    import subprocess

    cmd = ["git"] + list(args)
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if check and result.returncode != 0:
        logger.error(
            f"git command failed: {' '.join(cmd)}\n"
            f"  cwd: {cwd}\n"
            f"  stderr: {result.stderr.strip()}"
        )
        raise RuntimeError(f"git {args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


async def _run_git_async(cwd: Path, *args: str, check: bool = True) -> str:
    """Run a git command asynchronously."""
    return await asyncio.to_thread(_run_git, cwd, *args, check=check)


class GitSandboxSubscriber(ChangelogSubscriber):
    """Manages git clone/branch lifecycle for code-aware sandbox projects.

    Registered between ModelContextChangelogSubscriber (files to disk)
    and ParticipantNotifier (agent callbacks) in insertion order.

    Handles:
    - PROJECT_CREATED: git clone/init, create project branch, setup .gitignore
    - ROOM_CREATED: create chatroom branch (workers fetch from remote first)
    - BATCH_COMPLETE: coordinator fetches + merges chatroom branch (git_url only)
    - PROJECT_COMPLETED: coordinator tags the project branch (git_url only)
    - on_sync_complete: worker commits + pushes sandbox changes
    """

    def __init__(
        self,
        sandbox_dir: Path,
        coordinator_id: str,
        participant_id: str,
        project_dir: Path,
    ) -> None:
        """Initialize the git sandbox subscriber.

        The git_url and ignored_folder are read lazily from the
        PROJECT_CREATED changelog payload, making this subscriber
        per-project rather than globally configured.

        Args:
            sandbox_dir: Agent's sandbox directory (will contain the clone/repo)
            coordinator_id: The project coordinator's participant ID
            participant_id: This participant's ID
            project_dir: The synced project_dir where chatroom files appear
        """
        self._sandbox_dir = sandbox_dir
        self._git_url: Optional[str] = None  # Set from PROJECT_CREATED payload
        self._ignored_folder = ".bus-files"  # Default; overridden by payload
        self._is_coordinator = coordinator_id == participant_id
        self._project_dir = project_dir
        self._initialized = False

    async def on_entry(
        self,
        entry: ChangelogEntry,
        project_id: str,
        project_name: str,
    ) -> None:
        """Process a changelog entry for git operations."""
        if entry.entry_type == ChangelogEntryType.PROJECT_CREATED:
            await self._handle_project_created(entry, project_name)
            return

        # Skip all git operations if no git_url configured for this project
        if not self._git_url:
            return

        if entry.entry_type == ChangelogEntryType.ROOM_CREATED:
            await self._handle_room_created(entry, project_name)

        elif entry.entry_type == ChangelogEntryType.BATCH_COMPLETE:
            await self._handle_batch_complete(entry, project_name)

        elif entry.entry_type == ChangelogEntryType.PROJECT_COMPLETED:
            await self._handle_project_completed(entry, project_name)

    async def _on_llm_complete(self, sandbox_dir: Path, **kwargs) -> None:
        """Commit and push sandbox changes after successful LLM invocation.

        Only acts for workers — coordinators merge via _handle_batch_complete.
        No-ops if this project has no git_url configured.
        """
        if self._is_coordinator or not self._git_url:
            return
        await self._commit_and_push(sandbox_dir)

    async def _commit_and_push(self, sandbox_dir: Path) -> None:
        """Commit and push any uncommitted changes in a sandbox directory."""
        if not sandbox_dir.exists() or not (sandbox_dir / ".git").exists():
            return

        status = await _run_git_async(sandbox_dir, "status", "--porcelain")
        if not status:
            return

        await _run_git_async(sandbox_dir, "add", "-A")
        current_branch = await _run_git_async(
            sandbox_dir, "rev-parse", "--abbrev-ref", "HEAD",
        )
        await _run_git_async(
            sandbox_dir, "commit", "-m",
            f"Worker changes on {current_branch}",
        )
        logger.info(f"Committed worker changes on {current_branch}")

        if self._git_url:
            await _run_git_async(
                sandbox_dir, "push", "origin", current_branch,
                check=False,
            )

    async def _handle_project_created(
        self,
        entry: ChangelogEntry,
        project_name: str,
    ) -> None:
        """Initialize git repo on project creation.

        Reads git_url from the PROJECT_CREATED payload. If no git_url,
        marks as initialized and all future operations become no-ops.

        Coordinator: clone/init, create project branch, push.
        Worker: clone/init, fetch project branch.
        """
        if self._initialized:
            return  # Idempotent

        # Read git config from payload
        payload = entry.payload
        raw_url = getattr(payload, "git_url", "") or ""
        self._git_url = raw_url if raw_url else None
        self._ignored_folder = getattr(payload, "git_ignored_folder", self._ignored_folder)

        if not self._git_url:
            self._initialized = True  # No git needed — all future on_entry calls skip
            return

        sandbox = self._sandbox_dir

        if self._git_url:
            if not sandbox.exists() or not (sandbox / ".git").exists():
                sandbox.parent.mkdir(parents=True, exist_ok=True)
                await _run_git_async(
                    sandbox.parent,
                    "clone", self._git_url, sandbox.name,
                )
                logger.info(f"Cloned {self._git_url} into {sandbox}")
        else:
            sandbox.mkdir(parents=True, exist_ok=True)
            if not (sandbox / ".git").exists():
                await _run_git_async(sandbox, "init")
                # Create initial commit so branches work
                gitignore_path = sandbox / ".gitignore"
                if not gitignore_path.exists():
                    gitignore_path.write_text("")
                await _run_git_async(sandbox, "add", ".gitignore")
                await _run_git_async(
                    sandbox, "commit", "-m", "Initial commit",
                    "--allow-empty",
                )
                logger.info(f"Initialized standalone git repo at {sandbox}")

        # Setup .gitignore for the ignored folder
        await self._setup_gitignore(sandbox)

        # Create project branch
        project_branch = f"project/{project_name}"
        try:
            await _run_git_async(sandbox, "checkout", "-b", project_branch)
            logger.info(f"Created project branch: {project_branch}")
        except RuntimeError:
            # Branch may already exist (idempotent replay)
            await _run_git_async(sandbox, "checkout", project_branch, check=False)

        # Coordinator pushes project branch to remote
        if self._is_coordinator and self._git_url:
            await _run_git_async(
                sandbox, "push", "-u", "origin", project_branch,
                check=False,  # May fail if remote already has it
            )

        self._initialized = True

    async def _handle_room_created(
        self,
        entry: ChangelogEntry,
        project_name: str,
    ) -> None:
        """Create chatroom branch from project branch.

        Workers fetch from remote first (if git_url) to get latest
        coordinator pushes from prior milestones.
        """
        payload: RoomCreatedPayload = entry.payload  # type: ignore[assignment]
        room_name = payload.chatroom_name

        # Skip system rooms
        if room_name in ("shared-context", "user-communication"):
            return

        sandbox = self._sandbox_dir
        if not (sandbox / ".git").exists():
            return  # Not initialized yet

        project_branch = f"project/{project_name}"
        chatroom_branch = f"chatroom/{project_name}/{room_name}"

        # Workers: fetch to get latest code from coordinator's pushes
        if not self._is_coordinator and self._git_url:
            await _run_git_async(sandbox, "fetch", "origin", check=False)
            # Update project branch from remote
            await _run_git_async(
                sandbox, "checkout", project_branch, check=False,
            )
            await _run_git_async(
                sandbox, "pull", "origin", project_branch, check=False,
            )

        # Create chatroom branch from project branch
        try:
            await _run_git_async(
                sandbox, "checkout", "-b", chatroom_branch, project_branch,
            )
            logger.info(f"Created chatroom branch: {chatroom_branch}")
        except RuntimeError:
            # Branch may already exist (idempotent replay)
            await _run_git_async(sandbox, "checkout", chatroom_branch, check=False)

    async def _handle_batch_complete(
        self,
        entry: ChangelogEntry,
        project_name: str,
    ) -> None:
        """Coordinator: merge chatroom branch into project branch (git_url only).

        When git_url is not set, the coordinator doesn't need to do anything —
        file updates are artifacts in project_dir, synced via changelog.
        """
        if not self._is_coordinator:
            return
        if not self._git_url:
            return

        payload: BatchCompletePayload = entry.payload  # type: ignore[assignment]
        room_name = payload.chatroom_name
        sandbox = self._sandbox_dir

        if not (sandbox / ".git").exists():
            return

        project_branch = f"project/{project_name}"
        chatroom_branch = f"chatroom/{project_name}/{room_name}"

        # Stash local changes
        await _run_git_async(sandbox, "stash", check=False)

        # Fetch worker's pushed commits
        await _run_git_async(sandbox, "fetch", "origin", check=False)

        # Checkout project branch and merge chatroom branch from remote
        await _run_git_async(sandbox, "checkout", project_branch, check=False)
        await _run_git_async(
            sandbox, "merge", f"origin/{chatroom_branch}",
            "--no-edit", check=False,
        )
        logger.info(f"Merged {chatroom_branch} into {project_branch}")

        # Push updated project branch
        await _run_git_async(
            sandbox, "push", "origin", project_branch,
            check=False,
        )

        # Pop stash
        await _run_git_async(sandbox, "stash", "pop", check=False)

    async def _handle_project_completed(
        self,
        entry: ChangelogEntry,
        project_name: str,
    ) -> None:
        """Coordinator: tag the project branch as complete (git_url only)."""
        if not self._is_coordinator:
            return
        if not self._git_url:
            return

        sandbox = self._sandbox_dir
        if not (sandbox / ".git").exists():
            return

        tag_name = f"project/{project_name}-complete"
        await _run_git_async(
            sandbox, "tag", tag_name,
            check=False,  # May already exist
        )
        logger.info(f"Tagged project branch: {tag_name}")

        await _run_git_async(
            sandbox, "push", "origin", tag_name,
            check=False,
        )

    async def _setup_gitignore(self, sandbox: Path) -> None:
        """Add git_ignored_folder and debug files to .gitignore if not already present."""
        gitignore = sandbox / ".gitignore"
        existing = ""
        if gitignore.exists():
            existing = gitignore.read_text(encoding="utf-8")

        entries_to_add = []
        if self._ignored_folder not in existing:
            entries_to_add.append(f"{self._ignored_folder}/")
        if ".agent-prompt.txt" not in existing:
            entries_to_add.append(".agent-prompt.txt")

        if entries_to_add:
            with gitignore.open("a", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                for entry in entries_to_add:
                    f.write(f"{entry}\n")
            logger.debug(f"Added {entries_to_add} to .gitignore")
