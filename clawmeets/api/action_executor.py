# SPDX-License-Identifier: MIT
"""
clawmeets/api/action_executor.py
HTTP action execution for Claude Code actions.

This module executes actions parsed from Claude ```actions blocks via HTTP.
"""
from __future__ import annotations

import errno
import logging
import os
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Optional

import httpx

from .client import CURRENT_TURN_GRANT

if TYPE_CHECKING:
    from .actions import ActionBlock
    from .client import ClawMeetsClient

logger = logging.getLogger(__name__)


class UploadRefused(Exception):
    """An update_file path that resolves outside the turn's write roots."""


def _real(path: Path) -> Path:
    return Path(os.path.realpath(path))


def _open_beneath(root: Path, rel: Path, pin: Optional[tuple[int, int]] = None) -> int:
    """Open ``root/rel`` read-only, refusing a symlink at ANY component.

    Walks one component at a time with ``O_NOFOLLOW`` relative to the previous
    directory's descriptor, so a component swapped for a symlink between the
    caller's ``realpath`` check and this open is refused instead of followed.
    ``root`` itself is opened ``O_NOFOLLOW`` too and, given its ``pin``
    (``(st_dev, st_ino)`` taken at turn start), must still be that directory.
    Windows has no ``dir_fd`` (and runs no scoped turns): a plain open there.
    """
    if os.open not in os.supports_dir_fd:
        return os.open(root / rel, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise UploadRefused("the sandbox is a symlink") from exc
        raise
    try:
        if pin is not None:
            st = os.fstat(fd)
            if (st.st_dev, st.st_ino) != tuple(pin):
                raise UploadRefused("the sandbox was replaced during the turn")
        for part in rel.parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        # O_NONBLOCK so a FIFO planted in the sandbox cannot hang the runner.
        return os.open(rel.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise UploadRefused("path is a symlink") from exc
        raise
    finally:
        os.close(fd)


def resolve_upload_path(
    write_roots: tuple[Path, ...], room: str, file_path: str,
    pins: tuple[tuple[int, int], ...] = (),
) -> int:
    """Open the file an ``update_file`` action names, confined to the turn's
    write roots, and return an open descriptor to read it from.

    The runner reads this file outside any sandbox, so without this check an
    absolute path, a ``..`` segment or a symlink in the sandbox would make it
    upload ``credential.json``, ``env.json`` or a skill config into the room.
    Refuses absolute paths, drive letters and ``..``; joins against the first
    root (the project sandbox), then the room-files fallback under it; takes
    the real path and requires it under one of the roots; then opens that real
    path from the root one component at a time with ``O_NOFOLLOW``
    (``_open_beneath``), so a directory swapped for a symlink after the check
    (by a process the turn left behind) is refused; refuses anything but a
    regular file. The roots are real paths already (``TurnScope``) and are
    not resolved again: a root swapped for a symlink would resolve to its
    target. ``pins`` (aligned with ``write_roots``) are their
    ``(st_dev, st_ino)`` at turn start.

    Every action that takes a path must go through this function.
    Raises ``FileNotFoundError`` when no candidate exists, ``UploadRefused``
    for anything outside the roots.
    """
    if not file_path or "\x00" in file_path:
        raise UploadRefused("empty path")
    posix, win = PurePosixPath(file_path), PureWindowsPath(file_path)
    if posix.is_absolute() or win.is_absolute() or win.drive or file_path.startswith(("/", "\\")):
        raise UploadRefused("absolute paths are not allowed")
    if ".." in posix.parts or ".." in win.parts:
        raise UploadRefused("'..' is not allowed")
    if not write_roots:
        raise UploadRefused("no write roots")
    roots = tuple(Path(r) for r in write_roots)
    pin_of = dict(zip(roots, pins))
    sandbox = roots[0]
    candidates = [sandbox / file_path, sandbox / "chatrooms" / room / "files" / file_path]
    for candidate in candidates:
        if not os.path.lexists(candidate):
            continue
        real = _real(candidate)
        root = next((r for r in roots if r in real.parents), None)
        if root is None:
            raise UploadRefused("path resolves outside the turn's sandbox")
        fd = _open_beneath(root, real.relative_to(root), pin_of.get(root))
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise UploadRefused("not a regular file")
        except BaseException:
            os.close(fd)
            raise
        return fd
    raise FileNotFoundError(file_path)


def _read_fd(fd: int) -> bytes:
    with os.fdopen(fd, "rb") as fh:
        return fh.read()


class ActionBlockExecutor:
    """
    Parses Claude action blocks and executes them via HTTP.

    Combines parsing and HTTP emission in one class. Created and owned by
    Agent/Assistant. Executes actions directly instead of returning intermediate
    response objects.

    Role-based action restrictions are enforced at the schema level:
    - Workers use WORKER_ACTION_SCHEMA (reply, update_file only)
    - Coordinators use COORDINATOR_ACTION_SCHEMA (all actions)
    """

    def __init__(self, client: "ClawMeetsClient") -> None:
        """
        Initialize the action block executor.

        Args:
            client: ClawMeetsClient for HTTP operations
        """
        self._client = client

    async def process(
        self,
        action_block: "ActionBlock",
        project_id: str,
        sandbox_dir: Path,
        *,
        write_roots: Optional[tuple[Path, ...]] = None,
        write_pins: tuple[tuple[int, int], ...] = (),
        grant_id: Optional[str] = None,
    ) -> set[str]:
        """
        Process an action block and execute actions via HTTP.

        Server-First Sync Architecture:
        Claude writes files to sandbox_dir (isolated working area). This method
        reads file content from sandbox_dir and sends to the server via HTTP.

        4xx failures on individual actions are caught: a one-line failure note
        is posted into `user-communication` so the coordinator sees the failure
        on its next dispatch and can re-plan, and processing continues with the
        next action in the block. 5xx is re-raised — server brokenness should
        be loud. A 5xx on an ``update_file`` moved ahead of its reply waits
        until that reply is posted (with a line saying the file was not
        shared), so the move never loses the reply.

        Args:
            action_block: The extracted action block from Claude output
            project_id: The project ID
            sandbox_dir: Sandbox directory where Claude wrote files (isolated from synced data)

            write_roots: Folders an ``update_file`` may read from (default: the
                sandbox). ``update_file`` is confined to them for every turn
                kind (``resolve_upload_path``).
            write_pins: Each write root's ``(st_dev, st_ino)`` pinned at turn
                start (``TurnScope.write_pins``).
            grant_id: The scoped turn's grant; every request this block makes
                carries it as ``X-Clawmeets-Turn-Grant`` (the isolation gate
                requires it for writes in an isolated project).

        Returns:
            Set of chatroom names that received reply actions.
        """
        token = CURRENT_TURN_GRANT.set(grant_id)
        try:
            return await self._process(action_block, project_id, sandbox_dir, write_roots, write_pins)
        finally:
            CURRENT_TURN_GRANT.reset(token)

    async def _process(
        self,
        action_block: "ActionBlock",
        project_id: str,
        sandbox_dir: Path,
        write_roots: Optional[tuple[Path, ...]],
        write_pins: tuple[tuple[int, int], ...] = (),
    ) -> set[str]:
        replied_chatrooms: set[str] = set()
        source_version = action_block.source_version
        # A 5xx on an update_file moved ahead of its reply, per room: the
        # reply is still posted (as it was before the move), then it raises.
        failed_uploads: dict[str, list[tuple[str, Exception]]] = {}

        for action, moved in _files_before_replies(action_block.actions):
            action_type = action["type"]

            try:
                if action_type == "reply":
                    room_name = action["room"]
                    await self._client.post_message(
                        project_id=project_id,
                        chatroom_name=room_name,
                        content=action["content"] + _upload_failed_note(failed_uploads.get(room_name)),
                        source_version=source_version,
                    )
                    logger.debug(f"Emitted reply to {room_name}")
                    replied_chatrooms.add(room_name)

                elif action_type == "create_room":
                    name = action["name"]
                    await self._client.create_chatroom(
                        project_id=project_id,
                        name=name,
                        participant_names=action["invite"],
                        init_message=action["init_message"],
                        source_version=source_version,
                    )
                    logger.debug(f"Created chatroom {name}")

                elif action_type == "invite_agent":
                    room_name = action["room"]
                    await self._client.invite_to_chatroom(
                        project_id=project_id,
                        chatroom_name=room_name,
                        participant_names=action["invite"],
                        message=action["message"],
                        source_version=source_version,
                    )
                    logger.debug(f"Invited {action['invite']} to {room_name}")

                elif action_type == "update_file":
                    file_path = action["file_path"]
                    room_name = action["room"]
                    # Resolve the sandbox root so this read side matches the write
                    # side, which resolves write_root (api_provider `_build_file_tools`).
                    # On macOS the temp dir is a /var → /private/var symlink; without
                    # resolving here the two sides can name the same file differently.
                    sandbox_root = sandbox_dir.resolve()
                    roots = write_roots if write_roots else (sandbox_root,)
                    try:
                        fd = resolve_upload_path(tuple(roots), room_name, file_path, write_pins)
                    except FileNotFoundError:
                        logger.warning(
                            "File not found in sandbox: %s (sandbox_root=%s, exists=%s)",
                            file_path, sandbox_root, sandbox_root.exists(),
                        )
                        continue
                    except UploadRefused as refused:
                        summary = (
                            f"Action `update_file` (file {file_path!r} in chatroom "
                            f"{room_name!r}) refused: {refused}. Only files inside this "
                            f"turn's sandbox can be shared."
                        )
                        logger.warning(summary)
                        await self._post_failure_note(project_id, summary, source_version)
                        continue
                    await self._client.upload_file(
                        project_id=project_id,
                        chatroom_name=room_name,
                        filename=file_path,
                        content=_read_fd(fd),
                        source_version=source_version,
                    )
                    logger.debug(f"Updated file {file_path} in {room_name}")

                elif action_type == "project_completed":
                    await self._client.complete_project(
                        project_id=project_id,
                        source_version=source_version,
                    )
                    logger.debug(f"Marked project {project_id} as completed")

            except httpx.HTTPStatusError as e:
                if e.response.status_code // 100 != 4:
                    if moved:
                        # Raised once its reply is posted, so the move never
                        # costs the reply.
                        failed_uploads.setdefault(action["room"], []).append((action["file_path"], e))
                        continue
                    # 5xx (or anything non-4xx): server brokenness, not a coordinator
                    # mistake — let the dispatch loop log it loudly.
                    raise
                summary = _format_failure_summary(action, e)
                logger.warning(summary)
                await self._post_failure_note(project_id, summary, source_version)
            except (httpx.HTTPError, OSError) as e:
                # A timeout, a dropped connection or an unreadable file: no
                # status to judge, so a moved upload is handled like a 5xx.
                if not moved:
                    raise
                failed_uploads.setdefault(action["room"], []).append((action["file_path"], e))
                continue
            if action_type == "reply" and action["room"] in failed_uploads:
                raise failed_uploads[action["room"]][0][1]

        return replied_chatrooms

    async def _post_failure_note(
        self,
        project_id: str,
        summary: str,
        source_version: int,
    ) -> None:
        """Post a coordinator-authored failure note to user-communication.

        Silent-fails on its own error so a missing/unreachable user-communication
        room can't mask the original action failure (which is already logged).
        """
        try:
            await self._client.post_message(
                project_id=project_id,
                chatroom_name="user-communication",
                content=summary,
                source_version=source_version,
            )
        except Exception:
            logger.exception("Failed to post action-failure note to user-communication")


def _upload_failed_note(failed: Optional[list[tuple[str, Exception]]]) -> str:
    """The line a reply carries when a file shared with it failed to upload."""
    if not failed:
        return ""
    names = ", ".join(path for path, _ in failed)
    error = failed[0][1]
    if isinstance(error, httpx.HTTPStatusError):
        return f"\n\n(Not shared: {names}. The upload failed with a server error ({error.response.status_code}).)"
    return f"\n\n(Not shared: {names}. The upload failed ({type(error).__name__}).)"


def _files_before_replies(actions: list[dict]) -> list[tuple[dict, bool]]:
    """The block's actions, each with whether it was moved, with each
    ``update_file`` moved up to just before the first ``reply`` into its
    room that precedes it. A file the agent
    shares with its reply then exists when the reply is posted, so whatever
    the reply wakes or relays (a retained agent's room bridge sends only the
    files uploaded with a message) sees it. A file never moves ahead of
    what precedes that reply, so a room the block creates before replying in
    it is still created first."""
    out: list[tuple[dict, bool]] = []
    for action in actions:
        if action.get("type") == "update_file":
            at = next((i for i, (a, _) in enumerate(out)
                       if a.get("type") == "reply" and a.get("room") == action.get("room")), None)
            if at is not None:
                out.insert(at, (action, True))
                continue
        out.append((action, False))
    return out


def _format_failure_summary(action: dict, exc: httpx.HTTPStatusError) -> str:
    """Build a one-line, coordinator-readable summary of a failed action."""
    status = exc.response.status_code
    reason = exc.response.reason_phrase or ""
    try:
        detail = exc.response.json().get("detail", exc.response.text)
    except Exception:
        detail = exc.response.text

    action_type = action.get("type", "<unknown>")
    if action_type == "create_room":
        invite = action.get("invite") or []
        ctx = f"chatroom {action.get('name')!r} (inviting {invite!r})"
    elif action_type == "invite_agent":
        invite = action.get("invite") or []
        ctx = f"chatroom {action.get('room')!r} (inviting {invite!r})"
    elif action_type == "reply":
        ctx = f"chatroom {action.get('room')!r}"
    elif action_type == "update_file":
        ctx = f"file {action.get('file_path')!r} in chatroom {action.get('room')!r}"
    elif action_type == "project_completed":
        ctx = "project_completed"
    else:
        ctx = action_type

    return (
        f"Action `{action_type}` ({ctx}) failed: "
        f"{status} {reason} — {detail}"
    )
