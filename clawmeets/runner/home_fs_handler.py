# SPDX-License-Identifier: MIT
"""
clawmeets/runner/home_fs_handler.py

The runner's end of the home-folder relay: turns one ``fs_request`` frame (and,
for a write, the ``fs_chunk`` frames that follow it — ``api/fs_protocol.py``)
into the reply frames — chunks of a read's content, then one ``fs_reply`` — by
running the op through ``HomeFs`` on this runner's OWN home folder.

Two steps, because chunks must be taken in the order they arrive:

1. :meth:`HomeFsRequestHandler.on_frame` — synchronous, called from the socket's
   receive loop for every fs frame. It validates a request, collects a write's
   chunks, and returns an :class:`FsJob` once there is something to answer.
2. :meth:`HomeFsRequestHandler.answer` — runs a job in its own task and yields
   the frames to send.

Rules this module holds, whatever the frames say:

- **The home is ours.** The folder is the ``AGENT_DIR`` this runner was started
  with. An agent id, a root, or any other key in the request is never read.
- **The caller kind is the server's.** The server authenticated and authorized
  the request and says who is asking; ``HomeFs`` enforces paths and tiers for
  that caller.
- **Off the event loop.** ``HomeFs.run`` runs the op in a worker thread, so a
  large read or a recursive delete never stalls heartbeats or changelog pushes.
- **Never raises.** Every failure becomes an error reply (or, for a frame too
  broken to name its request, no reply at all — the server times it out). An
  exception here must not reach the socket loop.
- **Bounded.** A frame over ``FS_MAX_FRAME_CHARS`` is refused unparsed, file
  content is capped at ``FS_MAX_CONTENT_BYTES`` both ways, at most
  ``FS_MAX_CONCURRENT`` requests (collecting chunks or running) are held, and a
  listing that would not fit one frame is cut short and marked ``truncated``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import AsyncIterator, Optional, Union

from pydantic import ValidationError

from clawmeets.api.fs_protocol import (
    BUSY,
    FS_CHUNK,
    FS_MAX_CONCURRENT,
    FS_MAX_CONTENT_BYTES,
    FS_MAX_ENTRIES,
    FS_MAX_FRAME_CHARS,
    FS_REQUEST,
    INVALID_REQUEST,
    REQUEST_TOO_LARGE,
    TOO_LARGE,
    ChunkAssembler,
    FsRequestFrame,
    build_chunks,
    encode_chunk,
    encode_listing_reply,
    encode_reply_error,
    encode_reply_ok,
    peek_frame,
)
from clawmeets.runner.home_fs import (
    CallerKind,
    FsEntry,
    FsFile,
    FsListing,
    FsOp,
    HomeFs,
    HomeFsError,
)

logger = logging.getLogger("clawmeets")


def is_fs_frame(raw: Union[str, bytes]) -> bool:
    """True if a socket frame is one the handler takes — an ``fs_request`` or
    an ``fs_chunk`` (decided on its prefix)."""
    peeked = peek_frame(raw)
    return peeked is not None and peeked[0] in (FS_REQUEST, FS_CHUNK)


def _entry(e: FsEntry) -> dict:
    d = asdict(e)
    d["tier"] = e.tier.value
    return d


def result_to_wire(op: FsOp, result) -> dict:
    """``HomeFs`` result -> the ``result`` object of a success reply (a read's
    content goes separately, as chunks)."""
    if op is FsOp.LIST:
        assert isinstance(result, FsListing)
        return {
            "kind": "listing",
            "path": result.path,
            "tier": result.tier.value,
            "writable": result.writable,
            "truncated": result.truncated,
            "entries": [_entry(e) for e in result.entries],
        }
    if op is FsOp.READ:
        assert isinstance(result, FsFile)
        return {
            "kind": "file",
            "path": result.path,
            "tier": result.tier.value,
            "writable": result.writable,
            "size": result.size,
            "mtime": result.mtime,
        }
    if op in (FsOp.WRITE, FsOp.MKDIR):
        assert isinstance(result, FsEntry)
        return {"kind": "entry", "entry": _entry(result)}
    return {"kind": "deleted"}


@dataclass
class FsJob:
    """Something to answer: a request ready to run, or a refusal to send."""

    request_id: str
    frame: Optional[FsRequestFrame] = None
    content: Optional[bytes] = None
    error: Optional[tuple[str, str]] = None  # (code, message)


@dataclass
class _Upload:
    frame: FsRequestFrame
    chunks: ChunkAssembler


class HomeFsRequestHandler:
    """Answers ``fs_request`` frames for one runner's home folder."""

    def __init__(self, home: Path, *, max_concurrent: int = FS_MAX_CONCURRENT) -> None:
        self._home = Path(home)
        self._fs: Optional[HomeFs] = None
        self._max_concurrent = max_concurrent
        self._in_flight = 0
        # Writes still collecting their chunks, by request id.
        self._uploads: dict[str, _Upload] = {}

    def _home_fs(self) -> HomeFs:
        # Built on first use and kept: HomeFs pins the home folder's identity
        # at construction, and a long-lived pin is what lets it notice the
        # folder being swapped out later.
        if self._fs is None:
            self._fs = HomeFs(
                self._home,
                max_entries=FS_MAX_ENTRIES,
                max_read_bytes=FS_MAX_CONTENT_BYTES,
                max_write_bytes=FS_MAX_CONTENT_BYTES,
            )
        return self._fs

    def socket_closed(self) -> None:
        """The socket the uploads came in on is gone, and the server has
        failed them; drop their partial content."""
        self._uploads.clear()

    def on_frame(self, raw: Union[str, bytes]) -> Optional[FsJob]:
        """Take one fs frame, in arrival order. Returns the job to answer, or
        None (a chunk taken, or a frame that names no request). Never raises."""
        request_id: Optional[str] = None
        try:
            peeked = peek_frame(raw)
            if peeked is None or peeked[0] not in (FS_REQUEST, FS_CHUNK):
                return None
            frame_type, request_id = peeked
            if frame_type == FS_CHUNK:
                return self._on_chunk(request_id, raw)
            if len(raw) > FS_MAX_FRAME_CHARS:
                return FsJob(request_id, error=(
                    REQUEST_TOO_LARGE, f"request frame is larger than {FS_MAX_FRAME_CHARS} bytes",
                ))
            try:
                frame = FsRequestFrame.model_validate(json.loads(raw))
            except (ValueError, ValidationError):
                return FsJob(request_id, error=(INVALID_REQUEST, "malformed fs request"))
            if frame.request_id != request_id or request_id in self._uploads:
                return FsJob(request_id, error=(INVALID_REQUEST, "malformed fs request"))
            if frame.op == "write" and frame.size > FS_MAX_CONTENT_BYTES:
                return FsJob(request_id, error=(TOO_LARGE, "content is too large"))
            if self._in_flight + len(self._uploads) >= self._max_concurrent:
                return FsJob(request_id, error=(BUSY, "too many file operations in flight"))
            if frame.op == "write" and frame.size:
                self._uploads[request_id] = _Upload(frame, ChunkAssembler(request_id, frame.size))
                return None
            return FsJob(request_id, frame=frame, content=b"" if frame.op == "write" else None)
        except Exception as exc:  # never reaches the socket loop
            logger.error(f"fs frame failed: {exc}", exc_info=True)
            if request_id is None:
                return None
            self._uploads.pop(request_id, None)
            return FsJob(request_id, error=("io_error", "the runner could not complete the operation"))

    def _on_chunk(self, request_id: str, raw: Union[str, bytes]) -> Optional[FsJob]:
        upload = self._uploads.get(request_id)
        if upload is None:
            return None  # its request was refused, or never ours
        try:
            if len(raw) > FS_MAX_FRAME_CHARS:
                raise ValueError("chunk is too large")
            upload.chunks.add(raw)
        except ValueError as exc:
            del self._uploads[request_id]
            code = TOO_LARGE if "too large" in str(exc) else INVALID_REQUEST
            return FsJob(request_id, error=(code, str(exc)))
        if upload.chunks.size < upload.frame.size:
            return None
        del self._uploads[request_id]
        return FsJob(request_id, frame=upload.frame, content=upload.chunks.content())

    async def answer(self, job: FsJob) -> AsyncIterator[str]:
        """The frames answering ``job``: a read's content chunks, then one
        ``fs_reply``. Never raises."""
        rid = job.request_id
        if job.error is not None:
            yield encode_reply_error(rid, *job.error)
            return
        frame = job.frame
        assert frame is not None
        try:
            self._in_flight += 1
            try:
                op = FsOp(frame.op)
                result = await self._home_fs().run(op, frame.path, CallerKind(frame.caller), job.content)
            finally:
                self._in_flight -= 1
        except HomeFsError as exc:
            yield encode_reply_error(rid, exc.code, exc.message)
            return
        except Exception as exc:  # never reaches the socket loop
            logger.error(f"fs request failed: {exc}", exc_info=True)
            yield encode_reply_error(rid, "io_error", "the runner could not complete the operation")
            return
        wire = result_to_wire(op, result)
        if op is FsOp.LIST:
            yield encode_listing_reply(rid, wire)
            return
        if op is FsOp.READ:
            for chunk in build_chunks(rid, result.content):
                yield encode_chunk(chunk)
        yield encode_reply_ok(rid, wire)
