# SPDX-License-Identifier: MIT
"""
clawmeets/api/fs_protocol.py

The wire contract of the agent home-folder browser: the request / reply frames
the server and an agent's runner exchange on the runner socket
(``/ws/{agent_id}``) so the server can list, read, write, make folders in and
delete from the agent's home folder on whichever machine runs it.

Part of the API layer (Layer 0) and deliberately NOT part of ``control.py``,
for the same reasons ``host_protocol.py`` is not:

1. ``ControlEnvelope`` is documented as never carrying file content, and its
   ``payload`` is a non-discriminated Union that re-parses a dict as the first
   member whose shape fits. These frames carry file bytes in both directions,
   so they get their own ``type`` values and explicit models instead.
2. A bad frame must never cost the socket. Both ends route these frames to
   their own handler BEFORE the ``ControlEnvelope`` parse, inside their own
   error boundary: a malformed reply fails the one request it names, and a
   malformed request is answered with an error reply.

## Frames

Server -> runner, :data:`FS_REQUEST`::

    {"type": "fs_request", "request_id": str, "op": "list"|"read"|"write"|"mkdir"|"delete",
     "path": str, "caller": "owner"|"self"|"assistant", "size": int (write only)}

Runner -> server, :data:`FS_REPLY`::

    {"type": "fs_reply", "request_id": str, "ok": true,  "result": <one of the results below>}
    {"type": "fs_reply", "request_id": str, "ok": false, "error": {"code": str, "message": str}}

File content never rides in either of those. It travels as :data:`FS_CHUNK`
frames, in either direction::

    {"type": "fs_chunk", "request_id": str, "seq": int, "data_b64": str}

- **write:** the request names the content's ``size``, then the server sends
  ``ceil(size / FS_CHUNK_BYTES)`` chunks, ``seq`` 0, 1, 2, ... The runner runs
  the write once it holds exactly ``size`` bytes.
- **read:** the runner sends the chunks first, then the ``fs_reply`` whose
  ``file`` result carries the ``size`` they add up to.

``type`` and ``request_id`` are always the first two keys (see
:func:`peek_frame`): it lets either end name the request an oversized frame
belongs to without parsing it.

The request carries no agent id and no root. The runner always operates on its
own ``AGENT_DIR``; anything else in the frame is ignored. ``caller`` is the
server's authorization decision (who is asking, relative to the agent); the
runner trusts it and enforces paths and tiers itself (``runner/home_fs.py``).

## Capability gate (instead of a version compare)

A runner that understands these frames lists :data:`FS_CAPABILITY` in a
``capabilities`` array on its socket's auth frame, next to ``token`` and
``clawmeets_version``. The server sends a request only to a socket that
advertised it and otherwise answers ``runner_too_old`` at once. A capability
rather than ``clawmeets_version >= X`` because the version is ``None`` for
editable / source installs, pre-release ordering is easy to get wrong, and the
flag is sent by exactly the code that handles the frame — it cannot claim
support it does not have.

## Size caps, and why content is chunked

Both ends run on WebSocket stacks whose inbound limit is :data:`FS_WS_MAX_SIZE`
(the server passes it to uvicorn as ``ws_max_size``; the runner raises its
``websockets`` client from the 1 MiB default to it). A frame over that limit
makes the receiving stack CLOSE the socket, so every fs frame stays under
:data:`FS_MAX_FRAME_CHARS` (1 MiB) on both ends.

That frame cap is set by the keepalive, not by memory. Both stacks ping every
20 s and drop a socket whose pong is 20 s late, and a pong queues BEHIND any
frame already being written. One frame per file (8 MiB of content is ~10.7 MiB
of base64) would take over 20 s to drain on links slower than ~4-5 Mbit/s and
cost the socket mid-transfer. A 512 KiB chunk (~0.7 MiB on the wire) drains in
under 20 s down to ~0.3 Mbit/s, and pongs interleave between chunks. A listing
that would not fit one frame is cut short and marked ``truncated``.

:data:`FS_MAX_CONTENT_BYTES` caps a whole file, independent of the frame size.
The JSON is ASCII-only (``ensure_ascii``), so characters == bytes on the wire.
"""
from __future__ import annotations

import base64
import binascii
import json
import re
from typing import Annotated, Iterator, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

# --- frame types -----------------------------------------------------------

FS_REQUEST = "fs_request"   # server -> runner
FS_REPLY = "fs_reply"       # runner -> server
FS_CHUNK = "fs_chunk"       # file content, either way

# Advertised in the runner's auth frame: ``{"capabilities": [FS_CAPABILITY]}``.
FS_CAPABILITY = "home_fs.v1"

FS_OPS: tuple[str, ...] = ("list", "read", "write", "mkdir", "delete")
# Ops whose effect may have happened even if no reply arrived. A timeout on
# one of these is ``result_unknown``, never "failed".
FS_MUTATING_OPS: frozenset[str] = frozenset({"write", "mkdir", "delete"})
# Values of ``runner.home_fs.CallerKind``.
FS_CALLERS: tuple[str, ...] = ("owner", "self", "assistant")

# --- caps --------------------------------------------------------------------

# Inbound message limit both ends run with (uvicorn's ``ws_max_size``; the
# runner passes this as its ``websockets`` client ``max_size``).
FS_WS_MAX_SIZE = 16 * 1024 * 1024
# Largest fs frame either end sends or accepts (see "Size caps" above).
FS_MAX_FRAME_CHARS = 1024 * 1024
# File content per fs_chunk frame (base64: ~683 KiB).
FS_CHUNK_BYTES = 512 * 1024
# Largest file that can be read, downloaded or written through the relay.
FS_MAX_CONTENT_BYTES = 8 * 1024 * 1024
# Entries in one listing; past this the runner sets ``truncated``.
FS_MAX_ENTRIES = 2000
FS_MAX_PATH_CHARS = 4096
# Requests in flight per agent. More are refused with ``busy``.
FS_MAX_CONCURRENT = 8
# How long the server waits for a reply. Read / write move up to 8 MiB over
# whatever link the runner has, so they get longer.
FS_TIMEOUT_SECONDS = 30.0
FS_TRANSFER_TIMEOUT_SECONDS = 120.0

# --- error codes the relay itself produces --------------------------------
# (The runner's own refusals carry ``runner.home_fs.HomeFsError.code``.)

AGENT_OFFLINE = "agent_offline"          # no live socket, or it dropped mid-request
                                         # (a write / mkdir / delete then also
                                         # carries ``result_unknown``)
RUNNER_TOO_OLD = "runner_too_old"        # socket did not advertise FS_CAPABILITY
TIMEOUT = "timeout"                      # list / read: no reply in time
RESULT_UNKNOWN = "result_unknown"        # write / mkdir / delete: no reply in time
BUSY = "busy"                            # FS_MAX_CONCURRENT in flight
TOO_LARGE = "too_large"                  # content over FS_MAX_CONTENT_BYTES
REQUEST_TOO_LARGE = "request_too_large"  # runner got a frame over FS_MAX_FRAME_CHARS
REPLY_TOO_LARGE = "reply_too_large"      # a reply frame over FS_MAX_FRAME_CHARS
BAD_REPLY = "bad_reply"                  # reply did not parse / did not fit the op
INVALID_REQUEST = "invalid_request"      # request frame did not parse

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
_CODE_RE = r"^[a-z][a-z0-9_]{0,39}$"
# The fixed frame prefix every encoder here produces.
_PEEK_RE = re.compile(
    r'^\s*\{\s*"type"\s*:\s*"(fs_request|fs_reply|fs_chunk)"\s*,\s*"request_id"\s*:\s*"([A-Za-z0-9_-]{16,64})"'
)
_PEEK_WINDOW = 256

MAX_MESSAGE_CHARS = 500


def validate_request_id(value: object) -> str:
    """The request id, or raise ``ValueError``. It keys the pending table."""
    if not isinstance(value, str) or not _REQUEST_ID_RE.match(value):
        raise ValueError("invalid fs request id")
    return value


def peek_frame(raw: Union[str, bytes]) -> Optional[tuple[str, str]]:
    """``(frame_type, request_id)`` from a frame's first bytes, or None.

    Cheap and parse-free: used to route a frame before the ``ControlEnvelope``
    parse, and to name the request an oversized frame belongs to so it can be
    failed (server) or answered (runner) without decoding megabytes of JSON.
    """
    head = raw[:_PEEK_WINDOW]
    if isinstance(head, bytes):
        head = head.decode("utf-8", "replace")
    m = _PEEK_RE.match(head)
    return (m.group(1), m.group(2)) if m else None


def _dumps(frame: dict) -> str:
    # ASCII-only so len(text) is the byte count the socket limit applies to.
    return json.dumps(frame, ensure_ascii=True, separators=(",", ":"))


def b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64decode(text: object, *, max_bytes: int = FS_MAX_CONTENT_BYTES) -> bytes:
    """Strict base64 -> bytes, or raise ``ValueError`` (incl. over ``max_bytes``)."""
    if not isinstance(text, str):
        raise ValueError("content must be a base64 string")
    if len(text) > 4 * ((max_bytes + 2) // 3):
        raise ValueError("content is too large")
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("content is not valid base64") from exc
    if len(data) > max_bytes:
        raise ValueError("content is too large")
    return data


# --- request ---------------------------------------------------------------


def build_request(
    request_id: str, op: str, path: str, caller: str, size: Optional[int] = None
) -> dict:
    """The one request shape the server sends; a write's content follows as
    :func:`build_chunks`. Key order is part of the contract (``peek_frame``)."""
    validate_request_id(request_id)
    if op not in FS_OPS:
        raise ValueError(f"unknown fs op {op!r}")
    if caller not in FS_CALLERS:
        raise ValueError(f"unknown fs caller {caller!r}")
    frame: dict = {
        "type": FS_REQUEST,
        "request_id": request_id,
        "op": op,
        "path": path,
        "caller": caller,
    }
    if op == "write":
        frame["size"] = size or 0
    return frame


def encode_request(frame: dict) -> str:
    return _dumps(frame)


class FsRequestFrame(BaseModel):
    """A request as the runner validates it. Extra keys (an agent id, a root,
    anything) are ignored — the runner never reads them."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["fs_request"]
    request_id: str = Field(pattern=_REQUEST_ID_RE.pattern)
    op: Literal["list", "read", "write", "mkdir", "delete"]
    path: str = Field(max_length=FS_MAX_PATH_CHARS)
    caller: Literal["owner", "self", "assistant"]
    size: int = Field(default=0, ge=0)


# --- content chunks ----------------------------------------------------------


def build_chunks(request_id: str, data: bytes) -> Iterator[dict]:
    """``data`` as consecutive :data:`FS_CHUNK` frames (none for empty data).
    Lazy, so a sender holds one encoded chunk at a time."""
    for seq, start in enumerate(range(0, len(data), FS_CHUNK_BYTES)):
        yield {
            "type": FS_CHUNK,
            "request_id": request_id,
            "seq": seq,
            "data_b64": b64encode(data[start:start + FS_CHUNK_BYTES]),
        }


class FsChunkFrame(BaseModel):
    type: Literal["fs_chunk"]
    request_id: str = Field(pattern=_REQUEST_ID_RE.pattern)
    seq: int = Field(ge=0)
    data_b64: str


class ChunkAssembler:
    """Puts one request's content back together from its chunk frames.

    Chunks must arrive in order (``seq`` 0, 1, 2, ...) — a WebSocket delivers
    them in the order they were sent — and may not add up to more than
    ``limit`` bytes. Anything else raises ``ValueError``; the caller fails the
    request.
    """

    def __init__(self, request_id: str, limit: int) -> None:
        self.request_id = request_id
        self.limit = limit
        self._parts: list[bytes] = []
        self.size = 0

    def add(self, raw: Union[str, bytes]) -> None:
        try:
            frame = FsChunkFrame.model_validate(json.loads(raw))
        except UnicodeDecodeError as exc:
            raise ValueError("chunk is not JSON") from exc
        if frame.request_id != self.request_id:
            raise ValueError("chunk names another request")
        if frame.seq != len(self._parts):
            raise ValueError("chunk out of order")
        data = b64decode(frame.data_b64, max_bytes=FS_CHUNK_BYTES)
        if not data:
            raise ValueError("empty chunk")
        if self.size + len(data) > self.limit:
            raise ValueError("content is too large")
        self._parts.append(data)
        self.size += len(data)

    def content(self) -> bytes:
        return b"".join(self._parts)


# --- reply -----------------------------------------------------------------

Tier = Literal["secret", "system", "personal_skills", "open"]


class FsEntryWire(BaseModel):
    name: str = Field(min_length=1, max_length=1024)
    type: Literal["file", "dir", "symlink", "other"]
    size: int = Field(ge=0)
    mtime: float
    tier: Tier
    writable: bool
    readable: bool


class FsListingResult(BaseModel):
    kind: Literal["listing"]
    path: str = Field(max_length=FS_MAX_PATH_CHARS)
    tier: Tier
    writable: bool
    truncated: bool = False
    entries: list[FsEntryWire] = Field(default_factory=list, max_length=FS_MAX_ENTRIES)


class FsFileResult(BaseModel):
    """A read's metadata. The bytes arrive as chunks before it; the relay
    attaches them as :attr:`content` once they add up to ``size``."""

    kind: Literal["file"]
    path: str = Field(max_length=FS_MAX_PATH_CHARS)
    tier: Tier
    writable: bool
    size: int = Field(ge=0, le=FS_MAX_CONTENT_BYTES)
    mtime: float
    _content: bytes = PrivateAttr(default=b"")

    @property
    def content(self) -> bytes:
        return self._content


class FsEntryResult(BaseModel):
    kind: Literal["entry"]
    entry: FsEntryWire


class FsDeletedResult(BaseModel):
    kind: Literal["deleted"]


FsResult = Annotated[
    Union[FsListingResult, FsFileResult, FsEntryResult, FsDeletedResult],
    Field(discriminator="kind"),
]

# The result kind each op must answer with.
RESULT_KIND_FOR_OP: dict[str, str] = {
    "list": "listing",
    "read": "file",
    "write": "entry",
    "mkdir": "entry",
    "delete": "deleted",
}


class FsReplyError(BaseModel):
    code: str = Field(pattern=_CODE_RE)
    message: str = Field(default="", max_length=MAX_MESSAGE_CHARS)


class FsReplyFrame(BaseModel):
    type: Literal["fs_reply"]
    request_id: str = Field(pattern=_REQUEST_ID_RE.pattern)
    ok: bool
    result: Optional[FsResult] = None
    error: Optional[FsReplyError] = None


def parse_reply(
    data: object, op: str, request_id: str, content: Optional[ChunkAssembler] = None,
) -> Union[FsListingResult, FsFileResult, FsEntryResult, FsDeletedResult, FsReplyError]:
    """The validated result (or the runner's refusal) for a reply to ``op``.

    ``request_id`` is the id the frame was routed by (its prefix); the parsed
    frame must agree. ``content`` is what the request's chunks assembled to.

    Raises ``ValueError`` on anything malformed: wrong shape, another request's
    id, ``ok`` without a result (or the wrong kind of result for ``op``), a
    file whose chunks do not add up to ``size``, or chunks for a non-read.
    """
    frame = FsReplyFrame.model_validate(data)  # pydantic.ValidationError is a ValueError
    if frame.request_id != request_id:
        raise ValueError("reply names another request")
    if not frame.ok:
        if frame.error is None:
            raise ValueError("error reply without an error")
        return frame.error
    result = frame.result
    if result is None or result.kind != RESULT_KIND_FOR_OP.get(op):
        raise ValueError(f"reply to {op!r} carried the wrong result")
    if isinstance(result, FsFileResult):
        received = content.size if content is not None else 0
        if received != result.size:
            raise ValueError("file size does not match its content")
        result._content = content.content() if content is not None else b""
    elif content is not None and content.size:
        raise ValueError(f"content sent with a reply to {op!r}")
    return result


def encode_reply_ok(request_id: str, result: dict) -> str:
    """A success reply, or a ``reply_too_large`` error reply if it would not
    fit in one frame. Never returns a frame over :data:`FS_MAX_FRAME_CHARS`."""
    text = _dumps({"type": FS_REPLY, "request_id": request_id, "ok": True, "result": result})
    if len(text) > FS_MAX_FRAME_CHARS:
        return encode_reply_error(
            request_id, REPLY_TOO_LARGE,
            f"the answer is larger than {FS_MAX_FRAME_CHARS} bytes",
        )
    return text


def encode_listing_reply(request_id: str, result: dict) -> str:
    """A listing reply cut down to fit one frame: entries are dropped from the
    end, and ``truncated`` set, until it does."""
    entries = result["entries"]
    while True:
        text = _dumps({"type": FS_REPLY, "request_id": request_id, "ok": True, "result": result})
        if len(text) <= FS_MAX_FRAME_CHARS or not entries:
            return text if len(text) <= FS_MAX_FRAME_CHARS else encode_reply_ok(request_id, result)
        entries = entries[: len(entries) * 3 // 4]
        result = {**result, "entries": entries, "truncated": True}


def encode_chunk(frame: dict) -> str:
    return _dumps(frame)


def encode_reply_error(request_id: str, code: str, message: str) -> str:
    if not re.match(_CODE_RE, code or ""):
        code = "io_error"
    return _dumps({
        "type": FS_REPLY,
        "request_id": request_id,
        "ok": False,
        "error": {"code": code, "message": (message or "")[:MAX_MESSAGE_CHARS]},
    })
