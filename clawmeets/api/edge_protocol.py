# SPDX-License-Identifier: MIT
"""
clawmeets/api/edge_protocol.py

The link between core (``clawmeets server start``) and the websocket edge
(``clawmeets server ws-edge``): one websocket, dialled by core to the edge's
loopback ``/link`` endpoint, carrying one JSON object per text message.

The edge holds every public socket (runners, browsers, computers, terminals)
and knows nothing about what flows through them: core sends ready-made frame
text and the connection ids to deliver it to, and the edge reports sockets
opening, closing and talking. Every rule about WHO receives WHAT stays in core.
That split is what lets core restart without dropping a single socket.

Stdlib only, and imported by both sides; ``clawmeets/ws_edge/`` may import
nothing else from clawmeets besides ``api/host_protocol`` and
``api/fs_protocol``.

Ops, core -> edge:

  hello        {"op","v","boot_id"}                 first frame; edge answers snapshot
  deliver      {"op","pids":[str],"text"}           every socket of each participant
  deliver_conn {"op","conn","text"}                 exactly one socket
  close        {"op","pid","code","reason"}         every socket of a participant
  close_conn   {"op","conn","code","reason"}        exactly one socket
  host_send    {"op","host_id","text"}              every socket of a computer
  host_close   {"op","host_id","code","reason"}
  auth_result  {"op","req","ok","code","reason","info"}

Ops, edge -> core:

  snapshot     {"op","edge_boot_id","conns":[{conn, **ConnInfo}],
                "hosts":{host_id: {"owner_id","hello","last_state","conns"}},
                "heartbeats":{agent_id: iso}}
  auth_req     {"op","req","kind":"agent"|"host","id","conn","ip","hello"}
  connected    {"op","conn","info":{...},"in_flight_turns":[...]?}
  disconnected {"op","conn"}
  runner_raw   {"op","conn","raw"}                  fs_reply / fs_chunk, untouched
  heartbeats   {"op","at":{agent_id: iso}}          coalesced
  host_event   {"op","host_id","event":"connected"|"frame"|"disconnected",
                "owner_id","frame","remaining"?}

ConnInfo (each ``snapshot`` conn, and ``connected``'s ``info``) has exactly the
keys kind, pid, ip, clawmeets_version, owner_id, agent_name, capabilities (a
list), connected_since (UTC iso) and seq; a key that does not apply is null.

A computer can briefly hold two sockets across a reconnect, so core is told how
many remain: a ``host_event`` "connected" / "disconnected" carries
``remaining``, the sockets that host_id still holds after the event (>= 1 on
"connected"), and each snapshot ``hosts`` entry carries ``conns``, its open
socket count. ``hello`` never carries the token.

A ``conn`` is an opaque string the edge assigns; core only stores it and hands
it back. Frame bodies (``text``, ``raw``, host frames) can carry secrets — model
API keys, env values, OAuth codes — so neither side logs them.
"""
from __future__ import annotations

import json
from typing import Any, Optional

PROTOCOL_VERSION = 1

# Largest link message either end accepts. A client socket accepts 16 MiB
# frames, and wrapping one in ``deliver`` JSON-escapes it, which can grow it.
LINK_MAX_FRAME_BYTES = 64 * 1024 * 1024

# Path of the link endpoint on the edge's loopback port.
LINK_PATH = "/link"

# Close codes the edge uses on a public socket that core did not choose.
CLOSE_TRY_AGAIN = 1013       # core is not linked: a new runner/computer retries
CLOSE_SERVICE_RESTART = 1012  # a client too old for RESYNC: reconnect to catch up

# How often the edge forwards coalesced runner heartbeats.
HEARTBEAT_FLUSH_S = 5.0
# How long the edge waits for core to answer an auth_req before 1013.
AUTH_TIMEOUT_S = 10.0

# core -> edge
HELLO = "hello"
DELIVER = "deliver"
DELIVER_CONN = "deliver_conn"
CLOSE = "close"
CLOSE_CONN = "close_conn"
HOST_SEND = "host_send"
HOST_CLOSE = "host_close"
AUTH_RESULT = "auth_result"
# edge -> core
SNAPSHOT = "snapshot"
AUTH_REQ = "auth_req"
CONNECTED = "connected"
DISCONNECTED = "disconnected"
RUNNER_RAW = "runner_raw"
HEARTBEATS = "heartbeats"
HOST_EVENT = "host_event"

HOST_CONNECTED = "connected"
HOST_FRAME = "frame"
HOST_DISCONNECTED = "disconnected"


def encode(op: str, **fields: Any) -> str:
    """One link message. Compact; non-ASCII kept as is (it is UTF-8 on the wire)."""
    return json.dumps({"op": op, **fields}, separators=(",", ":"), ensure_ascii=False)


def decode(raw: str | bytes) -> Optional[dict]:
    """The message as a dict with a string ``op``, or None if it is not one."""
    try:
        msg = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(msg, dict) or not isinstance(msg.get("op"), str):
        return None
    return msg


def frame_text(data: Any) -> str:
    """Serialize a frame for a public socket exactly as Starlette's
    ``WebSocket.send_json`` does, so a client cannot tell edge delivery from
    in-process delivery."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
