# SPDX-License-Identifier: MIT
"""
clawmeets/api/host_protocol.py

The wire contract of the host socket (``/ws/host/{host_id}``) — the frames the
server and a user's computer exchange, and the complete list of things the
server is allowed to ask that computer to do.

Part of the API layer (Layer 0) alongside ``control.py``, and deliberately
NOT part of it. Two reasons:

1. ``ControlEnvelope.payload`` is a non-discriminated Union: a payload parsed
   from a dict resolves to the first member whose shape fits. Every inbound
   frame on the agent and browser sockets is a HEARTBEAT, so that hazard has
   been latent. The host socket carries real inbound payloads in both
   directions, so it validates against explicit models instead of feeding new
   members into that union.
2. The other end is a DIFFERENT distribution. ``clawmeets-daemon`` installs
   without ``clawmeets`` and without pydantic, so it builds these frames as
   plain dicts. Keeping the contract in one small stdlib-shaped module makes
   the mirrored copy in ``clawmeets_daemon/protocol.py`` reviewable, and
   ``tests/test_host_protocol_parity.py`` fails if the two drift.

## The allowlist is the feature

:data:`HOST_ACTIONS` is the complete set of things that can ever happen on a
user's machine at the server's request. It is checked TWICE — once here, before
the server will send a frame, and again on the machine, before it will act —
and that redundancy is deliberate rather than defensive clutter: the machine's
refusal is the fence that holds even if the server is wrong, and it is the
refusal a user can be told about.

``delete`` is absent and must stay absent. Deleting an agent destroys
credentials, memory and sandbox state, and is a user-only action in the browser
(``DELETE /agents/{id}`` accepts a user JWT and nothing else). A host command
that could delete would route around that decision.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

# --- frame types -----------------------------------------------------------

# Machine -> server.
HOST_HELLO = "host_hello"          # first frame; carries the token (auth)
HOST_HEARTBEAT = "host_heartbeat"  # keep-alive
HOST_STATE = "host_state"          # full agent snapshot (+ optional command result)

# Server -> machine.
HOST_COMMAND = "host_command"      # do one allowlisted thing
HOST_ACCEPTED = "host_accepted"    # auth succeeded; here is your host id

# --- the allowlist ---------------------------------------------------------

HostAction = Literal["start", "stop", "restart", "status", "update"]

HOST_ACTIONS: tuple[str, ...] = ("start", "stop", "restart", "status", "update")

# Actions that address ONE agent and are meaningless without a name. `status`
# and `update` are whole-machine.
HOST_AGENT_ACTIONS: frozenset[str] = frozenset({"start", "stop", "restart"})

# What the user is told the actions are, in their words. One source, used
# by the pairing dialog, the computer's page and the system skill, so the
# promise on the consent screen and the promise on the page are the same
# sentence rather than two that have to be kept in step by hand.
HOST_ACTION_LABELS: dict[str, str] = {
    "start": "Start one of your agents",
    "stop": "Stop one of your agents",
    "restart": "Restart one of your agents",
    "status": "Report which of them are running",
    "update": "Update the ClawMeets software on it (clawmeets and its connection software)",
}

# The refusals, also in the user's words, and also load-bearing product copy:
# the claim "this is the complete list" is only checkable next to what is
# excluded.
HOST_NEVER_LABELS: tuple[str, ...] = (
    "Read or change your agents' environment variables",
    "Install or change anything else",
    "Delete an agent — only you can, here in the browser",
    "Reach any other computer or account",
)

# The rest of the truth about this connection, next to the fixed list. The
# allowlist bounds what the SERVER can ask for; it is not a bound on what runs
# here, because the agents it starts act as the user.
HOST_AGENTS_NOTE = (
    "The agents it runs act as you and can run commands on this computer."
)
TERMINAL_ON_LABEL = (
    "Terminal: on. You can open a full shell on this computer, as you, from "
    "your Computer page. Turn it off on this machine with "
    "`clawmeets computer terminal disable`."
)
TERMINAL_OFF_LABEL = (
    "Terminal: off. Turn it on on this machine with "
    "`clawmeets computer terminal enable`."
)

# --- the terminal channel --------------------------------------------------
#
# NOT an action, and deliberately outside HOST_ACTIONS: the allowlist above is
# the fixed set of things the server can ask this computer to do, and the
# terminal is a separate, unconstrained shell the user opens from their own
# Computer page. It is on by default and the user turns it off ON THE MACHINE
# (`clawmeets computer terminal disable`); no frame can change that switch.
#
# It adds no reach the connection did not already have: every agent this
# computer runs executes commands as the user with permission prompts off, and
# "start an agent" is on the allowlist. The terminal gives the user that same
# access directly.

# Server -> machine.
TERM_OPEN = "term_open"        # {session_id, cols, rows}
TERM_INPUT = "term_input"      # {session_id, data_b64}
TERM_RESIZE = "term_resize"    # {session_id, cols, rows}
TERM_ACK = "term_ack"          # {session_id, bytes}  flow-control credit
TERM_CLOSE = "term_close"      # {session_id}
# Machine -> server.
TERM_OPENED = "term_opened"    # {session_id, ok, detail}
TERM_OUTPUT = "term_output"    # {session_id, data_b64}
TERM_EXIT = "term_exit"        # {session_id, code, reason}

TERMINAL_FRAMES: tuple[str, ...] = (
    TERM_OPEN, TERM_INPUT, TERM_RESIZE, TERM_ACK, TERM_CLOSE,
    TERM_OPENED, TERM_OUTPUT, TERM_EXIT,
)

TERM_MAX_SESSIONS = 3
# One input frame. A paste larger than this is chunked by the browser.
TERM_MAX_INPUT_BYTES = 16384
TERM_IDLE_SECONDS = 15 * 60
TERM_MAX_SECONDS = 8 * 3600
# The machine stops reading the shell's output once this much is sent and not
# yet acknowledged by the browser, so a runaway `yes` blocks in the kernel
# instead of flooding the relay and freezing the tab.
TERM_UNACKED_LIMIT = 256 * 1024
TERM_MAX_COLS = 1000
TERM_MAX_ROWS = 500

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def validate_session_id(value: object) -> str:
    """The session id, or raise ``ValueError``. It keys dicts on both ends."""
    if not isinstance(value, str) or not _SESSION_ID_RE.match(value):
        raise ValueError("invalid terminal session id")
    return value


def validate_size(cols: object, rows: object) -> tuple[int, int]:
    """``(cols, rows)`` clamped to a sane window, or raise ``ValueError``."""
    try:
        c, r = int(cols), int(rows)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError("terminal size must be two integers")
    return max(1, min(c, TERM_MAX_COLS)), max(1, min(r, TERM_MAX_ROWS))


# Close codes, matching the agent socket's vocabulary so a daemon and a runner
# react to the same number the same way.
CLOSE_BAD_TOKEN = 4001   # token rejected -> stop retrying with this credential
CLOSE_NO_HOST = 4004     # host record gone or revoked -> self-destruct, do not loop


class HostCommandRejected(ValueError):
    """An action outside :data:`HOST_ACTIONS`, or missing its agent name."""


def validate_host_action(action: str, agent: Optional[str]) -> tuple[str, Optional[str]]:
    """``(action, agent)`` normalized, or raise :class:`HostCommandRejected`.

    The single gate both ends call. Refuses three things: an action not on the
    allowlist, a per-agent action with no agent named, and an agent name
    carrying anything but the characters an agent short name can contain —
    the last because the name reaches a subprocess argument list, and "it is
    only ever passed as a list, never a shell string" is a property of today's
    call site rather than of the name.
    """
    cleaned = (action or "").strip().lower()
    if cleaned not in HOST_ACTIONS:
        raise HostCommandRejected(
            f"{action!r} is not one of the allowed actions "
            f"({', '.join(HOST_ACTIONS)})"
        )
    name = (agent or "").strip() or None
    if cleaned in HOST_AGENT_ACTIONS:
        if not name:
            raise HostCommandRejected(f"{cleaned!r} needs the name of one agent")
        if not all(c.isalnum() or c in "-_" for c in name):
            raise HostCommandRejected(f"{name!r} is not a valid agent name")
    else:
        name = None
    return cleaned, name


# --- frame payloads (server-side validation) -------------------------------


class HostAgentReport(BaseModel):
    """One agent as the machine sees it. Mirror of ``models.host.HostAgent``.

    Separate from the stored model on purpose: this is what an untrusted
    machine sent us, and the stored shape is what we chose to keep. They are
    identical today and may not always be.
    """

    short_name: str
    name: str = ""
    state: str = "stopped"
    pid: Optional[int] = None


class HostModelCLIReport(BaseModel):
    """One model CLI as the machine found it. Mirror of ``doctor.ModelCLIState``.

    This is the one fact in the zero-state checklist that no amount of
    server-side reasoning can produce: whether a ``claude`` / ``codex`` / ``agy``
    binary exists on someone's laptop and is signed in. The machine is the only
    witness, so it reports and we store what it said.

    ``None`` for ``model_clis`` on a frame means "this machine did not tell us",
    which is deliberately different from an empty list ("it looked and found
    nothing") — an un-upgraded computer, or one whose runner is missing, must
    leave the checklist row unconfirmed rather than turn it red.
    """

    id: str
    label: str = ""
    binary: str = ""
    present: bool = False
    logged_in: bool = False
    version: str = ""


class HostHelloFrame(BaseModel):
    """The first frame on a host socket. Auth plus the machine's description."""

    type: Literal["host_hello"]
    token: str = ""
    daemon_version: Optional[str] = None
    hostname: Optional[str] = None
    platform: Optional[str] = None
    os_version: Optional[str] = None
    agents: list[HostAgentReport] = Field(default_factory=list)
    model_clis: Optional[list[HostModelCLIReport]] = None
    # When the machine ran the check that produced ``model_clis`` — not when it
    # sent the frame. A machine re-sends its last answer on every frame and only
    # re-checks every few minutes, so the two differ.
    model_clis_checked_at: Optional[str] = None
    # Whether the terminal is switched on at this machine. None = the daemon
    # predates the terminal and cannot open one at all.
    terminal_enabled: Optional[bool] = None
    # The installed ``clawmeets`` runner's version. None = not reported (an
    # older machine, or one where the runner cannot be found).
    runner_version: Optional[str] = None


class HostCommandOutcome(BaseModel):
    """What happened when the machine ran one command."""

    command_id: str = ""
    action: str = ""
    agent: Optional[str] = None
    ok: bool = True
    detail: str = ""


class HostStateFrame(BaseModel):
    """A full agent snapshot, optionally reporting the command that caused it."""

    type: Literal["host_state"]
    agents: list[HostAgentReport] = Field(default_factory=list)
    daemon_version: Optional[str] = None
    result: Optional[HostCommandOutcome] = None
    model_clis: Optional[list[HostModelCLIReport]] = None
    model_clis_checked_at: Optional[str] = None
    terminal_enabled: Optional[bool] = None
    runner_version: Optional[str] = None


def command_frame(command_id: str, action: str, agent: Optional[str] = None) -> dict:
    """Build the one frame shape the server ever sends a machine.

    Goes through :func:`validate_host_action` so an unrepresentable command
    cannot be constructed, let alone sent.
    """
    cleaned, name = validate_host_action(action, agent)
    return {
        "type": HOST_COMMAND,
        "command_id": command_id,
        "action": cleaned,
        "agent": name,
    }
