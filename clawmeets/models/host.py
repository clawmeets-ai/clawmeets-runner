# SPDX-License-Identifier: MIT
"""
clawmeets/models/host.py

Host registry — the record of ONE user's computer, and the per-host credential
that lets the server ask that computer to start or stop the user's own agents.

This is the first record in the data model that describes a *machine* rather
than a participant, and it exists because "your computer is reachable but none
of your agents are running" was previously indistinguishable from "your
computer is off". Both rendered as silence. An agent's own socket cannot tell
them apart — it is gone in both cases — so something on the machine has to stay
connected when every agent is dead. That something authenticates with a host
token, and this module is where the token and the machine's reported state live.

Storage::

    {data_dir}/hosts/
      pairing/<code>.json          # PairingCode — single-use, 15-minute
      <owner_user_id>/
        <host_id>.json             # HostRecord

One file per host so two machines checking in at the same instant never fight
for a single document, and owner-scoped directories so a list read is one
``iterdir`` of the caller's own folder — the same shape ``brief_tab.py`` uses.

## What is stored and what is derived

``last_seen_at`` is STORED: an offline machine still has to say "seen 12 Aug".
Online-ness is DERIVED from a live websocket in ``WSHub`` and is never written
here, exactly as runner versions are never persisted — a connection that does
not exist cannot be reported as existing. :func:`derive_status` is the one
place the two are combined, so the REST route, the websocket push and the UI
cannot disagree about what "off" means.

## What is NOT stored

There is no command history. The five allowed actions are start / stop /
restart / status / update-self, and the record keeps only the OUTCOME OF THE
MOST RECENT one (:attr:`HostRecord.last_command`) — enough for a button to
report "couldn't start it, here's why", and deliberately not a log. The
computer's page shows live state and the machine's own agents; a durable
per-machine audit trail was considered and dropped as redundant with that.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from clawmeets.utils.file_io import FileUtil

_lock = asyncio.Lock()

HOSTS_DIR = "hosts"
PAIRING_SUBDIR = "pairing"

# How long a pairing code is good for. Short on purpose: the code is typed into
# a terminal on the machine being paired, seconds after the dialog renders it,
# and it grants the right to hold a host credential for that account.
PAIRING_CODE_TTL_SECONDS = 15 * 60

# A machine with no live socket whose last check-in is inside this window is
# "not answering" rather than "off" — it was here moments ago, so the honest
# reading is that it is awake and something between us broke, not that the lid
# is shut. Beyond it, "off" is the better guess and the better advice ("open
# the computer" rather than "run the reconnect command").
HOST_NOT_ANSWERING_WINDOW_SECONDS = 15 * 60

# Pairing-code alphabet: uppercase, no 0/O/1/I/L so a code read off a screen and
# typed into a terminal cannot be mistranscribed.
_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_CODE_GROUP = 4
_CODE_GROUPS = 2

_HOST_ID_RE = re.compile(r"^[0-9a-f]{16}$")

MAX_HOST_NAME_CHARS = 60

HostStatus = Literal["online", "not_answering", "off", "revoked"]
AgentRunState = Literal["running", "crashed", "stopped"]


class HostAgent(BaseModel):
    """One agent as the MACHINE sees it, not as the server remembers it.

    ``state`` comes from the machine scanning its own agents directory and
    checking each recorded process id (``utils/agent_processes.scan_agents``):

    - ``running``  — the pidfile names a live process.
    - ``crashed``  — a pidfile exists and the process is gone. Rendered as
      "Stopped on its own", which is a genuinely different fact from a stop the
      user asked for and is the reason this is three states rather than two.
    - ``stopped``  — no pidfile.
    """

    short_name: str
    name: str
    state: AgentRunState = "stopped"
    pid: Optional[int] = None


class HostModelCLI(BaseModel):
    """One model CLI as the MACHINE found it, not as the server guessed.

    Stored because it is the only route by which the server can honestly answer
    "can this computer actually invoke a model?" — the credential lives in an OS
    keychain or a dotfile on someone's laptop, and nothing server-side can see
    it. ``logged_in`` is the machine's best-effort reading (see
    ``clawmeets/doctor.py`` on why it is a heuristic), carried through verbatim
    rather than re-interpreted here.
    """

    id: str
    label: str = ""
    binary: str = ""
    present: bool = False
    logged_in: bool = False
    version: str = ""


class HostCommandResult(BaseModel):
    """The outcome of the most recent command the server sent this machine.

    A single slot, overwritten each time — not a log. It exists so a failed
    "Start" can say *why* instead of leaving a button that appears to do
    nothing; ``ok=False`` plus ``detail`` is the whole contract.
    """

    # Echoes the id the command route returned, so the page can tell "the
    # command I just sent finished" from "some earlier command finished" and
    # release that button at the right moment. Empty on records written before
    # the field existed.
    command_id: str = ""
    action: str
    agent: Optional[str] = None
    ok: bool = True
    detail: str = ""
    finished_at: str = ""


class HostRecord(BaseModel):
    """One registered computer.

    ``token_hash`` never leaves this process: :meth:`to_wire` drops it, and
    that is the only shape any route returns. The raw token exists exactly once,
    in the response to the pairing call that minted it.
    """

    id: str
    owner_user_id: str
    # Display name. Defaults to a humanized hostname and is renameable inline —
    # the page calls the machine what its owner calls it, never "runner" or
    # "daemon".
    name: str
    hostname: str = ""
    platform: str = ""
    os_version: str = ""
    token_hash: str
    created_at: str
    last_seen_at: str = ""
    revoked_at: Optional[str] = None
    daemon_version: Optional[str] = None
    agents: list[HostAgent] = Field(default_factory=list)
    agents_reported_at: str = ""
    last_command: Optional[HostCommandResult] = None
    # None = this machine has never told us. Distinct from [] ("it looked and
    # found none"), because a computer running an older release cannot report
    # this at all and must not be rendered as having nothing installed.
    model_clis: Optional[list[HostModelCLI]] = None
    model_clis_reported_at: str = ""
    # When the machine last ran the check, by its own clock. Older machines do
    # not say, and fall back to ``model_clis_reported_at``.
    model_clis_checked_at: str = ""

    @property
    def is_revoked(self) -> bool:
        return bool(self.revoked_at)

    def to_wire(self, *, connected: bool) -> dict:
        """The shape every route returns. No ``token_hash``, ever.

        ``connected`` is the caller's live-socket answer; it is a parameter
        rather than a field because nothing here can know it and nothing here
        should persist it.
        """
        data = self.model_dump(exclude={"token_hash"})
        data["connected"] = connected
        data["status"] = derive_status(self, connected=connected)
        data["running_count"] = sum(1 for a in self.agents if a.state == "running")
        data["agent_count"] = len(self.agents)
        # Three-valued on purpose, and the reason it is computed here rather than
        # in the frontend: True = this machine reported a usable model CLI,
        # False = it looked and found none, None = it never said. A checklist row
        # that cannot tell "no" from "don't know" either nags a working user or
        # reassures a broken one.
        data["model_cli_ready"] = self.model_cli_ready
        return data

    @property
    def model_cli_ready(self) -> Optional[bool]:
        """Did this machine report a model CLI that is installed AND signed in?

        Installed-but-signed-out counts as False: the runner will shell the
        binary, the binary will refuse, and the agent will go silent — which is
        indistinguishable, from the browser, from nothing being installed at all.
        That equivalence is the whole reason the machine reports ``logged_in``
        separately instead of just presence.
        """
        if self.model_clis is None:
            return None
        return any(c.present and c.logged_in for c in self.model_clis)


class PairingCode(BaseModel):
    """A single-use, short-lived grant to register ONE machine to one account."""

    code: str
    owner_user_id: str
    created_at: str
    expires_at: str


def derive_status(record: HostRecord, *, connected: bool) -> HostStatus:
    """The one definition of a machine's state, shared by every reader.

    Order matters. ``revoked`` wins over everything: a revoked host keeps its
    row (so the page can explain what happened) but must never read as online
    even if a socket is somehow still draining. Then a live socket is
    ``online``. Without one, a check-in inside
    :data:`HOST_NOT_ANSWERING_WINDOW_SECONDS` is ``not_answering`` (awake, lost
    contact — the remedy is a command on the machine) and anything older is
    ``off`` (the remedy is to open the computer).
    """
    if record.is_revoked:
        return "revoked"
    if connected:
        return "online"
    seen = _parse_iso(record.last_seen_at)
    if seen is None:
        return "off"
    age = (datetime.now(UTC) - seen).total_seconds()
    return "not_answering" if age <= HOST_NOT_ANSWERING_WINDOW_SECONDS else "off"


def default_host_name(hostname: str) -> str:
    """"Chengtaos-MacBook-Pro.local" -> "MacBook Pro".

    Best-effort and deliberately lossy: the user renames it if we guess wrong,
    and a wrong-but-human name beats a correct-but-unreadable one. Falls back
    to the raw hostname, then to "This computer", so the field is never empty.
    """
    raw = (hostname or "").strip()
    if not raw:
        return "This computer"
    base = raw.split(".")[0]
    parts = [p for p in re.split(r"[-_]+", base) if p]
    # Drop a leading possessive owner segment ("Chengtaos", "Alices").
    if len(parts) > 1 and parts[0].lower().endswith("s"):
        parts = parts[1:]
    pretty = " ".join(parts).strip()
    return (pretty or base or "This computer")[:MAX_HOST_NAME_CHARS]


def validate_host_name(name: str) -> str:
    """Normalize a user-supplied machine name. Raises ValueError if unusable."""
    cleaned = " ".join((name or "").split())
    if not cleaned:
        raise ValueError("A computer needs a name")
    if len(cleaned) > MAX_HOST_NAME_CHARS:
        raise ValueError(f"Name must be {MAX_HOST_NAME_CHARS} characters or fewer")
    return cleaned


def validate_host_id(host_id: str) -> str:
    """Guard the path segment before it becomes a filename.

    Host ids are server-minted 16-hex strings, so anything else is either a
    typo or a traversal attempt; either way it is not a host.
    """
    cleaned = (host_id or "").strip().lower()
    if not _HOST_ID_RE.match(cleaned):
        raise ValueError("Not a valid computer id")
    return cleaned


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _root(data_dir: Path) -> Path:
    return Path(data_dir) / HOSTS_DIR


def _owner_dir(data_dir: Path, owner_user_id: str) -> Path:
    return _root(data_dir) / owner_user_id


def _host_path(data_dir: Path, owner_user_id: str, host_id: str) -> Path:
    return _owner_dir(data_dir, owner_user_id) / f"{validate_host_id(host_id)}.json"


def _pairing_path(data_dir: Path, code: str) -> Path:
    return _root(data_dir) / PAIRING_SUBDIR / f"{normalize_pairing_code(code)}.json"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_iso(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def normalize_pairing_code(code: str) -> str:
    """"7qk2 m4rd" / "7QK2-M4RD" -> "7QK2M4RD".

    Grouping and case are presentation. Normalizing before lookup is what lets
    a user retype a code the way they read it without it failing.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]", "", code or "").upper()
    if len(cleaned) != _CODE_GROUP * _CODE_GROUPS:
        raise ValueError("A pairing code is 8 characters")
    return cleaned


def format_pairing_code(code: str) -> str:
    """"7QK2M4RD" -> "7QK2-M4RD" — the only form a user ever sees."""
    cleaned = normalize_pairing_code(code)
    return "-".join(
        cleaned[i:i + _CODE_GROUP] for i in range(0, len(cleaned), _CODE_GROUP)
    )


async def create_pairing_code(data_dir: Path, owner_user_id: str) -> PairingCode:
    """Mint a pairing code for one account.

    Codes are NOT reused and old ones are not invalidated: a user who opens the
    dialog twice gets two live codes, both of which work until they expire or
    one is spent. Invalidating the earlier one would break the common case of
    "I opened this on my phone and I'm typing it on my laptop".
    """
    async with _lock:
        _sweep_expired_codes(data_dir)
        code = "".join(
            secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_GROUP * _CODE_GROUPS)
        )
        now = datetime.now(UTC)
        record = PairingCode(
            code=code,
            owner_user_id=owner_user_id,
            created_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=PAIRING_CODE_TTL_SECONDS)).isoformat(),
        )
        FileUtil.write(_pairing_path(data_dir, code), record.model_dump(), "json")
        return record


def _sweep_expired_codes(data_dir: Path) -> None:
    """Delete codes that can no longer be spent. Best-effort housekeeping.

    Called from the mint path so the directory cannot grow without bound from
    dialogs that were opened and abandoned. Never raises — a failure to tidy
    must not fail the pairing the user is in the middle of.
    """
    pairing_dir = _root(data_dir) / PAIRING_SUBDIR
    if not pairing_dir.is_dir():
        return
    now = datetime.now(UTC)
    for entry in pairing_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".json":
            continue
        data = FileUtil.read(entry, "json")
        expires = _parse_iso((data or {}).get("expires_at"))
        if expires is None or expires < now:
            try:
                entry.unlink()
            except OSError:
                pass


async def consume_pairing_code(data_dir: Path, code: str) -> Optional[str]:
    """Spend a pairing code. Returns the owner's user id, or None.

    Single use: the file is unlinked BEFORE the caller registers the host, so
    two machines racing on one code cannot both win. None covers unknown,
    already-spent and expired with one answer — the caller must not tell a
    stranger which of the three it was.
    """
    async with _lock:
        try:
            path = _pairing_path(data_dir, code)
        except ValueError:
            return None
        data = FileUtil.read(path, "json")
        if not isinstance(data, dict):
            return None
        try:
            record = PairingCode.model_validate(data)
        except Exception:
            return None
        try:
            path.unlink()
        except OSError:
            return None
        expires = _parse_iso(record.expires_at)
        if expires is None or expires < datetime.now(UTC):
            return None
        return record.owner_user_id


# ---------------------------------------------------------------------------
# Hosts
# ---------------------------------------------------------------------------


def _load(path: Path) -> Optional[HostRecord]:
    data = FileUtil.read(path, "json")
    if not isinstance(data, dict):
        return None
    try:
        return HostRecord.model_validate(data)
    except Exception:
        return None


def _save(data_dir: Path, record: HostRecord) -> None:
    FileUtil.write(
        _host_path(data_dir, record.owner_user_id, record.id),
        record.model_dump(),
        "json",
    )


async def register_host(
    data_dir: Path,
    owner_user_id: str,
    *,
    hostname: str = "",
    platform: str = "",
    os_version: str = "",
    daemon_version: Optional[str] = None,
) -> tuple[HostRecord, str]:
    """Create a host record. Returns ``(record, raw_token)``.

    The raw token is returned exactly once and never stored — only its SHA-256
    is, mirroring ``PersistableParticipant.register``. A lost token is
    re-paired, not recovered.

    There is deliberately no "create a host and wait for it to connect" state:
    the record is created BY the pairing call the machine itself makes, so a
    listed-but-never-connected computer cannot exist.
    """
    async with _lock:
        host_id = secrets.token_hex(8)
        token = secrets.token_urlsafe(32)
        record = HostRecord(
            id=host_id,
            owner_user_id=owner_user_id,
            name=default_host_name(hostname),
            hostname=hostname,
            platform=platform,
            os_version=os_version,
            token_hash=hashlib.sha256(token.encode()).hexdigest(),
            created_at=_now(),
            last_seen_at=_now(),
            daemon_version=daemon_version,
        )
        _save(data_dir, record)
        return record, token


def get_host(data_dir: Path, owner_user_id: str, host_id: str) -> Optional[HostRecord]:
    try:
        return _load(_host_path(data_dir, owner_user_id, host_id))
    except ValueError:
        return None


def find_host(data_dir: Path, host_id: str) -> Optional[HostRecord]:
    """Resolve a host by id ALONE, scanning owner directories.

    The websocket endpoint needs this: a connecting machine presents its host
    id and token and nothing else, so there is no owner to scope the read to
    yet. Every other reader knows the owner and must use :func:`get_host`,
    which is one file open instead of a walk.
    """
    try:
        wanted = f"{validate_host_id(host_id)}.json"
    except ValueError:
        return None
    root = _root(data_dir)
    if not root.is_dir():
        return None
    for owner_dir in root.iterdir():
        if not owner_dir.is_dir() or owner_dir.name == PAIRING_SUBDIR:
            continue
        candidate = owner_dir / wanted
        if candidate.is_file():
            return _load(candidate)
    return None


def list_hosts(data_dir: Path, owner_user_id: str) -> list[HostRecord]:
    """Every computer one user has connected, newest first.

    Revoked hosts are INCLUDED — this is the store, not the view. The
    ``/me/computers`` route is what hides them from the user.
    """
    owner_dir = _owner_dir(data_dir, owner_user_id)
    if not owner_dir.is_dir():
        return []
    records: list[HostRecord] = []
    for entry in owner_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".json":
            continue
        record = _load(entry)
        if record is not None:
            records.append(record)
    records.sort(key=lambda r: r.created_at, reverse=True)
    return records


def verify_host_token(data_dir: Path, host_id: str, token: str) -> bool:
    """Timing-safe host-token check. A revoked host verifies as False.

    The revocation check lives HERE rather than at the call site so "Disconnect
    this computer" cannot be defeated by a second door: every path that
    authenticates a machine goes through this function.
    """
    record = find_host(data_dir, host_id)
    if record is None or record.is_revoked:
        return False
    return secrets.compare_digest(
        record.token_hash,
        hashlib.sha256((token or "").encode()).hexdigest(),
    )


async def touch_host(
    data_dir: Path,
    owner_user_id: str,
    host_id: str,
    *,
    daemon_version: Optional[str] = None,
    hostname: Optional[str] = None,
    platform: Optional[str] = None,
    os_version: Optional[str] = None,
) -> Optional[HostRecord]:
    """Record a check-in, and refresh whatever the machine re-reported.

    ``None`` for a field means "unchanged in this check-in" rather than
    "cleared", the same convention ``AgentSettingsChangePayload`` uses, so a
    daemon that reports less than it used to cannot blank out fields the page
    is showing.
    """
    async with _lock:
        record = get_host(data_dir, owner_user_id, host_id)
        if record is None:
            return None
        record.last_seen_at = _now()
        if daemon_version is not None:
            record.daemon_version = daemon_version
        if hostname is not None:
            record.hostname = hostname
        if platform is not None:
            record.platform = platform
        if os_version is not None:
            record.os_version = os_version
        _save(data_dir, record)
        return record


async def record_agent_snapshot(
    data_dir: Path,
    owner_user_id: str,
    host_id: str,
    agents: list[HostAgent],
    *,
    command_result: Optional[HostCommandResult] = None,
) -> Optional[HostRecord]:
    """Replace the machine's reported agent list wholesale.

    A full snapshot, never a delta — the machine re-scans its own directory on
    every report, so merging would keep agents alive in our copy that the
    machine no longer sees. ``last_seen_at`` moves too: a snapshot IS a
    check-in.
    """
    async with _lock:
        record = get_host(data_dir, owner_user_id, host_id)
        if record is None:
            return None
        record.agents = list(agents)
        record.agents_reported_at = _now()
        record.last_seen_at = record.agents_reported_at
        if command_result is not None:
            record.last_command = command_result
        _save(data_dir, record)
        return record


async def record_model_clis(
    data_dir: Path,
    owner_user_id: str,
    host_id: str,
    model_clis: list[HostModelCLI],
    checked_at: Optional[str] = None,
) -> Optional[HostRecord]:
    """Store what the machine found when it looked for model CLIs.

    A full replacement, like the agent snapshot and for the same reason: the
    machine re-probes and re-reports, so merging would keep a CLI alive in our
    copy after the user uninstalled it.

    Separate from :func:`record_agent_snapshot` even though both arrive on the
    same frames, because the two have different reasons to be absent. A frame
    with no ``agents`` means "nothing is running"; a frame with no ``model_clis``
    means "this machine's software is too old to tell you" — so the caller must
    be able to write one without touching the other, and an omitted report must
    never clear a good one.

    Does NOT move ``last_seen_at``: the frames that carry this already stamp it,
    and a second write here would be the only place a check-in could be recorded
    twice for one frame.
    """
    async with _lock:
        record = get_host(data_dir, owner_user_id, host_id)
        if record is None:
            return None
        record.model_clis = list(model_clis)
        record.model_clis_reported_at = _now()
        record.model_clis_checked_at = checked_at or record.model_clis_reported_at
        _save(data_dir, record)
        return record


async def rename_host(
    data_dir: Path, owner_user_id: str, host_id: str, name: str
) -> Optional[HostRecord]:
    """Rename a computer. Raises ValueError on an unusable name."""
    cleaned = validate_host_name(name)
    async with _lock:
        record = get_host(data_dir, owner_user_id, host_id)
        if record is None:
            return None
        record.name = cleaned
        _save(data_dir, record)
        return record


async def revoke_host(
    data_dir: Path, owner_user_id: str, host_id: str
) -> Optional[HostRecord]:
    """Disconnect a computer for good.

    The credential is DESTROYED, not parked: ``token_hash`` is replaced with a
    value no token can hash to, so re-arming the old key is impossible even by
    editing ``revoked_at`` back out. Reconnecting means a fresh pairing code,
    which is the promise the confirm dialog makes.

    Idempotent — revoking an already-revoked host returns it unchanged rather
    than erroring, so a double-click cannot produce a scary failure.
    """
    async with _lock:
        record = get_host(data_dir, owner_user_id, host_id)
        if record is None:
            return None
        if record.is_revoked:
            return record
        record.revoked_at = _now()
        record.token_hash = "revoked"
        _save(data_dir, record)
        return record
