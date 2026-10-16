# SPDX-License-Identifier: MIT
"""
clawmeets/runner/turn_sandbox.py

Wrap a scoped turn's subprocess in an OS sandbox (plan 4.3, items 5-8).

- macOS: ``/usr/bin/sandbox-exec -p <profile>``. The profile starts from
  ``(deny default)`` and allows only what a provider CLI needs (see
  :func:`build_macos_profile` and the pinned :data:`MACH_LOOKUP_ALLOW`), then
  explicitly denies the visibility table's denied areas, Apple Events, the
  pasteboard, LaunchServices, WindowServer, keychain services, IOKit and other
  processes' info, so a later or broader allow cannot reopen them.
- Linux: ``bwrap --unshare-user --unshare-pid --unshare-ipc --unshare-uts
  --unshare-cgroup --unshare-net --new-session --die-with-parent`` over an empty
  root with read-only binds of the system and the turn's read roots. The PID
  namespace hides the runner's ``/proc/<pid>/environ``; the network namespace
  removes host loopback and host abstract sockets. The netshim is the first
  process and the only listener on the namespace's loopback.

Scoped kinds never run unsandboxed: :func:`wrap_command` raises
:class:`SandboxUnavailable` when no working sandbox exists, and a full turn's
command is returned unchanged (full turns are byte-for-byte as before).
"""
from __future__ import annotations

import functools
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, Optional, Sequence

from ..llm.base import SandboxUnavailable
from .egress_proxy import NETSHIM_LISTEN, SANDBOX_SOCKET_PATH

if TYPE_CHECKING:
    from .turn_home import TurnHome
    from .turn_scope import TurnScope

SandboxKind = Literal["sandbox-exec", "bwrap"]

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

# Instruction files a provider auto-loads from ancestors and user config.
INSTRUCTION_FILES: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md", "GEMINI.md")
INSTRUCTION_DIRS: tuple[str, ...] = (".claude", ".codex", ".gemini")

# macOS mach services a turn may look up. Pinned; everything else is denied by
# ``(deny default)``. System logging, user/group lookup (libinfo via
# opendirectoryd), and the certificate-trust service for providers whose TLS
# uses the system trust store.
MACH_LOOKUP_ALLOW: tuple[str, ...] = (
    "com.apple.system.logger",
    "com.apple.logd",
    "com.apple.system.notification_center",
    "com.apple.system.opendirectoryd.libinfo",
    "com.apple.system.opendirectoryd.membership",
    "com.apple.trustd",
    "com.apple.trustd.agent",
)

# Explicitly denied (and checked by the canaries) so no later allow reopens them.
MACH_LOOKUP_DENY_PREFIXES: tuple[str, ...] = (
    "com.apple.pasteboard",          # pbpaste / pbcopy
    "com.apple.coreservices.",       # LaunchServices (open -a)
    "com.apple.lsd.",
    "com.apple.windowserver",        # WindowServer
    "com.apple.SecurityServer",      # keychain
    "com.apple.securityd",
    "com.apple.security.",
    "com.apple.ctkd",
    "com.apple.secd",
    "com.apple.tccd",
)

# Package-manager prefixes. Only their package dirs are readable: ``var/``
# (local databases, logs, caches), ``etc/`` (service configs) and anything else
# installed at the top (``/usr/local/aws-cli``) are not, and the first two are
# denied explicitly.
MACOS_PACKAGE_PREFIXES: tuple[str, ...] = ("/opt/homebrew", "/usr/local")
MACOS_PACKAGE_DIRS: tuple[str, ...] = (
    "bin", "sbin", "lib", "libexec", "opt", "Cellar", "Caskroom", "Frameworks", "share", "include",
)
MACOS_PACKAGE_DENIED: tuple[str, ...] = ("var", "etc")
# Re-allowed inside a prefix's etc/: Homebrew's OpenSSL refuses to start
# (node-based CLIs included) when its config is unreadable. Public trust
# material and config only; ``openssl@*/private`` stays denied.
MACOS_PACKAGE_ETC_ALLOW_RE: tuple[str, ...] = (
    r"/etc/openssl[^/]*/(openssl\.cnf|cert\.pem|ct_log_list\.cnf)$",
    r"/etc/openssl[^/]*/certs(/|$)",
    r"/etc/ca-certificates(/|$)",
)
_PACKAGE_PATHS = tuple(f"{p}/{d}" for p in MACOS_PACKAGE_PREFIXES for d in MACOS_PACKAGE_DIRS)

# System paths a provider CLI may read and execute from. ``/usr`` is listed by
# subdirectory so it does not cover ``/usr/local``.
MACOS_SYSTEM_READ: tuple[str, ...] = (
    "/usr/bin",
    "/usr/lib",
    "/usr/libexec",
    "/usr/sbin",
    "/usr/share",
    "/usr/standalone",
    "/bin",
    "/sbin",
    "/System",
    "/Library/Apple",
    "/Library/Frameworks",
    "/Library/Preferences",
    "/Library/Developer/CommandLineTools",
    "/Applications/Xcode.app/Contents/Developer",
    "/private/etc",
    "/private/var/db/timezone",
    "/private/var/db/dyld",
    "/private/var/select",
    *_PACKAGE_PATHS,
)
MACOS_SYSTEM_EXEC: tuple[str, ...] = (
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
    "/usr/libexec",
    "/System",
    "/Library/Developer/CommandLineTools",
    "/Applications/Xcode.app/Contents/Developer",
    *_PACKAGE_PATHS,
)
# Tools whose only purpose in a turn would be to drive the GUI session, the
# clipboard or the keychain. Exec denied outright (``osascript -e 'return 1'``
# needs no Apple Event, so denying appleevent-send alone does not stop it).
# Claude Code's own ``security`` call is answered by the turn home's shim
# (turn_home: shims/security, first on PATH), never by the real tool.
MACOS_DENIED_EXEC: tuple[str, ...] = (
    "/usr/bin/osascript",
    "/usr/bin/open",
    "/usr/bin/pbcopy",
    "/usr/bin/pbpaste",
    "/usr/bin/security",
    "/usr/bin/screencapture",
    "/usr/bin/shortcuts",
    "/usr/bin/automator",
)

MACOS_DEV_RW: tuple[str, ...] = (
    "/dev/null", "/dev/zero", "/dev/tty", "/dev/dtracehelper", "/dev/stdout", "/dev/stderr",
)
MACOS_DEV_R: tuple[str, ...] = ("/dev/random", "/dev/urandom", "/dev/autofs_nowait")

LINUX_SYSTEM_RO: tuple[str, ...] = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/etc", "/opt")
LINUX_ETC_HIDDEN: tuple[str, ...] = ("/etc/ssh", "/etc/shadow", "/etc/gshadow", "/etc/sudoers.d")


__all__ = ["SandboxUnavailable", "detect_sandbox", "wrap_command", "build_macos_profile", "build_bwrap_args"]


def _real(p: Path | str) -> Path:
    return Path(os.path.realpath(os.path.expanduser(str(p))))


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _probe(cmd: list[str]) -> bool:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


@functools.lru_cache(maxsize=1)
def _detect_cached() -> Optional[str]:
    if sys.platform == "darwin":
        if os.path.exists(SANDBOX_EXEC) and _probe(
            [SANDBOX_EXEC, "-p", "(version 1)(allow default)", "/usr/bin/true"]
        ):
            return "sandbox-exec"
        return None
    if sys.platform.startswith("linux"):
        bwrap = shutil.which("bwrap")
        if bwrap and _probe([
            bwrap, "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
            "--unshare-cgroup", "--unshare-net", "--new-session", "--die-with-parent",
            "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev", "true",
        ]):
            return "bwrap"
    return None


def detect_sandbox() -> Optional[SandboxKind]:
    """Which OS sandbox works on this host, or ``None`` (probed once, cached)."""
    return _detect_cached()  # type: ignore[return-value]


def reset_detection() -> None:
    _detect_cached.cache_clear()


# ---------------------------------------------------------------------------
# What a command needs to execute
# ---------------------------------------------------------------------------


def _shebang_interpreter(path: Path, env_path: Optional[str]) -> Optional[Path]:
    try:
        with open(path, "rb") as f:
            head = f.read(256)
    except OSError:
        return None
    if not head.startswith(b"#!"):
        return None
    line = head[2:].split(b"\n", 1)[0].decode("utf-8", "replace").strip().split()
    if not line:
        return None
    interp = line[0]
    if os.path.basename(interp) == "env" and len(line) > 1:
        args = [a for a in line[1:] if not a.startswith("-")]
        if not args:
            return None
        found = shutil.which(args[0], path=env_path)
        return _real(found) if found else None
    return _real(interp)


# A package prefix itself is never an install root (that would reopen var/, etc/).
_SYSTEM_PREFIXES = tuple(Path(p) for p in (*MACOS_SYSTEM_READ, *MACOS_PACKAGE_PREFIXES, *LINUX_SYSTEM_RO))


def _install_root(real_bin: Path) -> Optional[Path]:
    """Install root of a binary, or ``None`` when the system read set already
    covers it. Never returns a path shallower than three components, so a
    binary in ``/bin`` can never open ``/``."""
    root = _install_root_raw(real_bin)
    for sysp in _SYSTEM_PREFIXES:
        if root == sysp or sysp in root.parents:
            return None
    if len(root.parts) < 4:  # ("/", "Users", "name") is a home dir: too broad
        return real_bin if real_bin.is_file() else None
    return root


def _install_root_raw(real_bin: Path) -> Path:
    """The directory tree a binary needs to read: its package, not just itself.

    ``…/lib/node_modules/<pkg>/dist/index.js`` -> ``…/lib/node_modules/<pkg>``;
    ``…/<prefix>/bin/<tool>`` -> ``…/<prefix>``; a lone versioned binary
    (``~/.local/share/claude/versions/X``) -> its own directory.
    """
    parts = real_bin.parts
    if "node_modules" in parts:
        i = len(parts) - 1 - parts[::-1].index("node_modules")
        n = i + (3 if i + 1 < len(parts) and parts[i + 1].startswith("@") else 2)
        return Path(*parts[:n])
    if real_bin.parent.name == "bin":
        prefix = real_bin.parent.parent
        home = _real(Path.home())
        # Only a real install prefix (bin/ + lib/), never the user's home,
        # ~/.local, or anything above them: those hold credentials.
        if (prefix / "lib").is_dir() and prefix not in (home, home / ".local") and prefix not in home.parents:
            return prefix
    return real_bin.parent


def command_roots(cmd: Sequence[str], env_path: Optional[str] = None) -> tuple[list[Path], list[Path]]:
    """(exec paths, read roots) the command's program and interpreter need."""
    execs: list[Path] = []
    reads: list[Path] = []
    if not cmd:
        return execs, reads
    found = shutil.which(cmd[0], path=env_path) if os.sep not in cmd[0] else cmd[0]
    if found:
        real_bin = _real(found)
        execs.append(real_bin)
        reads.append(real_bin)
        root = _install_root(real_bin)
        if root is not None:
            reads.append(root)
        if Path(found) != real_bin:
            reads.append(Path(os.path.abspath(found)))
        interp = _shebang_interpreter(real_bin, env_path)
        if interp is not None:
            execs.append(interp)
            reads.append(interp)
            root = _install_root(interp)
            if root is not None:
                reads.append(root)
    # The runner's own interpreter (netshim, probes, python-based CLIs).
    for p in {sys.prefix, sys.base_prefix, sys.exec_prefix}:
        rp = _real(p)
        if len(rp.parts) >= 4:
            reads.append(rp)
    execs.append(_real(sys.executable))
    return execs, reads


# ---------------------------------------------------------------------------
# macOS profile
# ---------------------------------------------------------------------------


def _q(path: Path | str) -> str:
    s = str(path).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _ancestors(p: Path) -> list[Path]:
    return [a for a in p.parents]


def instruction_denies(cwd: Path, read_roots: Iterable[Path], instruction_roots: Iterable[Path]) -> list[Path]:
    """Instruction files the turn may not read: every one in an ancestor of the
    cwd, and at the top of every read root other than the knowledge dirs. The
    cwd's own (runner-generated) files stay readable."""
    allowed = {_real(r) for r in instruction_roots}
    dirs: list[Path] = list(_ancestors(_real(cwd)))
    dirs.extend(r for r in (_real(x) for x in read_roots) if r not in allowed and r.is_dir())
    out: list[Path] = []
    for d in dirs:
        for name in (*INSTRUCTION_FILES, *INSTRUCTION_DIRS):
            out.append(d / name)
    return out


def build_macos_profile(
    scope: "TurnScope",
    home: "TurnHome",
    *,
    cwd: Path,
    exec_paths: Iterable[Path] = (),
    extra_read: Iterable[Path] = (),
) -> str:
    """The deny-by-default sandbox-exec profile for one scoped turn."""
    if home.egress_port is None:
        raise SandboxUnavailable("egress proxy is not bound for this turn")
    turn_root = _real(home.root)
    reads = [*(_real(p) for p in MACOS_SYSTEM_READ), *scope.read_roots, *(_real(p) for p in extra_read)]
    writes = [*scope.write_roots, turn_root]
    execs = [*(_real(p) for p in MACOS_SYSTEM_EXEC), *(_real(p) for p in exec_paths)]
    user_home = _real(Path.home())

    L: list[str] = [
        "(version 1)",
        "(deny default)",
        # Processes: fork, and exec only of system tools + the provider install.
        "(allow process-fork)",
        "(allow process-exec " + " ".join(
            f"(subpath {_q(p)})" if Path(p).is_dir() else f"(literal {_q(p)})" for p in execs
        ) + ")",
        "(allow process-info* (target self))",
        "(allow process-info-pidinfo process-info-setcontrol (target same-sandbox))",
        "(allow signal (target same-sandbox))",
        "(allow sysctl-read)",
        "(allow mach-lookup " + " ".join(f"(global-name {_q(n)})" for n in MACH_LOOKUP_ALLOW) + ")",
        "(allow ipc-posix-shm-read-data (ipc-posix-name \"apple.shm.notification_center\"))",
        # Path traversal (realpath, getcwd) needs stat on each ancestor of an
        # allowed root; only those literals, so no directory can be listed.
        "(allow file-read-metadata " + " ".join(
            f"(literal {_q(a)})" for a in sorted({a for r in [*reads, *writes, *execs] for a in _ancestors(Path(r))}, key=str)
        ) + ")",
        '(allow file-read* (literal "/") (literal "/private") (literal "/var") (literal "/etc") (literal "/tmp"))',
        # The Xcode shims (/usr/bin/git, …) read this symlink itself to find
        # the developer dir; resolving it (as every other read root is) would
        # allow only its target, so the link is allowed as a literal.
        '(allow file-read* (literal "/private/var/db/xcode_select_link"))',
        "(allow file-read* " + " ".join(f"(subpath {_q(p)})" for p in reads + writes) + ")",
        "(allow file-write* " + " ".join(f"(subpath {_q(p)})" for p in writes) + ")",
        "(allow file-read* file-write-data file-ioctl " + " ".join(f"(literal {_q(p)})" for p in MACOS_DEV_RW) + ")",
        "(allow file-read* " + " ".join(f"(literal {_q(p)})" for p in MACOS_DEV_R) + ")",
        '(allow file-read* file-write-data file-ioctl (regex #"^/dev/fd/[0-9]+$") (regex #"^/dev/ttys[0-9]+$"))',
        # Network: only the runner's egress proxy port on loopback.
        f'(allow network-outbound (remote ip "localhost:{int(home.egress_port)}"))',
        # ---- explicit denies: nothing above may reopen these ----
        "(deny appleevent-send)",
        "(deny process-exec " + " ".join(f"(literal {_q(p)})" for p in MACOS_DENIED_EXEC) + ")",
        "(deny iokit-open)",
        "(deny process-info* (target others))",
        "(deny mach-lookup " + " ".join(f"(global-name-prefix {_q(n)})" for n in MACH_LOOKUP_DENY_PREFIXES) + ")",
        f"(deny file-read* file-write* (subpath {_q(user_home / 'Library' / 'Keychains')}))",
        '(deny file-read* file-write* (subpath "/Library/Keychains"))',
        "(deny file-read* file-write* " + " ".join(
            f"(subpath {_q(_real(Path(p) / d))})" for p in MACOS_PACKAGE_PREFIXES for d in MACOS_PACKAGE_DENIED
        ) + ")",
        "(allow file-read* " + " ".join(
            f'(regex #"^{re.escape(str(_real(p)))}{r}")' for p in MACOS_PACKAGE_PREFIXES for r in MACOS_PACKAGE_ETC_ALLOW_RE
        ) + ")",
    ]
    denies = [*scope.deny_roots, *instruction_denies(cwd, scope.read_roots, scope.instruction_roots)]
    if denies:
        L.append("(deny file-read* file-write* " + " ".join(
            f"(subpath {_q(p)})" for p in denies
        ) + ")")
    # The turn home lives under the denied turns root; re-allow this one only.
    L.append(f"(allow file-read* file-write* (subpath {_q(turn_root)}))")
    # The home's stand-ins for denied tools (turn_home.MACOS_SHIMS): exec only
    # these files, and the turn may not rewrite them into something else.
    shims = home.shim_paths()
    if shims:
        L.append("(allow process-exec " + " ".join(f"(literal {_q(_real(p))})" for p in shims) + ")")
        L.append(f"(deny file-write* (subpath {_q(_real(home.shim_dir))}))")
    # A subpath allow covers the root itself, so the turn could rmdir its
    # sandbox and put a symlink there that the runner, unsandboxed, would
    # follow. Everything below a write root stays writable; the root does not.
    L.append("(deny file-write* " + " ".join(f"(literal {_q(p)})" for p in writes) + ")")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# macOS: processes a turn left behind
# ---------------------------------------------------------------------------

_SANDBOX_FILTER_NONE = 0
_SANDBOX_FILTER_PATH = 1
_SANDBOX_CHECK_NO_REPORT = 0x40000000


@functools.lru_cache(maxsize=1)
def _libsystem():
    """``(proc_listallpids, check(pid, op, path))`` from libSystem, or ``None``."""
    import ctypes
    import platform

    try:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        listpids, sandbox_check = lib.proc_listallpids, lib.sandbox_check
    except (OSError, AttributeError):
        return None
    listpids.restype = ctypes.c_int
    listpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
    sandbox_check.restype = ctypes.c_int
    fixed = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    # sandbox_check(pid, op, type, ...) is variadic. Apple arm64 passes a
    # variadic argument on the stack, which ctypes cannot express: five
    # padding arguments fill the remaining registers so the path lands there.
    pad = [None] * 5 if platform.machine() == "arm64" else []
    sandbox_check.argtypes = fixed + [ctypes.c_void_p] * len(pad) + [ctypes.c_char_p]

    def check(pid: int, op: Optional[str], path: Optional[Path]) -> int:
        if op is None:
            return sandbox_check(pid, None, _SANDBOX_FILTER_NONE, *pad, None)
        return sandbox_check(pid, op.encode(), _SANDBOX_FILTER_PATH | _SANDBOX_CHECK_NO_REPORT,
                             *pad, os.fsencode(path))

    def pids() -> list[int]:
        n = listpids(None, 0)
        buf = (ctypes.c_int * (max(n, 0) + 256))()
        n = listpids(buf, ctypes.sizeof(buf))
        return [p for p in buf[:max(n, 0)] if p > 0]

    return pids, check


def kill_macos_turn_leftovers(own: Path, denied: Path, *, max_rounds: int = 1000) -> int:
    """SIGKILL every process still running under one turn's sandbox profile.

    The turn's process group is killed when its subprocess returns, but a
    child can leave the group with ``setsid()`` (bwrap's pid namespace ends it
    on Linux; sandbox-exec has none). What it cannot leave is its sandbox, so
    a leftover is found by asking the kernel about each process
    (``sandbox_check``): it is sandboxed, may not read ``denied`` (the turns
    dir, which only a turn profile denies) and may read ``own`` (this turn's
    home root, under that dir: only this turn's profile re-allows it). ``own``
    itself is the probe, never a path inside it: the profile denies writing
    the root, so the turn cannot remove or swap it, while ``own/tmp`` it can
    (a probe on a missing or redirected path reports "denied" and spared the
    process). Matches are SIGSTOPped pass after pass until a pass finds no new
    one (a stopped process cannot fork, so this ends), then all are SIGKILLed.
    Returns how many were killed; a host without the calls kills none.
    """
    lib = _libsystem()
    if lib is None:
        return 0
    pids, check = lib
    me, stopped = os.getpid(), set()
    for _ in range(max_rounds):
        found = [
            pid for pid in pids()
            if pid != me
            and pid not in stopped
            and check(pid, None, None) == 1
            and check(pid, "file-read-data", denied) == 1
            and check(pid, "file-read-data", own) == 0
        ]
        if not found:
            break
        for pid in found:
            try:
                os.kill(pid, signal.SIGSTOP)
                stopped.add(pid)
            except (ProcessLookupError, PermissionError):
                pass
    killed = 0
    for pid in stopped:
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except (ProcessLookupError, PermissionError):
            pass
    return killed


# ---------------------------------------------------------------------------
# Linux bwrap
# ---------------------------------------------------------------------------


def build_bwrap_args(
    scope: "TurnScope",
    home: "TurnHome",
    *,
    cwd: Path,
    extra_read: Iterable[Path] = (),
    bwrap: str = "bwrap",
) -> list[str]:
    """bwrap argv (everything before the netshim) for one scoped turn."""
    if home.egress_socket is None:
        raise SandboxUnavailable("egress proxy is not bound for this turn")
    a = [
        bwrap,
        "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
        "--unshare-cgroup", "--unshare-net", "--new-session", "--die-with-parent",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/run",
    ]
    for p in LINUX_SYSTEM_RO:
        a += ["--ro-bind-try", p, p]
    for p in LINUX_ETC_HIDDEN:
        if os.path.isdir(p):
            a += ["--tmpfs", p]
        elif os.path.exists(p):
            a += ["--ro-bind", "/dev/null", p]
    for p in (*(_real(x) for x in extra_read), *scope.read_roots):
        a += ["--ro-bind-try", str(p), str(p)]
    for p in (*scope.write_roots, _real(home.root)):
        a += ["--bind", str(p), str(p)]
    # Denied paths inside a bound root are masked (dirs by tmpfs, files by /dev/null).
    bound = [*scope.read_roots, *scope.write_roots, *(_real(x) for x in extra_read)]
    for d in (*scope.deny_roots, *instruction_denies(cwd, scope.read_roots, scope.instruction_roots)):
        if not any(d == b or b in d.parents for b in bound):
            continue  # not bound at all => not visible
        if d.is_dir():
            a += ["--tmpfs", str(d)]
        elif d.exists():
            a += ["--ro-bind", "/dev/null", str(d)]
    a += ["--dir", os.path.dirname(SANDBOX_SOCKET_PATH)]
    a += ["--bind", str(home.egress_socket), SANDBOX_SOCKET_PATH]
    a += ["--chdir", str(_real(cwd))]
    return a


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def wrap_command(
    cmd: Sequence[str],
    scope: "TurnScope",
    home: Optional["TurnHome"],
    *,
    cwd: Optional[Path] = None,
    extra_read: Iterable[Path] = (),
    env_path: Optional[str] = None,
    sandbox: Optional[str] = None,
) -> list[str]:
    """Return ``cmd`` wrapped for the turn's sandbox.

    A full turn gets ``cmd`` back unchanged. A scoped turn gets the sandboxed
    argv, or :class:`SandboxUnavailable` when there is no working sandbox or
    no turn home; it never falls back to running unsandboxed.
    """
    cmd = list(cmd)
    if not scope.scoped:
        return cmd
    if home is None:
        raise SandboxUnavailable("scoped turn has no turn home")
    kind = sandbox or detect_sandbox()
    if kind is None:
        raise SandboxUnavailable(
            "this host has no working OS sandbox (sandbox-exec on macOS, bubblewrap on Linux); "
            "a scoped turn cannot run here"
        )
    cwd = _real(cwd or (scope.write_roots[0] if scope.write_roots else home.home))
    execs, reads = command_roots(cmd, env_path)
    reads = [*reads, *(_real(p) for p in extra_read)]
    if kind == "sandbox-exec":
        profile = build_macos_profile(scope, home, cwd=cwd, exec_paths=execs, extra_read=reads)
        # Exec the resolved binary itself. sandbox-exec's PATH search would
        # try the symlink (``~/.local/bin/claude``), which the profile does
        # not allow, and silently fall through to a later, other install.
        found = shutil.which(cmd[0], path=env_path) if cmd and os.sep not in cmd[0] else None
        if found:
            cmd[0] = str(_real(found))
        return [SANDBOX_EXEC, "-p", profile, *cmd]
    if kind == "bwrap":
        argv = build_bwrap_args(scope, home, cwd=cwd, extra_read=reads, bwrap=shutil.which("bwrap") or "bwrap")
        shim = [
            str(_real(sys.executable)), "-m", "clawmeets.runner.netshim",
            "--listen", f"{NETSHIM_LISTEN[0]}:{NETSHIM_LISTEN[1]}",
            "--upstream", SANDBOX_SOCKET_PATH, "--",
        ]
        # The shim is found through the runner's own source / site-packages root.
        src_root = Path(__file__).resolve().parents[2]
        argv += ["--ro-bind-try", str(src_root), str(src_root), "--setenv", "PYTHONPATH", str(src_root)]
        return [*argv, *shim, *cmd]
    raise SandboxUnavailable(f"unknown sandbox kind {kind!r}")
