# SPDX-License-Identifier: MIT
"""
clawmeets/runner/egress_proxy.py

The scoped turn's only network path (plan 4.3, hardening item 7).

``EgressProxy`` runs in the runner. Each scoped turn gets its own listener:

- Linux: a Unix socket inside the turn home, bound into the sandbox at
  ``/run/clawmeets/egress.sock``; the in-sandbox netshim
  (``clawmeets.runner.netshim``) listens on ``127.0.0.1:3128`` inside the
  network namespace and forwards there.
- macOS: a runner-chosen ``127.0.0.1`` port, the only network destination the
  sandbox profile allows.

What it does with a request:

- ``CONNECT host:port`` -> resolve the name ONCE, refuse unless every resolved
  address is public (no loopback, RFC 1918, CGNAT, link-local incl.
  169.254.169.254, fc00::/7, ::1, multicast, reserved), connect to the PINNED
  address and splice bytes. A later DNS answer can never reopen localhost
  because nothing resolves the name again for that connection.
- ``GET http://host/...`` (absolute form) -> same checks, forwarded in origin
  form with ``Proxy-*`` and hop-by-hop headers removed.
- An origin-form request (``POST /m/<nonce>/v1/messages``) goes to the
  ``local_handler`` (the model-auth proxy, commit 2c) when one is installed;
  otherwise 403.

It attaches no credential to tunnelled or forwarded traffic, ever: the
model-auth handler is the only code path that adds one, and it serves only
origin-form requests on its nonce prefix.
"""
from __future__ import annotations

import http.server
import ipaddress
import logging
import os
import selectors
import shutil
import socket
import socketserver
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional, Protocol
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# Networks a turn may never reach through the proxy. ``is_global`` covers most
# of these too; the explicit list documents intent and guards older Pythons.
BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(n)
    for n in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",      # CGNAT
        "127.0.0.0/8",
        "169.254.0.0/16",     # link-local, incl. 169.254.169.254 metadata
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "255.255.255.255/32",
        "::/128",
        "::1/128",
        "fc00::/7",           # unique local
        "fe80::/10",          # link-local
        "ff00::/8",           # multicast
    )
)

# Inside the sandbox (Linux) the shim listens here.
NETSHIM_LISTEN = ("127.0.0.1", 3128)
SANDBOX_SOCKET_PATH = "/run/clawmeets/egress.sock"

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade",
}

_CONNECT_TIMEOUT = 15
_IDLE_TIMEOUT = 600


def is_public_address(addr: str) -> bool:
    """True only for a globally routable unicast address."""
    try:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        mapped = ip.ipv4_mapped or ip.sixtofour
        if mapped is not None:
            ip = mapped
    if any(ip in net for net in BLOCKED_NETWORKS if net.version == ip.version):
        return False
    return bool(ip.is_global) and not ip.is_multicast


Resolver = Callable[[str, int], list[str]]
Connector = Callable[[str, int], socket.socket]


def default_resolver(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    out: list[str] = []
    for info in infos:
        a = info[4][0]
        if a not in out:
            out.append(a)
    return out


def default_connector(addr: str, port: int) -> socket.socket:
    return socket.create_connection((addr, port), timeout=_CONNECT_TIMEOUT)


class LocalHandler(Protocol):
    """The model-auth proxy's hook (commit 2c)."""

    def handles(self, path: str) -> bool: ...

    def handle(self, request: "http.server.BaseHTTPRequestHandler") -> None: ...


class EgressRefused(Exception):
    """A destination the proxy will not reach. ``str(e)`` is the reason."""


class EgressProxy:
    """Per-turn listeners sharing one address policy and one local handler."""

    def __init__(
        self,
        *,
        resolver: Optional[Resolver] = None,
        connector: Optional[Connector] = None,
        address_allowed: Optional[Callable[[str], bool]] = None,
        use_unix_socket: Optional[bool] = None,
    ) -> None:
        self._resolver = resolver or default_resolver
        self._connector = connector or default_connector
        self._address_allowed = address_allowed or is_public_address
        self._use_unix = (
            sys.platform.startswith("linux") if use_unix_socket is None else use_unix_socket
        )
        self.local_handler: Optional[LocalHandler] = None
        self._servers: dict[str, socketserver.BaseServer] = {}
        self._lock = threading.Lock()

    # -- policy ---------------------------------------------------------------

    def pin(self, host: str, port: int) -> str:
        """Resolve once and return the address to connect to, or raise.

        Every resolved address must be public: a name that answers with both a
        public and a private address is refused, not partially trusted.
        """
        host = host.strip("[]")
        if not host or not (0 < port < 65536):
            raise EgressRefused("bad destination")
        try:
            addrs = self._resolver(host, port)
        except OSError as e:
            raise EgressRefused(f"cannot resolve {host}: {e}") from e
        if not addrs:
            raise EgressRefused(f"cannot resolve {host}")
        for a in addrs:
            if not self._address_allowed(a):
                raise EgressRefused(f"{host} resolves to a non-public address ({a})")
        return addrs[0]

    def open_upstream(self, host: str, port: int) -> socket.socket:
        addr = self.pin(host, port)
        try:
            return self._connector(addr, port)
        except OSError as e:
            raise EgressRefused(f"connect to {host} failed: {e}") from e

    # -- listeners ------------------------------------------------------------

    def bind_for_turn(self, home) -> str:
        """Open this turn's listener. Returns the Unix socket path (Linux) or
        the loopback URL (macOS), and records it on ``home``.

        ``home.egress_url`` is always what the turn puts in HTTPS_PROXY: the
        loopback URL on macOS, the netshim's in-namespace address on Linux.
        """
        handler = _make_handler(self)
        if self._use_unix:
            # A short private dir: AF_UNIX paths are capped near 104-108 bytes,
            # which a turn home under a long agent name can exceed.
            sock_dir = Path(tempfile.mkdtemp(prefix="cm-egress-"))
            os.chmod(sock_dir, 0o700)
            path = sock_dir / "s"
            server: socketserver.BaseServer = _ThreadingUnixServer(str(path), handler)
            os.chmod(path, 0o600)
            home.egress_socket = path
            home.egress_port = None
            home.egress_url = f"http://{NETSHIM_LISTEN[0]}:{NETSHIM_LISTEN[1]}"
            result = str(path)
        else:
            server = _ThreadingTCPServer(("127.0.0.1", 0), handler)
            port = server.server_address[1]
            home.egress_socket = None
            home.egress_port = port
            home.egress_url = f"http://127.0.0.1:{port}"
            result = home.egress_url
        t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, name=f"egress-{home.nonce[:8]}", daemon=True)
        t.start()
        with self._lock:
            self._servers[home.nonce] = server
        return result

    def unbind_turn(self, home) -> None:
        with self._lock:
            server = self._servers.pop(home.nonce, None)
        if server is not None:
            server.shutdown()
            server.server_close()
            server.cut_connections()
        sock = getattr(home, "egress_socket", None)
        if sock is not None:
            shutil.rmtree(Path(sock).parent, ignore_errors=True)

    def close(self) -> None:
        with self._lock:
            servers = list(self._servers.values())
            self._servers.clear()
        for s in servers:
            s.shutdown()
            s.server_close()
            s.cut_connections()


class _TurnConnections:
    """Server mixin: the turn's open client connections, so unbinding the turn
    also ends its spliced CONNECT tunnels and forwards (stopping the listener
    alone leaves those threads running for up to the idle timeout)."""

    def __init__(self, *args, **kwargs) -> None:
        self._conns: set[socket.socket] = set()
        self._conns_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def track(self, conn: socket.socket) -> None:
        with self._conns_lock:
            self._conns.add(conn)

    def untrack(self, conn: socket.socket) -> None:
        with self._conns_lock:
            self._conns.discard(conn)

    def cut_connections(self) -> None:
        # shutdown, not close: wakes the handler's select/recv with EOF and
        # leaves closing (and the fd) to the handler thread that owns it.
        with self._conns_lock:
            conns = list(self._conns)
        for c in conns:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class _ThreadingTCPServer(_TurnConnections, socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _ThreadingUnixServer(_TurnConnections, socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def _splice(a: socket.socket, b: socket.socket) -> None:
    """Copy bytes both ways until either side closes or the idle timeout hits."""
    sel = selectors.DefaultSelector()
    sel.register(a, selectors.EVENT_READ, b)
    sel.register(b, selectors.EVENT_READ, a)
    try:
        while True:
            events = sel.select(timeout=_IDLE_TIMEOUT)
            if not events:
                return
            for key, _ in events:
                src: socket.socket = key.fileobj  # type: ignore[assignment]
                dst: socket.socket = key.data
                try:
                    data = src.recv(65536)
                except OSError:
                    return
                if not data:
                    return
                try:
                    dst.sendall(data)
                except OSError:
                    return
    finally:
        sel.close()


def _make_handler(proxy: EgressProxy) -> type:
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # Unbuffered request reader: bytes a client pipelines after CONNECT must
        # stay in the socket for the splice, not in a read-ahead buffer.
        rbufsize = 0
        server_version = "clawmeets-egress"
        sys_version = ""

        def setup(self) -> None:
            # UnixStreamServer hands an empty client_address; give it a name.
            if not self.client_address:
                self.client_address = ("unix", 0)
            self.server.track(self.request)
            super().setup()

        def finish(self) -> None:
            try:
                super().finish()
            finally:
                self.server.untrack(self.request)

        def log_message(self, fmt: str, *args) -> None:  # noqa: D401
            logger.debug("egress: " + fmt, *args)

        def _refuse(self, code: int, reason: str) -> None:
            body = (reason + "\n").encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def do_CONNECT(self) -> None:
            host, _, port_s = self.path.rpartition(":")
            try:
                port = int(port_s)
            except ValueError:
                return self._refuse(400, "bad CONNECT target")
            try:
                upstream = proxy.open_upstream(host, port)
            except EgressRefused as e:
                logger.info(f"egress: refused CONNECT {self.path}: {e}")
                return self._refuse(403, f"refused: {e}")
            try:
                self.send_response(200, "Connection Established")
                self.end_headers()
                self.wfile.flush()
                _splice(self.connection, upstream)
            finally:
                upstream.close()
                self.close_connection = True

        def _do_other(self) -> None:
            if not self.path.startswith(("http://", "https://")):
                lh = proxy.local_handler
                if lh is not None and lh.handles(self.path):
                    return lh.handle(self)
                return self._refuse(403, "refused: not a proxy request")
            parts = urlsplit(self.path)
            if parts.scheme != "http":
                return self._refuse(400, "use CONNECT for https")
            # An absolute-form request aimed at the proxy itself is local.
            lh = proxy.local_handler
            if lh is not None and parts.hostname in ("127.0.0.1", "localhost"):
                origin = parts.path + (f"?{parts.query}" if parts.query else "")
                if lh.handles(origin):
                    self.path = origin
                    return lh.handle(self)
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                return self._refuse(411, "chunked request bodies are not supported")
            try:
                upstream = proxy.open_upstream(parts.hostname or "", parts.port or 80)
            except EgressRefused as e:
                logger.info(f"egress: refused {self.command} {self.path}: {e}")
                return self._refuse(403, f"refused: {e}")
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
                lines = [f"{self.command} {target} HTTP/1.1"]
                for k, v in self.headers.items():
                    if k.lower() in _HOP_BY_HOP:
                        continue
                    lines.append(f"{k}: {v}")
                lines.append("Connection: close")
                upstream.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body)
                while True:
                    chunk = upstream.recv(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            except OSError:
                pass
            finally:
                upstream.close()
                self.close_connection = True

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _do_other

    return Handler
