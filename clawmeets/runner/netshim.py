# SPDX-License-Identifier: MIT
"""
clawmeets/runner/netshim.py — clawmeets-netshim (plan 4.3, hardening item 7).

The first process inside a Linux scoped-turn sandbox. The sandbox has its own
network namespace (``bwrap --unshare-net``): only its own loopback exists, so
the turn cannot reach host loopback services or host abstract sockets. The
runner's egress proxy is reachable only through a Unix socket bound into the
sandbox. This shim listens on ``127.0.0.1:3128`` INSIDE the namespace, forwards
every connection byte-for-byte to that socket, runs the turn's command as its
child and exits with the child's status. It is the only listener on the turn's
loopback.

Usage::

    python -m clawmeets.runner.netshim --listen 127.0.0.1:3128 \
        --upstream /run/clawmeets/egress.sock -- <cmd> [args...]

Stdlib only: it runs inside the sandbox with the runner's interpreter.
"""
from __future__ import annotations

import argparse
import os
import selectors
import signal
import socket
import subprocess
import sys
import threading


def _pipe(a: socket.socket, b: socket.socket) -> None:
    sel = selectors.DefaultSelector()
    sel.register(a, selectors.EVENT_READ, b)
    sel.register(b, selectors.EVENT_READ, a)
    try:
        while True:
            for key, _ in sel.select():
                try:
                    data = key.fileobj.recv(65536)  # type: ignore[union-attr]
                except OSError:
                    return
                if not data:
                    return
                try:
                    key.data.sendall(data)
                except OSError:
                    return
    finally:
        sel.close()
        a.close()
        b.close()


def _serve(listener: socket.socket, upstream: str) -> None:
    while True:
        try:
            conn, _ = listener.accept()
        except OSError:
            return
        try:
            up = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            up.connect(upstream)
        except OSError:
            conn.close()
            continue
        threading.Thread(target=_pipe, args=(conn, up), daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        i = argv.index("--")
        own, cmd = argv[:i], argv[i + 1:]
    else:
        own, cmd = argv, []
    ap = argparse.ArgumentParser(prog="clawmeets-netshim")
    ap.add_argument("--listen", default="127.0.0.1:3128")
    ap.add_argument("--upstream", required=True)
    args = ap.parse_args(own)
    host, _, port = args.listen.rpartition(":")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, int(port)))
    listener.listen(64)
    threading.Thread(target=_serve, args=(listener, args.upstream), daemon=True).start()

    if not cmd:
        signal.pause()
        return 0
    child = subprocess.Popen(cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)

    def _forward(signum, _frame):
        try:
            child.send_signal(signum)
        except ProcessLookupError:
            pass

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, _forward)
    rc = child.wait()
    listener.close()
    return rc if rc >= 0 else 128 - rc


if __name__ == "__main__":
    sys.exit(main())
