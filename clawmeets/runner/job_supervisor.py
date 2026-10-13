# SPDX-License-Identifier: MIT
"""The supervisor behind ``clawmeets job run``.

Runs the agent's command under a live managed job and does the bookkeeping
every job needs, so no agent has to write its own driver:

  * heartbeats on a timer (proof the process is alive, not that work moved);
  * output lines starting with ``CLAWMEETS_JOB_PROGRESS `` become progress
    messages in the room;
  * exit 0 marks the job complete, anything else marks it failed with the
    exit status and the last stderr lines;
  * when the server closes the job (stalled, deadline, cancelled), the command
    is stopped by default (``on_close="stop"``) or left to finish (``"keep"``);
  * SIGTERM / SIGINT to the supervisor is forwarded and the job cancelled.

The server's lease check is the authority on whether the job is alive: if the
server is unreachable the command keeps running and calls keep retrying.
A crashed command is never restarted here; the agent is woken to decide.
"""
from __future__ import annotations

import argparse
import collections
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Callable

import httpx

from clawmeets.api.managed_job_client import JobApiError, JobClosed, ManagedJobClient
from clawmeets.utils.agent_processes import IS_WINDOWS, popen_detached_kwargs

logger = logging.getLogger("clawmeets.runner.job_supervisor")

PROGRESS_PREFIX = "CLAWMEETS_JOB_PROGRESS "
#: Exit status of ``job run`` when the server closed the job and the command
#: was stopped.
EXIT_JOB_CLOSED = 3
_STDERR_TAIL = 20
_FINAL_REPORT_ATTEMPTS = 10


class JobSupervisor:
    def __init__(
        self,
        client: ManagedJobClient,
        job_id: str,
        argv: list[str],
        *,
        heartbeat_every: float | None = None,
        on_close: str = "stop",
        grace: float = 10,
        log: IO[str] | None = None,
        retry_sleep: float = 15,
    ) -> None:
        if on_close not in ("stop", "keep"):
            raise ValueError("on_close must be 'stop' or 'keep'")
        self._client = client
        self._job_id = job_id
        self._argv = argv
        self._heartbeat_every = heartbeat_every
        self._on_close = on_close
        self._grace = grace
        self._log = log or sys.stdout
        self._retry_sleep = retry_sleep
        self._lock = threading.Lock()  # serializes every server call
        self._log_lock = threading.Lock()
        self._sequence = 0
        self._closed = threading.Event()
        self._signalled: int | None = None
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=_STDERR_TAIL)
        self._child: subprocess.Popen | None = None

    # -- server calls -------------------------------------------------------

    def _call(self, op: str, **fields) -> dict | None:
        """One event under the lock. None when the server was unreachable."""
        with self._lock:
            if self._closed.is_set() and op in ("heartbeat", "progress"):
                return None
            try:
                job = self._client.event(
                    self._job_id, op, expected_sequence=self._sequence, **fields
                )
            except JobClosed as exc:
                self._sequence = exc.job.get("event_sequence", self._sequence)
                self._note(f"job closed by the server: {exc.detail}")
                self._closed.set()
                return None
            except (httpx.HTTPError, JobApiError) as exc:
                self._note(f"{op} not delivered: {exc}")
                return None
            self._sequence = job["event_sequence"]
            return job

    def _final(self, op: str, **fields) -> None:
        """complete / fail / cancel: keep retrying while the server is away."""
        for attempt in range(_FINAL_REPORT_ATTEMPTS):
            with self._lock:
                try:
                    job = self._client.event(
                        self._job_id, op, expected_sequence=self._sequence, **fields
                    )
                    self._sequence = job["event_sequence"]
                    return
                except JobClosed as exc:
                    self._note(f"{op} not recorded, job already closed: {exc.detail}")
                    return
                except JobApiError as exc:
                    self._note(f"{op} refused: {exc}")
                    return
                except httpx.HTTPError as exc:
                    self._note(f"{op} not delivered (attempt {attempt + 1}): {exc}")
            time.sleep(self._retry_sleep)

    # -- output -------------------------------------------------------------

    def _note(self, text: str) -> None:
        self._write(f"[clawmeets job run] {text}\n")

    def _write(self, line: str) -> None:
        with self._log_lock:
            self._log.write(line)
            self._log.flush()

    def _pump(self, stream: IO[str], is_stderr: bool) -> None:
        for line in iter(stream.readline, ""):
            self._write(line)
            if is_stderr:
                self._stderr_tail.append(line.rstrip("\n"))
            if line.startswith(PROGRESS_PREFIX):
                summary = line[len(PROGRESS_PREFIX):].strip()
                if summary:
                    self._call("progress", summary=summary)
        stream.close()

    def _heartbeats(self, every: float) -> None:
        while not self._closed.wait(every):
            if self._child is not None and self._child.poll() is not None:
                return
            self._call("heartbeat")

    # -- the child ----------------------------------------------------------

    def _spawn(self) -> subprocess.Popen:
        kwargs: dict = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
                        "stdin": subprocess.DEVNULL, "text": True, "bufsize": 1}
        if IS_WINDOWS:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        return subprocess.Popen(self._argv, **kwargs)

    def _signal_group(self, sig: int) -> None:
        child = self._child
        if child is None or child.poll() is not None:
            return
        try:
            if IS_WINDOWS:
                child.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                os.killpg(child.pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def _stop_child(self) -> None:
        child = self._child
        if child is None:
            return
        self._signal_group(signal.SIGTERM)
        try:
            child.wait(self._grace)
            return
        except subprocess.TimeoutExpired:
            pass
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(child.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            self._signal_group(signal.SIGKILL)
        child.wait()

    def _on_signal(self, signum, _frame) -> None:
        self._signalled = signum
        self._signal_group(signum)

    # -- main ---------------------------------------------------------------

    def run(self) -> int:
        job = self._client.inspect(self._job_id)
        self._sequence = job["event_sequence"]
        if job.get("status") not in ("running", "queued", "retry_wait"):
            self._note(f"job is {job.get('status')}; not starting the command")
            return EXIT_JOB_CLOSED
        every = self._heartbeat_every or max(5.0, job["heartbeat_timeout_seconds"] / 3)

        previous = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.signal(sig, self._on_signal)
        try:
            return self._supervise(every)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def _supervise(self, every: float) -> int:
        self._child = self._spawn()
        self._note(f"started pid {self._child.pid}: {' '.join(self._argv)}")
        pumps = [
            threading.Thread(target=self._pump, args=(self._child.stdout, False), daemon=True),
            threading.Thread(target=self._pump, args=(self._child.stderr, True), daemon=True),
        ]
        for thread in pumps:
            thread.start()
        threading.Thread(target=self._heartbeats, args=(every,), daemon=True).start()

        stopped = False
        while self._child.poll() is None:
            if self._closed.wait(0.2) and self._on_close == "stop" and not stopped:
                self._note("stopping the command")
                self._stop_child()
                stopped = True
        code = self._child.wait()
        for thread in pumps:
            thread.join(5)

        if self._signalled is not None:
            self._final("cancel", reason=f"supervisor received signal {self._signalled}")
            return 128 + self._signalled
        if self._closed.is_set():
            return EXIT_JOB_CLOSED if stopped else code
        self._closed.set()  # no more heartbeats
        if code == 0:
            self._final("complete", summary="exit 0")
        else:
            reason = f"exit {code}" if code > 0 else f"killed by signal {-code}"
            tail = "\n".join(self._stderr_tail)
            self._final("fail", reason=reason + (f"\n{tail}" if tail else ""))
        return code


def supervise(
    job_id: str,
    argv: list[str],
    *,
    heartbeat_every: float | None = None,
    on_close: str = "stop",
    grace: float = 10,
    log_path: Path | None = None,
    client_factory: Callable[[], ManagedJobClient] = ManagedJobClient.from_env,
) -> int:
    """Run ``argv`` under job ``job_id`` in the foreground; returns its exit status."""
    log = open(log_path, "a", buffering=1) if log_path else None
    try:
        with client_factory() as client:
            return JobSupervisor(
                client, job_id, argv, heartbeat_every=heartbeat_every,
                on_close=on_close, grace=grace, log=log,
            ).run()
    finally:
        if log:
            log.close()


def spawn_detached(
    job_id: str,
    argv: list[str],
    *,
    log_path: Path,
    heartbeat_every: float | None = None,
    on_close: str = "stop",
    grace: float = 10,
) -> dict:
    """Start the supervisor detached, so it outlives the agent's turn."""
    log_path = log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "clawmeets.runner.job_supervisor", job_id,
           "--on-close", on_close, "--grace", str(grace), "--log", str(log_path)]
    if heartbeat_every:
        cmd += ["--heartbeat-every", str(heartbeat_every)]
    cmd += ["--", *argv]
    with open(log_path, "a") as sink:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=sink, stderr=sink,
            **popen_detached_kwargs(),
        )
    return {"job_id": job_id, "pid": proc.pid, "log": str(log_path)}


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clawmeets job run")
    parser.add_argument("job_id")
    parser.add_argument("--heartbeat-every", type=float)
    parser.add_argument("--on-close", choices=("stop", "keep"), default="stop")
    parser.add_argument("--grace", type=float, default=10)
    parser.add_argument("--log", type=Path)
    args = sys.argv[1:] if args is None else args
    if "--" not in args:
        parser.error("a command is required after --")
    split = args.index("--")
    ns, command = parser.parse_args(args[:split]), args[split + 1:]
    if not command:
        parser.error("a command is required after --")
    return supervise(
        ns.job_id, command, heartbeat_every=ns.heartbeat_every,
        on_close=ns.on_close, grace=ns.grace, log_path=ns.log,
    )


if __name__ == "__main__":
    sys.exit(main())
