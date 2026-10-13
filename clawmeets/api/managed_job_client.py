# SPDX-License-Identifier: MIT
"""HTTP client for the managed background-job API (``/jobs``).

Shared by ``clawmeets job …`` and the ``job run`` supervisor
(``runner/job_supervisor.py``), so the request code exists once.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime

import httpx

from clawmeets.api.retry_transport import SyncRetryingTransport

#: Job statuses after which nothing should keep working for the job.
CLOSED_STATUSES = {"stalled", "succeeded", "failed", "cancelled"}


class JobApiError(RuntimeError):
    """The server refused a job request."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"Error ({status_code}): {detail}")
        self.status_code = status_code
        self.detail = detail


class JobClosed(JobApiError):
    """The job is no longer live (stalled, past its deadline, or ended)."""

    def __init__(self, job: dict) -> None:
        reason = job.get("failure_reason") or job.get("status")
        super().__init__(409, f"job is {job.get('status')}: {reason}")
        self.job = job


def is_closed(job: dict) -> bool:
    if job.get("status") in CLOSED_STATUSES:
        return True
    deadline = job.get("deadline_at")
    return bool(deadline) and datetime.fromisoformat(deadline) <= datetime.now(UTC)


class ManagedJobClient:
    """Authenticated as the agent from ``CLAWMEETS_SERVER_URL`` /
    ``CLAWMEETS_AGENT_TOKEN`` / ``CLAWMEETS_AGENT_ID``."""

    def __init__(
        self,
        server_url: str,
        token: str,
        agent_id: str,
        *,
        timeout: float = 30,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(
            base_url=server_url.rstrip("/"),
            timeout=timeout,
            transport=transport or SyncRetryingTransport(),
            headers={"Authorization": f"Bearer {token}", "X-Agent-ID": agent_id},
        )

    @classmethod
    def from_env(cls) -> "ManagedJobClient":
        names = ("CLAWMEETS_SERVER_URL", "CLAWMEETS_AGENT_TOKEN", "CLAWMEETS_AGENT_ID")
        values = [os.environ.get(name, "").strip() for name in names]
        missing = [name for name, value in zip(names, values) if not value]
        if missing:
            raise JobApiError(0, f"${missing[0]} is not set")
        return cls(*values)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ManagedJobClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def _ok(response: httpx.Response) -> dict:
        if response.is_success:
            return response.json()
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise JobApiError(response.status_code, str(detail))

    def start(self, **fields) -> dict:
        body = {key: value for key, value in fields.items() if value is not None}
        return self._ok(self._http.post("/jobs", json=body))

    def inspect(self, job_id: str) -> dict:
        return self._ok(self._http.get(f"/jobs/{job_id}"))

    def event(
        self, job_id: str, op: str, *, expected_sequence: int,
        retry_stale: bool = True, **fields,
    ) -> dict:
        """POST ``/jobs/{id}/{op}``. A stale sequence is re-read and retried
        once unless ``retry_stale`` is False. Raises ``JobClosed`` when the job is no longer live, including
        a heartbeat answered with a stalled job."""
        body = {"expected_sequence": expected_sequence, **fields}
        for attempt in (1, 2):
            response = self._http.post(f"/jobs/{job_id}/{op}", json=body)
            if response.is_success:
                job = response.json()
                if op in ("heartbeat", "progress") and is_closed(job):
                    raise JobClosed(job)
                return job
            if response.status_code != 409:
                self._ok(response)
            current = self.inspect(job_id)
            if is_closed(current):
                raise JobClosed(current)
            if retry_stale and attempt == 1 and "stale sequence" in response.text:
                body["expected_sequence"] = current["event_sequence"]
                continue
            self._ok(response)
        raise AssertionError("unreachable")
