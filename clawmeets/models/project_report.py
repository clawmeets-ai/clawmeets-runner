# SPDX-License-Identifier: MIT
"""
clawmeets/models/project_report.py

Project-scoped interactive report — one artifact per project, authored
by the coordinator at wrap-up and rendered in the project detail view.

Reuses the self-serve tab publishing protocol (``data`` JSON +
``render_code_js`` body executed in the browser via
``new Function('mount', 'data', 'lib', ...)`` with the same ``lib``
namespace as brief-tabs). What differs is the scope: a report is keyed
by the project, not the user, and surfaces inside the project's own UI.

Storage::

    {data_dir}/metadata/projects/<name>-<id>/report.json

One file per project — overwrite on re-publish is the contract.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from clawmeets.utils.file_io import FileUtil

_lock = asyncio.Lock()


class ProjectReport(BaseModel):
    """A single project's interactive report.

    ``data`` is opaque JSON the render body consumes. ``render_code_js``
    is the body of ``function(mount, data, lib)`` — no signature, no
    surrounding braces. The frontend never edits either.
    """

    project_id: str
    title: str = ""
    generated_by_agent_id: str
    generated_by_agent_name: str
    data: dict | list = Field(default_factory=dict)
    render_code_js: str = ""
    generated_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _report_path(metadata_project_dir: Path) -> Path:
    return Path(metadata_project_dir) / "report.json"


def _load_report(path: Path) -> ProjectReport | None:
    data = FileUtil.read(path, "json")
    if not isinstance(data, dict):
        return None
    try:
        return ProjectReport.model_validate(data)
    except Exception:
        return None


def get_report(metadata_project_dir: Path) -> ProjectReport | None:
    return _load_report(_report_path(metadata_project_dir))


async def upsert_report(
    metadata_project_dir: Path,
    project_id: str,
    generated_by_agent_id: str,
    generated_by_agent_name: str,
    title: str,
    data: dict | list,
    render_code_js: str,
) -> ProjectReport:
    """Create or replace the project's report. Always succeeds —
    overwrite is the contract (the coordinator owns it and re-runs on
    every refresh)."""
    async with _lock:
        report = ProjectReport(
            project_id=project_id,
            title=(title or "").strip(),
            generated_by_agent_id=generated_by_agent_id,
            generated_by_agent_name=generated_by_agent_name,
            data=data,
            render_code_js=render_code_js,
            generated_at=_now(),
        )
        FileUtil.write(
            _report_path(metadata_project_dir),
            report.model_dump(),
            "json",
        )
        return report


async def delete_report(metadata_project_dir: Path) -> bool:
    """Delete the project's report. Returns True if it existed."""
    async with _lock:
        path = _report_path(metadata_project_dir)
        if not path.is_file():
            return False
        path.unlink()
        return True


def backfill_report_published_at(model_ctx) -> int:
    """One-shot idempotent reconcile of ``Project.report_published_at`` against
    the on-disk ``report.json`` for every project. Run once at server startup.

    For each project: if ``report.json`` exists, stamp meta.json's
    ``report_published_at`` from its ``generated_at``; otherwise clear it.
    Delegates to :meth:`ProjectState.set_report_published_at`, which writes only
    on a real delta — so this is safe to run on every boot and touches only
    projects whose meta.json is out of sync with their report. Returns the number
    of projects rewritten (logged once at startup).

    Why a backfill exists: projects with a ``report.json`` predating the new meta
    key would serialize ``report_published_at = null`` until their next
    re-publish. This makes the field correct from first deploy without waiting on
    a re-publish, and re-derives the value purely from the source of truth
    (``report.json``), so it can never diverge from ``getProjectReport``.

    Persistence-safety: ``report_published_at`` lives in meta.json OUTSIDE the
    changelog, consistent with ``report.json`` itself being a non-changelog
    artifact. Every ``ProjectState`` mutation is a read-modify-write of the full
    meta dict (``complete`` / ``touch_activity`` / ``add_participant`` / …), which
    preserves this key; the only wholesale rewrite, ``ProjectState.create()``,
    runs at ``PROJECT_CREATED`` — before any report can exist. So no existing
    write path clobbers it.
    """
    from .project import Project  # local import: avoid any import-time coupling

    count = 0
    for project in Project.list_all(model_ctx):
        report = get_report(project.meta_dir)
        desired = report.generated_at if report is not None else None
        if project.state().set_report_published_at(desired):
            count += 1
    return count
