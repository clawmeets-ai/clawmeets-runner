# SPDX-License-Identifier: MIT
"""
clawmeets/models/desk_todo_link.py

The link between a desk to-do and the projects it spawned — and the ONE place
a to-do's ticket state is derived.

**Why this is a third module and not a computed field on ``DeskTodo``.**
``models/desk_todo.py`` stores project ids and knows nothing about projects,
exactly as it stores label slugs and knows nothing about the label registry.
This module may import both ``models/desk_todo.py`` and ``models/project.py``;
**neither imports it**. That is the same direction rule that already lets
``models/desk_label.py`` import ``models/desk_todo.py`` and never the reverse,
and it is what keeps the plate loadable by a runner that has no projects
directory at all.

**The two axes, and that they never move each other.** A to-do has a *ticket
state* — New / Working / Completed — which is DERIVED, read-only, and owned
entirely by the association set; and it has ``archived``, a disposal the owner
sets by hand. Nothing here writes ``archived`` and nothing here reads it as
"the work finished". An archived to-do whose project is still running still
derives Working, and it says so on the wire (AC-1.4, AC-1.7).

**No packaging manifest row is needed.** ``scripts/build-runner-package.sh``
rsyncs the whole ``models/`` subtree, so a new module here is wheel-available
for free. The corollary is a constraint this module must respect: it may import
only things that ship in that wheel. In particular it may NOT import
``server/routes/_batch.py``'s ``accessible_projects``, which is BUSL-licensed
server code — such an import installs cleanly and crashes on first use on every
released runner. That is why :func:`check_association_target` takes an
already-resolved project rather than resolving one itself.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from clawmeets.models.desk_todo import DeskTodo, list_todos
from clawmeets.models.project import Project
from clawmeets.sync.changelog import ProjectStatus

if TYPE_CHECKING:
    from clawmeets.models.context import ModelContext

logger = logging.getLogger("clawmeets.models.desk_todo_link")

# The derived ticket state, as lowercase slugs. Slugs and not display strings:
# the UI owns the words "Working" and "Completed", and the terminal and the
# browser must agree on the value rather than on the copy (AC-4.2).
TICKET_STATES: tuple[str, ...] = ("new", "working", "completed")

STATE_NEW = "new"
STATE_WORKING = "working"
STATE_COMPLETED = "completed"

# The two refusal sentences for the association surface, written ONCE here so a
# sentence cannot be phrased one way at one call site and another way at the
# next — the same discipline ``LabelError`` already applies to the coded label
# strings. They are BARE sentences with no ``<namespace>.<code>:`` prefix, and
# that is a rule rather than a preference: the ``labels.*`` prefix exists so the
# browser can dispatch on the code and rewrite the copy using context the server
# lacks, and neither of these has that shape — the 404 already names the id the
# owner typed and the 403 already names the project. A code with nothing to
# dispatch on is just a string the client has to strip before showing. Every
# existing refusal in ``server/routes/desk_todos.py`` is bare for the same
# reason.


class AssociationRefused(Exception):
    """An association target the owner may not link to.

    Carries the HTTP ``status`` and the exact ``sentence`` the route answers
    with, so the decision and its copy live together and the route is left with
    ``raise HTTPException(e.status, e.sentence)``.
    """

    def __init__(self, status: int, sentence: str) -> None:
        super().__init__(sentence)
        self.status = status
        self.sentence = sentence


def _project_label(project: "Project") -> str:
    """What to call a project in a sentence the owner reads.

    ``display_name`` is the human label and is ``None`` only on legacy rows, so
    ``name`` (the slug) is the fallback — the same rule every other
    owner-facing surface applies.
    """
    return project.display_name or project.name


def check_association_target(
    project: "Project | None",
    owner_user_id: str,
    requested_id: str,
) -> "Project":
    """Decide whether ``owner_user_id`` may associate ``requested_id``, or
    raise :class:`AssociationRefused` carrying the sentence to answer with.

    The caller supplies the resolution and the visibility check — ``project`` is
    the resolved project **only if the owner may see it**, and ``None``
    otherwise. That split is not squeamishness about I/O: the visibility
    predicate lives in ``server/routes/_batch.py``, which is not shipped in the
    runner wheel that carries this module (see the module docstring).

    Two refusals, and NEITHER is generic (AC-3.5):

    * **404** — the id resolves to nothing, OR it resolves but this owner
      cannot see it. Deliberately ONE sentence covering both: splitting them
      would make the route an existence oracle for other people's project ids,
      which is the same reason the to-do attachment download route answers 404
      rather than 403.
    * **403** — the owner can see it (it was shared to them as a viewer) but
      did not create it.

    The predicate is ``created_by == user.id`` — **the rename predicate, not
    the wider "can see it" predicate.** That scoping is load-bearing well past
    authorization: it is what keeps "which to-dos does this project affect?"
    answerable from one file read with no reverse index, because the only plate
    that can reference a project is its creator's.

    A DM-thread id passes through untouched and is never refused as "not a
    project": ``_create_own_dm_thread_project`` writes ``created_by =
    user.id``, so an own-DM thread satisfies the predicate by construction.
    That matters because the desk's **command** button spawns a DM thread and
    it is the feature's most common creation path (AC-3.3).
    """
    if project is None:
        raise AssociationRefused(
            404,
            f"No project or thread with id {requested_id!r} — check the id, "
            "or it may have been deleted.",
        )
    if project.created_by != owner_user_id:
        raise AssociationRefused(
            403,
            f"{_project_label(project)!r} belongs to someone else — you can "
            "only link to-dos to projects and threads you created.",
        )
    return project


def _status_memo(
    projects: dict[str, "Project"],
) -> dict[str, ProjectStatus]:
    """Read every resolved project's status EXACTLY ONCE.

    ``Project.status`` is a ``@computed_field`` that re-reads ``meta.json`` on
    every attribute access. Reading it inside the per-row loop would cost one
    metadata read per (to-do, project) pair, so a project associated to three
    to-dos would cost three reads of the same file. This is a REQUIREMENT of
    the design, not an optimisation. The memo's lifetime is one call.

    A project whose ``meta.json`` cannot be read is treated as unresolvable —
    i.e. it stops contributing, exactly as a deleted one does — rather than
    failing the owner's whole plate fetch over one bad file.
    """
    memo: dict[str, ProjectStatus] = {}
    for pid, project in projects.items():
        try:
            memo[pid] = project.status
        except Exception as e:  # pragma: no cover - filesystem/meta edge
            logger.warning(f"Could not read status for project {pid!r}: {e}")
    return memo


def derive_plate(
    todos: list[DeskTodo], ctx: "ModelContext"
) -> dict[str, tuple[str, int]]:
    """Derive ``(state, association_count)`` for every to-do in ONE pass.

    Returns ``{todo_id: (state, association_count)}``.

    **The rule is three branches with NO precedence order**, over the
    CONTRIBUTING associations only::

        every contributing association COMPLETED  -> "completed"
        any contributing association not complete -> "working"
        no contributing associations at all       -> "new"

    Non-contributing, and identical in effect to a dangling id: a project that
    did not resolve (deleted), and a project whose status is **FAILED**. That
    is the reference-not-a-foreign-key property the whole design rests on, and
    it is what makes AC-1.8 true — a to-do whose only project failed reads New,
    not Working forever with no way out. One completed project is enough when
    it is the only one; the rule is "every contributing association is
    complete", never "more than one".

    Exactly one value comes back per to-do, which is what AC-1.5 is tested
    against. (There is no server-side at-most-one-state rule in the product
    today and never was — ``clawmeets todo update --label next --label someday``
    succeeds right now — so AC-1.5 asserts this function's return, not a
    regression against an invariant that never existed.)

    ``association_count`` is ``len(todo.project_ids)`` — the **STORED** count,
    whether or not the ids still resolve. Deliberately not the resolved count:
    a row reading ``(association_count=2, state="new")`` is how a dangling
    association becomes legible to its owner, where a resolved count would
    report 0, agree with New, and hide it.

    TWO PERFORMANCE PROPERTIES THAT ARE REQUIREMENTS, NOT OPTIMISATIONS:

    1. **One batched** ``Project.get_many`` over the union of every row's ids.
       Single-id resolution globs the metadata dir, O(P) per id — a measured
       850 ms for 50 ids against 20,000 projects, and the reference box already
       carries 539. Batched is one ``scandir``. A 30-row plate with 2 ids each
       would otherwise cost 60 globs on EVERY desk fetch, and that fetch is
       hot: the desk refetches on both the to-do sync and the label sync.
    2. **Status is memoized per call** (:func:`_status_memo`), so one project
       associated to three to-dos costs one ``meta.json`` read.

    A plate whose rows carry no ids at all short-circuits before touching disk.
    """
    counts = {t.id: len(t.project_ids) for t in todos}
    wanted = {pid for t in todos for pid in t.project_ids}
    if not wanted:
        # Nothing to resolve — no scandir, no meta.json read, no disk at all.
        return {tid: (STATE_NEW, n) for tid, n in counts.items()}

    resolved = Project.get_many(wanted, ctx)
    statuses = _status_memo(resolved)

    out: dict[str, tuple[str, int]] = {}
    for todo in todos:
        contributing = [
            statuses[pid]
            for pid in todo.project_ids
            # Absent from `statuses` == deleted, unreadable, or dangling.
            if statuses.get(pid) is not None
            and statuses[pid] is not ProjectStatus.FAILED
        ]
        if not contributing:
            state = STATE_NEW
        elif all(s is ProjectStatus.COMPLETED for s in contributing):
            state = STATE_COMPLETED
        else:
            state = STATE_WORKING
        out[todo.id] = (state, counts[todo.id])
    return out


def as_wire(todos: list[DeskTodo], ctx: "ModelContext") -> list[dict]:
    """The wire shape: ``todo.model_dump()`` plus ``state`` and
    ``association_count``.

    **Every** to-do-returning route goes through this instead of
    ``.model_dump()`` — LIST, CREATE, PATCH, PUBLISH and REORDER alike. The
    browser resolves records from all of them, so a create or a patch response
    missing ``state`` makes the row flicker to undefined until the next full
    fetch. That includes a bare archive toggle that changes no association:
    one batched lookup over that row's ids, not zero.

    ``state`` is typed-required and never null. A single-row call is
    ``as_wire([t], ctx)[0]``; when that row carries no ids the derivation
    short-circuits and touches no disk.
    """
    derived = derive_plate(todos, ctx)
    rows: list[dict] = []
    for todo in todos:
        state, count = derived[todo.id]
        row = todo.model_dump()
        row["state"] = state
        row["association_count"] = count
        rows.append(row)
    return rows


def resolve_associations(todo: DeskTodo, ctx: "ModelContext") -> list[dict]:
    """One to-do's associations, resolved, for ``GET
    /me/desk/todos/{id}/projects``.

    A DM thread appears on exactly the same footing as a regular project — same
    id space, same row shape. ``is_dm`` is DESCRIPTIVE and never a filter:
    nothing here decides what may be listed or associated on the strength of it.

    Row shape, and the split matters because the client types against it:

    * **ALWAYS PRESENT, on every row**: ``{"id", "resolved", "contributes"}``.
      The three are total so the client can type them required and never has to
      read an ABSENT ``contributes`` as falsy — which would work right up until
      a genuine ``contributes: false`` arrived from a resolved-but-FAILED
      project, at which point two different wire shapes would mean the same
      thing.
    * **PRESENT ONLY WHEN** ``resolved``: ``{"display_name", "name", "status",
      "surface", "is_dm"}`` — **omitted** on an unresolved row, never emitted
      as ``null``.

    ``status`` is the closed three-value ``ProjectStatus``
    (``"active" | "completed" | "failed"``); ``models/project.py`` states there
    will deliberately never be a fourth member. ``is_dm`` is
    ``Project.is_dm_shaped``, which is TRUE for ``surface: "dm"`` **and**
    ``"frontdesk"`` — so it is not ``surface == "dm"`` and must not be derived
    from ``surface``, or every Front Desk thread reads wrong.

    An id that does not resolve is still emitted, as ``{"id": ...,
    "resolved": false, "contributes": false}``: a dangling association the
    owner cannot see is one they cannot remove either, and ``dissociate`` is
    the only way out.
    """
    if not todo.project_ids:
        return []
    resolved = Project.get_many(todo.project_ids, ctx)
    statuses = _status_memo(resolved)

    rows: list[dict] = []
    for pid in todo.project_ids:
        project = resolved.get(pid)
        status = statuses.get(pid)
        if project is None or status is None:
            rows.append({"id": pid, "resolved": False, "contributes": False})
            continue
        rows.append(
            {
                "id": pid,
                "resolved": True,
                "contributes": status is not ProjectStatus.FAILED,
                "display_name": _project_label(project),
                "name": project.name,
                "status": status.value,
                "surface": project.surface,
                "is_dm": project.is_dm_shaped,
            }
        )
    return rows


def todos_referencing_project(
    data_dir: Path, owner_user_id: str, project_id: str
) -> list[str]:
    """THE reverse lookup: which of this owner's to-dos carry ``project_id``.

    Returns the matching to-do ids in plate order.

    One file read and an in-memory scan. No reverse index exists and none is
    needed — and that holds PRECISELY because association is scoped to projects
    the owner created (:func:`check_association_target`), so there is exactly
    one plate that could possibly reference a given project. It would stop
    holding the moment a shared project became associable, which is one more
    reason that scoping is the rename predicate rather than the wider
    can-see-it one.

    Two callers, one scan. :func:`plate_references_project` is this function
    asked a yes/no question, and it is written in terms of it rather than
    beside it: two scans of the same document that must agree about what
    "references" means is exactly the pair that drifts, and the drift would be
    invisible — each would keep passing its own tests.

    The id is matched VERBATIM against what is stored. Callers holding a
    ``{name}-{id}`` slug must resolve it through ``Project.get`` and pass
    ``project.id``; this module knows nothing about projects and cannot do it
    for them.
    """
    return [
        t.id
        for t in list_todos(data_dir, owner_user_id)
        if project_id in t.project_ids
    ]


def plate_references_project(
    data_dir: Path, owner_user_id: str, project_id: str
) -> bool:
    """Does this owner's plate reference this project at all?

    Used by the project-event notifier to answer "is this event worth an
    envelope?" without loading anything else. It is
    :func:`todos_referencing_project` with the ids thrown away — see there for
    why there is only one scan.
    """
    return bool(todos_referencing_project(data_dir, owner_user_id, project_id))
