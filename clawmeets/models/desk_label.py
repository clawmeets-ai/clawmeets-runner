# SPDX-License-Identifier: MIT
"""
clawmeets/models/desk_label.py

The owner's label registry — presentation for the bare slugs stored on
``DeskTodo.labels``. One JSON document per user::

    {data_dir}/desk-labels/<owner_user_id>.json
    {"version": 1, "labels": [{slug, name, kind, color, created_at, updated_at}, ...]}

Array position IS the ordering. There is no ``order`` field and no ``revision``
counter (both ruled declined) — ``reorder_labels`` rewrites the array and every
other verb preserves position.

**The registry is presentation, NOT a foreign key.** A slug on a to-do with no
row here is valid and renders unregistered; the plate is fully readable with
this document missing. That is why this module imports from ``desk_todo`` and
``desk_todo`` never imports from here — the property is enforced by the import
graph rather than merely documented. If you ever need the reverse edge, the
property has quietly stopped being true and that is the thing to fix.

The seed is a **first-write materialization only** and is never re-asserted. An
implementation that re-adds missing defaults on read is wrong, and it is wrong
in the way that silently resurrects ``home`` every time the owner deletes it.

**THE ``state`` KIND IS RETIRED, AND IT IS RETIRED BY READING, NOT BY WRITING.**
Lifecycle now lives on ``DeskTodo.state``, derived from the projects linked to
an item (``models/desk_todo_link.py``); a label can no longer mean "the work is
running". ``KINDS`` is down to one member and nothing new can be stored as a
state — but owners have ``kind: "state"`` rows on disk today, and **not one byte
of them is destroyed**. The row keeps its slug, its name, its colour and its
position, and every to-do carrying it keeps carrying it. It simply stops
carrying lifecycle meaning, and it reads as a context.

That retirement is a projection applied on the READ path and nowhere else, which
is the whole reason ``_rows_raw`` and ``_rows`` are two functions:

  * ``_rows_raw``  — today's parse, verbatim. The STORED kind. Every WRITE
    starts here (``_rows_for_write``, ``auto_register``), so a read-modify-write
    round-trips a legacy row's ``kind`` unchanged.
  * ``_rows``      — raw, plus the coercion. What ``list_labels`` answers.

Coercing inside the write path instead would persist ``kind: "context"`` over a
row the owner stored as ``state`` — a silent rewrite of their data on the next
unrelated recolour — and it would blind the kind guards in ``create_label`` and
``merge_label``, which read the stored kind and are what keep a legacy row from
being shadowed or folded into a context. There is **no migration pass, no
schema-version bump and no bulk write**; a registry holding a state row is
byte-identical after any number of reads.

**A WRITE NEVER STARTS FROM AN UNREADABLE FILE.** Every verb here is a
read-modify-write of the whole array, so "the file exists and will not parse"
must not be allowed to look like "the registry is empty" — it once did, and one
``create_label`` turned twelve rows into one. The predicate is the PARSE
OUTCOME (``_Registry.readable``), NEVER ``rows == []``: an empty-but-valid
registry is the ordinary state after the owner deletes their last label, and it
must keep accepting writes. The six curation verbs refuse
(``labels.registry_unreadable``); ``auto_register`` skips. See
``_read`` / ``_rows_for_write``.

Mutations are broadcast to the owner via ``DESK_LABEL_SYNC`` (see
``server/routes/desk_labels.py``), mirroring ``DESK_SOP_SYNC``.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel

from clawmeets.models import desk_todo
from clawmeets.models.desk_todo import (
    DeskTodo,
    LabelError,
    normalize_label,
)
from clawmeets.utils.file_io import FileUtil

LABELS_DIR = "desk-labels"

# The mock's PALETTE constant, verbatim and in order. Order is load-bearing
# twice: it is the tie-break in ``default_color`` and it is the contrast-checked
# sequence @designer authored, so "earliest free" also means "most likely to be
# visually distinct from what is already on screen".
PALETTE = ("blue", "green", "indigo", "teal", "amber", "pink", "slate", "plum")

# The kind vocabulary, down to ONE member. Collapsing the tuple is the entire
# enforcement of the retirement on the write side: ``_validate_kind`` already
# refuses anything outside it with ``labels.invalid_kind``, so no new code path
# and no new error code are needed. The CODE STRING does not change — it is
# published in the to-do data contract and the browser maps it at
# ``utils/labelErrors.ts``; only the sentence changes, to say the axis is gone
# rather than to recite an enum that now has one member.
KINDS = ("context",)

# What a stored row may say that ``KINDS`` no longer accepts. This is a READ
# vocabulary, never a write one: rows carrying it are shown as ``context`` by
# ``_rows`` and are left exactly as they are on disk by ``_rows_raw``. Naming it
# is what keeps the coercion a deliberate one-value projection rather than an
# "anything unknown becomes a context" rule, which would quietly repair a
# corrupt row and hide a real bug.
RETIRED_KINDS = ("state",)

MAX_REGISTRY_ROWS = 64
MAX_NAME_LEN = 48
SCHEMA_VERSION = 1

# The owner's starting vocabulary. Three rows, ALL deletable — there is no
# ``locked`` field and no guard anywhere in this module protecting a row. A
# helpful start, not a fixture.
#
# It was six: ``next`` / ``wait-for`` / ``someday`` seeded the state axis, and
# they are gone with it — a new owner is not handed three labels whose whole
# purpose was to say what ``state`` now derives. THIS IS NOT A MIGRATION AND IT
# TOUCHES NOBODY'S REGISTRY. The seed is a first-write materialization, so it
# only ever decides what an owner who has never written a registry sees; an
# owner who already has those three rows keeps all six, position and colour
# intact, and ``list_labels`` shows them as contexts. Nothing re-asserts and
# nothing removes.
# The seed's timestamps are a FIXED constant, not the clock. Two reasons, and
# the first is the one that bit: the seed is materialized in memory on every
# read of an unwritten registry, so minting `now()` there made two consecutive
# GETs return different `created_at` values for the same row — churn a client
# could legitimately notice, and non-determinism in a value nothing needs. The
# second is that it is simply more truthful: the owner did not create these,
# so they carry the epoch rather than a moment that pretends they did. Rows
# keep the constant when the first write materializes them to disk; anything
# the owner creates afterwards is stamped with the real clock and sorts after.
SEED_TIMESTAMP = "1970-01-01T00:00:00+00:00"

SEED: tuple[dict[str, str], ...] = (
    {"slug": "office", "name": "office", "kind": "context", "color": "blue"},
    {"slug": "home", "name": "home", "kind": "context", "color": "green"},
    {"slug": "phone", "name": "phone", "kind": "context", "color": "teal"},
)

# INNER lock. ``desk_todo.plate_lock`` is OUTER and the order is never reversed
# — see the header comment on that module. Registry-only writes take this one
# alone; ``delete_label`` and ``merge_label`` take the plate lock first.
_registry_lock = asyncio.Lock()

_UNSET: object = object()


def _display_name(raw: str) -> str:
    """A display name with its leading sigils stripped.

    The `@` is chrome the renderer prepends, never data — storing one makes the
    same character sometimes data and sometimes decoration. Mirrors the mock's
    `raw.replace(/^[@!]+/, '')`.

    `!` is still stripped even though the axis it marked is retired, and it
    keeps being stripped: owners and assistants will go on typing `!next` at a
    label that has been on the plate for a year, and the alternative is a label
    literally named `!next` sitting beside the real one.
    """
    return raw.strip().lstrip("@!").strip()


class DeskLabel(BaseModel):
    """One registry row.

    ``slug`` is identity and is NEVER rewritten by any verb here. A true slug
    change is expressed as a merge into a newly created slug, which is honest
    about being a plate rewrite rather than a rename.
    """

    slug: str
    name: str
    kind: str = "context"
    color: str | None = None
    created_at: str
    updated_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _path(data_dir: Path, owner_user_id: str) -> Path:
    return Path(data_dir) / LABELS_DIR / f"{owner_user_id}.json"


class _Registry(NamedTuple):
    """ONE read of the registry file: the document, and whether it was READABLE.

    ``readable`` is the write predicate, and the whole point is that it is NOT
    ``rows == []``. Three states, and the middle two are the ones that collapse
    into each other if you decide on the row count:

      * no file                  -> readable, document None -> the seed
      * ``{"labels": []}``       -> readable, zero rows      -> the ordinary
        state after the owner deletes their last label. Auto-registration MUST
        keep working here or the next to-do they label silently gets nothing.
      * exists, cannot be read   -> NOT readable, zero rows  -> a write that
        starts from ``[]`` replaces rows nobody has seen.

    ``readable`` asks one question: *can we account for every row currently in
    the file?* A JSON object whose ``labels`` is absent or a list, yes — even
    when some rows inside it are individually corrupt, because ``_rows`` skips
    those one at a time and the ones it keeps are the ones we would rewrite.
    Anything else — truncated bytes, a top-level list, ``labels`` holding a
    string — no.
    """

    document: dict | None
    readable: bool


def _read(data_dir: Path, owner_user_id: str) -> _Registry:
    """The single read. Every caller in this module goes through it, so the
    missing / empty / unreadable distinction is drawn in exactly one place and
    the file is parsed exactly once per operation.

    THE TRAP THIS FUNCTION EXISTS TO AVOID. ``FileUtil.read(..., default=...)``
    returns the default on a ``JSONDecodeError`` as well as on a missing file
    (``utils/file_io.py``), so deciding "seed?" on the parsed value RE-SEEDS a
    corrupt registry — which the contract forbids in as many words, and which
    would resurrect ``home`` the first time a half-written file is read. The
    seed decision is therefore made on ``path.exists()`` and on nothing else.
    """
    path = _path(data_dir, owner_user_id)
    if not path.exists():
        return _Registry(None, True)
    raw = FileUtil.read(path, "json", default=None)
    if not isinstance(raw, dict) or not isinstance(raw.get("labels", []), list):
        return _Registry({}, False)
    return _Registry(raw, True)


def _document_or_none(data_dir: Path, owner_user_id: str) -> dict | None:
    """The raw registry document, or ``None`` when NO FILE EXISTS.

    Unchanged for every reader:

      * no file            -> None -> seedable
      * file, unparseable  -> {}   -> present and authoritative -> zero rows
      * file, parsed       -> dict -> authoritative

    Missing is not empty, and unparseable is not missing either. What this
    return value CANNOT say is which of the last two you got — both are a dict
    with no usable rows — so every WRITE reads through ``_read`` /
    ``_rows_for_write`` instead. Reads are happy without the distinction; a
    write that lacks it destroys the file.
    """
    return _read(data_dir, owner_user_id).document


def _seed_rows() -> list[DeskLabel]:
    return [
        DeskLabel(created_at=SEED_TIMESTAMP, updated_at=SEED_TIMESTAMP, **row)
        for row in SEED
    ]


def _rows_raw(document: dict | None) -> list[DeskLabel]:
    """Parse rows one at a time inside try/except, mirroring ``_load`` in
    ``desk_todo.py``: a corrupt, truncated or hand-edited row is skipped, never
    raised.

    ``None`` (no file) materializes the seed in memory. A document that exists
    but parses to nothing yields ZERO rows — never the seed as a repair.

    THE STORED KIND, UNPROJECTED. This is what every WRITE starts from, so a
    read-modify-write puts a legacy ``kind: "state"`` row back exactly as it
    found it. Readers want ``_rows`` instead; the difference is the retirement
    and it is explained at the top of this module.
    """
    if document is None:
        return _seed_rows()
    raw = document.get("labels")
    if not isinstance(raw, list):
        return []
    out: list[DeskLabel] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            out.append(DeskLabel.model_validate(row))
        except Exception:
            continue
    return out


def _visible(row: DeskLabel) -> DeskLabel:
    """The row as a READER is shown it — the projection applied to ONE row.

    The mutation verbs return through this while saving the raw row, so that a
    kind the GET would never show cannot come back on a PATCH / DELETE / merge
    response either. Without it the wire contradicts itself: the same legacy row
    reads ``"context"`` on the list and ``"state"`` on the response to a
    recolour, and a client that caches what it just wrote ends up holding a
    value the server refuses to accept back.
    """
    return row.model_copy(update={"kind": "context"}) if row.kind in RETIRED_KINDS else row


def visible_kind(row: DeskLabel) -> str:
    """The kind a READER is told, which is ``"context"`` for a retired state.

    One value in, one value out, and the only input it rewrites is a member of
    ``RETIRED_KINDS`` — a row carrying anything else is returned untouched so a
    genuinely corrupt kind stays visible as itself instead of being laundered.
    """
    return "context" if row.kind in RETIRED_KINDS else row.kind


def _rows(document: dict | None) -> list[DeskLabel]:
    """``_rows_raw`` with the retirement projected onto it — the READ path.

    A stored ``kind: "state"`` row is answered as a context. Nothing else about
    it moves: same slug, same name, same colour, same position in the array, and
    the copy is made with ``model_copy`` so the projection cannot leak back into
    a row a caller might later hand to ``_save``.
    """
    return [_visible(row) for row in _rows_raw(document)]


def _rows_for_write(data_dir: Path, owner_user_id: str) -> list[DeskLabel]:
    """The rows a DIRECT registry mutation is allowed to start from.

    THE DEFECT THIS EXISTS TO CLOSE. ``_rows`` of an unreadable document is
    ``[]``, and every verb here reads-modifies-writes the whole array — so
    ``create_label`` against a registry whose file will not parse appended one
    row to nothing and ``_save``d it. Twelve labels became one, and the original
    bytes were gone.

    So the six curation verbs refuse rather than rewrite. The owner is already
    being told on screen that their labels are still on their to-dos and only
    the LIST failed to load; silently making that a lie is the one outcome the
    banner cannot survive. In practice the browser never reaches this — the UI
    shuts every registry write under an orphaned registry — so it surfaces to
    ``cli_todo.py`` and the two desk-todo skills, which is exactly who could
    otherwise destroy the file with no UI in front of them.

    ``auto_register`` deliberately does NOT come through here: it skips instead
    of raising, because the to-do write has already succeeded. It parses through
    ``_rows_raw`` directly, for the same reason this does.

    ``_rows_raw``, NOT ``_rows``: every verb downstream of this is a
    read-modify-write of the whole array, so a row that arrived here with the
    retirement already projected onto it would be ``_save``d as a context and
    the owner's stored ``state`` would be gone — destroyed by a recolour they
    asked for on a different row. The stored kind goes out and comes back.
    """
    registry = _read(data_dir, owner_user_id)
    if not registry.readable:
        raise LabelError.registry_unreadable()
    return _rows_raw(registry.document)


def _save(data_dir: Path, owner_user_id: str, rows: list[DeskLabel]) -> None:
    FileUtil.write(
        _path(data_dir, owner_user_id),
        {"version": SCHEMA_VERSION, "labels": [r.model_dump() for r in rows]},
        "json",
    )


def _find(rows: list[DeskLabel], slug: str) -> DeskLabel | None:
    for row in rows:
        if row.slug == slug:
            return row
    return None


def _validate_kind(kind: object) -> str:
    """The strict wire enum. A forgiving parser lives at the CLI boundary; the
    wire stays strict, because the inverse is how a third kind value gets stored
    by accident — and the kind is immutable once it lands.

    ``KINDS`` now holds one member, so this is also the whole of the retirement
    on the write side: ``kind: "state"`` arrives here and leaves as
    ``labels.invalid_kind``, with no new branch and no new code. Stored state
    rows are unaffected — they never come through here.
    """
    if not isinstance(kind, str) or kind not in KINDS:
        raise LabelError.invalid_kind()
    return kind


def _validate_name(name: str) -> str:
    if len(name) > MAX_NAME_LEN:
        raise LabelError.name_too_long(MAX_NAME_LEN)
    return name


def _validate_color(color: str | None) -> str | None:
    if color is not None and color not in PALETTE:
        raise LabelError.unknown_color(color, PALETTE)
    return color


def default_color(rows: list[DeskLabel]) -> str:
    """The default colour for a NEW label: the LEAST-USED palette token, ties
    broken by ``PALETTE`` order (first wins).

    NOT ``PALETTE[len(rows) % 8]``, which is wrong in the ordinary case rather
    than the exotic one: delete one label, create another, and the count lands
    back on a token already in use while two sit free. Least-used maximises the
    distance to the first repeat and then spreads the unavoidable repeats evenly
    past eight labels.

    Rows carrying ``None`` or an unrecognised token contribute to no count.
    Pure and deterministic — no clock, no randomness — which is what makes
    asserting the exact "pink, then plum, then blue" sequence possible at all.

    Strict ``<``, not ``<=``: a tie keeps the EARLIER token, matching the mock's
    own ``reduce``. If the two disagree, the two surfaces disagree about the
    same registry.
    """
    used = {token: 0 for token in PALETTE}
    for row in rows:
        if row.color in used:
            used[row.color] += 1
    # ``min`` scans PALETTE in order and keeps the FIRST minimum, which is the
    # tie-break: earliest free token wins.
    return min(PALETTE, key=used.__getitem__)


def list_labels(data_dir: Path, owner_user_id: str) -> list[DeskLabel]:
    """The owner's registry in stored order.

    No file -> the SEED, materialized in memory and NOT written to disk (a read
    never creates the file). A file holding ``{"labels": []}`` -> an empty list,
    and it stays empty forever: the seed is a first-write materialization, never
    re-asserted.

    THE ONE PROJECTION: a stored ``kind: "state"`` row is answered as a context
    (``_rows``). That is the whole of the state axis's retirement — nothing is
    deleted, nothing is rewritten, and the next read of the same untouched file
    returns the same thing. A caller that needs to know what is actually on disk
    wants ``_rows_raw``, and the only callers that need it are the ones that
    write.

    Never raises on content, so "the client has no registry" narrows to a
    genuine transport or auth failure. Never returns per-label counts — the
    client already holds the plate and can count locally; coupling the two
    documents on the read path buys nothing and goes stale on the next
    ``DESK_TODO_SYNC``.
    """
    return _rows(_document_or_none(data_dir, owner_user_id))


def get_label(data_dir: Path, owner_user_id: str, slug: str) -> DeskLabel | None:
    return _find(list_labels(data_dir, owner_user_id), slug)


async def create_label(
    data_dir: Path,
    owner_user_id: str,
    *,
    name: str,
    slug: str | None = None,
    kind: str = "context",
    color: str | None = None,
) -> DeskLabel:
    """Append one row. ``slug`` derives from ``name`` when omitted.

    Check order, which decides which sentence the owner hears:

      1. kind not in KINDS             -> labels.invalid_kind       (400)
      2. slug normalization            -> labels.invalid_slug /
                                          labels.slug_too_long      (400)
      3. len(name) > 48                -> labels.name_too_long      (400)
      4. color not in PALETTE          -> labels.unknown_color      (400)
      5. slug held, SAME kind          -> labels.duplicate          (409)
      6. slug held, OTHER kind         -> labels.kind_conflict      (409)
      7. len(rows) >= 64               -> labels.registry_full      (400)

    5 and 6 sit before 7 deliberately: "you already have that one" is the useful
    sentence even on a full registry, and it is about THIS slug.

    The collision is ONE symmetric condition with one code, either direction;
    the sentence is generated from the OTHER row's kind.

    ``kind="state"`` is no longer creatable and fails check 1 with
    ``labels.invalid_kind``. The collision check still reads the STORED kind, so
    creating ``next`` as a context while a legacy ``next`` state row sits in the
    registry is ``labels.kind_conflict`` — *"'next' is a state, not a context"*.
    That refusal is what keeps the legacy row from being shadowed by a second
    row with the same slug, and it is the reason this check must not be moved
    onto the projected kind: under the projection the two would look identical
    and the create would fall through to ``labels.duplicate``, which is a
    different sentence about a different situation.

    ``color`` omitted or null -> ``default_color(rows)``. An explicit null is
    treated as omitted: the owner cannot deliberately create a colourless label
    and does not need to, because ``PATCH {color: null}`` clears one.
    """
    kind = _validate_kind("context" if kind is None else kind)
    name = _display_name(name or "")
    resolved = normalize_label(slug if slug else name)
    if not name:
        name = resolved
    _validate_name(name)
    _validate_color(color)

    async with _registry_lock:
        rows = _rows_for_write(data_dir, owner_user_id)
        existing = _find(rows, resolved)
        if existing is not None:
            if existing.kind == kind:
                raise LabelError.duplicate(resolved)
            raise LabelError.kind_conflict(resolved, existing.kind, kind)
        if len(rows) >= MAX_REGISTRY_ROWS:
            raise LabelError.registry_full(MAX_REGISTRY_ROWS)
        now = _now()
        row = DeskLabel(
            slug=resolved,
            name=name,
            kind=kind,
            color=color or default_color(rows),
            created_at=now,
            updated_at=now,
        )
        rows.append(row)
        _save(data_dir, owner_user_id, rows)
        return row


async def patch_label(
    data_dir: Path,
    owner_user_id: str,
    slug: str,
    *,
    name: object = _UNSET,
    color: object = _UNSET,
    kind: object = _UNSET,
) -> DeskLabel:
    """Change display name and/or colour IN PLACE — position is preserved and
    NO to-do is touched. This is the common rename; the slug never moves.

    ``kind`` is accepted only as an echo, and this is the unusual rule a lazy
    implementation gets wrong in one of two ways (both have a test):

      * absent              -> normal, the overwhelming majority
      * present, == VISIBLE -> accepted, no-op. A manage sheet that PATCHes the
                               whole row back must not 400 — including a row we
                               showed as a context while it is stored as a
                               retired state.
      * present, != visible -> labels.kind_immutable (400), naming
                               delete-and-re-add as the repair.
      * present, "state"    -> labels.invalid_kind (400) before any of the
                               above: the axis is retired and nothing can ask
                               for it, so the refusal is about the vocabulary
                               rather than about this row.

    Silently ignoring a CHANGED kind would let the sheet believe it promoted a
    label when it hadn't — the worse of the two failures.

    With ``KINDS`` down to one member the middle case is no longer reachable
    from the wire: every accepted value equals every visible kind. The guard
    stays because the rule it encodes — a label's kind is not editable — is
    still true, and a second kind arriving later must land on it rather than on
    a gap where it used to be.

    ``color`` is three-way: absent untouched, null clears, token sets.
    ``name`` absent leaves it; null or blank resets it to the slug rather than
    400ing, because clearing the field in the manage sheet is a real intent and
    "display it as its slug" is what it means.
    """
    if color is not _UNSET:
        _validate_color(color if isinstance(color, str) else None)
    if name is not _UNSET and isinstance(name, str):
        _validate_name(_display_name(name))

    async with _registry_lock:
        rows = _rows_for_write(data_dir, owner_user_id)
        row = _find(rows, slug)
        if row is None:
            raise LabelError.unknown_slug(slug)
        if kind is not _UNSET:
            wanted = _validate_kind(kind)
            # Compared against the VISIBLE kind, not the stored one. For every
            # row but a retired state the two are the same string and this is
            # the guard exactly as it was. For a retired state they differ, and
            # comparing against the stored kind would 400 a manage sheet that
            # echoed back the ``"context"`` WE told it on the read — the precise
            # failure the echo rule exists to prevent, and it would answer with
            # ``kind_immutable``'s sentence, which tells the owner to delete the
            # label. Instructing someone to delete a row to get out of an error
            # our own projection caused is the one outcome forbidden outright.
            #
            # This costs no safety: ``patch_label`` never assigns ``row.kind``
            # under any input, so the stored kind is not reachable from here in
            # either version. The guard refuses; it does not protect a write.
            if wanted != visible_kind(row):
                raise LabelError.kind_immutable(row.slug, row.kind, wanted)
        if name is not _UNSET:
            cleaned = _display_name(name) if isinstance(name, str) else ""
            row.name = cleaned or row.slug
        if color is not _UNSET:
            row.color = color if isinstance(color, str) else None
        row.updated_at = _now()
        _save(data_dir, owner_user_id, rows)
        # The STORED row was saved; the PROJECTED one is returned. See _visible.
        return _visible(row)


async def reorder_labels(
    data_dir: Path, owner_user_id: str, ordered_slugs: list[str]
) -> list[DeskLabel]:
    """Rewrite the array to match ``ordered_slugs``.

    Unknown slugs are IGNORED and omitted rows are appended in prior order —
    not an error, exactly like ``reorder_todos``, so a partial list (say, only
    the contexts) never drops the rest. Position is the ordering, so this verb
    IS the ordering write.

    Returns through ``_visible``, for exactly the reason ``patch_label`` /
    ``delete_label`` / ``merge_label`` do: this is a MUTATION RESPONSE, and a
    kind the GET would never show must not come back on one. Without it
    ``clawmeets todo labels reorder`` prints ``"kind": "state"`` for a legacy
    row that ``labels list`` reports as ``"context"`` — two answers about one
    row, one of which the server will refuse if it is ever sent back.
    """
    async with _registry_lock:
        rows = _rows_for_write(data_dir, owner_user_id)
        by_slug = {r.slug: r for r in rows}
        seen: set[str] = set()
        ordered: list[DeskLabel] = []
        for slug in ordered_slugs:
            row = by_slug.get(slug)
            if row is not None and slug not in seen:
                ordered.append(row)
                seen.add(slug)
        for row in rows:
            if row.slug not in seen:
                ordered.append(row)
        # The STORED rows were saved; the PROJECTED ones are returned.
        _save(data_dir, owner_user_id, ordered)
        return [_visible(row) for row in ordered]


async def delete_label(
    data_dir: Path, owner_user_id: str, slug: str
) -> tuple[DeskLabel, int]:
    """Delete always DETACHES: the slug leaves the registry and every to-do that
    carries it. Returns ``(removed row, detached_from)``.

    GUARANTEE, and it is in both SKILL.md files: deleting a label never deletes
    a to-do. An item whose only label is deleted becomes unlabelled.

    A legacy state row deletes exactly like a context — no guard, no
    confirmation, no special case. The retirement changes nothing here: the
    owner has always been able to delete any row, and it is THEIR delete. What
    the retirement does not do is delete one for them.

    Ordering, and it is the whole reason this is not two calls::

        async with desk_todo.plate_lock:      # OUTER
            async with _registry_lock:        # INNER, never reversed
                validate -> 404 before either write
                write the PLATE first
                write the REGISTRY second

    Plate-first means a crash between the two writes leaves a registry row with
    no references — invisible and harmless — rather than orphaned slugs on
    items. Rows whose labels actually changed get ``updated_at`` bumped;
    untouched rows do not.
    """
    async with desk_todo.plate_lock:
        async with _registry_lock:
            rows = _rows_for_write(data_dir, owner_user_id)
            row = _find(rows, slug)
            if row is None:
                raise LabelError.unknown_slug(slug)

            todos = desk_todo.load_plate(data_dir, owner_user_id)
            detached = _rewrite_plate(todos, lambda labels: [s for s in labels if s != slug])
            if detached:
                desk_todo.save_plate(data_dir, owner_user_id, todos)

            _save(data_dir, owner_user_id, [r for r in rows if r.slug != slug])
            return _visible(row), detached


async def merge_label(
    data_dir: Path, owner_user_id: str, slug: str, into: str
) -> tuple[DeskLabel, DeskLabel, int]:
    """Replace ``slug`` with ``into`` on every referencing to-do, de-duplicating
    in place while preserving first-seen order, then remove the source row. The
    target keeps its position. Returns ``(source, target, moved)``.

    ALL FOUR REFUSALS RUN BEFORE EITHER DOCUMENT IS TOUCHED — which is why the
    guard lives in the model and not in the route. The CLI and the assistant
    reach merge through this same call, and a route-level guard can refuse the
    response after the plate write has already landed::

        1. slug == into   -> labels.merge_self           (400)
        2. source missing -> labels.unknown_slug         (404)
        3. target missing -> labels.unknown_merge_target (404)
        4. kinds differ   -> labels.kind_mismatch        (400)

    Same-kind only, symmetric, and it reads the STORED kind — deliberately, and
    this is the guard that does the most work after the retirement. context ->
    context is the typo repair and is the overwhelming case. Two legacy state
    rows still merge into each other, which is the same repair on the axis that
    used to exist and must keep working.

    A legacy state merged into a context is REFUSED, and refusing is the safe
    direction rather than the tidy one: merge is the verb that rewrites the
    plate and then removes the source row, so letting it run across the
    projection is how an owner's stored ``state`` row gets deleted by a command
    they issued about two labels that looked alike on screen. Nothing is lost by
    refusing; the row and its to-dos stay exactly where they are.

    There is NO ``state_conflicts`` in the return: after the same-kind rule it
    is structurally always zero, and a field that always reads 0 is worse than
    no field — it reads as a live signal, so a client eventually branches on it.
    """
    if slug == into:
        raise LabelError.merge_self(slug)

    async with desk_todo.plate_lock:
        async with _registry_lock:
            rows = _rows_for_write(data_dir, owner_user_id)
            source = _find(rows, slug)
            if source is None:
                raise LabelError.unknown_slug(slug)
            target = _find(rows, into)
            if target is None:
                raise LabelError.unknown_merge_target(into)
            if source.kind != target.kind:
                raise LabelError.kind_mismatch(
                    source.slug, source.kind, target.slug, target.kind
                )

            def _swap(labels: list[str]) -> list[str]:
                out: list[str] = []
                for existing in labels:
                    replacement = into if existing == slug else existing
                    if replacement not in out:
                        out.append(replacement)
                return out

            todos = desk_todo.load_plate(data_dir, owner_user_id)
            moved = _rewrite_plate(todos, _swap)
            if moved:
                desk_todo.save_plate(data_dir, owner_user_id, todos)

            _save(data_dir, owner_user_id, [r for r in rows if r.slug != slug])
            return _visible(source), _visible(target), moved


def _rewrite_plate(todos: list[DeskTodo], rewrite) -> int:
    """Apply ``rewrite`` to every to-do's labels in place; return how many rows
    actually changed. Only changed rows get ``updated_at`` bumped — an untouched
    item must not look edited on the owner's rail."""
    changed = 0
    for todo in todos:
        after = rewrite(todo.labels)
        if after != todo.labels:
            todo.labels = after
            todo.updated_at = _now()
            changed += 1
    return changed


async def auto_register(
    data_dir: Path, owner_user_id: str, slugs: list[str]
) -> list[str]:
    """Create registry rows for slugs that have none. Returns the new slugs.

    Called ONLY from the owner-authenticated to-do write paths (POST and PATCH
    ``/me/desk/todos``). Never from PUT publish — without that line, fifty
    agents each inventing ``desk`` / ``office-stuff`` / ``at-office`` silently
    fill the owner's group list.

    An auto-created row is ALWAYS ``kind="context"`` with ``color=None``: the
    server never invents an axis and never picks a colour. That was already the
    only kind this function could produce, and with ``KINDS`` down to one member
    it is now the only kind anything can produce — so the paragraph that used to
    stand here, arguing this line was what kept the state axis from drifting and
    at-most-one-state intact, is describing an axis that no longer carries
    meaning. The behaviour is unchanged; only the reason to care is gone.

    Best-effort by contract: a full registry simply leaves the slug
    unregistered. It NEVER raises into the caller — the to-do write has already
    succeeded and must not be failed by a bookkeeping side effect.

    AN UNREADABLE REGISTRY IS SKIPPED, NOT REFUSED, and this is the one path
    where that asymmetry with the six curation verbs is deliberate. Every plate
    slug looks unregistered when the file will not parse, so without this guard
    ANY label edit on ANY to-do — including taking a chip OFF, which is a
    REPLACE that resends the survivors — rewrote the whole document from zero
    rows. There is no browser route left into the curation verbs under that
    state, but there is no avoiding this one: the frontend cannot suppress it
    without freezing row-label editing, which would contradict the banner
    telling the owner their labels are still on their to-dos.

    So: the to-do write succeeds, the slugs stay unregistered (they render
    solid-neutral, a treatment that already exists), and the file is not
    touched. 400ing the to-do write instead — refusing to save someone's to-do
    because a label INDEX is corrupt — is the worse outcome of the two.
    """
    if not slugs:
        return []
    try:
        async with _registry_lock:
            registry = _read(data_dir, owner_user_id)
            if not registry.readable:
                return []
            # ``_rows_raw`` because this function ``_save``s — see the note in
            # ``_rows_for_write``. This is the OTHER write path, and it is the
            # easier one to miss: it does not go through ``_rows_for_write`` at
            # all, so repointing only that one would have left every ordinary
            # label edit on any to-do rewriting the owner's stored states to
            # contexts as a side effect.
            rows = _rows_raw(registry.document)
            held = {r.slug for r in rows}
            created: list[str] = []
            now = _now()
            for slug in slugs:
                if slug in held or len(rows) >= MAX_REGISTRY_ROWS:
                    continue
                rows.append(
                    DeskLabel(
                        slug=slug,
                        name=slug,
                        kind="context",
                        color=None,
                        created_at=now,
                        updated_at=now,
                    )
                )
                held.add(slug)
                created.append(slug)
            if created:
                _save(data_dir, owner_user_id, rows)
            return created
    except Exception:
        return []
