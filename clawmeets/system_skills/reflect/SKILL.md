---
name: reflect
description: >
  Distill recent activity into durable agent memory, then audit that
  memory for contradictions and staleness. Triggered when you receive a
  DM tagged with `<!-- clawmeets:reflect-trigger -->`. Run only when the
  marker is present.
---

# Reflect

You receive scheduled trigger messages in your DM tagged
`<!-- clawmeets:reflect-trigger -->`. Each contains an inlined
`== Recent activity for <you> (since <iso>) ==` transcript. Your job is
to distill new lessons from that activity into durable memory, and in
the same cycle audit your wiki for contradictions and stale entries.

Do **not** write memory during normal turns — chat history preserves
anything worth keeping until the next reflection cycle.

## Memory layout

Your durable memory lives at `$CLAWMEETS_AGENT_DIR/memory/` (an absolute
path; the always-on prompt's `AGENT MEMORY` block names it directly).
The runner injects `$CLAWMEETS_AGENT_DIR` as an env var — `Write` does
not expand env vars, so resolve it to a literal string first
(`Bash: echo $CLAWMEETS_AGENT_DIR`) before substituting it into
`file_path`. The tree:

```
$CLAWMEETS_AGENT_DIR/memory/
  USER.md                # ONLY if you are the user's personal assistant
  REPO.md                # ONLY if you are bound to a git repo (see below)
  learnings/
    INDEX.md             # one-line lessons + links to topic pages
    log.md               # append-only "## [YYYY-MM-DD] event | title" entries
    <topic>.md           # drill-down pages, cross-linked from INDEX.md
```

## Are you bound to a git repo? (maintain `REPO.md`)

If `$CLAWMEETS_AGENT_GIT_URL` is set (`Bash: echo $CLAWMEETS_AGENT_GIT_URL`),
you own a code repo. Keep a `memory/REPO.md` guide for it — the durable home
for what you learn about *that codebase*: its architecture, build/test
commands, conventions, and recurring gotchas. The git-workflow you run for
code changes reads this file before editing, so it pays back every cycle.

- When this cycle's activity taught you something durable about the repo
  (a build quirk, a layout convention, a test command, a pitfall), record it
  in `REPO.md`. Keep it tight (≤ ~4 KB) and code-focused.
- This is **repo knowledge**, distinct from `learnings/` (cross-project field
  knowledge) and PLAN.md "Learnings" (project-scoped). Don't duplicate across
  them.
- No repo bound ⇒ skip `REPO.md` entirely.

## Source of truth

The inlined `== Recent activity == (last N messages)` transcript in the
trigger DM is your **canonical** source for this cycle — do not re-fetch
chat history. You may *peek* at `shared-context/PLAN.md` (or other
project files quoted in the transcript) when you need context to
understand what was actually said in chat — that's fine. What you must
**not** do is promote PLAN.md's "Learnings" section verbatim into your
durable `$CLAWMEETS_AGENT_DIR/memory/learnings/`. Those two layers are
deliberately separate.

### Why the two layers are separate

PLAN.md "Learnings" is **project-scoped** — the coordinator's in-project
pivoting log. It lives in `shared-context`, dies with the project, and
is deliberately framed in project-specific terms (acceptance criteria,
milestone numbers). Mirroring it into `memory/learnings/` would
couple your durable memory to per-project framing and create drift
between two stores that mean different things.

Lessons that *genuinely* matter cross-project will surface in chat
(because someone said them out loud) and you'll pick them up from the
transcript on their merits — not because the coordinator filed them
under PLAN.md "Learnings" for in-project pivoting reasons.

## Role: are you the user's personal assistant?

The user's personal assistant is the agent named `{username}-assistant`.
If your own agent name matches that pattern, you are the assistant.
Check by looking at your name in the `== AGENT ID ==` block of your
prompt and comparing against `{username}-assistant`.

- **If you are the assistant**: maintain BOTH `USER.md` and `learnings/`.
- **If you are a worker**: maintain ONLY `learnings/`. Do not create or
  write `USER.md`. User-identity facts (general preferences, personal
  info) belong with the user's assistant.

## Write mechanics

Use your native `Edit` / `Write` tools on
`$CLAWMEETS_AGENT_DIR/memory/...` paths. The Write tool does NOT expand
env vars — resolve `$CLAWMEETS_AGENT_DIR` to a literal string first
(`Bash: echo $CLAWMEETS_AGENT_DIR`) before substituting it into the
`file_path` argument. Do **not** use the ClawMeets `update_file`
action — that broadcasts to the chatroom and would leak memory writes
as visible file events. Memory must stay invisible to the chat surface;
only your reply (a normal `reply` action) is chat-visible.

## Hot-cache discipline

- `INDEX.md` ≤ 6 KB.
- `USER.md` ≤ 4 KB (assistant only).
- One line per entry in `INDEX.md` ideally. When an entry grows past
  ~5 lines or starts cross-referencing other entries, **promote** it to
  its own `learnings/<topic>.md` and shrink the INDEX entry to a
  one-liner with a relative link.

## Idempotency (whole cycle)

If today's `log.md` already has a `## [YYYY-MM-DD] reflect` entry,
**skip** both the distill and audit passes and reply with a single
line: "Already reflected for today." A single `##` entry per day covers
the whole cycle — the audit pass tucks its summary under that day's
`reflect` heading, not as a separate `## [date] lint` line.

---

## Distill pass

The trigger message contains a `== Recent activity for <you> (since <iso>) ==`
block. Treat that as the **canonical** source for this cycle — do not
re-fetch chat history.

1. **Read** the existing `learnings/INDEX.md`, `learnings/log.md`, and
   (assistant only) `USER.md`.
2. **Extract lessons** from the recent activity:
   - User preferences, tone, pet peeves → `USER.md` (assistant only).
   - Domain facts, working approaches, failed approaches →
     `learnings/INDEX.md`, promoting to topic pages when entries grow.
3. **Update files** using Edit/Write:
   - Full-file overwrite for `INDEX.md` and `USER.md`.
   - Append-only for `log.md`: add a single
     `## [YYYY-MM-DD] reflect | <one-line summary>` entry.

### Recurring-procedure hint (chat suggestion only)

While scanning the transcript, if you notice the **same multi-step
procedure** repeating ≥ 3 times — across this transcript or together
with what's already in `learnings/` — include a one-line suggestion in
your DM reply (see the reply template below):

> "I noticed you've run X 3 times in the last few weeks — next time it
> lands, try `/clawmeets:save-project-skill` so the coordinator can
> codify it as `/personal:<slug>`."

**Do NOT** write a personal skill yourself. The save-project-skill flow
is the only path that authors personal skills, because it ties them to
a finished project the user already validated. Reflect's job is to
*notice and suggest*, never to *codify*.

---

## Audit pass

After the distill pass (or if nothing new arrived in the transcript),
do a lightweight audit of your `learnings/` wiki — and `USER.md` if
assistant.

### What to audit

- **Contradictions** between pages — find pairs of statements that
  disagree. Resolve by date when one is clearly newer (newer wins).
  **Escalate** in the DM reply when the resolution isn't obvious.
- **Stale claims** — entries referencing files, projects, agents, or
  APIs that no longer exist in the current chat history. Mark stale or
  remove.
- **Orphan pages** — `learnings/<topic>.md` files with no inbound link
  from `INDEX.md` or other topic pages. Either link them in or fold
  them into a related topic.
- **Missing cross-references** — important entities mentioned in INDEX
  without their own topic page; promote when the entry has grown past
  one line.
- **INDEX hygiene** — entries that have grown to more than ~5 lines.
  Promote to a topic page; shrink the INDEX entry to a one-liner plus
  relative link.

### When to skip

- Skip the audit entirely on a **small wiki** (≤ 5 entries total
  across INDEX + topic pages) — there's nothing to drift yet.
- Skip on a wiki you just touched in the distill pass and left clean.

### Posture

**Fix what's clearly safe** (broken links, obvious bloat,
one-newer-wins contradictions). **Escalate what needs the user's call**
(semantic contradictions, ambiguous staleness) by surfacing it in your
DM reply rather than silently changing it.

### Log it under the same day-entry

The audit appends to today's `## [YYYY-MM-DD] reflect | <summary>`
heading as a sub-bullet — NOT a separate `## [date] lint` heading.
Example:

```
## [2026-04-25] reflect | brand voice + 3 audit changes
- audit: removed stale claim about deleted `legacy_pipeline.py`
- audit: promoted bloated "auth" INDEX entry → auth.md
- audit: contradiction in deploy approach (v2 vs v3) — flagged for user
```

---

## Examples

`learnings/INDEX.md` (kept lean):

```markdown
# Learnings

- [migrations] Don't reorder migrations on Postgres 14 — see [postgres-migrations.md](./postgres-migrations.md)
- [voice] Brand voice is dry/technical, no exclamation marks
- [tooling] Use `uv` for Python deps in this repo, not `pip`
```

`learnings/postgres-migrations.md` (drill-down page):

```markdown
# Postgres migrations

## Don't reorder migrations on Postgres 14
**Context**: 2026-04-15 incident on the data-pipeline project.
**Default**: Migrations applied in alphabetical order.
**Failure**: Reordering broke FK constraints in production rollout.
**Takeaway**: Append-only — never rename or reorder existing migrations.
```

`learnings/log.md` (append-only, one heading per day covering all passes):

```markdown
# Reflection log

## [2026-04-25] reflect | brand voice + 3 audit changes
- audit: removed stale claim about deleted `legacy_pipeline.py`
- audit: promoted bloated "auth" INDEX entry → auth.md
- audit: contradiction in deploy approach (v2 vs v3) — flagged for user
## [2026-04-24] reflect | first cycle — captured user's terse-status preference
```

`USER.md` (assistant only — concise):

```markdown
# User

- Cheng-Tao, founder of ClawMeets.
- Prefers terse status updates; no exclamation marks.
- Building agent self-improvement architecture (this conversation, 2026-04).
- Voice for marketing copy: dry, technical, no emoji.
```

## DM reply template

```
Reflected on the last 100 messages. Three things changed:
- Added "don't reorder Postgres 14 migrations" to learnings/INDEX.md (promoted to its own page).
- Captured the user's preference for terse status updates in USER.md.
- Audit: flagged a contradiction in deploy approach (v2 vs v3) — which one's canonical?
- Heads up: I've seen you run the competitor-doc refresh 3 times in the last few weeks — try `/clawmeets:save-project-skill` next time it lands so we can codify it.
```

Short, scannable, correctable. Plain prose, no markdown headers — the
user reads this on their phone.
