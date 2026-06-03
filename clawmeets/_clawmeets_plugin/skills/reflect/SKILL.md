---
name: reflect
description: >
  Distill recent activity into durable agent memory and maintain your
  private skill hub. The reflect-trigger DM kicks off three sub-modes in
  one cycle: Reflect (distill recent activity into learnings), Promote
  (codify a recurring procedure as a `/personal:<name>` skill), and
  Correct (patch a personal skill that misfired). Triggered when you
  receive a DM tagged with `<!-- clawmeets:reflect-trigger -->`. Run only
  when the marker is present.
---

# Reflect

You receive scheduled trigger messages in your DM tagged
`<!-- clawmeets:reflect-trigger -->`. Each contains an inlined
`== Recent activity for <you> (since <iso>) ==` transcript. Your job is
to distill new lessons into durable memory and, in the same cycle,
codify or patch personal skills.

The reflect-trigger fires all three sub-modes (Reflect, Promote,
Correct) in one cycle — they share the same inlined transcript and the
same `learnings/log.md` cursor. Do **not** write memory or author
personal skills during normal turns — chat history preserves anything
worth keeping until the next reflection cycle.

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
  learnings/
    INDEX.md             # one-line lessons + links to topic pages
    log.md               # append-only "## [YYYY-MM-DD] event | title" entries
    <topic>.md           # drill-down pages, cross-linked from INDEX.md
```

You also have a personal skill hub at the **absolute path**
`$CLAWMEETS_AGENT_DIR/personal-skill-hub/` (the same path is named in
your always-on MEMORY block under "personal skill hub anchored at this
absolute path:"). Personal skills are agent-private and never leave the
runner; writing outside `$CLAWMEETS_AGENT_DIR/personal-skill-hub/`
won't be loaded by Claude's plugin loader. The tree:

```
$CLAWMEETS_AGENT_DIR/personal-skill-hub/
  skills/
    INDEX.md             # one line per skill: "<name> — when to invoke"
    <name>/
      SKILL.md           # frontmatter (name, description) + procedure body
```

Each `<name>/SKILL.md` is auto-registered as a `/personal:<name>`
slash-command on your next invocation — no sync, no server round-trip.
Personal skills are agent-private and never leave your runner.

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
`$CLAWMEETS_AGENT_DIR/memory/...` and
`$CLAWMEETS_AGENT_DIR/personal-skill-hub/...` paths. The Write tool
does NOT expand env vars — resolve `$CLAWMEETS_AGENT_DIR` to a literal
string first (`Bash: echo $CLAWMEETS_AGENT_DIR`) before substituting it
into the `file_path` argument. Do **not** use the ClawMeets
`update_file` action — that broadcasts to the chatroom and would leak
memory writes (and personal-skill writes) as visible file events.
Memory and personal skills must stay invisible to the chat surface;
only your reply (a normal `reply` action) is chat-visible.

## Hot-cache discipline

- `INDEX.md` ≤ 6 KB.
- `USER.md` ≤ 4 KB (assistant only).
- One line per entry in `INDEX.md` ideally. When an entry grows past
  ~5 lines or starts cross-referencing other entries, **promote** it to
  its own `learnings/<topic>.md` and shrink the INDEX entry to a
  one-liner with a relative link.

---

## Reflect sub-mode

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
4. **Reply in the DM** with a 3-bullet summary of what changed. Plain
   prose, no markdown headers — the user reads this on their phone.

### Idempotency (reflect)

If today's `log.md` already has a `## [YYYY-MM-DD] reflect` entry,
**skip** all three sub-modes (Reflect, Promote, Correct) and reply with
a single line: "Already reflected for today." A single `##` entry per
day covers the whole cycle — Promote and Correct are logged as bullets
*under* that day's `reflect` heading, not as separate `## [date]
promote` lines.

---

## Promote sub-mode (runs in the same cycle as Reflect)

After Reflect finishes (or in the same pass), look at the **whole**
inlined transcript — not just the new portion since your last reflection
— and decide whether to codify a recurring procedure as a
`/personal:<name>` skill.

### When to Promote

Use your judgment. Good Promote candidates have **at least one** of:

- **Recurring procedure**: the same multi-step workflow appears across
  several conversations or projects in the transcript. Three-ish hits is
  a useful floor; one is too few; ten is well past time.
- **Multi-step with prereqs and checks**: the procedure has setup steps,
  ordered actions, and verifiable success conditions — i.e. it's an
  actual *procedure*, not a single tip. Single-tip lessons belong in
  `learnings/`, not the personal-skill hub.
- **Explicit user request**: the user said "remember how to do X" or
  "save this as a skill". Always honor — even for one-shot procedures.

**Bad** Promote candidates (don't):

- A one-off task with no obvious re-use ("plan my Tuesday").
- A simple tip or fact (lives in `learnings/INDEX.md`).
- Something that overlaps an existing personal skill or a system skill
  (e.g. `/clawmeets:reflect`) — patch the existing one instead.

### How to Promote

1. **Pick a name**: lowercase, hyphens or dots only, ≤ 40 chars,
   describes the procedure (e.g. `weekly-replan`,
   `competitor-doc-refresh`, `inbox-triage`).
2. **Check for collisions**: list `$CLAWMEETS_AGENT_DIR/personal-skill-hub/skills/` — if a
   skill with that name already exists, switch to the Correct flow
   below (patch it) rather than overwriting.
3. **Write `$CLAWMEETS_AGENT_DIR/personal-skill-hub/skills/<name>/SKILL.md`** with this shape:
   ```markdown
   ---
   name: <name>
   description: <one-line summary; agent reads this when deciding to invoke>
   ---

   # <Procedure title>

   ## When to use this
   <2-3 sentences. Concrete signals — what does the task look like?>

   ## Prerequisites
   - <inputs / context the procedure assumes>

   ## Steps
   1. <ordered, imperative>
   2. ...

   ## Success checks
   - <how you know it worked>

   ## Pitfalls
   - <gotchas you discovered while doing this>
   ```
   Use Edit/Write — **never** `update_file`.
4. **Update `$CLAWMEETS_AGENT_DIR/personal-skill-hub/skills/INDEX.md`** with one new line:
   `- <name> — <when to invoke; one short clause>`. Sort alphabetically.
   Create the file if it doesn't exist yet.
5. **Log under today's `learnings/log.md` reflect entry**, as a bullet:
   `- promoted: /personal:<name>`. Don't create a separate
   `## [YYYY-MM-DD] promote` heading.

### Idempotency (promote)

The "today's log already has a reflect entry → skip" check at the top of
the cycle covers Promote too. Within a single cycle, if you Promote and
the same skill name was already promoted today (visible as a `promoted:
/personal:<name>` bullet under today's entry), don't write the SKILL.md
twice — that's a sign you missed the dedup check.

---

## Correct sub-mode (runs in the same cycle as Reflect)

Scan the inlined transcript for `/personal:<name>` invocations that
**misfired**: produced an error, got user pushback ("that's not what I
meant", "skip step X"), or led you to deviate from the documented steps.

### When to Correct

Trigger a patch when **any** of:

- The user explicitly told you a personal skill was wrong or out of date.
- A `/personal:<name>` invocation produced an error and you had to work
  around its instructions.
- You deviated from the SKILL.md steps because the world changed (a tool
  was renamed, a URL moved, a precondition no longer holds) and your
  workaround succeeded.

If a personal skill worked exactly as written, **don't touch it** — only
patch on observed failure.

### How to Correct

1. **Read** `$CLAWMEETS_AGENT_DIR/personal-skill-hub/skills/<name>/SKILL.md`.
2. **Patch** with `Edit` — keep the change minimal. Targeted edits to a
   single step or pitfall are better than rewriting the whole skill.
   - If a step was wrong: edit the step in place.
   - If a pitfall was missing: add a bullet under "Pitfalls".
   - If the skill is broken beyond a small patch: rewrite it with `Write`.
3. **Log under today's `learnings/log.md` reflect entry**, as a bullet:
   `- patched: /personal:<name> — <one-line reason>`.

### Idempotency (correct)

Same day-level guard as Reflect — if today's log already has a `reflect`
entry and a `patched: /personal:<name>` bullet for this skill, skip.

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

`learnings/log.md` (append-only, all cycles share one entry per day):

```markdown
# Reflection log

## [2026-04-25] reflect | learned brand voice + promoted weekly-replan
- promoted: /personal:weekly-replan
- patched: /personal:competitor-doc-refresh — Crunchbase URL moved
## [2026-04-24] reflect | first cycle — captured user's terse-status preference
## [2026-04-22] lint | 3 changes / 1 escalation
```

`$CLAWMEETS_AGENT_DIR/personal-skill-hub/skills/INDEX.md`:

```markdown
# Personal skills

- competitor-doc-refresh — pull latest deck, diff against last quarter
- inbox-triage — when inbox > 50, sort sender → priority → deadline
- weekly-replan — reshuffle calendar when Monday's plan slips
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
- Codified weekly-replan as /personal:weekly-replan — invoke when Monday's plan slips.
- Patched /personal:competitor-doc-refresh: Crunchbase URL moved, fixed step 3.
```

Short, scannable, correctable.
