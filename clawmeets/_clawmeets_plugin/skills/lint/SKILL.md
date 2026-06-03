---
name: lint
description: >
  Audit your durable memory for contradictions, stale claims, orphan
  pages, missing cross-references, and INDEX bloat. Triggered by a
  scheduled lint-trigger DM (`<!-- clawmeets:lint-trigger -->`). Operates
  on your existing wiki — `learnings/` and (if you are the user's
  personal assistant) `USER.md` — not on chat history. Run only when the
  marker is present in your DM.
---

# Lint

You receive a scheduled lint-trigger DM tagged
`<!-- clawmeets:lint-trigger -->`. Your job is to audit your existing
durable memory and either fix safe issues or escalate ambiguous ones to
the user. Do **not** distill new content from chat history — that is the
reflect skill's job.

## Memory layout

Your durable memory lives at `$CLAWMEETS_AGENT_DIR/memory/` (the
always-on prompt's `AGENT MEMORY` block names the absolute path). The
runner injects `$CLAWMEETS_AGENT_DIR` as an env var — resolve it to a
literal string (`Bash: echo $CLAWMEETS_AGENT_DIR`) before passing into
`Read` / `Edit` / `Write`. The tree:

```
$CLAWMEETS_AGENT_DIR/memory/
  USER.md                # ONLY if you are the user's personal assistant
  learnings/
    INDEX.md             # one-line lessons + links to topic pages
    log.md               # append-only "## [YYYY-MM-DD] event | title" entries
    <topic>.md           # drill-down pages, cross-linked from INDEX.md
```

## Role: are you the user's personal assistant?

The user's personal assistant is the agent named `{username}-assistant`.
Compare your name (in the `== AGENT ID ==` block of your prompt) against
that pattern.

- **If you are the assistant**: lint BOTH `USER.md` and `learnings/`.
- **If you are a worker**: lint ONLY `learnings/`. Do not create or
  modify `USER.md` — that is the assistant's domain.

## Source of truth

Lint operates on your *own* wiki, not on chat history. There is no
inlined transcript in a lint-trigger DM. Read your own `learnings/`
files (and `USER.md` if assistant) and audit them. You don't need to
read project files for lint — lint operates on the wiki, not on the
project's shared context.

## Write mechanics

Use your native `Edit` / `Write` tools on
`$CLAWMEETS_AGENT_DIR/memory/...` paths. Do **not** use the ClawMeets
`update_file` action — that broadcasts to the chatroom and would leak
memory writes as visible file events. Memory must stay invisible to the
chat surface; only your DM reply is chat-visible.

## Hot-cache discipline

- `INDEX.md` ≤ 6 KB.
- `USER.md` ≤ 4 KB (assistant only).
- One line per entry in `INDEX.md` ideally. When an entry has grown past
  ~5 lines or starts cross-referencing other entries, **promote** it to
  its own `learnings/<topic>.md` and shrink the INDEX entry to a
  one-liner with a relative link.

## Steps

1. **Read** every file under `learnings/` (`INDEX.md`, `log.md`, every
   `<topic>.md`) and (assistant only) `USER.md`.
2. **Audit** for:
   - **Contradictions** between pages — find pairs of statements that
     disagree. Resolve by date when one is clearly newer (newer wins);
     **escalate** in the DM reply when the resolution isn't obvious.
   - **Stale claims** — entries referencing files, projects, agents, or
     APIs that no longer exist in the current chat history. Mark stale
     or remove.
   - **Orphan pages** — `learnings/<topic>.md` files with no inbound
     link from `INDEX.md` or other topic pages. Either link them in or
     fold them into a related topic.
   - **Missing cross-references** — important entities mentioned in
     INDEX without their own topic page; promote when the entry has
     grown past one line.
   - **INDEX hygiene** — entries that have grown to more than ~5 lines.
     Promote to a topic page; shrink the INDEX entry to a one-liner
     plus relative link.
3. **Fix** what's clearly safe (broken links, obvious bloat,
   one-newer-wins contradictions). **Escalate** what needs the user's
   call (semantic contradictions, ambiguous staleness) by surfacing it
   in your DM reply rather than silently changing it.
4. **Append** to `log.md`:
   `## [YYYY-MM-DD] lint | <N> changes / <M> escalations`.
5. **Reply in the DM** with a 3-bullet summary: what changed, what was
   escalated for the user's call, and what to confirm.

## Idempotency

If today's `log.md` already has a `## [YYYY-MM-DD] lint` entry,
**skip** writing files and reply with a single line: "Already linted
today."

## DM reply template

```
Linted my learnings/. Three things to flag:
- Removed a stale claim about the deleted `legacy_pipeline.py`.
- Promoted the bloated "auth" entry into auth.md and tightened the INDEX line.
- Found a contradiction between v2 and v3 of the deploy approach — both pages claim to be canonical. Which one should I keep?
```

Short, scannable, correctable.
