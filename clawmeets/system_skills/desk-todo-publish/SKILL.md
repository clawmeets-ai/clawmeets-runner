---
name: desk-todo-publish
description: >
  Hand ONE task back to the owner's My Desk plate — only when the ball is
  genuinely in *their* court. NOT for work you can just do yourself, NOT for
  status updates (those are a reply or a briefing), NOT to nag. INVOKE when
  ANY of: (1) a message contains `<!-- clawmeets:desk-todo-publish-trigger -->`;
  (2) the user or coordinator asks you to "add this to my to-do / plate",
  "flag this to me", or "put this on my desk"; (3) you surfaced something
  mid-task that needs the *user's own hand* — an approval, a sign-off, a
  decision only they can make — and want to hand it back with context.
  Package the task and shell `clawmeets todo publish`. The item appears on
  the owner's To-do rail, agent-badged, and clicking it opens a guided
  take-over pre-loaded with your suggested prompt + files. You may retract
  only an item YOU published. Managing or firing what is already on the
  plate is the owner's assistant's job — the `desk-todo` skill.
---

# Desk to-do — hand a task back to the manager

My Desk's right rail is the manager's **plate**: things that still sit
with *them*. Most items they capture themselves. This skill is the other
source — **you**, an agent, pushing a task back when it needs the user's
own hand.

You publish ONE to-do per invocation via `clawmeets todo publish`.

## § When to publish

Publish when the ball is genuinely in the *user's* court and you can make
their next move cheap by packaging what you already know:

- an **approval / sign-off** only they can give (a PO, a contract, a spend);
- a **decision** that needs their judgment (which segment, which direction);
- a **hand-off** you can't complete without their input.

Do **not** publish for work you can just do, for status updates (those are
a reply or a briefing), or to nag. The plate is the user's own attention,
and every item you add spends some of it. If in doubt, ask in chat instead
of adding to the plate.

Triggers:
- **Marker**: a message contains
  `<!-- clawmeets:desk-todo-publish-trigger -->` — the body after it
  describes what to flag; follow it.
- **Direct request**: the user / coordinator says to add something to
  their to-do / desk / plate.
- **Discretion**: you surfaced a user-hand item mid-task and want to hand
  it back cleanly.

## § What to package

Everything is optional except `--text`. The more you supply, the sharper
the take-over the manager opens.

- `--text` (required) — the task as it reads on the plate, e.g.
  `"Approve the Provi restock PO ($6.8k) waiting on your sign-off"`.
- `--suggest <agent>` — the agent you'd hand it to (short or full name).
  Pre-selected as the recipient in the take-over.
- `--draft-prompt "…"` — a ready-to-refine request that seeds the
  take-over composer. Write it as the message you'd send the suggested
  agent, so the manager edits rather than composes.
- `--context-file ctx.md` — a small `.md`/`.txt` (≤ 16 KB) of supporting
  detail. Becomes an attachable context chip on the composer, so the
  recipient can consult it. Write it to your sandbox first with `Write`.
- `--file "name::sub"` (repeatable) — a reference file you gathered, shown
  as an informational chip (name + a short sub-label). Chip only — no
  bytes are uploaded; the name rides along so the recipient knows to
  consult it.
- `--done "step"` (repeatable) — a step you already completed, listed
  under **What's been done** so the manager sees the groundwork.
- `--fact "label::value"` (repeatable) — a key fact, listed under
  **Available & relevant** (e.g. `"PO total::$6,821.40 · net-30"`).
- `--due "Today"` / `--due "Fri"` — an optional due hint (`Today` renders
  urgent).
- `--linked "label::icon"` — a source to open from the take-over (e.g.
  `"Finance briefing::chart"`).

Keep provenance honest — only list steps you actually did and facts you
actually verified.

## § Publish

1. If you have supporting detail, `Write` it to `ctx.md` in your sandbox.
2. Shell (one invocation, one to-do):
   ```bash
   clawmeets todo publish \
     --text "Approve the Provi restock PO ($6.8k) waiting on your sign-off" \
     --suggest api_sync \
     --draft-prompt "Review Provi restock PO #4471 ($6,821.40, net-30). If the line items and pricing check out against the last order, approve it and confirm the delivery window with Provi." \
     --context-file ctx.md \
     --due Fri \
     --linked "Finance briefing::chart" \
     --done "Reconciled every line item against the last 3 Provi orders — pricing matched" \
     --done "Confirmed it covers the Etna Rosso + Sancerre low-stock flags" \
     --fact "PO total::$6,821.40 · net-30" \
     --fact "Budget::Within Q3 F&B — 62% used" \
     --file "PO-4471-provi.pdf::purchase order · 2pp"
   ```
   The CLI resolves your agent id + token + server URL from the env the
   runner injects (`CLAWMEETS_AGENT_ID`, `CLAWMEETS_AGENT_TOKEN`,
   `CLAWMEETS_SERVER_URL`). No `--token` flag needed.
3. Reply ONE line in `user-communication`, e.g.
   `Flagged "Approve the Provi PO" to your desk — open it to review and dispatch.`
   Don't restate the whole task; the plate item IS the deliverable.

To see what's already on the plate (e.g. to avoid publishing a duplicate):

```bash
clawmeets todo list
```

To retract a to-do you published:

```bash
clawmeets todo delete <id>
```

## § What you cannot do here

`delete` is scoped to **your own** items: retracting a to-do published by
another agent, or one the user captured in the browser, returns

```
Error 403: Only the publishing agent, the owner's assistant, or the
owning user can delete this to-do
```

That is the rule working, not a bug — don't retry it and don't ask for a
wider token. Same for the plate-management verbs (`todo update` / `done` /
`reopen` / `trigger`): those carry the *owner's* authority and return
`Error 401: Invalid token` for you. They belong to the owner's own
`{username}-assistant` (the `desk-todo` skill). If the user asks you to
mark something done or send a saved draft, say it's their assistant's job
and stop — don't work around it by publishing a second item about it.
