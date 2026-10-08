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
  decision only they can make — and want to hand it back ready to act on.
  Package the task and shell `clawmeets todo publish`, tagging it with the
  owner's OWN labels via `--label` (e.g. `--label office`) so it lands in the
  right group on their rail: run `clawmeets todo labels list` first and reuse
  what is there — none of it is built in, and a label you invent still shows
  but stays unregistered.
  The item appears on the owner's To-do rail; what they see is your title,
  your suggested prompt and your suggested recipient, all of which they can
  edit before sending. You
  may retract only an item YOU published. Managing or firing what is already on
  the plate — and renaming, merging, recolouring or deleting the labels
  themselves — is the owner's assistant's job, the `desk-todo` skill.
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

Everything is optional except `--text`. What you pass is exactly what the
owner sees on the item — the suggested prompt and recipient land in the same
fields the owner edits themselves, so write them ready to send.

- `--text` (required) — the task as it reads on the plate, e.g.
  `"Approve the Provi restock PO ($6.8k) waiting on your sign-off"`.
- `--draft-prompt "…"` — the suggested prompt. Write it as the message
  you'd send the suggested agent, self-contained (put the key facts — amounts,
  ids, dates, what you already checked — in it), so the owner edits rather
  than composes.
- `--suggest <agent>` — the suggested recipient (short or full name). It
  becomes the draft's recipient, which the owner can change.
- `--label <name>` (repeatable) — a GTD context the owner files this under,
  e.g. `--label office`. Write `office` or `@office`; both land as the same
  label. Use the owner's **existing** vocabulary — run `clawmeets todo labels
  list` first and reuse what's there rather than inventing a near-duplicate.
  **That list is the owner's own and none of it is built in** — do not assume
  `office` / `home` / `phone` exist, and do not assume any particular label
  does. A label you invent still shows on the item, but it stays
  **unregistered** — dashed outline, no colour, sorted last, in its own group
  rather than the owner's curated ones — until the owner adopts it. At most 8
  labels fit on one item; a 9th is refused whole rather than partly applied.

  **A label cannot say the work is running.** Every label is a context; the
  `state` kind is retired. Whether an item is New / Working / Completed is
  DERIVED from the projects linked to it and is read-only — nothing you pass
  here moves it, and there is no label that means "in progress". If you want the
  item to read Working, link it to the project actually doing the work
  (`clawmeets todo associate <id> <project_id>`), which is true rather than
  decorative.

There are no other flags — no context file, file chips, due date, facts or
"what's been done" list. Anything the owner needs to know goes in the
`--draft-prompt`. Keep it honest — only state what you actually did and
verified.

## § Publish

1. Shell (one invocation, one to-do):
   ```bash
   clawmeets todo publish \
     --text "Approve the Provi restock PO ($6.8k) waiting on your sign-off" \
     --suggest api_sync \
     --label office \
     --draft-prompt "Review Provi restock PO #4471 ($6,821.40, net-30, due Fri). I reconciled every line item against the last 3 orders and pricing matched. If it checks out, approve it and confirm the delivery window with Provi."
   ```
   The CLI resolves your agent id + token + server URL from the env the
   runner injects (`CLAWMEETS_AGENT_ID`, `CLAWMEETS_AGENT_TOKEN`,
   `CLAWMEETS_SERVER_URL`). No `--token` flag needed.
2. Reply ONE line in `user-communication`, e.g.
   `Flagged "Approve the Provi PO" to your desk — open it to review and dispatch.`
   Don't restate the whole task; the plate item IS the deliverable.

To see what's already on the plate (e.g. to avoid publishing a duplicate), and
the owner's label vocabulary before you tag anything:

```bash
clawmeets todo list
clawmeets todo labels list
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

**Labels are the same split.** You may READ the vocabulary —
`clawmeets todo labels list` works for you and you should run it — but every
other `labels` verb curates the owner's own list and answers
`Error 401: Invalid token`. If the user asks you to rename, merge, recolour,
reorder or delete a label, say it's their assistant's job rather than retrying.
Applying an existing label with `--label` on your own publish is always yours
to do.
