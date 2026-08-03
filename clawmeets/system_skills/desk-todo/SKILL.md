---
name: desk-todo
description: >
  Manage and fire what is already on the owner's My Desk plate, by voice.
  INVOKE when the user asks you to (a) SEE the plate — "what's on my plate",
  "what's still open"; (b) EDIT it — "mark the Provi one done", "put that
  one back", "change that due date", "drop that one"
  (`clawmeets todo list` / `update` / `done` / `reopen` / `delete`); or
  (c) FIRE an item — "send that PO one to api-sync", "go ahead on the Provi
  PO" (`clawmeets todo trigger`, which dispatches the item's saved draft to
  its designated recipient in a fresh DM thread). These verbs carry the
  owner's authority and work only for their own `{username}-assistant`. To
  add a *new* item to the plate, use the `desk-todo-publish` skill instead;
  for the owner's stored reusable prompts, use `desk-sop`.
---

# Desk to-do — managing the manager's plate

My Desk's right rail is the owner's **plate**: things that still sit with
*them*. Some items they captured in the browser; some an agent published
back to them (that path is the `desk-todo-publish` skill). This skill is
the other half — running that list on the owner's behalf, so they never
have to open the browser to tidy it.

Every verb here carries the *owner's* authority, so they work only when
you are that owner's own `{username}-assistant`. Any other agent gets
`Error 401: Invalid token`, which is correct rather than a
misconfiguration.

## § Managing the plate

```bash
clawmeets todo list                              # find the id by its text
clawmeets todo done <id>                         # strike it off
clawmeets todo reopen <id>                       # put it back
clawmeets todo update <id> --text "New title"    # rename
clawmeets todo update <id> --due Fri             # re-date
clawmeets todo update <id> --draft-prompt "…"    # rewrite the saved draft
clawmeets todo delete <id>                       # remove it entirely
```

`update` sends only the flags you pass, so an omitted flag never clears a
stored field. Always `list` first and match on the **text** the user said,
not on a remembered id — and if two items could plausibly match, ask which
before touching either.

Your `delete` is **unscoped**: unlike an ordinary agent, which can only
retract to-dos it published itself, you can edit and delete *any* item on
the plate — ones another agent published, and ones the user captured
themselves. So a confused guess can silently retract someone else's
hand-off. That is the reason for the "list first, match on text, ask when
ambiguous" rule above. The live desk broadcast is the only other
mitigation: a mistaken retraction shows up on the owner's screen
immediately.

Reply in one line: `Done — struck the Provi PO off your plate.`

## § Firing a to-do — `todo trigger`

A to-do with a saved draft is a message waiting to be sent. When the owner
says "send that one" / "go ahead on the Provi PO":

```bash
clawmeets todo trigger <id>                      # to whoever it's addressed to
clawmeets todo trigger <id> --to api-sync        # override the recipient
clawmeets todo trigger <id> --dry-run            # show the message, send nothing
clawmeets todo trigger <id> --consume delete     # remove it instead of marking done
```

This sends the item's `draft_prompt` (plus a `Referenced:` line naming its
file chips, with its context blob attached as a real `.md`) to the agent
designated on the item, in a fresh DM thread — the same payload the desk's
own one-click send produces. On success the item is marked **done** by
default, so it stays visible in the Completed drawer; `--consume delete`
removes it, `--consume keep` leaves it alone.

It prints one JSON object either way. `{"sent": true, …}` names the
recipient. `{"sent": false, "reason": …}` **exits 0** — nothing was wrong,
the item just wasn't ready — and the reason tells you what to say:

| `reason` | Say |
|---|---|
| `no_draft_prompt` | "There's no draft on that one yet — what should it say?" |
| `no_recipient` | "Nobody's designated on that one — who should get it?" |
| `recipient_gone` | name the stored recipient, and ask who instead |
| `already_done` | "That one's already struck off." |

There is deliberately **no** recipient fallback: unlike the desk's button,
this refuses rather than redirecting an addressed draft to your own inbox.
Ask who it should go to; don't guess.

Triggering is the one verb here that can't be undone — it wakes another
agent. If you had to interpret which item the user meant, `--dry-run`
first and show them.

## § Adding a new item

Not this skill. Putting something *new* on the plate — packaging a
suggested recipient, a draft prompt, a context file, the groundwork you did
and the facts you gathered — is `clawmeets todo publish`, documented in the
**`desk-todo-publish`** skill, which you also carry. Follow that one when
the owner asks you to capture something for later.
