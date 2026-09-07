---
name: desk-todo
description: >
  Manage and fire what is already on the owner's My Desk plate, and curate the
  label vocabulary the plate is organised by, by voice.
  INVOKE when the user asks you to (a) SEE the plate — "what's on my plate",
  "what's still open", "what's on my plate at the office", "what am I waiting
  on" (`clawmeets todo list`, with `--label` to filter by context or state, and
  `--match-all` to require every one); (b) EDIT it — "mark the Provi one done",
  "put that one back", "change that due date", "drop that one", "tag that one
  @office", "take @home off that one" (`clawmeets todo list` / `update` /
  `done` / `reopen` / `delete`); (c) FIRE an item — "send that PO one to
  api-sync", "go ahead on the Provi PO" (`clawmeets todo trigger`, which
  dispatches the item's saved draft to its designated recipient in a fresh DM
  thread); or (d) CURATE THE LABELS themselves — "what labels do I have",
  "rename @offce to @office", "merge @errands into @town", "make @home purple",
  "put !next at the top", "get rid of !someday" (`clawmeets todo labels list /
  add / rename / recolor / reorder / merge / delete`).
  A leading `@` or `!` in what the user says means a LABEL, never an agent.
  These verbs carry the owner's authority and work only for their own
  `{username}-assistant`. To add a *new* item to the plate, use the
  `desk-todo-publish` skill instead; for the owner's stored reusable prompts,
  use `desk-sop`.
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

clawmeets todo list --label office --label next  # what's next at the office
clawmeets todo list --label office --label home  # either context (OR within an axis)
clawmeets todo list --label office --label home --match-all   # both, strictly
clawmeets todo update <id> --add-label next      # add one label
clawmeets todo update <id> --remove-label wait-for
clawmeets todo update <id> --label office --label next   # replace the whole set
clawmeets todo update <id> --clear-labels
```

`update` sends only the flags you pass, so an omitted flag never clears a
stored field. Always `list` first and match on the **text** the user said,
not on a remembered id — and if two items could plausibly match, ask which
before touching either. That rule extends to labels unchanged: run
`clawmeets todo labels list` before you merge or delete one.

`--add-label` / `--remove-label` adjust; `--label` replaces. **Prefer the
adjusting flags** — they can't clobber a label the owner just set in the
browser, and a retried command can't double-apply. Both are idempotent, so
adding a label an item already carries and removing one it doesn't are
successes, not errors. Passing `--label` together with either is refused
outright rather than guessed at.

Filtering matches the rail: **several contexts are OR'd, a state narrows
them** — `--label office --label home --label next` reads "next actions I
could do at the office or at home". Add `--match-all` for the strict reading.
At most 8 labels fit on one to-do; a 9th is refused whole, nothing is
partly applied, and the message says how many the item would have ended up
with.

Your `delete` is **unscoped**: unlike an ordinary agent, which can only
retract to-dos it published itself, you can edit and delete *any* item on
the plate — ones another agent published, and ones the user captured
themselves. So a confused guess can silently retract someone else's
hand-off. That is the reason for the "list first, match on text, ask when
ambiguous" rule above. The live desk broadcast is the only other
mitigation: a mistaken retraction shows up on the owner's screen
immediately.

Reply in one line: `Done — struck the Provi PO off your plate.`

### Reading a row back

`labels` is a list of bare slugs. To say "context" or "state" out loud, read
`labels_detail[].kind` — but note it is present **only when you filtered**,
because that is the only path that already has the label list in hand. If you
didn't filter and you need kinds, run `clawmeets todo labels list`; do **not**
infer a kind from how the slug is spelled. `wait-for` looks like a state and
`someday` looks like one too, but only the registry knows, and the owner may
have made either one a context.

`labels_detail[].registered: false` means that label isn't in the owner's list
at all — an agent invented it on a publish. Say so ("that one isn't in your
list") rather than treating it as one of theirs.

## § Curating the labels

The plate is organised by the owner's own **contexts** (`@office`, `@home`)
and **states** (`Next`, `Wait For`). A leading `@` or `!` in what the user
says means a label, never an agent.

```bash
clawmeets todo labels list                        # the owner's vocabulary
clawmeets todo labels add "@errands" --kind context --color pink
clawmeets todo labels add "Waiting on bank" --kind state --color plum
clawmeets todo labels rename office --name "the office"   # display only
clawmeets todo labels recolor office --color teal
clawmeets todo labels reorder office home phone next wait-for someday
clawmeets todo labels merge offce --into office   # fold a typo into the real one
clawmeets todo labels delete errands              # removes it from every to-do
```

**Every label is the owner's — nothing is built in.** `office`, `home`,
`phone`, `Next`, `Wait For` and `Some Day` are a *starting* vocabulary, not a
fixed one: any of them can be renamed, recoloured, merged or deleted, and the
owner can add new **states** as well as new contexts. Five rules to hold on to:

1. **`labels delete` never deletes a to-do.** It takes the label off every item
   carrying it and those items stay on the plate. The response says how many —
   say the number: *"Dropped @errands — it came off 6 items, all still on your
   plate."* That is true of states too; `labels delete someday` is allowed.

2. **A label's kind can't change after it's created.** To turn a context into a
   state, delete it and add it again — and say first how many items that will
   detach it from **and that you cannot put the new one back on them
   automatically**. There is no `set-kind` flag and you should not look for one;
   the two-step *is* the honest version.

3. **You can only merge a label into one of the same kind.** A context into a
   context, a state into a state. Anything else comes back "labels can only be
   merged into the same kind" — not a bug to work around, it is the same
   kind-change asking to be done the long way.

4. **A to-do should carry at most one state.** The server won't stop you writing
   two, so don't — check `labels list` for which of the owner's labels are
   states before you add one.

5. **`labels.registry_unreadable` means STOP, and say so.** Every curation verb
   refuses with it when the owner's label file exists on disk but cannot be
   read. Their labels are still on their to-dos and nothing has been lost — the
   *list* is what failed to load, and the server is refusing to overwrite a file
   it cannot read first. Do **not** "repair" it by re-adding the labels you
   remember: that is exactly the write that would replace the real ones with
   your guess. Tell the owner their label list needs a look and leave the file
   alone. To-do writes are unaffected — you can still capture, edit and label
   items, and any label you use simply stays unregistered until the list is
   back.

Deleting every state is allowed and is **not** a mistake to correct. The owner
is running contexts only; nothing re-seeds, and you should not offer to put them
back unless asked.

If the owner asks for a label that already exists on the other axis, the error
names which — *"next is a state, not a context"* — so say that rather than
retrying.

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
