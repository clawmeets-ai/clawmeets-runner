---
name: desk-todo
description: >
  Manage and fire what is already on the owner's My Desk plate, and curate the
  label vocabulary the plate is organised by, by voice.
  INVOKE when the user asks you to (a) SEE the plate — "what's on my plate",
  "what's still open", "what am I working on", "what's finished", "what's on my
  plate at the office", "what am I waiting on" (`clawmeets todo list`, with
  `--state new|working|completed` to filter by ticket state, `--label` to filter
  by context, `--archived`/`--no-archived` to narrow by disposal, and
  `--match-all` to require every label); (b) EDIT it — "file that one away",
  "put that one back", "change that due date", "drop that one", "tag that one
  @office", "take @home off that one" (`clawmeets todo list` / `update` /
  `archive` / `unarchive` / `delete`); (c) LINK an item to the work it spawned —
  "that one's being handled in the pricing project" (`clawmeets todo associate`
  / `dissociate` / `projects`); (d) FIRE an item — "send that PO one to
  api-sync", "go ahead on the Provi PO" (`clawmeets todo trigger`, which
  dispatches the item's saved draft to its designated recipient in a fresh DM
  thread); or (e) CURATE THE LABELS themselves — "what labels do I have",
  "rename @offce to @office", "merge @errands into @town", "make @home purple",
  "put @office at the top", "get rid of @errands" (`clawmeets todo labels list /
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
clawmeets todo archive <id>                      # file it away
clawmeets todo unarchive <id>                    # put it back
clawmeets todo update <id> --text "New title"    # rename
clawmeets todo update <id> --due Fri             # re-date
clawmeets todo update <id> --draft-prompt "…"    # rewrite the saved draft
clawmeets todo delete <id>                       # remove it entirely

clawmeets todo list --state working              # what's actually in flight
clawmeets todo list --state new --state working  # either (OR within --state)
clawmeets todo list --state working --label office   # AND across the two
clawmeets todo list --archived                   # what's been filed away
clawmeets todo list --no-archived                # only what's still on the plate
clawmeets todo list --label office --label home  # either label (a flat OR)
clawmeets todo list --label office --label home --match-all   # both, strictly
clawmeets todo update <id> --add-label errands   # add one label
clawmeets todo update <id> --remove-label home
clawmeets todo update <id> --label office --label errands  # replace the whole set
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

Filtering matches the rail: **every named label is OR'd together, flatly** —
`--label office --label home` reads "at the office or at home". Add
`--match-all` for the strict reading; it applies to labels only. At most 8
labels fit on one to-do; a 9th is refused whole, nothing is partly applied, and
the message says how many the item would have ended up with.

> **This rule CHANGED, and it changed quietly — read this once.** It used to be
> "OR within a kind, AND across kinds", so `--label office --label home --label
> next` meant *"next actions I could do at the office or at home"*: `next` was a
> **state** and narrowed the two contexts. Every label is a context now, so
> there is only one kind to group by and the same command is a **flat OR** — it
> returns everything carrying office, home, *or* next. Nothing errors; you just
> get a wider list than that phrasing used to produce.
>
> **`--state` is what narrows now.** "What's next at the office or at home" is
> `--label office --label home --state working` — a predicate over the item's
> DERIVED state, AND'd against the labels. That is a stronger answer than the
> old one, because it reflects whether the work is actually running rather than
> whether someone remembered to put a `!next` chip on it.

### § The two axes — say the right one

Every to-do carries **two independent facts**, and confusing them is the
mistake to avoid in this skill:

| | what it means | who sets it |
|---|---|---|
| `state` | `new` / `working` / `completed` — whether the WORK is running | derived from the projects linked to the item; **read-only** |
| `archived` | whether the OWNER has filed the row away | the owner, via `archive` / `unarchive` |

They never move each other, in either direction. Archiving a to-do does not
say the work finished, and a project completing does not file anything away.
So an archived to-do whose project is still running reads `working` — that is
correct, not a contradiction to reconcile.

There is **no verb that sets `state`** and there must not be one, and there is
no LABEL that sets it either — a label has never been able to, and since the
`state` kind was retired it cannot even appear to. If the user says "mark that
one done", they mean **archive it** — say "filed away", not "completed", or you
are reporting a fact about the work that you did not establish. `association_count` is how many projects are linked; a to-do
reading `new` with a non-zero count has links that no longer resolve, and
`clawmeets todo projects <id>` shows which.

```bash
clawmeets todo associate <id> <project_id>    # to-do id FIRST, then the project
clawmeets todo dissociate <id> <project_id>
clawmeets todo projects <id>                  # what it is linked to, and why
```

A **DM thread never completes**, so a to-do linked only to one reads `working`
for as long as it exists. That is expected. The disposal for such an item is
`archive`, not a wait for a Completed that is not coming.

Your `delete` is **unscoped**: unlike an ordinary agent, which can only
retract to-dos it published itself, you can edit and delete *any* item on
the plate — ones another agent published, and ones the user captured
themselves. So a confused guess can silently retract someone else's
hand-off. That is the reason for the "list first, match on text, ask when
ambiguous" rule above. The live desk broadcast is the only other
mitigation: a mistaken retraction shows up on the owner's screen
immediately.

Reply in one line: `Done — filed the Provi PO away.`

### Reading a row back

`labels` is a list of bare slugs. `labels_detail[]` carries each slug's display
name and whether it is registered — present **only when you filtered by
`--label`**, because that is the only path that already has the label list in
hand. If you didn't filter, run `clawmeets todo labels list`.

`labels_detail[].kind` is `"context"` on every registered row, always. The
`state` kind is retired, so the field no longer distinguishes anything and
**nothing should branch on it**. Do not describe a label as "a state" —
`wait-for` and `someday` look like states and are not; they are labels like any
other, and whether the WORK is running is the item's `state`.

`labels_detail[].registered: false` means that label isn't in the owner's list
at all — an agent invented it on a publish. Say so ("that one isn't in your
list") rather than treating it as one of theirs.

## § Curating the labels

The plate is organised by the owner's own **contexts** (`@office`, `@home`).
A leading `@` or `!` in what the user says means a label, never an agent — `!`
still works and is still stripped, because people go on typing it at labels
they have had for years.

```bash
clawmeets todo labels list                        # the owner's vocabulary
clawmeets todo labels add "@errands" --color pink  # --kind is optional now
clawmeets todo labels add "Waiting on bank" --color plum
clawmeets todo labels rename office --name "the office"   # display only
clawmeets todo labels recolor office --color teal
clawmeets todo labels reorder office home phone errands
clawmeets todo labels merge offce --into office   # fold a typo into the real one
clawmeets todo labels delete errands              # removes it from every to-do
```

**Every label is the owner's — nothing is built in.** `office`, `home` and
`phone` are a *starting* vocabulary, not a fixed one: any of them can be
renamed, recoloured, merged or deleted, and the owner can add as many more as
they like. Many owners also still have `Next`, `Wait For` and `Some Day` — those
were seeded back when labels carried lifecycle, and they are ordinary labels
now. Do not offer to clean them up; they are the owner's, they still group the
plate, and nothing about them is broken. Four rules to hold on to:

1. **`labels delete` never deletes a to-do.** It takes the label off every item
   carrying it and those items stay on the plate. The response says how many —
   say the number: *"Dropped @errands — it came off 6 items, all still on your
   plate."* Every label deletes the same way — `labels delete someday` is
   allowed, and so is deleting one the owner has had for years.

2. **There is one kind, `context`, and `--kind` is optional.** The `state` kind
   is retired: `labels add … --kind state` is refused with *"every label is a
   context now"*, and so is `--kind !`. If the user asks for "a state", they
   want to track whether the work is running — that is the item's `state`, it is
   derived, and `clawmeets todo list --state working` is the answer. Adding an
   ordinary label named `Waiting on bank` is also fine if what they want is a
   place to file things.

3. **A merge can still be refused for being across the same kind.** The error
   reads *"labels can only be merged into the same kind"* and reaches you only
   on a registry written before the retirement: such a row shows as a context
   but is STORED as a state, and the server refuses to fold it into a real
   context. That is the server protecting a row of the owner's, not a bug to
   work around — merge deletes the source row and rewrites every to-do carrying
   it. Leave it where it is, or `labels delete` it if the owner asks outright.

4. **`labels.registry_unreadable` means STOP, and say so.** Every curation verb
   refuses with it when the owner's label file exists on disk but cannot be
   read. Their labels are still on their to-dos and nothing has been lost — the
   *list* is what failed to load, and the server is refusing to overwrite a file
   it cannot read first. Do **not** "repair" it by re-adding the labels you
   remember: that is exactly the write that would replace the real ones with
   your guess. Tell the owner their label list needs a look and leave the file
   alone. To-do writes are unaffected — you can still capture, edit and label
   items, and any label you use simply stays unregistered until the list is
   back.

Deleting every label is allowed and is **not** a mistake to correct. Nothing
re-seeds, and you should not offer to put anything back unless asked.

On a pre-retirement registry, adding a label whose slug matches one of those
older rows comes back *"'next' is a state, not a context"*. The wording mentions
an axis the owner can no longer see; what it means is simply **they already have
that label**. Say that, and don't retry.

## § Firing a to-do — `todo trigger`

A to-do with a saved draft is a message waiting to be sent. When the owner
says "send that one" / "go ahead on the Provi PO":

```bash
clawmeets todo trigger <id>                      # to whoever it's addressed to
clawmeets todo trigger <id> --to api-sync        # override the recipient
clawmeets todo trigger <id> --dry-run            # show the message, send nothing
clawmeets todo trigger <id> --consume delete     # remove it instead of archiving
```

This sends the item's `draft_prompt` (plus a `Referenced:` line naming its
file chips, with its context blob attached as a real `.md`) to the agent
designated on the item, in a fresh DM thread — the same payload the desk's
own one-click send produces. On success the item is **archived** by default,
so it stays visible in the Archived drawer and `clawmeets todo unarchive` can
undo it; `--consume delete` removes it, `--consume keep` leaves it alone.

Firing does **not** link the thread it opens to the to-do, so the item's
`state` does not move. That is why archiving is the default disposal here.

It prints one JSON object either way. `{"sent": true, …}` names the
recipient. `{"sent": false, "reason": …}` **exits 0** — nothing was wrong,
the item just wasn't ready — and the reason tells you what to say:

| `reason` | Say |
|---|---|
| `no_draft_prompt` | "There's no draft on that one yet — what should it say?" |
| `no_recipient` | "Nobody's designated on that one — who should get it?" |
| `recipient_gone` | name the stored recipient, and ask who instead |
| `already_archived` | "That one's already filed away." |

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
