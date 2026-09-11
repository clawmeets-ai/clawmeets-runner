---
name: plan
description: >
  The project's PLAN.md — the contract everyone works against. INVOKE when you
  need to read what was agreed, when you disagree with it, or when you are the
  coordinator and need to write it. One plan per PROJECT, never per to-do; a DM
  never has one. The rule that decides which verb you want: the
  coordinator DRAFTS and KEEPS the document, the user ALONE DECIDES, and a plan
  note connects exactly those two and nobody else. If you are neither,
  `plan update` and `plan note` both refuse you — say it in `shared-context`
  (before acceptance) or in your workroom (after) and the coordinator files it.
  Also invoke to record that work went off-contract (an ordinary note to the
  user, after acceptance), to answer a note addressed to you
  (`plan note --reply-to`), or to send a batch of notes for review.
---

# The project plan

`PLAN.md` in `shared-context` is the contract. It is **plain Markdown** — no
annotations, no state machine, no schema — and everything this system knows
about it is derived on every read. You can open the file directly from your
synced project directory with no network call at all; that is the fastest way
to read it and it is what the always-on prompt expects you to do.

Everything below is for **changing** it, or for saying something about it.

## Before anything else: which side of the line are you on?

**Drafting and deciding are two different rights, held by two different
people.** The coordinator DRAFTS and KEEPS the document. The user ALONE
DECIDES: acceptance, applying or rejecting a note, and any change to what an
accepted plan says.

| You are | You may | You may not |
|---|---|---|
| the project's **coordinator** (its keeper) | write any section **until the user accepts the plan** — after their first review round each spec change also needs `--why` (below); send review batches; close a note addressed to you (`--answered` / `--dismiss`); relay the user's questions in `shared-context` | **accept** the plan, **apply** or **reject** a note — including notes you wrote yourself. **Change what an ACCEPTED plan says** — that is a proposal (below). **Close your own note to the user before it has been sent** — the user has not seen it, so there is nothing to report. And you may not write `## Approval` while the approval note is open: that section is the user's |
| the **owner** (the human, from the UI or their own terminal) | everything the coordinator may, **and every decision**: accept, apply, reject | — |
| **any other agent** | read the plan; answer in `shared-context` (before acceptance) or in your workroom (after) — and when a consult asks you to sign an acceptance criterion, answer **per criterion**, not per plan | write the file (`plan update` is a **403**), send a batch (`plan review` is a **403**), and **file a plan note at all** (`plan note` is a **400**, in both phases) |

**This is not a permission to route around.** A specialist that wants a
different plan says so in the room; the coordinator files it as a note, the
owner accepts it, and the owner's write is what lands. A refusal here means
*use the other channel*, not *try again* — and each one names the channel in
its own error text.

### Two lines, and they do different things

**You write the plan until the user ACCEPTS it. You do not propose it.** Before
acceptance nothing has been ratified, so a diff is not a change *to* anything
the user agreed — and rejecting one hunk of five leaves the plan in a state
nobody designed. Plan sections are entangled: narrow M2 and you change what M4
depends on, and no diff view shows that. So the user reads the document.

The first line is **the user's first review round**, and it starts a
*record*, not a refusal. From there every `plan update` that moves what the plan
**says** must carry a changelog line:

```bash
clawmeets plan update <project> --section m2 --body-file ./m2.md \
  --why "M2 now owns auth setup, moved out of M3 — backend flagged that M3's endpoints can't be built before it"
```

That is enforced: without `--why` the write is a **400** and nothing lands. The
line is filed as a **receipt** — a comment on the user's plan, no diff, already
closed — and their next review round opens with every receipt since they last
read the document, under a banner naming the span it moved over.

**Write a changelog line, not a summary.** The user answers by quoting one of
your lines back at you. *"M2 now owns auth setup, moved out of M3 — backend
flagged the ordering"* is answerable. *"Incorporated feedback from all agents"*
is not: it hands them nothing to object to and pushes them into a full re-read
to find the one thing they disliked. That is the exact cost this shape exists to
remove, so a lazy `--why` gives back everything it bought.

The second line is **acceptance**, and it reverses the rule. From there a
`plan update` that moves the spec is a **403**; your text is not lost, it is
filed for the user as a note carrying your proposal and you get back the note
ids. Do not retry it smaller — a smaller spec edit is still a spec edit. The
diff is right *there* because there is finally a ratified baseline for it to be
a bounded delta against.

What needs no `--why` and is never refused, in either phase, because the rule
compares what the document *says* and not its bytes:

- ticking and unticking milestone checkboxes as work completes;
- editing `<!-- … -->` provenance comments;
- reflowing a table, reindenting a list, rewrapping a paragraph;
- rewriting `## Milestones` or any other `<!-- layer: detail -->` section.

So **keep reporting progress exactly as you did before.** The schedule is yours.

**Batch, don't drip.** One round of feedback folded into the document and sent
as one `plan review` costs the user one message and one read. The same changes
dripped out over five rounds cost them five.

**Why the receipt exists.** On one real project the coordinator made 26 direct
writes to the plan after the user's first review round closed — Goal,
Guardrails, Sequencing and every milestone — against zero records the user could
see. The plan was never accepted, so the accept-time refusal never engaged once.
The failure was not that it wrote. It was that 26 acts of judgement about what
the user's feedback "settled" left no trace. You are still not the judge of
that — but now you do not have to be, because you say what you did and they
answer it.

## The subcommands

Quoted verbatim from `clawmeets plan --help`, so this table and the listing
cannot drift.

| Command | What it does |
|---|---|
| `clawmeets plan conflicts` | Blocked writes, beside what each section says now, with the fix. |
| `clawmeets plan consult` | Sync the specialist roster into `shared-context` so you can reach them there. |
| `clawmeets plan create` | Create PLAN.md from `--body-file` or the one template. 409 if it exists. |
| `clawmeets plan list-notes` | List notes, filtered on any documented axis. `--thread` reads a thread. |
| `clawmeets plan note` | File a note: a comment, with or without a proposal. The user and the coordinator only. |
| `clawmeets plan resolve` | Close a note out: `--apply` / `--reject` / `--answered` / `--dismiss`. |
| `clawmeets plan review` | Send a review batch. Coordinator or owner only. |
| `clawmeets plan review-status` | Per addressee: sent / replied / resolved, and the round's state. |
| `clawmeets plan show` | Read the plan: the file, a section, a note, the index, the versions. |
| `clawmeets plan update` | Write one section. Owner or keeper only; no whole-body form, no retry. |

Every command takes the project as its first argument — a name, a display name
or an id.

## Reading

```bash
clawmeets plan show <project>                     # the whole document
clawmeets plan show <project> --sections          # the derived index: slugs, depths, box counts, criteria
clawmeets plan show <project> --section milestones
clawmeets plan show <project> --clean             # every HTML comment stripped, for pasting into a renderer
clawmeets plan show <project> --revision          # the revision label, the acceptance state, the spec digest
clawmeets plan show <project> --versions          # who wrote what, newest first
```

`--clean` is a **render, never a write**: the stored bytes are untouched.

Three things `--sections` tells you that nothing else does. A section's **slug**
is what every other command addresses it by, and two sections with the same
heading get `x` and `x-2` — if you see that, say which one you mean. `boxes` is
`(done, total)`, counting only that section's own extent; it is display, not a
state machine. And under those, the **criteria table**: every `AC-<m>.<n>` on
the plan, its section, whether its box is ticked, and its text. Those ids are
what `plan note --ac` takes, and an id listed twice is flagged `duplicate id` —
the same way a heading listed twice is flagged.

## Saying something: `plan note`

**The note channel connects the user and the coordinator. Nobody else is ever
an end of it.**

| `to` | `by` must be | Phase |
|---|---|---|
| `user` | the owner, or the coordinator | both |
| the coordinator | the OWNER only | both |
| any other agent | — refused | both |
| agent → agent | — refused | both |

One shape in both phases. **A specialist is reached over the `shared-context`
room, by the coordinator, on the user's behalf — never by a note.** The user
may still write "this one is for backend" inside a note; that is **prose the
coordinator reads and routes**, not an addressing mode the channel enforces.

That table is the whole rule, quoted in full on purpose. A partial restatement
of it is exactly what produces an agent that files a legal note in the wrong
phase — the phase column reads "both" on every row, and the temptation is to
assume the executing rule is looser or the spec rule is tighter. Neither is.

```bash
# THE COORDINATOR, to the owner — a plain comment
clawmeets plan note <project> --section milestones --to user \
    -m "M2 assumes the API is ready in week 1; it is not."

# THE COORDINATOR, to the owner — a PROPOSAL, the section's replacement text
clawmeets plan note <project> --section milestones --to user \
    -m "Splitting M2 so the API work is its own milestone." \
    --edit-file ./new-milestones.md

# THE OWNER, to the coordinator — the only direction that names an agent.
# `<coordinator-name>` is the coordinator's actual agent name, as `plan show`
# prints it; there is no role alias.
clawmeets plan note <project> --section goal --to <coordinator-name> \
    -m "Does this cover the mobile case? This one is for backend."

# EITHER PARTY, about ONE acceptance criterion. `--ac` resolves to the
# criterion's section AND its text, so the note is drawn against that one line
# instead of against the whole milestone that contains it. Pass `--ac` OR
# `--section`/`--quote`, never both — it already is a section and a quote, and
# a mismatched pair is refused rather than silently resolved.
clawmeets plan note <project> --ac AC-2.3 --to user \
    -m "@backend says a 409 here needs the idempotency key we cut from M1."

# either party, replying. `--reply-to` is the whole of it: the server reads the
# parent and fills in ITS section, ITS quote and the other end of the thread as
# `--to`, so the answer is drawn directly under the question it answers.
clawmeets plan note <project> --reply-to n-abc123 -m 'It does — see `## Not Authorized`.'

# replying WITH the change, which is the form the user can accept in one click.
# `--section` is not needed: with `--reply-to` it is read off the parent.
clawmeets plan note <project> --reply-to n-abc123 \
    -m 'Applied — `## Not Authorized` now says so explicitly.' \
    --edit-file ./new-goal.md
```

**A note is drawn as close to what it names as it can be**, and there are three
places, not two. Name a section (or `--ac`) and the server derives the excerpt,
so the note lands **against the line** — a proposal anchors to the base line it
changes, a note whose message opens on `AC-2.3` anchors to that criterion
provided the criterion is in the section you named, anything else to the
section's heading. Where no line can be derived it falls exactly one step, to
the **end of that section**, still beside the text it argues about. Only a note
that names nothing the document still contains reaches the **"Notes" list at the
bottom of the page**.

You do not have to do anything for any of that. Pass `--quote` yourself when you
want a specific line; an explicit one always wins. And `--quote` alone is
enough — with no `--section` the server looks the excerpt up and files the note
in whichever section holds it, so the two flags do not have to be remembered
together. The one way to file a note against no part of the document at all is
to pass neither, which is how you say *"about the document as a whole"*.

**The slug you pass must be one the document has.** A `--section` that does not
resolve is refused — `clawmeets plan show <project> --sections` prints the
slugs — unless your `--edit-file` opens on a heading that would create exactly
that section, which is the legitimate way to propose a new one. If your note
carries a `--quote` the server will look the section up from it first, so a
section that was retitled out from under you costs you nothing.

`clawmeets plan list-notes <project> --ac AC-2.3` reads back every note filed
against one criterion — the argument's whole history in one command, without
scanning a milestone.

**If you are a specialist, none of the above is yours.** `plan note` answers
you with a `400` that names the room instead: `shared-context` while the plan is
being specced, your own workroom once it is accepted. A `400` rather than a
`403` on purpose — `403` reads *stop*, `400` reads *re-form the call*, and
re-forming is the truth: there is a legal act, it is just not this command.

Seven things worth knowing before you type one.

**Answering a note means `--reply-to` it.** Not a fresh note that happens to be
about the same thing. Do **not** answer by filing a new note and then running
`plan resolve --answered` on the original — that is two acts where one is
correct, and the answer it files is worse than the one act: with no `--reply-to`
there is no parent to read, so it carries no section and no quote and lands in
the "Notes" list at the bottom of the page instead of under the line being
argued about, and with no `--to` derived from the parent it is *recorded and
never sent*, so the person who asked never hears it. `--reply-to` does all of
that from the one flag — **including closing the parent**, which lands on
`answered` in the same write. That is the whole of the second act, so there is
no `plan resolve --answered` to run afterwards and nothing left open on the
asker's screen for them to dismiss. The go-note is the one exception: replying
to it leaves it open, because closing it releases the execution gate and only
accepting it means yes.

**Answer with the change, not just about it.** When your answer changes the
plan, file it as a proposal (`--edit-file`) rather than prose. A proposal
renders as a diff with **Accept** and **Reject** beside it, which is a decision
the user makes in one click; prose is a paragraph they then have to ask you to
act on. Reserve prose for an answer that genuinely changes nothing.

**A proposal is the section's whole replacement text.** Write out what the
section should say, in full, and pass the file. No surface anywhere accepts a
diff — not a flag, not a field, not a route — because a diff carries line
positions and a plan document moves under you.

**`--to` has exactly two legal values and which one depends on who you are.**
The owner writes `--to <coordinator>`; the coordinator writes `--to user`.
Repeating it is still supported and still produces *N sibling notes with
independent statuses*, but under the table above there is no longer an N > 1
worth writing. **Omit it and the note is recorded and never sent** — a real and
useful thing to do, and why a note nobody was addressed to never badges anyone.
**On a reply this is not what omitting it does.** With `--reply-to`, an omitted
`--to` resolves to the other end of the thread — whoever asked — because an
answer addressed to nobody is a failure, not a choice. Pass `--to ''` when you
really do mean recorded-and-never-sent.

**`--to user` is what reaches the human.** It is the number on their desk card,
and while a project is executing an open one holds new work rooms. Use it when
you need a decision. Do not use it to report progress — that belongs in chat,
not on the plan. ("Reply" has a precise meaning on this page: a note filed with
`--reply-to`, not a chat message.)

**A number you re-measured is not a decision.** Durations are not in the
document at all (`create-project` rule 4), and the current number is yours as
keeper — say it in the round-up prose. It becomes a note **only when the
re-measurement changes what gets built**: a milestone dropped, a criterion cut,
work re-sequenced. Then the note is that scope call, named as such, and the
arithmetic is the reasoning inside it rather than the thing being asked. If you
can absorb the change without dropping anything, the user does not hear it as a
decision at all. The tell is blunt: **if you can predict the answer, it was not
a decision.** A note that comes back *"accept your recommendation"* was a
progress report wearing a decision's clothes, and it cost the user a turn and
held their work rooms while it sat open.

**Filing a note is not sending it.** A note badges the card; `clawmeets plan
review` is what renders it and posts it into `user-communication`, which is
what the user actually reads. File, then send.

**A note is never refused for being out of date.** You can write one against a
view several revisions old; a stale objection is still a real objection, and a
note cannot clobber anything.

## Recording that work went off-contract

**Anchor it to the criterion it broke, not to the milestone that was building
it.** `--ac AC-<m>.<n>` fills in both the section and the quote for you, and
because criteria live in the unmarked `## Acceptance Criteria` section it lands
in the spec layer — which is the layer the execution gate and the editor's red
comment box read. A departure anchored to `milestones` is filed into the layer
you were free to rewrite anyway, so it records but does not hold.

```bash
# A REPORT. The work already diverged; there is nothing here to accept.
clawmeets plan note <project> --ac AC-1.2 --to user \
    -m "Two of the six recommendations rest on pricing we could not source, so they cite no opener-able source. AC-1.2 is not met and cannot be met from public data."

# AN ASK. You want the contract moved, so the new section text rides along.
clawmeets plan note <project> --ac AC-3.2 --to user \
    -m "AC-3.2 needs a clause: useLogout owns its navigation, or logout lands on \"Project not found\". Why, below." \
    --edit-file ./ac-3.md
```

**"Shipped 4 slides, the plan says 6" is NOT a deviation and must not be filed
as one.** Slide count was never the promise — a criterion states a behaviour,
so the only thing that can deviate from it is a behaviour that does not hold.
The example above is a real one: a criterion the work cannot satisfy. If you
find yourself filing a note about a count, a filename or a format, check what
the criterion actually says before you spend the user's attention on it.

**A deviation has two shapes and only one of them is prose.** If your note asks
the user to change what the plan *says*, it is a proposal and it carries
`--edit-file` — the same rule as answering with the change above, and for the
same reason: with the replacement text the row renders a diff with **Accept**
and **Reject**, and without it the row offers Dismiss and Reply and nothing
else. A note that argues *the proposal* and *what happens if you refuse it* in
prose has named a decision and then given them no way to make it; agreeing costs
them a round trip to ask you for text you had already written. Report-only is
the right shape when nothing in the document should move — not when you have not
got round to writing the section out.

**There is no `--deviation` flag. That IS the note.** Once the user has accepted
the plan, a coordinator's note to them is by construction a report that the work
and the contract have parted company — there is no second kind of message for it
to be, so the flag had nothing left to assert and was removed. Before acceptance
nothing has been agreed, so the same note is an ordinary comment or proposal and
no flag could honestly have said otherwise.

A deviation makes the parting **recordable and visible**. Nothing here makes one
**detected** — no checker compares work to plan, and pretending otherwise would
be worse than the gap. File one when you know; it shows on the owner's card with
the clause it departed from (`--ac`, or `--section` if what moved is not a
criterion), and they either move the contract or hold the line. **Moving it is one click only if the replacement
text rode along.**

**The coordinator files it, not the agent that hit the problem.** A deviation is
a plan note, so the two-party table above applies unchanged: a specialist raises
it with the coordinator in the workroom and the coordinator files it.

## Writing, if you may: `plan update`

```bash
clawmeets plan update <project> --section goal --body-file ./goal.md
clawmeets plan update <project> --section milestones --append --body-file ./m4.md
clawmeets plan update <project> --section m2 --title "M2 — API and ingest"
clawmeets plan update <project> --section scratch --delete
```

**One section at a time. There is no whole-body form and there is no retry.**

The body you send is that section and nothing after it, heading line included.
Regenerating a section into a file and copying one heading too far is refused
with a `400` naming the heading you duplicated — two headings slug to one id,
and the copy that comes first in the document takes it, so every later note,
quote and edit addressed to that section would land on the wrong one.

A write carries the section's current text as its precondition. If somebody
changed that section since you read it, the write is refused, **nothing is
written**, and a conflict note is filed carrying your text so it is not lost.
`clawmeets plan conflicts` lists every refusal beside what the section says now,
with the exact command to close it. **Do not loop.** Read what it says now,
decide, write once.

Two writers on two *different* sections never collide, in either order. That is
the whole reason the unit is a section.

**Once the user has reviewed this plan, a spec change here also needs
`--why "<one changelog line>"`** — a `400` without it, and the line becomes the
receipt they read at their next round. **Once they have ACCEPTED it**, the same
write is a `403` and your text comes back filed as a proposal instead. A
checkbox tick, an HTML comment, a reflow and a `## Milestones` re-cut need
neither, in either phase. See *"Two lines, and they do different things"* at the
top.

## Closing a note out: `plan resolve`

```bash
clawmeets plan resolve <project> n-abc123 --apply        # take the proposal into the document
clawmeets plan resolve <project> n-abc123 --reject -m "We need the wider scope."
clawmeets plan resolve <project> n-abc123 --answered     # the addressee has replied
clawmeets plan resolve <project> n-abc123 --dismiss
```

**Notes are never deleted — the ladder closes them.**

**`--apply` and `--reject` are the OWNER's, and only the owner's.** They are
decisions about the document, and the whole shape of this system is that the
coordinator drafts and keeps while the user alone decides. A coordinator that
could apply and reject could dispose of the notes it wrote itself. Both are a
`403` for every agent, on every note, including a note the agent is the
addressee of.

**You may not close your own note to the user before it has been sent.** One
shape only: you wrote it, it is addressed to `user`, and no review batch has
carried it yet — so the user has never seen it. Closing it there decides it on
their behalf, which is the same authority `--apply` and `--reject` withhold,
reached through a report. Send the batch and let them decide, or file the
correction as another note in the same batch. Once a note HAS been sent,
`--answered` and `--dismiss` on it are reports again and are open to you.

**Otherwise `--answered` and `--dismiss` are unchanged** — they are reports,
not decisions, and a note's own addressee may still close its own. Under the
two-party rule the only agent that can be an addressee is the coordinator, so
on any note filed from here on these two verbs are exercised by the coordinator
and the owner and by nobody else. A note filed *before* the two-party rule and
addressed to a specialist stays closable by that specialist, which is the point
of not gating a report.

**A reply closes the note it replies to, from either door** — the user's review
tray and `plan note --reply-to` both land the parent on `answered` in the same
write that files the outgoing question. `answered` is the ladder's own word for
*the addressee has replied*, and that is exactly what a reply is, so there is
no `--answered` to run afterwards.

The two doors used to disagree, and because the tray is the user's alone the
disagreement ran one way: the user's reply closed the coordinator's question,
while the coordinator's reply left the user's question open forever with a
`Dismiss` button on it. The reason given was that the plan editor draws only
OPEN notes, so closing a question takes it off screen and leaves the answer
standing alone — real, and now paid for on the answer instead: the reply's
*"Answering your question"* line carries the parent's own first line, so the
exchange still reads with the question closed.

**Two notes do not close on reply.** The go-note, because closing it releases
the execution gate and accepting it is the only act that means yes; and a
proposal of your own to the user that has never been sent, because the user has
not seen it yet and closing it would decide it on their behalf.

**A user's note is a REQUEST TO YOU, and you close it when you have dealt with
it — replying is not the only way to deal with something.** A reply closes it
for free, so most of the time there is nothing extra to do. But when the right
response is an ACT rather than an answer — you re-cut the milestone they
questioned, or you file a fresh proposal that does what they asked — the reply
never happens and the note sits open. Close it yourself: `plan resolve <id>
--answered` when you did the thing, `--dismiss` when you decided not to and the
reason belongs elsewhere. You are the addressee, so both are already yours.

**Do not leave it for the user to dismiss.** An open note addressed to you
badges nobody and blocks nothing — but it is still on their plan editor with a
control on it, and asking the person who made the request to also file the
paperwork closing it is the whole of what made this channel feel like a chore.

If the section changed since the proposal was written, `--apply` is refused and
prints both texts. That is a second, explicit look, not an error.

## Sending a batch: `plan review` (coordinator or owner)

```bash
clawmeets plan review <project> --dry-run             # see exactly what would be posted
clawmeets plan review <project> -m "Answers to your three questions."
clawmeets plan review-status <project>
```

One batch is one write and one message per addressee. A sent note is not sent
again unless its text changed, so overlapping batches carry only what the
earlier one did not.

**Under the two-party rule every batch goes to `user-communication`, and no
room is created.** The addressees are the user and the coordinator, which is
what `review_room_for` routes there. So the question "which room?" has one
answer for every batch you will send.

**This is what puts a note in front of somebody.** `plan note` moves the
number on the desk card; `plan review` renders the note — the section as it
stands now, the diff, the thread, and the four commands to respond with — and
posts it. Run it after you file, every time. Notes you file and never send are
a badge with nothing behind it.

### One proposal per section per batch

**Two proposals on one section in one batch is a bug you wrote, not a conflict
the user gets to resolve.** A proposal replaces its section WHOLE. Stage two of
them against `m2` and the user can tick Accept on both — but at submit only the
last survives. The other resolves `rejected` with the reason `superseded by
@<name>'s proposal on the same section`, and everything it carried is gone,
into a rejection reason nobody reads. Nothing merges them; that is deliberate,
and it assumes you did not file both.

**So group by section BEFORE you file, not after.** Three specialists coming
back with three fixes to three criteria of Milestone 2 is ONE note carrying ONE
rewrite of Milestone 2 that satisfies all three, with the criteria it covers
enumerated in the comment so the user can check the fold:

```bash
clawmeets plan note <project> --section m2 --to user \
    -m "Folds three consult answers: AC-2.1 gains the fixture it is demonstrated by, AC-2.3 drops the p99 nobody would own, AC-2.4 moves to frontend." \
    --edit-file ./m2.md
```

Composing one careful section out of three inputs is drafting, and drafting is
yours. It is also the last point where it is possible: by submit time the
server holds N opaque blobs of section text and can only keep one.

**`--ac` does not narrow what a proposal replaces.** It resolves to the
criterion's section AND its text as a quote — the quote narrows what the
*reader* sees, the section is still the whole milestone, and the replacement is
still the whole milestone. Per-criterion anchoring stays right for a **comment**
(a note with no proposal never collides, and the fine address is worth keeping).
It is wrong for a proposal, and reliably so: any milestone with two contested
criteria produces the collision above every time.

**Re-filing on a section REPLACES your own unsent proposal on it.** File a
second proposal on the same section, addressed to the same person, before the
batch goes out and the first is closed for you — `dismissed`, with a reason
naming the note that replaced it. You do not clear it, ask the user to clear
it, or mention it in the batch message. It applies to every door: a `plan note
--edit-file`, and equally the deviation note the server files on your behalf
when `plan update` is refused on an accepted plan, which is the case you cannot
avoid by being careful.

So the last write wins and that is the whole rule. **Send the section as it
should read now**, not as a delta against what you filed ten minutes ago — the
row that survives is the one the user applies, and applying an older one after
it would undo part of the newer.

**What is still not a take-back**, and the distinction is worth holding: closing
your own unsent proposal with `plan resolve --dismiss` remains a **403** (`may
not close <id> — you wrote it, it is addressed to the user, and it has not been
sent yet`). That ends with nothing on the user's desk — a question you raised
and disposed of behind them. Replacing ends with the same question in its
current wording. If you want a filed proposal gone and have nothing to put in
its place, send it and let the user decide.

**Once it HAS been sent, it is theirs.** A note that went out in a batch is
never superseded by a later one; both stand until the user closes them. That is
the round you save by re-filing before you send, and the reason `clawmeets plan
review <project> --dry-run` is worth reading first.

## Reaching a specialist: `plan consult`

```bash
clawmeets plan consult <project>          # seat the roster in `shared-context`; print who is in it
```

A specialist is never reached by a note. `shared-context` is the room — the one
that also holds `PLAN.md` — and it exists from the moment the project does, but
seeded with the coordinator alone. During `spec-ing` you cannot invite anyone
into it yourself: the server refuses every `create_room` until the user accepts,
and the auto-add that fills the room lives inside that same refused handler.
This command fills it for you, with every agent on the project's invitable
roster. It is idempotent per participant — an unchanged roster appends nothing,
an agent registered since your last run is seated by the next one — and it posts
nothing.

Once it exists, consult with **ONE** `reply` into it that `@`-mentions every
agent you want an answer from. One message is one batch and one wake-up when
they have all answered; a **second** addressed message into the same room
collides on the batch key and is dropped with a warning.

**Consult BEFORE the user's first review round, not only after it.** The
coordinator's spec-stage contract ends its first turn here: draft the plan,
`plan consult`, one `@`-mentioning reply, stop. The `BATCH_COMPLETE` that
follows is the turn the answers are folded in and the still-open questions go
to the user — so what the user reviews is a draft the domain agents have
already read. The one exception is a project whose roster is the coordinator
alone: `plan consult` lists YOU ALONE in the room — it seeds the coordinator
first, so it never prints `(nobody)` — and a reply mentioning no one opens no
batch, so stopping there would mean nothing ever wakes you. In that case go
straight to the notes and the review batch in the same turn.

### Ask each agent to sign the criteria it will be judged by

**The consult message has a required shape, and it is organised BY AGENT, not
by topic.** Under each name you `@`-mention, put:

- the milestones that agent owns, by id;
- under each milestone, its acceptance criteria, **quoted from the plan
  verbatim**;
- three questions per criterion, and only these three:
  1. Can you satisfy `AC-x.y` as written — yes / no / not mine?
  2. If no: what should it say instead?
  3. What will you **show** to demonstrate it?

**Why the executor and not the roster.** A criterion is a promise somebody has
to keep. Asking everyone *"any comments on the draft?"* collects opinions about
a document; asking one named agent *"can you keep this promise, and how will
you prove it?"* collects a commitment. Only the second is checkable before
the work starts, and only the second is a signature you can point at later.

**Question 3 is not a formality.** A criterion whose demonstration nobody can
name is not testable, and that gap is the single most common source of the
post-approval argument this exists to prevent — the coordinator and the
specialist each held a different idea of *done* and neither knew it. Ask about
every criterion, including the ones you are confident about; the confident ones
are cheap to answer and are not reliably the ones that hold.

The coordinator-alone exception above applies unchanged: a roster of one has
nobody to sign, so go straight to the notes and the review batch in the same
turn.

**If you are the specialist, answer per criterion, not per plan.** One line
each: the id, yes / no / not mine, and the demonstration. Prose approval of the
whole draft is exactly what this replaces and is not an answer. Answer about
the criteria you **own** — *"not mine"* is a real answer and a useful one,
because it tells the coordinator a milestone has no executor, which is a scope
problem the user needs to see before acceptance rather than after.

On the way back, split what came in: an answer that settles something goes
into the document with `plan update` — you are the keeper — and only what needs
the *user's* judgment (a trade-off, a scope call, two agents who disagree)
becomes a note. For the signatures specifically:

**The split survives the user's first review round; what changes is the
paperwork.** Before they have seen anything, a settled answer goes into the
document silently. After their first round it still goes into the document —
you are the keeper right up to acceptance — but the write carries
`--why "<changelog line>"`, and they read that line rather than a diff. Only
after ACCEPTANCE does the split go away for the spec layer: from there **every**
answer that changes what a criterion promises is a proposal, including the ones
you are confident about.

**The evidence is the exception, and it is the row that used to generate most
of the traffic.** This table's first row once said to fold the demonstration
*into* the criterion. That welded the evidence into the contract text, so every
later substitution of a test or an artifact became a change to what the plan
*says* and therefore a deviation the user had to accept. Keep the two apart and
the problem disappears with no ceremony: the **behaviour** is the criterion's
sentence and is theirs; the **evidence** is a `<!-- evidence: … -->` comment
beside it and is yours, in both phases, because comments never move the digest.

```markdown
- [ ] **AC-2.1** — A registered user with valid credentials receives a session
      that survives a server restart.
      <!-- evidence: tests/test_auth_session.py::test_survives_restart -->
- [ ] **AC-3.1** — A reader can name the formats gaining traction and open
      posts that show each one.
      <!-- evidence: deliverables/ig/trends.md — two live post links per format -->
```

A test is the evidence where one exists. Where none can, the evidence is
whatever the user could check the claim against — a file, a link, a sample
someone can open. What it must never be is a restatement of the criterion.

| what came back | what you do with it |
|---|---|
| yes + a demonstration | fold the demonstration into the criterion's `<!-- evidence: … -->` comment, **never into the criterion's own sentence**. A comment is inert to the spec digest, so this lands with `plan update` in every phase — before the user's first review, after it, and after acceptance — and needs no `--why`, no note and no decision from them. |
| no | a note to the user, anchored to it: `plan note <project> --ac AC-x.y --to user -m "…"`. Carry the specialist's own words and the alternative it proposed; the scope call is the user's. |
| not mine, or no answer at all | a note to the user. A milestone with no executor is a roster gap and must not reach acceptance unstated. |

**Never carry an unsigned criterion silently into the review batch.** The whole
point of consulting before the user's first round is that what the user reviews
is a draft the executing agents have already agreed to.

**A consult fold is the case that collides.** One `BATCH_COMPLETE` hands you
every specialist's answer at once, and several of them routinely land in one
milestone — so this is where "one proposal per section per batch" earns its
keep. Sort the answers by the section they change first, then write one
proposal per section.

Then send the batch, and say in `user-communication` what changed and what is
still open; the rendered batch is diffs and threads, and on its own it is a
changelog the user has to interpret unaided.

## Acceptance: a note the user accepts, and a line in the document

**There is no approve command and no approve button.** When a project is
created the server files one note against the plan's `## Approval` section,
addressed to the user, proposing to replace its body with:

```
User approves the plan.
```

Accepting that note IS accepting the plan. It applies the proposal, so the
section stops reading `_Not yet approved._` and starts reading the line above;
in the same transaction the acceptance is stamped, the note closes, and the
project's blocking-note count drops — which is what releases the coordinator.
One verb for every decision the user makes about their plan, instead of a
special one for the first decision.

```bash
clawmeets plan list-notes <project> --to user     # the go-note's id is in here
clawmeets plan resolve <project> <id> --apply     # this is the acceptance
```

**Acceptance is one-way. There is no undo, and that is not a gap.** An accepted
plan is not un-decided; it is CHANGED, by filing a note the user applies — and
that apply re-signs the acceptance, so the record follows the document rather
than freezing at the moment of the first yes. If the user wants to stop work
while something is renegotiated, file a note addressed to them: an open note to
the user **on a spec-layer section** is what the execution gate counts, so
filing one holds the project without touching the acceptance at all. Anchor it
to what is actually being renegotiated — a note on a `<!-- layer: detail -->`
section does not hold anything, for the reason in *The two layers* below.

### If you are the coordinator: read the section, not the chat

`## Approval` is your go signal and nothing else is. The steady-state prompt
prints what it currently says on every turn, so you do not have to go looking —
but if you want to check it yourself:

```bash
clawmeets plan show <project> --section approval
```

**A user saying "looks good" or "go ahead" in chat has not approved the plan.**
Neither has a thumbs-up, and neither has silence. Those are sentences you would
have to interpret, and the interpretation that starts work is the expensive one
to get wrong. The document is not a sentence to interpret. If they say
something like that while `## Approval` still reads `_Not yet approved._`, tell
them where to accept and stop.

The server holds the same line from the other side: while the go-note is open,
`create_room` is refused, so a coordinator that reads the chat wrong cannot
open a workroom anyway. What you lose by ignoring this section is not the
guarantee — it is knowing *why* you were refused, and a refusal you cannot
explain is one you will narrate as though it succeeded.

**You may not write that line yourself.** While the go-note is open, a
`plan update` that changes `## Approval`'s text is refused, and so is one that
removes its heading — the acceptance is a diff, and a diff needs its heading to
land on. Every other section is yours to rewrite as freely as ever, including
the seed template's Goal and Not Authorized; `## Approval` is the one exception,
and only until the plan is accepted.

Point the user at the note and **stop**. You do not have to stay in the turn to
hold the project open: their acceptance is a change of state and it wakes you
by itself.

### `--as-user`, the owner's second door

```bash
clawmeets plan resolve <project> <id> --apply --as-user
```

`--as-user` authenticates as the **human**, not as the agent — it sends the
owner's own credential instead of the agent token, so the server sees the owner
making the decision, which is what makes it legal.

**It is not a way around the section above.** It exists for one caller: the
user's own assistant, in the user's own DM, acting on a specific request the
user just made. Its rules live in the **`decide-plan-as-owner`** skill, which
is the only documentation of when it may be used — read that before you type
it. In short: only in your owner's own DM, only on an explicit request for
*this* plan, and a standing instruction ("approve plans when they look done")
is not a request. If you are a project coordinator reading this, `--as-user` is
not yours; ask the user.

**An apply on an already-accepted plan re-signs it.** The same verb, doing the
same thing it always did — the user is putting their name to a piece of text —
except that the plan already carries their name and this moves it onto what the
document now says. It is why there is no revoke to miss: `## Approval` stays
true, and the *"the plan changed since approval"* warning means the one thing it
can still mean, that the document moved **without** them.

## The two layers

**A section is SPEC if changing it would change what the user said yes or no
to. It is DETAIL if it only changes how the same yes gets delivered.** That one
sentence decides every case, and it is deliberately not about software — it
works the same for a research plan, a design brief or an operations rollout.

| layer | holds | marker | may the keeper edit it after the user's first review? |
|---|---|---|---|
| **Intent** | Goal, Not Authorized, Acceptance Criteria | none (spec is the default) | no — refused, filed as a proposal |
| **Approach** | the shape of the answer, *only where the user chose the shape* — a named architecture, a named report format, a named method | none | no |
| **Execution** | milestones, sequencing, ownership, evidence | `<!-- layer: detail -->` on the heading | yes, freely |

**Approach is the layer people get wrong in both directions.** If the user said
"deliver it as a board memo, not a deck", that is spec — they chose it. If they
said "figure out the right structure", it is detail and it is yours. The test is
*did the user decide this*, not *is this important*.

`## Milestones <!-- layer: detail -->` is the marked section in the seed
template, and every `### M<n>` inside it inherits the marker. Unmarked means
spec, so a plan that carries no markers is locked exactly as strictly as it was
before layers existed.

**Two edits inside a detail section are still refused.** Changing a section's
`layer:` marker — because otherwise you could unlock any section with one write
that looks inert. And dropping an `<!-- advances: … -->` claim so that a
criterion ends up claimed by nobody — because that is how a re-cut quietly
deletes the work behind a promise. Re-cutting, splitting, merging, reordering
and re-assigning milestones are all free.

**You lose nothing except the user's automatic sight of a re-plan.** Nobody will
be asked to accept your new schedule, so **say in `user-communication` that you
re-planned and why.** The plan's version history records it either way; the
announcement is what makes it visible in time to matter.

**A note anchored to a detail section does not stop the project.** The execution
gate counts open notes to the user in the **spec** layer only, and the plan
editor's comment box goes red on the same test. The note is still filed, still
sent, still on their screen and still answerable — it just does not halt every
agent on the project while it waits.

That is the layer being consistent with itself rather than a loophole: you did
not need permission to make that edit, so asking for it should not be able to
stop the work. **The completion is upstream — decide it, do it, and say so.** If
you genuinely want the user's taste on something inside your own layer, ask in
`user-communication`, where a question costs them a reply instead of costing the
project a stop.

**And the corollary, which is the one that bites:** if the thing you are
departing from is a criterion, anchor the note to the criterion — `plan note
--ac AC-2.3` fills in the section and the quote for you — and not to the
milestone that happened to be building it. `## Acceptance Criteria` is unmarked
and therefore spec, so a departure filed that way lands where it will be seen
and where it will hold. A real objection anchored to `milestones` is silent.

## `## Not Authorized` — the fourth question

Goal says **why**. Acceptance Criteria say **what desired behaviour looks
like**. Milestones say **who does what**. `## Not Authorized` says **what the
user's yes did not buy** — and it is a claim about authority, not about
behaviour, which is why it is neither of the other two.

**One test decides whether a line belongs there: violating it cannot be undone
by more work.** An email sent, an order placed, a deploy shipped, data published
or deleted, money spent. Extra code is deletable and extra research is
discardable, so scope creep is not this; a quality bar is a criterion; the
sequencing you chose is milestone ordering. Sequencing the *user* mandated is a
denial and goes here phrased as one — *"no deploy before the security review
passes"* — because the milestone layer is yours to re-cut and would not hold it.

**Why it is a section and not a criterion.** The deviation channel fires when
the spec MOVES: someone notices Goal or Acceptance Criteria no longer describe
the work, and files a note to the user. An agent that submits the order has not
moved the spec — it has **over-satisfied** it. Sending the questions completes
the deliverable; placing the order is more of what the Goal asked for. No
criterion changed, so there is nothing to notice and nothing to file, and by
the time a note could be read the act cannot be taken back. That asymmetry is
the whole justification for the heading — not importance, and not frequency.

**Unlike every other section, this one is injected verbatim into every agent's
prompt** — the coordinator's and each worker's — capped at a few lines. Write
it short. Most plans should say *"None."*, and a plan that says "None" has not
skipped anything.

**Legacy plans carry this under `## Guardrails`.** That heading is read as a
fallback and never written, so a plan created before the rename keeps working
without being rewritten. If you inherit one, expect it to be full of scope
rules and quality bars that now have better homes.

## Two conventions this system teaches and does not enforce

**Acceptance criteria live in ONE `## Acceptance Criteria` section**, grouped
under `### G<m>` headings and labelled `AC-<m>.<n>` — where `<m>` is the
**group**, not a milestone. Milestones cite the ids they advance
(`### M2: Session layer <!-- advances: AC-1.1, AC-1.3 -->`) rather than owning
them.

**They used to live inside their milestone, and the move is what makes the
layers mean anything.** "Milestones are detail, criteria are spec" is
self-contradictory while the criteria are nested in a milestone, because
freeing the milestone frees whatever is inside it.

**A criterion is an observable outcome, not an artifact and not a method.** One
plain sentence saying what must be TRUE for whoever receives the work, checkable
by them without being told how it was made. For software that is behaviour; for
research or analysis it is what the reader can now see and trust; for design,
what someone looking at it can tell. *"Produces 6 slides"* and *"a TSV with 10
rows"* are artifacts; *"every recommendation cites a source the reader can
open"* is a criterion.

**The altitude test is the whole rule.** If an ordinary change in **how the work
gets done** would break the criterion, it is too fine-grained. That is what this
section is protecting: a criterion pinned to the method turns every routine
decision into a change to what the plan *says*, and therefore into a review the
user has to sit through. What counts as "how" depends on the work — a rename or
a refactor in code, a different source, sample window or tool in research, a
different layout or file in design. None of them may cost the user a decision.

| too fine-grained | right altitude |
|---|---|
| the block's quote goes through `capQuote` | a note filed from a block quotes that block |
| the pending highlight does not render as `.plan-mark` | a draft comment is visibly distinct from one already filed |
| the top 10 hashtags by engagement rate from the Graph API | a reader can name the formats gaining traction and open posts that show each one |
| a 200-account sample scraped weekly | every trend is backed by more than one account, so no single viral post carries one |

A quality bar a reader can check IS a criterion. A scope rule (*"nothing
outside this directory changes"*, *"public posts only, last 90 days"*) restates
the Goal and belongs there — breaking it already moves something observable, so
it already produces a deviation.

**The format is now PARSED — and still not validated.** This paragraph used to
say there was no parser, and that is no longer true: `plan show --sections`
lists every `AC-<m>.<n>` on the plan and `plan note --ac` resolves one to a
section and a line. Nothing *checks* the label, though, so the sentence that
matters is unchanged — a plan that ignores the convention still parses fine and
simply has no criteria. What you lose by ignoring it is the address: an argument
about one criterion has to point at the whole milestone instead, which is a note
the user has to cold-read a section to place. Keep the convention. The
completion report also walks criteria per milestone and reads them out of the
plan rather than restating them — per criteria group, naming which milestones
claimed each.

**One plan per project.** Not per to-do, not per milestone, and never on a DM —
a DM has no `shared-context` room to hang one on.

## What this skill does not do

- It does not design the plan. What a good plan says is your judgement; this is
  how to move the bytes.
- It does not merge. Two changes to one section meet in front of a human — there
  is no three-way merge, no fuzzy patch and no model-resolved reconciliation
  anywhere in this feature, by design.
- It does not detect a deviation. It records one.
