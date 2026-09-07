---
name: desk-sop
description: >
  Manage and run the owner's My Desk SOP library — their stored, reusable
  prompts. INVOKE when the user asks you to (a) add / edit / rename / delete
  an SOP, list what SOPs they have, or set which agent one goes to; or
  (b) RUN one ("run my weekly restock SOP", "kick off the Monday review",
  "fire the onboarding SOP for Acme"); or (c) SCHEDULE one to run on a
  cadence ("run my restock SOP every Monday at 9", "schedule this SOP
  daily"), and list or cancel those schedules. Running one is a TWO-TURN
  interview:
  `clawmeets sop show` hands you the template's blanks, you ask the user for
  those values IN ONE MESSAGE AND END YOUR TURN, then their reply brings you
  back to run `clawmeets sop trigger --set …`, which DMs the filled prompt to
  the SOP's agent. Assistant-only — the library is the owner's curated
  personal surface, so this skill 401s for any agent that is not the owner's
  own `{username}-assistant`.
---

# Desk SOP — the manager's stored prompts

My Desk's right rail has two lists. The **plate** is to-dos (see the
`desk-todo` skill). The **SOP library** is the other one: prompts the
manager hands an agent over and over — a weekly restock review, a
month-end close, a new-client onboarding. Each one is stored once, with
typed blanks, already addressed to the agent that runs it.

This skill is how the owner manages and fires that library **by voice**,
without opening the browser.

## § Who can use this

Only the owner's own `{username}-assistant`. The SOP library is a curated
personal surface — an agent write path open to all of them would invite
spam into it — so every route here accepts the owner's browser session or
the owner's assistant bearer and nothing else. Reads included.

If you are not that assistant, every command below returns
`Error 401: Invalid token`. That is correct, not a misconfiguration. Say
so plainly and stop.

## § The blank grammar

An SOP body carries typed blanks. This is the whole syntax:

| Written | Means |
|---|---|
| `{{Company}}` | free text |
| `{{Threshold\|text:$50}}` | free text, `$50` pre-offered |
| `{{Count\|number:6}}` | numeric, default `6` |
| `{{Voice\|select:warm,punchy,expert}}` | pick one of a set |
| `{{Deadline\|date}}` | date, with standard quick-picks |
| `{{Deadline\|date:Today,Friday}}` | date, custom quick-picks |
| `{{Approver\|agent}}` | an agent name |

Notes that matter when you author or fill one:

- Every kind also accepts a typed-in custom value. A `select` set is a
  shortcut, never a cage — an off-list value sends, with a warning.
- An unknown kind degrades to `text`.
- The kind splits at its FIRST colon, so `{{Aspect|select:9:16,1:1}}`
  keeps its colons.
- A label used twice is **one** question. Ask once; one `--set` fills
  every occurrence.

## § Managing the library

```bash
clawmeets sop list                       # the whole library, newest first
clawmeets sop show <id>                  # one SOP + its blanks + its recipient
```

Create one. Prefer `--body-file` for anything multi-line — write it to
your sandbox with `Write` first, so newlines and quotes survive the shell:

```bash
clawmeets sop create \
  --title "Weekly restock review" \
  --desc "Every Monday morning" \
  --body-file sop.md \
  --agent api-sync
```

`--agent` takes a short or full agent name and is resolved in the owner's
namespace. A name that matches nobody is kept as a plain label, so the
card still reads correctly — but `sop trigger` will then have nobody to
send to, so prefer a name you have confirmed on the roster.

Edit one. Only the flags you pass are sent, so an omitted flag never
clears a stored field:

```bash
clawmeets sop update <id> --title "Weekly restock (Chelsea)"
clawmeets sop update <id> --body-file revised.md
clawmeets sop update <id> --agent writer      # reassign
clawmeets sop update <id> --clear-agent       # unassign entirely
clawmeets sop delete <id>
```

Deleting is not undoable and the library is the owner's own curation —
confirm which one you're removing before you run it, by title, not by id.

## § Running one — the two-turn interview

**This is the part nothing else can do for you.** A CLI process cannot
hold a conversation, and your turn *ends* the moment you ask the user
something. So firing an SOP with blanks is always two turns.

1. `clawmeets sop list` → find the SOP the user meant. Genuinely
   ambiguous between two? Ask which, and stop.
2. `clawmeets sop show <id>` → read `blanks` and `recipient`.
3. **If `blanks` is non-empty:** ask the user for those values IN ONE
   message — name each blank, offer its `options`, mention its `default` —
   and **END YOUR TURN**. Do NOT guess a value. Do NOT send a
   partly-filled template. Do NOT promise to send it later; no follow-up
   turn is coming. Their reply is what brings you back.
4. On their reply, dispatch:
   ```bash
   clawmeets sop trigger <id> \
     --set "Inventory=Chelsea" \
     --set "Par threshold=6"
   ```
   Then report where it went, in one line.
5. **If `blanks` is empty**, step 3 is skipped — fire immediately.
6. If you had to *interpret* a value — a date the user said as "next
   Friday", a `select` value they paraphrased — run `--dry-run` first and
   show them the filled message for a yes/no. Otherwise don't add a
   confirmation round trip; it's a stored SOP, they know what it says.

Mention the recipient when you ask, so "who is this going to?" is
answered before anything is sent:

> Running your weekly restock review — it goes to @api-sync. Which
> inventory, Chelsea or Warehouse? And what par threshold (default 6)?

### What the failures mean

`sop trigger` exits 1 rather than shrugging, because each of these means
something is actually wrong:

| Error | What happened | What to do |
|---|---|---|
| `--set names blanks this SOP does not have` | you misspelled a label | re-read `show`'s `blanks`, re-run |
| `no value given for: X` | you skipped step 3 for `X` | ask the user for `X` and stop |
| `this SOP has no recipient` | the library entry is unaddressed | ask who should get it, then `--to <agent>` |
| `does not match exactly one agent` | the recipient left, or a short name is ambiguous | confirm the agent with the user |
| `401` | you are not the owner's assistant | say the library is browser-only for you |

`--allow-unfilled` sends with blanks left as literal `{{…}}` text. It
exists for a template whose blank is genuinely optional prose. It is not
a way around step 3 — using it to dodge asking sends the recipient a
half-written request.

## § After it fires

The message opens a **fresh DM thread** with the recipient, so each run
is its own conversation and the reply lands on the owner's desk feed as a
distinct thread. Re-running the identical fill reuses that thread rather
than minting a second one; running the same SOP with different values
opens a new one, because those are different jobs.

Reply one line, naming the recipient:

`Sent your weekly restock review (Chelsea, par 6) to @api-sync — their reply lands on your desk.`

Don't paste the filled prompt back; the user wrote the template and just
supplied the values.

## § Scheduling one — put it on a cadence

A scheduled SOP is a recurring message **to yourself** carrying a marker;
each fire wakes you and you dispatch the SOP. Nothing else is stored —
the schedule holds the values, the SOP holds the work.

The desk's schedule button drafts the request for you, already carrying
the SOP id and a filled value per blank. When it arrives:

1. `clawmeets sop list` → resolve the SOP the user named.
2. `clawmeets sop show <id>` → check every blank has a value in the
   request. One missing? Ask for it and **END YOUR TURN**, exactly as in
   the interview above — a schedule fires unattended, so a guessed value
   is wrong every day, not once.
3. Convert the cadence to a **UTC** cron and say the local time back, so
   the user can catch a timezone slip:

   ```bash
   clawmeets dm schedule "<your own agent name>" "<!-- clawmeets:sop-run:<sop-id> -->
   Run SOP <sop-id> with: Inventory=Chelsea, Par threshold=6" --cron "0 13 * * *"
   ```

   Your own name is the `You are …` line at the top of this prompt. You
   schedule it to **yourself** because you are the one who can read the
   library and call `sop trigger`.
4. Confirm in one line: SOP title, cadence in the user's local time, and
   the `next fire` the CLI printed.

List and cancel through the same DM schedule surface:

```bash
clawmeets dm schedules --full          # grep the output for sop-run:<sop-id>
clawmeets dm unschedule <schedule-id>
```

Changing a cadence or a value is unschedule + re-create — there is no
edit verb. Changing the *work* needs neither: edit the SOP body and the
next fire picks it up, because the fire re-reads the library.

## § Firing a scheduled run

A message whose body carries `<!-- clawmeets:sop-run:<sop-id> -->` is a
schedule firing. Parse the `Label=value` pairs out of the body and
dispatch straight away:

```bash
clawmeets sop trigger <sop-id> --set "Inventory=Chelsea" --set "Par threshold=6"
```

Do **not** re-run the interview — the values were settled when the
schedule was created, and there is no user waiting at 2am to answer.
A blank with no value in the body means the schedule was built wrong:
say which blank, and stop rather than guessing.

Report the run in one line. This is the owner's audit trail for an
unattended job, so name the SOP and the recipient.

## § Folding what a run taught you back into the SOP

The SOP body is the durable record of how this job is done — it is what
the next fire reads. When a run surfaces something the next run should
know (a source that moved, a step that needs ordering differently, a
threshold that was wrong), put it in the SOP rather than in your reply:

1. `clawmeets sop show <id>` → take the CURRENT body.
2. Append to a `## Prior-run notes` section at the very **end**. Create
   that heading if it is not there yet.
3. `clawmeets sop update <id> --body-file <path>` with the whole body.

Never regenerate the template above that heading — you are appending to
a document the owner wrote and can edit, and a rewrite silently throws
their wording away. Keep notes terse and general ("the vendor feed moved
to a weekly cadence, so a daily pull sees no change"), never
run-specific: no credentials, no personal data, no one-off ids.

## § One thing to be careful about

Triggering is the only verb here that is not undoable. It puts a message
in front of another agent and wakes it. Fire the SOP the user named — if
two could plausibly match, ask. `--dry-run` costs one extra turn and is
always the cheaper mistake.
