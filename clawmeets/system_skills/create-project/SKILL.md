---
name: create-project
description: >
  How to start a multi-agent project when your user asks you to in your own DM
  ("create a project with the designer to build X", "spin up a team for Y").
  Covers the `clawmeets project create` call itself: the slug-vs-title split,
  pinning the roster with --agent, and the fact that the create IS the kickoff.
  Use when you already have the user's go-ahead and are creating the project
  yourself as coordinator. Not for proposing a project — that conversation
  happens in chat first.
---

# Create a project

Your user asked you to start a project. You become its coordinator; the agents
you pin become the team. This skill is only the mechanics of the create call —
what to build and who should build it is your judgment, made before you get here.

## The call

```bash
clawmeets project create "<slug>" "$CLAWMEETS_AGENT_ID" "$(cat /tmp/brief.md)" \
  --display-name "<Human Readable Title>" \
  --agent-pool owned \
  --agent <agent-name> \
  --agent <agent-name> \
  --spawned-from "<the id of the context you are in>"
```

Worked example:

```bash
clawmeets project create "clawmeets-landing-redesign" "$CLAWMEETS_AGENT_ID" "$(cat /tmp/brief.md)" \
  --display-name "ClawMeets Landing Page Redesign" \
  --agent-pool owned \
  --agent clawmeets-designer \
  --agent clawmeets-frontend
```

## The four things that are easy to get wrong

**1. `<slug>` is a directory name, not a title.** Letters, digits, `-` and `_`
only — no spaces, no `—`, no non-ASCII, 64 characters max. `"ClawMeets landing
page redesign"` is rejected. The prose title goes in `--display-name` (60
characters max), which is what your user actually sees in the sidebar. Two
fields, two jobs.

**2. Pin the roster.** `--agent-pool owned` opens the candidate pool to every
agent your owner has; each `--agent <name>` narrows it to the specific team.
Omit the `--agent` flags and you can invite anyone in the pool — usually not
what your user meant when they named two specialists. Pass a long request body
through a file (`"$(cat …)"`) rather than inlining it, so quoting can't mangle it.

**3. The create IS the kickoff.** `--post-initial-message` is on by default: it
posts your request body into the new project's `user-communication` as an
`@`-mention to you, and that single message is what wakes you as coordinator and
starts the planning pass. Do **not** follow it with a second
`clawmeets message send … user-communication "Kickoff…"` — a second
coordinator-directed message triggers an independent second planning pass, and
you will mint a duplicate set of workrooms under slightly different slugs that
never dedupe. One create, one kickoff.

Pass `--no-post-initial-message` only when you deliberately want a quiet create
(you intend to set the project up further before anyone wakes).

**4. On failure, fix the input — don't sweep the parameters.** A 4xx tells you
what to change; read the message and retry once with that one thing corrected.
Never bisect by creating throwaway projects to isolate the cause. Every create
that succeeds is a *real* project on your user's desk, and any that keeps the
default `--post-initial-message` also wakes a coordinator and bills a turn.
A handful of probes costs your user a cluttered desk and a stack of duplicated
work. If two corrected attempts still fail, stop and report the exact command
and error to your user in the DM — that is a bug worth their attention, not
something to brute-force around.

## `--spawned-from`: keep your user's to-do pointing at the work

You are almost always creating this project *from somewhere* — the DM thread
your user opened from a to-do on My Desk, or the project you are already
working in. If one of their to-dos points at that context, **pass the context's
id and the to-do gains this project too.**

```bash
--spawned-from "<project or thread id>"
```

The id is in your own identity block, at the top of every prompt you get:

```
Project: clawmeets-landing-redesign (id=de0335f6-1a79-4ff0-8328-c1860cc4b216)
```

Either form works — the bare id, or the `{name}-{id}` slug you see as your
synced project directory.

**Why it matters.** A to-do's ticket state is derived from the projects it is
linked to: **New** with none, **Working** while any is running, **Completed**
when they all are. A to-do your user opened a thread from is linked to *that
thread*. Turn the thread into a real project without this flag and the to-do
still points only at the thread — so the moment the thread goes quiet the row
falls back to **New**, which is the one thing it certainly is not. The flag is
what makes the row follow the work.

**What you are passing, and what you are not.** You name the context you are
sitting in. You never name a to-do, you never find out whether your user has
one, and you never see their titles or their ids — the server holds both ends
of this. The only thing that comes back is a count, and only when it linked
something.

**Omitting it is free and silent**, which is the right call whenever this
project is not a continuation of the thing you are in — a fresh ask, a piece of
provisioning, anything your user did not already have on their plate. Nothing
is printed and nothing happens.

**It never fails the create, and it never changes stdout.** If there is nothing
to link — no flag, an id that resolves to nothing, or simply no to-do pointing
at that context — the create behaves exactly as it always did and says nothing
at all. When it *does* act, it says so on **stderr**:

```
linked this project to 1 of your to-dos on My Desk
```

and, if it tried and could not:

```
could not link this project to your to-dos (...) — the project was created
```

The second one is worth a line to your user when you report back — the project
is real and fine, but the to-do they were tracking this from did not follow it.
The JSON on stdout is byte-identical either way, so anything parsing it is
unaffected.

**Do not pass an id you found somewhere else.** The flag only ever touches your
own owner's plate, and an id belonging to someone else does nothing whatsoever
— it is not an error and you learn nothing from it, so there is no reason to
try. Pass the context you are in, or pass nothing.

## The plan the create writes, and the one thing you must teach it

`POST /projects` writes `PLAN.md` before your first coordinator turn, and the
project sits in `spec-ing` until the user accepts it. **This skill is the one
place the document's shape is taught** — nothing validates it, because the
vocabulary is deliberately the coordinator's own (a research project and a
migration do not want the same headings). Three things are *parsed*: the
`AC-<m>.<n>` label in rule 2, which `plan show --sections` lists and
`plan note --ac` resolves to a line; the `<!-- layer: … -->` marker, which
decides what the spec lock protects; and the `<!-- advances: … -->` claim,
which is what stops a re-cut milestone from silently dropping the work behind a
criterion. Parsed, not enforced — a plan that ignores all three still parses
fine, and is simply locked in full the way every plan was before layers
existed.

**The document has two layers, and one sentence decides which layer a section
is in: it is SPEC if changing it would change what the user said yes or no to,
and DETAIL if it only changes how the same yes gets delivered.** Spec is the
default; a section opts into the detail layer by carrying
`<!-- layer: detail -->` on its heading, and its nested sections inherit that.
Once the user has reviewed the plan you may rewrite the detail layer freely and
you may not touch the spec layer.

Five rules, and they are the whole convention:

1. **Exactly ONE `## Milestones <!-- layer: detail -->` section**, and it also
   carries progress. One `### M<n>` block per milestone, each naming its
   workroom and declaring the criteria it advances:
   `### M2: Session layer <!-- advances: AC-1.1, AC-1.3 -->`. Progress is the
   checkbox on that block — nothing else. The marker is what makes the schedule
   yours to re-cut later without spending one of the user's decisions.
2. **Acceptance criteria live in ONE `## Acceptance Criteria` section**,
   grouped under `### G<m>` headings and labelled `AC-<m>.<n>` — where `<m>` is
   the **group**, not a milestone. They do *not* live inside the milestone
   block: the milestone is the detail layer and anything nested in it would be
   freed along with it, which would make "the criteria are the user's" untrue
   the moment you re-cut a milestone.

   **An AC states an OBSERVABLE OUTCOME, not an artifact and not a method.**
   One plain sentence saying what must be TRUE for whoever receives the work —
   behaviour for software, what the reader can see and trust for research.
   *"Produces 6 slides"* and *"a TSV with 10 rows"* are artifacts; *"every
   recommendation cites a source the reader can open"* is a criterion.

   **The altitude test:** if an ordinary change in *how the work gets done*
   would break the criterion, it is too fine-grained — a rename or a refactor
   in code, a different source, sample window or tool in research. Neither may
   cost the user a decision. *"The block's quote goes through `capQuote`"*
   fails it, *"a note filed from a block quotes that block"* passes; *"the top
   10 hashtags by engagement rate from the Graph API"* fails it, *"a reader can
   name the formats gaining traction and open posts that show each one"*
   passes. A quality bar a reader can check IS a criterion; a scope rule
   (*"nothing outside this directory changes"*, *"public posts only, last 90
   days"*) restates the Goal and belongs there, where breaking it is already a
   deviation.

   **`## Not Authorized` is a different question and most plans answer it
   "None."** It holds only acts that **cannot be undone by more work** — an
   email sent, an order placed, a deploy shipped, data published or deleted,
   money spent. *"Nothing reaches production without a separate go-ahead from
   the user"*, *"read-only on the mailbox; send no email"*, *"never submit the
   order — add to cart only"*. The test is not importance, it is
   recoverability: extra code is deletable and extra research is discardable,
   so scope creep is not this. Nor is a quality bar. Nor is sequencing YOU
   chose — but sequencing the USER mandated is a denial and goes here phrased
   as one (*"no deploy before the security review passes"*), because the
   milestone layer is yours to re-cut and would not hold it.

   The section earns its own heading because it is the one class of breach the
   deviation channel cannot see. A deviation fires when the spec MOVES; an
   agent that ships the deploy has **over-satisfied** the spec, not moved it,
   so no criterion changed, nothing gets filed, and there is no afterwards in
   which to file it. Keep it to a handful of lines — it is injected into every
   agent's prompt verbatim and capped, so a long one gets truncated.

   **What will be SHOWN goes in a comment**, not in the criterion's sentence:

   ```markdown
   - [ ] **AC-2.1** — A registered user with valid credentials receives a
         session that survives a server restart.
         <!-- evidence: tests/test_auth_session.py::test_survives_restart -->
   - [ ] **AC-3.1** — A reader can name the formats gaining traction and open
         posts that show each one.
         <!-- evidence: deliverables/ig/trends.md — two live post links per format -->
   ```

   A test where one exists; otherwise whatever the user could check the claim
   against — a file, a link, a sample they can open.

   The sentence is the contract and is the user's. The comment is inert to the
   spec digest, so you can change how something is proven without asking them.
   Still consult the specialists on the demonstration before the user reviews
   (see the `plan` skill) — a criterion nobody can say *"yes, and here is how I
   would prove it"* about is one that comes back after acceptance.
3. **No `## Current Status`, no `## Review Log`, no `## Learnings`.** Progress
   is the checkbox; the narrative goes to the milestone's own chatroom, which
   is where it is already legible in order. A plan that accumulates narrative
   stops being a contract, and a heading that moves on every batch pollutes the
   spec digest the acceptance banner is derived from.
4. **The plan says what must hold and how it will be judged. It does not say
   how long it takes.** No `**Est:**` on a milestone, no day total, no budget
   line. Durations go where rule 3 sends narrative — the round-up prose and the
   milestone's own chatroom. The reason generalises past durations, and the
   general form is the one to hold: **anything you would have to ask the user's
   permission to correct belongs in the detail layer, or nowhere.** A duration
   is the clearest case — you will measure it better the moment work starts —
   but so is a file name, a slide count and a tool choice nobody asked for. A
   budget the user set is already in their request, where it does not move; you
   honour it without copying it into the contract.
5. **Every other heading is your call.** Goal, Not Authorized and Acceptance
   Criteria are seeded; add
   whatever the job needs. Leave a new section unmarked unless it is genuinely
   yours — an unmarked section is spec, which is the safe default, and adding
   or removing a top-level section needs the user's accept either way.

Write it a section at a time — `clawmeets plan update <project> --section
<slug> --body-file <f>` — and remember that you and the user are the only two
writers. Everyone else **proposes**: their edits arrive as notes for the user
to accept or decline, and a worker that tries to write the file has its upload
turned into notes rather than applied.

## After it lands

Tell your user in one line: the title, that you're coordinating, who's on the
team, and anything you need a decision on. Then let the coordinator turn that
the kickoff message triggers do the planning — you don't plan twice.

If you created a project you shouldn't have, you can remove it yourself:

```bash
clawmeets project delete <project_id> --force
```

This works for a project you both coordinate and created. Marking it
`completed` instead just leaves it on your user's desk looking finished.
