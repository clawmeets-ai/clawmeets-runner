---
name: author-brief-tab
description: >
  Used in the owned DM (your user's 1:1 with you, their assistant) when
  the user asks to add, change, or remove a RECURRING briefing on their
  My Desk — "track X on my desk, refresh daily", "add top-5 AI news to
  my briefings, weekly", "make the wine briefing weekly instead",
  "remove the news briefing". Judge that it's a recurring-briefing ask
  (one-off questions are answered directly or delegated), route to the
  owned agent capable of producing the data, draft the refresh message
  (brief-tab trigger marker + self-contained SOP), show it for
  approval as a normal DM reply, and only on approval schedule it and
  fire the first run. Assistant-only. Does not publish briefings itself
  when the executor is a peer.
---

# Author brief tab

The user describes WHAT they want on their My Desk briefings and how
often; you translate it into a per-tab pipeline: one executor agent, one
scheduled message carrying the brief-tab trigger marker plus a
self-contained SOP, one slug. Each briefing is its own pipeline — the
user builds the desk up one briefing at a time, and the slug is the join
key between the briefing and its schedule.

## You only run if you are the user's assistant in an owned DM

This skill belongs to `{username}-assistant` in their own 1:1 DM. If
your name doesn't end with `-assistant`, or the room isn't an owned
`dm-*` chatroom, stop and reply with one line: "author-brief-tab is
for the user's personal assistant in their own DM."

## Step 1 — judge and classify

Fire only when BOTH signals are present:

1. **A My Desk destination** — "on my desk", "briefing", "the
   dashboard", or the user is clearly extending a briefing they set up
   this way before.
2. **Recurrence** — "daily", "every Monday", "keep it updated",
   "weekly".

One-off asks ("what Burgundies does Acker have right now?") are NOT
this skill — answer directly or delegate via a plain DM. Classify the
ask:

- **New briefing** — obtain the KEY from the user (the name they give);
  if they didn't name it, derive a candidate slug and surface it in the
  proposal (Step 4) for confirmation — never lock in a slug the user
  hasn't seen. First run `clawmeets brief list-tabs`: if a tab from a
  prior one-off publish matches the user's name, reuse its slug so the
  schedule upserts that same tile instead of creating a duplicate.
  Slugify with the SAME deterministic rule the `brief` skill uses
  (lowercase → trim → each run of `[^a-z0-9]` → single `-` → strip
  leading/trailing `-`); keep it `[a-z0-9_-]+` and ≤ ~20 chars so it
  stays greppable in schedule listings. Also pick a briefing title.
- **Update** — the user references an existing briefing (criteria change,
  cadence change, retitle). Keep the slug stable so the briefing
  overwrites in place.
- **Remove** — tear the pipeline down (Step 7).

If the source or criteria are ambiguous, ask 1–2 pointed questions
before drafting ("Acker's daily auction list or their retail site?";
"price in USD per bottle?"). Don't guess.

## Step 2 — route the executor

List the peer cards and shortlist by capability:

```bash
ls "$CLAWMEETS_AGENT_DIR/agents/"
```

Read each candidate's `card.json` (`description`, `capabilities`,
`user_teams`). The executor must be able to (a) produce the data —
browse, search, or read its own warehouse — and (b) publish the briefing.
Confirm installed skills (install state is NOT on peer cards):

```bash
clawmeets skill installed <agent-name>
```

It needs the `brief` skill plus whatever the data side requires
(e.g. a browser skill for scraping a retailer). Also check the slug
isn't already taken by another agent's briefing:

```bash
clawmeets brief list-tabs
```

Outcomes:

- **One fit** → proceed.
- **Zero or multiple** → ask the user, naming the candidates.
- **Capable agent missing a skill** → include the install in the
  proposal; on approval install it via the `/clawmeets:install-skill`
  flow.
- **No owned agent can do it** → design ONE new specialist
  (name / description / capabilities) and include it in the proposal.
  On approval, register and start it exactly as
  `/clawmeets:propose-project` Steps 6a/6b do (`clawmeets agent
  register … --as-user <username>`, then targeted
  `clawmeets start --user <username> --agent <name>`). Reuse before
  creating — a rough fit on the existing roster beats a new agent.

You may route to **yourself** when you're the natural executor (e.g.
a briefing built from data you already own) — self-install the
brief-publishing skill if missing (`clawmeets skill install self brief`)
and schedule the trigger DM to yourself.

## Step 3 — choose the pipeline shape

**Default — single executor, scheduled DM.** The refresh message body
is the trigger marker followed by a complete, self-contained SOP:

```
<!-- clawmeets:brief-tab-trigger:<slug> -->
Publish the briefing `<slug>` (title: "<Briefing title>").
Source: <where to look — site, warehouse table, search scope>.
Criteria: <exactly what qualifies — filters, price bounds, count>.
Output: <the rows/fields the briefing should show, and any sort order>.
```

It fires verbatim every time and the executor sees only its DM — so
the body must carry everything; never rely on this conversation for
context. The executor's `brief` skill owns the publish protocol
(render code, styling, upload); your SOP only says what data to show.

**Rare — genuinely multi-agent.** Only when one agent truly can't
produce the briefing (e.g. data gathering and analysis live on different
specialists): create a dedicated per-tab project with yourself as
coordinator whose request ends with "final step: publish briefing
`<slug>` via the brief protocol", and schedule a recurring DM **to
yourself** whose body is `<!-- clawmeets:rerun-<project-name> -->` so
each fire resets and reruns the project via `/clawmeets:rerun-project`.
Justify this shape in the proposal — it costs several LLM turns per
refresh instead of one.

## Step 4 — propose as a normal DM reply

Reply with: a one-paragraph plain-language summary (executor, slug +
title, cadence in the user's local time, what gets checked, roughly
one LLM turn per refresh — more for the project shape, plus any
new-agent registration or skill install) + the exact refresh message
body in a fenced block + "Say **go** to set it up, or tell me what to
change." **No trigger marker outside the fenced block.** State the
briefing key explicitly in the summary ("key: `wine` — reused on every
refresh") so the user can correct it before it is locked in.

## Step 5 — on the next turn

Re-read the recent chat. Clear approval → execute. Redirect → revise
the draft in place and re-propose. Off-topic → the proposal is
implicitly dropped; answer the new message normally. Ambiguous → one
clarifying question.

## Step 6 — execute on approval

1. If a new agent or skill install was approved, do that first (the
   cross-referenced flows in Step 2). If a `register` or `start` step
   fails, surface the error verbatim and stop.
2. Schedule the refresh via the `/clawmeets:schedule-message`
   conventions (cron is UTC — convert from the user's local time and
   say so):

   ```bash
   clawmeets dm schedule <agent-name> "<marker + SOP body>" --cron "<utc-cron>"
   ```

   For the project shape, the scheduled DM targets yourself with the
   rerun marker instead.
3. First run now (default yes, skip only if the user declined): send
   the same body once via `/clawmeets:direct-message` so the briefing
   appears immediately:

   ```bash
   clawmeets dm send <agent-name> "<marker + SOP body>"
   ```
4. Confirm in one line: briefing title + slug, executor, cadence in
   local time, and the `next fire` timestamp the CLI printed.

## Step 7 — update / remove an existing pipeline

Find the schedule that owns the slug:

```bash
clawmeets dm schedules --full
```

and match `brief-tab-trigger:<slug>` (or `rerun-<project-name>` for
the project shape) in the content.

- **Update**: `clawmeets dm unschedule <schedule-id>`, then re-create
  with the revised body and/or cron (Step 6). Same slug — the briefing
  overwrites in place on the next fire.
- **Remove**: unschedule, then send a one-time DM asking the
  publishing agent to run `clawmeets brief delete-tab <slug>` — only
  the publishing agent (or the user, from My Desk) can delete
  a briefing. Confirm both halves to the user in one line.

If no schedule matches the slug, say so and show the briefings
(`clawmeets brief list-tabs`) so the user can point at the right one.

## Hard rules

- **Never execute before approval on the current turn.** A proposal
  is not execution.
- **Never** Edit/Write peer cards or skill configs directly.
- **One pipeline per briefing, one briefing per pipeline** — the slug is
  the join key; never reuse a slug across executors.
- **Do not** publish or refresh the briefing yourself when the executor
  is a peer — the executor runs it on its own runner when triggered.
- **Do not** install skills on peers or register agents without
  explicit user approval.

## Worked examples

*"I wanna check Acker for good-price Burgundy red between $100 and
$300, refreshed daily, on my desk"* → executor: the wine agent
with a browser skill; slug `acker-burgundy`; body =
`<!-- clawmeets:brief-tab-trigger:acker-burgundy -->` + SOP (source:
acker site; criteria: Burgundy, red, $100–300/bottle; output: wine,
vintage, price, link, sorted by price); cron `@daily`; first run now.

*"Add top 5 AI news to my briefings, weekly"* → executor: the
news/research agent; slug `ai-news-top5`; body = marker + SOP (top 5 AI
developments of the past week, one line each: headline, why it
matters, source link); cron `0 16 * * 1` (Mondays 9am PDT).
