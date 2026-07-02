---
name: author-etl-rule
description: >
  Used in the owned DM (your user's 1:1 with you, their assistant)
  when the user describes a RECURRING data pipeline over their synced
  personal data — "extract X from my (gmail / photos / calendar /
  drive) into a table with columns …, refresh daily", "keep a running
  table of …", "track every <thing> that lands in my <source>". Judge
  that it's a recurring derivation (not a one-off question — answer
  those directly or delegate), identify the owned data agent whose
  warehouse has the source and who has the etl skill, draft the
  etl.json rule (sources, columns, merge_policy, key, on_source_delete,
  prompt, howto), show it for approval as a normal DM reply, and only
  on approval apply it and optionally schedule the trigger.
  Assistant-only. Does not create projects, does not run ETL itself.
---

# Author ETL rule

The user describes WHAT they want tracked; you translate it into a
rule the deterministic ETL harness can run. The user supplies the
intent (source, extraction, table shape, cadence); you fill in the
mechanical fields (watermarking, provenance columns, merge policy,
stable key, source paths) from the conventions below and from what you
can see in the target agent's warehouse.

## You only run if you are the user's assistant in an owned DM

This skill belongs to `{username}-assistant` in their own 1:1 DM. If
your name doesn't end with `-assistant`, or the room isn't an owned
`dm-*` chatroom, stop and reply with one line: "author-etl-rule is for
the user's personal assistant in their own DM."

## Step 1 — judge whether this is a rule

Fire only when ALL THREE signals are present:

1. **A synced source** — the data lives (or should live) in a data
   agent's warehouse `raw/` layer (gmail, photos, calendar, drive
   docs, database dumps, …).
2. **A tabular output** — columns are stated or clearly inferable
   ("airline, flight number, dates", "producer, wine, vintage").
3. **Recurrence** — "daily", "whenever new mail arrives", "keep
   updated", "running table of".

One-off questions ("what wines were offered last week?") are answered
from the existing derived tables or delegated to the data agent's DM —
do NOT author a rule for them. If the recurrence is there but source
or columns are ambiguous, ask 1–2 pointed questions before drafting
("from your gmail receipts or your credit-card statements?"; "just
name + price, or renewal date too?"). Don't guess.

## Step 2 — identify the owning data agent

List the peer cards and shortlist data agents:

```bash
ls "$CLAWMEETS_AGENT_DIR/agents/"
```

Read each candidate's `card.json` — you're looking for
`local_settings.dwh_dir` (set ⇒ the agent owns a warehouse). Confirm
the etl skill is installed (install state is NOT on peer cards):

```bash
clawmeets skill installed <agent-name>
```

Verify the source actually exists: `ls <dwh_dir>/raw/` (peer
runners share this host). Outcomes:

- **One match** → proceed.
- **Zero or multiple** → ask the user, naming the candidates.
- **Agent has the warehouse but not the etl skill** → tell the user;
  offer to install it (via `/clawmeets:install-skill`) only with their
  explicit approval, otherwise point at Agent Settings → Skills.
- **Source not synced at all** → that's a sync-setup conversation
  first; say so rather than authoring a rule over a missing source.

## Step 3 — read the current config (fresh, via HTTP)

Peer-card `skill_configs` can be stale — always read live:

```bash
curl -s -H "Authorization: Bearer $CLAWMEETS_ASSISTANT_TOKEN" \
     "$CLAWMEETS_SERVER_URL/agents/<agent_id>/skills/etl/config" | jq .
```

The response carries `config` (current rules), `starter_config`, and
`starter_config_text` (the commented JSONC — your style reference for
drafting). An existing rule with the same `name` means this is an
UPDATE — say so in the proposal.

## Step 4 — draft the rule

Conventions (mirror the starter config and the agent's existing rules):

- `name`: kebab `[a-z0-9_-]+`. It becomes the trigger marker segment
  (`<!-- clawmeets:<name>-etl-trigger -->`) and the output filename.
- `sources[].path`: a warehouse source under `raw/` — `gmail/inbox`,
  `mailbox/inbox`, `osxphotos`, a sheet-tab source like
  `google-drive/ledger`; or a `derived/<rule>` table to chain. The CLI
  tracks a per-source checkpoint and hands the rule new/changed/deleted
  candidates — there is NO `watermark_field` or `format` to set.
- `columns`: include the provenance columns the existing rules carry
  (`source_type`, `source_id`, `source_updated_at`, `updated_at`) plus
  the user's requested fields. If a row will be rendered with a link,
  emit one pre-computed `source_url` column.
- `merge_policy` + `key`: upsert with a stable natural key for
  incremental streams; replace for full-rollup snapshots.
- `on_source_delete` (optional): a column matched against tombstoned
  source ids to delete the derived row when its source row is removed
  upstream; default `ignore`.
- `prompt`: the per-candidate procedure — detailed and mechanical.
  Copy ingestion boilerplate from sibling rules verbatim (how to Read
  `body_html_path` / `attachments[*].path` joined to `dwh_dir`; the
  scaled_path → export-jpeg fallback chain for photos). Spell out each
  column's derivation and the skip conditions.
- `howto`: grain, key semantics, what downstream consumers should know.

Worked example — user says *"whenever a flight confirmation lands in
my gmail, keep a running table of my upcoming trips — airline, flight
number, airports, times, confirmation code; refresh every morning"*:
source `gmail/*` + watermark `ts`; upsert keyed on
`<confirmation_code>-<flight_number>-<depart-date>`; provenance columns
plus the requested fields and a Gmail `source_url`; "refresh every
morning" is NOT part of the rule — it becomes the scheduled trigger in
Step 7.

## Step 5 — propose as a normal DM reply

Reply with: a one-paragraph plain-language summary (which agent, what
gets scanned, output table path, cadence, whether this is new or
replaces an existing rule, `max_per_run` cost note) + the rule as a
fenced JSON block + "Say **go** to apply, or tell me what to change."
**No trigger marker** in the reply.

## Step 6 — on the next turn

Re-read the recent chat. Clear approval → apply. Redirect → revise the
draft in place and re-propose. Off-topic → the proposal is implicitly
dropped; answer the new message normally. Ambiguous → one clarifying
question.

**Apply** = the assistant-on-another-agent flow of
`/clawmeets:update-skill-config`: build the FULL config (current
`rules[]` with your rule appended or replaced — the PUT replaces the
whole object), write it to a scratch file, then:

```bash
clawmeets skill set-config <agent-name> etl "$tmpfile"
```

Never Edit/Write the agent's `etl.json` or peer card files directly —
a direct edit drifts from the server mirror and gets clobbered.
Surface CLI errors verbatim; don't retry blindly.

## Step 7 — offer the schedule

The rule runs when the data agent receives a DM containing
`<!-- clawmeets:<rule>-etl-trigger -->`. Offer (don't auto-create):

- **Recurring**: a scheduled DM via `/clawmeets:schedule-message`,
  e.g. daily after the relevant sync has run. Confirm the time in the
  user's local timezone.
- **First run now**: send the trigger DM once so the user sees output
  immediately.

If the user's pipelines run through a recurring orchestration project
(e.g. a daily briefing project whose coordinator fires the derivation
triggers), mention that adding the rule there is the alternative —
but changing that project is out of scope for this skill.

## Hard rules

- **Never apply before approval on the current turn.** A proposal is
  not execution.
- **Never** Edit/Write skill configs or peer cards directly — always
  `clawmeets skill set-config`.
- **Do not** install skills on peers without explicit user approval.
- **Do not** create projects or modify existing orchestration projects.
- **Do not** run the ETL rule yourself — the data agent runs it on its
  own runner when triggered.
- **Do not** invent secrets or paths — what you can't see, you ask.
