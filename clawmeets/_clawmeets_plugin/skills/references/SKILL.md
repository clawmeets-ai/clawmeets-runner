---
name: references
description: >
  One-shot index of user-pre-seeded reference files in your
  knowledge_dir. Triggered by a references-trigger DM
  (`<!-- clawmeets:references-trigger -->`) posted by
  `clawmeets bootstrap references`. Reads each file the orchestrator
  lists (absolute paths) and writes a one-line "when to invoke" entry
  per file into `$CLAWMEETS_AGENT_DIR/memory/REFERENCES.md`. Run only
  when the marker is present in your DM.
---

# References

You receive a one-shot trigger DM tagged
`<!-- clawmeets:references-trigger -->` after the user runs
`clawmeets bootstrap references`. The DM lists the **absolute paths** of
user-pre-seeded reference files under your knowledge_dir. Your job is
to read each file and produce a thin index — `REFERENCES.md` — under
your agent memory directory (`$CLAWMEETS_AGENT_DIR/memory/`) so that on
future tasks you know what's there and when to consult it.

## What you write

Only `$CLAWMEETS_AGENT_DIR/memory/REFERENCES.md`. Do **not** touch:

- `$CLAWMEETS_AGENT_DIR/memory/learnings/` (agent-distilled lessons;
  owned by `/clawmeets:reflect`)
- `$CLAWMEETS_AGENT_DIR/memory/USER.md` (assistant-only profile;
  owned by `/clawmeets:interview` on first run and by
  `/clawmeets:reflect` thereafter)
- `$CLAWMEETS_AGENT_DIR/memory/KNOWLEDGE_PACKS.md` (rebuilt by the
  runner's KnowledgePackManager on every install/uninstall)
- `$CLAWMEETS_AGENT_DIR/personal-skill-hub/` (procedures; owned by
  `/clawmeets:reflect`)
- `$CLAWMEETS_AGENT_DIR/memory/learnings/log.md` (references-build is
  not a reflection cycle)
- The user's knowledge_dir itself — references are read-only from your
  perspective; you only emit the index.

The user-curated reference layer and the agent-distilled learning layer
stay in separate trees, by design. Don't promote reference content into
`memory/learnings/` from this skill — if something in a reference file
genuinely matters cross-task, it'll surface in chat and the next
reflect cycle will pick it up on its merits.

## Source rule (read the listed files yourself)

The references skill *requires* reading the files listed in the trigger
body. Use your native `Read` tool with the absolute path as given —
don't transform or relativize it. The trigger DM does **not** inline
file contents — only paths — because reference files can be large and
chat-history bloat is real.

For files larger than ~5 KB, **skim** instead of reading the full body:
read the first ~50 lines and scan H1/H2 headings. The goal is one good
"when to invoke" line per file, not a synthesis of the file's content.

## Write mechanics

Use your native `Edit` / `Write` tool on
`$CLAWMEETS_AGENT_DIR/memory/REFERENCES.md`. Resolve
`$CLAWMEETS_AGENT_DIR` to a literal string first
(`Bash: echo $CLAWMEETS_AGENT_DIR`) before passing it into `Write`. Do
**not** use the ClawMeets `update_file` action — that broadcasts to
the chatroom and would leak the index as a visible file event.

## Steps

1. **Parse** the file list from the
   `== REFERENCE FILES (absolute paths) ==` block in the trigger DM.
   Each line is one absolute path.
2. **Read** each listed file with the Read tool (pass the absolute
   path verbatim). Skim files larger than ~5 KB (first ~50 lines +
   H1/H2 scan).
3. **Write** `$CLAWMEETS_AGENT_DIR/memory/REFERENCES.md`:
   ```markdown
   # References

   User-pre-seeded reference files. Read a file when the description
   below matches the task at hand. Paths are absolute so they resolve
   regardless of your current working directory.

   - [<basename>](<abs-path>) — <when to invoke / what's inside>
   - ...
   ```
   - One line per file. Lead with the basename (or `<parent>/<basename>`
     for files under nested subdirs of the knowledge_dir) as the link
     label, then the **absolute path** as the link target; follow with
     " — " and a short, concrete "when to invoke" clause (≤ 15 words
     ideal). Avoid generic phrasings like "documentation for the
     project"; aim for a signal the future-you can scan: "when the
     task involves <X>", "the canonical glossary for <Y>".
   - Optional H2 grouping: if the file count is ≥ 10 and natural
     clusters emerge (e.g. `domain/*` vs `playbooks/*`), group entries
     under H2 headings. Otherwise a flat list is fine.
   - Skip the file entirely if it's binary or unreadable; mention it
     in the DM reply summary instead.
4. **Reply in the DM** with a 3-bullet summary:
   - How many files you indexed.
   - Whether you used H2 grouping (and what the clusters were).
   - Any files that were too opaque to summarize confidently — surface
     them so the user can either rewrite the file or override the
     entry.

   Plain prose, no markdown headers — the user reads this on their
   phone.

## Idempotency

If `$CLAWMEETS_AGENT_DIR/memory/REFERENCES.md` already exists,
**skip** the write and reply with a single line: "Already indexed —
re-run with `clawmeets bootstrap references --force` to redo."
`--force` on the orchestrator side just re-triggers; the skill is what
gates the actual overwrite. To redo from scratch, the user should
`rm` the file before re-running, or pass `--force`.

## Example

A worker named `chengtao-marketer` whose knowledge_dir at
`/Users/chengtao/Knowledge/marketer/` contains:

```
test-strategy.md             # 200 words on go-to-market
playbooks/email-cadence.md   # 500 words on outreach cadence
playbooks/discovery-script.md # 1200 words on discovery calls
domain/icp.md                # 400 words on ideal customer profile
```

After the references skill runs,
`$CLAWMEETS_AGENT_DIR/memory/REFERENCES.md`:

```markdown
# References

User-pre-seeded reference files. Read a file when the description
below matches the task at hand. Paths are absolute so they resolve
regardless of your current working directory.

- [test-strategy.md](/Users/chengtao/Knowledge/marketer/test-strategy.md) — go-to-market strategy; consult before any GTM proposal
- [playbooks/email-cadence.md](/Users/chengtao/Knowledge/marketer/playbooks/email-cadence.md) — outreach cadence rules; consult before drafting sequences
- [playbooks/discovery-script.md](/Users/chengtao/Knowledge/marketer/playbooks/discovery-script.md) — discovery-call script; consult before booking customer interviews
- [domain/icp.md](/Users/chengtao/Knowledge/marketer/domain/icp.md) — ideal customer profile; consult when scoring leads or filtering accounts
```

DM reply:

```
Indexed 4 reference files into REFERENCES.md. Two clusters (playbooks/, domain/) but kept the list flat — too few files to warrant grouping. All four were specific enough to summarize without ambiguity.
```
