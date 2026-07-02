---
name: consult-proprietary-knowledge
description: >
  Search the user's proprietary reference files (their knowledge_dir) for
  specific content. Invoke when the always-on `memory/REFERENCES.md` preview
  index doesn't already pinpoint the right file — e.g. you need the file(s)
  that mention a specific term, name, number, client, product, or rule, or the
  preview snippets aren't enough to be sure. Complements the index: the index
  is the file map; this skill greps the live files.

  Use when the task needs user-specific domain context beyond chat history and
  the fallback `learnings/` layer — drafting in the user's voice, applying
  domain rules (pricing, ICPs, regulations), or looking up business specifics.

  ALSO invoke to refresh the index when the user asks you to "refresh / rebuild
  your knowledge index" or tells you they added/changed/removed files in their
  knowledge folder (see "Refresh the map" below).

  Do NOT invoke for: generic field knowledge already in `learnings/`, basic
  ops, or code edits that don't depend on user content.
---

# Consult proprietary knowledge

The user keeps domain-specific reference material under the knowledge_dir(s)
listed in your prompt's `User-curated reference material` line. The runner
keeps a deterministic, always-fresh map of those files at
`$CLAWMEETS_AGENT_DIR/memory/REFERENCES.md` — one entry per file with its path
and a short content preview. Your job is to **find and read** the file(s) the
current task needs, using that map plus live search over the files themselves.

Resolve `$CLAWMEETS_AGENT_DIR` to a literal string via
`Bash: echo $CLAWMEETS_AGENT_DIR` once and substitute it into paths below —
the Read tool does not expand env vars.

## Step 1 — check the map

Read `$CLAWMEETS_AGENT_DIR/memory/REFERENCES.md`. It lists every reference
file (absolute path) with a preview of its opening words. If a file's path or
preview clearly matches the task → skip to step 3 and Read it.

(If REFERENCES.md is absent, the user has no knowledge_dir configured — say so
and proceed with general knowledge.)

## Step 2 — search the live files

When the map's previews don't pinpoint the file (the term you need may appear
deeper in a file than its preview, or across several files), grep the
knowledge_dir(s) directly. Substitute each `<kdir>` from your prompt's
`User-curated reference material` line:

```bash
# files mentioning a term (case-insensitive, recursive, names only)
grep -ril "<term>" <kdir>

# then pull line-level context from a hit
grep -ni "<term>" "<kdir>/path/to/file.md"

# locate by filename pattern
find <kdir> -type f -iname "*<pattern>*"
```

Use a few targeted terms (client name, product, rule keyword) rather than one
broad query. grep reads the live files, so it also covers anything edited or
added since the map was last built.

## Step 3 — read + answer

Read the matching file(s) with the native `Read` tool (paths are absolute) and
use the content in your answer. If nothing is relevant after searching, say so
plainly and proceed with general knowledge.

## Refresh the map (on request)

The runner rebuilds `REFERENCES.md` at startup and whenever the knowledge_dir
setting changes — but NOT when the user edits files inside the folder directly.
When the user asks you to refresh/rebuild the index, or tells you they
added/changed/removed reference files, rebuild the map by shelling the runner's
deterministic builder (one `--knowledge-dir` per path on your prompt's
`User-curated reference material` line, which are already absolute):

```bash
clawmeets knowledge-dir reindex --knowledge-dir <kdir> [--knowledge-dir <kdir2>]
```

Then confirm to the user (e.g. "Reindexed — N files now in the map."). You do
not need this for content lookups — grep (step 2) already reads live files; it
only refreshes the file map / previews in `REFERENCES.md`.

## Hard rules

- **Do NOT hand-write `memory/REFERENCES.md`.** The runner owns it; the only
  sanctioned refresh is `clawmeets knowledge-dir reindex` (above), which runs
  the same deterministic builder. Anything you write by hand is overwritten.
- **Do NOT use the `update_file` action.** Use native Read/Bash only — this is
  private lookup, not a chat-visible file event.
- **Do NOT write into knowledge_dir itself.** It's user-owned; read-only.
- **Do NOT promote contents into `learnings/`.** Durable distillation is
  `/clawmeets:reflect`'s job, on its own cadence.
