---
name: fs
description: >
  Browse and manage an agent's home folder (its AGENT_DIR: memory, sandbox,
  skill configs, knowledge packs, project files) with `clawmeets fs --agent
  <name> ls|cat|write|mkdir|rm|download <path>`. Every op is relayed by the
  server to that agent's own runner, wherever it runs, under the same access
  rules as the owner's file navigator in the web app. Invoke when the user asks
  what is in an agent's folder, to read / change / delete a file there, to
  install a file into it, or to fetch a file out of it. A worker or coordinator
  can reach only its OWN folder; the owner's assistant can reach any of the
  owner's agents' folders.
---

# Agent home folders: `clawmeets fs`

An agent's home folder lives on whichever machine runs that agent, often not
this one. `clawmeets fs` never touches a disk itself: the server checks who you
are and relays the op to that agent's runner, which applies the path and access
rules to its own folder and answers. So the answer is the same one the owner
sees in the web app's file navigator.

## Who can reach what

- **You, a worker or coordinator:** only your own folder. `--agent` defaults to
  you. Naming any other agent is refused (`forbidden`), even one of the same
  owner. Do not try other names to work around it.
- **The owner's assistant:** any of the owner's agents, by short or full name
  (`--agent backend` or `--agent alice-backend`). Never another account's
  agents.
- Names resolve only among the owner's own agents.

## Commands

Paths are relative to the home folder. A leading `/` is the home root, so
`ls /` lists the whole home folder. Both `/` and `\` work as separators.

```bash
clawmeets fs ls /                               # your own home root
clawmeets fs --agent backend ls /memory         # assistant: a peer's folder
clawmeets fs cat memory/MEMORY.md               # print a text file
clawmeets fs write sandbox/notes.md --text "hello"
clawmeets fs write sandbox/report.pdf --from ./report.pdf   # any bytes
echo "draft" | clawmeets fs write sandbox/draft.txt --stdin # stdin, only when asked
clawmeets fs mkdir sandbox/out                  # the parent must exist
clawmeets fs rm sandbox/old.txt                 # file, link, or a whole folder
clawmeets fs download sandbox/report.pdf -o ./report.pdf    # exact bytes
clawmeets fs --json ls /                        # the server's JSON, for parsing
```

- `cat` prints UTF-8 text only. For a binary file it refuses and tells you to
  use `download`.
- `write` needs exactly one of `--text`, `--from FILE` or `--stdin` (`--from -`
  is the same as `--stdin`); with none it refuses rather than empty the file.
- `write` creates or replaces ONE file. To install a multi-file skill, run
  `mkdir` for each folder, then one `write --from` for each file.
- `rm` does not ask for confirmation, and a folder goes with everything in it.
  Confirm with the user first unless they named exactly what to delete.
- `download` will not overwrite a local file unless you pass `--force`.
  `-o -` writes the bytes to stdout.
- Files are capped at 8 MiB. Listings stop at 2000 entries; a cut listing ends
  with a `(truncated ...)` line.

## What you will see marked

`ls` marks each entry that you cannot change:

| Mark | Meaning | What you may do |
|---|---|---|
| `[read-only]` | System-managed: AGENTS.md, agent.pid, logs, and the `metadata`, `projects`, `system-skill-hub`, `mcp-hub`, `skill-hub`, `knowledge_packs` folders. Also `personal-skill-hub`, which only the owner edits. | Read it if it is readable. Never write, create or delete there. |
| `[..., secret]` | Credentials and sign-in tokens: `credential.json`, `env.json`, `card.json`, `agents/`, `mcp-hub/configs`, `mcp-hub/servers`, `skill-hub/configs`, `skill-hub/state`. | Nobody can change these here. The assistant cannot read another agent's secrets. Never copy one into chat, a file, or a message. |
| `[no access]` | You may see that it exists, nothing more (e.g. logs, for agents). | Nothing. |
| `[link: ...]` | A symlink. It is never followed, and its target is not shown. | Only `rm`. |

To change skill or MCP settings, use the `update-skill-config` /
`update-mcp-config` skills. Do not edit those files through `fs`; it refuses
anyway.

## When it fails

The command prints `Error: <message> (<code>)` to stderr. The exit code tells
you what to do next:

| Exit | Meaning | Do this |
|---|---|---|
| 0 | Done. | — |
| 1 | Refused or failed (`forbidden`, `read_only`, `not_found`, `exists`, `not_empty`, `too_large`, ...). | Read the message; do not retry the same call. |
| 3 | **Result unknown.** A write, mkdir or rm got no answer: it timed out, the connection dropped, a 5xx came back without the server's own error body, or the agent disconnected mid-op. It may or may not have happened. The command then prints the parent folder's current contents. | Check that listing. If the change is there, it happened. Never report it as failed, and never blindly repeat a delete. |
| 4 | The agent is offline (its runner is not connected), or its runner is too old for home folders. | Tell the user. The agent has to be started, or upgraded and restarted. An offline agent is not an empty folder. |

## Rules

- Act only on the user's request, and only in your owner's own conversations.
  Never use this for a front-desk / external requester.
- Every op on another agent's folder, including the owner's and the
  assistant's, is recorded in that agent's access log with the path and the
  byte count (never the content).
- Do not run this skill against your own folder to get around a refusal you got
  some other way. The rules are the same on every surface.
