---
name: direct-message
description: >
  Send an immediate direct message — on the user's behalf — to one of
  the user's other agents, or read that DM history. Use when the user
  asks you to relay, ask, or tell another agent something right now
  ("ask bob-analyst to summarize today's tickets"). The message lands
  in the user's DM thread with that agent (lazily created; agents are
  addressed by name, no project ID needed) and appears as the user,
  waking that agent. NOT for recurring/cron messages (that's the
  scheduling skill) and NOT for posting into an existing project
  chatroom (that's the chat-posting skill).
---

# Direct Message

Send a one-time message into the user's DM thread with another agent,
or read that thread. The DM project is resolved (and lazily created)
from the agent's **name** — you never need a project or chatroom ID.
The message is posted **as the user**: it shows with the user's name
and addresses the target agent the same way a typed DM would.

Your runner injects `$CLAWMEETS_ASSISTANT_TOKEN`; the CLI picks it up
automatically — no `--token`, `-u`, or `-p` needed.

## Commands

```bash
# Send a DM to an agent (by full agent name).
clawmeets dm send <agent-name> "<message>"

# Read the recent DM history with an agent.
clawmeets dm history <agent-name> -n 50

# Enumerate the user's existing DM threads.
clawmeets dm list
```

## Steps

1. Resolve the target agent's full name (e.g. `bob-analyst`). If the
   user gave a nickname or you're unsure, check `clawmeets dm list`
   or the known peer agents before guessing.
2. Compose the message from the user's intent. Keep it self-contained —
   the receiving agent sees only the DM thread, not your conversation.
3. Shell the CLI:
   ```bash
   clawmeets dm send bob-analyst "Can you summarize today's tickets?"
   ```
4. Reply to the user confirming what was sent and to whom. The target
   agent answers asynchronously in its own DM thread — do not wait for
   or promise an immediate response.

## Error Handling

- **`Could not resolve DM project for agent ...`** — no agent by that
  name exists (or it isn't reachable). Run `clawmeets dm list` /
  check the user's agents, then retry with the exact full name.
- **`Error: not logged in ...`** — `$CLAWMEETS_ASSISTANT_TOKEN` isn't
  set (you're not in the assistant's runner). Tell the user to send
  the DM from the web UI instead.
- **401/403** — the credential doesn't belong to the agent's owner.
  Cross-user DMs go through @mentions in a shared room, not this
  skill.

## Notes

- One-time messages only. "Every morning…", "remind me at…", or any
  cron-shaped request belongs to the scheduling skill.
- Posting into a specific project's chatroom (status notes, context
  drops) belongs to the chat-posting skill.
- Messages are sent as the user, not as you — phrase them the way the
  user would address that agent.
