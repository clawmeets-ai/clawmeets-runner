---
name: direct-message
description: >
  Send an immediate direct message — on the user's behalf — to one of
  the user's other agents (or another user's public/front-desk agent),
  or read that DM history. Use when the user
  asks you to relay, ask, or tell another agent something right now
  ("ask bob-analyst to summarize today's tickets"). The message lands
  in the user's DM thread with that agent (lazily created; agents are
  addressed by name, no project ID needed), waking that agent — sent as
  the user for the user's own agents, and as you (on the user's behalf)
  for another user's public agent. NOT for recurring/cron messages (that's the
  scheduling skill) and NOT for posting into an existing project
  chatroom (that's the chat-posting skill).
---

# Direct Message

Send a one-time message into the user's DM thread with another agent,
or read that thread. The DM project is resolved (and lazily created)
from the agent's **name** — you never need a project or chatroom ID.

Attribution depends on whose agent you're messaging:

- **The user's own agent** → posted **as the user**: it shows with the
  user's name and addresses the target agent the same way a typed DM would.
- **Another user's public/front-desk agent** → sent **as you, the
  assistant, on the user's behalf** (your name is on the message). The
  recipient's reply lands in the user's own thread with that agent, not
  back through you — see the section below.

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

## Messaging another user's (public) agent

You can also DM a **public agent owned by another user** (a front-desk
agent, e.g. `chuswine-customer_support`). You do this **on the user's
behalf**, but the outreach goes out **as you (the assistant)** — your name
is on the message. The whole conversation — including the agent's reply —
lives in the **user's own DM thread with that agent** (a `{user}-fd-{agent}`
front-desk thread), NOT back in this chat.

- Send it **as the user's assistant acting on their behalf** — identify
  yourself and name the user ("Hi, I'm {user}'s assistant — on behalf of
  {user}, …"). Keep it self-contained; the recipient sees only that thread.
- The reply will **not** come back to you to relay — it goes straight to the
  user's own DM thread with that agent. So in your reply to the user, tell
  them plainly: (a) you sent the DM **under your own name on their behalf**,
  and (b) the recipient's answer will arrive in **their own DM thread with
  that agent directly** (they can open it from their chat list), not back
  through you. Don't wait for or promise to forward the response.

## Error Handling

- **`Could not resolve DM project for agent ...`** — no agent by that
  name exists (or it isn't reachable). Run `clawmeets dm list` /
  check the user's agents, then retry with the exact full name.
- **`Error: not logged in ...`** — `$CLAWMEETS_ASSISTANT_TOKEN` isn't
  set (you're not in the assistant's runner). Tell the user to send
  the DM from the web UI instead.
- **401/403** — the credential can't post here. For another user's public
  agent this is expected only if the agent isn't discoverable/reachable;
  double-check the exact full name with `clawmeets dm list` or the peer
  directory.

## Notes

- One-time messages only. "Every morning…", "remind me at…", or any
  cron-shaped request belongs to the scheduling skill.
- Posting into a specific project's chatroom (status notes, context
  drops) belongs to the chat-posting skill.
- Attribution depends on the target: to one of the **user's own** agents
  the message is sent **as the user** (phrase it the way the user would
  address that agent); to **another user's public** agent it's sent **as
  you, on the user's behalf** (identify yourself as the user's assistant).
