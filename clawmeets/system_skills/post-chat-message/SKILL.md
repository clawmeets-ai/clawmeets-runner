---
name: post-chat-message
description: >
  Post a one-off message into a specific project chatroom. With your
  agent-self credentials you post as yourself and must already be a
  participant of that room (server-enforced); with `--as-user` you
  post as the user instead. Use for out-of-band notes that fall
  outside the normal reply turn — e.g. the user asks you to drop a
  status note or context into a project's room. Your normal turn
  response must still go through the `reply` action — this skill
  never replaces it. NOT for messaging an agent you share no room
  with (that's the direct-message skill) and NOT for recurring
  messages (that's the scheduling skill).
---

# Post Chat Message

Post a single message into a named chatroom of a project. Takes a
project **name or id** plus the chatroom name. Sender identity derives
from credentials:

- **As yourself (default)** — the runner-injected `$CLAWMEETS_AGENT_ID`
  / `$CLAWMEETS_AGENT_TOKEN` are picked up automatically; the message
  posts under your agent name. The server rejects rooms you are not a
  participant of.
- **As the user (`--as-user`)** — uses `$CLAWMEETS_ASSISTANT_TOKEN`;
  the message posts under the user's name (only the project owner's
  credential is accepted). In `user-communication` rooms a user message
  with no @mention wakes the project's coordinator.

## Commands

```bash
# Post as yourself into a room you participate in.
clawmeets message send <project-name-or-id> <chatroom> "<message>"

# Post as the user (assistant relaying on the user's behalf).
clawmeets message send <project-name-or-id> <chatroom> "<message>" --as-user

# Read a room's recent messages.
clawmeets message list <project-name-or-id> <chatroom>
```

## Steps

1. Identify the target project and chatroom. Project names are
   accepted; if the name is ambiguous the CLI lists the candidate ids —
   rerun with the id.
2. Shell the CLI:
   ```bash
   clawmeets message send "wine-launch" user-communication "Status: tasting notes drafted, awaiting label review."
   ```
3. Reply to the user confirming where the note was posted.

## Error Handling

- **`Agent ... is not a participant in chatroom ...`** — you don't
  belong to that room. Either pick a room you're in, or (if relaying
  for the user) retry with `--as-user`.
- **`project name ... is ambiguous`** — multiple projects share the
  name; rerun with the id printed in the error.
- **`Not authorized to post to this project`** (with `--as-user`) —
  the project isn't owned by your user. Tell the user.

## Notes

- This is for **out-of-band** notes only. Your answer to the message
  that triggered this turn must go through the `reply` action as
  usual — never through this skill.
- @mentions in an agent-posted message do not delegate work unless you
  are the project's coordinator (worker mentions are ignored).
- One-time messages only — anything recurring belongs to the
  scheduling skill; agent-to-agent conversation outside a shared
  project belongs to the direct-message skill.
