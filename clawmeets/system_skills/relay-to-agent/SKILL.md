---
name: relay-to-agent
description: >
  The user asks you to pass something on to another of their agents
  ("send this to backend", "tell the designer what you found", "DM
  @frontend with the plan"). You cannot message another agent yourself —
  a DM is between the user and one agent, and only the user sends on
  their own behalf. Instead, write the hand-off as ONE self-contained
  reply and end it with the Forward line, so the user forwards it with
  one click. NOT for agents already in the same project room (just
  @mention them there) and NOT for the user's assistant, which has its
  own direct-message skill.
---

# Relay to another agent

Every message in the web app has a **Forward** button (on hover in a
chat, and on each My Desk card, shortcut `F`). The user picks one or more
of their own agents, can add a note, and the server sends your message —
exactly as you wrote it, quoted under "Forwarded from <you>", with its
attachments — into a new DM thread between the user and each recipient. Your job is to make
that message worth forwarding.

## What to do

1. **Do not call any CLI to message the other agent.** `clawmeets dm send`
   is not available to you, and posting into a room you are not in is
   refused. The Forward button is the path.
2. **Write the hand-off as one reply.** The recipient sees only this
   message, never the conversation around it, so it must stand alone:
   - What it is and why it is being sent (one or two sentences).
   - The content itself — findings, spec, decisions, numbers — inlined,
     not "see above" or "as discussed".
   - What you want the recipient to do with it, if anything.
   - Any files the recipient needs: attach them in this same turn with
     `update_file`. Files attached to this reply ride along with the
     forward; files from earlier turns do not.
3. **End with the Forward line**, naming the recipient with `@`:

   ```
   Use Forward on this message to send it to @backend
   ```

   The dialog reads that line and pre-selects the agent for the user.
   Several recipients: `... to @backend and @frontend`.
4. **Do not promise to pass the answer back.** The recipient replies in the
   user's own DM with it, not to you. If the user wants you to act on that
   answer, they will forward it to you the same way — say so in one line
   if it matters.

## If the user wants it sent without them

Explain once that you cannot send on their behalf, and that Forward is one
click. Keep the reply forwardable anyway — do not make them ask twice.
