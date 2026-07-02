---
name: uninstall-skill
description: >
  Uninstall a skill from an agent. Use when the user says "remove
  <skill> from <agent>", "uninstall <skill> on <agent>", or similar.
  Two callsites: assistant-on-peer (`<agent>` argument) and
  worker/coordinator self-uninstall (`self` argument).
---

# Uninstall Skill

Remove a previously installed skill from an agent.

There are two callsites:

- **Assistant-on-another-agent** — you are `{username}-assistant` and
  the user asked you to uninstall a skill on one of their other agents.
  Pass the agent's name/id; the CLI picks up
  `$CLAWMEETS_ASSISTANT_TOKEN` via `_resolve_user_session`.
- **Worker/coordinator self-uninstall** — you (a worker/coordinator)
  were asked to uninstall a skill on yourself. Use `self` as the agent
  argument; the CLI picks up `CLAWMEETS_AGENT_ID` /
  `CLAWMEETS_AGENT_TOKEN` / `CLAWMEETS_SERVER_URL` from the
  runner-injected env.

## Steps

1. **Confirm the agent and skill name**. If unsure, list what's installed:
   ```bash
   clawmeets skill installed <agent|self>
   ```

2. **Uninstall**:
   ```bash
   clawmeets skill uninstall <agent|self> <skill_name>
   ```

3. **Confirm** what was removed.

## Error handling

- "skill not installed" → the agent didn't have it; report and continue.
- "Only the agent itself, its owner, the user's assistant, or an admin
  can access this" → the target agent is owned by someone else; tell
  the user.
- "'self' requires CLAWMEETS_AGENT_ID..." — the env vars aren't set;
  you're probably not in an agent subprocess. Pass an explicit agent
  name/id instead.
