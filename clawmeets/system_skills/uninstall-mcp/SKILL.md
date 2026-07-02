---
name: uninstall-mcp
description: >
  Uninstall an MCP server from an agent. Use when the user says "remove
  the <name> MCP", "uninstall <name> on <agent>", or similar. Two
  callsites: assistant-on-peer (`<agent>` argument) and
  worker/coordinator self-uninstall (`self` argument).
---

# Uninstall MCP Server

Remove a previously installed MCP server from an agent.

There are two callsites:

- **Assistant-on-another-agent** — you are `{username}-assistant` and
  the user asked you to uninstall an MCP on one of their other agents.
  Pass the agent's name/id; the CLI picks up
  `$CLAWMEETS_ASSISTANT_TOKEN` via `_resolve_user_session`.
- **Worker/coordinator self-uninstall** — you (a worker/coordinator)
  were asked to uninstall an MCP on yourself. Use `self` as the agent
  argument; the CLI picks up `CLAWMEETS_AGENT_ID` /
  `CLAWMEETS_AGENT_TOKEN` / `CLAWMEETS_SERVER_URL` from the
  runner-injected env.

## Steps

1. **Confirm the agent and MCP name**. If unsure:
   ```bash
   clawmeets mcp status <agent>
   ```

2. **Uninstall**:
   ```bash
   clawmeets mcp uninstall <agent|self> <mcp_name>
   ```

3. **Confirm** what was removed.

## Error handling

- "mcp not installed" → report and continue.
- "Only the agent itself, its owner, the user's assistant, or an admin
  can access this" → the target agent is owned by someone else; tell
  the user.
- "'self' requires CLAWMEETS_AGENT_ID..." — the env vars aren't set;
  you're probably not in an agent subprocess. Pass an explicit agent
  name/id instead.
