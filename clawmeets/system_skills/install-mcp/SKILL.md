---
name: install-mcp
description: >
  Install an MCP server on an agent. Use when the user says "install
  the <name> MCP on <agent>", "add the <name> integration", or
  similar. Two callsites: assistant-on-peer (`<agent>` argument) and
  worker/coordinator self-install (`self` argument).
---

# Install MCP Server

Install one or more MCP (Model Context Protocol) servers on a target
agent. After install, an OAuth-bearing MCP triggers an auto-auth
browser flow on the runner; a config-only MCP needs
`/clawmeets:update-mcp-config` next.

There are two callsites:

- **Assistant-on-another-agent** — you are `{username}-assistant` and
  the user asked you to install an MCP on one of their other agents.
  Pass the agent's name/id; the CLI picks up
  `$CLAWMEETS_ASSISTANT_TOKEN` via `_resolve_user_session`.
- **Worker/coordinator self-install** — you (a worker/coordinator)
  were asked to install an MCP on yourself. Use `self` as the agent
  argument; the CLI picks up `CLAWMEETS_AGENT_ID` /
  `CLAWMEETS_AGENT_TOKEN` / `CLAWMEETS_SERVER_URL` from the
  runner-injected env.

## Steps

1. **List MCPs already installed on an agent if asked**:
   ```bash
   clawmeets mcp list --agent <agent_name>
   ```
   Note: this only shows what's already installed on that agent (with
   auth status). To discover what's installable, ask the user — the
   server-curated MCP registry is small post-2026-06 (per `CLAUDE.md`,
   `chess` is the sole surviving MCP; new MCPs are added rarely).

2. **Install**:
   ```bash
   clawmeets mcp install <agent|self> <mcp_name> [<mcp_name_2> ...]
   ```

3. **Confirm** what was installed.

4. **If the MCP needs config**, suggest `/clawmeets:update-mcp-config`.
   If it needs OAuth, the runner pops a browser automatically; tell the
   user to complete the consent screen.

## Error handling

- "mcp not found in registry" → check the project's MCP registry (server-side `mcps/registry.json`); `clawmeets mcp list --agent <agent>` only shows what's already installed on that agent, not the catalog.
- "Only the agent itself, its owner, the user's assistant, or an admin
  can access this" → the target agent is owned by someone else; tell
  the user.
- "'self' requires CLAWMEETS_AGENT_ID..." — the env vars aren't set;
  you're probably not in an agent subprocess. Pass an explicit agent
  name/id instead.
