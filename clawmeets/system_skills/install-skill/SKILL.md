---
name: install-skill
description: >
  Install a server-curated skill on an agent. Use when the user asks
  "install <skill> on <agent>", "add the <skill> capability to
  <agent>", "give <agent> the <skill> skill", or similar. Two
  callsites: assistant-on-peer (`<agent>` argument) and
  worker/coordinator self-install (`self` argument).
  ALSO use when the user's request needs a capability you don't
  have — they reference a product page, dashboard, integration, or
  data source you have no installed skill for, or they need a real
  website driven (login walls, JavaScript-rendered pages, multi-step
  interaction) beyond what plain fetching can do. Browse the catalog
  (`clawmeets skill list`); if a skill's description matches the ask,
  self-install it and continue with the original request in the same
  turn. Canonical case: the /today page — a built-in dashboard agents
  populate by publishing tabs. Any ask to put, show, or refresh
  something "on my today page / today tab / today dashboard" means:
  install the `today` skill on yourself if missing, then follow it.
---

# Install Skill

Install one or more skills (from the server's skill registry) on a target
agent.

There are two callsites:

- **Assistant-on-another-agent** — you are `{username}-assistant` and
  the user asked you to install a skill on one of their other agents.
  Pass the agent's name/id; the CLI picks up
  `$CLAWMEETS_ASSISTANT_TOKEN` via `_resolve_user_session`.
- **Worker/coordinator self-install** — you (a worker/coordinator)
  were asked to install a skill on yourself. Use `self` as the agent
  argument; the CLI picks up `CLAWMEETS_AGENT_ID` /
  `CLAWMEETS_AGENT_TOKEN` / `CLAWMEETS_SERVER_URL` from the
  runner-injected env.

## Discover (capability-gap path)

The user didn't name a skill — they asked for something you can't do
with what's installed (drive a real website behind a login or
JavaScript, read a mailbox, generate images, publish to a product
page you don't recognize). Before saying you can't — and before
settling for a plain web fetch against a site that needs a real
browser:

1. Browse the catalog (no auth needed):
   ```bash
   clawmeets skill list
   ```
   Scan name + description for a match to the ask. The one case you
   already know without browsing: anything about the user's **/today
   page** (today tab, today dashboard) is the `today` skill.
2. On a candidate, confirm fit from the SKILL.md preview:
   ```bash
   clawmeets skill show <name>
   ```
3. Match → `clawmeets skill install self <name>`, then follow the
   freshly installed skill and complete the user's original ask in the
   same turn. Mention the install in one clause of your reply ("set
   myself up to publish to your today page, then …") — don't make it a
   separate confirmation round-trip.
4. No match → say plainly what you can't do. Don't install
   speculatively.

Self-install in direct service of the user's ask needs no approval
round-trip — it's reversible and visible in their catalog. Installing
on a **peer** still requires the user's explicit approval first.

Example: the user DMs you "show your best deals on my today page".
That's the /today case → `clawmeets skill install self today` → follow
the today skill to publish the tab → reply with its one-line
confirmation, noting the install.

## Steps

1. **Identify the agent and skill(s)**. If the user named the agent without
   the owner prefix (e.g. "research" instead of "{username}-research"), use
   the bare name — the CLI resolves it via the saved session.

2. **List available skills if asked**:
   ```bash
   clawmeets skill list
   ```

3. **Install**:
   ```bash
   clawmeets skill install <agent|self> <skill_name_1> [<skill_name_2> ...]
   ```

4. **Confirm** what was installed. If the CLI reports nothing new
   (already installed), say so.

5. **If the skill needs config**, mention that the user can run
   `/clawmeets:update-skill-config` (or use the web UI's Configure pill)
   to set it up.

## Error handling

- "skill not found in registry" → list `clawmeets skill list` so the user
  can pick a valid name.
- "Only the agent itself, its owner, the user's assistant, or an admin
  can access this" → the target agent is owned by someone else; tell
  the user.
- "'self' requires CLAWMEETS_AGENT_ID..." — the env vars aren't set;
  you're probably not in an agent subprocess. Pass an explicit agent
  name/id instead.
- "not logged in" → `$CLAWMEETS_ASSISTANT_TOKEN` is missing; ask the user
  to restart the assistant runner.
