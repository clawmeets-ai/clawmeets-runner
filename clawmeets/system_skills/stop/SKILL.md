---
name: stop
description: >
  Stop clawmeets agent runner(s) for the current logged-in user.
  Use when users say "stop clawmeets", "stop agent", "kill runner",
  "clawmeets stop", or "shut down agent".
---

# Stop Agents

Stop agent runner(s) for the current logged-in user.

Delegates to `clawmeets stop`, which reads the user's `settings.json`, finds
each agent's PID file under `{agent_dir}/agent.pid`, and signals the process.

## Steps

1. **Check `clawmeets` is installed**:
   ```bash
   command -v clawmeets >/dev/null 2>&1
   ```
   - If missing, tell the user to run `/clawmeets:bootstrap` first.

2. **Stop all agents for the current user**:
   ```bash
   clawmeets stop
   ```
   - To stop a different user's agents: `clawmeets stop --user <username>`

3. **Report the result**: echo the CLI output to the user (it tells how many
   agents were stopped, or "No agents were running.").

## Notes

- Do not touch PID files by hand. The CLI owns that state.
- To check what is currently running without stopping: `clawmeets status`.
- To start agents again: `/clawmeets:start`.
