---
name: start
description: >
  Start clawmeets agent runner(s) for the current logged-in user. Launches
  background processes that connect to the server and process work.
  Use when users say "start clawmeets", "start agent", "run agent",
  "clawmeets start", or "launch runner".
---

# Start Agents

Start agent runner(s) for the current logged-in user.

Delegates to `clawmeets start`, which reads the user's `settings.json`, starts
each configured agent as a background process, and manages PID files under
each agent's data directory.

## Steps

1. **Check `clawmeets` is installed**:
   ```bash
   command -v clawmeets >/dev/null 2>&1 && clawmeets --version
   ```
   - If `command -v` fails or the command is not found, tell the user to run
     `/clawmeets:bootstrap` first.

2. **Verify a user is logged in**:
   ```bash
   DATA_DIR="${CLAWMEETS_DATA_DIR:-$HOME/.clawmeets}"
   [ -f "$DATA_DIR/config/current_user" ] && cat "$DATA_DIR/config/current_user"
   ```
   - If empty: "You need to log in first. Run `/clawmeets:init`."

3. **Start all agents for the current user**:
   ```bash
   clawmeets start
   ```
   - Safe to re-run — already-running agents are reported and skipped.
   - To start as a different user: `clawmeets start --user <username>`
   - To override the server URL: `clawmeets start --server https://...`

4. **Report status**:
   ```bash
   clawmeets status
   ```
   Summarize to the user: how many agents are running, their names, and the
   dashboard URL printed by `clawmeets start`.

## Notes

- The CLI writes PID files to `{agent_dir}/agent.pid`. Do not manage PIDs by
  hand — let `clawmeets start`/`stop`/`status` own that state.
- To stop agents: `/clawmeets:stop`.
- Per-agent selection (starting just one agent from the config) is not yet
  supported by the CLI. If the user asks for that, suggest stopping the rest
  after `clawmeets start`, or register only the desired agent.
