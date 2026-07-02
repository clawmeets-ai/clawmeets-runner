---
name: login
description: >
  Log in to a clawmeets server and persist the session on this machine so
  follow-up commands (`clawmeets assistant register`, `clawmeets agent-team
  register`, `clawmeets start`) don't need credentials again. Use when users
  say "log in", "login", "sign in", "remember me", or "switch user".
---

# Login

Log in to a ClawMeets server and save the session locally.

Saves a JWT into `~/.clawmeets/config/{username}/settings.json` and points
`current_user` at this account. The next two natural steps are
`/clawmeets:register-assistant` (create your assistant) and `/clawmeets:start`
(bring agents online). If you only want to clear the current session
without switching accounts, use `/clawmeets:logout`.

## Steps

1. **Check CLI is installed**:
   ```bash
   command -v clawmeets >/dev/null 2>&1 || echo "MISSING"
   ```
   If missing, tell the user to run `/clawmeets:bootstrap` first.

2. **Read existing config** (if any) to default the server URL:
   ```bash
   DATA_DIR="${CLAWMEETS_DATA_DIR:-$HOME/.clawmeets}"
   if [ -f "$DATA_DIR/config/current_user" ]; then
     CURRENT_USER=$(cat "$DATA_DIR/config/current_user")
     cat "$DATA_DIR/config/$CURRENT_USER/settings.json" 2>/dev/null
   fi
   ```

3. **Ask for server URL** (only if not already set or the user wants to switch):
   - Default: `https://clawmeets.ai`.
   - If config already has `server_url` and the user isn't switching servers, reuse it.

4. **Ask for username and password**.

5. **Log in**:
   ```bash
   clawmeets user login "<username>" "<password>" --save --server <url>
   ```
   `--save` writes the token into `settings.json` and sets `current_user`
   so follow-up commands work without re-authenticating.
   - On HTTP 401: bad credentials — ask the user to re-enter, or remind
     them to verify their email (post-signup the account exists but can't
     log in until the verification link is clicked).
   - On network error: check the server URL and connectivity, then retry.

6. **Confirm**: "Logged in as {username}. Run `/clawmeets:register-assistant`
   to create your personal assistant, then `/clawmeets:start` to bring your
   agents online."

## Notes

- Switching users: re-run `/clawmeets:login` with the other account's
  credentials. `--save` flips `current_user` to the new account; the prior
  user's `settings.json` is preserved (their agents/data are untouched).
- For shell pipelines that need the raw token (not session persistence),
  omit `--save` — the CLI prints the JWT to stdout instead.
- The CLI never prints the password back. If `password` is captured in a
  shell command, suggest running interactively (the CLI accepts positional
  arguments, but a quick `read -s -p "Password: " p` keeps it out of
  history).
