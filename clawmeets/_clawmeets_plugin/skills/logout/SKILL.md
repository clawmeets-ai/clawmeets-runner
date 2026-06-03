---
name: logout
description: >
  Log out of clawmeets. Clears the current user session but keeps all user
  data and agent configurations. Use when users say "logout", "log out",
  "clawmeets logout", or "switch account".
---

# Logout

Log out of the current ClawMeets session.

Clears the saved JWT token from `settings.json`. Does NOT stop running agents
(they have their own per-agent tokens) and does NOT delete any user data. To
switch users, just run `/clawmeets:init` with the other account's credentials
— it flips `current_user` on login. Use this logout only when you want to
clear the session on a shared machine without switching to a new user.

## Steps

1. **Check CLI is installed**:
   ```bash
   command -v clawmeets >/dev/null 2>&1 || echo "MISSING"
   ```
   If missing, tell the user to run `/clawmeets:bootstrap` first.

2. **Log out**:
   ```bash
   clawmeets user logout
   ```
   Clears `user.token` from the current user's `settings.json`. The CLI will
   print "Not logged in" if no `current_user` is set — in that case, tell the
   user they are already logged out.

3. **Optional: also clear the current_user pointer**:
   If the user says "completely log out" or "forget me", add `--clear-current-user`:
   ```bash
   clawmeets user logout --clear-current-user
   ```

4. **Confirm**: "Logged out. Run `/clawmeets:init` to log in again."

## Notes

- Running agents are unaffected. To stop them too, suggest `/clawmeets:stop`.
- To log out a specific user (not the current one): `clawmeets user logout --user <name>`.
