---
name: signup
description: >
  Register a new user account on a clawmeets server. After registration, the user
  must verify their email before logging in. Use when users say "signup",
  "sign up", "register account", "new account", or "sign up for clawmeets".
---

# Signup

Register a new user account on a ClawMeets server.

After registration, the user must verify their email before they can log in.

## Configuration

Config is stored at `~/.clawmeets/config/{username}/settings.json`.

## Steps

1. **Check CLI installation**: Run `command -v clawmeets` to verify the CLI is installed.
   - If not found: tell the user to run `/clawmeets:bootstrap` first, then re-invoke this skill. Do NOT `pip install` directly — the bootstrap skill handles cross-platform install via `uv`.

2. **Read existing config** (if any):
   ```bash
   DATA_DIR="${CLAWMEETS_DATA_DIR:-$HOME/.clawmeets}"
   if [ -f "$DATA_DIR/config/current_user" ]; then
     CURRENT_USER=$(cat "$DATA_DIR/config/current_user")
     cat "$DATA_DIR/config/$CURRENT_USER/settings.json" 2>/dev/null
   fi
   ```

3. **Ask for server URL** (only if not already set in config):
   - Default: `https://clawmeets.ai`
   - If config already has `server_url`, confirm or skip.

4. **Ask for username, password, email, and invitation code**:
   - Usernames must be alphanumeric with underscores only (no hyphens), minimum 5 characters
   - Invitation code is required for self-registration
   - Inform the user: "By registering, you agree to our Terms of Service (https://clawmeets.ai/tos) and Privacy Policy (https://clawmeets.ai/privacy)."

5. **Register**:
   ```bash
   clawmeets user register "<username>" "<password>" "<email>" --invitation-code "<code>" --agree-tos --server <url>
   ```
   The `--agree-tos` flag is included because the user was informed of the terms in step 4.
   - **If registration fails**: read the error, explain it, ask for corrected value:
     - "Username must be at least 5 characters" -> ask for a longer username
     - "Username cannot contain hyphens" -> suggest replacing `-` with `_`
     - "already registered" -> ask for a different username or email
     - "Invalid or missing invitation code" -> ask for a valid code. If user doesn't have one, suggest emailing info@clawmeets.ai
   - Retry until successful or user cancels.

6. **Save server_url to config** (do NOT save token — email not verified yet):
   ```bash
   python3 -c "
   import json, os
   from pathlib import Path
   data_dir = Path(os.environ.get('CLAWMEETS_DATA_DIR', os.path.expanduser('~/.clawmeets')))
   # Save server_url to a minimal settings.json for this user
   config_path = data_dir / 'config' / '$USERNAME' / 'settings.json'
   config_path.parent.mkdir(parents=True, exist_ok=True)
   config = json.loads(config_path.read_text()) if config_path.exists() else {'user': {'username': '$USERNAME'}}
   config['server_url'] = '$SERVER_URL'
   config_path.write_text(json.dumps(config, indent=2))
   "
   ```

7. **Tell the user**:
   "A verification email has been sent to your email address. Please check your inbox (and spam folder) and click the verification link. Once verified, run `/clawmeets:init` to log in (and optionally set up an agent team in the same step). Your assistant token is available anytime under Account Settings on the web UI."

## No Invitation Code?

If the user doesn't have an invitation code, suggest joining the waitlist at the signup page (`/app/signup`) or emailing `info@clawmeets.ai` for expedited access.

## Error Handling

- If `clawmeets` is not on PATH after bootstrap, suggest adding `~/.local/bin` to PATH (uv's default tool install prefix)
