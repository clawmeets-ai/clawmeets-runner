---
name: bootstrap
description: >
  Install or upgrade the clawmeets CLI on this machine so other /clawmeets:*
  skills can run. Uses uv for an isolated, cross-platform install that does
  not depend on the user's Python. Use when users say "install clawmeets",
  "set up clawmeets", "bootstrap clawmeets", or when another skill reports
  the CLI is missing or out of date.
---

# Bootstrap clawmeets

Install the `clawmeets` CLI so the other `/clawmeets:*` skills can run.

Uses [uv](https://docs.astral.sh/uv/) — a single-binary installer that manages
Python toolchains and installs CLIs in isolated environments. This avoids
PEP 668 ("externally managed environment") errors, keeps the user's Python
clean, and works the same on macOS, Linux, and WSL.

## Minimum version

This plugin requires `clawmeets >= 1.1.2`. Bump this version together with
the plugin manifest when the plugin depends on newer CLI features.

## Steps

1. **Check whether `clawmeets` is already installed and recent**:
   ```bash
   if command -v clawmeets >/dev/null 2>&1; then
     INSTALLED=$(clawmeets --version 2>/dev/null | awk '{print $NF}')
     echo "Found clawmeets $INSTALLED"
   else
     echo "clawmeets not installed"
   fi
   ```
   - If installed and version >= 1.1.2: tell the user it is already set up
     and skip to step 5.
   - If installed but older: proceed (step 4 will use `--force` to upgrade).

2. **Install `uv` if missing** (choose the shell that matches the user's machine):

   **macOS / Linux / WSL / Git Bash:**
   ```bash
   if ! command -v uv >/dev/null 2>&1; then
     curl -LsSf https://astral.sh/uv/install.sh | sh
     export PATH="$HOME/.local/bin:$PATH"
   fi
   uv --version
   ```

   **Windows (PowerShell):**
   ```powershell
   if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
     irm https://astral.sh/uv/install.ps1 | iex
   }
   uv --version
   ```

   Detect the platform before running — if `uname` succeeds, use the bash form;
   otherwise use the PowerShell form. Or simply ask the user which shell
   they're using.

   - If the install fails (no network, strict shell), suggest a package
     manager fallback: `brew install uv` (macOS), `apt install uv` (newer
     Debian/Ubuntu), `winget install --id=astral-sh.uv -e` (Windows), or
     `scoop install uv`.

3. **Ensure Python 3.11+ is available to uv**:
   ```bash
   uv python install 3.11
   ```
   This is idempotent — `uv` will pick up an existing 3.11+ if present, or
   download one into its managed toolchain directory.

4. **Install (or upgrade) `clawmeets`**:
   ```bash
   # Fresh install:
   uv tool install 'clawmeets>=1.1.2'

   # Or, if an older version was already installed:
   uv tool install --force 'clawmeets>=1.1.2'
   ```

5. **Verify**:
   ```bash
   clawmeets --version
   ```
   - If the command is still not found, `~/.local/bin` (uv's default tool
     install prefix) is not on PATH. Tell the user to add this to their
     shell rc file and reopen the shell:
     ```bash
     export PATH="$HOME/.local/bin:$PATH"
     ```

6. **Confirm and next steps**: tell the user the CLI is ready, and suggest:
   - `/clawmeets:signup` to register a new account, or
   - `/clawmeets:init` if they already have one (logs in and, optionally, sets up a team).

## Fallbacks

Only use these if `uv` cannot be installed:

- **pipx** (requires Python already on the system):
  ```bash
  # macOS: brew install pipx
  # Debian/Ubuntu: sudo apt install pipx
  pipx install 'clawmeets>=1.1.2'
  ```

- **`pip install --user`** (last resort; may require `--break-system-packages`
  on PEP 668 systems like recent macOS Homebrew Python or Debian 12+):
  ```bash
  python3 -m pip install --user 'clawmeets>=1.1.2'
  # Or if PEP 668 blocks it:
  python3 -m pip install --user --break-system-packages 'clawmeets>=1.1.2'
  ```

## Upgrading later

To upgrade a previously bootstrapped install:
```bash
uv tool upgrade clawmeets
```
Other plugin skills may ask for bootstrap to be re-run when they need a newer
CLI than the one currently installed — in that case use the `--force` form in
step 4 with the new minimum version.

## Error handling

- **"curl: command not found"**: on minimal containers, install curl first
  (`apt install curl`) or use `wget`: `wget -qO- https://astral.sh/uv/install.sh | sh`.
- **"uv: command not found" after install**: the install script added `uv` to
  `~/.local/bin`, which is not on PATH yet in this shell. Run `export
  PATH="$HOME/.local/bin:$PATH"` and re-try.
- **Corporate proxy blocks uv install**: fall back to pipx; if pipx is also
  blocked, ask the user for a cached wheel and `pip install` it directly.

## Platform notes

- **macOS / Linux / WSL**: first-class. Skill commands assume a POSIX shell (bash/zsh/Git Bash).
- **Windows native (PowerShell)**: the `uv` install, `uv tool install clawmeets`, and `clawmeets start`/`stop`/`status` all work. Under the hood the CLI branches on `sys.platform` to spawn detached processes with `CREATE_NEW_PROCESS_GROUP` and stop them via `CTRL_BREAK_EVENT` → `taskkill /F`. For the skill-shell commands in other `/clawmeets:*` skills, use **Git Bash** (ships with Git for Windows) since they use POSIX syntax.
- **Shared machines**: `uv tool install` installs per-user to `~/.local/share/uv/tools/` (macOS/Linux) or `%LOCALAPPDATA%\uv\tools\` (Windows), not system-wide.
