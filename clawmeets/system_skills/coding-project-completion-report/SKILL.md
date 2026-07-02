---
name: coding-project-completion-report
description: >
  Publish a final interactive report for a git-shaped project (FILES &
  STATE block shows a `Git repo` line). The report is the summary of the
  change PLUS the coordinator's overall review of the change PLUS an
  annotated diff review — what was done, why, the holistic verdict, key
  files, risk areas, and the diff with inline reviewer comments. Invoke
  BEFORE `project_completed` when wrapping a coding project. For
  non-coding projects, use `project-completion-report` instead.
---

# Coding project completion report

You — the coordinator — are wrapping up a coding project. The user has
been following along in chat, but the chat is long and the actual
*shape* of the change is buried across diffs in many commits. This
skill is how you produce the one place the user opens when they want
to know what was built and where to look first.

The output is a tab on the project page, rendered via the same
self-serve protocol as `/today` tabs: a `data` JSON + a small
`render_code_js` body that runs against an inline mount with the same
`lib` namespace (Chart.js, design tokens, helpers).

## § When to use

Author this iff ALL of:

- The project's FILES & STATE block shows a `Git repo` line (it's
  git-tracked).
- The diff is non-trivial: more than one file, or a single file with
  a non-obvious change.
- Acceptance criteria have passed.

Skip when:

- The diff is one tiny edit a chat reply already explained.
- The project never produced code (use `project-completion-report`).
- The project failed and you're recording why (use a plain chat note).

## § When in the project lifecycle

Publish the report **just before** you emit `project_completed`.
Sequence:

1. Verify acceptance criteria pass.
2. Gather the diff and write the report (this skill).
3. Post one short line in `user-communication`: *"Wrap-up done — the
   review is on the project page."*
4. Emit `project_completed`.

You CAN also publish mid-project as a live review board if the user
asks for one; re-running overwrites and the open page re-renders in
place.

## § Gather the diff

The agent-env injects `$CLAWMEETS_AGENT_DIR` and the FILES & STATE
block shows the repo subdir (`sandbox/projects/<name-id>/repos/<repo>`).
Inside the repo:

```bash
REPO="$( … repo dir from FILES & STATE … )"
PROJ_BRANCH="project/<project_name>"

# Pick the base: the commit the project branched from
BASE=$(git -C "$REPO" merge-base origin/main "$PROJ_BRANCH" \
       || git -C "$REPO" merge-base main "$PROJ_BRANCH")

# Overview: per-file +/- stats
git -C "$REPO" log --stat "$BASE..$PROJ_BRANCH"

# Full diff to read + summarize
git -C "$REPO" diff "$BASE..$PROJ_BRANCH"

# Per-commit messages (useful for the "summary" bullets)
git -C "$REPO" log --pretty=format:"%h %s" "$BASE..$PROJ_BRANCH"
```

If the project doesn't have a `project/<name>` branch (no merges yet —
unlikely at wrap-up), fall back to `HEAD` against the repo's default
branch.

Capture the full `git diff "$BASE..$PROJ_BRANCH"` output verbatim into
`unified_diff` in `data.json` (it's plain text — escape backslash + quote
when JSON-encoding). The renderer (`lib.Diff2HtmlUI`) consumes the raw
unified-diff text directly; do NOT pre-parse it into structured lines.

Then walk the diff with the LLM lens to produce the human layer:

- For each file, identify the *intent* in one sentence and put it in
  `diff_files[].summary` (keyed by `path`).
- For lines that need a reviewer note — security boundary, surprising
  side effect, "is this intended?" — emit an `annotations` entry with
  `file_path` + `new_line` (or `old_line` for deletions) + a short
  `comment`. Skip the lines that don't need a note.
- Flag the highest-stakes ones in `risk_areas` so they bubble up at
  the top of the page before the reader scrolls into the diff.

Then step back from the per-file / per-line detail and write `review` —
your overall verdict on the change *as a whole*: does it do what the
request asked, is it correct and complete, what's the quality /
architecture call, and is there anything that worries you across files
(not tied to one line). This is the senior-reviewer sign-off, not a
restatement of the summary bullets.

## § What to populate

You write two artifacts:

1. **`data.json`** — opaque JSON; the render body consumes it.
   Recommended shape:

   ```json
   {
     "executive": "Migrated session-based auth to JWT; backward-compatible cookie shim for legacy clients.",
     "summary": [
       "Replaced session storage with JWT issuance in `auth/login.py`.",
       "Added shim middleware so cookie-only clients still authenticate.",
       "Removed `sessions` Redis dependency from prod config."
     ],
     "review": "The change fully satisfies the request: every entry point that issued a session cookie now mints a JWT, and the cookie shim keeps legacy clients working, so nothing is left half-migrated.\n\nThe split between issuance (`login.py`) and the verification middleware is clean and the test coverage tracks the new paths. My one reservation is operational rather than correctness: the shim is meant to be temporary, so this needs a follow-up to retire it — otherwise we carry two auth paths indefinitely.",
     "risk_areas": [
       {"file": "src/auth/login.py", "issue": "Token returned in response body — verify no caller logs the body."},
       {"file": "config/prod.yaml", "issue": "Redis stanza removed; downstream tooling that connected here will fail."}
     ],
     "metrics": {"files_changed": 14, "additions": 312, "deletions": 187, "commits": 9},
     "unified_diff": "diff --git a/src/auth/login.py b/src/auth/login.py\n--- a/src/auth/login.py\n+++ b/src/auth/login.py\n@@ -45,7 +45,12 @@\n def login(req):\n-    session.user_id = user.id\n+    token = jwt.encode(payload, key)\n+    return {'token': token}\n",
     "diff_files": [
       {
         "path": "src/auth/login.py",
         "summary": "Swap session cookie for JWT issuance."
       }
     ],
     "annotations": [
       {
         "file_path": "src/auth/login.py",
         "new_line": 49,
         "comment": "Returning token in body — make sure callers don't log this."
       }
     ]
   }
   ```

   - `review` is your **holistic review of the entire change** — 1–3
     tight paragraphs (plain text, blank line between paragraphs) giving
     the coordinator's overall verdict: correctness, completeness vs the
     request, the quality/architecture call, and cross-cutting concerns.
     Distinct from `summary` (what was done) and from `risk_areas`
     (specific spots to inspect). This is the one place the reader gets
     your judgment of the change as a whole.
   - `unified_diff` is the **raw output of `git diff`** — multi-file,
     standard unified-diff format. The `diff2html` renderer in `lib`
     consumes it directly. Don't pre-parse into structured lines.
   - `diff_files` carries the per-file **summary** (one sentence on
     intent) — keyed by `path` so the renderer can pair each file's
     diff2html block with its summary. Omit a file entry to use the
     default file header alone.
   - `annotations` is the **annotated** half: zero or more reviewer
     comments anchored to a specific file + line. The post-render
     overlay finds the matching row in the rendered diff and injects
     a chip next to it. Anchor with `new_line` (line number in the
     NEW file) for additions / context, or `old_line` for deletions.

2. **`render.js`** — the BODY of `function(mount, data, lib)`. No
   signature, no surrounding braces. See render recipes below.

## § Size budget

Both files cap at 64 KB. The full diff of a moderate project can
exceed that. Strategies, in priority order:

- **Filter `git diff` to what matters** — use `git diff "$BASE..HEAD"
  -- ':!path1' ':!path2'` to exclude lockfiles, snapshot tests,
  generated bundles before piping into `unified_diff`. The diff is for
  *review*, not as a complete change archive.
- **Truncate per-file via git pathspec or `--stat` summary** — for
  vendored / autogenerated files, mention them as a one-liner in
  `summary` or `risk_areas` instead of inlining their diff.
- **Annotate sparingly** — `annotations` adds noise per chip; flag the
  5–10 lines that genuinely warrant a reviewer note, not every change.
- **Last resort** — split into two reports (e.g.,
  "backend changes" + "frontend changes"). The route only stores one
  report per project; pick the highest-signal one.

## § Render recipes

The render runs inside a freshly-cleared `mount` div with `data` and
`lib` in scope. Available helpers (same as the `today` skill):
`lib.Chart`, `lib.React`, `lib.tokens` (`accent`, `accentDeep`, `fg`,
`muted`, `border`, `warn`, `warnBg`, `fontSans`, `fontMono`),
`lib.esc`, `lib.row`, `lib.linkOrText`, `lib.Diff2HtmlUI`.

**Reading-order contract — high level first.** The reader scrolls
top → bottom and stops when they have what they need. Author the body
in exactly this order:

1. **Verdict** (Recipe A's executive line) — one sentence, displayed
   prominently. The reader could read JUST this and know the outcome.
2. **Summary bullets** (Recipe A) — 3-5 lines of what was done.
3. **Reviewer's assessment** (Recipe A) — your `review`: the holistic
   verdict on the whole change, in prose. Sits above the per-spot detail.
4. **Risk areas** (Recipe A) — 2-5 places the reviewer should look
   first, in a warn-styled callout. Surface here, not inside the diff.
5. **Metrics doughnut** (Recipe B) — at-a-glance change shape.
6. **Annotated diff viewer** (Recipe C) — the deep dive, last.

Don't reorder. If risks are load-bearing for the verdict, surface
them in step 4 — never hide them in chip-only annotations inside
step 6, which the reader may never reach.

### Recipe A — Header + executive + summary + reviewer's assessment + risk areas

```js
mount.style.font = '14px ' + lib.tokens.fontSans;

// Verdict: visually elevated above the bullets so the reader can read
// just this and know the outcome. Accent border-left + larger font.
var exec = document.createElement('p');
exec.style.font = '700 20px ' + lib.tokens.fontSans;
exec.style.color = lib.tokens.fg;
exec.style.lineHeight = '1.35';
exec.style.margin = '0 0 20px 0';
exec.style.padding = '4px 0 4px 16px';
exec.style.borderLeft = '4px solid ' + lib.tokens.accent;
exec.textContent = data.executive;
mount.appendChild(exec);

if (data.summary && data.summary.length) {
  var sumWrap = document.createElement('div');
  sumWrap.style.marginBottom = '20px';
  var ul = document.createElement('ul');
  ul.style.margin = '0';
  ul.style.paddingLeft = '20px';
  ul.style.color = lib.tokens.fg;
  ul.style.lineHeight = '1.6';
  data.summary.forEach(function (s) {
    var li = document.createElement('li');
    li.textContent = s;
    ul.appendChild(li);
  });
  sumWrap.appendChild(ul);
  mount.appendChild(sumWrap);
}

// Reviewer's assessment: the holistic verdict on the whole change.
// Neutral (muted eyebrow) — it's the sign-off, not a warning.
if (data.review) {
  var rev = document.createElement('div');
  rev.style.margin = '0 0 24px 0';
  rev.style.padding = '12px 14px';
  rev.style.border = '1px solid ' + lib.tokens.border;
  rev.style.borderRadius = '6px';
  var rh = document.createElement('div');
  rh.style.font = '700 12px ' + lib.tokens.fontSans;
  rh.style.textTransform = 'uppercase';
  rh.style.letterSpacing = '0.08em';
  rh.style.color = lib.tokens.muted;
  rh.style.marginBottom = '8px';
  rh.textContent = "Reviewer's assessment";
  rev.appendChild(rh);
  data.review.split(/\n\s*\n/).forEach(function (para) {
    if (!para.trim()) return;
    var p = document.createElement('p');
    p.style.margin = '0 0 8px 0';
    p.style.color = lib.tokens.fg;
    p.style.lineHeight = '1.6';
    p.textContent = para.trim();
    rev.appendChild(p);
  });
  mount.appendChild(rev);
}

if (data.risk_areas && data.risk_areas.length) {
  var risk = document.createElement('div');
  risk.style.background = lib.tokens.warnBg;
  risk.style.border = '1px solid ' + lib.tokens.warn;
  risk.style.borderRadius = '6px';
  risk.style.padding = '12px 14px';
  risk.style.margin = '0 0 24px 0';
  var h = document.createElement('div');
  h.style.font = '700 12px ' + lib.tokens.fontSans;
  h.style.textTransform = 'uppercase';
  h.style.letterSpacing = '0.08em';
  h.style.color = lib.tokens.warn;
  h.style.marginBottom = '8px';
  h.textContent = 'Review carefully';
  risk.appendChild(h);
  data.risk_areas.forEach(function (r) {
    var line = document.createElement('div');
    line.style.marginTop = '6px';
    line.style.color = lib.tokens.fg;
    line.innerHTML = '<code style="font:600 12px ' + lib.tokens.fontMono
      + ';color:' + lib.tokens.accentDeep + '">'
      + lib.esc(r.file) + '</code> — '
      + lib.esc(r.issue);
    risk.appendChild(line);
  });
  mount.appendChild(risk);
}
```

### Recipe B — Metrics doughnut (additions vs deletions)

```js
if (data.metrics) {
  var head = document.createElement('div');
  head.style.font = '600 12px ' + lib.tokens.fontSans;
  head.style.textTransform = 'uppercase';
  head.style.letterSpacing = '0.08em';
  head.style.color = lib.tokens.muted;
  head.style.margin = '12px 0 8px 0';
  head.textContent = data.metrics.files_changed + ' files changed · '
    + data.metrics.commits + ' commits';
  mount.appendChild(head);

  var wrap = document.createElement('div');
  wrap.style.maxWidth = '240px';
  wrap.style.margin = '0 0 24px 0';
  var canvas = document.createElement('canvas');
  wrap.appendChild(canvas);
  mount.appendChild(wrap);

  new lib.Chart(canvas, {
    type: 'doughnut',
    data: {
      labels: ['Additions', 'Deletions'],
      datasets: [{
        data: [data.metrics.additions, data.metrics.deletions],
        backgroundColor: ['#22863a', '#b31d28'],
        borderWidth: 0,
      }],
    },
    options: {
      cutout: '65%',
      plugins: { legend: { position: 'bottom' } },
    },
  });
}
```

### Recipe C — Annotated diff viewer (diff2html + annotation overlay)

```js
// 1. Render the diff via diff2html. `lib.Diff2HtmlUI` is the class
//    exposed by the page bundle; CSS is loaded for you. Side-by-side
//    is friendlier for big diffs; line-by-line for compact ones.
var diffMount = document.createElement('div');
diffMount.style.marginTop = '8px';
mount.appendChild(diffMount);

var ui = new lib.Diff2HtmlUI(diffMount, data.unified_diff || '', {
  outputFormat: 'line-by-line',  // or 'side-by-side' for wide diffs
  drawFileList: true,
  matching: 'lines',
  highlight: true,
});
ui.draw();

// 2. Overlay per-file summaries from data.diff_files[].
var summaries = {};
(data.diff_files || []).forEach(function (f) {
  if (f.summary) summaries[f.path] = f.summary;
});
diffMount.querySelectorAll('.d2h-file-wrapper').forEach(function (wrap) {
  var nameEl = wrap.querySelector('.d2h-file-name');
  if (!nameEl) return;
  var path = nameEl.textContent.trim();
  var summary = summaries[path];
  if (!summary) return;
  var header = wrap.querySelector('.d2h-file-header');
  if (!header) return;
  var note = document.createElement('div');
  note.style.padding = '6px 14px';
  note.style.borderTop = '1px solid ' + lib.tokens.border;
  note.style.background = '#fafbfc';
  note.style.font = '400 13px ' + lib.tokens.fontSans;
  note.style.color = lib.tokens.muted;
  note.textContent = summary;
  header.parentNode.insertBefore(note, header.nextSibling);
});

// 3. Inject reviewer-comment chips from data.annotations[].
//    Anchor by file_path + (new_line OR old_line). diff2html
//    line-by-line mode emits each row with two line-number cells —
//    `.line-num1` (old) and `.line-num2` (new); we match against
//    those, then append a chip into `.d2h-code-line-ctn`.
(data.annotations || []).forEach(function (ann) {
  var wrap = Array.prototype.find.call(
    diffMount.querySelectorAll('.d2h-file-wrapper'),
    function (w) {
      var n = w.querySelector('.d2h-file-name');
      return n && n.textContent.trim() === ann.file_path;
    }
  );
  if (!wrap) return;
  var targetCol = ann.old_line != null ? 'line-num1' : 'line-num2';
  var targetVal = String(ann.old_line != null ? ann.old_line : ann.new_line);
  wrap.querySelectorAll('tr').forEach(function (tr) {
    var cell = tr.querySelector('.' + targetCol);
    if (!cell || cell.textContent.trim() !== targetVal) return;
    var line = tr.querySelector('.d2h-code-line-ctn');
    if (!line) return;
    var chip = document.createElement('span');
    chip.style.marginLeft = '12px';
    chip.style.background = lib.tokens.warnBg;
    chip.style.color = lib.tokens.warn;
    chip.style.padding = '1px 8px';
    chip.style.borderRadius = '10px';
    chip.style.font = '600 11px ' + lib.tokens.fontSans;
    chip.style.whiteSpace = 'normal';
    chip.textContent = ann.comment;
    line.appendChild(chip);
  });
});
```

Combine all three recipes in one render body — header + bullets +
reviewer's assessment + risks at top (Recipe A), metrics doughnut in
the middle (Recipe B), annotated diff viewer below (Recipe C).

## § Publish

In your sandbox cwd:

1. Find the project's id — it's in the FILES & STATE block of your
   prompt header. Use that exact UUID.
2. Write the two files:
   - `report-data.json` — the data shape above.
   - `report-render.js` — the body, copying + adapting the recipes.
3. Shell:
   ```bash
   clawmeets project upsert-report <project_id> \
     --title "<short title — e.g., 'JWT migration — review'>" \
     --data report-data.json \
     --render-code report-render.js
   ```
   The CLI resolves your agent token + server URL from
   `CLAWMEETS_AGENT_ID` / `CLAWMEETS_AGENT_TOKEN` /
   `CLAWMEETS_SERVER_URL`. No `--token` flag needed. The server
   verifies you're the project's coordinator.
4. The frontend receives `PROJECT_REPORT_SYNC` over WebSocket and the
   "Report" tab on the project page appears (or refreshes in place).

To remove a stale report:

```bash
clawmeets project delete-report <project_id>
```

## § Voice

The review is the user's polished take-away. Match the voice of a
trusted senior reviewer:

- Lead with the verdict (`executive`), not the setup.
- The `review` is your overall sign-off on the whole change — give a
  clear, honest verdict (call out weaknesses, not just praise). A few
  sentences, not a wall of text.
- Comments on lines are short, concrete, actionable — "verify callers
  don't log this" beats "this might be sensitive."
- Risk-areas section flags 2–5 places to look first, not every change.
- No emoji, no exclamation marks, no "we ran git diff."

The page IS the deliverable. Don't summarize it in chat afterward.
