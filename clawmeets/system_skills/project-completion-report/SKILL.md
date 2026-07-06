---
name: project-completion-report
description: >
  Publish an interactive, chart-rich report for the project you are
  coordinating, so the user absorbs the findings + supporting rationale
  in one polished surface instead of scrolling chat history. INVOKE
  before wrapping up EVERY regular project — a completion report is
  required, not optional. The report renders inside the project's own UI
  ("Report" tab) — not in chat, not on /today. Two exceptions only: for
  git-shaped projects (FILES & STATE block shows a `Git repo` line) use
  `coding-project-completion-report` instead (it pairs the summary with
  an annotated diff review); and DM-shaped projects (a one-to-one direct
  message, no PLAN.md / milestones) don't get a report — a chat reply is
  the deliverable there.
---

# Project completion report

You — the coordinator — are wrapping up a project. The user has been
following along in chat, but the chat is long, scattered across rooms,
and hard to revisit. This skill is how you produce the *one place* the
user opens when they want to know what the project found and why.

## § When to use

A completion report is **required for every regular project** you
coordinate — publish one before `project_completed`, every time. It is
the user's polished take-away; the chat is not. Shape the report to what
the project produced (numerical findings, ranked options, a
recommendation with rationale, or structured data to scan) — but always
publish one.

Only two cases are exempt:

- **git-shaped projects** (FILES & STATE block shows a `Git repo` line) —
  use `coding-project-completion-report` instead; its diff-aware review
  is what users want for code changes. (You still publish a report — just
  the coding variant.)
- **DM-shaped projects** — a one-to-one direct message (no PLAN.md,
  milestones, or setup). There the chat reply IS the deliverable; don't
  publish a report.

Everything else — including projects whose deliverable is a file — gets a
report. When a file is the deliverable, keep the report short and link to
the file in **Sources**; the report still gives the user the verdict and
where to look. If you would otherwise write a 200+ word markdown summary
in `user-communication`, write the report instead — do not write both.

## § When in the project lifecycle

Publish the report **just before** you emit the `project_completed`
action — last thing, after all worker batches have returned and you've
verified acceptance criteria. Sequence:

1. Verify all acceptance criteria pass.
2. Author + publish the report (this skill).
3. Post one short line in `user-communication`: e.g., *"Wrap-up done —
   the report is on the project page."*
4. Emit `project_completed` in the same turn.

You CAN also publish (and re-publish) mid-project as a live progress
board if that's useful — the report is keyed by the project, not by
status. Re-running with the same project overwrites the previous
artifact and the user's open page re-renders in place.

## § What the report should contain

Default structure (deviate when the project shape demands it):

1. **Executive verdict** — one sentence at the top. The user should be
   able to read just this and know the outcome.
2. **2–6 conclusions** — each with a title and an expandable rationale
   drawer. The drawer is where the chain-of-reasoning, supporting
   numbers, and chat quotes live.
3. **Supporting charts** — one or more Chart.js charts (bar / line /
   doughnut / scatter / radar) showing the numerical evidence. Inline
   in the page, not behind a click.
4. **Methodology** (optional) — a short note on what data sources you
   used, what you excluded, and known caveats. The user wants to know
   what to trust.
5. **Sources** — links back to chatroom files / external URLs the user
   can verify. Use absolute server URLs for chatroom files:
   `/api/projects/<project_id>/chatrooms/<room>/files/<name>`.

## § The render contract

The page renders your report by invoking
`function(mount, data, lib)` — you supply the body. Same contract as
the `today` skill, same `lib` namespace. Recap:

- `lib.Chart` — Chart.js v4, all controllers pre-registered. Drop a
  `<canvas>` in `mount` and `new lib.Chart(canvas, { type, data,
  options })`.
- `lib.React`, `lib.ReactDOM` — React 18 for stateful widgets (the
  expandable conclusion list, tabs within the report).
- `lib.tokens` — design tokens (accent, fg, muted, border, fontSans,
  fontMono). Use these for any inline `style=` so the report stays on
  brand.
- `lib.row`, `lib.rowWithTime`, `lib.esc`, `lib.linkOrText` — HTML
  string helpers. Always `esc()` agent / user / file text before
  injecting into `innerHTML`.
- CSS classes shipped by the shell: `today-row`, `today-row__title`,
  `today-row__meta`, `today-row__right`, `today-link`, `today-empty`,
  `today-err`. Reuse them so the report visually matches the rest of
  the product.

Constraints:

- `data` ≤ 64 KB serialized JSON. `render_code_js` ≤ 64 KB.
- Render body must be self-contained — only `mount` / `data` / `lib`
  plus standard browser globals (`document`, `Math`, `Date`, …) are in
  scope.
- Errors are caught and shown as a `Render error: <message>` pill in
  the report tab. Test locally before publishing if you can.

## § How to publish

In your sandbox cwd:

1. Identify the project's id — it is in the `== FILES & STATE ==`
   block of your prompt header. Use that exact UUID.
2. Write two files:
   - `report-data.json` — your structured data (executive, conclusions,
     charts, sources).
   - `report-render.js` — the body of `function(mount, data, lib)`.
3. Shell:
   ```bash
   clawmeets project upsert-report <project_id> \
     --title "<Report title>" \
     --data report-data.json \
     --render-code report-render.js
   ```
   The CLI resolves your agent token + server URL from env
   (`CLAWMEETS_AGENT_ID`, `CLAWMEETS_AGENT_TOKEN`,
   `CLAWMEETS_SERVER_URL`). No `--token` flag needed. The server
   verifies you're the project's coordinator.
4. The frontend receives `PROJECT_REPORT_SYNC` over WebSocket and the
   "Report" tab appears (or refreshes in place) in real time.

To remove a stale report:

```bash
clawmeets project delete-report <project_id>
```

## § Recipes

A canonical data shape worth starting from:

```json
{
  "executive": "SE region drove all of Q1 growth; APAC slipped on weak retail.",
  "conclusions": [
    {
      "title": "SE region grew 23% YoY, vs. 4% company-wide.",
      "rationale": "Comparing Q1-2025 to Q1-2026 by region (SE: $4.1M → $5.0M; rest flat). Driver was the enterprise renewal cohort closing on time, not new logos."
    },
    {
      "title": "APAC weakness concentrated in retail vertical.",
      "rationale": "APAC retail down 18% YoY; APAC SaaS up 6%. Two anchor accounts churned in Feb."
    }
  ],
  "charts": [
    {
      "title": "YoY revenue by region",
      "kind": "bar",
      "labels": ["NA", "EU", "SE", "APAC"],
      "datasets": [
        {"label": "Q1-2025", "data": [12.0, 8.1, 4.1, 6.0]},
        {"label": "Q1-2026", "data": [12.4, 8.3, 5.0, 5.2]}
      ]
    }
  ],
  "sources": [
    {"label": "raw Q1 export", "href": "/api/projects/<id>/chatrooms/data/files/q1.csv"},
    {"label": "APAC churn notes", "href": "/api/projects/<id>/chatrooms/discussion/files/churn-2026.md"}
  ]
}
```

### Recipe A — Chart.js bar

```js
var head = document.createElement('div');
head.style.font = '14px ' + lib.tokens.fontSans;
head.style.color = lib.tokens.muted;
head.style.marginBottom = '4px';
head.textContent = data.charts[0].title;
mount.appendChild(head);

var canvas = document.createElement('canvas');
canvas.style.maxHeight = '280px';
mount.appendChild(canvas);

new lib.Chart(canvas, {
  type: data.charts[0].kind,
  data: {
    labels: data.charts[0].labels,
    datasets: data.charts[0].datasets.map(function (d, i) {
      return Object.assign({}, d, {
        backgroundColor: i === 0 ? lib.tokens.muted : lib.tokens.accent,
      });
    }),
  },
  options: { responsive: true, plugins: { legend: { position: 'bottom' } } },
});
```

### Recipe B — Expandable conclusions (vanilla DOM)

```js
var list = document.createElement('div');
data.conclusions.forEach(function (c) {
  var row = document.createElement('details');
  row.style.borderTop = '1px solid ' + lib.tokens.border;
  row.style.padding = '12px 0';
  var summary = document.createElement('summary');
  summary.style.cursor = 'pointer';
  summary.style.font = '600 15px ' + lib.tokens.fontSans;
  summary.style.color = lib.tokens.fg;
  summary.textContent = c.title;
  var body = document.createElement('div');
  body.style.marginTop = '8px';
  body.style.color = lib.tokens.muted;
  body.style.font = '14px ' + lib.tokens.fontSans;
  body.style.lineHeight = '1.55';
  body.textContent = c.rationale;
  row.appendChild(summary);
  row.appendChild(body);
  list.appendChild(row);
});
mount.appendChild(list);
```

### Recipe C — Sources list

```js
var html = '<div style="margin-top:24px;font:600 12px ' + lib.tokens.fontSans
  + ';text-transform:uppercase;letter-spacing:0.08em;color:' + lib.tokens.muted
  + '">Sources</div>';
data.sources.forEach(function (s) {
  html += lib.row(lib.linkOrText(s.href, s.label), '', '');
});
mount.insertAdjacentHTML('beforeend', html);
```

## § Voice

The report is the user's polished take-away. Match the voice of a
trusted analyst:

- Lead with verdict, not setup.
- Numbers in the conclusion line; rationale (how you got there) in the
  drawer.
- No "we ran a query and found…" — the user knows you ran a query.
- No emoji, no exclamation marks. Quiet confidence.

The page IS the deliverable. Don't summarize it in chat afterward.
