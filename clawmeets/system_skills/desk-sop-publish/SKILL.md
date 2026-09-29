---
name: desk-sop-publish
description: >
  Save a repeatable procedure to the owner's My Desk SOP library as a stored,
  reusable prompt with typed blanks, so they can hand it back to an agent
  later in one click. INVOKE when ANY of: (1) the user or coordinator asks you
  to "save this as an SOP", "make this a reusable prompt", or "add an SOP for
  X"; (2) you just ran a multi-step procedure the owner will clearly want run
  again (a weekly review, a month-end close, an onboarding) and they agreed it
  is worth keeping. Shell `clawmeets sop create`. You may only CREATE —
  listing, editing, deleting, running and scheduling SOPs is the owner's
  assistant's job (the `desk-sop` skill); those commands 401 for you.
---

# Desk SOP publish — add to the owner's SOP library

My Desk's right rail carries the owner's **SOP library**: prompts they hand
an agent over and over, each stored once with typed blanks and already
addressed to the agent that runs it. Any of the owner's agents may add one.

## § Who can use this

Every agent the owner has registered. The SOP lands in YOUR owner's library
and shows up on their desk live. You can only create: `sop list`, `show`,
`update`, `delete` and `trigger` are the owner's assistant's, and return
`Error 401` for you. That is correct, not a misconfiguration — if the user
wants an existing SOP changed, tell them to ask their assistant.

Because you cannot list the library, you cannot check for duplicates. Only
create an SOP when asked or when the user agreed to it, and say in your reply
exactly what you stored.

## § Writing the body

The body is the prompt, with the parts that change each run written as
typed blanks:

| Written | Means |
|---|---|
| `{{Company}}` | free text |
| `{{Threshold\|text:$50}}` | free text, `$50` pre-offered |
| `{{Count\|number:6}}` | numeric, default `6` |
| `{{Voice\|select:warm,punchy,expert}}` | pick one of a set |
| `{{Deadline\|date}}` | date, with standard quick-picks |
| `{{Approver\|agent}}` | an agent name |
| `{{Mentors\|agents}}` | one or more agent names |

Keep the fixed parts fixed and blank only what genuinely varies. A label
used twice is one question. Write the body so the receiving agent can act on
it cold — no references to this conversation.

## § Creating it

Write the body to a file in your sandbox with `Write` first, so newlines and
quotes survive the shell:

```bash
clawmeets sop create \
  --title "Weekly restock review" \
  --body-file sop.md \
  --agent <agent-that-runs-it>
```

- `--title` is how the card reads in the rail — short and verb-first.
- `--agent` is the default recipient (short or full name, resolved in the
  owner's namespace). Usually that is you, if you are the one who runs it.
  Omit it and the SOP goes to the owner's assistant when fired.

The command prints the stored SOP as JSON. Tell the user its title, who it
goes to, and the blanks it will ask for.
