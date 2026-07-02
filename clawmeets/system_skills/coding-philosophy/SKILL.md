---
name: coding-philosophy
description: >
  The stance to hold whenever you write, restructure, or review code in a
  git-tracked project (FILES & STATE block shows a `Git repo` line) — pairs with
  `coding-plan`. Optimize for comprehensibility above all else; the biggest
  question for any change is finding the right semantic home for each piece of
  logic, and when no home fits, extending the system rather than hacking. Read
  it before deciding WHERE new logic belongs, and revisit it when a change
  starts to feel awkward — that awkwardness is usually a misplaced-ownership
  signal. Skip for non-code work.
---

# Coding Philosophy

There is one thing to optimize for: **comprehensibility** — code a brand-new
developer can read, understand, and safely modify quickly and reliably. Speed
is the only other axis worth trading against it, and on modern hardware you
rarely need to. Everything below is in service of comprehensibility.

## The two axes

Code quality is a **bivariate** optimization — speed and comprehensibility — not
a juggling act across a dozen goals. Premature optimization is a trap; hardware
is fast. So comprehensibility wins by default.

Comprehensibility is also **foundational**: code needs to be comprehensible
before it can be modular, testable, reusable, or reliable. Spaghetti can't be
any of those things. Get comprehensibility right and the rest become reachable;
get it wrong and no amount of test coverage or cleverness saves you.

## The governing question: find the right home for every piece of logic

For any change, before you type, answer this first: **what is the semantic owner
of this logic?** Every piece of behavior belongs on the module, class, or
concept whose responsibility it actually is. Place it there.

- Put a behavior on the thing that *owns the concept* — not on whatever object
  happens to be in scope at the call site, and not in a `utils` grab-bag.
- Name the home by its responsibility. If you can't name what a module/class is
  *for* in one phrase, its boundaries are wrong.
- The home should make the call site read like a plain statement of intent.

This single decision — right home vs. convenient home — is the largest lever on
whether the codebase stays comprehensible as it grows.

## When no home fits: extend the system, don't hack

Sometimes a piece of logic has no good home. The tempting move is to wedge it
into the nearest semantically-unfit module — a flag here, a special case there,
a helper bolted onto an object that doesn't really own the concept. **Don't.**
That compromise is the single biggest source of incomprehensibility, and it
compounds: each hack makes the next one look normal.

Instead, treat "no home fits" as a signal that the system is missing a concept.
Ask: *what concept could I introduce that would naturally own this logic — and,
in doing so, make the whole system more capable and more flexible?* Introduce
that concept (a new class, abstraction, module, or boundary), give the logic its
natural home, and the change strengthens the system instead of eroding it.

**Smell → response:**

| Smell | Likely meaning | Response |
|---|---|---|
| A helper "can't use" the obvious instance method | It's a *different* semantic wearing a familiar name | Tease the two semantics apart; give each its own owner |
| A `utils`/`helpers`/`misc` module keeps accreting unrelated functions | A concept is missing | Name the concept; promote the cluster into it |
| You're threading the same flag/param through many layers | A responsibility is in the wrong place | Move the decision to the layer that owns it |
| A `switch`/`if` on "kind" appears in several files | A type/strategy abstraction is missing | Introduce it; let each kind own its behavior |
| The change needs a comment to explain why it lives *here* | It probably doesn't | Find (or create) where it would need no explanation |

When in doubt, extending the system is cheaper than the long tail of confusion a
hack creates.

## Practices that keep code comprehensible

1. **Read from the reader's perspective.** Review your own diff as if you'd just
   joined the team. Revise until a stranger could follow it.
2. **Treat code smells as signals.** Duplicated code, long methods, large
   parameter lists, feature envy — each points at a deeper structural problem,
   not a cosmetic one. Fix the structure.
3. **Keep it DRY, with clear ownership.** Don't repeat logic — but de-duplication
   only works when each module has a clearly defined responsibility to own the
   shared piece. DRY without ownership just moves the mess.
4. **Hold consistent conventions.** Naming, structure, and patterns that match
   the surrounding code let readers infer intent. Match the code you're editing.
5. **Speak design patterns as shared vocabulary.** Patterns give the team a
   common language ("this is a factory", "that's an adapter") — use them to
   communicate intent, not to show off.
6. **Make dependencies explicit (dependency injection).** Inject dependencies
   rather than constructing them inline. It clarifies lifecycles and separates
   creation from business logic, so each piece reads in isolation.

## Structural conventions

- **Layered, acyclic dependencies.** Modules form a layered DAG: lower layers
  never import from higher ones, and there are no cycles. Import directly from
  the module that *owns* a definition rather than re-exporting through a
  convenient intermediary. A reader should be able to place any module in the
  layering at a glance.
- **Expose capabilities through multiple surfaces.** Pair every meaningful
  backend capability with three coordinated surfaces over the *same* logic: an
  HTTP API, a web UI built on that API, and a CLI. The CLI is what users script
  and automate against and what other agents' skills shell into; don't leave a
  capability reachable from only one entry point. The capability lives in one
  place; the surfaces are thin and consistent on top of it.

## The test

Your code should read like a story in which each line clearly tells the reader
what it does. The acceptance bar isn't "it works" — it's "a developer who has
never seen this could understand it and change it safely, quickly." If a part of
the diff would make that person pause, that's where the design needs another
pass: usually a piece of logic in the wrong home, or a concept the system is
still missing.
