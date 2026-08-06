# Session onboarding — ask, don't assume

User-directed 2026-08-06. Status: **accepted design, not yet implemented.**
This document is the contract; the checker and prose changes named in
§ Enforcement are the work it implies.

Subsystem **A** of two. Subsystem B — the orchestration harness an operator
follows to run a row through subagents with an independent review — is a
separate spec and lands after this one. They meet at exactly one point: this
interview's "long campaign" answer hands off into B.

## Scope

What happens in the first minute of an agent session: which role this session
holds, which row it is taking, whether it may write at all, and where the
machine-specific values come from.

Out of scope: what work to do (the roadmap), how to do it (`AGENTS.md` and
`.agents/directives.md`), and the orchestration loop (subsystem B).
`.agents/developer-preferences.md` generation is deliberately excluded — it is
a per-developer profile, not a per-session decision.

## Our baseline — the obligation exists, nothing triggers it

Every piece of this already exists as prose or tooling. None of it fires.

- `.agents/specs/operator-helper-protocol.md` already says the session "ASKS
  before doing anything else", and explains why derivation cannot work: several
  sessions launch from the same checkout, so nothing distinguishes them until a
  role has already been taken.
- `scripts/agent-role.py` already does the hard part — `claim`, materialize
  (operator lock in the git COMMON dir, helper = worktree), `heartbeat`,
  `release`. It exits 3 when undeclared.
- `scripts/agent-preflight.sh` already has `--require-role`, and it is
  **opt-in**. `.agents/NOW.md` records it as "still opt-in".
- `AGENTS.md` already says that when `.env` is missing it "is asked, never
  inferred".
- There is **no `.claude/settings.json` and no hook of any kind**. Nothing runs
  at session start.

So the protocol depends entirely on an agent reading a 328-line index and
choosing to comply. The 2026-08-04 incident is what that costs: two sessions
pushed to `main` within minutes, neither claimed anything, and a three-way merge
silently produced a VARIANT of the other session's binding numbers.

**This spec adds no new concepts. It makes the existing obligation fire, and
makes it pleasant when it does.**

## Design

### The trigger is preflight, not a hook

`--require-role` becomes the **default**. `--no-require-role` is the escape for
scripted and CI use.

A hook was considered and rejected: it would be Claude-Code-specific, and this
protocol is harness-neutral by design (`AGENTS.md` is read by other harnesses
too). Preflight is the harness-neutral trigger the protocol already mandates,
and the push chain (`gate && git push`) means a session that skips preflight
also never lands anything. The role is therefore demanded at the latest before
the first durable effect.

**A script cannot ask a question.** No harness-neutral mechanism exists for a
shell script to run an interactive prompt, and a hook cannot ask either — hooks
inject text, they do not converse. So the split is fixed:

| Job | Owner |
|---|---|
| Detect and report what is unresolved | `scripts/agent-onboard.py --probe` |
| Ask the human | the agent, in its own UI |
| Make the answer a fact | `scripts/agent-role.py claim`, `--env-set` |

The canonical interview text therefore lives in `.agents/workflow.md`, where an
agent actually reads it, not in a script that cannot perform it.

### Ask about the work, not the vocabulary

The interview asks what the session is here to do. The role follows from the
answer, and the answer is something the developer already knows.

| "What are you here to do?" | Role | What it means |
|---|---|---|
| A long or multi-step campaign — several changes, a benchmark grid, a whole row block | **operator** | Owns `main` and the GPU. Merges PRs first. Drives feature work through sub-agents rather than writing it. One at a time, repo-wide. |
| One scoped change — a fix, a port, a single row | **helper** | Isolated worktree on `row/<ROW-ID>`, draft PR opened at the START. That PR **is** the claim. Never touches `main`. |
| Just looking — reading code, answering a question | **read-only** | No lock, no worktree, no claim. |

`read-only` is **not a third role.** It is a declared *absence* of claim,
recorded so the gate can tell "decided not to claim" from "never asked". The
two-role model in
[operator-helper-protocol.md](operator-helper-protocol.md) is unchanged.

It exists because the alternative punishes the most common session. Forcing a
question-answering session to claim `operator` would take the repo-wide lock and
block a real one; forcing `helper` would create a throwaway worktree. Faced with
either, a developer reaches for `--no-require-role`, and the gate erodes to
nothing. A cheap honest answer keeps the gate credible.

A `read-only` session **passes preflight and is refused by every write path**:
commit, push, and any matrix or record edit. Escalating is one command — it is a
declaration, not a sentence.

### Mode: interactive by default, headless when asked

`~/_git/skills/spec-driven-development` forbids asking anything, because it runs
unattended overnight and "a question is a hang, not a pause". This spec is built
on asking. Both are right in their context, so the mode is declared explicitly,
in the same breath as the role — one question, not two.

| Mode | When | Behaviour on ambiguity |
|---|---|---|
| **interactive** (default) | a human is present | ask |
| **headless** | the human explicitly says the session is unattended | decide, record the decision in `.agents/state.md`, never block, never merge, park what will not go green |

Headless is never inferred — not from the hour, not from silence, not from a
long-running task. It is stated, exactly like the role.

### `.env` is asked just in time, never up front

A missing `.env` does **not** block the role claim, and the developer is never
walked through a template for values the session may never use.

The agent asks for a value at the moment a gate actually needs it — the oracle
path before an oracle comparison, the gate host before a device run,
`${GPU_LOCK}` before touching the GPU. Anything unanswered is recorded as
empty, and `AGENTS.md` already defines what empty means: the gates that need it
stay `PENDING`. That is an honest state, not a failure.

`scripts/agent-onboard.py --env-set KEY=VALUE` performs the write, creating
`.env` from `.env.example` on first use. The agent asks; the script writes.
Never substitute another developer's paths, and never infer a value from a
username, a filesystem path or a machine identity.

## Enforcement

**`scripts/agent-onboard.py --probe`** (new, harness-neutral, read-only) reports
machine-readable state and asks nothing:

- role: `operator` | `helper <ROW-ID>` | `read-only` | `undeclared`
- mode: `interactive` | `headless`
- `.env`: `present` | `missing` | `incomplete: KEY,KEY`
- helper queue: the `READY` row IDs, from the existing `ready-for-helper.py`

**`scripts/agent-preflight.sh`**: `--require-role` on by default;
`--no-require-role` to opt out. On an undeclared role it fails with the
interview to run and the exact claim commands — not a bare error code. A gate
that tells you what to do next is the difference between a protocol people
follow and one they route around.

**`scripts/agent-role.py`** gains `claim read-only` and a `--headless` flag on
`claim`, so mode and role are one materialized fact.

**Write paths refuse a `read-only` session.** Preflight `--staged` fails, and the
role check runs before any record edit.

**Prose and gate move together.** `AGENTS.md` T0's role bullet, the
`.agents/workflow.md` session protocol (which carries the canonical interview),
and `operator-helper-protocol.md` are updated in the SAME change as the checker.
`scripts/check-protocol-consistency.py` exists precisely because an obligation
was once migrated in `AGENTS.md` and the checker but not in the manual, which
went on instructing agents to do the thing the migration had removed. Extend it
to assert the interview table appears in `workflow.md`.

**Never weaken a checker to make a transition pass.** If preflight is red
because the role is undeclared, the answer is to declare it.

## Work breakdown

Each item is independently landable.

| # | Work |
|---|---|
| 1 | `scripts/agent-onboard.py --probe` + mutation tests; reports state, writes nothing |
| 2 | `agent-role.py claim read-only` and `--headless`; the mode becomes a materialized fact |
| 3 | `--require-role` default-on, `--no-require-role` escape, actionable failure text; write paths refuse `read-only` |
| 4 | `--env-set KEY=VALUE`, creating `.env` from `.env.example` on first use |
| 5 | Prose: `AGENTS.md` T0, `workflow.md` interview table, `operator-helper-protocol.md` — in the same change as the `check-protocol-consistency.py` extension |

## Risks and decisions

**Accepted: preflight is only as binding as the habit of running it.** The
mitigation is structural rather than aspirational — `AGENTS.md` T0 already
requires it at session start and before committing, and chains the push to it,
so the unclaimed session cannot land anything. A hook would close the remaining
gap for one harness at the cost of neutrality; rejected on that basis.

**Accepted: `read-only` can be over-used.** Someone can declare `read-only` and
then find real work. That is fine and expected — escalation is one command. The
failure it prevents (silent unclaimed writing) is far worse than the one it
allows (an extra claim command mid-session).

**Rejected: derive the role from the environment.** Already tried and already
recorded as an error in `operator-helper-protocol.md`. Several sessions launch
from one checkout; a helper only becomes environmentally distinguishable *after*
it has taken a worktree, so derivation is circular.

**Rejected: full `.env` walkthrough at session start.** It asks for values most
sessions never use, and front-loads friction onto the first minute — the exact
moment a developer is least willing to spend it.

**Rejected: generating `.agents/developer-preferences.md` in this interview.**
It is a per-developer profile, not a per-session decision, and folding it in
would make the first prompt of every session long enough to be skipped.

**Open, deferred to subsystem B:** an operator that declares `headless` still
has no written orchestration loop to run. That is B's subject, and B is where
the independent reviewer subagent — the single highest-value missing piece —
gets specified.
