# Current-main prerequisite repairs — 2026-08-08

## Scope

Repair the three independent current-`main` regressions that prevent otherwise
clean row PRs from passing preflight:

1. make the new landed-detached role-discipline unit test hermetic without
   changing `scripts/check-role-discipline.py` or its enforcement; and
2. reconcile `MODEL-EMBED-llama-llama-for-causal-lm` from `ACTIVE` to
   `PARTIAL`, because only `LlamaModel` of the eight listed memberships landed
   and the real-checkpoint vLLM cosine gate remains pending; and
3. make landed range validation inspect first-parent arrival events, so a
   recognized row-PR merge supplies the review evidence for its internal
   commits without blessing direct pushes or unrecognized merges.

Affected policy IDs: `POL-PR-REQUIRED`, `POL-CHECKER-CHANGE`,
`POL-PATH-CLASSIFICATION`, `POL-PR-SIZE`, `POL-ROW-SYNC`, `POL-KEYED-MERGE`,
`POL-DOC-STATUS`, `POL-DOC-BENCHMARKS`, `POL-DOC-FEATURES`,
`POL-NOW-COUPLING`, `POL-PREFLIGHT`, and `POL-PR-DISPOSITION`.
The landed-range change follows `POL-CHECKER-CHANGE`: policy and procedure move
together with RED-before and mutation evidence. The separate hermetic-fixture
repair does not itself alter production semantics.

## Verified gaps and root causes

### Role-discipline test

RED on `ba8d867c` after the helper reservation commit:

```text
FAIL: test_landed_detached_commit_remains_strict_without_pending_evidence
AssertionError: 0 != 1
```

The test patched `has_reached_main()` but evaluated ambient `HEAD`.
`main()` first calls `inspect(HEAD)`; a merge, empty reservation, or
integration-only commit has no governed-path violation, so the failure
classification branch is never reached. The test therefore measures the
checkout's current tip rather than its stated strictness guarantee. Make the
input hermetic by stubbing `inspect()` to return one violation while retaining
`has_reached_main() == true`; do not alter the checker.

### Embedding lifecycle row

PR #158 CI job `93159545283` reports:

```text
MODEL-EMBED-llama-llama-for-causal-lm | ACTIVE | ABANDONED
no branch, no commit on main mentioning the row ID
```

The feature did land on main at `57ed063e`, but its commit and integration
branch use the umbrella `EMBEDDINGS-ONE-SURFACE` name rather than the exact
matrix ID. More importantly, the row itself lists eight memberships while the
landing implements only `LlamaModel`; seven memberships and the real oracle
cosine gate remain. `PARTIAL` is the truthful lifecycle state. This repairs the
record rather than weakening `audit-live-rows.py`.

### Multi-commit reviewed PR range

RED on the landed #167 range `e6dbd5ce..ba8d867c`: CI's detached checkout
classifies every PR-internal commit independently. Those commits do not carry
the enclosing merge's `row/*` / PR evidence, so `check-role-discipline.py`
reports or fails them even though the recognized merge is the reviewed arrival
event. The range must walk `git rev-list --first-parent --reverse`: a reviewed
merge then represents its full diff, while an earlier direct first-parent
commit and an unrecognized merge remain individually visible and strict.

### Tracked manifesto path

Current main tracks root `MANIFESTO.md`, but the fail-closed PR path classifier
did not assign it an explicit class; its own tracked-tree mutation test was RED.
Classify that exact path as `public_document`. No regex, budget, exemption, or
other path changes. This checker delta is governed by both
`POL-PATH-CLASSIFICATION` and `POL-CHECKER-CHANGE`; the existing synchronized
registry and workflow wording already requires explicit documentation-path
classification, so their policy meaning stays unchanged.

## Changes

- Add a hermetic violation fixture to the existing role-discipline test.
- Validate landed ranges as first-parent arrival events and synchronize
  `POL-PR-REQUIRED` in the registry and workflow procedure.
- Change the embedding row `ACTIVE` to `PARTIAL`, update model rollups and the
  engaged-architecture summary, and point evidence at main `57ed063e`.
- Remove that row from the exact ACTIVE-scoped runnable-command baseline while
  preserving its commands in the feature spec as historical gate evidence.
- Reconcile the owning roadmap/coordination entries and keyed public status,
  feature, and benchmark rows. Append this checkpoint and refresh `NOW.md`.
- Do not change product code, checker code, benchmark numbers, or performance
  claims.

## Gates and mutation evidence

Focused:

```sh
python3 tests/scripts/test_agent_role.py
python3 scripts/check-agent-record.py
python3 scripts/audit-live-rows.py --check
python3 scripts/check-model-checklist.py
python3 scripts/check-public-doc-tables.py
python3 scripts/check-doc-checkpoint.py --base origin/main --head HEAD
```

Full: `scripts/agent-preflight.sh --staged`, then `scripts/agent-preflight.sh`.

Targeted mutations for review:

- remove the `inspect()` violation fixture: the landed-detached test must RED
  on a non-feature ambient `HEAD`;
- remove `--first-parent`: the recognized two-commit row merge must RED;
- inspect only merge tips: the direct commit before a later reviewed merge must
  incorrectly pass, and the dedicated test must RED;
- accept every merge or remove second-parent message evidence: respectively the
  unrecognized-merge and synthetic-merge tests must RED;
- restore the model row to `ACTIVE`: a clean CI-style checkout with no exact
  row ref must make `audit-live-rows.py --check` RED.

## Stop conditions

Stop rather than changing semantics if strict landed-commit enforcement cannot
remain intact, if another row owns the lifecycle correction, or if current
main changes any keyed surface during reconciliation; in the latter case take
the new main version wholesale and reapply only this scoped correction.
