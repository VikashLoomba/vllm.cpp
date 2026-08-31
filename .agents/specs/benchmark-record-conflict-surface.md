# `.agents/benchmark-record.md` is a lock, and `merge=union` cannot unlock it

Issue: [#1373](https://github.com/mudler/vllm.cpp/issues/1373).
Row: `ENG-RECORD-CONFLICT-SURFACES`.
Measured against `origin/main` `9fb40279d` on 2026-08-31.

[#1373](https://github.com/mudler/vllm.cpp/issues/1373) asks for one line of
`.gitattributes`: give `.agents/benchmark-record.md` the `merge=union` attribute
that `.agents/issue-index.md` has, so that two pull requests appending an entry
stop conflicting.

**The answer is no, on three independent grounds, and this spec records why so
that nobody re-derives it.** The attribute the issue asks us to copy no longer
exists. The shape it belongs to was retired from `AGENTS.md` §Records two days
before this spec. And it would be unsound here even if both of those were false,
because **23.4% of the writes to this file are not appends**, and union against a
non-append edit silently produces a file neither side wrote.

## Scope

**In scope.** The soundness question for `merge=union` on
`.agents/benchmark-record.md`; the write-pattern measurement that answers it; a
comparison of the four available shapes against `AGENTS.md` §Records; a
recommendation; and two documentation repairs that this investigation found and
that nothing else owns.

**Out of scope, deliberately.** `.agents/parity-ledger.md`, which
`scripts/check-pr-size.py:78-83` classifies in the same `APPEND_ONLY_FILES` set
and which was not measured here. Any rewrite, reflow, compaction or re-ordering
of the 396 existing entries: the file is a forensic archive and its content is
evidence. Any change to `docs/BENCHMARKS.md`, which is a keyed table and a
different surface with a different defect (#460).

**Not a correctness change.** No product source, kernel, gate semantic, or
measured number moves.

## The issue's premise is falsified by the tree

`#1373` cites `.gitattributes:7`. At `9fb40279d` the whole file is three lines:

```text
# Vendored Triton AOT artifacts are generated code (embedded cubins): collapse
# in diffs/review and exclude from language stats. Regen: scripts/regen-triton-aot.sh
src/vt/cuda/triton_aot_vendored/** linguist-generated=true
```

There is no line 7 and no `merge=union` anywhere in the tree. Commit
`7dc2ef1ea` (2026-08-29, `feat(ENG-RECORD-CONFLICT-SURFACES): W6`) deleted both
the attribute and its subject:

```text
-# The issue index is append-only, so two branches appending a row merge
-# cleanly instead of conflicting. See .agents/specs/issue-intake.md
-.agents/issue-index.md merge=union
```

`.agents/issue-index.md` now lives in `.agents/completed/`. The same commit
removed the shape from `AGENTS.md` §Records, which today reads: *"An append-only
file with `merge=union` is NOT one of them."* GitHub does not run
`.gitattributes` merge drivers, so the driver resolves the collision on the
author's machine and the forge conflicts anyway (#883).

So the requested remedy is barred by policy and its factual premise is gone.
Per §Records — *"An issue the tree falsifies closes with that evidence"* — the
remedy half of #1373 is closable on this spec. **The complaint half is not.** The
file is still a shared surface that concurrent pull requests write, and that part
of the issue is answered below on its own evidence rather than dismissed.

## The soundness question, answered on the writes themselves

`merge=union` is sound only while every write is genuinely an append. It is not
a merge strategy that understands the file; it concatenates both sides' version
of a conflicting region. Against a non-append edit it emits both.

### Method

Every non-merge commit touching `.agents/benchmark-record.md` was classified
from `git show --unified=0` against the file's line count at the commit's parent.
A commit is `APPEND` iff every hunk's old-side start is at or past the parent's
last line **and** the diff deletes zero lines. Everything else is `NON_APPEND`.
222 commits carry a diff for the file.

### Result

| Class | commits | share |
|---|---|---|
| pure tail append | 169 | 76.1% |
| **non-append, with deletions** | **18** | **8.1%** |
| **non-append, mid-file insert** | **17** | **7.7%** |
| **non-append, head prepend at line 21** | **17** | **7.7%** |
| initial create | 1 | 0.5% |
| **all non-append** | **52** | **23.4%** |

**The rate is rising, not decaying:**

| window | non-append | share |
|---|---|---|
| last 25 commits | 12 / 25 | **48.0%** |
| last 50 commits | 18 / 50 | 36.0% |
| last 100 commits | 23 / 100 | 23.0% |
| all 222 commits | 52 / 222 | 23.4% |

The newest write to the file, `b426de5ac` (2026-08-30), is itself a non-append:
two hunks, 165 insertions and **6 deletions**, one hunk at `-28726` against a
29,129-line parent. This is not a historical artefact that a convention could
retire. It is what the file's writers do now, and they do it more than they used
to.

This reproduces the observation that prompted this investigation. A cumulative
merge of `origin/main` showing a header hunk at `@@ -19` plus two entry hunks
deep in the file plus an append is exactly the shape of several commits in the
`NON_APPEND` classes above — `6756f9131` (4 hunks at 21, 289, 474, 26503),
`1db7e59cf` (6 hunks, 11 deletions), `438305e15` (2 hunks, 7 deletions at 24325
and 24336). No single commit carries that exact hunk triple, because the observed
diff was cumulative over a range rather than one commit; the class it belongs to
is confirmed 52 times over.

### Two demonstrations that union corrupts this file

Run in throwaway repositories under the scratchpad, with
`merge.union.driver` configured exactly as `.gitattributes` would invoke it. No
build, no product code.

**Demo 1 — two concurrent head-prepends, the file's own second convention.**
Both branches insert a new entry directly under the `# Benchmarks` heading,
which is what 17 commits in the table above do, at line 21 every time.

```text
# Benchmarks

## Entry NEW-2 (2026-08-06)
## Entry NEW-1 (2026-08-05)
Placement. Newest-first. This sits above Entry A.

## Entry A (2026-08-01)
FA_USABLE=0. Unresolved.
```

`git merge` exited **0** with no conflict marker. `## Entry NEW-2` is now a
**heading with no body**: its evidence was silently dropped, because the two
entries' body lines were identical and the union collapsed them. A forensic
record acquired an entry that claims a measurement and contains none.

**Demo 2 — a retraction merged against a concurrent edit to the same entry.**
This is the `NON_APPEND` deletions class, 18 commits, and retraction is precisely
what those commits do.

```text
## Entry A
SPEED: 1.50x vs vLLM.
Gate: PASS (reconfirmed 08-30).
SPEED: RETRACTED, the arm was misbuilt.
Gate: FAILING.
```

Again exit **0**, no conflict. The retracted claim and its retraction now stand
in the same entry, as do `PASS` and `FAILING` for the same gate. A reader cannot
tell which is live, and neither can a checker.

**Verdict: `merge=union` is UNSOUND for `.agents/benchmark-record.md`.** It
converts a visible conflict, which a human resolves, into a silent corruption of
the one record whose entire purpose is to be trustworthy about superseded
numbers. Adding it would be strictly worse than the problem it solves, and it
would not even fix that problem, because the forge ignores the driver.

## Who writes this file, and how

**No generator exists.** `.agents/benchmark-record.md:10-11` tells the reader:

> `scripts/roll-benchmark-record.py` moves any narrative section that
> accumulates in the scoreboard down into this file.

That script was added by `8a0744ae0` and **deleted by `1db7e59cf`** (2026-08-22,
`docs: retire shared status and split benchmark details (#1714)`). It is not in
the tree. Every write to this file is hand-authored by an agent, and nothing
mechanical constrains a write to be an append. This is repair **R1** below.

**No gate requires or budgets a write.** `scripts/check-pr-size.py:78-83` places
the path in `APPEND_ONLY_FILES`, but that is a classification feeding
`classify()` at `:448`, and the per-class line budgets were retired on
2026-08-10, so the classification now carries no obligation and no limit. The
only instruction to write the file is prose:
`.agents/benchmarking.md:221` — *"Accepted and pending results go in
`benchmark-record.md`"*. The append-only property is a convention, gated by
nothing, which is what the dflash2 spec already recorded at
`.agents/specs/dflash2-spec-decode.md:3033-3036` when it declined to annotate an
entry in place.

**The file contradicts itself about placement, three ways.** Line 3 calls it an
"Append-only forensic record". 169 commits append at EOF. 17 commits prepend at
line 21, and the entry now sitting at line 22 opens with *"**Placement.**
Newest-first. This sits above `QUANT-QWEN38-27B-GGUF-ARM W3`"*. So the header,
the entry prose, and the actual writes each assert a different convention. This
is repair **R2**, and it matters for more than tidiness — see the anchors below.

## Structure: is the header separable from the entries?

Yes, and cleanly. Lines 1-18 are the header: purpose, the pointer to
`docs/BENCHMARKS.md`, and the provenance note for the 2026-08-04 migration. Line
18 is a `---` rule, line 20 is `# Benchmarks`, and the 396 entries are `## `
headings from line 22 to EOF.

An append-only rule could therefore be enforced for the entry region while the
header stayed deliberately editable. **That does not rescue `merge=union`,**
because the attribute binds per file and cannot be scoped to a line range, and
because 35 of the 52 non-append commits edit the *entry* region rather than the
header. It is recorded because it is the one structural fact that would matter
if a future row ever wants an append-only gate on the entries, which this spec
does not propose.

## The options, compared

### Option 1 — add `merge=union`

**REJECTED, three times over.** Unsound (23.4% non-append, and the two demos
above). Ineffective (GitHub does not run merge drivers; #883). Barred (§Records
names the shape as not admissible, since `7dc2ef1ea`). Nothing further is owed to
this option.

### Option 2 — split per row, read with a glob

This is the shape §Records prefers, and it removes the lock rather than
mitigating it. **REJECTED as specified, and deferred in its workable variant.**

*As literally specified — one file per row — it is not mechanically possible.*
The 396 entry headings yield **313 distinct leading tokens**, and many are not
row IDs at all: `CPU` (10), `GEMM`, `CUDA`, `OPEN`, `PR`, `BENCHMARKS`, `MLX`,
`TT`, `Q38-27B-BF16`. Only `SPEC-DSPARK` (36) and a handful of others cluster.
The owning row simply cannot be derived from the heading for most entries, so the
migration would require a human to classify 396 forensic entries, and a
misclassification silently files evidence where no one will look for it.

*The workable variant is one file per entry* —
`.agents/benchmark-records/<date>-<slug>.md`, chronology derivable from the
filename, index derived at read time. That genuinely removes the lock: two
concurrent entries never touch the same path. Its costs are concrete and were
measured:

- **252 citing lines across 117 files** point at the file. All become wrong.
- **35 of those name a line number**, including two in product source:
  `src/vt/cuda/cuda_mamba2_ssd.cuh:54` (`benchmark-record.md:532`) and
  `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:290`
  (`benchmark-record.md:10722`).
- **All 201 relative links *inside* the file break.** They are `.agents/`-relative
  (`](specs/multimodal-speed.md)` 12 times, `](parity-ledger.md)` 5,
  `](completed/state-events/)` 4, and so on). Moving the content one directory
  deeper re-bases every one of them, so the migration must either rewrite 201
  links — which violates the byte-for-byte preservation this archive exists to
  provide — or extend `link_bases()` in `scripts/check-agent-record.py:1078-1107`,
  which today already carries a special second base for this exact file and
  documents at length (#460) why that is weaker than followability.

The cost is real and the benefit is proportional to the conflict rate, which is
measured below and is low. **Filed as owed debt with a trigger, not done now.**

### Option 3 — derive it at read time

**REJECTED, and not close.** There is no source to derive from. The entries are
hand-authored narrative: refuted hypotheses, profiler tables, reasoning about why
a lever was abandoned, and the reasons a default has its value. The only
generator this file ever had was deleted (above). Derivation requires an
authority that already holds the facts in structured form; for a prose archive of
human judgement, none exists and none can be built.

### Option 4 — leave the file, record why

**RECOMMENDED**, with repairs R1 and R2, and with option 2's workable variant
filed as owed.

The conflict rate does not justify migrating a 24,000-line forensic archive
today, and the comparison is against a surface this repository has already
migrated for exactly this reason:

| surface | commits writing it | open PRs conflicting |
|---|---|---|
| `.agents/issue-index.md` (retired `7dc2ef1ea`) | **115 of last 200 (57.5%)** | 16 of 21, 4 of them on that file alone |
| `.agents/benchmark-record.md` (this file) | **91 of last 1000 (9.1%)**, 2 of last 200 | 2 of 29, per the measurement below |

The 1000-commit window spans 2026-08-12 to 2026-08-31, 19 days; the 200-commit
window spans only the last two days, which is why the two figures differ. Either
way the file is written at roughly one-sixth the rate that justified retiring the
index.

That is also not a new finding. `.agents/specs/retire-shared-record-surfaces.md`
measured this file on 2026-08-11 at **2 of 29 conflicting open pull requests**,
the same tier as `docs/FEATURES.md`, and placed it under *"Out of scope,
deliberately… measured below as not implicated; removing them would be scope this
evidence does not support."* Nothing in the present measurement overturns that.
The write rate is still an order of magnitude below the surface that did get
retired.

**The honest counter-argument, recorded rather than buried.** A conflicted pull
request carries zero check-runs, because GitHub never schedules CI for one, so it
reads as unverified rather than red (#2248). Each collision therefore costs more
than its frequency suggests. And the file grows by roughly 500 lines a day, so
the rate is not static. This is why option 2's variant is filed as owed with a
trigger rather than refused.

## Recommendation

1. **Do not add `merge=union`.** Close the remedy half of #1373 citing this spec.
2. **Keep the file as one file.** Record the exception in the landing commit
   message, per §"Changing the rules or a checker" — the project has no waiver
   registry, so the argument lives with the diff.
3. **R1 — correct the header.** Lines 10-11 name `scripts/roll-benchmark-record.py`
   as if it maintained the file. It was deleted at `1db7e59cf`. A reader currently
   believes a generator owns this file and that their hand-edit is unusual; the
   opposite is true.
4. **R2 — standardise on tail-append, and say so in the header.** Not for
   tidiness: **head-prepends are what break the line-number citations.** An
   insert at line 21 shifts every one of the 35 recorded anchors, and it has
   happened 17 times. Tail-append leaves every existing line number stable, which
   is the only reason those anchors can survive at all. This also halves nothing
   and costs nothing — 76% of writes already do it.
5. **File as owed** (`## Owed` below): the per-entry split with its trigger, and
   the conversion of line-number citations to heading anchors.

## Owed

- **O1 — the per-entry split.** `.agents/benchmark-records/<date>-<slug>.md`,
  read with a glob, content preserved byte-for-byte, `link_bases()` extended
  rather than links rewritten. **Trigger:** the file is written by more than 25%
  of the last 200 non-merge commits, or more than 4 open pull requests report
  `CONFLICTING` with it as a path. Below that the migration costs more than it
  returns. Needs its own issue.
- **O2 — the 35 line-number citations.** They are already unreliable. Checked at
  `9fb40279d`: `benchmark-record.md:532`, cited from
  `src/vt/cuda/cuda_mamba2_ssd.cuh:54` as recording *"a pre-rounded `v²`
  differing by <= 1 ulp… and flipping a near-tie"*, resolves to *"A container
  cannot drop the host page cache, so this run took an `rc hold` on"*. Four more
  spot-checked anchors (`:4654`, `:10722`, `:17209`, `:19021`, `:24510`) likewise
  land mid-sentence on unrelated prose. They should cite heading text, which
  survives every insert. Needs its own issue.
- **O3 — `scripts/check-pr-size.py:93`** still lists `.agents/issue-index.md` in
  `PROJECT_RECORD_FILES`, a path deleted by `7dc2ef1ea`. Harmless today, because
  classification carries no budget, but it is a stale reference to a retired
  surface. Needs its own issue.

## Risks

- **R-1: the recommendation is a decision not to act, so it decays.** If the
  write rate climbs, option 4 silently becomes wrong and nothing fires. Mitigated
  by O1's explicit numeric trigger, which is checkable with the two commands in
  §Gates rather than by judgement.
- **R-2: R2 standardises a convention that is not gated,** so it can drift back.
  Accepted deliberately. An append-only gate on the entry region is possible
  (§Structure) but would fail every legitimate retraction, which is 8.1% of
  writes, and this spec's whole finding is that retractions are legitimate here.
  A gate that fires on ordinary work is the defect.
- **R-3: the conflict-rate denominators are windows over `main`,** and `main`
  moves. Both numbers are pinned to `9fb40279d` in §Evidence so a re-measurement
  is comparable rather than merely different.
- **R-4: closing the remedy half of #1373 while its complaint half survives**
  risks the complaint being lost. Mitigated by O1, which carries the complaint
  forward with a trigger and an owner.

## Tests

This spec changes no executable behaviour, so it ports no test and adds no gate.
The two claims it makes that could be wrong are both executable, and both were
run:

1. **The union-corruption claim** is demonstrated by the two throwaway
   repositories in §"Two demonstrations". Each is reproducible in under ten
   seconds with no build: `git init`, set `merge.union.driver`, write
   `rec.md merge=union` to `.gitattributes`, branch, edit both sides, merge.
   Both exit 0 and both produce the corrupt file shown.
2. **The write-classification claim** is reproducible from `git` alone with the
   commands in §Gates. It reads only committed history, so it is stable against
   anything in the working tree.

The mutation a reviewer should perform: drop the deletion term from the `APPEND`
predicate, so that a commit is judged by hunk position alone. **The non-append
total falls from 52 to 50, not from 52 to 34.** That result was measured, and it
corrects the number this section first asserted from arithmetic.

The two-commit delta is the finding, not a weakness in the predicate. It says
that **16 of the 18 deletion-bearing commits are already non-tail by position**:
when a writer edits this file destructively, they almost always do it in the
middle of the archive, because that is where the entry being corrected lives.
Only `7c84d3710` and `c3fb0d247` delete lines while writing at the tail, one
line each. So the
23.4% figure does not depend on the deletion term at all — position alone
recovers 50 of the 52 — and a reviewer who suspects the deletion count is doing
the work can delete it and still reach the same verdict.

## Gates

Both run offline, in seconds, with no build and no GPU.

```sh
# G1 — the write-pattern classification, at the pinned base.
git log --no-merges --format=%H -- .agents/benchmark-record.md | wc -l   # 222

# G2 — the conflict-rate denominators.
git log --no-merges -1000 --name-only --format=%H \
  | grep -c '^\.agents/benchmark-record\.md$'                            # 91

# G3 — the premise check: no merge driver is configured anywhere.
# NOTE: written as an absence test. A bare `grep -c` prints 0 and EXITS 1 here,
# and rc 1 on a passing gate is how a reader mistakes this for a failure.
! grep -q 'merge=union' .gitattributes                                   # rc 0

# G4 — the generator named in the header does not exist.
test ! -e scripts/roll-benchmark-record.py                               # rc 0
```

`scripts/agent-preflight.sh` is the full gate for a records-only change and was
run on the head that carries this spec; its result is in §Evidence.

## Evidence

Measured on `origin/main` `9fb40279d79f327cfa5e64358b6aa2e822b782b5`,
2026-08-31, authoring host, in the worktree
`.claude/worktrees/agent-ab6244bdf23ad2bda`. No GPU was used, no lease was
taken, and nothing was built — the host had **7.0G free on `/` (99% used)** and
a load average of 53.17, which is why every check in §Gates reads committed
history instead of compiling anything.

| Claim | Source |
|---|---|
| `merge=union` is absent from the tree | `.gitattributes`, 3 lines, at `9fb40279d` |
| the attribute and its subject were deleted together | `git show 7dc2ef1ea -- .gitattributes` |
| the shape is no longer admissible | `AGENTS.md` §Records, since `7dc2ef1ea` |
| the forge ignores merge drivers | #883, quoted in `7dc2ef1ea`'s body |
| 52 of 222 writes are non-append | classifier over `git show --unified=0`, §Method |
| the newest write is non-append | `b426de5ac`, 2 hunks, +165 −6, parent 29,129 lines |
| union drops an entry body | scratch repo demo 1, `git merge` rc 0 |
| union keeps a retracted claim and its retraction | scratch repo demo 2, `git merge` rc 0 |
| the generator was deleted | `git log --diff-filter=AD -- scripts/roll-benchmark-record.py` → A `8a0744ae0`, D `1db7e59cf` |
| no gate requires or budgets a write | `scripts/check-pr-size.py:78-83,448`; budgets retired 2026-08-10 |
| the only instruction is prose | `.agents/benchmarking.md:221` |
| append-only is a convention, not a gate | `.agents/specs/dflash2-spec-decode.md:3033-3036` |
| 396 entries, 313 distinct heading prefixes | `grep '^## '` over the file |
| 252 citing lines across 117 files | `grep -r` excluding the file itself |
| 35 citations name a line number | `grep -rnoE 'benchmark-record\.md:[0-9]+'` |
| `:532` no longer resolves to what cites it | `sed -n 532p`, vs `cuda_mamba2_ssd.cuh:50-58` |
| 201 internal links are `.agents/`-relative | `grep -oE '\]\([^)h][^)]*\)'` over the file |
| this file was already measured not-implicated | `.agents/specs/retire-shared-record-surfaces.md`, §Scope and §baseline |
| the retired index wrote 115 of 200 commits | `7dc2ef1ea` body, measured at `e541be98` |

## Stop conditions

- **Stop and return `NEEDS_DECISION`** if the developer or operator wants the
  per-entry split (O1) executed now rather than filed. That is a defensible
  reading of §Records and this spec does not foreclose it; it is deferred on
  cost, not on principle, and the trigger in O1 is a number somebody chose.
- **Stop** before rewriting, reflowing, re-ordering or compacting any existing
  entry. The archive's content is evidence. R1 and R2 touch only the header.
- **Stop** before adding an append-only checker over the entry region. It would
  fail the 18 legitimate retractions this spec identifies.
- **Stop** if a re-measurement at a later base puts the write rate above O1's
  trigger. At that point option 4 has expired and O1 is the work.
