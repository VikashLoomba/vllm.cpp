# `ENG-PREFLIGHT-COMPILES` — compile what a change can break, before the push

Issue: [#2401](https://github.com/mudler/vllm.cpp/issues/2401).

| Section | Content |
|---|---|
| Scope | IN: `scripts/check-tree-compiles.py`, its suite, its preflight and CI wiring, and one sentence in `AGENTS.md` §"Landing work". OUT: linking, running tests, every non-default configuration, every CI build job, `.githooks/pre-push`, and what preflight demands of a `read-only` session |
| Upstream chain | NO vLLM analogue. This is local protocol machinery, so the mirror rule does not apply and there is no upstream `file:line` to port. Governed by `AGENTS.md` §"Changing the rules or a checker", which requires a spec, a red-before test or mutation, and green-after evidence. The nearest in-tree precedents are `.agents/specs/gate-prepush-fail-loud.md` (a gate that presented as six checks while running three) and `.agents/specs/gate-preflight-skip-report.md` (the three-state `ok`/`FAIL`/`SKIP` protocol this checker plugs into) |
| Our baseline | `scripts/agent-preflight.sh` runs 30 record checkers and 60 Python suites and compiles nothing. `.githooks/pre-push` runs three of the same checkers and compiles nothing. `main` was pushed twice on 2026-08-31 in a state that does not build, green on every one of them: `5263ac31f` and `08fa2f5aa` (#2395) |
| Port map | Nothing is ported. `compile_commands.json` (`CMAKE_EXPORT_COMPILE_COMMANDS`, already `ON` at `CMakeLists.txt:38`) → the exact per-TU flags; `c++ -MM -MG` → the exact reverse-include map; `c++ -fsyntax-only` → the front-end-only compile. Preflight's existing `run()`/`skip()` pair → the `ok`/`FAIL`/`SKIP` reporting |
| Tests to port | Nothing to port; `tests/scripts/test_check_tree_compiles.py` is written against this tree. The historical red-before trees are `5263ac31f^` and `08fa2f5aa^`, which pass every existing gate and must fail this one |
| Gates | `python3 tests/scripts/test_check_tree_compiles.py`; `python3 scripts/check-tree-compiles.py --base origin/main`; `scripts/agent-preflight.sh` |
| Dependencies | Row IDs: none blocking. `GATE-PREPUSH-FAIL-LOUD` owns the hook's checker list and is untouched; `GATE-PREFLIGHT-SKIP-REPORT` owns the three-state protocol and is reused, not changed. Toolchain: `cmake` ≥ 3.24, Ninja, a host C++20 compiler. Hardware: none; no GPU lease is taken |
| Work breakdown | (1) the spec and its records, committed first; (2) `scripts/check-tree-compiles.py`; (3) `tests/scripts/test_check_tree_compiles.py`, red before (2); (4) the preflight block and `SUITES` entry, plus the CI script lane; (5) the `AGENTS.md` sentence; (6) the red-before/green-after evidence on the two historical trees |
| Risks/decisions | DECISION — no cap on the affected-TU count: a cap below the real fanout is a mute switch, and the worst case measured over 60 commits is 783 TUs / ~3.1 min, still 4x cheaper than the build it replaces. DECISION — exit 2 maps onto `SKIP` rather than `FAIL`, because a box without a compiler is not a defective change, and preflight already refuses the green banner over a skip. RISK — a stale `compile_commands.json` would compile the wrong file; answered by configuring fresh into a scratch directory every run, which costs 1.67 s. RISK — parallel agent builds have OOM-killed this box; `-fsyntax-only` writes no object file and jobs default to half the CPU count capped at 8 |

## Scope

On 2026-08-31 `main` was pushed twice in a state that does not compile, and
`scripts/agent-preflight.sh` was green both times.

| Fix | Defect | Shape |
|---|---|---|
| `5263ac31f` | `tools/bench/ltx2_connector_gemm_probe.cpp` ended seven `//` comment lines with a shell continuation, which `-Wcomment` plus this tree's `-Werror` rejects | one translation unit, and the diff that introduced it named that file |
| `08fa2f5aa` | `tests/vllm/models/test_glm_moe_dsa_schedule.cpp:304` passed `MlaSharedSelection*` into `ForwardMlaAttentionBlock`'s `vt::Tensor*` parameter (#2395) | one translation unit, and **no contributing diff named it** |

Preflight runs 30 record checkers and 60 Python suites. Not one of them compiles
a translation unit. `check-commit-trailers`, `check-commit-style`,
`check-agent-record`, `check-symbol-anchors`, `check-env-doc`,
`check-attention-rung-consistency` and `check-issue-index-append-only` all
validate records, prose, anchors and trailers against a tree, and every one of
them passes on a tree that does not build.

`.githooks/pre-push` runs three of those record checkers against the pushed
commit's worktree. It compiles nothing either.

CI compiles. `build-test-cpu`, `build-newest-gcc`, `cuda-fat-build` and the
sanitizers would have caught both. They land a verdict up to two hours after the
push, and both defects reached `main` by a direct push, so the verdict arrived
after the damage. The gap is not "does anything build". It is "does anything
build BEFORE the push".

**The second defect fixes the shape of the answer.** Neither contributing commit
carried the broken call. `e799f7d2c` added `attn_pre_o_proj` to the header;
`ee5c86031` added the test that calls it. Each side compiled alone. A gate that
compiles "the `.cpp` files this diff names" sees the first defect and is blind to
the second. The scope has to follow `#include` edges out of the changed headers.

Out of scope: linking, running tests, any configuration other than the default
Linux host build, and the CI lanes, which keep their job unchanged.

## Design

One checker, `scripts/check-tree-compiles.py`, invoked from preflight.

**1. Resolve the scope, and say what it is.** Changed paths come from the union
of `git diff --name-only <base> <head>`, the staged diff and the unstaged diff.
`<base>` defaults to `origin/main` resolved to a commit, `<head>` to `HEAD`. Each
component's path count prints, so a reader can see which of the three carried the
change.

**2. An empty scope is derived, not skipped.** When no C++ source, header, or
build file is in scope, the affected-TU set is provably empty, the checker says
so in words with the range it read, and exits 0. It does not print a compile
count it did not earn. This is the arm that keeps the gate off the 30 of the last
60 commits on `main` that touch no C++ at all.

**3. Configure, do not build.** `cmake -S . -B <dir> -G Ninja` writes
`compile_commands.json` and nothing else: measured 1.7 s and 14 MB on this box.
That file carries the exact flags CMake would use for every TU, `-Werror`
included, so the checker cannot drift from the build by reconstructing flags of
its own.

**4. Follow the include edges, exactly.** When a header or build file is in
scope, the checker runs `c++ -MM -MG` over every TU in `compile_commands.json`
with that TU's own flags, and inverts the result into a header → TUs map. This is
the real preprocessor with the real include path, not a textual `#include` scan,
so a conditional include, a macro-formed path and a generated header all resolve
the way the compiler resolves them. Measured 15.9 s for all 1218 TUs at `-j8`
under a load average of 28. The scan is skipped entirely when only `.cpp` files
are in scope, because then the affected set is exactly those files.

**5. Compile with `-fsyntax-only`.** The recorded command minus `-c` and `-o`,
plus `-fsyntax-only`. The front end runs, `-Werror` applies, and no object file
is written, so the check costs no disk and cannot collide with another agent's
build directory. Measured 0.27–3.26 s per TU, mean 1.76 s, on a loaded box.

**6. Three exit codes, and the third is never a pass.** `0` every TU in scope
compiled, or the set was provably empty. `1` at least one TU failed, and the
compiler's own message prints verbatim under the TU's path. `2` the check could
not run: no `cmake`, no compiler, a configure failure, an unresolvable base, or
no `compile_commands.json`. Preflight maps `2` onto its existing `SKIP` state
with the checker's reason, which already denies the "All gates green." banner and
already exits 1 under `--fail-on-skip`. Unknown is not absence and not success.

### Why not the obvious shapes

**Not a full build in preflight.** ~12 minutes warm, 9.4 GiB, and it fires on
every records-only change. AGENTS.md §Gates: a gate that fires on ordinary work
is the defect. The per-class line budgets were retired for exactly this.

**Not "compile the changed `.cpp` files".** That is the cheap diff scope, and it
is blind to the second defect, which is the one this row exists for.

**Not a textual `#include` graph.** It over-approximates on `#if`-guarded
includes and under-approximates on generated and macro-formed ones, and it would
have to be maintained against the include path. `-MM` answers the same question
with the compiler, in 15.9 s.

**Not a new CI job.** CI already compiles four ways. Adding a fifth moves no
verdict earlier than the push.

**Not prose alone.** A documented obligation to build before pushing is what
AGENTS.md already implies, and both of these landed anyway; the cost of a full
build is exactly why it does not get paid. The mechanism has to make the cheap
thing the correct thing. The prose change rides along: §Landing work gains one
sentence naming the checker, so the obligation and the affordance land together.

### The known diff-scope failure mode, and what is done about it

`.agents/specs/` records that a diff-scoped checker can SKIP when its base moves
and still exit 0 — a green that covered less than the reader thinks. Three
things keep that out of here.

- The empty-scope arm prints the base SHA, the head SHA and the derived TU count
  every time, so a zero is visible as a zero rather than inferred from silence.
- An unresolvable base is exit `2`, not exit `0` with an empty diff. A base that
  cannot be read is the `CANNOT-VERIFY` arm.
- The scope is a union with the staged and unstaged diffs, so a moved base
  narrows the committed component but cannot empty the set for work in hand.

It stays true that a base which moves *forward past your own commits* narrows the
range. That is the same exposure every range gate in preflight already carries,
it is stated here rather than argued away, and §Owed names it.

## Tests

`tests/scripts/test_check_tree_compiles.py`, registered in preflight's `SUITES`
and in the CI script lane. One case per guarantee, each red before the checker
exists:

1. A TU in scope that does not compile exits 1 and prints the compiler's message.
2. A changed header pulls in a TU that includes it and that the diff does not
   name. This is the second defect's shape, and it is the case a changed-`.cpp`
   scope fails.
3. An empty scope exits 0, says the set is empty in words, and does not print a
   compiled count.
4. An unresolvable base exits 2, not 0.
5. A missing `compile_commands.json` exits 2, not 0.
6. The report names the base, the head and the TU count, so the instrument states
   what it compared.
7. Preflight maps exit 2 onto `SKIP` and exit 1 onto `FAIL`, executed rather than
   grepped.

Mutations, each run against the build matrix first: delete the reverse-include
closure and case 2 must fail; turn the exit-2 arm into exit 0 and cases 4 and 5
must fail; drop `-Werror` from the reconstructed command and case 1 must fail.

## Risks

- **The checker is itself a TU consumer.** If `compile_commands.json` goes stale
  against a moved source tree, the checker compiles the wrong file. Mitigated by
  configuring fresh into a scratch directory on every run; 1.7 s buys that.
- **Parallelism on a shared box.** `-fsyntax-only` allocates far less than a
  codegen-and-link job, and jobs default to half the CPU count capped at 8.
  Parallel agent builds have OOM-killed this box before.
- **A wide header change costs minutes.** `3bfd1a738` touched a core `vt`
  header and reaches 783 TUs, ~3.1 min. That is the correct answer for that
  change and it is still 4x cheaper than the build it replaces. No cap is
  imposed, because a cap below the real fanout is a mute switch.

## Gates

- `python3 tests/scripts/test_check_tree_compiles.py`
- `python3 scripts/check-tree-compiles.py --base origin/main`
- `scripts/agent-preflight.sh`

## Evidence

Measured on this box, 20 cores, load average ~28 from concurrent agents, so
every figure is pessimistic.

| Quantity | Value |
|---|---|
| `cmake` configure, Ninja, Release | 1.67 s, 14 MB |
| TUs in `compile_commands.json` | 1218 |
| `c++ -MM -MG` per TU | 0.02–0.10 s |
| whole-tree dependency scan at `-j8` | 15.9 s |
| `c++ -fsyntax-only` per TU | 0.27–3.26 s, mean 1.76 s over 10 sampled TUs |
| full `cmake --build`, warm | ~12 min, 9.4 GiB |

Affected-TU fanout over the 60 commits ending at `9fa3be388`:

| TUs affected | Commits | Gate cost |
|---:|---:|---|
| 0 | 30 | ~0.1 s, no configure |
| 1–4 | 20 | 3–22 s |
| 18–37 | 6 | ~26 s |
| 358–783 | 4 | 1.4–3.1 min |

Red-before, on the two trees that pass every existing gate:

| Range | Head | Expected |
|---|---|---|
| `76fefdca7..77cedc9a5` | `5263ac31f^` | RED on `ltx2_connector_gemm_probe.cpp` |
| `76fefdca7..5263ac31f` | the fix | GREEN |
| `bcadc3a64..c27246d37` | `08fa2f5aa^` | RED on `test_glm_moe_dsa_schedule.cpp` |
| `bcadc3a64..08fa2f5aa` | the fix | GREEN |

## What this does NOT cover

Stated here because a checker's message defines what it enforces, and no gate
checks that this prose and that message agree.

1. **It does not link.** An undefined symbol, a duplicate definition, a missing
   vtable, an ODR violation and an unregistered CTest target all pass.
2. **One configuration.** The default Linux host `c++`, `Release`, with CUDA,
   HIP, Metal and MSVC off. A `.cu`, `.hip` or `.mm` TU is not in
   `compile_commands.json` and is not compiled. `444` — ROCm `main` does not
   build — is exactly the class this cannot see.
3. **One compiler.** A `clang`-only or `gcc-15`-only diagnostic stays CI's.
4. **It runs nothing.** A tree that compiles can fail every test in it.
5. **Include edges only.** A TU a change reaches through a CMake option, a
   generated header written by a script the diff changed, or an embedded data
   file is not pulled into scope.
6. **Build files partially.** A `CMakeLists.txt` edit is covered for configure
   errors and for TUs the diff names. An edit that changes flags for an existing
   TU it does not name is not re-verified.
7. **The tree at HEAD, not each commit.** A range red at commit 3 and green at
   commit 5 reads green, which is the right answer for what is about to be
   pushed and the wrong one for a bisect.

## Owed

- A base that moves forward past a branch's own commits narrows the committed
  component of the scope. Every range gate in preflight carries this; it is not
  introduced here and it is not closed here.
- Limit 6: comparing `compile_commands.json` at base and at head would cover a
  flag-only build edit exactly, at the cost of a second configure and a base
  worktree. Not built; #2401 owns it.
- Limit 2: the non-default configurations stay CI-only. No row claims moving
  them earlier.

## Stop conditions

Return `NEEDS_DECISION` rather than widening what preflight demands of a
`read-only` session, or adding a cap that suppresses a wide fanout.

## Now

Spec committed ahead of the implementation. Row `ACTIVE`.
