# Main Regression Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the two integration regressions introduced by merged PR #140
without reverting its ROCm/Gemma-4 functionality, adding an exception, or
raising a baseline; then reconcile and merge release PR #141.

**Architecture:** Gemma-4 expert GeGLU must enter the shared
`layers::MlpGateUpMethodBase` family, preserving both host-backed and already
device-resident expert layouts. Runner async eligibility must ask two portable
capabilities—whether the allocation is host-readable and whether the platform
implements the device mirror—instead of naming CUDA in shared code.

**Tech stack:** C++17, `vt::` op/provider and Platform seams, doctest, Python
policy/mutation gates, CMake/CTest, GitHub row-PR workflow.

## Global constraints

- Preserve every #140 ROCm and Gemma-4 BF16/FP8/resident-expert behavior.
- The source calls are `vt::GeluAndMul`, not `SiluAndMul`; Gemma uses GeGLU.
- This is the merged-GEMM/MoE seam, not the residual-norm `vt::FusedChain` glue
  seam. Do not add a decorative `FusedChain` token.
- Do not add or retain a `gemma4_moe` entry in
  `scripts/merged-gemm-consistency-allowlist.txt`.
- Do not add `DSR-ALLOW`, raise `scripts/device-leakage-baseline.json`, or move
  the CUDA test behind another shared-layer helper.
- Make two independently testable production commits: Gemma-4 seam repair
  first, runner capability repair second. Each commit carries its tests and any
  same-commit public-record projection required by the current policy.
- No performance claim is made. GPU/ROCm runtime evidence is correctness and
  inertness evidence only.
- Rebase onto current `main` before implementation and again before review.
  Keyed records are taken from `main` wholesale and the narrow repair clause is
  reapplied; they are never three-way merged.

---

## Intake and verified gap

The intake search on 2026-08-08 found merged PR
[#140](https://github.com/mudler/vllm.cpp/pull/140), which introduced both
affected files, and open PR
[#154](https://github.com/mudler/vllm.cpp/pull/154), which overlaps
`gemma4_moe.cpp` but is a separate resident-expert performance campaign. No
open issue or row PR already claimed this repair. This plan is reserved by
draft PR #158 on row `MODEL-TEXT-gemma4-gemma4-for-causal-lm`; that existing
model row owns the Gemma-4 PLE/YOCO/MoE backbone and avoids inventing a row.

The dispatched base is `1ce0d662`:

- `src/vllm/model_executor/models/gemma4_moe.cpp:27-47` and `:50-74`
  implement host and device expert GeGLU separately. Both end in direct
  `vt::GeluAndMul` calls at lines 46 and 73 and reference none of the shared
  gate-up constructs.
- `scripts/check-fusion-consistency.py:61-76,112-121,203-226` classifies those
  two calls as merged-GEMM drift. `python3 scripts/check-fusion-consistency.py`
  reports `gemma4_moe.cpp (2 gated-MLP epilogue hand-call site(s))`.
- `tests/scripts/test_check_fusion_consistency.py:117-192` contains the
  merged-GEMM mutation contract; its shipped-tree assertion at lines 156-163
  is red with `['gemma4_moe']`.
- `include/vllm/model_executor/layers/linear.h:82-142` already provides the
  shared base and GeGLU arm (`UnquantizedMlpGateUpGeluMethod`).
- `tests/vllm/model_executor/layers/test_linear_method.cpp:342-378` already
  proves the owned merged-weight GeGLU method is byte-identical to
  `{MatmulBT; GeluAndMul}`.
- `src/vllm/v1/worker/gpu/runner.cpp:88-111` adds
  `QueueSupportsAsyncInputCombine`; line 107 names `DeviceType::kCUDA` in the
  shared runner. Both constructors consume it at lines 347-348 and 388-389.
- `scripts/device-leakage-baseline.json:12-18` binds `kcuda=0`. Therefore
  `python3 scripts/check-device-leakage.py` is red with `kcuda: 1 > baseline
  0`; the baseline is already correct and must not change.
- `include/vllm/platforms/interface.h:120-128` already exposes the composed
  backend and host/device address-space query. `runner.cpp:2169-2193` already
  implements the CUDA device mirror, but the availability is not represented
  as Platform policy.
- `tests/scripts/test_device_leakage.py:155-219` proves a planted shared-layer
  `kCUDA`, including one hidden behind a helper, fails the DSR.

After dispatch, `origin/main` advanced to `1a021b1b` and added a
`gemma4_moe` known-drift allowlist entry. That makes the coarse gate green but
does not repair the execution seam and is explicitly outside this design. The
implementation starts by rebasing and removing that entry in the Gemma-4
repair commit. PR #154's current head must be inspected again immediately
before editing because it changes the same model file.

## File map

| Path | Responsibility |
|---|---|
| `include/vllm/model_executor/layers/linear.h` | Shared GeGLU gate-up method; add only the minimum borrowed/resident expert-weight form needed by Gemma-4. |
| `src/vllm/model_executor/models/gemma4_moe.cpp` | Route both host-backed and device-resident expert paths through that method. |
| `tests/vllm/model_executor/layers/test_linear_method.cpp` | Red-first byte-equivalence and layout/lifetime coverage for the expert-weight form. |
| `tests/scripts/test_check_fusion_consistency.py` | Pin both Gemma-4 call sites to the real method seam so a decorative token cannot satisfy the coarse checker. |
| `scripts/merged-gemm-consistency-allowlist.txt` | Remove the temporary `gemma4_moe` exception if present after rebase. |
| `include/vllm/platforms/interface.h` | Add a default-false `supports_async_device_mirror()` policy capability. |
| `src/vllm/platforms/cuda.cpp` | Override the capability true where the mirror is implemented. |
| `src/vllm/v1/worker/gpu/runner.cpp` | Select async input combine from unified host readability or the mirror capability; contain CUDA implementation details in the existing CUDA arms. |
| `tests/vllm/platforms/test_platform.cpp` | Capability defaults and CUDA override. |
| `tests/vllm/v1/worker/test_runner.cpp` | CPU/unified-memory default-on and rollback/inertness behavior. |
| `scripts/check-device-leakage.py`, `tests/scripts/test_device_leakage.py` | Existing unchanged DSR and mutation proof; no baseline edit. |

## Task 1: Fold Gemma-4 expert GeGLU onto the shared method family

**Interfaces**

- Consumes: `layers::MlpGateUpMethodBase::Apply(Dev, const vt::Tensor&) ->
  DBuf`, the existing `UnquantizedMlpGateUpGeluMethod`, and contiguous Gemma-4
  expert weights `[2I,H]` / `[H,I]`.
- Produces: one shared GeGLU method form that can bind either a host-backed
  expert slab (staged through `DBuf`) or an already resident `vt::Tensor`, with
  both `ExpertGeGLUHost` and `ExpertGeGLUDevice` calling it.

- [ ] **Step 1: Rebase and repeat intake.** Fetch `main`; inspect PR #154 and
  all open row PRs; confirm the two direct calls and allowlist state at the new
  head. Resolve any overlap before editing.

- [ ] **Step 2: Add the red source-routing assertion.** Extend
  `test_check_fusion_consistency.py` to inspect comment-stripped
  `gemma4_moe.cpp` and require zero direct `vt::GeluAndMul` calls plus two real
  shared-method applications in the two expert helpers. Run:

  ```sh
  python3 tests/scripts/test_check_fusion_consistency.py
  ```

  Expected before production edits: FAIL naming both direct sites. The test
  must still fail if either one of the two method applications is replaced by
  the old direct sequence.

- [ ] **Step 3: Add the red numerical expert-layout test.** Extend
  `test_linear_method.cpp` with deterministic BF16 `[2I,H]` expert data and
  compare the new host-backed and resident-view method form against the exact
  pre-repair sequence: two current projections/packing operations,
  `GeluAndMul`, then down projection. Assert raw BF16 bytes, shape `[M,I]`, and
  method name. Run:

  ```sh
  cmake --build build --target test_linear_method -j2
  ctest --test-dir build -R '^test_linear_method$' --output-on-failure
  ```

  Expected before implementation: compile failure because the expert method
  form does not exist.

- [ ] **Step 4: Implement the minimum shared method form.** Keep device choice
  out of the method. A host slab is staged into the queue's backend before
  binding its tensor; a resident slab is bound directly. Preserve the existing
  activation order and BF16 store boundary. Do not introduce a Gemma-private
  fused kernel or route GeGLU through the SwiGLU-only
  `MoeGateUpSwiGLUGrouped` descriptor.

- [ ] **Step 5: Route both production helpers.** Replace both direct
  `GeluAndMul` sequences with the shared method. Remove the temporary
  `gemma4_moe` allowlist entry after rebasing. Run:

  ```sh
  python3 scripts/check-fusion-consistency.py
  python3 tests/scripts/test_check_fusion_consistency.py
  ctest --test-dir build -R '^(test_linear_method|test_gemma4_paged_engine)$' --output-on-failure
  ```

  Expected: the merged-GEMM drift list is empty; all method and available
  Gemma-4 gates pass. An unavailable real-checkpoint/GPU gate remains explicitly
  pending rather than being simulated.

- [ ] **Step 6: Commit independently.** Include required keyed public-record
  projections in this same commit, describing a structural correctness repair
  with no speed number.

  ```text
  fix(gemma4): route expert GeGLU through shared gate-up method
  ```

## Task 2: Replace the runner's CUDA identity test with capabilities

**Interfaces**

- Consumes: `Platform::is_unified_memory()` and the existing CUDA
  `GPUModelRunner::async_device_mirror()` implementation.
- Produces: `Platform::supports_async_device_mirror() -> bool`, default false,
  CUDA true; `QueueSupportsAsyncInputCombine` returns
  `is_unified_memory() || supports_async_device_mirror()`.

The truth table is binding:

| Platform allocation/mirror | Async input combine |
|---|---|
| CPU or any genuinely unified-memory backend | true |
| Discrete CUDA with the implemented device mirror | true |
| Discrete ROCm without a HIP mirror | false |
| A future discrete backend after it implements and advertises a mirror | true |

- [ ] **Step 1: Add red capability tests.** In `test_platform.cpp`, assert the
  base/CPU capability is false and the compiled CUDA platform capability is
  true. Add a pure truth-table test for the eligibility predicate so the
  discrete-ROCm case is exercised without pretending CPU memory is discrete.

- [ ] **Step 2: Confirm the DSR is red first.** Run:

  ```sh
  python3 scripts/check-device-leakage.py
  python3 tests/scripts/test_device_leakage.py
  ```

  Expected before the production edit: the real-tree checker fails with
  `kcuda: 1 > baseline 0`; the mutation suite itself passes.

- [ ] **Step 3: Implement capability selection.** Add the Platform virtual and
  CUDA override, then rewrite `QueueSupportsAsyncInputCombine` using only those
  two capabilities. Keep CUDA headers, launches, and compile guards in the
  existing CUDA implementation arms; this task removes only shared policy's
  device identity test.

- [ ] **Step 4: Prove behavior and leakage.** Run:

  ```sh
  python3 scripts/check-device-leakage.py
  python3 tests/scripts/test_device_leakage.py
  cmake --build build --target test_platform test_runner test_loaded_engine_dense -j2
  ctest --test-dir build -R '^(test_platform|test_runner|test_loaded_engine_dense)$' --output-on-failure
  ```

  Expected: DSR remains `kcuda=0` with no baseline edit; CPU/unified async stays
  default-on; the sync rollback and scheduler construction matrix remain green.
  Where the authorized ROCm host is available, additionally compile the HIP
  runner and run `test_rocm_backend`; a discrete board must resolve async input
  combine false until a HIP mirror exists.

- [ ] **Step 5: Commit independently.** Include required keyed public-record
  projections in this same commit, with no performance claim.

  ```text
  fix(runner): select async combine by platform capability
  ```

## Task 3: Review, integrate the repair, then reconcile release PR #141

- [ ] **Step 1: Run focused and full gates on the repair head.** At minimum:

  ```sh
  python3 scripts/check-fusion-consistency.py
  python3 tests/scripts/test_check_fusion_consistency.py
  python3 scripts/check-device-leakage.py
  python3 tests/scripts/test_device_leakage.py
  scripts/agent-preflight.sh
  python3 scripts/check-doc-checkpoint.py --base origin/main --head HEAD
  ```

- [ ] **Step 2: Dispatch a fresh mutation reviewer.** It must independently
  replace each of the two Gemma-4 method applications with the old direct
  activation and reintroduce the runner `kCUDA` branch. Each mutation must make
  its named gate red. Findings go to a fresh implementer and then a scoped fresh
  re-review.

- [ ] **Step 3: Rebase and merge the repair PR.** Re-run the operator's full
  gate on the exact reviewed head and merge in the same session when CI is green.

- [ ] **Step 4: Reconcile #141 from the repaired `main`.** For
  `docs/STATUS.md`, `docs/BENCHMARKS.md`, `docs/FEATURES.md`, `.agents/NOW.md`,
  matrices, and coordination records, take repaired `main` wholesale and reapply
  only #141's W5 release-manifest clauses. Union-append append-only evidence and
  run `scripts/sort-state-tail.py --apply` if required. Verify unrelated repaired
  main lines are byte-identical.

- [ ] **Step 5: Review and merge #141.** Re-run its manifest suites, full
  preflight, exact-head CI, and a fresh mutation review; then mark ready and merge
  in the same session. This lands the W5 manifest/static-boundary contract only;
  it does not claim that binary artifacts are already published.

## Success criteria

- Both Gemma-4 expert paths invoke a real `MlpGateUpMethodBase` family method;
  neither contains a direct `vt::GeluAndMul` call.
- The structural mutation test fails when either call site is unfolded.
- `gemma4_moe` is absent from the merged-GEMM allowlist.
- The shared runner contains no code-level `kCUDA` reference for async combine.
- Async eligibility follows the four-row capability truth table and preserves
  CPU, CUDA, unified-memory, and discrete-ROCm behavior.
- Device leakage stays at `kcuda=0`; no baseline or waiver changes.
- Focused tests, full preflight, exact-head CI, and independent mutation review
  pass for the repair; then reconciled #141 passes its own gates and is merged.
