# Sync cycle `e126687a9a`, wave STEP6

Row: `UPSTREAM-SYNC-HEADPIN`.
Issue: [#2771](https://github.com/mudler/vllm.cpp/issues/2771).
Predecessors: [#2593](https://github.com/mudler/vllm.cpp/issues/2593) wave
HEADPIN, [#2611](https://github.com/mudler/vllm.cpp/issues/2611) wave RUNHALF,
and [#2764](https://github.com/mudler/vllm.cpp/issues/2764) wave
PORTQ-RECONCILE, which ruled the 290-entry PORT-NOW queue discharged as a
reconciliation obligation and named three blockers that stand. This wave is
blocker 2.
Row spec: [`upstream-sync-headpin.md`](upstream-sync-headpin.md).

## Now

**The affected set is determined, and it is two gates, both FlashInfer's.** Of
the four denominators `.agents/oracles/vllm.md:66-69` names, only FlashInfer
reaches anything committed:

| Denominator | Reaches a committed gate? |
|---|---|
| FlashInfer `0.6.15.post1` to `0.6.18` | **YES**, `vllm-online-serving` and `speculative-decoding` |
| CUTLASS DSL `4.6.0` to `4.6.2` | **NO**, discharged, §2.3 |
| `transformers` floor `>= 5.5.3` to `>= 5.10.4` | **NO**, discharged, §2.4 |
| `VLLM_ALLREDUCE_USE_FLASHINFER` default `0` to `1` | **NO**, discharged, §2.5 |

**The re-measurement was NOT run, on either gate.** No CUDA device was free at
any of three `rc` readings, `dgx:gpu0` ended the wave
`unhealthy (no contact 2h9m47s)`, `thor:gpu0` is `sm_110` and cannot carry a
GB10 baseline, and the committed harness structurally refuses to run at the
target anyway (§2.6). The report's §7 C1a and C1b name the two exact jobs.

The pin did **not** advance and nothing here is a reason to move it. The active
parity pin remains `5559679229bc961848b121ccdeaa8fa5d79bec98`.

## 1. Scope

**Two questions, in order.**

1. Which committed gates have a recorded denominator that was measured under a
   value one of the four has since moved? A gate is affected when the moved
   value was on the executed path of the run that produced its recorded number,
   or when the committed harness asserts that value as a precondition of the
   measurement.
2. For each affected gate, re-measure the baseline at `e126687a9a`, on a fleet
   device, under vLLM's production graphed configuration, never
   `--enforce-eager`, with identical artifacts, prompts, token counts,
   batching, concurrency and sampling on both sides.

**In scope.** The four denominators, the committed gate surfaces
(`docs/BENCHMARKS.md`, `docs/benchmarks/*.md`, `docs/bench-evidence/*`,
`.agents/benchmark-record.md`, `.agents/parity-ledger.md`), the committed
harness (`tools/bench/`), and the pinned and target vLLM sources.

**Out of scope.** Advancing the pin. Porting a PORT-NOW entry. The declared
token-exact gate at the target (blocker 1). A reading on `dgx:gpu0` beyond
whatever this wave's own job obtains (blocker 4). Re-deriving any disposition
the PORTQ waves recorded.

**Not a substitute for measurement.** A discharge in §2 is a claim that a moved
value is not on a committed gate's executed path. It is argued from the
committed record and from upstream source, and where it is a source argument
this spec says so rather than dressing it as a measurement.

## 2. Design

### 2.1 What counts as a committed gate here

`docs/BENCHMARKS.md:7-19` is the public index; nine benchmark IDs, each owning
one detail file. **Two** of them carry a number the record calls binding against
a vLLM denominator at the current pin.

- `vllm-online-serving`, rows `docs/benchmarks/vllm-online-serving.md:66-67`
  (Qwen3.6-27B NVFP4 `nvidia` @`0893e160`) and `:106-107` (Qwen3.6-35B-A3B
  NVFP4), both marked "**BINDING at the pin, graphed, `--language-model-only`,
  clocks 2184 MHz**", and carried internally by the clock-controlled grid at
  `.agents/benchmark-record.md:22318-22343`.
- `speculative-decoding`, row `docs/benchmarks/speculative-decoding.md:6`
  (DFlash, 1.003x, `DONE`), carried internally at
  `.agents/benchmark-record.md:5161-5165` with `benchmark_binding=true`, over
  four earlier binding rungs at `:5360-5361`, `:5437-5438`, `:5469-5470` and
  `:5503-5504`.

Everything else on those pages is SUPERSEDED, VOID, PENDING or owes no number,
and `docs/benchmarks/open-gaps.md` carries the reasons. The one binding row
outside GB10, Qwen3.5-4B on the `sm_120` workspace box
(`.agents/parity-ledger.md:884`, `benchmark_binding=true`), records the full
oracle stack including `flashinfer 0.6.15.post1` and `nvidia-cutlass-dsl 4.6.0`,
and records `FLASH_ATTN` as the backend actually selected.

### 2.2 FlashInfer reaches both of them

Three facts, all committed. The first two are the load-bearing ones because
they are about what executed; the third is about what the harness will let run
at all.

**FlashInfer is on the denominator's executed path, and that was observed.** On
GB10 (`sm_121`) vLLM's NVFP4 linear-kernel auto-selection resolves to
`FlashInferCutlassNvFp4LinearKernel`, which calls
`flashinfer_scaled_fp4_mm(backend="cutlass")`.
`.agents/parity-ledger.md:66` records the A/B on the box that established it:
the auto-selected native run produced token 6 = 198, and forcing the
alternatives through `VLLM_DISABLED_KERNELS` produced 271.
`.agents/specs/prefill-gap-scan-2026-07-08.json:91` states the conclusion,
names the earlier assumption it supersedes, and records that the cute-dsl
kernel above it in priority is skipped on `sm_121`.
`.agents/specs/nvfp4-bf16-producer-vectorization.md:215` and
`.agents/specs/qwen27b-w4a4-notes.md:427,589` carry the same selection.

The tactic that call uses is chosen by vLLM's warmup autotuner and cached in a
directory **keyed by the FlashInfer version**:
`tests/fixtures/nvfp4_flashinfer_v025_gb10/manifest.json:2` records the source
path `.../flashinfer_autotune_cache/0.6.13/121a/<hash>/autotune_configs.json`,
and `:10` records `"flashinfer_version": "0.6.13"` beside 64 fp4 entries. A
different FlashInfer is a different tactic search over a different candidate
list.

**FlashInfer is also on the numerator's build path.**
`tools/bench/online_gate.py:3610` fingerprints
`<flashinfer package root>/data/cutlass` and records it as
`cutlass_source_tree`; `:3766-3767` refuses if that tree drifts before
execution; `:3793-3801` refuses unless the build's `VLLM_CPP_CUTLASS_DIR`
resolves to that same path.
`tools/bench/gdn_packed_component.py:1509-1515` and `:1592-1595` impose the
same contract. So our own `src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu` is
compiled against CUTLASS headers vendored inside the pinned FlashInfer wheel,
and `docs/bench-evidence/qwen3-4b-binding-20260721.log:3-4` records a binding
build configured exactly that way
(`-DVLLM_CPP_CUTLASS_DIR=.../flashinfer/data/cutlass`).

**One driver of this gate also asserts the version outright, and one does
not.** `tools/bench/serve_low_common.py:53,109` read `flashinfer_version` out
of the ` ```parity-pin ` block in `.agents/upstream-sync.md`.
`tools/bench/online_gate.py:3552-3560` refuses the run when the installed
FlashInfer distribution or runtime version differs from it, and `:3714-3720`
re-asserts the same pair at execution against the manifest on disk;
`scripts/dgx-online-serving.sh:305` and `scripts/dgx-gdn-packed-component.sh:175`
run that `record-oracle` step. **`tools/bench/run_serve_low.py`, the driver of
the clock-controlled series, does not**: it imports `VLLM_COMMIT`
(`run_serve_low.py:32`) and records it (`:592`), and carries no
`FLASHINFER_VERSION` reference at all. `docs/benchmarks/reproduce.md:6` says
"identity asserted per leg", and the identity it means is the oracle's, not its
dependency's. This spec does not claim more than that.

**On `speculative-decoding` it is the oracle's attention backend, not a GEMM
under it.** `.agents/benchmark-record.md:5233-5234` records "Backend
auto-selects `flashinfer-native` fp8-KV sm121 (NOT FLASH_ATTN)", restated at
`:12985-12986`, and `:5455` draws the D8 consequence that our kernels "cannot
bit-match vLLM's flashinfer/cutlass draft+paged-attn".

**What that adds up to.** Moving FlashInfer moves the kernel one denominator
runs, the attention backend the other denominator runs, the tactic the first
picks, and the headers the numerator is compiled from. Neither recorded ratio is
transferable across the move, and re-measurement is owed for both.

**One recorded case shows this is not hypothetical.**
`.agents/benchmark-record.md:23161-23163` records two `FAILED` FlashInfer JIT
compiles of `gen_gemm_sm120` on a rebuilt stack, and `:23181-23183` states the
consequence: "a partial fallback would change which kernels the oracle actually
runs".

**What the public record does NOT contain, which is itself a finding.** Not one
of the binding GB10 rows records a FlashInfer version, a CUTLASS DSL version, a
torch version, or the attention backend the oracle selected. The single public
evidence file that records the oracle's full dependency stack,
`docs/bench-evidence/qwen35-4b-pinned-oracle-20260728.md:31-36`, belongs to a
different box (RTX 5070 Ti, `sm_120`) and a different arm (bf16 dense
Qwen3.5-4B), and records `FLASH_ATTN` as the selected backend out of four
candidates. The GB10 side has only the prose claim at
`docs/benchmarks/how-we-measure.md:21-24`. So the FlashInfer version under the
binding GB10 numbers is asserted by the harness and by the pin record, and is
nowhere written into the evidence beside the numbers themselves.

### 2.3 CUTLASS DSL is discharged

**It is recorded as installed, and it is not on any executed path this fleet
can take.** `docs/bench-evidence/qwen35-4b-pinned-oracle-20260728.md:33` is the
one committed measurement that names it, `nvidia-cutlass-dsl 4.6.0`, in the
oracle stack of the binding Qwen3.5-4B row. That is an inventory of what `pip`
resolved, not evidence that a kernel from it ran.

`nvidia-cutlass-dsl` is the Python package vLLM imports as `cutlass`. Nineteen
modules import it at the pin. Three of them can be reached by the models this
project gates, and **every one is gated behind device capability family 100**,
which no host in this fleet is: GB10 is `sm_121` and the local workspace box
`sm_120`, both family 120; Thor is `sm_110`, family 110.

1. **The NVFP4 linear kernel.** `FlashInferCuteDslNvFp4LinearKernel.is_supported`
   returns `False, "FlashInfer cutedsl requires sm_10x"` unless
   `is_device_capability_family(100)`
   (`vllm/model_executor/kernels/linear/nvfp4/flashinfer.py:35-42` @
   `5559679229`). It is first in `_POSSIBLE_NVFP4_KERNELS`
   (`kernels/linear/__init__.py:448`), so it is skipped on every gate host and
   selection falls to `FlashInferCutlassNvFp4LinearKernel`, which §2.2 records
   as observed.
2. **The GDN prefill kernel**, which matters because every Qwen3.5/3.6/3.8 arm
   on this gate is a hybrid GDN model.
   `_resolve_gdn_prefill_backend` sets `supports_cutedsl` only inside the
   `is_device_capability_family(100) and head_k_dim == 128 and
   cuda_runtime_major >= 13` branch, and the `cutedsl` return additionally
   requires the backend to be **explicitly requested**, never `auto`
   (`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:112-133` @
   `5559679229`; the docstring at `:97-100` says "opt-in only"). On a family-120
   host neither branch fires and the function returns `triton`, the FLA path.
3. **FA4.** The `import cutlass` calls in
   `vllm/v1/attention/backends/flash_attn.py:1401-1403` and `:1489-1490` are
   function-local, inside `vllm_flash_attn.cute` helpers on the FA4 path. The
   recorded runs selected FA2 (`docs/bench-evidence/qwen35-4b-pinned-oracle-20260728.md:35-36`,
   `FLASH_ATTN` out of four candidates) and `.agents/sync/2026-09-03-e126687-runhalf.md`
   §2 records the same selection at the target.

`scripts/dgx-online-serving.sh:506-508` records the parallel MXFP4 case, where
the cute-dsl backend does not merely lose the selection on `sm_121` but aborts
engine start, and `docs/bench-evidence/mxfp4-qwen/W0-W1-oracle-run.md:26`
carries the exception verbatim.

No committed harness asserts a `nvidia-cutlass-dsl` version. The only
occurrence outside records is `scripts/rc-sglang-oracle-lease.sh:227`, which
greps a `pip freeze` for an SGLang lease and gates nothing.

**Scope of the discharge.** It covers the gate hosts this project has. It is a
source argument, not a measurement, and it does not hold on an `sm_100` device,
where the cute-dsl NVFP4 kernel is first in priority and the GDN cutedsl branch
becomes reachable.

### 2.4 The `transformers` floor is discharged

The floor is a **lower bound**, and the version the pinned environment actually
resolved is above both floors. `.agents/upstream-sync.md` records
`transformers 5.14.1` beside the pin; `.agents/oracles/transformers.md:35-45`
pins the oracle at `5.14.1` and says in as many words that the pin "is the one
resolved inside the pinned vLLM environment". `5.14.1 >= 5.10.4`, so no
denominator measured at `5.14.1` falls outside the target's requirement.

**What the floor move does not cover, and this wave says so rather than hiding
it.** A rebuild of the oracle environment at the target resolves the open floor
to whatever is newest that day; `.agents/sync/2026-09-02-e126687.md` §5.2
measured `transformers 5.16.1` there. That is a real move of the installed
version, but its cause is pip resolving an open bound on a rebuild, not the
floor's value, and it would happen identically had the floor not moved. It
belongs to the pin advance's own re-validation, which
`.agents/specs/pin-advance.md` §3 already treats as a golden-drift question, not
to this denominator.

### 2.5 `VLLM_ALLREDUCE_USE_FLASHINFER` is discharged

`vllm#52998` (`6a962071bd`) flips the default from `False` to `True`
(`vllm/envs.py:244` @ `5559679229` against `vllm/envs.py:266` @ `e126687a9a`).
**At `tensor_parallel_size == 1` the flag cannot change behavior**, at either
revision, on all three of its consumers:

1. `vllm/distributed/device_communicators/cuda_communicator.py:50-55` @
   `e126687a9a` (`:48-53` @ the pin) forces `use_flashinfer_allreduce = False`
   whenever `"tp" not in unique_name`.
2. `cuda_communicator.py:108` @ `e126687a9a` (`:103` @ the pin) constructs
   `FlashInferAllReduce` only `if self.use_flashinfer_allreduce and
   self.world_size > 1`.
3. The remaining reader, `flashinfer_all_reduce.py:190-191` @ `e126687a9a`, is
   inside `prepare_fi_ar_workspace`, reached from `AllReduceFusionPass`. That
   pass is added only when `pass_config.fuse_allreduce_rms` is true
   (`vllm/compilation/passes/pass_manager.py:176-180`), which resolves through
   `enable_allreduce_rms_fusion` — whose CUDA branch requires
   `cfg.parallel_config.tensor_parallel_size > 1`
   (`vllm/config/vllm.py:159-183`, and whose docstring states the condition).

**Every committed gate runs `tensor_parallel_size = 1`.** The bench harness
never sets a parallel size, so vLLM's default of 1 applies; every explicit
setting anywhere in `tools/` and `scripts/` is the literal `1`
(`tools/parity/dump_27b_emulation_greedy.py:84`, `tools/parity/dump_qwen36.py:248`,
`tools/parity/dump_qwen3_5_mtp.py:274`). The project has no multi-GPU
denominator at all: `docs/benchmarks/open-gaps.md:24` records tensor parallelism
as `benchmark_binding=false`, records-only, with its perf gate TP-W6 `PENDING-HW`
for want of a two-GPU box.

**Scope of the discharge.** It holds exactly while no committed gate runs
`tensor_parallel_size > 1`. The moment TP-W6 gets its hardware, this flag
becomes the first thing that must be held fixed between the arms.

### 2.6 The committed harness refuses to measure at the target

This is a finding, not an obstacle to be worked around. `FLASHINFER_VERSION` is
read from the `parity-pin` block (`serve_low_common.py:109`) and asserted at
`online_gate.py:3552-3560` and again at `:3714-3720`. Running the canonical
online-serving driver against a target whose FlashInfer is `0.6.18` therefore
requires editing that block first — which is the pin advance, the very step this
re-measurement is a precondition of.

The dependency is real and this spec does not resolve it by loosening an
assertion. §4 records the two admissible shapes and which one this project's
rules select.

## 3. Risks

- **An instrument whose failure looks like a result.** Three of four reds in the
  RUNHALF wave were its own instruments. Any job this wave writes prints an
  explicit rc per leg and a `SUM` line, and a leg that did not run is reported
  as not run rather than as a red.
- **A zero read as an absence.** Every negative claim in §2 is paired with a
  positive control through the same probe form, and the report records both.
- **A speed number without an idle host.** Any re-measurement is refused unless
  the host is idle, the clocks are pinned, and the arms are interleaved under one
  lease.
- **Over-reading a source argument.** §2.3 and §2.5 are source arguments. They
  are labelled as such and carry their own scope limits.

## 4. Gates

**G1, the affected set.** Each of the four is either named with the committed
gate it reaches, with `file:line`, or discharged with evidence and an explicit
scope limit. Met by §2.2 to §2.5.

**G2, the re-measurement.** For each affected gate, a baseline at the target on
a fleet device under the production graphed configuration, recorded with the
exact build and run recipe, revisions, model hashes, environment and contention
state. **NOT MET.** No CUDA fleet device was free, and §2.6's ordering
constraint stands regardless. The report's §7 C1a and C1b carry the two exact
jobs, both on `dgx:gpu0`, sharing one source build of vLLM at the target and one
rebuild of our arm against that wheel's `data/cutlass`.

**G3, the ordering constraint.** §2.6 is recorded as a property of the harness,
with the two admissible resolutions named, and no assertion is loosened to make
a red gate green. Met.

## 5. Stop conditions

- **Stop and report** if no CUDA fleet device can be had in reasonable time. The
  affected set with its citations is the deliverable that lets the next wave
  measure. This condition fired.
- **Stop** rather than edit the `parity-pin` block. Advancing it is the pin
  advance and has its own review.
- **Stop** rather than weaken any assertion in `tools/bench/`.
- **Return `NEEDS_DECISION`** rather than pick a resolution to §2.6 unilaterally.

## Owed

- **G2, the re-measurement itself**
  ([#2771](https://github.com/mudler/vllm.cpp/issues/2771)). Not run; §4 names
  the job.
- **The §2.6 ordering constraint** (#2771). The step-6 obligation and the
  harness's own fail-closed pin assertion are in tension, and nothing in
  `.agents/upstream-sync.md` says which moves first.
- **A reading on `dgx:gpu0`**, blocker 4 of `.agents/oracles/vllm.md:70`. Not
  discharged here, and now blocked on the environment: the box read
  `unhealthy (no contact 2h9m47s)` at this wave's last `rc devices`. Nobody can
  discharge blocker 4, or run this wave's own G2, while that holds.
- **The `sm_100` scope limit on §2.3.** The CUTLASS DSL discharge does not hold
  on a device of capability family 100, and this fleet has none to test it on.
