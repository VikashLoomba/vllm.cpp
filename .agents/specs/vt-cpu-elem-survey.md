# SPEC — `VT-CPU-ELEM-SURVEY`: which of the other 62 CPU kernels the per-element dtype dispatch actually costs

Issue: [#2416](https://github.com/mudler/vllm.cpp/issues/2416).
Predecessor: [`.agents/specs/vt-cpu-elem-dispatch.md`](vt-cpu-elem-dispatch.md),
whose `## Owed` names this work, states the sizing method, and leaves
[#2376](https://github.com/mudler/vllm.cpp/issues/2376) open against it.

## The gap

`VT-CPU-ELEM-DISPATCH` hoisted the loop-invariant dtype dispatch out of
`AttentionKernel` and `AttentionCrossKernel` and measured 8.75x-11.16x for the
pair. It hoisted two of the 64 functions that call `vt::cpu::LoadF32` /
`StoreF32`. The other 62 still resolve a dtype per element, and its `## Owed`
gives the method for sizing them verbatim:

> for each candidate, `perf record -e cpu-clock` over a probe that runs THAT op
> alone at a shape a shipped model actually uses, and read `LoadF32`'s self
> percentage. Above ~30% the hoist is worth a row; below it the kernel is bound
> by something else and a hoist buys the fraction the profile names. **Do not
> sweep**: each hoist needs its own byte-equality gate.

Nobody has run it. Without the ranking the only ways forward are a 62-kernel
sweep nobody can review, or a guess about which kernel is next.

**The call sites, counted on this branch's base `43553262c`.** 271 calls across
67 enclosing functions, all of them in `src/vt/cpu/cpu_ops.cpp`.
**`src/vt/cpu/cpu_paged_attn.cpp` contains none** — the five matches there are
its own already-hoisted `KvKind` resolver and its comments, so the predecessor
row's "two files" is now one. Of the 67, five are the shared helpers rather than
kernels (`LoadF32`, `StoreF32`, `AttnResolveOrRefuse`, `FusedLoad`, `FusedStore`)
and two are the pair already hoisted, which is where the "other 62" comes from.

## Scope

IN scope:

- **W1** — a committed probe that runs ONE op alone at a grounded shape, and the
  ranked `LoadF32` self-percentage table over it. **The ranking is this row's
  product**: it lets every future row prioritize by measurement instead of
  guessing, and it is what stops the remaining work becoming a 62-kernel sweep.
- **W2** — hoist ONLY what the ranking justifies, each with its own byte-equality
  gate. One kernel here; see `## Outcome` for why that one and not the next.
- **W3** — the `__attribute__((always_inline))` verdict, with the i-cache
  evidence the predecessor row said it lacked, either way.

OUT of scope, declared rather than approximated:

- **Hoisting the rest of the ranked set.** The measurement says roughly half the
  measured kernels clear the bar. That is a finding, not a licence: each hoist
  needs its own gate and its own mutations, and doing them in one change is the
  sweep the predecessor row forbade. They are in `## Owed`, ranked.
- **Changing any output value.** Byte equality is the bar. A kernel that cannot
  be hoisted bit-exactly stops and is reported.
- **The aarch64 half.** Every number here is x86-64 AVX-512. See `## Owed`.
- **Folding `cpu_paged_attn.cpp`'s private `StoreRowF32` into the shared
  `NarrowRowFromF32` this row adds.** They differ only in the refusal MESSAGE,
  so folding them changes another kernel's observable behaviour and needs its own
  gate. `## Owed`.

## Design

### The instrument

`tools/bench/vt_cpu_elem_survey_probe.cpp`. `--op <name>` runs one op in a loop
with no other `vt` work in the process, so `perf report`'s WHOLE-PROCESS self
percentages are that kernel's and no call-graph attribution is needed. Every case
reaches its kernel through the production `vt::` entry point, never by
constructing the kernel: a probe that called the kernel directly would measure a
function, not the path a model takes. `--list` prints, for every case, the
extents AND the model whose config they come from.

**The shapes are grounded, and that is load-bearing rather than decorative.** The
question a hoist turns on is whether the operand stream is dispatch-bound or
DRAM-bound, and that is a property of the extents, so a synthetic shape ranks
kernels wrongly. Qwen3.6-27B and Qwen3.6-35B-A3B config values come from
[`qwen36-forward-notes.md`](qwen36-forward-notes.md) §1 (read off the real
checkpoints), the GDN derived dims from [`gdn-semantics.md`](gdn-semantics.md)
§1, and the LTX-2.5 connector geometry from
`tools/bench/ltx2_connector_gemm_probe.cpp`. The two token counts are the two
regimes this repo's own harness runs: PREFILL 1024 and DECODE 16
(`tools/bench/online_gate.py:59` `INPUT_LEN`, `:74` `TRACE_CONCURRENCY`).

**Single thread, and why.** The predecessor's profile was taken at the shipped
thread count and carried `Threadpool::Barrier()` at 16.69%. This devbox ran at
loadavg 63-146 throughout, where the same barrier reads 46.77% and swamps
everything else, so a multithreaded ranking here would rank box contention. Every
ranking number below is therefore `VLLM_CPP_CPU_THREADS=1`, which measures the
kernel's instruction mix and not the spin-wait. **The bar is recalibrated onto
that instrument rather than carried across** — see `## Outcome`.

### The hoist

The same transformation `AttentionKernel` and `MatmulOneChunk` already apply in
the same file: resolve the dtype once per call and go through the shared
`WidenRowToF32` / `NarrowRowFromF32` row helpers. `NarrowRowFromF32` is new and
lands beside `WidenRowToF32` in `src/vt/cpu/cpu_matmul_elem.{h,cpp}`, which is
where a caller already looks for the pair.

## Reachability

Nothing new lands unreached. `RmsNormKernel` is already registered and already
reached from `ModelRegistry::Forward` through `vt::RmsNorm` — every text model in
the tree calls it twice per layer, and the `FusedChain` tier-0 walker dispatches
`kRmsNorm` to it. This row changes the body of a reached kernel and adds no entry
point. The gate enters through `vt::RmsNorm`, never by constructing the kernel,
and M7/M8 in `## Outcome` are the reachability mutations: making the new row
helpers inert reds the gate through the production entry point.

`NarrowRowFromF32` is new and is reached by that same call site at its own merge
commit. It is not a staged slice.

## Tests to port

There is no upstream test: upstream's RMSNorm is `torch`. The tests are this
tree's own, and each is an executable observable.

| ID | Assertion |
|---|---|
| T1 | CPU `vt::RmsNorm` is `memcmp`-identical to a per-element reference over the whole dtype matrix the op accepts, at ragged extents, gemma and non-gemma |
| T2 | the residual stream's add / round-on-store / RE-READ order is `memcmp`-identical, in BOTH residual dtypes, and the residual bytes are checked as well as the output bytes |
| T3 | a non-float operand is still refused at the op boundary |

`tests/vt/test_ops_rmsnorm_elem_dispatch.cpp`. The reference is the ORIGINAL
per-element loop, re-derived from the layout contract in `include/vt/ops.h` and
sharing nothing with `src/vt/cpu` — including its own hand-written f16 and bf16
conversions, because a gate that compared the kernel against a helper the kernel
also uses would prove consistency, not correctness.

**The dtype matrix is bounded by what the OP accepts.** `vt::RmsNorm` takes x and
weight in {f32, f16, bf16} (`IsFloat`) and out and residual in {f32, bf16}
(`IsOutFloat`, `src/vt/ops.cpp:25`). FOUR tensors carry INDEPENDENT dtypes, which
is exactly the shape a typed table indexed by the wrong operand gets wrong, and
before this row every CPU RmsNorm test ran them all f32 or all bf16.

## Gates

1. `tests/vt/test_ops_rmsnorm_elem_dispatch.cpp` green, and green on a pristine
   base tree too, which is what says the reference is a valid oracle rather than
   a transcription of the new code.
2. The CPU `vt` norm suites green with the SAME case and assertion counts on both
   trees, so nothing was silently skipped.
3. A mutation per claimed guarantee, each verified to have APPLIED and BUILT
   before its result is read.
4. Before/after at one thread AND at the shipped thread count, arms interleaved,
   `n >= 3`, with a same-arm control leg and the box's load stated.
5. `scripts/agent-preflight.sh`.

## Risks/decisions

- **The devbox is not idle and cannot be made idle.** Three other agents compiled
  throughout; loadavg ran 63 to 146. Every comparison is INTERLEAVED with a
  same-arm control leg, and the control's own spread is quoted beside every
  ratio. A gap inside its control is not a result.
- **No CMake build tree fits, and a bare `ninja` writes 9.4 GiB with three other
  agents already compiling.** Every binary here is a direct `g++` over the `vt`
  TU set at the project's own flags, `-j 4` at most. `-ffp-contract=off` is not
  optional: the bit-identity contracts depend on it.
- **A mutation that does not build reads as a passing test.** The harness asserts
  the edit changed the file AND that the object compiled and linked before it
  reads a result; its first run reported eight BUILD-FAILED rather than eight
  passes, which is the behaviour that makes the later results readable.
- **This is x86-64 AVX-512 only.** The predecessor's own single-thread claim
  inverted between x86 and aarch64, so nothing here may be read as a GB10
  statement. `## Owed`.

## Stop conditions

- a kernel that cannot be hoisted bit-exactly — report it, never trade correctness;
- ENOSPC, or a build that fails for memory;
- a same-arm control that swallows the effect being claimed.

## Now

`ACTIVE`. W1, W2 and W3 are complete on x86-64. The aarch64 half and the
unhoisted remainder of the ranked set are owed and named in `## Owed`.
