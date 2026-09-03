# QUANT-EXL3-PERF — the EXL3 decode-throughput arm set, and the envelope that decides it

Row: `QUANT-EXL3-PERF`
Issues: [#2570](https://github.com/mudler/vllm.cpp/issues/2570) (primary)
Base SHA: `3d045ba1b`
Parent row: [`QUANT-EXL3`](quant-exl3-shared.md)
Sibling row: [`QUANT-EXL3-MUL1`](quant-exl3-mul1.md) — that row ports the FORMAT,
this one owns what it COSTS.
Matrix: [`.agents/quantization-matrix.md`](../quantization-matrix.md)

Upstream pin: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` — **vLLM implements
no EXL3** at the pin, so the kernels are mirrored from the registered secondary
oracle [`exllamav3`](../oracles/exllamav3.md) @
`2398c05635fbbad01a0a51dce63c85c6c8a8450e` (v1.4.3, MIT). The seam is vLLM's;
exllamav3 supplies the trellis kernels only.

## Now

`ACTIVE`. This row exists because no row owned EXL3 PERFORMANCE. `QUANT-EXL3`
owns the format and `QUANT-EXL3-MUL1` owns the `mul1` widths; both are
correctness rows and both say so. #2570 named a throughput gap with no owner,
and an unowned gap is one nobody reruns.

Slice A (the `(3, 2)` GEMV instantiation) is described below. Everything else is
under `## Owed`, itemised, with the reason it is not closed here.

**No throughput number is claimed by this row yet.** The arm is instantiated and
its numeric gate and A/B are QUEUED on `dgx:gpu0` behind other work. Until that
lease runs, every device claim here is PENDING and is reported as PENDING; a
queued job nobody could gate is a partial result and never a pass.

## The gap, as #2570 states it

Our `m <= 8` EXL3 GEMV instantiated exactly one arm, `(bits = 3, cb = 1)`.
`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` — the #2495 benchmark checkpoint — contains
**zero** tensors of that arm, so the small-m fast path was unreachable on the
model the benchmark is about, one arm at a time and silently, because a declined
GEMV falls through to the regular block-pipelined kernel rather than refusing.

Upstream instantiates seven pairs at the pin
(`exllamav3/exllamav3_ext/quant/exl3_gemv.cu:83-86`):

```c
SEL_GRID(4, 0, false) SEL_GRID(4, 1, false) SEL_GRID(4, 2, false)
SEL_GRID(2, 1, false) SEL_GRID(2, 2, false) SEL_GRID(2, 1, true) SEL_GRID(2, 2, true)
SEL_GRID(3, 1, false) SEL_GRID(3, 2, false) SEL_GRID(3, 1, true) SEL_GRID(3, 2, true)
```

## The census, recomputed for this row rather than inherited

Read from the LOCAL safetensors headers of
`/mnt/nas_share/rc/qwen38-exl3-bench/ckpt/target-3.5bpw`, both shards, on
2026-09-02. 2426 tensors, of which **409 are `.trellis`**, **409 carry a `mul1`
marker and zero carry `mcg`** — so every quantized linear in the artifact is
codebook 2, and `LinearEXL3`'s presence rule (`exl3.py:74-77`) has no other
answer for it.

Bit widths are read from the trellis shape's third axis (`16 * bits`), and every
shape is 128-aligned on both `k` and `n`, so none is excluded by the GEMV's
`size_k % 128 || size_n % 128` hard test:

| bits | k | n | modules | which |
|---:|---:|---:|---:|---|
| 3 | 5120 | 17408 | 92 | `mlp.gate_proj`, `mlp.up_proj` |
| 3 | 17408 | 5120 | 45 | `mlp.down_proj` |
| 4 | 5120 | 1024 | 34 | narrow attention/GDN projections |
| 4 | 5120 | 6144 | 48 | |
| 4 | 5120 | 10240 | 48 | |
| 4 | 5120 | 12288 | 17 | |
| 4 | 5120 | 17408 | 38 | |
| 4 | 6144 | 5120 | 64 | |
| 4 | 10240 | 5120 | 1 | |
| 4 | 17408 | 5120 | 20 | |
| 5 | 6144 | 5120 | 1 | |
| 6 | 5120 | 248320 | 1 | `lm_head` |

Totals `{3: 137, 4: 270, 5: 1, 6: 1}` = 409. This reproduces the count
`QUANT-EXL3-MUL1` slice F recorded, from the artifact and not from a summary of
it, which is the discipline that row's own lesson demands.

## THE INSTANTIATION IS NOT THE ONLY GATE, AND ON GB10 IT IS NOT THE BINDING ONE

This is the finding that shapes the whole row, and #2570 does not contain it.

`Exl3GemvSelectConfig` (`src/vt/exl3_policy.cpp:154-179`) is upstream's
`exl3_gemv_cfg` (`exl3_gemv.cu:46-72`) verbatim, and it returns `-1` to DECLINE.
Read it at the shapes above, on Blackwell (GB10 is `Exl3Cc::kBlackwell`), at the
default `mode == 1`:

```
if (K == 2)                      -> config chosen        (no bits-2 tensor here)
if (K == 3 && cc == kAda)        -> config chosen        (GB10 is NOT Ada)
if (size_n / 32 <= narrow_coresident) -> 0
if (size_k <= 2048 && size_n <= 8192) -> 0               (min k here is 5120)
if (K == 3)                      -> -1                   <-- every bits-3 shape
if (size_n >= 8192 && size_k <= 4096) -> 1               (min k here is 5120)
if (... && cc == kAmpere)        -> 1                     (GB10 is NOT Ampere)
                                 -> -1
```

`narrow_coresident` is `GemvOccupancy(narrow kernel, 512 threads) * num_sms`,
the only term a device supplies. Every other term is fixed by the shape. So on
GB10 the arm is admitted at the default mode **only** where
`n / 32 <= blocks_per_sm * num_sms`:

| bits | k | n | n/32 | admitted at mode 1? |
|---:|---:|---:|---:|---|
| 3 | 5120 | 17408 | 544 | only if `narrow_coresident >= 544` |
| 3 | 17408 | 5120 | 160 | only if `narrow_coresident >= 160` |
| 4 | 5120 | 1024 | 32 | almost certainly YES |
| 4 | 6144 | 5120 | 160 | only if `narrow_coresident >= 160` |
| 4 | 5120 | 6144 | 192 | only if `narrow_coresident >= 192` |
| 4 | 5120 | 10240 | 320 | only if `narrow_coresident >= 320` |
| 4 | 5120 | 12288 | 384 | only if `narrow_coresident >= 384` |
| 4 | 5120 | 17408 | 544 | only if `narrow_coresident >= 544` |

An occupancy of one or two 512-thread blocks per SM on a device with a few dozen
SMs puts `narrow_coresident` in the low hundreds. **The instantiation is
therefore necessary and possibly not sufficient**, and a measurement that only
compares "arm present" against "arm absent" at the default mode cannot tell the
two apart: a zero would be indistinguishable from an envelope that never
admitted the arm.

That confusion has already cost this tree one published number. The comment at
`cuda_exl3.cu:2086` records a `VT_EXL3_GEMV=1` vs `=0` A/B reported as an 8%
GEMV effect **when neither arm could take the GEMV at all** — it ran the same
path twice. This row does not repeat it.

So the measurement carries THREE legs, not two:

- `VT_EXL3_GEMV=0` — the arm is off. On this checkpoint this is byte-identical
  in behaviour to the pre-change binary, because the pre-change predicate
  admitted only `(3, 1)` and the artifact has zero `(3, 1)` tensors. That
  equivalence is asserted by a fourth leg on the PRE-CHANGE binary, not assumed.
- `VT_EXL3_GEMV=1` — the default. Upstream's measured envelope decides, and this
  is the only leg that is a production claim.
- `VT_EXL3_GEMV=2` — upstream's own "use wherever the hard constraints allow"
  testing mode (`exl3_gemv.cu:22`). This leg separates "the envelope declined"
  from "the arm ran and did not help". It is a DIAGNOSTIC and is never reported
  as a production number.

`narrow_coresident` itself is printed from the device, so the table above stops
being a prediction.

**WHAT SEPARATES G0 FROM G1 WHEN THE ENVELOPE DECLINES, exactly.** Read
`Exl3GemvTryLaunch` in order: `force_gemv == 0`, then the arm predicate, then
`Exl3GemvHardEligible`, then the mode, then `GemvKernel`, then `GemvOccupancy`,
then `Exl3GemvSelectConfig`. `VT_EXL3_GEMV=0` returns at the MODE test, which is
before the occupancy query. So if the envelope declines at mode 1, the whole
measured difference between G0 and G1 is one
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` per (device, kernel) pair,
memoised in a static map. That is why `G1 == G0` would be the EXPECTED reading
under a declining envelope rather than a surprise, and why G2 is needed to say
anything about the kernel itself.

The same order is what makes the pre-change binary and `VT_EXL3_GEMV=0`
equivalent on this checkpoint: the old predicate returned false one test EARLIER
than the mode test does, and neither reaches a device call. Both fall through to
`exl3_gemm_kernel` with identical arguments.

## Scope

**In scope, slice A:** instantiate the GEMV at `(bits = 3, cb = 2)`; extend the
device numeric case to gate both codebooks at tier 3c; measure the end-to-end
decode effect on the real checkpoint on `dgx:gpu0`; record the envelope's actual
verdict per shape.

**Out of scope and owed, not silently dropped:** the `(4, 2)` kernel port, the
fused MoE arm, `kExl3HadR` on ROCm, the four-shape coverage of the regular
kernel's shape table. Each is under `## Owed` with its reason.

## Upstream chain

vLLM implements no EXL3 at the parity pin, so the chain below is the secondary
oracle's, read at `2398c05635fbbad01a0a51dce63c85c6c8a8450e` and cited by
`file:line` as `.agents/porting.md` requires.

| Upstream path | What it defines | Where it lands here |
|---|---|---|
| `exllamav3_ext/quant/exl3_gemv.cu:29-42` | the `EXL3_GEMV` and `EXL3_GEMV_SMEM` knobs | `Exl3GemvParseMode` / `Exl3GemvParseSmemMode`, `src/vt/exl3_policy.cpp` |
| `exllamav3_ext/quant/exl3_gemv.cu:46-72` | `exl3_gemv_cfg`, the narrow/wide/decline envelope | `Exl3GemvSelectConfig`, `src/vt/exl3_policy.cpp:154-179`, verbatim |
| `exllamav3_ext/quant/exl3_gemv.cu:83-86` | `SEL_GRID`, the instantiated `(bits, cb)` arms | `Exl3GemvArmInstantiated` / `GemvKernel`, `src/vt/cuda/cuda_exl3.cu` |
| `exllamav3_ext/quant/exl3_gemv.cu:108-114` | the hard eligibility tests, in upstream's order | `Exl3GemvHardEligible`, `src/vt/exl3_policy.cpp:141-152` |
| `exllamav3_ext/quant/exl3_gemv.cu:171-241` | the direct entry point that ERRORS on an ineligible call | `Exl3GemmArgs::force_gemv`, used by the device gate |
| `exllamav3_ext/quant/exl3_gemv_kernel.cuh:31` | `EXL3_GEMV_MAX_M` | `vt::kExl3GemvMaxM` / `kExl3GemvMaxMDev`, asserted equal |
| `exllamav3_ext/quant/exl3_gemv_kernel.cuh:37-52` | the fp16-accumulate `mma.m16n8k16` fragment | `ptx_mma_ab_h`, `src/vt/cuda/cuda_exl3.cu` |
| `exllamav3_ext/quant/exl3_gemv_kernel.cuh:120-134` | the register-form `dq8` with per-lane funnel alignment | `dq8_regs_3bits<cb>` |
| `exllamav3_ext/quant/exl3_gemv_kernel.cuh:138-402` | the kernel, both configs, both m-modes | `exl3_gemv_kernel`, narrowed to `bits == 3` |
| `exllamav3_ext/quant/codebook.cuh:56-90` | `decode_3inst<cb>` for all three codebooks | `decode_3inst_2<cb>`, landed by `QUANT-EXL3-MUL1` slice A |
| `exllamav3_ext/quant/exl3_gemm.cu:220-236` | a DECLINED GEMV falls through to the shape table | `Exl3GemmKernelCuda`'s `TryGemv` call site |

The chain that EXECUTES on a decode step is: the model's linear method reaches
`vt::Exl3Gemm` through `ModelRegistry::Forward`; the CUDA arm calls
`Exl3GemvTryLaunch`; that reads the arm predicate, then `Exl3GemvHardEligible`,
then the env mode, then an occupancy query, then `Exl3GemvSelectConfig`; a
non-negative config launches `exl3_gemv_kernel` cooperatively and a `-1` falls
through to `exl3_gemm_kernel`. Every one of those is a place the arm can be lost,
which is why the measurement observes the kernel rather than inferring it.

## Our baseline

Measured on `dgx:gpu0` (GB10, `sm_121a`, driver 580.173.02) under an `rc` lease,
prompt `The capital of France is`, 64 tokens, greedy, `--repeat 5` with run 1
discarded as cold, `VT_DFLASH_PAGED=0` (#2274):

| arm | decode tok/s | spread |
|---|---|---|
| target only | 16.670 – 16.796 | 0.75%, two interleaved legs |
| target + DFlash2 draft, k = 7 | 48.446 – 49.079 | two interleaved legs |

Reproduced by this row's own job before anything is compared, on the same
binary, because a sequential A/B on this box measures drift as much as it
measures a change: one unchanged binary has read 36.82 and 78.86 tok/s in a
single session here.

The GEMV's own baseline is that it NEVER RUNS on this checkpoint. Every one of
its 409 trellis modules is codebook 2, and the arm predicate admitted only
`(3, 1)`.

## Port map

| Item | Upstream | Here | Status |
|---|---|---|---|
| `(3, 1)` GEMV arm | `SEL_GRID(3, 1, *)` | `GemvKernelForArm<3, 1>` | landed, `MODEL-DSV4-EXL3` W2c |
| `(3, 2)` GEMV arm | `SEL_GRID(3, 2, false)` | `GemvKernelForArm<3, 2>` | **this row, slice A** |
| `(4, 2)` GEMV arm | `SEL_GRID(4, 2, false)` | — | OWED, kernel port |
| `(4, 0)`, `(4, 1)` | `SEL_GRID(4, 0/1, false)` | — | OWED with `(4, 2)`; same kernel work |
| `(2, 1)`, `(2, 2)` | `SEL_GRID(2, *, *)` | — | no 2-bit artifact has reached this tree |
| `(3, 1)`, `(3, 2)` smem-staged | `SEL_GRID(3, *, true)` | `SMEM_STAGE` template arm, both instantiated | landed |
| the envelope | `exl3_gemv_cfg` | `Exl3GemvSelectConfig` | landed, verbatim, per-K so `(3, 2)` inherits `(3, 1)`'s |
| the hard tests | `exl3_gemv.cu:110-114` | `Exl3GemvHardEligible` | landed, upstream's order |

## Tests to port

Upstream ships no C++ unit test for `exl3_gemv`; it exercises the path from
Python through `exl3_gemv` and `exl3_gemm` with `EXL3_GEMV` set, and its
correctness reference is the same linear evaluated by the regular kernel. The
adaptation is recorded rather than hidden: the envelope is extracted as a pure
function and gated directly, which upstream cannot do because its copy is
`static` inside the `.cu`, and the kernel is gated against this tree's CPU arm,
which `test_exl3_gemm` already gates against an f64 chain built from
DEFINITIONS. That keeps ONE reference for both device arms instead of two that
must agree.

What is preserved from upstream: `EXL3_GEMV_MAX_M`, the mode values and their
`atoi` parsing including the unset defaults, every branch of the envelope, the
`K != 4 && cb == 0` refusal, the 128-alignment tests, and the `force` semantics
of the direct entry point.

## Dependencies

- `QUANT-EXL3` (#2181) — the format, the CPU reference, `had_r_128`, and the
  tier-3c bound this row inherits rather than restates.
- `QUANT-EXL3-MUL1` (#2495) — `decode_3inst_2<2>` and `decode_mul1_product_2`.
  Without codebook 2 in the shared decoder there is no `(3, 2)` arm to
  instantiate; this row adds no decode of its own.
- `.agents/oracles/exllamav3.md` — the pin. Advancing it re-opens every
  `file:line` above.
- An `rc` lease on `dgx:gpu0` for anything device-shaped. There is no local CUDA
  toolchain, so every device claim in this row is PENDING until the lease runs.
- #2274 (`VT_DFLASH_PAGED`) — the workaround the baseline carries, applied to
  every leg alike so it cancels in the ratio.

## Risks/decisions

- **D1. The A/B is env-driven on ONE binary, not two binaries.** Decided,
  because on this checkpoint `VT_EXL3_GEMV=0` and the pre-change binary take the
  same code path: the old predicate admitted only `(3, 1)` and the artifact has
  zero `(3, 1)` tensors. That removes binary drift from the comparison. The
  equivalence is not assumed — a mutation that takes `(3, 2)` back out of the
  predicate rebuilds the pre-change product behaviour and is measured.
- **D2. Three legs, not two.** `VT_EXL3_GEMV=2` is upstream's testing mode and
  is carried as a DIAGNOSTIC so an envelope decline stays distinguishable from a
  null effect. It is never reported as a production number. R: quoting it as one
  would be the 8%-from-nothing failure again.
- **D3. The tier is 3c and it is inherited.** R: a `(3, 2)` arm that misses
  `6.0e-3` is wrong; widening the bound to admit it would delete the only check
  that can see a mis-threaded codebook.
- **R1. The envelope may decline every bits-3 shape on GB10.** Then the arm is
  correct, upstream-faithful and worth nothing on this device, and the next
  hypothesis is the narrow config's occupancy. Recorded in `## Owed` as the
  outcome it would be, not as a failure of the row.
- **R2. The fat build compiles one more kernel set.** `(3, 2)` is 16 more
  kernels in a translation unit the fat build compiles for ten architectures.
  Upstream answers this with a per-`K` compilation-unit split
  (`comp_units/exl3_comp_unit_K_cbX.cu`); this tree has one unit and the split
  stays owed by `QUANT-EXL3-MUL1`, which argued it first.
- **D4. The `QUANT-*EXL3*` sibling ratchet is widened by NAME, not by
  predicate.** `tests/scripts/test_agent_record.py` pins the exact set of
  `QUANT-` rows carrying `EXL3`, and it went RED on this row before its argument
  was written, which is the gate working. The argument is that this row sits on
  a different AXIS rather than being a third scheme: the two siblings answer
  "does this width RUN" and this one answers "what does it COST", and
  `QUANT-EXL3-MUL1`'s own claim file EXCLUDES the GEMV by name, so this row is
  the owner that exclusion implies. A FOURTH row still fails there and must
  argue for itself.
- **D5. `docs/FEATURES.md` is edited, `docs/BENCHMARKS.md` is not.** The GEMV
  arm set is a quantization surface and its row said the mul1 GEMV was owed, so
  that sentence is now false and is repaired. No benchmark ID is added, because
  no number exists yet; the cell says the measurement is PENDING rather than
  omitting it.
- **R3. Measuring on a box that crashes.** `dgx:gpu0` has crashed roughly hourly
  under long ladders. The job prints results incrementally and orders the A/B
  ahead of the mutations, so a crash costs the cheapest evidence rather than the
  most expensive.

## Why `(3, 2)` is an instantiation and `(4, 2)` is a port

`exl3_gemv_kernel` in `src/vt/cuda/cuda_exl3.cu` is already
`template <int bits, bool c_fp32, int cb, int MMODE, int CFG, bool SMEM_STAGE>`
and threads `cb` all the way down: the only decode site in it is
`dq8_regs_3bits<cb>`, which calls `decode_3inst_2<cb>`, and that function has
carried all three codebooks since `QUANT-EXL3-MUL1` slice A. `GemvKernelForArm`
is already `template <int BITS, int CB>`. Nothing about the kernel's geometry
depends on `cb`: `LSTRIDE`, `TWORDS`, `FOLD`, `PF` and `LOADS` are all functions
of `bits`, `CFG` and `MMODE`.

So `(3, 2)` changes ONE template argument and ONE predicate. It covers 137 of
the 409 modules.

`(4, 2)` does not. `LSTRIDE` is the literal `24` with the comment
`// uint32 per load, bits == 3`, and `TWORDS` is `8 * bits`, so at bits 3 the
two are equal and one warp load covers one whole 16x16 tile. At bits 4
`TWORDS == 32` and they are not equal; the prefetch ring depth `PF`, the fold
cadence `FOLD` and the load count `LOADS` are all tuned around that equality,
and no `dq8_regs_4bits` register extractor exists in this tree. That is a kernel
port from `exl3_gemv_kernel.cuh`, and it is not attempted here.

## The envelope is ALREADY ported, and that matters

#2570 asks that any port bring `exl3_gemv.cu:55-71` with it. It is already here,
verbatim, including the `K == 4` wide-config admission and the commented-out
`cc != CC_AMPERE` guard that upstream disabled
(`src/vt/exl3_policy.cpp:154-179`), and `tests/vt/test_exl3_gemv.cpp:101-122`
gates its branches on any machine. Slice A adds no envelope change, because
there is none to add: upstream's envelope is per-`K`, not per-`cb`, so `(3, 2)`
inherits `(3, 1)`'s exactly.

## The numeric tier is 3c, and it is NOT widened

The GEMV accumulates in `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` and
folds to an f32 pair only every `FOLD` iterations, so an fp16 accumulator
absorbs up to `FOLD * 16` k-elements. `QUANT-EXL3`'s `## W2cd design` W2c-3 sets
its bound at tier 3c, relative RMS `6.0e-3`, rather than tier 3's `1.0e-3`. A
new arm INHERITS that bound. If `(3, 2)` cannot meet `6.0e-3` the arm is wrong;
the bound does not move.

`(3, 1)` and `(3, 2)` are the confusable pair and the reason the device case
gates both. Both are three bits wide, both take the same `dq8` route, both use
the same tile shapes. A `cb` threaded wrongly between them does not fail to
compile and does not change a shape: it decodes with the other codebook's tail
and yields a weight with the right DISTRIBUTION and no correlation to the true
one. The mutation table below makes that failure red.

## Work breakdown

- **A1.** Extend `tests/vt/test_exl3_gemv.cpp`'s device case over `cb` in
  `{1, 2}`, referenced against the CPU arm at the same `cb`. RED first: `(3, 2)`
  declines the arm before A2, so the forced call throws.
- **A2.** Add `(3, 2)` to `Exl3GemvArmInstantiated` and `GemvKernel`, and
  correct the comment block above them, which asserts the arm set is `(3, 1)`
  only.
- **A3.** Add a machine-independent case that reads the ENVELOPE at this
  checkpoint's real shapes, so the table in this spec is executable rather than
  prose.
- **A4.** Measure on `dgx:gpu0` under an `rc` lease: the four legs above,
  interleaved, one binary per arm, `narrow_coresident` printed.

## Tests

- `tests/vt/test_exl3_gemv.cpp` — the envelope cases (any machine) and the
  device tier-3c case (CUDA only, skips loudly and still asserts).
- The device case is FORCED through `Exl3GemmArgs::force_gemv`, mirroring
  upstream's direct entry point (`exl3_gemv.cu:171-241`). Forcing is what makes
  it a gate rather than a coin flip on an occupancy query: without it a device
  whose envelope declines the shape measures the REGULAR kernel and reports
  tier 3c green.

## Gates

```sh
ctest --test-dir build -R '^test_exl3_gemv$' --output-on-failure
ctest --test-dir build -R '^test_exl3_gemm$' --output-on-failure
scripts/agent-preflight.sh --staged
```

The device arm additionally needs an `rc` lease and a CUDA build:

```sh
rc run --device dgx:gpu0 -- ctest --test-dir build-cuda -R '^test_exl3_gemv$' -V
```

A CPU-only green is not a device result and is never reported as one. A doctest
`assertions: 0` line is a skip wearing a pass; read the ctest exit code.

## Owed

- **`(4, 2)`, the 270-module arm.** The largest single population in the
  checkpoint, and the one #2570 leads with. It is a kernel port for the reason
  under "Why `(3, 2)` is an instantiation" above, and it is not attempted in
  this flow. Tracked by [#2570](https://github.com/mudler/vllm.cpp/issues/2570),
  which stays OPEN for it.
- **The envelope may decline `(3, 2)` at every shape this checkpoint has, on
  GB10.** If it does, the instantiation is correct, upstream-faithful, and worth
  zero end to end on this device — and the next traceable hypothesis is the
  occupancy of the narrow config, which is the only device-supplied term in the
  admission test. That is a measurement, not a ceiling.
- **The fused MoE arm is `(3, 1)` only** (`kMoeBits`/`kMoeCb` in
  `cuda_exl3.cu`), and it is CUDA-only. The #2495 checkpoint is dense so nothing
  here reaches it, and it is named so the next MoE EXL3 artifact does not
  rediscover it.
- **`kExl3HadR` on ROCm runs the portable CPU reference tier, not a native
  kernel.** It is correct and it is not fast, and no row owned that either.
- **The regular kernel's shape table is gated at ONE of its four shapes.**
  `Exl3SelectGemmShape` picks shape 2 at the tested dimensions and shape 4 is
  unreachable at `n = 256`. `force_shape_idx` already exists, so this is a test
  loop rather than a port.
- **`bits` 5 and 6 have no GEMV upstream either** (`exl3_gemv.cu:110-111`
  refuses `K < 2 || K > 4`), so the one 5-bit tensor and the 6-bit `lm_head`
  falling to the regular kernel is upstream's behaviour and not a gap. Recorded
  so it is not re-filed.
- **`docs/USAGE.md` owes this checkpoint's file names, sizes, repo and
  REVISION.** Owed by `QUANT-EXL3-MUL1`, which loads it; repeated here because
  this row measures it.

## Stop conditions

- `(3, 2)` and `(3, 1)` agree elementwise on the device case → the case is not
  discriminating; the two codebooks must produce different numbers at the same
  width. Stop and fix the fixture before reading any tolerance.
- The tier-3c bound fails → the arm is wrong. Never widen the bound.
- The `(4, 2)` port needs a kernel structure that is not upstream's → return
  `NEEDS_DECISION` rather than inventing one.
- The lease never arrives → report `(3, 2)` as instantiated-and-ungated, and say
  so. A queued job nobody could gate is a partial result, never a pass.
- An A/B leg shows the GEMV arm never ran → the measurement is void, not a zero.
  Print the envelope's verdict before reading any tok/s.
