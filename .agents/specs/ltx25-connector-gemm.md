# SPEC — `LTX25-CONNECTOR-GEMM`: what the connector's 224.9 s of host f32 GEMM is actually spent on

Issue: **NONE — `REMOTE_UNVERIFIED`.** The `mudler-agent` GitHub account returns
`Your account is suspended` (HTTP 403) on `gh api user`, so no issue could be
opened for this row. The work it takes up is the first `## Owed` item of
[`.agents/specs/ltx25-text-cond-device.md`](ltx25-text-cond-device.md), which
records the same block and the same reason:

> **THE COMPUTE LEVER IS UNOWNED, and it is 43.52% of the render.** 224.882 s of
> host f32 GEMM at ~37 GFLOP/s [...] **No issue was filed for this: the GitHub
> account was suspended mid-row.**

Owner row: `LTX25-CONNECTOR-GEMM`. Predecessors:
[#2296](https://github.com/mudler/vllm.cpp/issues/2296) (the 5.53x reading) and
[#2354](https://github.com/mudler/vllm.cpp/issues/2354) (the weights/compute
split). An issue is owed the moment the account is restored.

## Scope

`LTX25-TEXT-COND-DEVICE` split `conditioning.connector` and found that the time
is **arithmetic, not weight materialization**: connector compute is 224.882 s
over both guidance passes, 43.52% of the render, against 13.707 s for all four
materializations. It named two traceable next steps and moved neither. This row
takes the first one.

IN scope:

- **W1** — establish **BY EXECUTION** which CPU GEMM kernel the connector's
  `vt::MatmulBT` calls actually enter, on both architectures that matter: this
  x86-64 devbox and the GB10 aarch64 cores the 37 GFLOP/s was measured on.
- **W2** — measure the **achievable rate** for the connector's exact GEMM shapes
  in each orientation the tree can express: `MatmulChunked<true>` (what runs),
  `MatmulChunked<false>` over a `[K,N]`-repacked weight (what
  `b.elem_kn_repacked` unlocks), and the historical reference tile
  (`VT_CPU_MATMUL_TIER=ref`).
- **W3** — decompose `Ltx2ConnectorForward` itself, so the claim "the connector
  is GEMM" is measured rather than assumed.
- **W4** — a repair **only if** it is bit-exact and the measurement supports it;
  otherwise the attribution and the next hypothesis.

OUT of scope, declared rather than approximated:

- **A device arm for the connector.** `Ltx2Attention` interleaves host
  `RmsNormRows` and `Ltx2ApplyRotaryEmb` on raw `float*` between its GEMMs and
  `Ltx2ConnectorForward` reads its weights as host `std::vector<float>`, so that
  is a weight-arm port with its own numerics gate, not a queue swap. #2354 says
  so and this row does not contradict it.
- **A published benchmark ID.** `docs/BENCHMARKS.md` gains nothing. A probe of
  one module's GEMM shapes is an instrument, not a benchmark.
- **Changing `vt`'s numeric contract.** Every kernel in `cpu_matmul_elem.cpp`
  keeps each output's K reduction strictly sequential, which is what makes the
  SIMD tiers bit-identical to the scalar reference. Nothing here may weaken it.

## The two predicates, and why reading them is not the deliverable

The brief this row was dispatched with named two static questions. Both are
answerable by reading and **both were read before this spec was written**, so
they are recorded here as inputs rather than as findings:

| question | answer | evidence |
|---|---|---|
| does f32 have an `ElemKind`? | YES | `src/vt/cpu/cpu_matmul_elem.cpp`, `ElemKindOf`, `case DType::kF32` |
| so does the dtype force `MatmulOneChunkRef`? | NO | the `!ElemKindOf(...)` disjunct at `cpu_ops.cpp:122` is false for f32 |
| is the connector's weight repack-eligible? | YES | `ElemRepackEligible` returns true for f32 with `n, k > 0` |
| does anything repack it? | NO | the only `ElemRepackWeight` caller and the only `elem_kn_repacked = true` assignment are both in `qwen3_5_gguf_weights.cpp` |

**So the connector takes `MatmulChunked<true>`, the BT orientation, through
whatever tier the process resolved.** That is a reading, and this repository has
been wrong before about what a predicate implies at runtime. Two things are still
unknown after it and neither is readable:

1. **Which tier resolves on the GB10's cores.** If `ElemGemmTier()` names `"ref"`
   there — from an environment variable, from a failed feature probe, from a
   build that never compiled the NEON TU — the reference tile is running and
   every sentence above is moot.
2. **What the BT orientation costs.** `MatmulChunked<true>` is not a materialized
   transpose; it is a different read pattern with different locality and, on the
   SIMD tiers, a 4x4 register transpose per weight group. Whether that is 10% or
   3x is a measurement, and `.agents/specs/kernel-gemm-cpu-tiled.md`'s lever 2
   exists precisely because nobody has measured it at these shapes.

## W1/W2/W3 — the instrument

`tools/bench/ltx2_connector_gemm_probe.cpp`, in the shape
`tools/bench/conv1d_scaling_probe.cpp` and `tools/bench/bpe_encode_cost.cpp`
already establish: a probe the build compiles, that CI runs never, and whose own
header carries the run recipe. It has three modes.

- `--mode tier` prints `vt::cpu::ElemGemmTierName()` and the resolved
  `mr`/function-pointer table. This is the **resolver's own output at runtime**,
  which is what W1 asks for and what reading `BuildTier()` cannot give.
- `--mode gemm` runs `vt::MatmulBT` at the connector's exact shapes — `M = 1024`
  and the `(N, K)` pairs the checkpoint's own config implies — in three
  orientations: as shipped, over a `[K,N]`-repacked weight with
  `elem_kn_repacked` set, and under `VT_CPU_MATMUL_TIER=ref`. It reports seconds
  and GFLOP/s per shape and **asserts every arm bit-identical to the reference
  arm**, so a rate is never reported for a kernel that computed something else.
- `--mode connector` runs `Ltx2ConnectorForward` itself at the shipped geometry
  with a configurable layer count, so W3's decomposition is taken on the product
  function rather than on a model of it.

**The shapes are read out of the checkpoint, not assumed.** The
`__metadata__.config` of
`ltx-2.5-22b-dev-transformer-bf16.safetensors` (the manifest-pinned DiT) gives
`connector_num_attention_heads = 32`, `connector_attention_head_dim = 128`,
`connector_num_layers = 8`, `audio_connector_attention_head_dim = 64`,
`connector_apply_gated_attention = true`, `connector_num_learnable_registers = 128`,
`rope_type = split`, `frequencies_precision = float64`. So the video stream is
`inner_dim = 4096` and the audio stream `inner_dim = 2048`, both 8 layers, and
the per-call GEMM total is `12 * dim^2 * rows` per layer:

```
video  8 * 12 * 4096^2 * 1024 = 1.6492e12 MAC
audio  8 * 12 * 2048^2 * 1024 = 0.4123e12 MAC
total                           2.0615e12 MAC = 4.123 TFLOP
```

which is the 4.2 TFLOP #2354 predicted in advance, now read off the checkpoint
instead of estimated.

**The symbol-level proof is `perf`, not a counter.** A counter added to
`MatmulOneChunk` would be a product-code change made to answer a question about
product code, and it would only report the sites it was added to. `perf record -e
cpu-clock` over the probe names the executed symbols — `Bt16*`, `BtM*`, `Nk*`,
`MatmulOneChunkRef`, `AttentionCrossKernel` — and needs no PMU, so it runs inside
a VM guest and inside a lease.

**No full CMake build is possible on this devbox.** `/` is at 99% with 4.7 GB
free and a bare `ninja` writes 9.4 GiB, which is the ENOSPC that has previously
produced FALSE policy refusals in this tree's records. The probe therefore also
carries a **direct `g++` recipe over its own translation-unit set** in its
header, which is the `bpe_encode_cost.cpp` precedent verbatim. The CMake target
exists so the file cannot rot behind a `vt::MatmulBT` or `Ltx2ConnectorForward`
signature change; the recorded runs are taken from the direct compile.

## Tests to port

There is no upstream test. Upstream's connector is a `torch.nn.Module` and its
GEMM is cuBLAS. The tests are this tree's own and each is an executable
observable:

| ID | Assertion | Red before |
|---|---|---|
| T1 | `vt::MatmulBT` over a `[K,N]`-repacked weight with `elem_kn_repacked` set is **byte-identical** to the same GEMM over the un-repacked `[N,K]` weight, at the connector's own shapes | nothing asserts it at a non-square `(N != K)` shape |
| T2 | `ElemRepackWeight` followed by the repacked GEMM equals the direct GEMM for f32 at `N != K` and at a K that is not a multiple of the lane count | — |

T1 is the guarantee any repair in W4 would rest on, written as byte equality
rather than a tolerance, because both orientations accumulate each output over K
in strict increasing order and therefore have no tolerance to argue about.
`tests/vt/test_ops_matmul_elem.cpp` already asserts the tier-vs-reference
identity; what it does not assert is the **orientation** identity at a shape the
connector actually uses.

## Gates

1. `--mode tier` on each architecture measured, output recorded verbatim.
2. `--mode gemm`, `n >= 3` per arm, spread stated, every arm asserted
   bit-identical before any rate is quoted.
3. `perf record` symbol attribution over `--mode connector`, recorded as the
   executed proof of W1.
4. Only if W4 lands a change: `ninja test_ops_matmul_elem` green, the mutation
   for each claimed guarantee, and `scripts/ltx25-text-cond-ab.sh`'s
   `pixel_files_differing=0` plus the #1864 blockiness gate under a lease.
5. `scripts/agent-preflight.sh`.

## Risks/decisions

- **This devbox is not the measured machine.** The 37 GFLOP/s was taken on
  `dgx:gpu0`, a GB10 whose CPU is aarch64; this devbox is an AMD Zen 5 with
  AVX-512. The tier tables differ, `mr` differs (4 on NEON, 2 on SSE2, whatever
  AVX2/AVX512 fill), and an absolute GFLOP/s taken here **is not** a statement
  about the GB10. Where this row quotes a rate it names the machine. The x86
  numbers answer "does the orientation matter" and "is the specialized kernel
  entered"; only a run on the GB10 answers "is 37 GFLOP/s what those cores do".
- **A lease is the only way to reach the GB10 and it is for the CPU.** `rc` is
  the mutex; never `ssh`. The probe needs no GPU, so the lease is short and its
  cost is queue wait, not runtime.
- **A microbenchmark is warm and the render is not.** The connector streams 8 GB
  of f32 weights once per call; a probe that loops one shape measures it out of
  L2/L3. Both forms are reported and the warm one is labelled as warm.
  `warm-probe-loops-are-l2-artefacts` is the failure being avoided.
- **`n = 3` bounds a spread; it does not establish a distribution.** A same-arm
  control is run and reported beside every comparison, because a gap smaller than
  the same-arm spread is not a result.

## Evidence

- `--mode tier` output on each machine, verbatim.
- `perf` symbol table over the connector run.
- Per-shape seconds and GFLOP/s for each orientation, `n >= 3`, spread stated,
  with the bit-identity assertion's own output beside them.
- The machine each number was taken on, its core count, and its clock.
- If W4 lands anything: red-before, green-after, the mutation, byte equality and
  the blockiness verdict.

## Stop conditions

Stop and report, do not work around:

- a speedup that cannot be made bit-exact — report it, never trade correctness
  for it (`AGENTS.md` `## Gates`);
- an unhealthy or unreachable fleet device;
- a same-arm spread that swallows the effect being claimed;
- ENOSPC. The disk is at 99% and a build that fills it makes unrelated checkers
  emit false policy refusals.

## Work breakdown

- **W1** — this spec, the probe, the tier proof.
- **W2** — the orientation and tier rate measurements.
- **W3** — the `Ltx2ConnectorForward` decomposition.
- **W4** — the repair, or the measured negative and the next hypothesis.

## Now

`ACTIVE`. W1, W2 and W3 are complete on x86-64 and the aarch64 half is under a
queued lease. W4 is a **measured redirection** rather than a repair, and the
reason is in `## Outcome`.

## Outcome

### The headline, stated before the evidence because it overturns a landed record

**`conditioning.connector.compute` is not dominated by the GEMM.** Two
independent instruments, sharing no code, put `vt::AttentionCross` at 54% to 66%
of `Ltx2ConnectorForward` and the specialized GEMM micro-kernels at 29% to 31%.
The connector's attention performs **4.2% of the layer's arithmetic and takes
66% of its time.**

**So #2354's 37 GFLOP/s is not the GEMM's rate.** That number is
`leaf_seconds / gemm_flops`, which is the GEMM's rate only if the leaf is the
GEMM. On the machine measured here the same construction reads **50.7 GFLOP/s**
for a video layer while the GEMM inside it runs at **153.3 GFLOP/s**. The
agreement between #2354's predicted 34 GFLOP/s and its measured 37 was real and
was a coincidence of construction: both sides divided the whole leaf by the
GEMM's flops, so both were bound to agree whatever else was in the leaf.

`a-number-quoted-often-becomes-treated-as-measured` is the shape, and this row is
where it is caught, one row after it was written.

### W1 — which kernel runs, proved by execution

`--mode tier`, run on the devbox, verbatim:

```
tier_name=avx512
elem_gemm_use_ref=0
mr=6
f32_bt=set f32_nk=set f32_btm=set f32_nkm=set
elem_kind_of_f32=1
repack_eligible_f32_4096x4096=1
```

and the resolver moves when told to, which is what says it was read rather than
printed from a constant: `VT_CPU_MATMUL_TIER=ref` gives `tier_name=ref`,
`elem_gemm_use_ref=1`; `=portable` gives `tier_name=portable`, `mr=4`,
`f32_btm=NULL`.

**The symbol-level proof is a `perf` profile of `Ltx2ConnectorForward` itself**,
not of a model of it. `sudo perf record -e cpu-clock -F 199`, two video layers,
devbox at loadavg 10:

| symbol | self |
|---|---:|
| `LoadF32(Tensor const&, long)` | 23.09% |
| `vt::SizeOf(vt::DType)` | 20.49% |
| `BtM6Avx512<kF32>` | 17.40% |
| `AttentionCrossKernel(...)::{lambda(long, long)#1}` | 10.74% |
| `Bt16Avx512<kF32>` | 6.81% |
| `Transpose16(__m512*)` | 6.75% |
| `Threadpool::Barrier()` | 6.10% |
| `Threadpool::PollForWork(...)` | 4.32% |
| `__memmove_avx512_unaligned_erms` | 1.11% |

**`BtM6Avx512` and `Bt16Avx512` ARE the tier's f32 `[N,K]` entry points, so the
specialized kernel is what executes.** `MatmulOneChunkRef` does not appear
anywhere in the flat profile down to a 0.05% limit; `MatmulOneChunk<true>` shows
0.10%, which is the driver frame whose work is in the two inlined tier calls
above it. The `Transpose16` line is the BT orientation's own 4x4-group register
transpose, which is what says the call took `MatmulChunked<true>` rather than the
repacked branch.

An earlier profile of ONE layer taken while the box was at loadavg 40 gives the
same ranking with the threadpool terms inflated (Barrier 14.74%, PollForWork
2.72%) and everything else within 3 points. Both are recorded because the
difference between them is the contention, not the finding.

### W2 — the rate, and the orientation lever refuted on this architecture

`--mode gemm`, M = 1024, the connector's own `(N, K)` set, arms **interleaved**
with a same-arm control leg, n = 3, devbox at loadavg 15-27 (**not idle, and
that is stated rather than smoothed**):

| | seconds for one `RunConnector` call | rate |
|---|---:|---:|
| `MatmulChunked<true>`, the shipped orientation | **26.924** | **153.3 GFLOP/s** |
| `MatmulChunked<false>` over an `ElemRepackWeight`-ed `[K,N]` weight | 55.435 | 74.4 GFLOP/s |

**The repack is 2.06x SLOWER here, and every shape is byte-identical**, so this
is a layout result and not a numerical one. Per shape the ratio is 1.47x to
1.84x on the four large shapes; the same-arm control ran at 0.96x to 1.17x, so
the effect is far outside its own control's spread. The two `heads x dim`
projections (N = 32) go the other way and are 0.4% of the call's flops.

**This qualifies a record.** `include/vt/quant.h` states the elementwise repack
as "measured 1.16x to 1.30x on dgx and BYTE-IDENTICAL". The byte-identity half
reproduces exactly. The speed half does not generalize: it was measured on
aarch64 at another row's shapes, and on x86-64 AVX-512 at the connector's shapes
the same lever is a 2x regression. The plausible mechanism is that the AVX-512
tier's `Transpose16` costs less than the `[K,N]` path's 16 KB-strided weight
walk, which is exactly the trade that inverts between ISAs. **This row does not
edit that header**, because the aarch64 measurement that would say whether the
sentence needs a scope qualifier or a correction is still queued.

### W3 — the decomposition, and it closes

`--mode connector` and `--mode attn`, same binary, same box, same hour:

| | video (dim 4096) | audio (dim 2048) | x8 layers, both streams |
|---|---:|---:|---:|
| `Ltx2ConnectorForward`, one layer | 8.132 s (spread 12.04%) | 3.366 s (7.18%) | **91.98 s** |
| `vt::AttentionCross` at that layer's shape | 5.361 s (3.07%) | 2.966 s (7.69%) | **66.62 s** |
| the layer's six GEMMs | — | — | **26.92 s** |

**66.62 + 26.92 = 93.54 against 91.98 measured, which closes to 1.7%.** That is
what makes this a decomposition rather than a set of intervals: there is no third
term of any size, and the two named terms are the whole leaf.

**The efficiency gap is the finding.** One video layer's attention is
1.718e10 FLOP against the layer's 4.123e11 of GEMM -- **4.2% of the
arithmetic** -- and it takes **65.9% of the layer**. Measured rates: the GEMM at
**153.3 GFLOP/s**, the attention at **3.2 GFLOP/s**. A 48x gap between two
kernels in the same loop.

### Why the attention kernel is 48x off, and what the repair is

`AttentionCrossKernel` (`src/vt/cpu/cpu_ops.cpp`) reads every operand element
through `LoadF32(const Tensor&, int64_t)`, which switches on `t.dtype` and
computes its byte offset with `vt::SizeOf(t.dtype)`. **`vt::SizeOf` is an
out-of-line function in `src/vt/dtype.cpp` and the build enables no LTO**
(`CMakeLists.txt` sets no `INTERPROCEDURAL_OPTIMIZATION` and passes no `-flto`),
so it is a cross-translation-unit call that cannot be inlined away. The profile
shows the consequence directly: `LoadF32` 23.09% plus `SizeOf` 20.49% is
**43.6% of a connector layer spent resolving an element type and an address**,
inside a loop whose body is one multiply and one add.

**That attribution is proved, not inferred.** A `perf` profile of `--mode attn`,
which runs `vt::AttentionCross` and the probe's hoisted reference and NO GEMM at
all, isolates it:

| symbol | self |
|---|---:|
| `LoadF32(Tensor const&, long)` | 36.14% |
| `vt::SizeOf(vt::DType)` | 28.41% |
| `AttentionCrossKernel(...)::{lambda(long, long)#1}` | 15.60% |
| `Threadpool::Barrier()` | 8.18% |
| `ModeAttn(long, int)` -- the hoisted reference, inlined | **4.76%** |
| `Threadpool::PollForWork(...)` | 4.45% |

**64.6% of the attention kernel's own CPU time is resolving an element type and
an address.** The arithmetic and the softmax are the 15.60% line. And the last
column is the same profile's own control: the hoisted reference computes the
identical output for **4.76%** of the process's CPU against the shipped kernel's
80.15%, so the shipped kernel burns **16.8x the CPU for the same result** --
measured in ONE process, with no cross-run drift to argue about.

**The repair is the transformation `MatmulOneChunk` already applies against
`MatmulOneChunkRef`:** resolve the element type once, outside the loops, and walk
typed pointers. It touches no output's accumulation order -- the same indices are
summed in the same sequence -- so it is **bit-exact, not merely close**.

`--mode attn` prices it. `AttnCrossHoisted` in the probe is that transformation,
written in the probe rather than in product code so the headroom could be
measured before anything was changed:

| | shipped kernel, 20 threads | hoisted reference, ONE thread | equality |
|---|---:|---:|---|
| video, heads 32, d 128 | 5.361 s | **3.775 s** | byte-equal |
| audio, heads 32, d 64 | 2.966 s | **2.161 s** | byte-equal |

**A single thread with the dtype hoisted beats twenty threads without it**, on
both streams, with `memcmp`-identical output. That is the headroom, measured, and
it bounds nothing from above: the hoisted form here is scalar and unthreaded.

### W4 — a redirection, and why no kernel was changed

The repair is obvious, bit-exact, and prototyped byte-equal. **It is not landed
here, and that is a scope decision rather than a lack of one.**

`AttentionCrossKernel` is a `vt` shared seam. Every model that reaches
`vt::AttentionCross` runs it -- the LTX-2.5 DiT among them -- and
`AttentionKernel` beside it carries the identical defect. Doing this properly is
the `CPU-ELEM-GEMM` shape: keep the current scalar kernel as the reference arm,
add a typed one, gate the two byte-identical, and put a same-binary A/B switch
between them. `AGENTS.md` `## Changing the rules or a checker` and `## Spec
before code` both point that work at its own row with its own spec and a fresh
reviewer, and this row's own scope excludes it.

Three further reasons, each measured rather than asserted:

- **This devbox is not the machine the render was measured on.** Every number
  above is x86-64 AVX-512. The GB10's aarch64 tier is NEON with `mr = 4` and a
  weaker GEMM, so the attention/GEMM split there is an open question that the
  queued lease answers and this row does not guess at.
- **The disk cannot hold a CMake build tree** (`/` at 99%), so the project's full
  suite could not be run against a change to a seam every model uses. What COULD
  be run is `tests/vt/test_ops_attention_cross.cpp`, compiled directly:
  **21 cases / 33 assertions / 0 failed** on this tree, which is the baseline the
  next row starts from and is recorded here so it does not have to re-derive it.
- **The existing byte-identity gate for the orientation lever already exists**
  and needed nothing added. `test_ops_matmul_elem.cpp`'s "load-time [N,K]->[K,N]
  repack is byte-identical" covers T1 for three dtypes at five ragged shapes, and
  the probe's own per-shape `memcmp` extends it to the connector's shapes. **No
  new test was written, because the guarantee was already gated** and a second
  copy is how two rules start.

### What could not be measured

- **The aarch64 side.** `rc` jobs `ea2631f3-aed2-46df-b419-3628078f9882`
  (`dgx:gpu0`) and `75800c9e-0b55-406e-9958-ba0048a5a751` (`thor:gpu0`) were
  submitted with the full probe and were still queued at positions 5 and 3 when
  this row was written. `orin:gpu0` was tried first and refused: its `/workspace`
  is local to that host and is NOT the shared NAS the other two mount, which is
  recorded here because it is not written anywhere else.
- **An idle box.** Every devbox number was taken with other agents compiling on
  the same 20 cores, at loadavg 10 to 40. The RATIOS are what this row rests on
  and each carries its own same-arm control; the absolute GFLOP/s are lower
  bounds.
- **A render.** No end-to-end before/after exists, because nothing changed.

## Owed

- **THE ATTENTION KERNEL'S PER-ELEMENT DTYPE SWITCH IS UNOWNED.** 43.6% of a
  connector layer is `LoadF32` plus a cross-TU `vt::SizeOf`, and the fix is
  bit-exact and prototyped. It is a `vt` seam change and needs its own row, its
  own spec, and a fresh reviewer. It is not LTX-2.5-specific: `AttentionKernel`
  carries the same defect and every CPU attention path pays it. Owner: this row,
  until that row exists.
- **`include/vt/quant.h`'s "1.16x to 1.30x on dgx" needs a scope qualifier or a
  correction.** On x86-64 AVX-512 at the connector's shapes the same lever is
  2.06x SLOWER. Which of the two it needs depends on the queued aarch64 run.
  Owner: this row.
- **The aarch64 numbers.** Both leases are submitted and neither had started.
  Owner: this row.
- **No issue could be filed.** `gh api user` returns `Your account is suspended`.
  `REMOTE_UNVERIFIED`; the branch is pushed over SSH and there is no pull
  request. Owner: this row, until the account is restored.
