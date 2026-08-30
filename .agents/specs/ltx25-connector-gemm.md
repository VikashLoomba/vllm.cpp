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

`ACTIVE`. W1 in this change.
