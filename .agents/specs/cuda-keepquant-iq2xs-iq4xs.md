# CUDA keep-quant kernels for IQ2_XS and IQ4_XS — `QUANT-CUDA-IQ4XS-IQ2XS`

Issue: [#2260](https://github.com/mudler/vllm.cpp/issues/2260).
Branch: `row/QUANT-CUDA-IQ4XS-IQ2XS`.
Extends [`cuda-keepquant-gemm.md`](cuda-keepquant-gemm.md), whose
`KERNEL-QUANT-CIQ-GEMM-CUDA` owns the CUDA `kMatmulBTQuant` provider. Discharges
the CUDA half of the `QUANT-GGUF-IQ2_XS` and `QUANT-GGUF-IQ4_XS` rows of
[`quantization-matrix.md`](../quantization-matrix.md), and the premise of O19 in
[`glm5-next-flash.md`](glm5-next-flash.md).

## Now

`ACTIVE`. The CPU arm of both encodings landed with #2247; this row is the
device arm of the same two `vec_dot`s.

## The gap, as measured on `673464ee1`

`src/vt/cuda/cuda_quant_dot.cu::IsCudaKeepQuantSupported` maps ten `DType`s to a
`WType`: `IQ2_XXS`, `IQ3_XXS`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ2_S`,
`IQ1_S`, `IQ1_XXXS`. **`IQ2_XS` and `IQ4_XS` are absent**, and `git grep -n
'kIQ2_XS\|kIQ4_XS' src/vt/cuda/` returns nothing at all.

An absent dtype does not refuse. The three CUDA consumers of that predicate
behave in two different ways, and the difference is the whole reason this row
exists:

| Seam | Today, for these two dtypes |
|---|---|
| `MatmulBTQuantKernelCuda` | `cudaStreamSynchronize`, then the CPU kernel over the same unified tensors. Correct numbers, host speed, and **not capturable** — the sync invalidates a decode graph. |
| `MatmulBTQuantGroupedKernelCuda` | the same drain-and-fall-back. |
| `MoeGateUpSwiGLUGroupedCuda` | **throws** `gate/up must be the SAME CUDA keep-quant dtype`. There is no fallback behind that seam. |

`gguf_keep_quant.cpp::DeviceKeepQuantSupported` returns `true` for every dtype
on CUDA precisely because of the first row of that table, so the loader admits
these tensors as keep-quant blocks and the residency is already right. **This
row does not touch that function**, and the reason is recorded rather than
inferred: narrowing it to the CUDA kernel list would push `Q4_0`, `IQ4_NL` and
`MXFP4` back to `expand_bf16` on CUDA, which is a residency regression on
already-shipped models and no part of #2260.

### What each artifact needs, verified against the tree

- **GLM-5.3-Flash `UD-Q2_K_XL`**: 82 `IQ2_XS` + 3 `IQ4_XS` tensors. Both are
  needed. `docs/FEATURES.md:153` currently instructs the reader to use
  `--device cpu` for exactly this reason, citing #2260.
- **GLM-5.3 non-flash `UD-IQ1_S`**: 4 `IQ4_XS` tensors; `IQ1_S`, `IQ3_XXS`,
  `IQ2_XXS`, `Q2_K` and `Q3_K` are already in `IsCudaKeepQuantSupported`. **So
  `IQ4_XS` alone completes that arm's device admission.**

## Upstream anchors

Oracle: llama.cpp `b10451` = `10bf611e533d81f739128304991c5e133c6aebd8`
([`oracles/llama-cpp.md`](../oracles/llama-cpp.md)). vLLM implements neither
encoding, so the secondary oracle rule applies exactly as it did for the CPU arm.

| Ported | From | To |
|---|---|---|
| `ggml_vec_dot_iq2_xs_q8_K_generic` | `b10451 ggml/src/ggml-cpu/quants.c:948` | `cuda_quant_dot.cu::DotIQ2XS` |
| `ggml_vec_dot_iq4_xs_q8_K_generic` | `b10451 ggml/src/ggml-cpu/quants.c:1283` | `cuda_quant_dot.cu::DotIQ4XS` |
| `iq2xs_grid` (512 u64) | `b10451 ggml/src/ggml-common.h:627` | `cuda_quant_iq_tables.cuh::d_iq2xs_grid` |
| `kvalues_iq4nl` (16 i8) | `b10451 ggml/src/ggml-common.h:1007` | `cuda_quant_iq_tables.cuh::d_kvalues_iq4nl` |

The in-tree behavioural reference is the CPU port of those same two functions:
`src/vt/cpu/cpu_quant_dot.cpp::VecDotIQ2_XSQ8_K` and
`src/vt/cpu/cpu_quant_dot.cpp::VecDotIQ4_XSQ8_K`, with the traits rows in
`src/vt/cpu/cpu_quant_traits.cpp` and the block layouts
`src/vt/cpu/cpu_quant_blocks.h::BlockIQ2_XS` (74 B) and `::BlockIQ4_XS` (136 B).

**`IQ4_XS` pairs with `Q8_K`, not `Q8_0`.** Read off `type_traits_cpu` at
`b10451 ggml/src/ggml-cpu/ggml-cpu.c:385-390` (`.vec_dot_type = GGML_TYPE_Q8_K`)
against `:379-384`'s `GGML_TYPE_Q8_0` for `IQ4_NL`. The 16-entry codebook is
shared with `IQ4_NL`; the block geometry is not, and it is the geometry that
picks the activation encoding. This has been got wrong before, so the pairing is
asserted in the test rather than assumed.

## Design

Both encodings are 256-element super-blocks against a `Q8_K` activation, so both
slot into the existing warp-per-output `QuantDotGemmKernel` with no new GEMM
shape, no new activation quantizer and no new launcher. The change is:

1. Two `__device__` codebooks in `cuda_quant_iq_tables.cuh`, mechanically
   derived from `src/vt/cpu/cpu_quant_iq_tables.h` by
   `scripts/gen-cuda-iq-tables.py` rather than hand-transcribed.
   `d_iq2xs_grid` is a `__device__` GLOBAL for the same measured reason
   `d_iq2xxs_grid` and `d_iq2s_grid` are: the 32 lanes of a warp own different
   super-blocks and therefore read different grid rows, and `__constant__`
   serialises a divergent read. `d_kvalues_iq4nl` is `__constant__`, following
   `d_kvalues_mxfp4`: 16 bytes, already resident in every cache.
2. `DotIQ2XS` and `DotIQ4XS`, each a literal transcription of its CPU twin,
   which is itself a 1:1 port of the oracle. **The accumulation order is
   upstream's and is not "improved".** `IQ2_XS` keeps its integer `bsum` and
   folds `0.125` once after the warp reduction, via `FinalFactor<kIQ2_XS>()`,
   exactly as `IQ2_XXS` and `IQ2_S` do. `IQ4_XS` keeps upstream's eight `f32`
   `sumf +=` steps per super-block — one per sub-block, `d1`/`d2` formed before
   the integer sums fold in — because that association is what the oracle's
   golden numbers were produced with.
3. `WType::kIQ2_XS = 10`, `kIQ4_XS = 11`, their `DotSuperblock` specialisations,
   the `FinalFactor` arm for `kIQ2_XS`, the two `IsCudaKeepQuantSupported` rows,
   and a `case` in each of the **three** dispatch switches — dense, grouped, and
   fused gate+up+SwiGLU. #967 is the reason the count is stated: it taught the
   predicate to say yes for two dtypes and extended only ONE switch, which
   turned a named refusal into a launch that never happened and an output tensor
   that kept whatever it held.
4. The drift seal grows two members. `IqTableSnapshot` gains `iq2xs_grid[512]`
   and `kvalues_iq4nl[16]`; `SnapshotIqTablesFromDevice` `static_assert`s each
   extent against `sizeof(d_…)` and copies it out, and the existing seal case
   memcmps both against the CPU tables.

No `f32` widening anywhere: the weight stays in its file blocks, the activation
is `Q8_K` as on CPU, and the output dtype is whatever the caller asked for
(`f32` or `bf16`), unchanged from the other ten encodings.

## Alignment

`BlockIQ2_XS` is 74 B and holds `uint16_t` members; `BlockIQ4_XS` is 136 B and
holds `uint16_t` members at offsets 0 and 2. Every block pointer in the GEMM is
`base + row * (nsb * block_bytes) + sb * block_bytes` with a 256-B-aligned
`cudaMalloc` base, and both 74 and 136 are even, so every `uint16_t` read is
2-byte aligned. This is the same argument that already holds for the 66-B
`BlockIQ2_XXS`. Neither dot does a 4-byte `__dp4a` read, so no stronger
alignment is claimed.

## Tests

**RED first, on the device.** The test additions land in a commit of their own,
BEFORE the kernels, and are built and run on the leased GPU so the failure is
observed rather than predicted. Expected RED: the fused-MoE case throws by name;
the graph-capture case fails because the CPU fallback synchronizes.

| Gate | What it can see that the others cannot |
|---|---|
| `IQ2_XS`/`IQ4_XS` rows in `kCases` | drives all five existing CUDA cases: dense vs CPU oracle at NMSE ≤ 1e-6, dense vs an independent f64 dequantize-then-dot at ≤ 5e-4, grouped vs CPU golden over a POISONED buffer, and the fused gate+up+SwiGLU seam. The last one is today's throw. |
| the oracle's own `vec_dot`, on the DEVICE, over REAL checkpoint bytes | `tests/vt/iq2xs_iq4xs_dot_golden.h` holds llama.cpp `b10451`'s OWN output for 4 super-blocks of `blk.3.ffn_gate_exps.weight` (IQ2_XS) and `blk.11.ffn_down_exps.weight` (IQ4_XS) of the staged artifact. At **k=256 the CUDA result is asserted BIT FOR BIT** against `kIq*DotPerBlockBits[b]`: one super-block means one contributing lane, and the warp tree then only adds exact zeros, so the reduction order cannot differ. At k=1024 four lanes contribute and the tree sums `(v0+v2)+(v1+v3)` where the oracle sums `((v0+v1)+v2)+v3`, so that case is asserted to a bound and the reason is written beside it. **This is the LOWER bound the arm needs**: a wrong grid row, a swapped scale nibble or a mis-shifted `scales_h` pair moves a reduction a little and passes any correlation check. |
| graph capture over `vt::MatmulBTQuant` | the only assertion that separates "ran on the device" from "drained the stream and ran on the host". Today's CPU fallback calls `cudaStreamSynchronize`, which is illegal under capture. |
| the device-codebook seal | a slipped literal in the generated `.cuh` fails on its own instead of waiting for a weight sample to address the drifted entry. |

Every float comparison is guarded with `std::isfinite` before it is believed: a
comparison against `NaN` is false, so an all-`NaN` forward reads as a perfect
match to a mismatch counter.

**Reachability.** The dtype arrives through `GgufFile::OpenOne` on a real header
(the existing `tests/vllm/test_gguf_dequant.cpp` "GgufFile reads IQ2_XS and
IQ4_XS tensors" case builds one), the loader routes it to `kKeepQuant`, and
`vt::MatmulBTQuant` / `vt::MoeGateUpSwiGLUGrouped` are the production ops the
model calls. The reachability mutation deletes the two `case` labels from the
fused-MoE switch in a scratch copy and reruns the focused gate; a green gate
would mean the gate measures the `__device__` function rather than the seam.

## Gates

- `scripts/agent-preflight.sh --fail-on-skip`, zero skips.
- Host C++ suites by hand, with counts: `test_ops_quant_dot`,
  `test_gguf_keep_quant`, `test_gguf_dequant`, `test_cuda_quant_dot` (CPU arm).
- On a leased `dgx:gpu0` (`sm_121a`) worker: a CUDA build and
  `test_cuda_quant_dot`, RED at the test-only commit and GREEN at the
  implementation commit, from the same build directory.
- On the leased box, `scripts/check-cuda-fat-gencode.py --compile-commands` over
  that build's `compile_commands.json`, which no CPU preflight can supply.

### The end-to-end check this row was asked for cannot run, and #2260 is not why

Driving the staged GLM-5.3-Flash `UD-Q2_K_XL` through `--device cuda` was the
decisive test on this row's brief. It is blocked one level above these kernels,
and the block is in another row's code:

```text
src/vllm/model_executor/models/glm5_next_forward.cpp::Glm5NextHostForward
  "this forward is a HOST f32 reference and was handed a non-CPU queue.
   Every glm5_next primitive on this row -- the KDA recurrence, the DSA
   indexer, the mHC blocks, the MoE router and the attention -- is host code,
   and `vt::MoeRouterTopK` dispatches on the queue's device ..."
```

That refusal is unconditional on `queue.device.type != vt::DeviceType::kCPU`,
and it fires before any GEMM. So `--device cuda` on this artifact refuses BY
NAME whether or not these kernels exist, and a CUDA token from this model is not
available to be observed on this row at any effort. **No token is claimed.** The
device arm of that model is owed by
`MODEL-MM-glm5-next-glm5-next-for-conditional-generation` and its campaign
[#1998](https://github.com/mudler/vllm.cpp/issues/1998).

What #2260 does discharge is real and is the prerequisite: the fused MoE seam
stops throwing, the dense and grouped seams stop draining the stream to the
host, and the encodings become capturable. The remaining `--device cpu`
instruction in `docs/FEATURES.md` and `docs/USAGE.md` is therefore kept and its
REASON is corrected, because a record correction that leaves a wrong reason
standing is not a correction.

## Evidence

### Host pre-check of the ported math, before any lease was spent

`nvcc` is not on the x86 development box, so a bad port would otherwise have
been found only after a queue and a build on the leased GB10. The two new
`__device__` functions were therefore EXTRACTED from
`src/vt/cuda/cuda_quant_dot.cu` by line range — the exact bytes that will be
compiled for the device, not a retyped copy — compiled by `g++` with `__device__`
and `__constant__` defined away, given the GENERATED `cuda_quant_iq_tables.cuh`
as their codebooks, and run against the oracle's own numbers:

```text
iq2_xs block 0: got 0x3FED7254 (1.85505152)    oracle 0x3FED7254  BIT-EXACT
iq2_xs block 1: got 0xBE366684 (-0.178125441)  oracle 0xBE366684  BIT-EXACT
iq2_xs block 2: got 0xBF0B2FD5 (-0.543698609)  oracle 0xBF0B2FD5  BIT-EXACT
iq2_xs block 3: got 0x3FAC2A65 (1.34504378)    oracle 0x3FAC2A65  BIT-EXACT
iq4_xs block 0: got 0xC084FEC1 (-4.15609789)   oracle 0xC084FEC1  BIT-EXACT
iq4_xs block 1: got 0xBFC39E5A (-1.52827001)   oracle 0xBFC39E5A  BIT-EXACT
iq4_xs block 2: got 0x3E173704 (0.147670805)   oracle 0x3E173704  BIT-EXACT
iq4_xs block 3: got 0x3FFF0F4C (1.99265432)    oracle 0x3FFF0F4C  BIT-EXACT
iq2_xs k=1024: tree 0x401E9BFF  oracle-seq 0x401E9C00  margin 2.38e-07  bound 1.87e-06
iq4_xs k=1024: tree 0xC062D199  oracle-seq 0xC062D19B  margin 4.77e-07  bound 3.73e-06
bad=0
```

**Say what this is and what it is not.** It is `x86_64` `g++`, not `sm_121a`
`nvcc`, so it measures the ALGORITHM and not the device: it cannot see a wrong
`__constant__` qualifier, a misaligned device read, a launch that never happens,
or a codebook that failed to reach the GPU. It settles the port's arithmetic
against the pinned oracle before a lease is spent, and the device gates settle
the rest. It also confirmed the k=1024 design decision: the warp tree's order
and the oracle's sequential order differ by ONE ULP on both encodings, so
asserting bit equality against the oracle's total would have been red on the
device for a reason that is not a defect.

### The f64-dequant band was PREDICTED before the lease, not discovered on it

The CUDA gate asserts each dtype against an independent f64
dequantize-then-dot at upstream's 5e-4 (`test-backend-ops.cpp:4277`), and that
comparison holds a Q8_K-quantized activation against an f32 reference, so what
it measures is ACTIVATION error and how far that error travels depends on the
weight distribution the encoding produces. `IQ1_S` and `IQ1_XXXS` needed a
per-case ceiling for exactly that reason. A third harness therefore ran the
device dots over the SAME `RandomBlocks` weights, the same `GenerateData`
activation and the same M{1,4,32,512} x N{1,7,16} shapes the device case uses,
with the production `Q8_K` encoder and `BlockToFloat` as the reference:

```text
iq2_xs worst NMSE vs f64 dequant = 1.601e-04 at M=32  N=1   (band 5e-4)  UNDER
iq4_xs worst NMSE vs f64 dequant = 6.027e-05 at M=512 N=1   (band 5e-4)  UNDER
```

Both sit under the shared band with room, so **neither dtype takes a per-case
ceiling and none was added**. Had either come out over, the answer would have
been `NEEDS_DECISION`, not a wider band.

### `check-pr-size.py` is red on this branch and it is BASE-CAUSED

```text
ERROR: PR size check could not classify the change:
       unclassified repository path '.agents/scripts/glm53-dsa-first-load.sh'
```

Measured both ways. Against this row's own four commits
(`673464ee1..HEAD`) the checker reports `OK: every explicit path class is
within its review budget`. Against `origin/main..origin/land/glm53-flash-and-dsa`
alone, with none of this row's commits present, it reports the identical error.
The unclassified path is `land/glm53-flash-and-dsa`'s, added by `fe2117c63`, and
that branch cannot pass this gate today whoever merges it. Reported rather than
repaired: the fix is a path class in another row's change, and adding one here
would put a checker edit into a kernel port.

The generated `d_iq2xs_grid` was checked a second way, independently of the
device seal: its 512 entries serialized little-endian FNV-1a-64 to
`0xc9b1ee61e79909bd`, the digest `cpu_quant_iq_tables.h` states for `kIq2xsGrid`
and `test_ops_quant_dot` re-derives.

## Risks

- **A number from `thor:gpu0` is not a number for `dgx:gpu0`.** `sm_110` and
  `sm_121a` are different targets; every result records its box.
- The k=1024 golden case cannot be bit-exact and saying so is part of the gate,
  not an excuse discovered afterwards.
- `dgx:gpu0` has crashed under long sequences; the lease script is written to be
  resumable and reports `df -h` before and after.

## Stop conditions

`NEEDS_DECISION` rather than widening a tolerance, bending a golden, or claiming
device parity not measured in the same tool. `NEEDS_CONTEXT` if the staged
artifact is absent from the leased worker.

## Owed

- **These two dispatch arms are landed but not yet SELECTED by any running
  model, and that is declared rather than implied.** The call sites are live and
  hot -- `vt::MatmulBTQuant`, `vt::MatmulBTQuantGrouped` and
  `vt::MoeGateUpSwiGLUGrouped` run for every keep-quant model on CUDA, and the
  loader already routes both encodings to `kKeepQuant` on that device. What no
  checkpoint reaches today is the DATA: `IQ2_XS` and `IQ4_XS` appear only in the
  two GLM-5.3 artifacts, `GlmMoeDsaForCausalLM` forwards nothing at all, and
  `Glm5NextForConditionalGeneration` refuses a non-CPU queue by name. This is
  the "unselected branch" shape in
  [`reachability.md`](../reachability.md), the wiring is owed by
  `MODEL-MM-glm5-next-glm5-next-for-conditional-generation`, and the campaign
  that tracks it is [#1998](https://github.com/mudler/vllm.cpp/issues/1998).
  The reachability mutation below therefore proves the SEAM, not a model step.
- The ROCm arm of both encodings. `rocm-gg-keep-quant.md` owns it and already
  records `IQ2_*` / `IQ3_*` as owed; this row does not touch
  `src/vt/rocm/rocm_grouped_gemm.hip`.
- Speed. This row lands a correct kernel on the same MMVQ warp-per-output
  structure as the other ten encodings and claims no throughput number. No
  `__dp4a` vectorisation of either dot is attempted here; `DotQ4K` and `DotQ5K`
  show what that would look like when a row measures it.
- The `MXFP4` / `Q4_0` / `IQ4_NL` / `Q5_0` Q8_0-activation GEMM variant, which
  is unchanged and still CPU-fallbacks.
