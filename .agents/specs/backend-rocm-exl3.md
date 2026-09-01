# BACKEND-ROCM-EXL3 — the two operations an EXL3 checkpoint still runs on the CPU

Row: `BACKEND-ROCM-EXL3`
Issues: [#2433](https://github.com/mudler/vllm.cpp/issues/2433) (primary)
Base SHA: `c5df2b1ed` (`row/QUANT-EXL3-MUL1`, unmerged at the time of writing)
Parent rows: [`QUANT-EXL3`](quant-exl3-shared.md), [`BACKEND-ROCM`](rocm-backend-w0.md)
Matrix: [`.agents/backend-matrix.md`](../backend-matrix.md)

Upstream pin: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` — **vLLM implements
no EXL3** at the pin, so the format is mirrored from the registered secondary
oracle [`exllamav3`](../oracles/exllamav3.md) @
`2398c05635fbbad01a0a51dce63c85c6c8a8450e` (MIT). The op seam, the reference
tier and the backend structure are vLLM's and this tree's; exllamav3 supplies
the trellis format only.

**Why the base is not `origin/main`.** Codebook 2 (`mul1`) landed on
`row/QUANT-EXL3-MUL1` and had not merged when this row started. The device arm
here instantiates all three codebooks, so it cannot be written against a tree
that only knows two. When that branch lands, merge `origin/main` and re-verify.

## Now

`ACTIVE`. The two ROCm kernels are written and the CPU-side gate is green. The
device half is what a `strix:gpu0` lease still owes; §Gates says exactly which
lines are unmeasured until it runs.

## The gap, as measured

The ROCm model sweep ran `llama32-1b-exl3-3bpw` on `strix:gpu0` (`gfx1151`,
ROCm 7.2.4) through the public `vllm-cli` path (#2433, rc job
`43267dc3-52c5-43b2-b6b7-538aff68e6b6`). One greedy token completed, and
provider statistics reported exactly two CPU reference-tier operations:

```text
[vt reference-tier] op=CastF16 device=rocm:0 ...
[vt reference-tier] op=Exl3Gemm device=rocm:0 ...
```

1.436 s for one output token. The BF16 control `llama32-1b-instruct-bf16`
completed on the same binary and device with **zero** reference-tier hits, which
isolates the gap to the EXL3 path and not to the Llama loader, paged attention,
the sampler or the dense model.

**On a DISCRETE Radeon this is not slowness, it is a refusal.** The reference
tier installs only where `Backend::UnifiedMemory()` is true
([`docs/ROCM.md`](../../docs/ROCM.md) "Understand fallback behavior"), so on a
discrete board `GetOp` throws and an EXL3 checkpoint does not run at all. R2
below is what fixes that. **We cannot test that fix**: `strix:gpu0` is the only
AMD device on the fleet and it is an APU.

## Scope

Exactly two ops.

- **R1 `kCastF16` on ROCm.** Registered today on CPU
  (`src/vt/cpu/cpu_ops.cpp` `kCastF16` registrar) and CUDA
  (`src/vt/cuda/cuda_glue.cu` `CastF16KernelCuda`) while its two siblings
  `kCastBf16`/`kCastF32` have six backends each.
  `.agents/specs/quant-exl3-shared.md` `## Owed` already records this gap in
  those words; this row discharges the ROCm quarter of it.
- **R2 `kExl3Gemm` on ROCm**, transcribed from the portable CPU reference and
  **not** ported from `src/vt/cuda/cuda_exl3.cu`.

Out of scope, each for a stated reason:

- `kExl3MoeMlp`. No AMD board on this fleet can hold the artifact that reaches
  it (~99.5 GiB; strix has 67 GB of system RAM), and the tree-wide fused-MoE
  contract is codebook-1-only. Owed, not done.
- `kExl3HadR128` as a registered ROCm op. It is not on a dense forward path, so
  registering it would add a surface nothing reaches. The transform itself IS
  implemented here — `Exl3Gemm` cannot exist without it — as an internal
  device function, and it is gated transitively (§Gates).
- Any matrix-core (`v_mfma`/WMMA), LDS-staged or split-K fast path. This row
  buys CORRECTNESS and the end of the reference tier. Speed is a later row.

## Why the CUDA kernel is not the donor

`src/vt/cuda/cuda_exl3.cu` cannot run on gfx1151, and the reasons are
structural rather than instruction-level. The first one alone settles it:

| Anchor | What it is | Why gfx1151 cannot have it |
|---|---|---|
| `cuda_exl3.cu:103` `constexpr int kSmemMax = 90 * 1024;`, asserted at `:694`, set at `:1893` via `cudaFuncAttributeMaxDynamicSharedMemorySize` | the kernel's shared-memory budget | **AMD LDS is 64 KiB per workgroup.** 90 KiB does not fit, and the tile arithmetic at `:694` is sized against that number, so no flag or launch bound recovers it |
| `:168` and `:1248` `mma.sync.aligned.m16n8k16` | the tensor-core inner product | `m16n8k16` has no AMD shape. RDNA3.5 WMMA is `16x16x16`, and the `FragA`/`FragB`/`FragC` blocking at `:151-155` is NVIDIA's fragment layout, not a width |
| `:257` `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | the fragment load | no equivalent instruction |
| `:241`, `:247`, `:251` `cp.async` / `cp.async.commit_group` / `cp.async.wait_group` | asynchronous global→shared staging, which is what the `SH_STAGES` pipeline at `:694` is made of | gfx1151 has no async global→LDS copy |

`src/vt/rocm/` is hand-written HIP ported from vLLM's ROCm `csrc/`, not a
hipify of our CUDA, and this tree ships no hipify script. That property is
preserved here.

## The donor that IS used

`src/vt/cpu/cpu_exl3_kernels.cpp` and `src/vt/cpu/cpu_exl3_dequant.cpp`, which
are pure portable C++ with no intrinsics. They are therefore both the SOURCE and
the ORACLE, which is what makes this row's gate a byte gate rather than a
tolerance:

| Device function | Transcribed from |
|---|---|
| `HadBlock128` levels 1-2 | `cpu_exl3_kernels.cpp` `HadBlock128`, the `s0/d0/s1/d1` butterfly |
| `HadBlock128` levels 4-64 | `cpu_exl3_kernels.cpp` `ShuffleHadWarp`, five xor-partner steps over 32 lanes, sign flip by XORing the f32 sign bit |
| the four (in, out) width arms and where each scale lands | `cpu_exl3_kernels.cpp` `HadRowBlock` |
| the three-step fused chain | `cpu_exl3_kernels.cpp` `Exl3GemmKernelCpu` |
| the tail-biting 16-bit window read | `cpu_exl3_dequant.cpp` `Exl3TileCodeword` |
| the three codebooks | `cpu_exl3_dequant.cpp` `Exl3DecodeCodeword` |
| the tensor-core permutation | `cpu_exl3_dequant.cpp` `Exl3TileRowMajorIndex` |

All three codebooks are instantiated, because an AMD box has no reason to see
fewer artifacts than an NVIDIA one:

- cb 0, 3INST: `x*89226354u + 64248484u`
- cb 1, MCG: `x*0xCBAC1FEDu`
- cb 2, mul1: `x*0x83DCD12Du`, then the **unsigned byte sum** of the product's
  four bytes into `0x6400`, reinterpreted as an fp16 bit pattern, then the fp16
  affine `k_inv = 0x1eee`, `k_bias = 0xc931`

For the byte sum this file follows the tree's own precedent rather than
inventing one: `src/vt/rocm/rocm_grouped_gemm.hip` `Dp4a` is a scalar
implementation documented as "bit-identical to `__dp4a`", with the hardware
instruction named as "a perf lever, not a correctness requirement". The same
sentence applies to `v_dot4` here.

## The exemplar mirrored

`src/vt/rocm/rocm_grouped_gemm.hip`, which serves `kMatmulBTQuant` — the same
"decode the weight INSIDE the GEMM" shape this op has. From it: the file
header naming the donor and its anchors, `#include <hip/hip_fp16.h>` +
`vt/rocm/rocm_device_bind.h`, the `__device__` bit-exact f16/f32 codec ports
rather than vendor intrinsics, the scalar-with-a-named-perf-lever comment
convention, and the registration through `rocm_ops.hip`.

## Design

### R1 — `CastF16KernelRocm`

A template on the source type, in `rocm_dense_basic.hip` beside the two
siblings, with the same packed-view row handling they have (`row_size`,
`in.stride[0]`) — which the CPU arm does NOT have and the CUDA arm does. It
accepts an f32 or a bf16 source and refuses an f16 one, which is what the
header contract says the op does.

The narrowing round is `DF32ToF16`, the bit-exact port, and not
`__float2half`. The two agree on RNE today; only one of them is a transcription
of `vt::F32ToF16`, and this op's whole gate is byte-equality with that function.

### R2 — `Exl3GemmKernelRocm`, three kernels in `src/vt/rocm/rocm_exl3.hip`

**Step 1, the input Hadamard.** `A[m,k]` fp16 with `pre_scale = suh`, into
`a_had` (which may alias `A`). One 32-lane group per 128-block, four groups per
workgroup; `h[4][32]` in LDS, so the five xor-partner levels are literally
`ShuffleHadWarp` with `__syncthreads()` where the CPU has a `memcpy`. Every
lane reads its four inputs BEFORE the first barrier and writes after the last,
which is what makes the in-place alias safe. LDS cost: 2 KiB.

**Step 2, the GEMM.** Grid `(n/16, ceil(m/16))`, 256 threads. Per k-tile: each
thread decodes ONE of the 256 codewords into `tile[Exl3TileRowMajorIndex(t)]`
in LDS — that is `Exl3DecodeTile` with the loop unrolled across threads — then
thread `(r, cc)` accumulates `acc += xv * tile[rr*16 + cc]` for `rr` ascending,
with `ti` ascending across the outer loop.

**That order is the CPU's order, element for element**, including its
`if (xv == 0.0f) continue;`, which is kept for a reason and not by habit: with
`acc == -0.0f` an added `+0.0f` product would flip the sign of the zero. So the
f32 accumulation is not merely close to the CPU arm, it is the SAME SEQUENCE OF
IEEE OPERATIONS, and the gate can require equality. LDS cost: 1 KiB.

**Step 3, the output Hadamard**, over the f32 `raw` into `c` with
`post_scale = svh`, in the f32→f32 or f32→f16 arm depending on `c.dtype` —
the same two arms `Exl3GemmKernelCpu` picks between.

`raw` is a device scratch allocated per call from the backend allocator. It is
`m*n` f32; for the Llama-3.2-1B EXL3 shapes at batch 1 that is at most 8192
floats. Named as a cost, not hidden: a fused kernel would not need it, and a
fused kernel is the later speed row.

**No matrix cores, no async copies, no cooperative launch, and 3 KiB of LDS
against a 64 KiB budget.** The structural blockers in the table above are not
worked around; they are not encountered.

## Tests

`tests/vt/test_exl3_rocm.cpp`, modelled on the device section of
`tests/vt/test_exl3_gemm.cpp` and sharing `tests/vt/exl3_fixture.h`, so no
second copy of the fixture exists.

1. **Decode, all three codebooks, bit-exact.** The device decode compared to
   `Exl3DecodeTile`/`Exl3DecodeCodeword` as INTEGERS (the fp16 bit patterns),
   asserted equal, not within a tolerance. Reached through `Exl3Gemm` with a
   trellis chosen so the GEMM reduces to the decode.
2. **GEMM against `Exl3GemmKernelCpu`** on identical inputs, asserted
   **byte-equal**. The bound is exact and the justification is in §Design:
   the three steps run the same IEEE f32 operations in the same order. This is
   a stronger claim than the CUDA arm's `1.0e-3`, and it is available only
   because this arm does not use tensor cores.
3. **`CastF16` against the CPU arm**, byte-equal, on an f32 and on a bf16
   source.
4. **Registration**, so a build that compiles the file but fails to register it
   is red.

Every device case SKIPS when no ROCm backend is registered, says so, and STILL
ASSERTS the precondition it skipped on — `assertions: 0` with `SUCCESS!` is a
skip wearing a pass and this family has already paid for it once.

## Gates

Run from the worktree; `ctest`, never a raw binary.

```sh
cmake -S . -B build -G Ninja -DVLLM_CPP_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
ninja -C build -j 2 test_exl3_rocm test_cast_f16 test_exl3_gemm test_exl3_dequant
ctest --test-dir build -R '^(test_exl3_rocm|test_cast_f16|test_exl3_gemm|test_exl3_dequant)$' --output-on-failure
```

On the device, inside an `rc` lease on `strix:gpu0` (never `ssh`):

```sh
cmake -S . -B build-hip -G Ninja -DVLLM_CPP_HIP=ON -DVLLM_CPP_HIP_ARCHITECTURES=gfx1151 \
      -DVLLM_CPP_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
ninja -C build-hip -j 4 test_exl3_rocm
ctest --test-dir build-hip -R '^test_exl3_rocm$' --output-on-failure
VT_OP_PROVIDER_STATS=1 ./build-hip/examples/vllm-cli --model <exl3 ckpt> \
      --prompt "The capital of France is" --max-tokens 8 --temperature 0
```

The acceptance criterion of #2433 is the last line: **zero reference-tier
hits**, `GetReferenceTierHits() == 0`, and `CastF16`/`Exl3Gemm` absent from the
fallback report.

**What a run on `strix:gpu0` can and cannot conclude.** gfx1151 is an RDNA3.5
**APU** with unified memory (`managedMemory=1`, `integrated=1`). Any number from
it is a gfx1151 APU number and generalizes to no discrete Radeon. And because
the reference tier is OFF on a discrete board by design, the discrete arm of R2
is the arm that turns a refusal into a run — and it is the arm no device here
can execute.

## Owed

- **`kExl3MoeMlp` on ROCm.** Unreached, unwritten, and unmeasurable on this
  fleet for the memory reason in §Scope.
- **`kExl3HadR128` as a registered ROCm op.** The transform exists; the
  registration does not, because nothing on a dense forward path calls it.
- **Speed.** This row records a per-token time as a diagnostic. It offers no
  throughput result and no AMD clock attribution, which #2433's fourth
  acceptance line asks for; that line stays open on this row.
- **A discrete AMD board.** Until one exists here, "EXL3 now runs on discrete
  ROCm" is a code claim and not a measurement.
- **`kCastF16` on Vulkan, Metal and Tenstorrent**, the other three quarters of
  the gap `quant-exl3-shared.md` records.

## Risks/decisions

- **A byte gate is a strong claim and can red on a compiler.** `-ffp-contract=off`
  is already set for HIP in `CMakeLists.txt`, which is what stops an FMA from
  contracting the `acc += xv * w` into a different value. If the byte gate reds,
  the answer is to find the contraction, never to widen the bound to green.
- **Decoding a codebook as the wrong one is invisible to a shape check.** The
  wrong multiplier yields a weight with the right DISTRIBUTION and no
  correlation to the true one. That is why case 1 asserts integer equality on
  the fp16 bit patterns.
- **`strix:gpu0` may not free.** Then the CPU-gated half lands and the device
  half is reported UNMEASURED by name. It is not reported as passing.

## Stop conditions

- The device decode disagrees with `Exl3DecodeTile` on any codeword → stop, and
  re-derive from `codebook.cuh:56-90`. Never tune a constant to green.
- The byte gate needs a tolerance to pass → stop. The design claim in §Design
  is then false and the design, not the gate, is what changes.
- A reference-tier hit remains after both ops register → stop and read the
  provider report before writing a third kernel; the hit names the op.
