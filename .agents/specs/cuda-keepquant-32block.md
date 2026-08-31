# CUDA keep-quant GEMM for the 32-element encodings — `QUANT-CUDA-KEEPQUANT-32B`

Issue: [#2419](https://github.com/mudler/vllm.cpp/issues/2419).
Branch: `row/QUANT-CUDA-KEEPQUANT-32B`.
Extends [`cuda-keepquant-gemm.md`](cuda-keepquant-gemm.md), whose
`KERNEL-QUANT-CIQ-GEMM-CUDA` owns the CUDA `kMatmulBTQuant` provider, and
follows [`cuda-keepquant-iq2xs-iq4xs.md`](cuda-keepquant-iq2xs-iq4xs.md), which
added the last two Q8_K-activation encodings. Discharges the keep-quant half of
the debt [#2380](https://github.com/mudler/vllm.cpp/issues/2380) records under
`## Owed` for `MODEL-MM-QWEN4-EXP`.

## Now

`ACTIVE`.

## The gap, as measured on `4ab04afd6`

`IsCudaKeepQuantSupported` (`src/vt/cuda/cuda_quant_dot.cu:1736`) maps twelve
`DType`s to a `WType`. Every one is a **256-element super-block** encoding whose
`vec_dot` pairs it with a `BlockQ8_K` activation, and the templated GEMM behind
the predicate (`DotSuperblock<W>(const void*, const BlockQ8_K*)`) takes that
activation type in its signature.

**IQ4_NL, Q5_0 and Q4_0 are absent, and cannot be added by writing a `case`.**
They are 32-element encodings that dot a `BlockQ8_0` activation. The mismatch is
in the activation type and the block extent, so this row adds a *second*
templated GEMM beside the first rather than extending it.

Q8_0 is also absent from the predicate but is **not** part of this gap: it has a
dedicated on-device path (`MatmulQ8_0Cuda`, `MatmulQ8_0GroupedCuda`) dispatched
before the predicate is consulted, carrying its own tuned levers and gates. This
row does not touch it, and does not route it through the new template — doing so
would put the DeepSeek-V4 decode levers behind an untuned generic kernel for no
gain this row measures.

### What the three consumers do today

| Seam | Today, for IQ4_NL / Q5_0 / Q4_0 |
|---|---|
| `MatmulBTQuantKernelCuda:1993` | `cudaStreamSynchronize` then `GetOp(kMatmulBTQuant, kCPU)` over the same tensors. |
| `MatmulBTQuantGroupedKernelCuda:2090` | the same drain-and-fall-back. |
| `MoeGateUpSwiGLUGroupedCuda:2320` | **throws** `moe_gate_up_swiglu: gate/up must be the SAME CUDA keep-quant dtype`. |

The fallback is correct and host-speed where `Backend::Alloc`'s memory is
host-addressable. On CUDA it is a plain `cudaMalloc` (`cuda_backend.cu:104-108`),
so where it is not, the CPU kernel dereferences a device pointer — the SIGSEGV
`cuda-keepquant-iq2xs-iq4xs.md` measured on GB10 for the previous two dtypes.
Either way the sync **invalidates a decode graph capture**, so the fallback is
not merely slow: it is unusable by the captured decode path.

### Why the released artifact needs exactly these

`unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S stores its routed-expert towers in
IQ4_NL and Q5_0. That is forced, not chosen: `expert_feed_forward_length` is
**640**, so every routed expert row is indivisible by 256 and no K-quant can
encode it. `qwen4_exp_gguf_weights.cpp:117-119` states this in the tree.

The consequence is sharper than a missing `case`. `ffn_down_exps` is `[E*H, I]`
with `I = 640`, so **K = 640**. The Q8_K seams reject that shape by name
(`K must be a whole number of 256-element Q8_K super-blocks`) *regardless of
dtype*. The 32-element lane is the only lane this checkpoint's MoE can use.

## Scope

**In.** A `BlockQ8_0`-activation templated GEMM for IQ4_NL, Q5_0 and Q4_0:
device dot, dense kernel, grouped kernel, fused grouped gate+up+SwiGLU kernel,
and the dispatch in all three seams above.

**Out.** MXFP4 (the fourth 32-element encoding, `ffn_down` of the DeepSeek-V4
UD-IQ2_M arm). Its device math (`DotMXFP4`) is already written and unreferenced,
so the template this row adds is what it needs — but no artifact this row gates
uses it, and adding an ungated arm is how a dead path lands. Recorded under
`## Owed`.

**Out.** Vectorising the dot with `__dp4a` over packed nibbles (llama.cpp's
`vec_dot_q4_0_q8_1_impl` shape). This row's kernels are scalar loops that mirror
the CPU oracle statement for statement. That is a real throughput ceiling and it
is recorded under `## Owed`; it is not a correctness question, and landing a
correct on-device arm is what removes the host drain.

**Out.** `gguf_keep_quant.cpp::DeviceKeepQuantSupported` and `cuda_ops.cu`
(#2396), the four `qwen4_exp` op kernels (#2391), `qwen4_exp_qsa_block.cpp`
(wave QSADEV).

## Upstream anchors

vLLM implements none of these three `vec_dot`s — it has no GGUF k-quant CPU
reference tier at all — so the secondary-oracle rule applies exactly as it did
for the CPU arm and for `cuda-keepquant-iq2xs-iq4xs.md`. Oracle: llama.cpp
`b10451` = `10bf611e533d81f739128304991c5e133c6aebd8`
([`oracles/llama-cpp.md`](../oracles/llama-cpp.md)).

The **behavioural** oracle for the gate is closer than upstream: it is this
tree's own CPU arm, the same kernel the host drain runs today.

| Ported | From (llama.cpp b10451) | Via (this tree's CPU arm) | To |
|---|---|---|---|
| `ggml_vec_dot_q4_0_q8_0_generic` | `ggml/src/ggml-cpu/quants.c:174` | `cpu_quant_dot.cpp:50 VecDotQ4_0Q8_0` | `cuda_quant_dot.cu::Dot32<W32::kQ4_0>` |
| `ggml_vec_dot_q5_0_q8_0_generic` | `ggml/src/ggml-cpu/quants.c:365` | `cpu_quant_dot.cpp:92 VecDotQ5_0Q8_0` | `cuda_quant_dot.cu::Dot32<W32::kQ5_0>` |
| `ggml_vec_dot_iq4_nl_q8_0_generic` | `ggml/src/ggml-cpu/quants.c:1254` | `cpu_quant_dot.cpp:140 VecDotIQ4_NLQ8_0` | `cuda_quant_dot.cu::Dot32<W32::kIQ4_NL>` |

### The association order is part of the port

The three upstream kernels fold the scales in three *different* orders, and the
CPU arm's comments already say so. They are reproduced rather than normalised:

- Q4_0: `sumf += sumi * F16ToF32(x.d) * F16ToF32(y.d)` — left-associated, the
  integer sum multiplied first.
- Q5_0: `sumf += (F16ToF32(x.d) * F16ToF32(y.d)) * (float)sumi` — scale product
  formed first.
- IQ4_NL: `d = F16ToF32(y.d) * F16ToF32(x.d); sumf += d * (float)(s1 + s2)` —
  scale product first *and* the operands in the opposite order.

`-ffp-contract=off` is CXX-only (`CMakeLists.txt:55`) and never reaches `.cu`,
so nvcc would contract each of these into an FMA that rounds once where upstream
rounds twice. `cuda_quant_dot.cu:535-558` records that being MEASURED here — two
of eight real super-blocks off by 1 and 4 ULP. Every multiply-add in the new
dots is therefore spelled `__fmul_rn` / `__fadd_rn`, as `DotIQ4XS` is.

## Memory format

Checked against the oracle as `.agents/porting.md` requires, because a dtype
that is too wide passes a token gate while moving twice the bytes.

| Encoding | Block | Bytes | Layout |
|---|---|---|---|
| Q4_0 | 32 | 18 | `d` f16 @0, `qs[16]` @2 |
| Q5_0 | 32 | 22 | `d` f16 @0, `qh[4]` @2, `qs[16]` @6 |
| IQ4_NL | 32 | 18 | `d` f16 @0, `qs[16]` @2 |

The kernels read the weight bytes **in place**, at the block stride the loader
already produced, and the accumulator is f32 — the same width as the Q8_K
template's and as the CPU arm's `float sumf`. No dequantised f32 or bf16 copy of
the weight is materialised anywhere on this path.

## Design

A second enum `W32` and a second templated GEMM, beside the Q8_K one:

- `Dot32<W>(const void* wb, const BlockQ8_0* ab)` — one weight block against one
  activation block, mirroring the named CPU function statement for statement.
- `IsCuda32BlockKeepQuantSupported(DType, W32*)` — the new predicate, structured
  exactly like `IsCudaKeepQuantSupported` so the two read as siblings.
- `QuantDotGemm32Kernel<W, OutT>` — dense, one warp per output, lane-strided over
  blocks, 32-lane `__shfl_down_sync` tree reduce. Mirrors
  `QuantDotGemmQ8_0Kernel`.
- `QuantDotGemmGrouped32Kernel<W, OutT>` — weight row `expert_ids[p]*n + j`,
  activation row `bcast ? 0 : p`. Mirrors `QuantDotGemmGroupedQ8_0Kernel`.
- `QuantDotGemmGroupedFusedSwiGLU32Kernel<W>` — gate and up in one warp, the
  same `ClampedSwiGLU` epilogue as the Q8_K fused kernel.

The activation quantiser is **reused, not rewritten**: `QuantizeQ8_0Kernel` /
`QuantizeQ8_0PreqKernel` and the grow-only per-stream `EnsureScratch` are already
the Q8_0-activation prologue, already graph-safe, and already gated.

`K % 32 == 0` replaces `K % 256 == 0` on the new lane, which is what lets K=640
through.

## Risks

1. **A passing token gate proves nothing here.** Addressed by the capture gate
   below; stated again because it is the failure this row exists to make
   visible.
2. **A fixture that never crosses a tile boundary cannot see a cross-tile bug.**
   The dense kernel strides `b = lane; b < nb; b += 32`, so a fixture with
   `nb <= 32` never takes the loop twice and a lane-index defect is invisible.
   The gate therefore includes a shape with `nb > 32`.
3. **Scalar dot is a throughput ceiling.** Named in `## Owed`, not hidden.
4. **Q5_0's `qh` is read as a `uint32_t` from a 2-byte-aligned offset** (@2 of a
   22-byte block). The CPU arm uses `memcpy`; the device dot must not deref a
   `uint32_t*` there. Byte-assembled instead.

## Tests and gates

`tests/vt/test_cuda_quant_dot.cpp`, which is table-driven over `kCases`. Adding
three rows drives every case in the file at once.

**What the gates observe, and what they cannot:**

| Gate | Observes | Cannot observe |
|---|---|---|
| `CUDA keep-quant GEMM == CPU reference and f64 dequant` | that the device numbers match the CPU arm (NMSE) and an independent f64 dequant-and-dot | **whether the device ran at all** — it reads the host fallback as a pass. This is the trap, and this gate is not the discriminator. |
| `CUDA keep-quant runs every kCases dtype INSIDE a stream capture` | **the path.** `cudaStreamSynchronize` on a capturing stream fails, so this is red for exactly the dtypes that drain and green for the ones that do not. Asserts the counted property `captured == std::size(kCases)`. | *which* device kernel ran, or how fast |
| fused-MoE grouped case | that `MoeGateUpSwiGLUGrouped` returns values instead of throwing by name | end-to-end MoE numerics on the real checkpoint |

The capture gate is the red-first discriminator. The parity gate is what says
the new kernel is *correct* once it runs.

Neither needs the real checkpoint, and neither can speak for it. What is proven
on a fixture is the dot, the dispatch and the capturability; what is **not**
proven is the released artifact's own tensors flowing through a full forward.
That is stated in `## Owed` rather than implied by a green suite.

## Reachability

`vt::MatmulBTQuant`, `vt::MatmulBTQuantGrouped` and `vt::MoeGateUpSwiGLUGrouped`
are the production entry points; the tests call them, never a kernel directly.
The mutation deletes the new `case` arms from the seam dispatch in a scratch copy
and shows the capture gate's `captured` count fall — a counted property, so a
mutation that never applied cannot read as a pass.

## Owed

- **MXFP4 CUDA keep-quant arm.** `DotMXFP4` exists and is unreferenced; this
  row's template is the seam it needs. Not gated by any artifact this row
  measures. Owner: this row's issue [#2419](https://github.com/mudler/vllm.cpp/issues/2419).
- **`__dp4a`-vectorised 32-block dot.** The scalar loop here is correct and
  on-device; llama.cpp's `vec_dot_q4_0_q8_1_impl` is the shape to port. No
  throughput floor is claimed by this row.
- **An end-to-end `qwen4_exp` CUDA forward on the released checkpoint.** Blocked
  independently of this row by the block-decoding n-gram gather having no CUDA
  arm (#2380). This row removes the keep-quant blocker only.
