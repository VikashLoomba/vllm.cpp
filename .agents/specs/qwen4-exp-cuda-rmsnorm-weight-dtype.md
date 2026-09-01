# CUDA `RmsNorm` refuses an f32 gamma against a bf16 activation

**Row:** `MODEL-MM-QWEN4-EXP` (wave SANITIZE)
**Issue:** [#2477](https://github.com/mudler/vllm.cpp/issues/2477)
**State:** `ACTIVE`
**Base:** `origin/main` at `855905f59`

## Scope

Make the CUDA `RmsNorm` kernel read its gamma in the gamma's own dtype, so a
`--device cuda` forward of `qwen4_exp` on the released GGUF artifact stops
refusing at `src/vt/cuda/cuda_ops.cu:463`.

Out of scope, recorded under `## Owed`: the ROCm twin of the same weld, and
issue #2476.

## What is wrong, and which side of the mismatch it is

`VT_CHECK(w.dtype == x.dtype)` in `RmsNormKernelCuda` is not an arbitrary
restriction. `RmsNormRowKernel` is declared
`template <typename Tin, typename Tout, typename Tres>` and takes the gamma as
`const Tin* w` — **the activation's type**. The dispatcher's equality check
mirrors that weld exactly, so widening the check on its own would read a bf16
gamma buffer through a `const float*` and walk off the end of a `[128]`
allocation. The check is honest; the kernel behind it is too narrow.

**Neither side of the mismatch is the defect.** Both operands are the value
upstream says they should be:

- **The activation is bf16, and must be.** `qwen4_exp_qsa_block.cpp:446`
  refuses anything else (`VT_CHECK(hidden.dtype == DType::kBF16, ...)`), which
  mirrors vLLM's own refusal at
  `vllm/models/qwen4_exp/nvidia/qsa.py:188-189`
  (`if model_config.dtype != torch.bfloat16: raise NotImplementedError(
  "Qwen4Exp QSA currently requires BF16")`), read at vLLM `origin/main`
  `25efcfa788`. vLLM reaches `q_norm` through a plain `torch.chunk`
  (`vllm/model_executor/models/qwen3_next.py:426-435`), which does not change
  dtype, so upstream's norm input is bf16 too.
- **The gamma is f32, and must be.** Every norm tensor in the released
  `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S artifact is stored `F32`:
  `blk.N.attn_q_norm.weight` `[256]`, `blk.N.attn_k_norm.weight` `[256]`,
  `blk.N.indexer.q_norm.weight` `[128]`, `blk.N.indexer.k_norm.weight` `[128]`,
  `blk.N.hc_*_norm.weight` `[10240]`, `blk.N.ple_norm_*.weight` `[10240]`.
  Measured by parsing the three shards' GGUF tensor tables directly.

vLLM never requires the two to agree. `GemmaRMSNorm` upcasts the gamma on its
own (`ple_layer.py:80`: `normalized * (1.0 + self.weight.float())`), and the
`vllm.cpp` **CPU** kernel already does exactly that: `cpu_ops.cpp:554-557`
widens `w` through `WidenRowToF32(w.dtype, ...)` for f32/f16/bf16, and
`cpu_ops.cpp:576` widens `x` separately. That is why the CPU control produces
tokens on the same checkpoint where CUDA refuses.

The tree also already carries the correct CUDA pattern one file over:
`RmsNormGroupKernelCuda` passes an **independent** weight tag
(`cuda_rms_norm_group.cu:194`, `w_tag = TagOf(weight.dtype, "weight")`). That
is why the PLE and hyper-connection grouped norms — the same f32 gamma against
the same bf16 stream — pass, and why only the plain `RmsNorm` refuses.

`RmsNormRowKernel` is therefore the outlier among three siblings that all get
it right.

### Why `ResidentWeightF32` is not the fix here

`qwen3_5.cpp:1226` exists for the mirror-image pairing (f32 activation, gamma
on disk at the checkpoint dtype) and `qwen3_5.cpp:5329-5337` selects between
the raw and the upcast weight per site. That precedent does not reach this
case. It can only widen a gamma to f32; the failing site has a **bf16**
activation, so matching it would mean *narrowing* an f32 gamma to bf16 — a
precision loss vLLM does not have, applied to the value `GemmaRMSNorm`
explicitly calls `.float()` on.

## Which call site fires first

All three plain-`vt::RmsNorm` call sites in `qwen4_exp` are in the QSA block,
and the block is layer 3 of the repeating `3 x linear -> 1 x QSA` pattern:

| line | activation | dtype | gamma | verdict |
|---|---|---|---|---|
| `qwen4_exp_qsa_block.cpp:632` | `q_index_raw` (`:592`, `hidden.dtype`) | bf16 | f32 | **refuses, and is first** |
| `qwen4_exp_qsa_block.cpp:705` | `q_f32` (`:694`, `DType::kF32`) | f32 | f32 | passes |
| `qwen4_exp_qsa_block.cpp:736` | `k_raw` (`:732`, `hidden.dtype`) | bf16 | f32 | refuses |

`idx_k_norm` reaches the same kernel through `Qwen4ExpQsaIndex` (`:690`).

## Design

Give `RmsNormRowKernel` its own gamma type:

```
template <typename Tin, typename Tw, typename Tout, typename Tres>
__global__ void RmsNormRowKernel(Tout* out, const Tin* x, const Tw* w, ...)
```

`Load()` is already overloaded for `float` and `__nv_bfloat16`
(`cuda_ops.cu:51-52`), so the kernel body is unchanged. `LaunchRmsNormRes` and
`LaunchRmsNorm` forward `Tw`; `RmsNormKernelCuda` replaces the equality check
with a dispatch over the gamma's dtype and refuses an unsupported one by name.

Neither vectorized fast path changes. `TryLaunchRmsNormDecodeFast` already
requires `w.dtype == DType::kBF16` and `TryLaunchRmsNormDecodeFastF32` requires
`w.dtype == DType::kF32`, so both self-guard and keep their bit-identity.

## Risks

- **A too-wide dtype is invisible to a token gate.** This change widens no
  buffer: it reads the gamma the checkpoint already stores, at its own width,
  and adds no `f32` model-path allocation. The activation stays bf16.
- **Silently reading the wrong bytes.** This is the failure the current check
  prevents, so the check is narrowed rather than deleted: an unsupported gamma
  dtype is still refused, by name.
- Instantiation count goes from 8 to 16. Compile-time only.

## Tests

Red before green: a `RmsNorm` with bf16 `x`/`out` and an f32 `w` must produce
the CPU kernel's values rather than a refusal, on every registered device.

## Gates

- `tests/vt/test_ops_rmsnorm*` focused, then the full gate.
- End to end: `--device cuda` on the released UD-IQ1_S artifact, token ids
  compared against the CPU control `11751 13 15767 411 2029 11 1092 369`.

## Stop conditions

A third wall after #2477 and #2476 clear is a reportable result, not a failure.

## Owed

- `src/vt/rocm/rocm_rmsnorm.hip:105-177` carries the identical weld
  (`RmsNormRowKernel<Tin, Tout, ...>` with `const Tin* w`, and
  `VT_CHECK(w.dtype == x.dtype)` at `:170`). No ROCm device is in the fleet, so
  this wave cannot build or gate it; editing an untestable backend blind is
  worse than recording it.
- [#2476](https://github.com/mudler/vllm.cpp/issues/2476), the illegal memory
  access, whose ordering against this refusal this wave measures.

## Now

`ACTIVE`.
