# ROCm attention parity: the value-dtype probability and the FP32 Q/K carrier

Owning row: `BACKEND-ROCM-BF16-MOE`
Issue: [#3115](https://github.com/mudler/vllm.cpp/issues/3115), which asks for the
decode attention and Q/K preamble parity of the BF16 MoE row.
Base: `cd1ab3909a25d58a9676afa08263366b09ee0cb8` on
`row/BACKEND-ROCM-BF16-MOE-attn-parity`.
Integration: one pull request, with this spec committed before any product edit.
The test-only replay instrument `e5c7bfc4d` is already at the base and is the
measurement this repair answers.

## Now

State: `ACTIVE`. The first attention output of the L33/C2/R0 workload differs from
the pinned primary by 2902 BF16 words, and the committed instrument has now
measured both candidate terms on identical inputs: the ROCm kernel contributes
2918 of 8448 words with the primary's own Q/K/V replayed through it, and the Q/K
preamble contributes 1569 Q and 1542 K words. This spec designs the repair of both
and scopes every sibling site of the same arithmetic.

## Problem and scope

Two arithmetic differences produce the 2902-word parity gap on native gfx1100 at
layer 0 step 0 of the L33/C2/R0 workload (66 tokens, `hq = hkv = 1`, `dh = 128`,
`scale 0.08838834764831845`).

1. **The kernel keeps the softmax probability in f32.** The primary narrows it to
   the value dtype before the value dot and keeps only the running sum in f32.
   `PagedAttnDecodeOptBf16T` accumulates `o_reg = o_reg * corr + pw * v_reg` with
   `pw` in f32 (`src/vt/rocm/rocm_paged_attn.hip:527-531`).
2. **The preamble narrows the normalized Q/K to BF16 before RoPE.** The primary
   carries the normalized value in f32 through the rotation and narrows once, at
   the store, reading a BF16 cos/sin cache. The native preamble rounds twice.

Both repairs are mirroring repairs. Neither invents behavior: each replaces a
native rounding decision with the primary's own.

**In scope.** `src/vt/rocm/rocm_paged_attn.hip` and the ROCm attention preamble,
plus the test cases that witness them, plus this spec.

**Out of scope.** The CUDA, Metal and CPU realizations of the same two
arith-metics; the two rocWMMA attention kernels; the Tier-0 fused-chain composite
for `kAttnQkNormRope`. Each is named under `## Sibling sites` or `## Owed` with
its reason and is not silently changed.

## Primary anchors

Pin `e126687a9a828d513c01a07cd69f025f27d63280`
(`.agents/upstream-sync.md` §"parity-pin"); source tree
`/home/vikash/oracle/gfx1100-active-2773/source`.

The executing primary arm for this workload is Triton, not the ROCm custom
kernel: `vllm/platforms/rocm.py` gates its decode predicate on `gqa_ratio >= 3`
and this fixture has `qg == 1`, so `chunked_prefill_paged_decode.py` logs the
fallback and a 33-token request (`max_query_len > 1`) enters
`prefix_prefill.py::context_attention_fwd`.

| fact | anchor |
|---|---|
| prefill softmax probability, f32 | `vllm/v1/attention/ops/prefix_prefill.py:443` |
| prefill running sum taken BEFORE the narrowing | `prefix_prefill.py:445` |
| prefill probability narrowed to the value dtype | `prefix_prefill.py:471` |
| prefill value dot accumulates the narrowed probability | `prefix_prefill.py:473` |
| prefill denominator update keeps the f32 sum | `prefix_prefill.py:475` |
| prefill epilogue divides by the denominator | `prefix_prefill.py:478` |
| decode probability narrowed inside the dot | `chunked_prefill_paged_decode.py:265` |
| decode running sum taken BEFORE the narrowing | `chunked_prefill_paged_decode.py:251` |
| Qwen3 q/k norm then RoPE | `vllm/model_executor/models/qwen3.py:150-167` |
| the cos/sin cache is narrowed to the query dtype | `vllm/model_executor/layers/rotary_embedding/base.py:105-131` |
| the compiled preamble reads that BF16 cache | `RotaryEmbedding.forward_static`, `base.py:150-190` |

The last three rows are why the primary's boundary is
`bf16(rope_f32(rmsnorm_f32(bf16 qkv)))` against a BF16 cos/sin cache. The
validated CPU transcription of that boundary
(`/home/vikash/.cache/attn-parity-plan/check_attention_models.py`, model A)
reproduces the primary's captured output **exactly, 0 of 8448 words**, and its
preamble model reproduces the primary's captured Q to **1 of 8448 words**. Those
two numbers license the CPU model as the primary-arithmetic reference without
executing the primary, and they are the targets this repair is measured against.

## Measurement this repair starts from

Operator-verified and independently reproduced, identical in
`/home/vikash/.cache/moe-attn-parity/gpu-run-at-head.log` and
`operator-rerun.log`; 1322 of 1322 assertions, 1 case, 3 skipped. Native gfx1100,
L33/C2/R0 layer 0 step 0.

| row | meaning | words / 8448 |
|---|---|---|
| A0 | native run output vs primary output | 2902 |
| B | native run post-RoPE Q / K vs primary Q / K | 1569 / 1542 |
| C0 | replay self-consistency, native Q/K/V at the primary's geometry | 0 |
| C | native `PagedAttention` on the PRIMARY's own Q/K/V vs its output | 2918 |
| D | native `RmsNorm`+`RopeFromCache` on the primary's qkv vs primary Q/K | 1569 / 1542 |
| D' | the same through `RopeNeox` | 1776 / 1719 |
| E | native cos/sin table vs primary `cos_sin` | 0 |
| F | primary qkv q/k slices vs native pre-norm q/k | 0 |

C0 is the guard: the replay reproduces the native run's own output byte-for-byte
at the primary's cache geometry, so row C measures the kernel and not the
transplant. E and F exclude the cos/sin source and the projection, so the
preamble term is exactly the pre-RoPE BF16 store.

## Design 1 — the value-dtype probability

Mirror `p.to(v.dtype)` and nothing else:

* the value accumulate multiplies a probability narrowed to the **value dtype**;
* `corr` (the primary's `alpha`) stays f32 — `prefix_prefill.py:449`, `:258`;
* the running sum keeps the f32 probability — `prefix_prefill.py:445` against
  `:471`, and `chunked_prefill_paged_decode.py:251` against `:265`;
* the warp-combine rescale stays f32, because the primary's `acc = acc * alpha`
  is f32 at `:449`/`:258` and has no narrowed analogue.

One device helper expresses the narrowing for both value dtypes the file uses
(`src/vt/rocm/rocm_paged_attn.hip` uses only `float` and `__hip_bfloat16`; there
is no f16 KV path in the file):

```cpp
template <typename TV>
__device__ inline float ProbInValueDtype(float p) {
  if constexpr (std::is_same_v<TV, __hip_bfloat16>) {
    return __bfloat162float(__float2bfloat16(p));  // p.to(v.dtype)
  } else {
    return p;  // p.to(f32) is the identity
  }
}
```

Narrowing to f32 is the identity, so a site templated on the value dtype is
correct for both arms and no dispatch is added.

### Sibling sites in `src/vt/rocm/rocm_paged_attn.hip`

| site | kernel | value dtype | disposition |
|---|---|---|---|
| `:299` | `PagedAttnOnline` | `TKV` | **changed** — the generic fallback; the primary's rule is uniform |
| `:527` | `PagedAttnDecodeOptBf16T` | bf16 | **changed** — the measured site; d=128 at `:2242`, d=256 at `:2249`, d=512 at `:2256` |
| `:668` | `PagedAttnDecodeGqaBf16` | bf16 | **changed** — the QG-fused decode sibling (`:2213`, `:2220`, `:2227`) |
| `:812` | `PagedAttnDecodeGqaF32Q` | `TKV` | **changed** — the f32-query / bf16-KV arm |
| `:1138` | `PagedAttnPrefillFlashTile` | bf16 | **changed** — scalar online-V branch of the flash tile |
| `:1608` | `PagedAttnPrefillSharedK` | `TKV` | **changed** — the scoreless shared-K prefill |
| `:1367` | `PagedAttnPrefillWmmaWave` | bf16 | **owed** — body compiles only under `VT_ROCWMMA_OK` (`:8-10`, gfx1200/1201), so on gfx1100 neither the edit nor its result can be compiled or run here |
| `:1944` | `PagedAttnPrefillSharedKWmma` | bf16 | **owed** — same guard, and the host admits it only through `PrefillSharedKWmmaHostOk()` on gfx1200/1201 (`:2077-2080`) |

The two owed sites are also both reachable only from default-OFF lab toggles
(`VT_ATTN_PREFILL_FLASH`, `VT_ATTN_PREFILL_SHAREDK_WMMA`). They are recorded
under `## Owed`, not silently skipped.

**Not a site.** `FastExp` (`:196-198`) and the warp-strided key order (`:512`)
are native traits with no primary analogue in the executing arm — the primary's
key reduction is a `tl.dot` over a whole tile. They are left alone; if they hold
residual words after this repair they are attributed, not rewritten.

## Design 2 — the FP32 Q/K carrier through RoPE

The native preamble today is `RmsNorm` into a BF16 buffer followed by
`RopeFromCache` over that buffer (`include/vt/recipes.h:357-385` binds the 2-D
norm view and the 3-D rope view to the same BF16 buffer; the hand-call at
`include/vllm/model_executor/models/dense_attn_block.h:642-648` is the same
sequence). On ROCm the fused branch at `dense_attn_block.h:582-613` executes that
composite and not a fused kernel, because `src/vt/rocm/rocm_ops.hip:321`
registers `OpId::kAttnQkNormRopeGate` and **not** `OpId::kAttnQkNormRope`, so
`vt::FusedChain` takes its `OpRegistered` guard (`src/vt/ops.cpp:1448-1452`) and
falls through to `FusedChainComposite` (`src/vt/ops.cpp:1468`).

**Design: register the recipe's own fast op on ROCm, with the f32 carrier.**

`vt::OpId::kAttnQkNormRope` is registered for `DeviceType::kROCM`, implemented by
`AttnQkNormRopeKernelRocm` in `src/vt/rocm/rocm_ops.hip`. The kernel:

1. loads the raw `q3`/`k3` row (BF16), the per-head norm weight (BF16) and the
   cos/sin cache row (BF16 at the position `positions[token]` supplies);
2. computes `mean(x^2)` over the head with the **same** f32 accumulation and the
   same `kBlock` binary-tree shared reduction the shipped `RmsNormRowKernel` uses
   (`src/vt/rocm/rocm_rmsnorm.hip:118-147`), then
   `inv = 1 / sqrtf(mean + eps)`;
3. forms `v * inv * w[j]` in f32, rotates the first `rot` elements in f32 exactly
   as `RopeFromCacheK` does (`src/vt/rocm/rocm_dense_basic.hip:699-704`, NeoX and
   GPT-J pair orders both), and narrows **once**, at the store, with the bf16
   store helper the file already uses;
4. leaves elements at or beyond `rot` normalized and narrowed once, which is the
   same BF16 word the shipped `RmsNorm` store produced.

The kernel is modelled on the registered sibling `AttnQkNormRopeGateK`
(`src/vt/rocm/rocm_gdn_fused.hip:95-169`), which is already a fused
norm-plus-partial-NeoX-RoPE preamble on this backend. The `attn_f32` arm passes
f32 states, an f32 norm weight and the f32 cache; for that arm the kernel's
arithmetic is the shipped `RmsNorm` + `RopeFromCache` arithmetic with no
intermediate rounding at all, so its bytes are unchanged by construction.

**Both realizations must agree.** With the op registered, `vt::FusedChain` takes
the fast path for the fused branch, and the hand-call fallback at
`dense_attn_block.h:645-653` is changed to dispatch to the same
`vt::AttnQkNormRope` when the op is registered on the device and the bf16 cache
is in use. On a backend that registers no fast op (CPU), and on the `RopeNeox`
default branch (no cache), both realizations keep exactly today's sequence. So
the fused path and the hand-call fallback agree byte-for-byte on every backend,
which is what the recipe's byte-exact composite contract asks for
(`include/vt/recipes.h:378-383`) and what `tests/vllm/models/test_qwen3_forward.cpp`
already gates for the adoption switch.

**The Tier-0 composite stays as it is, and is owed.** Carrying f32 through the
composite would require an operand slot that holds an f32 intermediate
(`kMaxFusedOperands` is 8, `include/vt/fused_recipe.h:107-108`, and the recipe
already spends all eight) plus a `RopeFromCache` that reads an f32 state against
a BF16 cache. That is a recipe and op-contract change on four backends, which
this row does not own. On ROCm the composite is no longer the executing path
once the fast op is registered; every other backend keeps today's bytes. This is
recorded under `## Owed`.

## Risks

* **This changes the numerics of existing users of the six attention kernels.**
  Every ROCm paged-attention arm whose value dtype is bf16 moves toward the
  primary and away from its previous bytes. Committed ROCm device goldens and
  near-tie anchors on those arms may need re-derivation; the production gate
  `test_rocm_moe_bf16` is re-run in this change and any moved golden is reported.
* **This changes the numerics of the ROCm BF16 Qwen3-dense preamble** for every
  model with a qk-norm and the cos/sin cache (which is on by default,
  `dense_attn_block.h:92-98`). The `attn_f32` arm is unaffected by construction.
* **The kernel term may not reach exactly zero.** The primary's value dot is a
  tile `tl.dot` with its own accumulation order; the native arm keeps `FastExp`,
  a warp-strided key walk and a reciprocal epilogue. A residual of a few words is
  possible and must be attributed before any further edit.
* **A greedy anchor can move.** The change alters bf16 words; a token is decided
  by the full logits, so movement is possible in principle. The gate's recorded
  tokens are checked and any movement is reported rather than papered over.
* **Performance.** The preamble repair replaces three launches per layer with
  one; the kernel repair adds one bf16 round per key. Neither is a perf claim and
  no perf axis is accepted by this change.

## Tests

**Red first, focused, in the committed instrument**
(`tests/vllm/models/test_rocm_moe_bf16.cpp`, case "ROCm paged attention replays
the primary's captured attention boundary"):

* kernel: `CHECK(replay_diff.different == 0)` — red at 2918 before Design 1;
* preamble: `CHECK(run_q.different <= 1 && run_k.different <= 1)` — red at
  1569/1542 before Design 2;
* preamble on the primary's own qkv through the production op, which must reach
  the CPU model's 1 word — red before Design 2.

The instrument gains an observer on `OpId::kAttnQkNormRope` for the pre-norm and
post-RoPE native bytes, because after Design 2 the production preamble no longer
calls `kRopeFromCache`; the existing `kRopeFromCache`/`kRmsNorm` observers stay
and now measure the composite fallback.

**Mutation.** Each repair is reverted in place, its focused case must redden, and
the file is restored with a sha256 check.

**Suites.** The instrument's CPU-only case (no device), the instrument's device
case under `flock /home/vikash/gpu.lock` with `HIP_VISIBLE_DEVICES=0`,
`tests/vt/test_ops_paged_attn.cpp`, and the production gate
`tests/test_rocm_moe_bf16` with `VT_ROCM_MOE_FIXTURE`, `VT_ROCM_MOE_ORACLE` and
`VT_FUSED_CHAIN_ADOPT=1`.

## Gates and evidence

* Focused red and green runs, with the exact command and exit status.
* The device run's per-row numbers for C, B, D and D2, in the same table shape
  as `## Measurement this repair starts from`.
* The production gate's recorded tokens, compared to
  `[66,1,70,57,33,81,63,69]` at length 33, concurrency 2, request 0.
* `TMPDIR=/home/vikash/.cache/moe-attn-parity/tmp-preflight
  GIT_CONFIG_GLOBAL=/dev/null GIT_CEILING_DIRECTORIES=$TMPDIR
  PYTHONPATH=/home/vikash/.cache/rdna3-moe-impl/numpy-only-python`, the `python3`
  wrapper adding `--jobs 4` to `check-tree-compiles.py`, stdin closed, with the
  exit line appended to `/home/vikash/.cache/moe-attn-parity/preflight-repair.log`.

## Owed

* The two rocWMMA attention kernels (`rocm_paged_attn.hip:1367`, `:1944`) keep
  the f32 probability until a gfx1200/1201 host can compile and run them.
* The Tier-0 `kAttnQkNormRope` composite keeps the pre-RoPE BF16 store. It is the
  realization on every backend without a registered fast op, and changing it
  needs an f32 operand slot in the recipe and a mixed-dtype `RopeFromCache`.
* The CUDA, Metal and CPU realizations of both arithmetics are not repaired by
  this change and are not measured here.
* `FastExp`, the warp-strided key order and the reciprocal epilogue remain native
  traits; only a measured residual would justify touching them.

## Stop conditions

* If row C does not reach 0, the residual is attributed to one of the three named
  remaining traits before any further product edit, or the repair returns with
  the residual reported as an open gap.
* If the production gate's recorded tokens move, the golden is reported as moved
  with the failing positions; it is never silently re-derived.
* If the primary's boundary cannot be reproduced without changing an op contract
  outside this row, the preamble repair stops and returns `BLOCKED` with the
  contract named.
