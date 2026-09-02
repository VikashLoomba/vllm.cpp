# ROCM-FUSED-NORM-ROPE — `vt::FusedNormRope` on ROCm, so GLM-5.3 reaches a token on `gfx1151`

Row: `BACKEND-ROCM`
Issue: [#2564](https://github.com/mudler/vllm.cpp/issues/2564)

## Now

`ACTIVE`. Base `4fe3852b119e40cda05c0fbcf64d3e2a4796ada2`.

## Scope

Register a native `kFusedNormRope` kernel for `DeviceType::kROCM`, and correct
the MLA block's stale comment and refusal message so both causes of the split
A-projection branch are named.

Out of scope: the other seven missing MLA/DSA ops
(`.agents/specs/rocm-glm53-dsa.md` W1.3). They keep serving from the portable
reference tier and are recorded under `## Owed`.

## The defect

`src/vllm/model_executor/layers/attention/mla_attention.cpp:550-551`:

```cpp
const bool fused_nr = R > 0 && !has_k_rope_norm && MlaFusedNormRopeEnabled() &&
                      vt::OpRegistered(vt::OpId::kFusedNormRope, d.q.device.type);
```

`vt::OpRegistered` is a native-only probe by design
(`src/vt/op_provider.cpp:779-803`) and ROCm registers no `kFusedNormRope`. With
every environment variable unset, `fused_nr` is therefore false on `gfx1151`,
the split path row-slices `kv_a_proj_with_mqa`, and a `q8_0` weight has no row
slice — so GLM-5.3's first forward throws. The throw's own comment
(`:628-631`) says the only way to reach it is `VT_MLA_FUSED_NORM_ROPE=0`. That
enumerates the backends that HAVE the op and forgets the ones that do not; the
measured run in #2564 is the counterexample.

## Why repair 1, and not 2 or 3

#2564 prices three repairs. This spec takes the first.

1. **Port `kFusedNormRope` to ROCm** — taken. Both halves of the composite are
   already native ROCm kernels: the latent RMS reduction is
   `rocm_rmsnorm.hip`'s `RmsNormRowKernel` and the decoupled-pe rotation is
   `rocm_dense_basic.hip`'s `RopeFromCacheK`. The port is the same
   composition CUDA already makes, it makes the predicate true honestly, it
   runs the work on the device rather than the host, and it decrements the
   reference-tier hit count `docs/ROCM.md:60-61` gates on.
2. **Let `fused_nr` consider the reference tier** — rejected. It changes what
   "available" means at a shared seam for every backend and every op, and
   `src/vt/op_provider.cpp:779-786` states the contract it would break: a
   unified accelerator would report every op registered the moment its fallback
   installed, and the fused-recipe ladder would stop choosing its portable
   composite path. It also keys naturally on host-addressability, and the two
   host-addressability predicates on this board answer differently:
   `DeviceMemoryIsHostAddressable()` is true (`rocm_backend.hip:371`, which is
   what makes the reference tier eligible) while
   `HostMemoryIsDeviceAddressable()` is false, because `gfx1151` reports
   `pageableMemoryAccess=0` (#2515, measured twice on hardware). A predicate
   written against the wrong one of those two reads plausible and answers
   backwards on the only board that can test it.
3. **Teach the split path to slice a block-quantized merged row** — rejected
   for this wave. It is the largest of the three and it repairs a fallback that
   nobody wants taken: the fused arm is bit-identical and one launch cheaper.
   It stays owed, because a backend that registers neither op still needs it.

## Upstream anchors

Read on the pinned oracle, `~/_git/vllm` @ `5559679229`
(`.agents/upstream-sync.md`). vLLM composes the same two steps, unfused, in the
MLA A-projection:

- `vllm/model_executor/layers/mla.py:164-165` — `kv_c, k_pe =
  kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)` and then
  `kv_c_normed = self.kv_a_layernorm(kv_c)`. That is the LATENT half of the
  fused kernel, and the split it performs is exactly the row split the local
  block cannot make on a block-quantized weight.
- `vllm/model_executor/layers/mla.py:175-177` — `self.rotary_emb(positions,
  q[..., self.qk_nope_head_dim:], k_pe)`. That is the ROPE half, applied to the
  trailing slice of the SAME merged row and to nothing the latent half touches,
  which is why fusing the two is arithmetically inert.
- `vllm/model_executor/models/deepseek_v2.py:512-518` — the merged weight is
  `ReplicatedLinear(hidden_size, kv_lora_rank + qk_rope_head_dim)`, which fixes
  the `[L + R, H]` row shape both arms of the local branch assume.
- `vllm/model_executor/models/deepseek_v2.py:1930` — `class
  GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)`, which
  `vllm/model_executor/models/registry.py:117` maps the checkpoint's
  architecture string to. **Not `deepseek_v32.py`:** at this pin that file does
  not carry the registration, and a reader sent there finds nothing.

vLLM has no fused kernel for the pair, so upstream is the BEHAVIOURAL reference
and the in-tree CUDA sibling is the STRUCTURAL one, exactly as
`vt::FusedNormRope`'s own contract already records
(`include/vt/ops.h:3388-3410`).

Ported `file:line`, verbatim composition:

- `src/vt/cuda/cuda_ops.cu:1163-1245` — `FusedNormRopeKernel`,
  `LaunchFusedNormRope`, `FusedNormRopeKernelCuda`.
- `src/vt/rocm/rocm_rmsnorm.hip:67-98` — the `__syncthreads()` shared-memory
  tree reduce and the scale loop, reproduced element for element. It is
  wavefront-width agnostic, which is the property that makes it portable to a
  64-lane wavefront at all.
- `src/vt/rocm/rocm_dense_basic.hip:607-635` — the cache read and the
  neox/gpt-j pairing of `RopeFromCacheK`, reproduced element for element.

## Design

One new translation unit, `src/vt/rocm/rocm_mla_fused_norm_rope.hip`, holding
one kernel and its `FusedNormRopeFn` entry point, registered in
`rocm_ops.hip`'s `Registrar` and listed twice in `CMakeLists.txt` (the source
list and the `HIP_ARCHITECTURES` property list).

Block width stays 256, as `rocm_rmsnorm.hip:41` fixes it and for the reason
stated there: it is four whole wavefronts AND it keeps the reduction order
identical to the CUDA and CPU siblings, which is what keeps the NMSE bar
meaningful. Grid is one block per token, as CUDA's is.

The two halves address disjoint dims, so the fused output is the composite of
`RmsNorm(x[:, :off])` and `RopeFromCache(x[:, off:])` by construction, not by
inspection.

## Risks

- **A wavefront assumption.** The donor reduction uses no warp-level primitive,
  so 64-lane wavefronts are safe; the gate below measures it rather than
  asserting it.
- **A partial MLA arm hides which half ran.** With the reference tier eligible
  a half-ported arm still emits tokens. Mitigated by requiring
  `VT_OP_PROVIDER_STATS=1` on every leg and reporting the reference-tier hit
  count beside any token (#2505's silent-fallback failure).
- **No speed claim is admissible** from this board for this model while the
  hit count is non-zero (`docs/ROCM.md:60-61`). None is made.

## Tests

- `tests/vt/test_backend_cross_device.cpp` — a new `FusedNormRope` case,
  against the CPU oracle at NMSE <= 5e-4, in both rope styles and both
  dtypes, on every backend that registers the op. It is SKIPPED on a build
  where no device registers `kFusedNormRope`, which is stated plainly rather
  than counted as a pass.
- The e2e leg on `strix:gpu0` is the reachability gate, through the production
  entry point (`vllm-cli` -> `vllm_engine_load` -> `ModelRegistry::Forward`),
  never a by-hand construction.

## Gates

1. `ctest` for the focused unit target on `strix:gpu0`, case AND assertion
   counts both read.
2. GLM-5.3 `UD-IQ1_S` through `vllm-cli --device auto` with `VT_CPU_MOE=1`,
   greedy, on `strix:gpu0`: generated text printed verbatim, with the
   reference-tier hit count beside it.
3. Reachability mutation: remove the `RegisterOp(OpId::kFusedNormRope,
   DeviceType::kROCM, ...)` line, REBUILD, rerun the e2e leg. It must throw
   #2564's message again. Restore, verify by sha256, rebuild, rerun.

## Evidence

Recorded in `## Outcome` when the row reaches `DONE`.

## Stop conditions

- The board faults or resets in a way that is a property of the board rather
  than of this change (#2546 measured 12/12 GPU resets for a gate-sized native
  run). Report it as such; do not paper over it.
- A second MLA op turns out to block GENERATION rather than merely make it
  slow. Return `NEEDS_DECISION` naming which ops are in which class.

## Owed

- The seven remaining MLA/DSA ops on ROCm — `kFusedChain`, `kBatchedMatmul`,
  `kConcatAndCacheMla`, `kConcatMlaNopeRope`, `kDsaIndexerLogits`,
  `kDsaTopkSelect`, `kGatherMlaCache`, `kMlaDecodeAttention`. Each has a CPU
  registration, so each serves from the reference tier on this host-addressable
  board and none of them refuses. They are what makes a speed result
  inadmissible here. Owned by `BACKEND-ROCM`, recorded in
  `.agents/specs/rocm-glm53-dsa.md` W1.5 as a campaign this wave does not open.
- Repair 3 of #2564 — a block-quantized row slice for the split path — for a
  backend that registers neither `kFusedNormRope` nor a native alternative.
