# Native BF16 grouped MoE on ROCm

Owning row: `BACKEND-ROCM-BF16-MOE`

Owner: the `BACKEND-ROCM-BF16-MOE` implementer, reviewer, and coordinating operator.
Parent row: `BACKEND-ROCM`, whose lifecycle stays `ACTIVE`.
Issue: [#3094](https://github.com/mudler/vllm.cpp/issues/3094), local record
`ISSUE-GH-3094`.
Origin: [#1928](https://github.com/mudler/vllm.cpp/issues/1928), local record
`ISSUE-GH-1928`.
Base: `6db4bef906859e864c82523c01107473f7dcca29`.
Integration: one pull request, with this spec committed before implementation.
The implementation pull request carries closing keywords for both issues.
This session ends with a published, independently reviewed pull request.
The operator does not merge it without separate developer authorization.

## Now

State: `SPIKE`. Source assessment is complete. Implementation and hardware gates
are `PENDING`. This spec does not claim a measured MoE execution on the oracle.

The coordinating operator dispatches a fresh implementer from the committed spec.
The implementer first captures a failing production reachability test on `gfx1100`.
A fresh reviewer then tests the immutable implementation through negative mutations.
The operator reruns the declared hardware gate before accepting the result.

## Problem and scope

ROCm registers neither `kMoeGroupedGemmBf16` nor
`kMoeGroupedGemmBf16GateUpSilu` at the base revision.
The existing BF16 mixture of experts (MoE) fast path therefore remains unavailable
on ROCm. Its eligibility check probes the unfused operation but calls both operations.
The two providers must become available together.

Implement native HIP providers for grouped BF16 matrix multiplication and fused
gate, up, and SiLU multiplication. Route the feature through the existing shared
operations and Qwen3 MoE production path. Preserve resident expert pointer arrays
and the device router output. Do not add a host gather loop.

The pinned ROCm oracle rounds intermediate BF16 tensors differently from the
existing CUDA sibling contract. Add the minimum shared numeric modes needed to
represent those differences. Preserve the current default semantics for every
existing caller that does not select the new mode.

This work is independent of the RDNA3 quantized dot changes in pull request
[#2782](https://github.com/mudler/vllm.cpp/pull/2782). BF16 expert weights do not
reach the quantized RDNA3 dot kernel. Do not depend on that pull request or
change its implementation. Do not change continuous integration configuration,
checker semantics, or the upstream pin.

This row adds a backend capability to an existing model path. It does not add
a model architecture or a quantized arm. GGUF, FP8, integer quantization, expert
parallelism, LoRA, and expert load balancing remain outside this change.
Do not alter their acceptance, refusal, or numeric behavior.

## Inventory

The source anchors in this spec use the immutable revisions stated here.
Line numbers refer to those revisions, before implementation changes them.

| Stable ID | Upstream source | Local anchor | Required test and evidence | State |
|---|---|---|---|---|
| `ROCM-BF16-MOE-GROUPED` | `vllm/model_executor/layers/fused_moe/fused_moe.py:299-610,763-910` | `include/vt/ops.h:3198-3238`, new `src/vt/rocm/rocm_moe_grouped_bf16.hip` | Native grouped suite, oracle buffers, provider selection, generated kernel | `SPIKE` |
| `ROCM-BF16-MOE-GATEUP` | `vllm/model_executor/layers/fused_moe/experts/triton_moe.py:388-409,487-527`, `csrc/libtorch_stable/activation_kernels.cu:44,165-177` | `MoeGroupedGemmBf16GateUpSilu`, shared numeric descriptor | BF16 boundary fixtures, legacy byte comparison, native oracle comparison | `SPIKE` |
| `ROCM-BF16-MOE-WEIGHTED-DOWN` | `vllm/model_executor/layers/fused_moe/fused_moe.py:593-610`, `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu:395-459` | Shared grouped down and `MoeCombine` descriptors | Weight-before-narrowing and no-double-weight mutations | `SPIKE` |
| `ROCM-BF16-MOE-FORWARD` | `vllm/model_executor/models/qwen3_moe.py:199-237` | `src/vllm/model_executor/models/qwen3_moe_registry.cpp:63-86`, `src/vllm/model_executor/models/qwen3_5.cpp:6869-7010,7198` | `ModelRegistry::Load` and `ModelRegistry::Forward`, token gate, call-site mutation | `SPIKE` |

All inventory items belong to this spec and `ISSUE-GH-3094`.
The required tests and evidence are planned obligations, not completed measurements.

## Upstream contract

### Pin and executing chain

The primary oracle is vLLM at
`e126687a9a828d513c01a07cd69f025f27d63280`.
[Upstream sync](../upstream-sync.md) owns the repository pin.
The inspected checkout contains that exact revision.
Use the operator's recorded oracle runtime and GPU authority from the shared
environment. Machine paths in another spec are not defaults.

The executing source chain is:

1. `vllm/model_executor/models/qwen3_moe.py:199-237` constructs and invokes
   `FusedMoEFactory` for the routed experts.
2. `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:41-130`
   resolves the backend, creates BF16 expert weights, and defines ROCm padding.
3. `vllm/model_executor/layers/fused_moe/oracle/unquantized.py:61-65,208-325`
   orders and selects the unquantized backends.
4. That file's `:329-395` converts weight storage and builds the prepare,
   finalize, and expert operations.
5. `vllm/model_executor/layers/fused_moe/modular_kernel.py:1144,1349`
   allocates activation storage in the model dtype and invokes the expert method.
6. `vllm/model_executor/layers/fused_moe/experts/triton_moe.py:309-330,388-527`
   executes gate/up, activation, and weighted down through BF16 intermediates.
7. `vllm/model_executor/layers/fused_moe/fused_moe.py:763-910` launches the
   kernel with explicit tensor strides and the selected compute type.
8. `vllm/model_executor/layers/fused_moe/fused_moe.py:517-610` accumulates in
   FP32, applies optional route weights, narrows, and stores.
9. `vllm/model_executor/layers/fused_moe/activation.py:196-237` selects
   `torch.ops._C.silu_and_mul` for SiLU.
10. `csrc/libtorch_stable/activation_kernels.cu:44,110,165-177,299` defines
    the BF16 activation and multiplication boundaries.
11. `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu:395-459,759` sums
    already weighted BF16 expert results with an FP32 accumulator.

ROCm's source candidate order is AITER, Triton, then batched Triton.
`vllm/_aiter_ops.py:134,1891` restricts the AITER capability to qualifying CDNA
devices. `gfx1100` is RDNA3. Source inspection therefore predicts ordinary Triton
for this unquantized, single-device workload. This prediction is not runtime evidence.

Before choosing the provider's native default, capture the selected oracle backend,
generated kernel, launch arguments, tensor dtypes, shapes, and strides.
Run the identical workload in the pinned engine. A successful config construction
or an unrelated dense-model capture does not satisfy this gate.
If runtime selection differs, reconcile the executing source before implementation.
Do not replace production defaults with `--enforce-eager` for a denominator.

### BF16 boundaries

For the target path, `apply_router_weight_on_input` is false.
Let `b(x)` mean conversion to BF16 with the oracle's rounding behavior.
For token `t`, selected expert `e`, and route weight `r`, the native semantics are:

```text
g = b(dot_fp32(x[t], Wgate[e]))
u = b(dot_fp32(x[t], Wup[e]))
s = b(silu(float(g)))
a = b(float(s) * float(u))
d = b(dot_fp32(a, Wdown[e]) * float(r))
y[t] = b(sum_in_fp32(d for each selected expert))
```

The dot product's order follows the selected kernel and receives the upstream
comparison tolerance. The BF16 conversion points are mandatory independent of
that tolerance. A token match does not prove the memory or arithmetic format.

The existing CUDA fused sibling computes its gate/up intermediates in FP32.
That legacy mode remains byte-identical to its existing unfused composite.
Its FP32 intermediates are a compatibility exception to the native BF16 mode.
Annotate that reason beside each model-path buffer that keeps FP32 storage.
Do not silently make the old composite the new oracle.

The native down operation multiplies the route weight before narrowing to BF16.
The combine operation sums these preweighted values without multiplying again.
Post-store weighting is numerically different and fails the native contract.

### Weight storage and shapes

Upstream stores gate/up weights as `[E,2I,H]` and down weights as `[E,H,I]`.
Its ROCm padding can add 128 BF16 columns when a row occupies a multiple of
512 bytes. The default `VLLM_ROCM_MOE_PADDING` is enabled in `vllm/envs.py:1341`.
Triton conversion preserves the physical strides in
`vllm/model_executor/layers/fused_moe/oracle/unquantized.py:364-370`.
Record those strides in the oracle capture and retain padding cases in the tests.

The local shared ABI uses one device pointer per expert.
Each pointer names a Matmul-B matrix `[K,N]`, indexed as `k * N + n`.
The loader's transpose from checkpoint storage is an existing harness adaptation.
Do not reinterpret a checkpoint `[N,K]` array as the shared matrix format.
The local ABI does not require unused physical padding bytes to match upstream.
It does require BF16 storage, correct logical values, and explicit recorded strides.

## Shared design

### Operations and selection

Register both existing operation IDs for `DeviceType::kROCM` in the same change.
Keep their current typed call signatures and default semantics intact.
Add an explicit shared numeric descriptor or typed sibling operations for native
gate/up rounding, weighted down, and preweighted combine.
If a descriptor changes a function-pointer signature, use a separate typed sibling
instead of casting an old provider to a new signature.
The exact C++ names are implementation choices. The semantic modes are fixed here.

The Qwen3 MoE path selects the complete native capability through the shared
provider seam. Do not add a model-specific HIP kernel or a device-type branch
that changes model arithmetic. Probe every operation required by the native mode.
Never select half of the native sequence and continue with legacy weighting.
Existing callers that do not select this capability keep their defaults.

The existing row map maps pair `p` to its activation row.
A null row map means identity. Expert IDs and row maps remain on the device.
Inputs are contiguous BF16 activations, contiguous I32 indices, and I64 device
pointer arrays. Outputs accept the existing grouped BF16 and FP32 modes.
Preserve validation in `src/vt/ops.cpp:906-962`.
Preserve the zero-pair and zero-output-width no-op behavior.
Use valid router-produced expert IDs. This row does not add sentinel index semantics.

The fused operation accepts separate gate and up pointer arrays.
Use one grouped provider implementation for all callers, including the legacy mode.
Mask incomplete K and N tiles. Preserve repeated experts, empty experts, arbitrary
pair order, repeated rows, and dimensions that are not tile multiples.
Do not assume a particular top-k, expert count, or model geometry.

### Fusion and production reachability

The existing pointer-array BF16 sibling is a tracked exception to a literal
`vt::MergedGemmGroup` instance.
`include/vt/merged_gemm.h:111-118` documents its different weight representation.
[The fusion spec, Tier A4](arch-fusion-fold-plan-2026-07-30.md) records the
shared operation and its original compatibility contract.
This row extends that shared operation with explicit native numeric modes.
It does not create another exception or a per-model expert loop.
Update that local seam comment to distinguish legacy and native numerics when
the implementation introduces the mode.

Model fusion continues through `vt::FusedChain` where that seam applies.
Mergeable dense and shared-expert projections continue through
`layers::MlpGateUpMethodBase` and `vt::MergedGemmGroup`.
Do not duplicate their existing model orchestration for this backend.

The production vehicle is `Qwen3MoeForCausalLM` loaded from safetensors.
`src/vllm/model_executor/models/qwen3_moe_registry.cpp:63-86` exposes load and forward.
`src/vllm/model_executor/models/qwen3_moe.cpp:71` delegates to the shared MoE block.
`src/vllm/model_executor/models/qwen3_5.cpp:7198` selects the grouped BF16 path.
The resident gate/up and down calls occur at `:6997-6999`.
The existing `VT_MOE_BF16_FAST` default is enabled at `:909`.
Preserve the rollback setting and validate both sides with the same binary.

DeepSeek and dots3-note contain callers of the shared BF16 operations.
Their production routers require currently refused ROCm modes.
`src/vt/rocm/rocm_moe_router.hip:183` rejects grouped routing, correction bias, and non-softmax
scoring. This row does not claim that either complete model becomes runnable.
Keep that router debt under the parent `BACKEND-ROCM` issue
[#41](https://github.com/mudler/vllm.cpp/issues/41).

### Kernel and scratch lifetime

Place new hardware code in `src/vt/rocm/rocm_moe_grouped_bf16.hip`.
Use `src/vt/cuda/cuda_matmul_nvfp4.cu:943-1004,1413-1573` as a local
layout and deterministic-reduction donor. Its legacy arithmetic is not the
native oracle contract. Attribute every ported upstream kernel section at the pin.

Start with a complete deterministic kernel. Optimize only after correctness passes.
If split-K is used, reduce partials in a fixed order and apply conversion once
at the specified boundary. Do not use atomic output accumulation that changes
results across identical runs.

Key reusable scratch by both device and stream. The null default stream does
not identify a unique device. Serialize allocation and publication across threads.
Preserve allocation lifetime until all users and captured graphs finish.
`src/vt/grow_only_stream_scratch.h` supplies the existing growth and retirement
mechanism. Adapt that mechanism instead of copying CUDA globals with weaker keys.
Allocation failure must not publish partial capacity or invalid pointers.

Prewarm the required capacity before graph capture.
Do not allocate, free, or synchronize during capture.
Preserve captured addresses when another invocation grows scratch.
Validate streams independently and devices separately when two devices are available.
A missing second device leaves that case `PENDING`, not silently skipped.
Direct operation capture tests do not enable model graph capability.
This row does not depend on the separate graph-capability pull request #2777.

## Tests and evidence

### Red before implementation

Create a deterministic tiny safetensors fixture through the existing model loader.
Adapt `tests/vllm/models/test_moe_async_device_ids.cpp:111-203,415-477`.
Enter through `ModelRegistry::Load` and `ModelRegistry::Forward`.
Do not construct an internal MoE block by hand for the reachability claim.

Use this concrete Qwen3 MoE configuration:

```json
{
  "architectures": ["Qwen3MoeForCausalLM"],
  "model_type": "qwen3_moe",
  "hidden_size": 128,
  "num_hidden_layers": 2,
  "num_attention_heads": 1,
  "num_key_value_heads": 1,
  "head_dim": 128,
  "intermediate_size": 128,
  "moe_intermediate_size": 128,
  "shared_expert_intermediate_size": 0,
  "num_experts": 4,
  "num_experts_per_tok": 2,
  "norm_topk_prob": true,
  "decoder_sparse_step": 1,
  "mlp_only_layers": [],
  "hidden_act": "silu",
  "vocab_size": 128,
  "max_position_embeddings": 256,
  "rms_norm_eps": 0.000001,
  "rope_theta": 10000000.0,
  "tie_word_embeddings": false,
  "attention_bias": false,
  "torch_dtype": "bfloat16",
  "bos_token_id": 1,
  "eos_token_id": 127,
  "pad_token_id": 0
}
```

Preserve `BuildTensors` ordering from the fixture at the base revision.
Set its first tensor seed to 7 and increment the seed for each tensor.
Preserve `Bf16Bytes` at `:131-142`, including unsigned 32-bit wraparound and
the existing `F32ToBF16` conversion. Keep projection scale 0.08 and norm scale 0.5.
Only substitute the dimensions and initial seed stated here.
Export the generated safetensors and config once, then use those same bytes in
both runtimes. Record each file's byte count and SHA256.

For each length `L` in `{1,3,33}`, request `r` contains token IDs
`1 + ((11 + 29*r + 17*i) % 126)` for `i` from 0 through `L-1`.
Run request 0 at concurrency 1 and requests 0 and 1 together at concurrency 2.
Repeat each workload 3 times with fresh model state.
Use token-ID prompts with tokenizer initialization skipped on the oracle.
Generate exactly 8 tokens per request through the existing shared device sampler.
Set temperature 0, top-p 1, top-k -1, min-p 0, repetition penalty 1,
presence penalty 0, frequency penalty 0, seed 7, and `ignore_eos=true`.
Set both the minimum and maximum generated token counts to 8.
Record selected expert IDs and assert changing expert pairs across the request set.
Keep repeated expert choices in the component fixtures.

Assert provider selection for grouped down and fused gate/up.
Enable `OpProviderCallStats` and require native selections greater than zero,
zero native declines, and zero CPU fallbacks for the tested expert sequence.
On `gfx1100`, absence of the providers is a failure, not a skipped test.
The red result must fail that reachability assertion on the base revision.
Correct fallback tokens alone do not satisfy this test.

### Port upstream cases

Port `tests/kernels/moe/test_moe.py:test_fused_moe` at `:345` from the pin.
Preserve its seed 7, BF16 fixtures, `atol=0.02`, and `rtol=0`.
Retain these `[M,N,K]` parameter sets:

```text
[1,128,128]
[1,2048,128]
[33,2048,128]
[32768,2048,511]
[40000,1024,1024]
```

Retain expert counts 8, 64, and 192, top-k values 2 and 6, and both padding modes.
Run every applicable single-device BF16 combination from the upstream fixture.
The local pointer-array transpose is the required storage adaptation.
Expert parallel size 4 is inapplicable because this row adds no distributed path.
Tensor-descriptor mode is inapplicable on this hardware by
`vllm/model_executor/layers/fused_moe/utils.py:665-684`.
Record those exclusions with their source rationale.
Do not reduce large M or tail dimensions to make the tests fit.
Name the resource and retain the gate as `PENDING` if a required case cannot run.

Preserve the upstream reference decomposition, activation mode, route-weight
placement, and output comparison. Record generated kernels before interpreting
any unavailable lever. A scratch reference must be labeled as scratch evidence.
The upstream component fixture uses `renormalize=false`. Preserve that mode.
The production fixture uses `norm_topk_prob=true`. Preserve that mode separately.
Keep the upstream `use_compile=false` setting and its graph replay cases where
`N >= 1024` and `K >= 1024` on CUDA-alike hardware, including ROCm.

Add local cases from `tests/vt/test_ops_moe_grouped_bf16.cpp` and
`tests/vt/test_ops_moe_grouped_bf16_gate_up_silu.cpp`.
Cover row-map and identity-map inputs, BF16 and legacy FP32 output, large pair
counts, split-K candidates, K and N tails, repeated experts, and empty experts.
The legacy fused operation must remain byte-identical to its existing composite.
The native mode must match captured oracle boundaries within the upstream gate.
Do not compare native arithmetic only against the legacy composite.

Use boundary values that fail when each BF16 narrowing point is removed.
Use nontrivial route weights that distinguish weighting before and after narrowing.
Assert physical activation and expert-output dtypes independently of token results.
Compare greedy token IDs exactly for the complete production fixture.
Capture finite logits and their differences for diagnosis. Never widen the token
gate because an output difference appears numerically small.

### Scratch and negative mutations

Test repeated launches, simultaneous streams, scratch growth, allocation failure,
prewarmed graph capture, replay after growth, and zero-sized workloads.
Require capture and replay to preserve output and pointer lifetime.

A fresh reviewer applies each applicable mutation to an immutable scratch copy:

| Guarantee | Mutation that must fail |
|---|---|
| Both providers are reachable | Remove each registration separately |
| Production uses the new sequence | Delete its production call site but retain fallback |
| Native mode is selected | Force legacy mode without changing final fallback availability |
| Matrix and row layouts are correct | Transpose the weight stride or ignore the row map |
| Expert selection is correct | Substitute another expert or mishandle an empty expert |
| Tail masks are correct | Remove a K mask and an N mask separately |
| BF16 boundaries are correct | Remove gate/up narrowing and SiLU narrowing separately |
| Weight placement is correct | Move route weighting after down-output narrowing |
| Combine consumes preweighted output | Apply the route weight a second time |
| Scratch survives growth | Free a block still referenced by a captured graph |
| Scratch keys isolate users | Remove the stream or device key where hardware permits |

Record the focused command, nonzero result, first relevant failure, and byte-for-byte
restoration after each mutation. A source inspection does not replace these tests.

## Gates and acceptance

The implementer records exact commands after the test executable names are final.
The evidence must include these obligations:

1. Startup, role, and full `scripts/agent-preflight.sh --staged` at the implementation head.
2. A clean CPU build and existing shared-operation regression tests.
3. A HIP build for `gfx1100`, including the new source in the HIP compile options.
4. The focused native and legacy grouped suites with zero unexpected skips.
5. The pinned oracle's identical production fixture and exact token comparison.
6. Dtype, stride, backend-selection, and generated-kernel evidence on both sides.
7. Scratch, capture, concurrency, and negative-mutation results.
8. A fresh scoped review followed by the operator's own hardware gate.

Use the recorded lease or mutex required for the actual GPU.
Record device identity, ROCm and compiler versions, binary SHA256, revisions,
artifact hashes, launch recipes, sampling configuration, and contention state.
Do not perform GPU work outside the operator's recorded authority.

Only accept performance after the declared correctness gate passes.
Trace both runtimes with the same tool on identical inputs.
For each applicable decode and prefill case, record throughput, latency, peak
device memory, peak host memory, and ratios against the production oracle.
The throughput floor is 1.0 times the oracle. Latency and memory must not exceed
1.0 times the oracle. Keep an unmet axis open with its next traceable hypothesis.
Do not describe an unresolved implementation difference as an architectural ceiling.
Repeat accepted results on an idle device with the same-binary fallback comparison.

Report each gate as satisfied, narrowly waived, pending a named external authority
or resource, or failing. This row cannot become `DONE` with an unresolved required
axis. Add an `## Outcome` section when it reaches `DONE`, including measurements,
rejected approaches, and the reason for each default.

## Files and authority

The implementation owns the new HIP source, its registration in `rocm_ops.hip`,
the corresponding CMake source and compile-option entries, and focused tests.
It can extend `include/vt/ops.h`, `src/vt/ops.cpp`, provider metadata, shared
MoE descriptors, and the documented fusion seam to represent the fixed numeric modes.
It can update the Qwen3 shared MoE dispatch to select the complete native mode.
Mechanical adapters in existing providers are allowed when required by the shared
API. Those adapters must preserve their current numerics and defaults.

The implementation updates this spec and the issue records as their state changes.
The parent matrix remains unchanged because its lifecycle does not change.
This per-row inventory records the new capability without another shared matrix write.
Update `docs/FEATURES.md` and the operation inventory in `docs/ROCM.md` when the
native backend capability ships. Document a changed command or configuration in
`docs/USAGE.md` only if the implementation changes that public surface.
This row's spec commit makes no shipped capability claim and owes no public rewrite.

## Owed

No new unowned issue is introduced by this spec.
The origin issue keeps its existing single owner in
[the device-fit spec](gguf-device-fit-expand-policy.md#owed) until this
implementation closes it. That owner links here for the implementation handoff.
The new row owns its tracking issue directly.
Grouped routing and correction-bias work stays under `BACKEND-ROCM`, issue #41.

## Stop conditions

Return `NEEDS_CONTEXT` if the oracle runtime, model bytes, or GPU authority is
unavailable for the next gate. Continue independent CPU work within the spec.
Return `NEEDS_DECISION` if the executing oracle requires semantics outside the
fixed shared modes or if a new shared-seam exception is required.
Do not change unrelated routers, quantized kernels, model architecture, or checkers.
Do not suppress a correctable failure or mark a fallback-only result as native.
Keep a blocked performance axis visible without stopping correctable work.
