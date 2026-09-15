# Enable rocWMMA attention prefill on gfx1100

Row: `BACKEND-ROCM-RDNA3-WMMA-ATTN`.
Parent: `BACKEND-ROCM`.
Issue: `ISSUE-LOCAL-01M2HK0GFJDRXXAAA8F0014XQQ`.
Base: `31509d91f`.
Integration: one pull request, following the repository default.

## Now

`ACTIVE`. The developer requests architecture admission, correctness, compiled
resource inspection, performance measurement, and end-to-end validation.
The developer explicitly forbids subagents for this task. This session performs
implementation and validation. Independent human review remains due at the MR.
The existing quantized admission is separately implemented in PR #3187.

The kernel compiles on gfx1100/1200/1201 and passes ten primary array cases.
The original 96-token gate passes after linear RoPE and BF16 head corrections.
The expanded 256-token gate fails on four WMMA and three scalar requests.
Keep gfx1100 opt-in. Default enablement and full-model performance acceptance
remain blocked by ISSUE-LOCAL-01M2HQEEXHD2B0BT3N71HQ0CRZ. The draft MR is not
merge-ready. [Measured report](../../docs/bench-evidence/rocm-rdna3-attention-wmma/README.md).

## Scope

Admit gfx1100 to the existing BF16 SharedK rocWMMA attention prefill kernel.
Preserve the current gfx1200 and gfx1201 behavior. Keep other gfx11 targets
excluded. Preserve head dimension 256, query-to-KV head ratio two, one request,
at least 64 query tokens, BF16 buffers, and the existing scalar override. Gfx1100 requires explicit opt-in until
the expanded token gate passes; preserve the gfx12 default.
Do not enable the deferred dimension-512 WMMA arm or alter quantized dispatch.
The quantized implementation in PR #3187 remains a separate reviewed change.

## Sources and design

The local implementation is
`src/vt/rocm/rocm_paged_attn.hip::PagedAttnPrefillSharedKWmma`.
Its 16 by 16 by 16 BF16 fragments accumulate FP32 through rocWMMA's public
load, multiply, and store operations. Consumers read ordinary shared-memory
matrices. They do not index the hardware fragment register representation.
Installed rocWMMA 2.2.1 provides the corresponding gfx1100 operation in
`internal/wmma_impl.hpp`, using the gfx11 BF16 wave32 builtin.

Use a new attention-specific architecture predicate that accepts gfx1100 plus
the existing gfx1200/gfx1201 predicate. Preserve the old predicate because
quantized dispatch also uses it on this base. Admit gfx1100 at the device
include guard and use the new predicate at the attention runtime launch guard.
Neither guard alone supplies a working implementation.

The primary pin is vLLM `e126687a9a828d513c01a07cd69f025f27d63280`.
Read and execute `vllm/v1/attention/ops/prefix_prefill.py::context_attention_fwd`
on identical exported Q/K/V, cache layouts, windows, lengths, and softcaps.
Use the head-256, query/KV-ratio-two parameters from
`tests/kernels/attention/test_prefix_prefill.py`. That suite uses F16 inputs.
Record the BF16 input extension needed to exercise this existing BF16 kernel,
and preserve the upstream reference and tolerances. Local fixtures cover 64 and nonmultiple-of-16 query
lengths, causal and sliding masks, reordered blocks, and softcaps.

The existing attention-parity spec records that WMMA arithmetic still owes a
comparison with the primary. An architecture guard change does not discharge
that debt. Capture any numeric difference, and repair only a demonstrated
defect needed by this admission. Do not widen the correctness tolerance.
All model-path buffers keep their existing BF16 format. FP32 accumulation
remains the matrix operation's existing accumulation type.

The real-model artifact is `unsloth/gemma-3-4b-it` at
`bf46152c47f5dd20b896357cb51abc4c03b8ee8c`. It has eight query heads, four KV
heads, and head dimension 256. Its two BF16 shards contain a multimodal wrapper.
For the text-only public entry point, export only `language_model.*` tensors,
remove that prefix, and flatten `text_config` to `Gemma3ForCausalLM`.
Preserve every retained tensor's bytes and all text configuration values.
Run the primary with the identical exported artifact. Record original shard
hashes and exported tensor hashes. This is a harness adaptation, not a loader
change or a newly claimed multimodal capability.

## Gates

### Expanded-gate repair design (15 September 2026)

The developer requests resolving the opt-in blockers and publishing a ready MR.
ISSUE-LOCAL-01M2HQEEXHD2B0BT3N71HQ0CRZ owns the remaining numerical repair.
The executing primary at the same pin exposes three materialized boundaries
that the original Gemma implementation does not mirror:

- `activation.py:451-464` selects erf GELU on ROCm. Its generated GeGLU kernel
  retains the activation in FP32 through the gate/up multiply.
- The generated Q/K preamble combines Gemma normalization and cached rotation
  before its single BF16 output store. Extend the existing gate-free
  `AttnQkNormRope` realization to honor its Gemma flag and route Gemma through
  `FusedChain` with the corresponding recipe.
- The generated sandwich norms evaluate `Npost(a) + base`, then
  `Npost(delta) + (Npost(a) + base)` in FP32. They round only the normalized
  output and the next-layer residual to BF16. Preserve BF16 operand ownership
  across the MLP and extend the typed residual-expression/FusedChain seam to
  represent normalized operands. Do not allocate a persistent FP32 residual.

The generated module `cycnctzmkc6yjxbdnjsayvhkrt7wcw2yzliovagbbvctjna42lab.py`
contains the intermediate-layer stores and returns its BF16 residual. Module
`cduy4zdp3dldo7p3yvkxoff27y2ioqz3atgb4yqnwofcobarhk5l.py` contains the final
norm, without a residual consumer. Both are retained in the task's primary
compiler cache and will be sealed with hashes in the evidence.

Intermediate capture additionally establishes the initial embedding boundary:
the first norm's variance uses the FP32 scaled embedding, while its numerator
reloads the BF16 scaled embedding. Add a typed scaled-norm operation through
`FusedChain`; its outputs remain BF16. Matching this boundary made layer zero's
QKV input and output byte-identical for the failing unique7 prefill and decode.

The affected attention paths also owe the primary's aligned key tiles and
BF16 probability conversion before PV. Update dimension-256 SharedK scalar,
SharedK WMMA, and GQA decode with the executing ROCm attention softmax
expression. Preserve dimension-512 behavior. Export actual layer-zero Q/K/V
and attention outputs to distinguish kernel arithmetic from model-front-end
differences. A scratch FP32-buffer experiment is diagnostic evidence only;
the implementation must keep BF16 buffers and use typed shared fusion calls.

The executing chain is `rocm_attn.py:459-480`, then
`chunked_prefill_paged_decode.py`. Prefill enters `prefix_prefill.py::_fwd_kernel`;
decode enters `kernel_paged_attention_2d`. Current-chunk prefill uses 64 keys,
cached-prefix prefill uses 32, and decode uses `min(block_size, 128)`.
The primary production cache block is 16. Set the native gate harness to 16
for the identical workload, and retain separate coverage of its default 32.
The pinned ROCm backend passes `sliding_window - 1` into a strict-distance
kernel mask. Adapt this at the Gemma model boundary; the shared inclusive
window contract remains unchanged.

The generated ISA uses BF16 WMMA for QK and PV. With exact layer-zero Q/K,
unique4 still differs in seven prefill outputs and eight first-decode outputs.
Test a WMMA PV accumulator against these captures. Reuse the Q tile storage
for accumulator rescaling after QK, reload Q for each next tile, and reuse
score storage for BF16 probabilities after every score reader synchronizes.
Keep FP32 accumulator fragments in registers and keep total LDS below 64 KiB.
Use public rocWMMA load/store operations for accumulator layout conversion.
If necessary, carry the same matrix arithmetic into dimension-256 GQA decode.
Record this prerequisite separately from prefill acceleration and test its
resources and performance. Do not change dimension-512 or quantized paths.

A 15 September operand capture identifies a gfx11 rocWMMA 2.2.1 difference:
`PreMmaXFormA/B` swap the two eight-element halves in lanes 16 through 31.
The executing Triton kernel duplicates lanes 0 through 15 without this swap.
Both matrices remain mathematically equivalent, but 1,936,922 of 5,696,064
captured QK scores differ by up to 1.526e-5. The diagnostic primary kernel
reproduces its original output byte-for-byte, so the comparison is gateable.

Add a gfx1100-only adapter around rocWMMA's packed MMA input seam. Keep its
input/output transforms, accumulator representation, and matrix instruction.
Broadcast the lower-half packed inputs into the upper half before MMA.
This requires rocWMMA's header implementation types; record the tested 2.2.1
version and isolate that dependency in one architecture-specific header.
Do not index accumulator coordinates or alter gfx12's public MMA operation.
Verify the operand capture, score intermediates, token gates, and resources
before accepting this adapter. Retain the AMD license for adapted glue.

Add focused primary-generated fixtures for these expressions, including
nonuniform gamma, BF16 rounding boundaries, full/partial rotary dimensions,
and two/three operand residual expressions. Capture red before each repair,
then green and production reachability. Preserve existing backend defaults
outside the measured ROCm Gemma3 path. Run the unchanged original 96-token
and expanded 256-token gates with scalar and WMMA controls. Investigate any
remaining differences; these source findings alone do not prove token parity.

1. Before admission, the CPU predicate case rejects the requested gfx1100
   target and a physical production-dispatch witness fails to observe WMMA.
2. Compile the actual translation unit for gfx1100. Inspect generated ISA for
   BF16 WMMA and record VGPR count, spills, private bytes, and LDS bytes for
   the production specialization. Compile gfx1200 and gfx1201 as controls.
   A spill is measured debt, not an assumed architecture incompatibility.
3. Run the existing frozen SharedK fixture through `vt::PagedAttention` and
   compare finite output against its declared oracle. Extend physical cases
   through the same entry point for tails and masks. Run enabled and disabled
   controls in separate processes because the environment is cached.
4. Execute the pinned primary on the same arrays. Record absolute errors,
   output dtypes, and tolerance verdicts. Preserve every failed attempt.
5. Enter through the public load/completion API using a deterministic model
   fixture and a pinned real Gemma 3 4B text checkpoint. Trace the call site,
   compare generated IDs, and retain logits when tokens differ. The user
   authorized downloading weights on 14 September 2026. Do not claim a
   synthetic fixture establishes full-checkpoint correctness.
6. After correctness, run same-binary scalar/WMMA comparisons on an idle local
   RX 7900 XTX under `/home/vikash/gpu.lock`. Alternate order across repeats.
   Record prefill and decode rates, latency, memory, clock samples, and boot ID.
   Trace both primary and native with rocprofv3. Below-floor axes remain gaps.
7. Run CPU architecture tests, the HIP attention and cross-device tests, and
   full preflight. Qualify baseline skips and failures. Mutate admission and
   the production launch in a scratch copy and prove the focused gate fails.
   This session's mutation checks do not claim independent review.

## Risks and stop conditions

rocWMMA's gfx11 fragments need more input registers than gfx12 fragments.
Compilation can succeed yet spill or reduce occupancy. Shared-memory and
barrier assumptions need physical execution. BF16 arithmetic can change token
selection even when a float reference tolerance passes.
Keep the gfx1100 default disabled if correctness fails or the measured path
regresses. Record the exact failure and required repair. Do not hide a failure
behind a skipped test, a CPU fallback, or a model that misses the call site.
Do not merge without independent review. Prepare a reviewable MR only after
the stated end-to-end gate passes, or report the measured blocker precisely.

## Evidence

Store concise receipts under `docs/bench-evidence/rocm-rdna3-attention-wmma/`.
Retain full logs, arrays, traces, compiler output, source hashes, commands,
checkpoint provenance, and return codes in the task's ignored build directory.

## Row inventory

| ID | Upstream source | Local anchor | Tests and evidence | Spec | State | Owner | Issue |
|---|---|---|---|---|---|---|---|
| `BACKEND-ROCM-RDNA3-WMMA-ATTN` | vLLM prefix prefill at e126687a9a; rocWMMA 2.2.1 gfx11 BF16 | `PagedAttnPrefillSharedKWmma` | Gates in this spec | [This spec](rocm-rdna3-attention-wmma.md) | `ACTIVE` | Codex, single-agent user direction | `ISSUE-LOCAL-01M2HK0GFJDRXXAAA8F0014XQQ` |
