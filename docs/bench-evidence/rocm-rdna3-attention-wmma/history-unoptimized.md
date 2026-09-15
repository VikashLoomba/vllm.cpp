# Superseded measurements before the Release build

This report preserves the previous branch evidence. Its native build had no
`CMAKE_BUILD_TYPE`, with host optimization disabled and HIP at `-O1`. The
production-performance label and accepted full-model ratios below are invalid.
The scalar failures and host-memory lifetime issues were subsequently repaired.
Use the [current report](README.md) for the final implementation and gates.

# Default gfx1100 attention WMMA

Rows: `BACKEND-ROCM-RDNA3-WMMA-ATTN`, `MODEL-GEMMA3-LINEAR-ROPE`,
and `MODEL-GEMMA3-HEAD-DTYPE`.
[Attention specification](../../../.agents/specs/rocm-rdna3-attention-wmma.md),
[linear RoPE specification](../../../.agents/specs/gemma3-linear-rope.md).
Measured on 15 September 2026, RX 7900 XTX, physical `gfx1100`.

## Result and scope

**gfx1100 uses rocWMMA prefill by default.** The public Gemma 3 4B path matches
the pinned primary on the original 96-token gate and the expanded 256-token
gate. Both gates pass with 16-token and 32-token cache blocks. No tolerance,
prompt, or generated-token count was relaxed. See the
[expanded default and scalar verdicts](release-expanded-gates.json),
[original block-16 verdict](release-original16-default-gate.json), and
[original block-32 dispatch verdict](release-default-gate.json).

Admission requires BF16, head dimension 256, two query heads per KV head,
one request, and at least 64 query tokens. `gfx1200` and `gfx1201` retain their
default admission. Other gfx11 devices and the dimension-512 WMMA arm remain
excluded. Quantized admission belongs to [PR #3187](https://github.com/mudler/vllm.cpp/pull/3187).

`VT_ATTN_PREFILL_SHAREDK_WMMA=0` selects scalar prefill. The gfx1100
single-query decode repair remains common to both prefill controls, for cache
blocks at most 64. The broader decode/GQA switches retain their existing role.
The scalar control passes the original block-32 gate and all array tolerances.
It differs on four expanded block-16 requests and three expanded block-32
requests. These are retained failures, not distributional passes. Its scalar
FP32 dot accumulation order differs from the primary's WMMA arithmetic.
The default path resolves the reported opt-in blocker.

Independent human review remains due. The developer prohibited subagents.
The implementation session's mutation checks do not claim independent review.

## Why the guard change needed more work

The primary pin is vLLM `e126687a9a828d513c01a07cd69f025f27d63280`.
The executing path exposed numerical boundaries that standalone layer source
and float-tolerance attention tests did not establish:

1. Gemma's final projection stores BF16. The original implementation stored
   FP32. The existing runner widens the corrected BF16 logits for sampling.
2. The compiled embedding norm computes variance before narrowing the scaled
   embedding. Its normalized numerator and residual reload BF16 values.
3. Compiled Q/K normalization retains FP32 through rotation, then stores BF16.
   Reduction order and the contracted product in each rotary coordinate matter.
4. ROCm selects erf GeGLU. Its compiled expression narrows after multiplication
   by the up projection. Sandwich norms retain normalized operands in FP32
   through residual addition. Their owned model buffers remain BF16.
5. `rotary_embedding/base.py:89-112` builds the cache on the device. CPU libm
   differed even after BF16 narrowing. The device implementation matches all
   301,989,888 local/global BF16 cache words in the diagnostic full-cache run.
   [Byte comparison and hashes](release-cache-proof.json).
6. Primary attention narrows softmax probabilities to BF16 before PV. Prefill
   and decode use different aligned key tiles. The executing decode IR folds
   `acc += dot(P,V)` into the matrix accumulator. A separate delta then add
   changed two first-decode attention words and later token selection.
7. rocWMMA 2.2.1 rotates packed K operands in gfx11's upper half-wave. Triton
   duplicates the lower half unchanged. Both encode the same mathematical dot,
   but their FP32 rounding differs. The gfx1100 adapter preserves rocWMMA's
   transforms and accumulator layout, then aligns packed inputs before MMA.

`rocm_rdna3_wmma.h` adapts AMD's MIT-licensed rocWMMA glue. It depends on
header implementation types tested with **rocWMMA 2.2.1**. This dependency is
isolated in that file. gfx12 retains the public `mma_sync` operation. No code
indexes accumulator coordinates. QK and PV consumers use ordinary shared
matrices through public fragment loads and stores.

The compiled Gemma path enters through `ModelRegistry::Forward`, typed
`vt::FusedChain` bindings, the merged MLP seam, and `vt::PagedAttention`.
It selects linear-RoPE models on the compiled-expression backend policy.
Other model routes retain their materialized behavior. Resident cache
initialization uses the shared lazy resident-weight seam. It creates an FP32
cache temporarily, then stores BF16 once. No persistent FP32 residual is added.

## Identity and executing references

Base: `31509d91f1dcdc5285b845d998a3f7b6da1097f9`.
Native: HIP 7.15.26333, Clang 23, rocWMMA 2.2.1, production flags
`-O1 -ffp-contract=off -Wall -Wextra -Werror`.
Primary container: `sha256:80aab4c182a1f3eeebe286173977e57fcaf10a049b41f475655b35d285de31dc`.
Primary compilation and graphs remain enabled. No eager-only denominator is used.

Checkpoint: `unsloth/gemma-3-4b-it@bf46152c47f5dd20b896357cb51abc4c03b8ee8c`.
Both engines load the same lossless text export. All 444 retained tensors and
7,760,526,336 payload bytes are unchanged. The
[weight recipe](../../USAGE.md#gemma-3-4b-text-weights) records shard hashes,
sizes, and the export command. [Export verification](rope-export-verified.json).
This change does not add the vision tower or a Gemma GGUF loader.

Primary execution anchors at the pinned revision:

- `gemma3.py:159-189`: rotary and attention configuration.
- `activation.py:451-464`: ROCm erf selection.
- `rocm_attn.py:459-480`, `chunked_prefill_paged_decode.py`, and
  `prefix_prefill.py::_fwd_kernel`: prefill execution.
- `kernel_paged_attention_2d`: decode. Its generated TTGIR folds PV into the
  scaled accumulator at lines 320 to 331.
- Generated modules `cpjgc4xo3dj77l3xvynxdob4hutmkehaur7hymz34uyfdq5ygyxs.py`
  and `cycnctzmkc6yjxbdnjsayvhkrt7wcw2yzliovagbbvctjna42lab.py`: initial and
  intermediate compiled partitions. The fixture generator pins their SHA-256.

Every GPU command holds `/home/vikash/gpu.lock`, selects device zero, and
monitors PCI `0000:03:00.0`. Native and primary traces both use rocprofv3,
versions 1.3.5 and 1.3.2 respectively. [Trace identities and counts](release-traces.json).
Whole-run traces include initialization and capture. They do not claim GEMM
invocation parity. Clock and memory samples, boot identity, exact commands,
return codes, and executable hashes accompany the raw measurements.

## Correctness and production reachability

| Gate | Observed result | Evidence |
|---|---|---|
| Default admission | Old predicate fails two assertions. New predicate passes 18 cases and 128 assertions | [Red](default-admission-red.txt), [green](default-admission-green.txt) |
| Public default block 32 | 96/96 tokens, 102 WMMA prefill launches, zero scratch | [Default](release-default-gate.json) |
| Public scalar block 32 | 96/96 tokens, 102 scalar prefill launches, zero scratch | [Scalar](release-scalar-gate.json) |
| Expanded default | 256/256 tokens plus exact warm-up, blocks 16 and 32 | [Verdicts](release-expanded-gates.json) |
| Attention arrays | All ten declared cases pass in both arms, max absolute error 0.0078125 | [Default](release-arrays-default-comparison.json), [scalar](release-arrays-scalar-comparison.json) |
| Compiled expressions | Seven frozen primary cases, cache samples, ownership/error checks | [Focused regressions](release-final-regressions.txt) |
| Final focused regression set | 14/14 CTest entries pass | [CTest](release-final-regressions.txt) |
| Production mutation | Deleting WMMA prefill keeps 96 exact tokens and WMMA decode, but the prefill witness fails | [Mutation](release-launch-mutation.json) |

The trace validator distinguishes prefill from decode by launch geometry.
Decode shares the same WMMA kernel name. Counting that name alone would let
a deleted prefill branch pass. The scratch mutation leaves source bytes
unchanged and fails specifically because zero of 102 prefill launches remain.

Array cases cover tails, prefixes, shuffled cache blocks, causal/noncausal
attention, sliding windows, and softcap. The primary suite's small-input
`atol=1e-4, rtol=0` is unchanged. BF16 stress cases retain the previously
registered P1 `atol=0.015, rtol=0.01`. Real-model diagnostic captures establish
exact Q/K, GeGLU, sandwich-norm, and attention intermediates separately from
these tolerance gates. [Full prefill GEMM boundaries](normdiv-unique2-comparison.json),
[544 real GeGLU comparisons](real-gelu-replay-comparison.json).

Physical RDNA4 execution is unavailable. Cross-compilation is the narrower
result. Old Gemma 1B checkpoint-dependent test bodies remain skipped because
their checkpoint is absent. The public 4B gate supplies actual model execution.

The final review moves the oversized-row check before stride multiplication.
The [UBSan probe](review-overflow-gate.json) detects signed overflow before
the repair and passes afterward. [Fusion tests](review-validation-tests.txt)
pass five cases and 412 assertions. The rebuilt public capture repeats the
expanded 256-token gate and its warm-up exactly.
[Final source and executable identities](review-source-identity.json).

## Registers and spills

| Target | Compiler VGPR | Compiler SGPR | VGPR/SGPR spills | Private bytes | Total LDS bytes |
|---|---:|---:|---:|---:|---:|
| gfx1100 | 89 | 86 | 0 / 0 | 0 | 58768 |
| gfx1200 | 76 | 87 | 0 / 0 | 0 | 58768 |
| gfx1201 | 76 | 87 | 0 / 0 | 0 | 58768 |

[Compiler metadata](release-resources.json) describes
`PagedAttnPrefillSharedKWmma<2,8,16,64,false>`, wave32.
The launch uses 49,152 dynamic LDS bytes and 9,616 static bytes.
The gfx1100 runtime allocates 96 VGPR and 128 SGPR, with zero scratch.
Allocation units differ from compiler usage counts. The ISA contains
`v_wmma_f32_16x16x16_bf16`.

## Reproduce

Use the normal HIP build:

```sh
cmake -S . -B build-rdna3-attn -G Ninja \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_HIP=ON \
  -DVLLM_CPP_HIP_ARCHITECTURES=gfx1100 -DROCM_PATH=/opt/rocm \
  -DVLLM_CPP_BUILD_TESTS=ON
cmake --build build-rdna3-attn -j4
```

Exact link commands are retained for [array capture](attn-capture-final-build-command.json),
[public capture](model-capture-final-build-command.json), and
[benchmark capture](attn-warm-bench-build-command.json).
`model_capture.cpp MODEL MANIFEST OUTPUT` uses the public C ABI.
`bench.cpp MODEL MANIFEST OUTPUT` runs one untimed warm-up, then distinct
requests through `LoadedEngine` and `AsyncLLM`. A manifest can select
`block_size`, default 16. The native public API default remains 32.

Run `model_primary.py capture MODEL MANIFEST --output OUTPUT` in the pinned
primary runtime. Trace the native capture with rocprofv3, then run:

```sh
python3 tools/rocm_attn_wmma/validate_model.py \
  MANIFEST PRIMARY NATIVE TRACE --layers 34 --output RECEIPT
```

For a scalar trace, add `--arm scalar`. The validator checks every input and
output ID, expected prefill geometry, 102 prefill launches, and zero scratch.
The seven compiled fixtures are generated by
`tools/rocm_attn_wmma/compiled_gemma_primary.py` using the pinned generated
modules. `rope_cache_primary.py` exports sparse samples from the primary's
actual full device caches. Their manifests record hashes and parameters.

[Historical opt-in report](history-opt-in.md) preserves the prior failures,
rejected timing runs, and earlier receipts. Raw arrays, traces, generated
code, logits, and failed commands remain in `build-rdna3-attn/evidence`.

## Performance

All measurements follow token correctness. Three alternating process pairs use
one identical native binary. Each process loads once, runs an untimed warm-up,
and then measures distinct requests at concurrency one. No primary prefix-cache
or graph default is disabled. Clocks remain dynamic. The monitor records an
idle device baseline and serializes every job with the same GPU mutex.

The A/B workload uses the original 122-token request as warm-up, followed by
the original 516-token and 1205-token requests. Each generates 32 tokens.
Both native arms and both primary repeats match all outputs. Cache blocks are
32 tokens on both sides. These cases isolate the speed comparison from the
expanded scalar control's known numerical failures. The expanded correctness
gate remains unchanged and mandatory for the default path.

| Model A/B axis, median of three run means | Scalar | Default WMMA | WMMA/scalar |
|---|---:|---:|---:|
| Prefill tokens/s | 3357.61 | 3857.88 | 1.149 |
| Decode tokens/s | 34.237 | 34.278 | 1.001 |
| Time to first token, ms | 256.284 | 223.050 | 0.870 |
| Time per output token, ms | 29.208 | 29.173 | 0.999 |
| Request latency, ms | 1161.729 | 1127.421 | 0.970 |
| Sampled peak RSS, bytes | 9714163712 | 9444880384 | 0.972 |
| Sampled peak PSS, bytes | 9708867584 | 9439588352 | 0.972 |
| Sampled peak device VRAM, bytes | 11133571072 | 11133566976 | 1.000 |

The first WMMA run has slower prefill than its scalar pair. All samples remain
in the [values, ratios, commands, and monitor summary](release-model-performance.json).
The report uses medians without removing that sample. Decode changes by 0.12%,
which does not establish a decode speed gain. Memory samples include loading
and do not establish an activation-memory reduction.

The unchanged expanded workload has eight distinct 518/1207-token prompts,
one warm-up, and 256 measured output tokens. Cache blocks are 16 on both sides.
All three default native runs and both primary runs match exactly.

| Expanded workload axis | Primary median, two runs | Default median, three runs | Native/primary |
|---|---:|---:|---:|
| Prefill tokens/s | 6193.43 | 6445.24 | 1.041 |
| Decode tokens/s | 66.494 | 26.606 | 0.400 |
| Time to first token, ms | 139.264 | 133.820 | 0.961 |
| Time per output token, ms | 15.039 | 37.586 | 2.499 |
| Request latency, ms | 605.471 | 1299.306 | 2.146 |
| Sampled peak RSS, bytes | 8460455936 | 9579806720 | 1.132 |
| Sampled peak PSS, bytes | 7196875776 | 9574513664 | 1.330 |
| Sampled peak device VRAM, bytes | 23074590720 | 11133571072 | 0.483 |

The primary reserves a larger KV pool. Its VRAM ratio does not compare equal
cache capacity. Full-model decode, latency, and host-memory parity remain
**failing** performance axes. This MR establishes correct default attention
and a measured improvement over scalar prefill. It does not establish overall
performance parity with vLLM. The next traceable decode hypothesis is the
512-thread matrix specialization and accumulator rescaling for one query,
especially at block size 16. The next host-memory hypothesis is the lifetime
of loader caches and source mappings. Neither gap is an architectural ceiling.

The ten array cases use five warm-up calls and 1000 timed calls per process,
with three alternating pairs. Outputs repeat byte-for-byte against the final
correctness captures. Every case improves, from **1.12× to 2.92×**.
[Per-case timings and every repeat](release-kernel-performance.json).


## Repository checks and disposition

The [full host preflight](review-preflight-host.txt) exits zero with no failed
checks. All **642 of 642** affected host translation units compile.
The compile check covers syntax and semantics. The separate HIP build, physical
tests, and public runs supply linking and execution evidence.

The sweep reports twelve skips. The seven NumPy suites pass separately in an
isolated host environment. [Commands and results](review-numpy-gates.json).
Five unrelated CLIP-checkpoint subcases remain unavailable and are not passes.
The [CPU ISA audit](release-cpu-isa.txt) and
[path classification](review-pr-classification.txt) pass with their required
arguments. ARM ISA, CUDA fat-gencode, and Triton AOT packaging checks are
narrowly inapplicable to this change's build configuration. CUDA execution of
the positive-factor cache branch is unverified on this ROCm host.
[Exact gate disposition](review-gate-disposition.json).

An earlier audit selected a Python environment created inside the oracle
container. Its package links were unavailable on the host. That run was
terminated and retained. The normal host interpreter supplies the completed
preflight result. No checker, baseline, or test obligation was weakened.

Default gfx1100 admission, primary token correctness, production reachability,
resource checks, and scalar/WMMA prefill A/B are satisfied. Expanded scalar
fallback parity and full-model decode/latency/host-memory parity remain failing
axes in the owning issue. Independent human review is pending at the MR.
