# Default gfx1100 rocWMMA attention

Rows: `BACKEND-ROCM-RDNA3-WMMA-ATTN`, `MODEL-GEMMA3-LINEAR-ROPE`,
and `MODEL-GEMMA3-HEAD-DTYPE`.
[Attention spec](../../../.agents/specs/rocm-rdna3-attention-wmma.md),
[RoPE spec](../../../.agents/specs/gemma3-linear-rope.md).
Measured on 15 September 2026 with an RX 7900 XTX, physical `gfx1100`.

## Result and admission

Eligible gfx1100 attention uses rocWMMA by default. BF16 prefill admission
requires head dimension 256, two query heads per KV head, one request, and
at least 64 query tokens. `VT_ATTN_PREFILL_SHAREDK_WMMA=0` selects real scalar
prefill. The dedicated single-query WMMA decoder serves both prefill controls.
The scalar fallback now reproduces the primary's BF16 dot arithmetic.

Single-request Gemma 3 decode uses a HIP graph by default with linear RoPE,
head dimension 256, the same head ratio, and BF16 cache blocks of 16 or 32.
`VLLM_CPP_CUDAGRAPH=0` selects eager execution. Graph admission remains scoped
to `Gemma3ForCausalLM` on gfx1100. Other model policies retain their values.

The unchanged gfx1200 and gfx1201 paths compile. Physical RDNA4 execution is
unavailable on this host. Other gfx11 devices and the dimension-512 attention
arm remain excluded. Quantized WMMA admission belongs to
[PR #3187](https://github.com/mudler/vllm.cpp/pull/3187).
Independent human review remains due. The developer prohibited subagents.
The implementation session's source mutations are not independent review.

## Repairs that make the default correct

The primary is vLLM `e126687a9a828d513c01a07cd69f025f27d63280` with production
compilation and graphs enabled. No eager-only denominator is used.
The execution chain required these numerical boundaries:

- The final projection stores BF16. The runner widens its output for sampling.
- Compiled embedding normalization computes variance before narrowing the
  scaled embedding. The normalized numerator and residual reload BF16 values.
- Q/K normalization retains FP32 through rotation, then stores BF16. The
  reduction order and contracted rotary products match the generated primary.
- ROCm uses erf GeGLU. Its compiled expression narrows after multiplication by
  the up projection. Sandwich norms retain normalized operands through the
  residual expression. All owned model buffers remain BF16.
- RoPE caches are constructed on the device, then narrowed to BF16. CPU libm
  differed at long positions. The diagnostic compares all 301,989,888 cache
  words exactly. [Cache proof](release-cache-proof.json).
- Attention narrows probabilities to BF16 before PV. Decode uses the primary's
  aligned key tiles and accumulates PV directly into the scaled accumulator.
- gfx11 packed WMMA inputs need alignment with the executing Triton operation.
  The adapter preserves rocWMMA transforms and public accumulator loads/stores.
- Scalar BF16 pairs use gfx1100 DOT2, whose rounding matches the WMMA operation.
  Sequential FP32 multiply/add does not. A 4096-case hardware probe matches
  DOT2 and WMMA exactly. Fully masked scalar tiles preserve the primary's
  negative-infinity state instead of introducing a zero maximum.

The gfx1100 adapter in `rocm_rdna3_wmma.h` adapts AMD's MIT-licensed glue and
isolates a dependency on rocWMMA **2.2.1** implementation types. gfx12 retains
public `mma_sync`. Consumers do not index undocumented accumulator coordinates.

The production route uses `ModelRegistry::Forward`, typed `vt::FusedChain`
bindings, the merged MLP seam, and `vt::PagedAttention`. Graphs capture the
same Gemma layer body. `PersistentStepInput` owns stable upload staging, and
`DevicePool::PinForGraph` protects captured scratch across intervening prefills.

The dedicated decoder uses 128 threads and one workgroup per KV head. It
reuses accumulator selectors and prefetches small V tiles. Normalization
retains operands in registers and preserves the primary reduction order.
One-token merged QKV uses contiguous views, Q/K normalization shares one
launch, and large greedy selection uses 32 partitions with original-index ties.
Embedding reuses checked per-queue device and pinned host bounds records.
Invalid IDs still raise synchronously after the stream completes.

Weights are staged and released incrementally during loading. Device RoPE
construction avoids the unused host cache. The shared resident-weight
initializer validates dimensions and preserves exception-safe ownership.
Neither a wider persistent dtype nor a global residency-policy change is used.

## Build, artifacts, and source anchors

Use an optimized build explicitly:

```sh
cmake -S . -B build-rdna3-attn -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_HIP=ON \
  -DVLLM_CPP_HIP_ARCHITECTURES=gfx1100 -DROCM_PATH=/opt/rocm \
  -DVLLM_CPP_BUILD_TESTS=ON
cmake --build build-rdna3-attn -j4
```

Native: HIP 7.15.26333, Clang 23, rocWMMA 2.2.1. Host and HIP compile with
`-O3 -DNDEBUG -ffp-contract=off`; HIP warnings are errors.
Primary container:
`sha256:80aab4c182a1f3eeebe286173977e57fcaf10a049b41f475655b35d285de31dc`.

The identical checkpoint is the lossless BF16 text export of
`unsloth/gemma-3-4b-it@bf46152c47f5dd20b896357cb51abc4c03b8ee8c`.
Its 444 tensors retain all 7,760,526,336 payload bytes.
[Artifact pins, hashes, and export recipe](../../USAGE.md#gemma-3-4b-text-weights).
The change adds neither the vision tower nor a Gemma GGUF loader.
The existing Gemma 1B checkpoint is downloaded and its regression bodies run.
Its 48-token primary result is reconfirmed at the current pin.
[Native executing assertions](integrated-gemma1b-explicit.txt),
[primary outputs](resume-gemma1b-primary.json).

Executing primary anchors:

- `gemma3.py:159-189`: attention and rotary configuration.
- `activation.py:451-464`: ROCm erf activation.
- `rotary_embedding/base.py:89-112`: device cache construction and BF16 cast.
- `rocm_attn.py:459-480`, `chunked_prefill_paged_decode.py`, and
  `prefix_prefill.py::_fwd_kernel`: prefill dispatch and arithmetic.
- `prefix_prefill.py:312-317,442-448`: masked online-softmax tiles.
- `kernel_paged_attention_2d`: decode. Generated TTGIR folds PV into the
  scaled accumulator at lines 320 to 331.
- Generated modules `cpjgc4xo3dj77l3xvynxdob4hutmkehaur7hymz34uyfdq5ygyxs.py`
  and `cycnctzmkc6yjxbdnjsayvhkrt7wcw2yzliovagbbvctjna42lab.py`: compiled
  Gemma expressions. The fixture generator pins both modules by SHA-256.

Every GPU command holds `/home/vikash/gpu.lock` and selects device zero,
PCI `0000:03:00.0`. Monitors record baseline activity, dynamic clocks, memory,
commands, exit codes, boot identity, and output hashes. Both engines use
rocprofv3 kernel traces, native version 1.3.5 and primary version 1.3.2.
The primary's HIP API trace fails in this runtime, so matched primary traces
use kernel tracing. Native HIP API traces prove graph capture and replay.
The matching benchmark traces disable log-probability collection on both
engines. Earlier diagnostic captures requested log probabilities only on the
primary and cannot establish matching full-span costs.
[Workload correction](integrated-trace-workload-correction.json).
The accepted trace window selects the final request's 1054 attention calls in
time order. Mean block-16 attention is 202.00 microseconds native versus 201.66
primary. Block-32 attention is 173.01 versus 509.29 microseconds. Native scratch
is zero. The primary trace reports 824 to 896 and 2148 to 2160 private bytes,
respectively. [Matched kernel groups](integrated-matched-bench-traces.json).
Traced timings diagnose executed paths and do not replace untraced model
measurements. These traces do not claim GEMM invocation parity. The final
pinned host readback changes no device kernel in this trace comparison.

## Correctness, resources, and mutations

The gate keeps all prompts, outputs, and tolerances. The original gate has
96 output tokens. The expanded gate has 256 measured tokens and an exact
32-token warm-up. Each runs with block sizes 16 and 32, WMMA and scalar
prefill, and graph replay and eager execution. Block sizes use their own
immutable primary references because valid rounding differences affect ties.

The trace validator requires the expected prefill geometry, every decode
launch, zero scratch, one graph capture, and replay across later requests.
A deleted graph call site must fail the graph witness even when tokens match.
Stale input refresh fails exact tokens. Scratch ownership has a separate
probe that holds canary allocations across public graph replay.

Twenty decode fixtures require byte equality with the primary. They cover
physical blocks 16, 32, 64, and 19, sequence lengths 1, 17, 79, and 1217,
window edges, reordered pages, NaN tails, and one or four KV heads.
Ten prefill fixtures retain the original small-input `atol=1e-4, rtol=0` and
BF16 stress `atol=0.015, rtol=0.01`. Both prefill arms run the physical P1 test.
Seven frozen primary fixtures cover compiled Gemma expressions.

The focused CTest set contains 20 entries: loader/configuration, Gemma 1B,
rotary caches, shared residency, attention routes and dtypes, compiled Gemma,
platform admission, pool ownership, graph execution, and persistent inputs.
The large-vocabulary selector test checks cross-partition ties, tail maxima,
all-infinity rows, and the public FP32 contract. Embedding tests alternate
invalid and valid IDs to detect stale error records.

| Integrated gate | Result | Evidence |
|---|---|---|
| Public model and graph matrix | All 16 configurations pass, 3072 exact output tokens including warm-ups | [Model gates](pinned-final-model-gates.json) |
| Dedicated decode arrays | 20/20 byte-exact | [Hashes and cases](integrated-decode-arrays.json) |
| Prefill arrays | 10/10 in each control, no tolerance changes | [WMMA](integrated-prefill-arrays-r1-wmma-comparison.json), [scalar](integrated-prefill-arrays-r1-scalar-comparison.json) |
| Focused regressions | 20/20 CTest entries pass | [CTest log](pinned-final-regressions.txt) |
| Deleted graph call site and stale uploads | Both source mutations detected | [Mutation receipts](integrated-source-mutations.json) |
| Removed graph scratch pinning | Live allocation canaries detect aliasing | [Probe](../../../tests/vllm/models/gemma3_graph_pool_probe.cpp), [mutation](pinned-final-pool-canary-mutation.json) |
| Deleted WMMA prefill call site | Exact tokens remain, prefill witness fails | [Mutation receipt](integrated-prefill-mutation.json) |

All four decoder templates use 128 threads and wave32. The table reports
compiler register usage. Runtime allocation rounds VGPR and SGPR counts up.

| gfx1100 kernel | VGPR | SGPR | Private bytes | LDS bytes |
|---|---:|---:|---:|---:|
| Decode, tile/block 16 | 144 | 50 | 0 | 23568 |
| Decode, tile/block 32 | 179 | 50 | 0 | 33552 |
| Decode, tile/block 64 | 105 | 50 | 0 | 53520 |
| Decode, tile 32, generic physical block | 149 | 61 | 0 | 33552 |
| WMMA prefill | 98 | 102 | 0 | 58768 |

Prefill uses 9,616 static and 49,152 dynamic LDS bytes. gfx1200 and gfx1201
prefill compile with 92 VGPR, 101 SGPR, and zero private bytes. The changed
normalization and selection kernels also report zero private bytes.
[Compiler metadata](integrated-resources.json), [exact commands](integrated-resource-commands.json).
All listed WMMA kernels have zero VGPR and SGPR spills. Physical traces
independently report zero attention scratch.

The token-only pinning mutation initially stayed green. That result is
[retained](integrated-pinning-token-only-mutation.json). The added public probe
holds pool allocations from every recorded decode class across later requests.
Their canaries remain intact with pinning and are overwritten without it.
The source is tested through a separately linked scratch library; production
source bytes remain unchanged. [Build command](graph-pool-probe-build-command.json).

## Performance and memory

Three alternating primary/default/scalar rounds use eight distinct requests,
one load and one untimed warm-up per process, concurrency one, and 32 output
tokens per request. All 18 runs match all 256 measured output tokens.
The summary uses the median of three run means and retains every sample.
Decode rate is `1000 / mean TPOT` on both engines.

The final Release binary remains byte-identical across all eighteen runs.
Block-16 decode is 66.5825 native versus 66.4829 primary tokens per second.
That 0.15% margin clears the sampled floor and supports a parity-level claim.
Block-32 decode is 70.0106 versus 44.6519 tokens per second, or 1.568 times
primary throughput on this workload.
[Every run, value, ratio, command, and contention baseline](pinned-final-performance.json),
[immutable executable proof](pinned-final-source-identity.json).

| Axis | Block 16 primary | Block 16 default | Block 32 primary | Block 32 default |
|---|---:|---:|---:|---:|
| Prefill tokens/s | 6304.61 | 6996.77 | 6689.32 | 6948.02 |
| Decode tokens/s | 66.48 | 66.58 | 44.65 | 70.01 |
| Mean time to first token, ms | 136.805 | 123.271 | 128.937 | 124.136 |
| Mean time per output token, ms | 15.041 | 15.019 | 22.395 | 14.284 |
| Mean request latency, ms | 603.474 | 588.743 | 822.624 | 566.926 |
| Sampled peak RSS, bytes | 8411541504 | 1473593344 | 8327475200 | 1737592832 |
| Sampled peak PSS, bytes | 7149673472 | 1468430336 | 7068961792 | 1732421632 |

Default prefill is 2.31 times the corrected scalar control at block size 16,
and 2.29 times at block size 32. Both controls use the same decoder. All
samples remain in the receipt. Primary VRAM reserves more KV capacity and
is not presented as an equal-capacity improvement.

An initial integrated repeat missed the literal block-16 floor by 0.03%,
including one slow native request. The entire
[three-round measurement](integrated-final-performance.json) remains retained.
The final change pins the checked embedding readback. The
[sampler control](monitor-control-summary.json) did not establish sampling as
the pause's cause. The measurement method and median statistic remain fixed.
The [pre-integration Release comparison](release-validation-v1-performance.json)
is historical evidence and does not replace these final measurements.

These Release measurements and the equal-capacity memory supplement replace
the earlier unoptimized report. Host memory now releases source
weights and unused caches. The supplement gives both engines a logical
4096-token capacity. Native KV storage is 570,425,344 bytes. The primary needs
589,496,320 bytes because its grouping pads 34 layers to 35 and reserves
alignment and null pages. [Capacity calculation](release-cache-budget-design.json).
The supplement is a memory comparison; production defaults remain the speed
denominator. Sampled RSS/PSS are process-tree peaks and VRAM is whole-device
usage, not a continuous process-allocation measurement.

| Equal logical capacity, block size 16 | Primary supplement | Final native default |
|---|---:|---:|
| Sampled peak RSS, bytes | 8370483200 | 1473593344 |
| Sampled peak PSS, bytes | 7112563712 | 1468430336 |
| Sampled whole-device VRAM, bytes | 11330113536 | 11117326336 |

The primary supplement is one exact 256-token run. Native values are the
final three-run medians, whose default capacity already equals 4096 tokens.
[Primary outputs and settings](release-capacity4096-r1-primary-b16.json),
[primary memory samples](release-capacity4096-r1-primary-b16-memory.json).

Rejected tuning experiments remain in the raw evidence. Two workgroups per
KV head and paired QK tiles passed numerical checks but reduced model speed.
Asynchronous embedding allocation passed tokens but slowed decode through
allocator trimming. Larger V prefetch introduced spills. The retained choices
follow exact model measurements and zero-scratch compiler checks.

## Repository checks

The [full preflight](integrated-preflight.txt) exits zero with no failed checks.
All 657 affected host translation units compile. The linked Release HIP build,
physical tests, and model runs supply execution evidence separately.
The final pinned embedding readback changes only a HIP translation unit.
Its [target compilation](pinned-embedding-compile.json), 1121 backend assertions,
repeated 20-test regression set, and model gates validate that final delta.

The sweep's twelve skips have explicit dispositions: seven NumPy suites pass
in the isolated host environment, and the CPU ISA and PR-classification checks
pass with their required arguments. ARM ISA, CUDA fat-gencode, and Triton AOT
packaging checks are narrowly inapplicable to this build. Five unrelated CLIP
checkpoint subcases remain unavailable and are not passes. CUDA cache execution
and physical RDNA4 execution require other hardware.
[Gate disposition](integrated-gate-disposition.json),
[NumPy suites](integrated-numpy-gates.json), [CPU ISA](integrated-cpu-isa.txt),
[path classification](integrated-pr-classification.txt).

[Source and executable hashes](pinned-final-source-identity.json) identify the
integrated product. Independent human review remains due at the MR.

## Reproduction and evidence history

`tools/rocm_attn_wmma/model_capture.cpp MODEL MANIFEST OUTPUT` drives the public
C ABI. `bench.cpp` drives `LoadedEngine` and `AsyncLLM`. The primary adapter
uses the same token arrays and checkpoint. Trace a native capture, then run:

```sh
python3 tools/rocm_attn_wmma/validate_model.py \
  MANIFEST PRIMARY NATIVE KERNEL_TRACE --layers 34 \
  --graph on --hip-trace HIP_TRACE --output RECEIPT
```

Use `--arm scalar` for the scalar control and `--graph off` for eager execution.
`decode_primary.py` generates, captures, and compares the decoder fixtures.
Exact link recipes remain in [public capture](model-capture-final-build-command.json),
[array capture](attn-capture-final-build-command.json), and
[model benchmark](attn-warm-bench-build-command.json).

[Prior opt-in evidence](history-opt-in.md) and the
[superseded unoptimized report](history-unoptimized.md) retain failures and
rejected measurements. The old full-model performance ratios are invalid as
production comparisons. Raw arrays, traces, generated kernels, failed commands,
and experiments remain under `build-rdna3-attn/evidence` in the task worktree.
