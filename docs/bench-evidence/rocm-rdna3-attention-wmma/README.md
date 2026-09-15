# gfx1100 attention WMMA and Gemma 3 linear RoPE

Rows: `BACKEND-ROCM-RDNA3-WMMA-ATTN`, `MODEL-GEMMA3-LINEAR-ROPE`,
and `MODEL-GEMMA3-HEAD-DTYPE`.
Specifications: [attention](../../../.agents/specs/rocm-rdna3-attention-wmma.md) and
[linear RoPE](../../../.agents/specs/gemma3-linear-rope.md).
Measured on 14 September 2026, local RX 7900 XTX, physical `gfx1100`.
Independent human review is pending. The developer prohibited subagents.

## Result and scope

**Draft candidate. Default gfx1100 enablement remains blocked.** The expanded
256-token gate fails on four of eight requests with WMMA and three with scalar
attention. The primary repeats exactly. Some differences have a non-tied
primary winner. [Failure receipt](expanded-failures.json),
[tracking issue](../../../.agents/issues/BACKEND-ROCM-RDNA3-WMMA-ATTN/ISSUE-LOCAL-01M2HQEEXHD2B0BT3N71HQ0CRZ.md).
No full-model performance result is accepted from that workload.

The original public load/completion gate matches all **96 generated token IDs** from
pinned vLLM. A native trace records **102 WMMA dispatches**, one per layer in
three 34-layer prefills. [Final receipt](public-final-gate.json), [default and scalar](default-final-gate.json).
The same primary outputs repeat in a second run and in the profiler run.

Both compilation and runtime guards admit `gfx1100`, `gfx1200`, and `gfx1201`.
Other gfx11 targets remain excluded. The existing BF16 kernel body is unchanged.
Admission retains head dimension 256, two query heads per KV head, one request,
and at least 64 query tokens. On gfx1100, explicitly set
`VT_ATTN_PREFILL_SHAREDK_WMMA=1` to enter the experimental path. An unset knob
keeps scalar attention. The existing gfx12 default remains enabled. A value
starting with `0` selects the scalar control on every target.
The deferred dimension-512 arm stays excluded.

The real 4B checkpoint exposed a missing prerequisite. The new linear rotary
leaf mirrors the primary's position division and cache construction. Gemma 3
loads BF16 global and local caches. Global layers apply factor eight. Sliding
layers retain unscaled local rotary frequencies.

The broader gate then exposed an incorrect FP32 final projection. The primary
projects into BF16, as its running model confirms in the
[dtype receipt](primary-head-dtype-retry.json). The correction uses BF16 output
and widens it at the existing runner boundary. Tied and untied heads fail the
new assertion before correction. All 1510 focused forward assertions pass
[afterward](head-green.txt), and the [mutation fails](head-mutation.json).
This repair fixes the original scalar tied-token difference. It does not
resolve the expanded gate.

## Identity and method

Base: `31509d91f1dcdc5285b845d998a3f7b6da1097f9`.
Primary: vLLM `e126687a9a828d513c01a07cd69f025f27d63280`.
Native: HIP 7.15.26333, Clang 23, rocWMMA 2.2.1, production HIP flags including
`-O1`, target `gfx1100`. The controls compile the same translation unit for
`gfx1200` and `gfx1201`. Physical RDNA4 execution is unavailable on this host.

Checkpoint: `unsloth/gemma-3-4b-it@bf46152c47f5dd20b896357cb51abc4c03b8ee8c`.
The [usage recipe](../../USAGE.md#gemma-3-4b-text-weights) records source shard
sizes and hashes. The text export retains all 444 text tensors and their
7,760,526,336 payload bytes. Both engines consume that identical export.
[Independent reread of every exported tensor](rope-export-verified.json).
No text config value, rotary factor, or tensor payload is changed.

Every GPU run selects device zero and holds `/home/vikash/gpu.lock`.
The monitored device is PCI `0000:03:00.0`, sysfs `card1/device`.
Boot ID: `2c176325-618c-43da-9f8e-ea0491635f38`.
Clocks vary dynamically. Monitors record clock samples, GPU utilization, process
memory, device-wide memory, exact commands, and return codes. Memory peaks are
sampled values, not continuous maxima or equal-capacity tensor comparisons.

## Correctness and reachability

| Gate | Observed result | Receipt |
|---|---|---|
| Architecture red/green | Three admission assertions fail before the guard change. Initial green: 16 cases, 97 assertions. Final policy: 17 cases, 120 assertions | [Red](arch-red.txt), [initial green](arch-green.txt), [final](arch-final.txt) |
| Frozen physical P1 | 32,768 outputs, finite, zero tolerance violations, max absolute error 0.00390494 | Raw `wmma/` receipt |
| Primary attention arrays | All 10 cases pass with WMMA enabled and disabled | [Enabled](primary-vs-on.json), [scalar](primary-vs-off.json) |
| Linear rotary arrays | Twelve primary cases pass in both layouts and F32/BF16, including multiple factors | [Focused tests](rope-focused-tests.txt) |
| Public real-model gate | Prompt lengths 122, 516, 1205. Each generates 32 exact greedy IDs | [Inputs](model-manifest.json), [final native](source-final-trace-wmma1.json), [primary](model-primary.json) |
| Focused regressions | 13 registered suites pass | [CTest](source-final-ctest.txt) |
| Mutations | Removing gfx1100 admission, the production launch, linear scaling, or sliding-cache selection fails | [Final guard/default](final-guard-mutations.json), [launch](mutation-launch.json), [rotary](rope-mutations.json) |

The physical array extensions cover lengths 64, 79, 257, 1024, and 2048,
nonzero prefixes, shuffled blocks, causal and noncausal attention, local windows,
and softcap 30. The primary prefix-prefill suite supplies small uniform inputs
and `atol=1e-4, rtol=0`. BF16, single-request shapes, NumPy seed-zero generation,
and these tails adapt its F16 multi-request harness. They do not reproduce
Torch's random stream byte-for-byte. High-amplitude extensions retain the frozen
P1 gate's `atol=0.015, rtol=0.01`. No tolerance changed after measurement.
Small-input maximum absolute error is at most 0.000003814697. Stress error is
at most 0.0078125. Every output remains BF16.

Before the head correction, the scalar real-model control first differed on prompt zero at generated index
28. Its two candidate logits are 26.1496639 and 26.1048012. The primary gives
both candidates identical log probabilities and chooses the other token.
[Retained diagnostic](scalar-tie-diagnostic.json). Full scalar logits remain in
the raw evidence. Both explicit controls now pass the original 96-token gate unchanged. The
expanded 256-token gate remains failing.
The old Gemma 1B checkpoint-dependent test bodies skip without their checkpoint.
Synthetic routing checks and the actual public 4B run supply distinct evidence.

Primary source chain: `gemma3.py:159-189` selects rotary configuration and
attention, `linear_scaling_rope.py:37-127` builds the rotary cache, and
`prefix_prefill.py::context_attention_fwd` supplies the array reference.
The softcap extension uses `triton_unified_attention.py::unified_attention`.
Both real-model traces use rocprofv3: [native 1.3.5](native-profiler-version.txt)
and [primary 1.3.2](primary-profiler-version.txt). The primary trace contains
`kernel_paged_attention_2d`; the native trace contains the SharedK WMMA
specialization. Whole-run kernel counts include initialization and graph
capture and do not establish GEMM invocation parity.

## Registers, spills, and shared memory

| Attention target | Compiler VGPR | Compiler SGPR | VGPR/SGPR spills | Private bytes | Static LDS bytes |
|---|---:|---:|---:|---:|---:|
| gfx1100 | 73 | 64 | 0 / 0 | 0 | 4880 |
| gfx1200 | 68 | 67 | 0 / 0 | 0 | 4880 |
| gfx1201 | 68 | 67 | 0 / 0 | 0 | 4880 |

[Final compiler metadata](resources-final.json) describes
`PagedAttnPrefillSharedKWmma<2,8,16,32,false>`, wave32.
The gfx1100 ISA contains `v_wmma_f32_16x16x16_bf16`.
The runtime trace reports allocation units of **80 VGPR and 128 SGPR** and zero
scratch bytes. These allocation counts differ from compiler usage counts.
The launch requests 49,152 additional dynamic LDS bytes, totaling 54,032 bytes
with static LDS. The profiler's LDS column reports 5120 rounded static bytes.

The separately implemented [quantized PR #3187](https://github.com/mudler/vllm.cpp/pull/3187)
is not duplicated here. Its existing build object reports:

| Quantized kernel, F32 and BF16 outputs | VGPR | SGPR | VGPR spills | Private bytes | Static LDS bytes |
|---|---:|---:|---:|---:|---:|
| Q6_K | 188 | 23 | 0 | 0 | 24576 |
| Q4_K | 192 | 22 | 154 | 620 | 25600 |

[Metadata](quant-resources.json), [object identity](quant-object.json).
Q4_K spills remain a tuning cost. The public quantized regression still passes
all 1024 logits and its four generated IDs exactly against the scalar control.
[Recheck](quant-public-recheck.txt). This does not advance that PR's task oracle
pin or close its existing full-model performance gaps.

## Kernel timing

Four alternating scalar/enabled process pairs use the same binary, five warm-up
calls, and 1000 timed calls per case. Each timing includes enqueue and final
synchronization. [All kernel values](perf-summary.json), [monitor summary](perf-monitor-summary.json).

| Case | Scalar µs | WMMA µs | Speedup |
|---|---:|---:|---:|
| T64, small input | 61.31 | 35.36 | 1.73× |
| T79, local window | 29.00 | 19.52 | 1.49× |
| T257, prefix | 318.39 | 200.04 | 1.59× |
| T64, stress | 60.07 | 34.63 | 1.73× |
| T79, stress | 26.89 | 19.51 | 1.38× |
| T257, stress | 313.54 | 197.92 | 1.58× |
| Softcap | 29.86 | 19.39 | 1.54× |
| Noncausal | 152.35 | 47.94 | 3.18× |
| T1024 | 2052.02 | 1145.93 | 1.79× |
| T2048 | 6548.63 | 3796.02 | 1.73× |

## Full-model timing disposition

The first benchmark repeated two prompts and mostly measured prefix-cache
reuse plus first-use costs. Its results remain diagnostic. A second workload
uses eight distinct prompts of 518 and 1207 tokens, one untimed warm-up,
32 output tokens, and concurrency one. Both engines load the identical model
once per process and retain their production scheduling and cache defaults.
vLLM keeps compilation and graphs enabled. This expanded workload exposes the
remaining token failures, so its rates cannot establish a performance gain.

The [diagnostic values and ratios](model-diagnostic-metrics.json) retain prefill,
decode, latency, and sampled memory axes. They are rejected measurements.
The retained JSON files record all outputs and raw timing axes:
[WMMA](head-fixed-warm-wmma1.json), [scalar](head-fixed-warm-wmma0.json),
[primary](warm-primary-r0.json). Their sampled memory includes model loading.
The primary reserves a larger KV pool than the native 256-block pool. Memory
ratios do not establish equal-capacity tensor footprints. The next hypothesis
is the first layer and operation where intermediate BF16 values diverge on
`unique7`, whose first mismatch is the second generated token in both controls.
A teacher-forced layer capture must use the primary prefix before that token.

## Gate disposition

- Architecture compilation, physical array correctness, and zero attention
  spills: satisfied.
- Original public 96-token gate: satisfied in both explicit controls after
  the head correction.
- Expanded 256-token gate and accepted full-model performance: failing under
  the linked issue. No distributional waiver or reduced token set is used.
- Physical RDNA4 regression and the old 1B checkpoint body: pending their
  hardware and checkpoint resources. Cross-compilation and synthetic tests
  supply their narrower results.
- Independent review: pending a human reviewer under the user instruction
  prohibiting subagents.

## Reproduce

The implementation and harness use the normal HIP build:

```sh
cmake -S . -B build-rdna3-attn -G Ninja \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_HIP=ON \
  -DVLLM_CPP_HIP_ARCHITECTURES=gfx1100 -DROCM_PATH=/opt/rocm \
  -DVLLM_CPP_BUILD_TESTS=ON
cmake --build build-rdna3-attn -j4
```

Compile `tools/rocm_attn_wmma/capture.cpp`, `model_capture.cpp`, and `bench.cpp`
with C++20, the repository includes, whole-archive `libvllm.a`,
`libblake3_vendored.a`, and the HIP/hipBLAS libraries. Exact compiler commands are retained for [array capture](attn-capture-final-build-command.json),
[public capture](model-capture-final-build-command.json), and
[benchmark capture](attn-warm-bench-build-command.json). Environment snapshots
accompany the raw evidence.

`primary.py generate` creates the array manifest. Run its `primary` mode with
the pinned runtime, then `capture.cpp` under each explicit knob value.
`primary.py compare` enforces the recorded tolerances. The final source binary
passes all ten cases with [opt-in](source-final-optin-compare.json) and with
the [default policy](source-final-default-compare.json). `model_primary.py`
generates the original prompt IDs and captures primary outputs.
`model_capture.cpp MODEL MANIFEST OUTPUT` uses only the public C ABI.
`bench.cpp MODEL MANIFEST OUTPUT` runs the untimed warm-up and each distinct
request through `LoadedEngine` and `AsyncLLM`.

Trace the public capture with `rocprofv3 --kernel-trace --output-format csv`.
`validate_model.py MANIFEST PRIMARY NATIVE TRACE --layers 34 --output RECEIPT`
requires all prompt/output IDs, 102 WMMA calls, and zero scratch bytes.
An output-only unit test cannot replace this production-dispatch gate.

Raw evidence remains under the task's ignored `build-rdna3-attn/evidence`.
The prerequisite worktrees retain their initial red and mutation builds.
Failed commands, pre-correction results, diagnostics, traces, arrays, full
logits, and source identities remain available. No failed attempt is deleted.

## Repository checks

The final source build passes all 13 focused CTest entries. The final CPU
architecture suite passes 17 cases and 120 assertions. The actual translation
unit compiles for all three admitted targets. Removing admission or changing
the gfx1100 default to enabled makes the policy tests fail.

Full preflight retains the repository's argument-dependent checks and seven
NumPy-dependent skips on the host Python. The seven NumPy suites pass in a
separate isolated environment. Its five optional adherence-model subcases
remain unavailable. The x86 ISA check passes against the actual compile
commands. ARM, CUDA, and Triton AOT gates are narrowly inapplicable to this
HIP-only change. PR classification is checked separately at publication.
This qualified result is not an all-green readiness claim.
