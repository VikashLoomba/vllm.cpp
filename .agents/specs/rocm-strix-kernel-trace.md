# Strix Halo decode attribution

Row: `BACKEND-GATE-ROCM-LLAMACPP`

Issue: [#3015](https://github.com/mudler/vllm.cpp/issues/3015).

## Now

Investigation on base `f98b638673b4d2edc0250eec56d229357ea38ab1`.
The user directed autonomous resumption on 2026-09-07. This spec precedes
the diagnostic harness. One pull request follows the repository default.

## Scope

Measure the executing kernels on Strix Halo before selecting a product change.
Trace vllm.cpp and stock llama.cpp with the same profiler, artifact, prompt,
generation count, and single-request workload in one resource-controller lease.
Separate model loading, prefill, cold generation, and warm decode where the
trace permits it. Report unresolved boundaries explicitly.

Measure existing vllm.cpp switches independently against the default binary:
`VT_ROCM_Q8K_BLOCK=1` and `VT_ROCM_Q6K_SMALL_PRIVATE=1`. These are diagnostic
experiments for [#3018](https://github.com/mudler/vllm.cpp/issues/3018) and
[#3017](https://github.com/mudler/vllm.cpp/issues/3017). Do not change defaults.
The Q4/Q5 hypothesis remains owned by
[#3016](https://github.com/mudler/vllm.cpp/issues/3016).

## References and upstream anchors

- [Previous arm spec](bench-rocm-strix-vllmcpp-arm-head.md) records the
  workload and provenance failure this run must avoid.
- [Survey](../../docs/benchmarks/qwen38-27b-q4km-gfx1151.md) records earlier
  values and the failing correctness gate.
- [llama.cpp pin](../oracles/llama-cpp.md) requires stock `b10451`, commit
  `10bf611e` expanded and checked against the configured local oracle.
- [Primary pin](../upstream-sync.md) now requires vLLM `e126687a9`.
  Historical `5559679229` results cannot stand for the new pin.
- `src/vt/rocm/rocm_grouped_gemm.hip` contains `KQuantGemmK`,
  `KQuantDecodeCoopWarps`, `DotQ6K`, and the Q8_K selector.
- Read the stock llama.cpp executing quantized matrix-vector path before
  attributing a trace difference to its implementation.

## Design

Use `strix:gpu0` through a bounded `rc run`. The operator owns the lease.
Build from clean, asserted sources in unique worker-local directories. Use
ccache and at most four build jobs. Record revisions, archive hashes,
compiler versions, build commands, binary hashes, and inherited tuning flags.
Verify the artifact after copying to a unique worker-local directory:
`Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 bytes, SHA256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.

Use prompt `The capital of France is`, greedy generation, 64 output tokens,
and one sequence. Run both engines through their production CLI paths.
Confirm both token counts and prompt processing from logs. A refused or
early-EOS run is not a comparable timing leg.

For each vllm.cpp switch, use three default/candidate pairs with alternating
order. Each process loads once and generates four times; discard generation
one. Compare emitted output and token counts before interpreting a timing
change. Preserve full logs. An output difference prevents acceptance of a
default change. Trace timings are diagnostic; use unprofiled runs for timing.

Sample clocks to worker-local files and fold samples inside recorded warm
generation timestamps. Retain whole-process measurements with explicit labels.
GPU activity percentages do not measure occupancy or prove a numeric bound
on host stalls. Kernel traces, not byte shares, determine kernel time shares.

## Tests and gates

The implementer first proves harness validation rejects missing files, wrong
hashes, incomplete legs, and mismatched workload where the harness checks them.
The focused tests must fail before implementation and detect scratch mutations.
A fresh reviewer reviews an immutable commit and mutates each claimed guarantee.
Run the repository preflight and report every failure or omitted gate.
The operator reruns focused tests and the hardware recipe independently.

`TOKEN_GATE=FAIL` is carried from the previous survey, not remeasured by this
diagnostic run. No throughput ratio is a parity result. No default changes or
performance acceptance occur without the applicable correctness gate.

## Risks and stop conditions

Stop a hardware run on a GPU fault, lost lease, identity mismatch, or missing
profiler support. Install missing tools within the lease when practical.
Preserve completed evidence and identify the exact blocked measurement.
If the current vLLM pin cannot execute this ROCm workload, mark that arm
PENDING; do not substitute the historical pin and claim current parity.
Do not clear a quarantined device or use SSH without a lease.

## Evidence

Initial resource-controller inspection reports Strix ready and no running jobs.
Probe job `cf4724ba-1084-40a5-8699-8990ad9adbd7` confirms the staged artifact,
ROCm 7.2.4 directories, and the previous build image still exist.

## Owed

The product fixes in #3016, #3017, and #3018 stay on their owning rows.
This issue does not close until the clock window and paired trace obligations
are both satisfied. The broader survey and correctness work remain #2921
and #2497.
