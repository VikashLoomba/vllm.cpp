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

### Correct the historical clock window

Issue #3015 also owns the clock-window ambiguity in the published lease-2
survey. Reuse its committed `n*.err.txt` timestamps and `clock-n*.jsonl.gz`
samples. Preserve the historical throughput values, raw artifacts, existing
`RESULT` literals, and `rederive.py` contract. Label the existing clock means
as whole-leg samples, including model loading and teardown.

Add a separate narrow reproduction command beside `rederive.py`. It reads
committed evidence by default and joins sample Unix timestamps to each
generation's inclusive start and end. Require exactly four ordered,
nonoverlapping generation windows per leg and valid samples in each window.
Report each generation, warm generations 2 through 4, and whole-leg values.
Pool samples without averaging per-leg averages. Test the command entry with
synthetic cold, warm, and outside-window samples before implementation.
Mutate cold exclusion and timestamp bounds to prove the tests distinguish
the reported windows. Run the command on committed lease-2 evidence and
publish its warm clock and activity values with the exact recipe.

The warm windows include prefill and generation. Busy percentage measures
sampled activity, not occupancy or a quantitative bound on host idle time.
Keep `TOKEN_GATE=FAIL` explicit. This correction proves no new throughput or
correctness result. A fresh reviewer checks the immutable change before landing.

### Collect new paired traces

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

### Diagnostic harness validation

The runner is `tools/bench/strix_kernel_trace/worker.py`. Its tests are
`tests/tools/test_strix_kernel_trace.py`, discovered by the existing tools
suite in preflight and continuous integration. The runner changes no defaults.

Run the focused gate with
`python3 -m unittest tests.tools.test_strix_kernel_trace`.
The first run exited 1 because the runner did not exist. After implementation,
four tests passed. The archive revision and llama count tests then failed with
two missing-function errors. After those guards were implemented, seven tests
passed. `git diff --check` exited 0.

Fourteen in-memory scratch mutations were detected by the focused suite:
file hash, file size, archive revision, llama pin, llama counts, generation
count, prompt count, duration agreement, cold exclusion, clock window,
pair completeness, output equality, process status, and runtime output bound.
The original runner remained byte-identical. This set is not a completeness
claim. The llama-pin mutation triggered the downstream archive parser as an
error, so independent review must verify that guard with a valid wrong-pin
archive too.

The helper invoked full preflight before editing. That run remains PENDING
until its process returns. Hardware compilation and measurement are owned by
the operator and are not established by these Python tests.

The manifest supplies the model path, image, clock device, profiler command
array, and each source archive's full revision and SHA256. Create raw tar
archives with `git archive --format=tar <revision>`. The build verifies the
embedded commit ID and archive hash before extraction. The build phase writes
`build-state.json`. The measure phase requires that state and the identical
manifest, then rechecks the image, binary, library, and model hashes.

Run `python3 tools/bench/strix_kernel_trace/worker.py --phase build --manifest
<manifest.json> --output <build-output>` inside the operator's lease. Then run
the same command with `--phase measure --state <build-output>/build-state.json`
and a separate output directory. Both phases require `RC_DEVICE=strix:gpu0`
and `RC_JOB_ID`. Builds use ccache and four jobs in a new worker-local directory.

Matched baseline traces each run one generation. The llama production target
is `llama-completion`. At the pin, `tools/completion/completion.cpp:44`
documents `-no-cnv`, and `common/sampling.cpp:559` and `:574` print actual
sample and prompt counts. The runner rejects different counts before recording
matched traces. The profiler command comes from the operator's installed help.
Trace interpretation remains a manual gate.

The switch experiment checks emitted cold text and all completion counts.
Warm text equality remains PENDING because `examples/cli/main.cpp:321` prints
only the first completion. Clock summaries use generations 2 through 4 and
their emitted Unix timestamps. These windows contain prefill and generation,
not decode alone. The carried token gate remains FAIL.

### Review repair: container lifecycle and command-path guards

The first review rejected the harness on container cleanup, inherited image
tuning, output monitoring, and tests that did not enter the command path.
The repair runs each container with a unique name. A bounded `finally` block
stops and removes that name after success, process failure, timeout, or excess
output. Cleanup failure stops the harness with an error.

Both phases inspect `Config.Env` and reject `VT_`, `GGML_`, `HSA_`, `HIP_`,
`ROCR_`, and `PYTORCH_` variables. Containers run the inspected image ID.
Measurement records the image environment in `image-environment.json`.
The existing manifest and build-state format remain accepted. Measurement
rechecks the environment even when an older build state lacks `image_env`.

The host monitors the aggregate bytes in each leg directory, including logs
and profiler files, against 512 MiB every 100 ms. This is a sampled stop
threshold, not a strict disk quota. A writer can overshoot between samples.
The container's per-file limit remains an additional guard. Build commands
use the same lifecycle and monitor the build output directory.

The repair's red run exited 1 with missing container names and a missing
managed execution function. The focused suite then passed 20 tests. Tests
execute the actual `__main__` block with temporary archives, state, and logs.
External build and GPU commands are simulated. Real CPU subprocesses prove
timeout cleanup, failure cleanup, host-output detection, and profiler-file
detection. A real subprocess also exceeds host output through the CLI path.
Different cold and warm timings pin each pair's warm median and ratio.
Every vllm.cpp leg sets `VT_OP_PROVIDER_STATS=1` and retains the complete log.
An explicit `[vt reference-tier]` warning rejects a leg, as does a missing
kernel message. Absence of a warning alone does not prove zero fallback.

Repair evidence resides in `/tmp/strix-repair-red.log`,
`/tmp/strix-repair-green.log`, and `/tmp/strix-repair-mutations.log`.
All 35 scratch mutations were detected, including the 25 original reviewer
mutations. The runner remained byte-identical after the mutation run.
The final immutable head receives one full preflight run. Its result and
omitted environment or hardware gates belong to the handoff evidence.

### Second review: test command identity and artifact rejection

The second reviewer found eight surviving mutations in 43 checks at
`881a0199f`. A fresh test implementer reproduced those eight survivors before
editing. The repair changes tests and this evidence only. The runtime remains
byte-identical to `881a0199f`, SHA256
`bc3c23523f657b824897a433e8b1f1e394d7f08018a260b78d13580ef0ccae4a`.

The command tests now reuse the build state during measurement. Temporary
binaries and libraries carry real hashes. Separate cases corrupt the copied
model and tamper with each measured binary and library. The tests reject
missing and incorrect lease identities through the actual command entry.
Command assertions check the immutable image, profiler prefix, greedy flags,
prompt, token count, sequence count, alternating pair order, and four build jobs.
The wrong-pin archive contains an extractable member. Removing the pin guard
now fails with an unmet rejection assertion, rather than a tar parser error.
The CLI output-limit test requires at least 8192 captured stdout bytes, so
command metadata alone cannot satisfy its failure condition.

`python3 -m unittest tests.tools.test_strix_kernel_trace` passed 23 tests.
The baseline mutation run recorded eight survivors in
`/tmp/strix-test-repair-red.log`. After the repair, all 43 mutations were
detected in `/tmp/strix-test-repair-mutations.log`. The eight new detections
include intended assertion failures for the missing checks and changed
commands. The unchanged runner was checked after each full mutation pass.
`git diff --check` passed. Full preflight on the final committed head remains
PENDING until the implementing agent records its result in the handoff.

### Historical warm clock correction evidence

Spec commit `2c09026ef` precedes the correction. The initial focused command,
`python3 -m unittest tests.tools.test_strix_clock_windows`, failed because
`clock_windows.py` did not exist. The completed command passed five tests.
Eleven scratch mutations were detected, including cold inclusion, either
missing timestamp bound, strict instead of inclusive bounds, omitted windows,
overlap, missing samples, invalid samples, averaging leg means, and deletion
of the command entry. The script remained byte-identical after every mutation.

Run the reproduction from the repository root:

```sh
python3 docs/bench-evidence/qwen38-27b-q4km-gfx1151-ourarm-head-20260905/clock_windows.py
```

The committed samples produce 3,216 whole-leg readings, mean 2228.2011815920396
MHz and 71.30783582089552 percent busy. Generations 2 through 4 retain 1,703
samples, mean 2872.811509101585 MHz and 100 percent busy. Per-generation and
per-leg details appear in the command output. The original `rederive.py`
still reports 16 checked claims and zero mismatches. Its committed fallback
was exercised by making only the shared `RESULT.json` existence probe return
false during execution. No file or original calculation changed.

Local evidence: `/tmp/strix-clock-window-red.log`,
`/tmp/strix-clock-window-green.log`, `/tmp/strix-clock-window-mutations.log`,
`/tmp/strix-clock-window-result.json`, and
`/tmp/strix-clock-window-historical-committed.log`. These are reproduction
outputs, not replacements for the committed raw records. The final immutable
head's full preflight remains PENDING until reported in the handoff.

## Owed

The product fixes in #3016, #3017, and #3018 stay on their owning rows.
This issue does not close until the clock window and paired trace obligations
are both satisfied. The broader survey and correctness work remain #2921
and #2497.
