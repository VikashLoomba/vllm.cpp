# gfx1100 Qwen3.5-4B WMMA model measurements

Row: `KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA3`.
Implementation: `c3fe98ba6c55ce71e75746e1b944a27640464e0f`.
Measurement date: 13 September 2026 local time, 14 September 2026 UTC.

## Disposition

The completed primary, llama.cpp, and six native captures emit identical
generated token IDs. Each native process emits 128 IDs across eight requests.
The original 240 matrix cases and public 1024-logit gate remain independent
correctness guards. The operator separately reproduces all 240 original cases
with [comparison verdict `PASS`](model-operator-mmq-comparison.json).

The same binary records higher prefill throughput with WMMA enabled.
The three process medians are 224.12 tokens/s enabled and 173.86 tokens/s disabled.
The ratio is 1.2891. Median first-token latency is 386.51 ms versus 617.09 ms.
These are observations under dynamic clocks. They do not establish accepted
clock attribution or full-model performance parity.

The whole-model floor remains `FAILING` on the axes identified below.
Matching oracle traces, comparable timing windows, and accepted clock attribution
remain `PENDING` under
`ISSUE-LOCAL-01M2F4WCD6ZK5VH5S8TF83APD6`.
The admission row does not claim a performance ceiling.

## Workload and identities

The retained artifact is `Qwen3.5-4B-Q4_K_M-unsloth-e87f1764.gguf`,
2,740,937,888 bytes, from
`unsloth/Qwen3.5-4B-GGUF@e87f176479d0855a907a41277aca2f8ee7a09523`.
Its SHA256 is
`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.
Each adapter checks that hash before use.
The native driver also verifies input, executable, and shared-library identities
at process boundaries and after the complete series.

The task snapshots are vLLM
`39545e475d3627287ff69c25465dc0bd405f67e1`, llama.cpp
`093a2f86c3e37c54fa3e1f9efb17b304f3433abd`, and the GGUF plugin
`d4c1f0d082fc7cd4350da56689109a01c1f29d6c`.
The developer selected these current upstream revisions for this experiment.
This choice does not advance the repository-wide parity pin.
The [build handoff](model-logs-HANDOFF.md),
[build provenance](model-logs-build-provenance.json), and
[import provenance](model-logs-import-provenance.json) seal the actual runtimes.
Both source builds complete without upstream source changes.

The raw prompts contain 183 and 174 tokens. Their original order repeats four
times, for 1428 input tokens and 128 output tokens per eight-request workload.
The [native prompt receipt](model-native-prompt-identity-result.json) verifies
all eight input arrays against the primary arrays, not only their lengths.
The adapter links the benchmark's existing tokenizer archive objects.
Changing token 760 to 761 without changing its array length fails comparison.
The CPU-only syscall receipt records no GPU device access.
Every engine uses concurrency one, temperature zero, seed zero, and 16 output
tokens. No chat template or EOS-logit mask is applied.
The primary uses `min_tokens=0` and `ignore_eos=True`.
The native ignores EOS stopping, and llama.cpp samples the complete vocabulary
for the fixed output count.

The primary requests dtype auto, context 2048, block size 32, one sequence, and
512 maximum batched tokens. Prefix caching is disabled.
Compilation and graph capture use the upstream production defaults.
The llama.cpp adapter requests context 2048, 512 batch and microbatch tokens,
99 GPU layers, four threads, and BF16 key and value caches.
The native uses 64 blocks, 512 maximum batched tokens, and raw prompt admission.
The primary resolves block size 544, Mamba block size 2048, and 1024 GPU blocks.
Its cache reservation is approximately 17 GiB.
These engine-specific memory allocations are not equivalent cache capacities.
The sampled memory ratios cannot establish equal-capacity memory efficiency.
The original reports preserve requested settings, resolved settings, and their
unavailable fields.

## Timing and memory limits

The native driver runs `on1, off1, off2, on2, on3, off3` as separate processes.
Enabled processes unset `VT_ROCM_QUANT_WMMA`.
Disabled processes set `VT_ROCM_QUANT_WMMA=0` before initialization.
Each process loads the same executable and model once and executes eight requests.
The reported native values are medians of three complete process aggregates.
First-use effects remain included. No warm-only native result is inferred.

The primary and llama.cpp each load one engine and execute four two-request legs.
The aggregate preserves all four legs, including the first captured leg.
The primary's first attempt warms compilation caches before its successful retry.
The retry's `cold` label therefore does not mean an empty compilation cache.
Primary request timing excludes initialization and inter-leg memory calls.

The llama.cpp client includes finite-logit checks, full-vocabulary copies, memory
probes, and memory clearing. Logit file writes occur outside each measured leg.
Its client timings carry more instrumentation than the native benchmark.
The separate llama.cpp decode-call measurements exclude sampling and capture.
They report 1824.71 prefill tokens/s and 100.85 decode tokens/s.
Those call-only values are not divided into native client timings.

The external monitor duration includes startup, teardown, and monitor overhead.
Subtracting benchmark duration does not isolate startup latency.
Initialization takes 154.14 seconds for the primary retry and 1.11 seconds for
llama.cpp. Their initialization scopes differ.

Device-memory readings cover the whole GPU. Process memory is sampled over each
process tree, and shared mappings can appear more than once in RSS.
Peak values are the largest observed samples, not continuous allocator peaks.
The primary reserves a large production cache pool.
Its worker allocator peaks are recorded separately in each original leg.
No native allocator-equivalent peak is inferred from device-wide samples.

## Clock and contention evidence

Every GPU command runs under `/home/vikash/gpu.lock` with visible device zero.
The six native prelaunch GPU-busy readings are all zero.
All monitor captures share boot ID `2c176325-618c-43da-9f8e-ea0491635f38`.
The driver records CPU process snapshots before each native process.
The [derived receipt](model-derived-measurements.json) retains monitor start and
end records, observed peaks, sample counts, and active-clock summaries.
The full sampled records are preserved as gzip files.

The monitors sample process lifetimes, including initialization and teardown.
Observed clocks vary dynamically, and the records do not isolate benchmark clock
windows or satisfy the benchmarking guide's acceptance thresholds.
The small decode and memory differences cannot establish a stable regression
or improvement. Reproduction with accepted clock windows remains owed.

## Preserved failed attempts

The first primary capture initializes production graphs and then fails at its
first memory RPC because trusted callable serialization is disabled.
It emits no tokens. The [original report](model-adapters-primary-model.json)
and compressed log remain unchanged.
The launch retry enables serialization for the trusted local callable inside
the network-disabled container. The capture source remains unchanged.

The [successful primary report](model-adapters-primary-model-retry.json)
records four complete legs. Its SHA256 is
`caad2b4491b3fd7e8e538ec0a739cc9f39748d4e7a491365a7930a168b9f8040`.
The adapter leaves its engine alive after flushing that report.
The operator stops its own container, producing monitor exit 137.
The [teardown receipt](model-adapters-primary-retry-teardown.json) records that
separate action. Generation completion and process teardown retain separate
dispositions.

The first native invocation passes a model directory containing no safetensors
shards. Its [failed summary](model-adapters-native-ab-model-summary.json) and
logs remain. The successful rerun passes the explicit GGUF file to the unchanged
driver. Historical trace preparation with the wrong directory remains identified
by the [adapter archive map](model-adapters-history-trace-gguf-fix-before-archive-map.json).

## Executed model path

Both native `rocprofv3` captures complete on the identical eight-request workload.
The profiler is version 1.3.5, revision
`6b0e43f341195e203754e08f850e437ff2fc09f9`.
The enabled trace contains 1216 WMMA calls. Every prefill window contains 152
calls, and all 120 decode windows contain zero.
The disabled trace contains zero WMMA calls.
The enabled kernels execute 512 Q4_K F32 calls, 536 Q4_K BF16 calls, and
168 Q6_K BF16 calls.
Both traced completion arrays match the primary capture exactly.

The [trace summary](model-adapters-native-trace-pair-summary.json) retains counts,
source trace hashes, output identities, and the window method.
The method sorts kernels by `Start_Timestamp` and pairs 128 embedding starts
with 128 argmax ends. These nonoverlapping windows contain 16 forwards per
request, with the first forward identified as prefill.
The original enabled and disabled traces contain 94,673 and 93,457 rows.
Their full files remain at the recorded external paths.

The enabled and disabled recipes and kernel-statistics CSV files are retained.
Profiler duration includes tracing overhead and dynamic clock variation.
The kernel statistics establish executed paths and are not used to rank
clock-attributed performance. No primary or llama.cpp invocation-parity claim
follows from these native traces.

## Recorded axes

All ratios divide native WMMA enabled by the named denominator.
Native values are medians of three process aggregates.
Oracle values aggregate all eight requests after one load.
The cross-engine ratios retain the timing and allocation limits stated earlier.

| Axis | Unit | Enabled | Disabled | Enabled/disabled |
|---|---|---:|---:|---:|
| Measured duration | s | 8.6400 | 10.4700 | 0.8252 |
| Request throughput | requests/s | 0.9300 | 0.7600 | 1.2237 |
| Input throughput over whole run | tokens/s | 165.1900 | 136.3900 | 1.2112 |
| Output throughput over whole run | tokens/s | 14.8100 | 12.2300 | 1.2110 |
| Total throughput over whole run | tokens/s | 180.0000 | 148.6200 | 1.2111 |
| Mean per-stream decode rate | tokens/s | 53.3300 | 53.1500 | 1.0034 |
| Mean TTFT | ms | 796.4400 | 1026.6900 | 0.7757 |
| Median TTFT | ms | 386.5100 | 617.0900 | 0.6263 |
| P99 TTFT | ms | 3440.2100 | 3687.6200 | 0.9329 |
| Mean TPOT | ms | 18.7500 | 18.8100 | 0.9968 |
| Median TPOT | ms | 18.6600 | 18.7800 | 0.9936 |
| P99 TPOT | ms | 19.5300 | 19.1200 | 1.0214 |
| Mean ITL | ms | 18.7500 | 18.8100 | 0.9968 |
| Median ITL | ms | 18.6300 | 18.6300 | 1.0000 |
| P99 ITL | ms | 19.1200 | 21.0400 | 0.9087 |
| Mean E2EL | ms | 1080.5500 | 1308.7400 | 0.8256 |
| Median E2EL | ms | 666.3400 | 898.8600 | 0.7413 |
| P99 E2EL | ms | 3755.7600 | 3973.5700 | 0.9452 |
| Prefill input tokens per summed TTFT | tokens/s | 224.1200 | 173.8600 | 1.2891 |
| Sampled whole-device VRAM peak | GB, decimal | 4.8638 | 4.8567 | 1.0015 |
| Sampled process-tree RSS peak | GB, decimal | 4.0124 | 4.0122 | 1.0000 |
| Sampled process-tree PSS peak | GB, decimal | 4.0072 | 4.0070 | 1.0001 |

| Axis | Primary | Enabled/primary | llama.cpp | Enabled/llama.cpp |
|---|---:|---:|---:|---:|
| Measured duration | 8.1629 | 1.0584 | 2.1085 | 4.0977 |
| Request throughput | 0.9800 | 0.9489 | 3.7941 | 0.2451 |
| Input throughput over whole run | 174.9379 | 0.9443 | 677.2555 | 0.2439 |
| Output throughput over whole run | 15.6807 | 0.9445 | 60.7064 | 0.2440 |
| Total throughput over whole run | 190.6186 | 0.9443 | 737.9619 | 0.2439 |
| Mean per-stream decode rate | 60.0536 | 0.8880 | 92.8590 | 0.5743 |
| Mean TTFT | 770.5776 | 1.0336 | 100.9933 | 7.8861 |
| Median TTFT | 765.0631 | 0.5052 | 48.3423 | 7.9953 |
| P99 TTFT | 799.1171 | 4.3050 | 445.0011 | 7.7308 |
| Mean TPOT | 16.6518 | 1.1260 | 10.7690 | 1.7411 |
| Median TPOT | 16.2128 | 1.1509 | 10.2946 | 1.8126 |
| P99 TPOT | 19.9876 | 0.9771 | 12.3871 | 1.5766 |
| Mean ITL | 16.6517 | 1.1260 | 10.7690 | 1.7411 |
| Median ITL | 16.1977 | 1.1502 | 8.3664 | 2.2268 |
| P99 ITL | 21.6950 | 0.8813 | 52.0128 | 0.3676 |
| Mean E2EL | 1020.3545 | 1.0590 | 262.5286 | 4.1159 |
| Median E2EL | 1007.8903 | 0.6611 | 207.7424 | 3.2075 |
| P99 E2EL | 1068.2000 | 3.5160 | 629.9160 | 5.9623 |
| Prefill input tokens per summed TTFT | 231.6444 | 0.9675 | 1767.4432 | 0.1268 |
| Sampled whole-device VRAM peak | 24.4776 | 0.1987 | 4.2463 | 1.1454 |
| Sampled process-tree RSS peak | 28.6517 | 0.1400 | 3.1936 | 1.2564 |
| Sampled process-tree PSS peak | 8.3016 | 0.4827 | 3.1867 | 1.2575 |

TTFT means time to first token. TPOT means time per output token.
ITL means inter-token latency. E2EL means end-to-end request latency.
RSS means resident set size. PSS means proportional set size.

The primary comparison retains gaps in whole-run throughput, prefill, decode,
mean and tail first-token latency, mean and median token latency, and mean and
tail request latency. The llama.cpp comparison retains throughput, prefill,
decode, most latency axes, and all sampled memory gaps.
Enabled P99 TPOT and sampled memory also exceed the scalar medians slightly.
Those small differences require controlled reproduction before a stable
regression or equality conclusion.

The next hypothesis is different prefill and decode operator selection and
launch overhead across the engines. Matching traces and accepted clock windows
must distinguish those effects from request instrumentation and allocation.
No unmeasured operator is declared a performance limit.

## Reproduction and retained sources

The [adapter handoff](model-adapters-HANDOFF.md) contains the exact commands and
source-chain anchors. Its absolute source paths describe the measured workspace.
The [adapter manifest](model-adapters-adapter-manifest-gguf-trace.json) seals
the adapters, compiler, model, and library inputs.
The [copy manifest](model-receipt-manifest.json) records each retained file's
original path, size, hash, and current filename.
Compressed files decompress to their original bytes.

The copied C++ adapter uses the suffix `.cpp.txt` to meet the evidence classifier.
Restore its original `.cpp` filename before using the recorded build command.
The original adapter code remains unchanged.
Full model logits, matrix outputs, and complete profiler traces remain under
`/home/vikash/oracle/rdna3-wmma-latest` at their sealed source paths.
These large raw artifacts are not duplicated in the repository.

The [evidence path map](evidence-path-map.json) resolves the earlier nested copies
to the permitted single-directory layout. Original source locations and hashes
remain preserved. No classifier rule changes.
