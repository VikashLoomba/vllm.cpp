Measurement adapter preparation is COMPLETE. All files are external scratch artifacts under `/home/vikash/oracle/rdna3-wmma-latest`; no product or upstream model/kernel source was changed. The coordinating task owns GPU execution and `/home/vikash/gpu.lock`. These preparation checks do not establish model token parity or a performance floor.

The source/binary and completed check hashes are sealed in [adapter-manifest.json](/home/vikash/oracle/rdna3-wmma-latest/adapters/adapter-manifest.json). Earlier build and import evidence remains in [build-provenance.json](/home/vikash/oracle/rdna3-wmma-latest/logs/build-provenance.json), [import-provenance.json](/home/vikash/oracle/rdna3-wmma-latest/logs/import-provenance.json), and [the oracle build handoff](/home/vikash/oracle/rdna3-wmma-latest/logs/HANDOFF.md).

The clean upstream source revisions are vLLM `39545e475d3627287ff69c25465dc0bd405f67e1`, llama.cpp `093a2f86c3e37c54fa3e1f9efb17b304f3433abd`, and the GGUF plugin `d4c1f0d082fc7cd4350da56689109a01c1f29d6c`. The committed experiment spec is `a5b5c92`. The retained GGUF SHA256 is `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.

**Primary capture and retry.** [capture_primary.py](/home/vikash/oracle/rdna3-wmma-latest/adapters/capture_primary.py) exports exact raw `tokenizer.encode` IDs, checks the retained GGUF hash, uses latest vLLM with explicit GGUF config/load/quantization and the supported text-model class override, and retains the resolved engine configuration. Metadata comes from `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; the flat text config preserves `partial_rotary_factor=0.25`. The two raw prompts encode to 183 and 174 tokens. No chat template or tokenizer-regex repair is applied.

The requested configuration is dtype auto, max length 2048, one sequence, max batched tokens 512, block size 32, prefix caching disabled, generation config vLLM, and upstream production compilation/graph defaults. Greedy sampling uses seed 0, temperature 0, 16 output tokens, `min_tokens=0`, `ignore_eos=True`, and neutral penalties. Four legs run after one engine load: the first captured leg and three later legs. Worker Torch allocator peaks are reset before each leg. The field named `memory_before_reset_peak` contains the snapshot **after** resetting the peak counter. Device free/total readings have whole-device scope; the allocator readings belong to the actual model worker.

The coordinator's first invocation initialized the model and production graphs in 348.582 seconds, then failed at the first memory RPC because callable serialization was disabled. It did not generate tokens. Its unchanged [primary-model.json](/home/vikash/oracle/rdna3-wmma-latest/adapters/primary-model.json) has SHA256 `f152650d003011ab1bbb0155f40331a1839c466af0c4e5fc193b36705eade0ba`. The capture source stayed unchanged. The launchers now explicitly allow serialization for this trusted local callable inside the network-disabled container, set `XDG_CONFIG_HOME` under the task directory, and disable usage statistics. The first attempt warmed JIT caches; the retry's `cold` label denotes its first captured leg, not an empty JIT cache.

While already holding the mutex, run:

```sh
/home/vikash/oracle/rdna3-wmma-latest/adapters/retry_primary_under_lock.sh
```

This writes `primary-model-retry.json`, `prompt-ids-retry.json`, `primary-model-retry-monitor.jsonl`, and separate stdout/stderr logs. The bounded monitor uses the actual host PID of Docker container `rdna3-wmma-primary-retry`. Existing output files cause the monitor to refuse replacement. The full Docker argv, image identity, environment, and mounts are in [python-gpu-named-under-lock.sh](/home/vikash/oracle/rdna3-wmma-latest/adapters/python-gpu-named-under-lock.sh). Both task worktrees and `/home/vikash/models` are mounted; the source/model mounts are read-only.

The coordinator subsequently completed the primary retry: four legs and 128 generated IDs, which it checked exactly against the scalar native two-prompt baseline repeated four times. Engine initialization took 154.137 seconds with normal production graphs. The completed generation report has SHA256 `caad2b4491b3fd7e8e538ec0a739cc9f39748d4e7a491365a7930a168b9f8040`. After the script wrote `COMPLETE`, the engine process persisted because the scratch adapter has no explicit shutdown. The coordinator owns termination of its `rdna3-wmma-primary-retry` container. Consequently, the process monitor may report `COMMAND_FAILED` during teardown even though the immutable generation report is complete. Treat generation completion and process teardown as separate evidence; do not rerun successful generation merely to obtain a clean teardown exit. The completed report and capture source remain unchanged.

**Latest llama.cpp capture.** [capture_llama.cpp](/home/vikash/oracle/rdna3-wmma-latest/adapters/capture_llama.cpp) and [capture_llama](/home/vikash/oracle/rdna3-wmma-latest/adapters/capture_llama) use the latest public llama API and the isolated built shared libraries. The exact command is [build_llama_adapter.sh](/home/vikash/oracle/rdna3-wmma-latest/adapters/build_llama_adapter.sh); compilation with Clang, `-O2 -Wall -Wextra -Werror`, and one compiler process exited 0. The linked runtime resolves through the explicit task-library RPATH. No upstream source repairs were made.

The adapter requests 99 GPU layers, context 2048, batch/ubatch 512, one sequence, four threads, and BF16 K/V. It rejects context creation failure without a dtype fallback. It clears all recurrent/KV memory and checks the empty sequence before every prompt. Each of four legs processes the same two prompt arrays. The upstream greedy sampler receives the complete vocabulary, with no EOS/EOG masking or stop checks, and emits exactly 16 tokens. Each request performs one prefill call and 15 later decode calls. Every logits row must be finite; all 16 full-vocabulary rows are retained as hashed float32 binary files.

While already holding the mutex, run:

```sh
/home/vikash/.venv/bin/python /home/vikash/oracle/rdna3-wmma-latest/adapters/monitor_command.py \
  --output-jsonl /home/vikash/oracle/rdna3-wmma-latest/adapters/llama-model-monitor.jsonl \
  --interval-seconds 0.05 --timeout-seconds 600 \
  -- env HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
  /home/vikash/oracle/rdna3-wmma-latest/adapters/capture_llama \
  /home/vikash/oracle/rdna3-wmma-latest/qwen35-4b-text/model.gguf \
  /home/vikash/oracle/rdna3-wmma-latest/adapters/prompt-ids-retry.json \
  /home/vikash/oracle/rdna3-wmma-latest/adapters/llama-model.json
```

Logit file writing and hashing occur outside each measured leg. Client timing includes finite checks and full-vocabulary copies, so it is not an equal-overhead latency denominator. Separate decode-call timings contain `llama_decode` plus synchronization and exclude sampling/capture. Memory clear is reported separately. Public getters do not expose a per-context GPU allocation breakdown, final layer placement, or resolved K/V types: the report states these omissions and retains whole-device snapshots and process peak RSS. Load logs and matching traces remain necessary to establish the executing path.

**Native same-binary A/B.** [run_native_ab.py](/home/vikash/oracle/rdna3-wmma-latest/adapters/run_native_ab.py) runs six separate processes in `on1, off1, off2, on2, on3, off3` order. On unsets `VT_ROCM_QUANT_WMMA`; off sets it to `0`. Each process loads one model and runs eight requests: the original two dataset records repeated four times without text changes. The output directory retains the original and expanded dataset hashes and mapping `[0,1,0,1,0,1,0,1]`. Configuration is input-length bound 256, output length 16, concurrency 1, max batched tokens 512, 64 native blocks, temperature 0, seed 0, raw prompts, and EOS ignored. Native block count is not equated with vLLM's hybrid allocation.

While already holding the mutex, run:

```sh
/home/vikash/.venv/bin/python /home/vikash/oracle/rdna3-wmma-latest/adapters/run_native_ab.py \
  --output-dir /home/vikash/oracle/rdna3-wmma-latest/adapters/native-ab-model \
  --primary-report /home/vikash/oracle/rdna3-wmma-latest/adapters/primary-model-retry.json \
  --llama-report /home/vikash/oracle/rdna3-wmma-latest/adapters/llama-model.json
```

The driver checks card1 `gpu_busy_percent <= 2` before each process and records CPU process/load snapshots. It hashes the executable, resolved ELF shared libraries, model/config/tokenizer inputs, and datasets, then checks identities at each boundary and hashes them again afterward. Missing shared libraries or changed inputs reject the run. Each process gets a 0.05-second monitor and separate output IDs/logs. The driver enforces eight requests, 1428 input tokens, and 128 generated tokens, compares every generated-ID row across all six processes, and compares completed oracle captures in four-leg order. Missing or incomplete oracles remain explicitly unavailable.

`same_binary_output_gate` and `cross_oracle_output_gate` are separate. A shared mismatch against an oracle makes cross-oracle parity fail, while exact native A/B outputs still permit descriptive on/off medians and ratios. Any native arm/repeat mismatch suppresses those ratios. Native metrics aggregate all eight requests after one load; the external process duration includes startup, teardown, and monitor overhead. Its difference from benchmark duration is not a direct startup-latency measurement. No warm-only native result is inferred.

**Whole-workload oracle aggregation.** After both captures complete, run:

```sh
/home/vikash/.venv/bin/python /home/vikash/oracle/rdna3-wmma-latest/adapters/aggregate_oracle_captures.py \
  --primary /home/vikash/oracle/rdna3-wmma-latest/adapters/primary-model-retry.json \
  --llama /home/vikash/oracle/rdna3-wmma-latest/adapters/llama-model.json \
  --output /home/vikash/oracle/rdna3-wmma-latest/adapters/oracle-whole-workload.json
```

This reuses the original primary metrics helper on all eight request records and the sum of the four leg durations. It aggregates llama's eight requests while labeling its additional capture overhead, retains each original cold/warm leg, and checks primary/llama IDs exactly. Matching request counts do not erase differences in instrumentation or timing windows.

**Native rocprofv3 path traces.** The installed `/opt/rocm/bin/rocprofv3 --help` and `--version` were inspected without executing a GPU workload. [run_native_trace.py](/home/vikash/oracle/rdna3-wmma-latest/adapters/run_native_trace.py) retains complete argv, tool/binary/dataset identities, profiler output configuration, and bounded monitor logs. It uses the same original two records repeated four times and the same native configuration as the unprofiled A/B. It collects unfiltered HIP API, kernel dispatch, memory-copy/allocation, and ROCTx traces in CSV and JSON, without hardware counters or ATT. `ROCPROF_TMPDIR` and `TMPDIR` point to writable task cache directories outside `/dev/shm`.

While already holding the mutex, run these sequentially:

```sh
/home/vikash/.venv/bin/python /home/vikash/oracle/rdna3-wmma-latest/adapters/run_native_trace.py \
  --arm on --output-dir /home/vikash/oracle/rdna3-wmma-latest/adapters/native-trace-on
/home/vikash/.venv/bin/python /home/vikash/oracle/rdna3-wmma-latest/adapters/run_native_trace.py \
  --arm off --output-dir /home/vikash/oracle/rdna3-wmma-latest/adapters/native-trace-off
```

Each trace has a 600-second bound and its own output and temporary directories. Existing output directories are refused. The driver requires idle card1 before starting and retains exact generated IDs. Profiler timings carry tracing overhead; use unprofiled A/B for throughput. The prepared argv can be reviewed now in `native-trace-on-preparation/recipe-and-result.json` and `native-trace-off-preparation/recipe-and-result.json`; both were created with `--prepare-only` and did not launch a workload.

The common [monitor_command.py](/home/vikash/oracle/rdna3-wmma-latest/adapters/monitor_command.py) records boot ID, command argv, timestamps, process-tree RSS/PSS, host available memory, and card1 VRAM/busy/clock text. Device memory is whole-device; peak values are maxima of sampled observations. It never changes clocks. For Docker, pass `--docker-name` matching the command's unique `--name`; the monitor refuses an existing container and samples the container's actual host PID. Timeout cleanup only targets the process group and container started for that invocation.

Preparation checks completed without executing a GPU workload: raw tokenization [183,174]; llama compile and help; malformed-token JSON rejection before backend initialization; monitor CPU process sampling; bounded timeout cleanup; Docker CPU host-PID sampling; six-process native CPU orchestration; deliberate off-arm token mismatch; busy-gate rejection before process launch; cross-oracle mismatch retaining exact same-binary ratios; aggregate arithmetic for eight requests/1428 inputs/128 outputs; Python AST and shell syntax checks. CPU fixtures and logs are retained and named as such. They are not model measurements.

Source anchors for review:

- Primary argument/sampling and worker memory adaptation: [capture_primary.py:62](/home/vikash/oracle/rdna3-wmma-latest/adapters/capture_primary.py:62), [capture_primary.py:165](/home/vikash/oracle/rdna3-wmma-latest/adapters/capture_primary.py:165), [capture_primary.py:193](/home/vikash/oracle/rdna3-wmma-latest/adapters/capture_primary.py:193); repository metrics [run_closed_loop:59](/home/vikash/vllm.cpp-rdna3-wmma/tools/bench/vllm_closed_loop_metrics.py:59) and [derive_metrics:202](/home/vikash/vllm.cpp-rdna3-wmma/tools/bench/vllm_closed_loop_metrics.py:202).
- Latest model registration: [registry.py:204](/home/vikash/oracle/rdna3-wmma-latest/vllm-source/vllm/model_executor/models/registry.py:204), [registry.py:597](/home/vikash/oracle/rdna3-wmma-latest/vllm-source/vllm/model_executor/models/registry.py:597); plugin registration [plugin.py:130](/home/vikash/oracle/rdna3-wmma-latest/plugin-source/vllm_gguf_plugin/plugin.py:130), GGUF linear dispatch [linear.py:35](/home/vikash/oracle/rdna3-wmma-latest/plugin-source/vllm_gguf_plugin/quantization/linear.py:35), compiled MMQ entry [ops.py:200](/home/vikash/oracle/rdna3-wmma-latest/plugin-source/vllm_gguf_plugin/ops.py:200).
- Upstream min-token masking skips zero: [builtin.py:198](/home/vikash/oracle/rdna3-wmma-latest/vllm-source/vllm/v1/sample/logits_processor/builtin.py:198). Trusted callable serialization is explicitly guarded at [serial_utils.py:221](/home/vikash/oracle/rdna3-wmma-latest/vllm-source/vllm/v1/serial_utils.py:221).
- Latest llama memory clear and synchronization: [llama.h:739](/home/vikash/oracle/rdna3-wmma-latest/llama-source/include/llama.h:739), [llama.h:1027](/home/vikash/oracle/rdna3-wmma-latest/llama-source/include/llama.h:1027). Full-vocabulary sampling and greedy tie policy: [llama-sampler.cpp:895](/home/vikash/oracle/rdna3-wmma-latest/llama-source/src/llama-sampler.cpp:895), [llama-sampler.cpp:1053](/home/vikash/oracle/rdna3-wmma-latest/llama-source/src/llama-sampler.cpp:1053).
- Native arguments [main.cpp:89](/home/vikash/vllm.cpp-rdna3-wmma/examples/bench/main.cpp:89), model/cache configuration [bench_core.h:723](/home/vikash/vllm.cpp-rdna3-wmma/examples/bench/bench_core.h:723), output IDs [bench_core.h:1005](/home/vikash/vllm.cpp-rdna3-wmma/examples/bench/bench_core.h:1005).

Remaining runtime limits include the older installed Torch dependency warning (the latest vLLM wheel was built against the container's Torch 2.12 development build), absent optional Mooncake/Rust components, and pending latest llama/native A/B/trace results until the coordinator completes those runs. The first attempt's serialization failure is preserved separately from the successful primary retry.
