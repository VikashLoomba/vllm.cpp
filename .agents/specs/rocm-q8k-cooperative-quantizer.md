# Cooperative ROCm Q8_K activation quantizer

Issue: [#1876](https://github.com/mudler/vllm.cpp/issues/1876)

Row: `BACKEND-ROCM`

## Now

The specification is `READY`. Product implementation and tests are `PENDING`.
The first implementation must keep the cooperative arm off by default.

This commit adds only this specification and the issue ownership row. It makes
no matrix or lifecycle change.

## Issue ownership

Issue #1876 owns this `BACKEND-ROCM` performance slice. The issue-index row
names `BACKEND-ROCM` as the owner and `perf` as the kind.

The issue stays open until the accepted implementation and its evidence land.
A negative result also leaves the issue open.

## Git integration

Use one pull request for the specification and implementation. This choice is
the repository default and is recorded in the local developer preferences.

The specification commit must precede all product and test commits. A fresh
implementer must start from this committed specification.

Local correctness and performance evidence can proceed against the reviewed
external prompt source in the real-checkpoint gate. Publication and merge of
#1876 remain ordered after records-only pull request #2270 or an equivalent
maintainer-landed evidence commit that tracks the same reviewed prompt artifact.

## Scope

The implementation has these in-scope changes:

- Add one cooperative Q8_K activation quantizer for the ROCm backend.
- Add one shared ROCm Q8_K launcher for dense and grouped keep-quant calls.
- Add the same-binary `VT_ROCM_Q8K_BLOCK` selection lever.
- Add direct scratch-byte tests and production-route witnesses.
- Gate correctness and performance on local `gfx1100` hardware.
- Record a negative result if the candidate fails an acceptance rule.

The implementation must not change these surfaces:

- The keep-quant GEMM dot bodies.
- Issue #1910's cooperative Q6_K GEMM.
- Graph enablement or graph policy.
- Producer fusion or quantization inside a GEMM prologue.
- CUDA kernels, launch policy, or environment variables.
- Quantization formats or the `BlockQ8_K` layout.
- Area matrices, lifecycle state, or other keyed records.
- Public documentation before a real-checkpoint gate succeeds.
- Product code from pull request #1936.

## Reference hierarchy

### vLLM primary reference

The primary pin is `5559679229bc961848b121ccdeaa8fa5d79bec98`.
That tree has no executable GGUF Q8_K activation quantizer.

The gap check searched `vllm/model_executor/layers/quantization/**`,
`vllm/model_executor/**`, and `csrc/**`. Searches for `Q8_K`, `q8_K`, and
`quantize_q8` found no implementation. Therefore vLLM supplies no byte oracle
or launch decomposition for this slice.

This absence does not change mirror priority. If a future vLLM pin implements
this behavior, reconcile the row against vLLM before landing.

### Behavioral byte oracle

The byte oracle is the local CPU path:

- `include/vt/quant.h:69::BlockFromFloat` exposes the activation encoder.
- `src/vt/cpu/cpu_quant_act.cpp:88::QuantizeRowQ8_K` defines the arithmetic.
- `src/vt/cpu/cpu_quant_blocks.h:129::BlockQ8_K` defines the 292-byte layout.

The CPU encoder mirrors llama.cpp
`ggml/src/ggml-quants.c::quantize_row_q8_K_ref`. The block layout mirrors
llama.cpp `ggml/src/ggml-common.h::block_q8_K`.

The pinned llama.cpp source places the reference encoder at
`ggml/src/ggml-quants.c:2768::quantize_row_q8_K_ref` and the block at
`ggml/src/ggml-common.h:371::block_q8_K`. The current local CPU comment still
names the older line coordinate `ggml-quants.c:2696`. The persistent anchor is
the symbol at pin `10bf611e533d81f739128304991c5e133c6aebd8`.

### Local decomposition donor

The primary implementation donor is the current CUDA path:

- `src/vt/cuda/cuda_quant_dot.cu:238::QuantizeQ8KPreqKernel` maps one thread to
  each activation element.
- `src/vt/cuda/cuda_quant_dot.cu:1643::LaunchQuantizeQ8K` selects one quantizer and
  is shared by dense and grouped consumers.
- `src/vt/cuda/cuda_quant_dot.cu:176::QuantizeQ8KKernel` retains the serial control
  arm.

`QuantizeQ8KPreqKernel` uses a 256-thread block for one row superblock. Its
reduction carries `(abs, value, index)` and selects the lowest original index
on an exact absolute-value tie.

This change ports that decomposition to HIP. It does not port the CUDA default
or the CUDA environment variable.

### llama.cpp secondary oracle

The secondary pin is llama.cpp
`10bf611e533d81f739128304991c5e133c6aebd8`, tag `b10451`. The pin and its
gateability record are in `.agents/oracles/llama-cpp.md`.

The relevant dense and grouped executing chain is:

1. `ggml/src/ggml-cuda/ggml-cuda.cu:1812::ggml_cuda_mul_mat` selects MMVQ for a
   dense quantized matrix-vector operation.
2. `ggml/src/ggml-cuda/ggml-cuda.cu:1899::ggml_cuda_mul_mat_id` selects the same
   MMVQ path for eligible grouped expert operations.
3. `ggml/src/ggml-cuda/ggml-cuda.cu:1783::ggml_cuda_should_fuse_mul_mat_vec_q`
   governs fused eligible nodes that also call the same MMVQ entry point.
4. `ggml/src/ggml-cuda/mmvq.cu:1153::ggml_cuda_mul_mat_vec_q` allocates Q8_1
   activation scratch.
5. The same function calls
   `ggml/src/ggml-cuda/quantize.cu:558::quantize_row_q8_1_cuda`.
6. That launcher starts `ggml/src/ggml-cuda/quantize.cu:54::quantize_q8_1` with
   `CUDA_QUANTIZE_BLOCK_SIZE == 256`.
7. `ggml/src/ggml-cuda/mmvq.cu:1000::mul_mat_vec_q_switch_type` consumes the Q8_1
   scratch in the quantized matrix-vector kernel.

The Q8_1 kernel supplies decomposition and performance context only. Q8_1 has
32-element blocks, half precision metadata, and no Q8_K `bsums` field. It is
not a byte oracle for Q8_K.

### Local production chain

The dense path is:

1. `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:83::GgufQuantComputeAvailable`
   keeps eligible weights quantized when the ROCm provider exists.
2. Model linear calls reach `src/vt/ops.cpp:149::MatmulBT`.
3. A block weight routes to `src/vt/ops.cpp:191::MatmulBTQuant`.
4. `src/vt/rocm/rocm_ops.hip:212` registers
   `src/vt/rocm/rocm_grouped_gemm.hip:633::MatmulBTQuantKernelRocm`.
5. `MatmulBTQuantKernelRocm` allocates Q8_K scratch and launches
   `src/vt/rocm/rocm_grouped_gemm.hip:148::QuantizeQ8KK` at line 666.
6. `KQuantGemmK` or `KQuantGemmKCoopQ6K` consumes that scratch.

The grouped path is:

1. Model expert calls reach `src/vt/ops.cpp:221::MatmulBTQuantGrouped`.
2. `src/vt/rocm/rocm_ops.hip:215` registers
   `src/vt/rocm/rocm_grouped_gemm.hip:699::MatmulBTQuantGroupedKernelRocm`.
3. `MatmulBTQuantGroupedKernelRocm` allocates Q8_K scratch and launches
   `QuantizeQ8KK` directly at line 744.
4. `src/vt/rocm/rocm_grouped_gemm.hip:384::GroupedKQ8K` consumes that
   scratch for each routed expert.

Both launch sites must move to one shared ROCm launcher. No direct Q8_K launch
may remain in either production consumer.

## Observed gap

`src/vt/rocm/rocm_grouped_gemm.hip::QuantizeQ8KK` maps one thread to one
256-element superblock. The thread scans all elements, quantizes all elements,
and computes all 16 sums.

The amax loop calls `DLoadAct` to compute `fabsf`. It calls `DLoadAct` again
when an element wins. The candidate must load each element once and retain the
value used by the reduction.

Both `MatmulBTQuantKernelRocm` and `MatmulBTQuantGroupedKernelRocm` launch the
serial kernel directly. A dense-only change cannot close issue #1876.

The unmerged T27 commit
`05455b6a97f1fe60615af105b63b8611ac681873` is historical evidence. Its
8-thread candidate measured 91.532 to 93.417 tok/s on `gfx1100`. The 2.06%
change had 5 of 5 byte-identical winning pairs.

T27 was dense-only, had no direct scratch-byte suite, and stayed off by
default. Do not cherry-pick it. Its numbers do not establish a current result.

## Design

### Cooperative kernel

Port the CUDA block-per-superblock decomposition to HIP. Launch one 256-thread
block for each `(row, superblock)` pair.

Each thread must do these operations:

1. Load its source element once through `DLoadAct`.
2. Store its absolute value, signed value, and original index for reduction.
3. Participate in a lowest-index signed-amax reduction.
4. Write one `qs` byte after the scale is known.

The first 16 threads each compute one signed sum over 16 consecutive `qs`
bytes. One thread writes the f32 delta.

The implementation can use shared memory or wave operations. The result and
the original index rule are binding. A wave-specific shortcut must retain a
correct block-wide reduction on `gfx1100`.

### Shared launcher

Add one private launcher beside the ROCm kernels. The launcher accepts the
scratch pointer, source pointer, activation dtype, row stride, row count,
superblock count, stream, and production route.

The route is `dense` or `grouped`. The launcher selects the legacy or
cooperative kernel, increments its route witness, and performs the launch.

Both production consumers must call this launcher. The launcher must preserve
the existing scratch allocation, stream ordering, and error check.

### Same-binary selection

`VT_ROCM_Q8K_BLOCK` accepts only `0` and `1`:

- `0` selects the current serial `QuantizeQ8KK` arm.
- `1` selects the cooperative block arm.
- An unset value selects the recorded default.
- Any other value refuses by name instead of choosing an arm silently.

The first implementation default is `0`. Read the selection at launch time so
one test process can exercise both values.

A final default of `1` is allowed only by the acceptance rules in this spec.
If the default becomes `1`, value `0` remains the legacy same-binary arm.

### Test-only seams

Add private test-only functions in the ROCm translation unit. Tests may
forward-declare them, following the existing `KQuantCoopDispatchCount`
convention.

Extend `tests/vt/test_backend_cross_device.cpp`. Keep the existing test target
and do not add a CMake registration for this slice.

The seam must include these capabilities:

- Launch either arm explicitly into caller-provided Q8_K scratch.
- Reset and read dense legacy, dense candidate, grouped legacy, and grouped
  candidate counters.
- Resolve the environment policy without changing the public ABI.

The explicit-arm hook must not read `VT_ROCM_Q8K_BLOCK`. This separation lets
the direct byte test compare both kernels regardless of the default.

No declaration may enter `include/vllm.h` or another public header.

## Arithmetic invariants

Each Q8_K superblock contains exactly 256 source elements and 292 output bytes.
The output fields are one f32 `d`, 256 signed `qs` bytes, and 16 signed `bsums`.

The implementation must preserve these rules:

- Decode f32, f16, and bf16 source bytes through the existing `DLoadAct` rules.
- Treat `a_rs` as an element stride and never read row padding as data.
- Select the first source element with the largest absolute value.
- Let the lowest original index win an exact absolute-value tie.
- Preserve the selected element's sign in `mx`.
- Set `iscale` to `-127.0f / mx`.
- Use `DNearestInt(iscale * x)` for every quant.
- Clamp only the upper side with `v < 127 ? v : 127`.
- Set each `bsums[g]` to the exact integer sum of `qs[16*g..16*g+15]`.
- Set `d` to `1.0f / iscale`.
- For an all-zero superblock, write zero to all 292 bytes.

The tie rule is not optional. Opposite-sign tied maxima produce a different
scale sign when the wrong index wins.

## Red-first tests

Write and run the direct tests before product code. Preserve the failing output
under `/tmp`.

The direct test must compare all 292 bytes of every `BlockQ8_K` against both
references:

1. The legacy ROCm GPU arm.
2. The independent CPU `vt::cpu::BlockFromFloat(DType::kQ8_K)` arm.

Run the comparison for f32, f16, and bf16 sources. Convert f16 and bf16 source
values independently to f32 before calling the CPU encoder.

Cover these input classes:

- Deterministic random finite data.
- All-zero data with sentinel-filled output storage.
- Opposite-sign exact tied maxima in both sign orders.
- Ties that cross a wave and a reduction boundary.
- Nontrivial row stride with sentinel padding.
- More than one activation row.
- `nsb` values 1, 2, 3, 10, and 16.

The test must name the first mismatching byte, block, row, dtype, and input
class. A field-level equality check is insufficient.

The production tests must call `vt::MatmulBTQuant` and
`vt::MatmulBTQuantGrouped`. For each environment arm, they must assert the
matching route counter changes and the other route counter stays unchanged.

Also test unset, `0`, `1`, and one invalid value. Test the explicit-arm hook
independently from the environment policy.

The record-first specification commit has no product behavior to make red. Do
not fabricate a red result for this commit.

## Production reachability

Output equivalence cannot prove that production used the cooperative arm.
Route counters are the causal witnesses.

The dense witness must enter through `vt::MatmulBTQuant`. It must observe the
selected arm inside `MatmulBTQuantKernelRocm` through the shared launcher.

The grouped witness must enter through `vt::MatmulBTQuantGrouped`. It must
observe the selected arm inside `MatmulBTQuantGroupedKernelRocm` through the
same launcher.

The direct explicit-arm test proves kernel arithmetic. It does not replace
either production witness.

The profiler must also contain the selected kernel name and the matching route
counter totals. A counter-only result cannot establish GPU engagement.

## Build and correctness gates

Use a fresh Release build directory. Do not copy a configured build tree.

Configure with these binding values:

```sh
BUILD_DIR="$(mktemp -d /tmp/vllmcpp-rocm-q8k-release.XXXXXX)"
cmake -S . -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DROCM_PATH=/opt/rocm \
  -DVLLM_CPP_HIP=ON \
  -DVLLM_CPP_HIP_ARCHITECTURES=gfx1100
cmake --build "$BUILD_DIR" -j 4
```

Every HIP runtime command must use this environment:

```sh
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib
```

Every GPU command must hold this lock for the complete command:

```sh
flock /home/vikash/gpu.lock
```

Run the new focused case under the lock. Then run the full applicable Release
test gate under the same lock.

Use these complete test commands:

```sh
flock /home/vikash/gpu.lock env \
  LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib \
  "$BUILD_DIR/tests/test_backend_cross_device" "--test-case=*ROCm Q8_K*"
flock /home/vikash/gpu.lock env \
  LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib \
  ctest --test-dir "$BUILD_DIR" --output-on-failure
```

Run `scripts/agent-preflight.sh` after the focused and full tests pass. Run the
staged quiet preflight immediately before the implementation commit.

If a command writes through `tee`, enable `set -o pipefail` and record the
producer's exit status. A log file must not replace the pipeline return code.

## Real-checkpoint correctness gate

Use this exact model artifact:

- Path: `/home/vikash/models/Qwen3.5-4B-Q4_K_M-unsloth-e87f1764.gguf`.
- Size: 2,740,937,888 bytes.
- SHA-256: `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.
- Source: `unsloth/Qwen3.5-4B-GGUF`.
- Revision: `e87f176479d0855a907a41277aca2f8ee7a09523`.

Use `tools/tg200-prompt.txt` with SHA-256
`e2b801cc6a5739cd317c2f77adfb67040667de524ab60ca64aac39f79c846bba`.
Use these prompt sources in order:

1. After #2270 lands, use the tracked `tools/tg200-prompt.txt` and verify its
   SHA-256.
2. Until #2270 lands, local validation can use only
   `/home/vikash/vllm.cpp-rocm-launch-evidence/tools/tg200-prompt.txt`. Before
   each validation, assert that the worktree has HEAD
   `88b1b1bc80c7c7024d64b9ab10626a93ff279a95`, tree
   `e2fb82f523d9572a5a3e26437d912c5ea0a5f76c`, an empty
   `git status --porcelain=v1`, the tracked path `tools/tg200-prompt.txt`, and
   the required SHA-256.

If neither verified source exists, the real-checkpoint gate stays `PENDING`.
Never substitute prompt text copied from prose. Do not import the prompt into
this branch. No product file from pull request #1936 is imported.

The correctness run uses batch 1, greedy decode, `--max-tokens 256`,
`--temperature 0`, and `--seed 0`. Both arms must return byte-identical
completion bytes and identical token counts.

## Performance protocol

Build one Release binary and record its SHA-256. Compare only these two arms:

- Legacy: `VT_ROCM_Q8K_BLOCK=0`.
- Candidate: `VT_ROCM_Q8K_BLOCK=1`.

Do not compare two binaries. Do not compare an unset arm against an explicit
arm.

Run one discarded warmup completion for each arm. Then run five measured pairs
in alternating order: `0,1`, `1,0`, `0,1`, `1,0`, and `0,1`.

Keep the model, prompt, batch, token count, sampling values, device, binary,
and other environment values identical. Hash every measured completion.

Hold `/home/vikash/gpu.lock` around the complete series. Record the lock
holder, GPU identity, ROCm version, boot identifier, host load, GPU processes,
free memory, utilization, temperature, power, and clock state before and after
each leg.

Record generation seconds, engine tok/s, peak device memory, and peak host
resident set size for every leg. Record absolute values and candidate-to-legacy
ratios.

The candidate wins the engine gate only if all these rules hold:

- All five candidate legs beat their paired legacy legs.
- Median paired engine tok/s improves by at least 2.0%.
- The exact one-sided sign test is 5 of 5, with `p = 0.03125` under equal odds.
- Completion bytes and token counts match in all ten measured legs.
- Device memory and host resident memory do not increase beyond instrument
  resolution.
- No leg has invalid contention, clock, thermal, or power evidence.

The 2.0% threshold is an advance decision. It matches the historical effect
size without accepting a smaller unratified change as signal.

## Profiler protocol

Use `rocprofv3 --kernel-trace --stats` for both arms. Use the same binary,
model, prompt, device, sampling values, and environment.

After a discarded profiler warmup, capture two valid pairs of
`--max-tokens 4` and `--max-tokens 36` for each arm. Subtract each 4-token
capture from its paired 36-token capture, then divide by 32 decode tokens.

For each arm, record these values:

- Quantizer calls per token.
- Quantizer microseconds per call.
- Quantizer kernel milliseconds per token.
- Total kernel milliseconds per token.
- Engine tok/s.
- Exact binary SHA-256.
- Full environment and contention evidence.
- Dense and grouped route counter values.
- The selected kernel name and arm engagement.

Repeat the profile pair if either capture has invalid contention evidence.
Use one tool and one workload for both arms.

The candidate profile must reduce quantizer kernel milliseconds per token by
more than the within-arm repeat spread. Total kernel milliseconds per token
must not regress.

llama.cpp at clean pin `b10451` is a secondary floor when it builds and runs on
the same artifact. Record its clean tree, binary hash, ignored tensors, and
exact workload.

Do not compare llama.cpp Q8_1 bytes with Q8_K bytes. The floor never changes
the local CPU oracle's priority.

## Negative mutation plan

A fresh reviewer must use a scratch copy of the immutable implementation head.
The reviewer must rebuild after each mutation and after each restoration.

Run these mutations separately:

1. Corrupt the exact-tie selector so a higher index can win.
2. Bypass the shared launcher in the dense production consumer.
3. Bypass the shared launcher in the grouped production consumer.
4. Force the legacy arm when the candidate arm is selected.
5. Corrupt one `qs` byte equality guarantee.
6. Corrupt one `bsums` equality guarantee.
7. Corrupt the `d` byte equality guarantee.

Each mutation must fail the focused test for the intended reason. The reviewer
must restore the scratch tree byte-for-byte after each mutation.

After each restoration, rebuild the affected target and rerun the focused
test. The final scratch tree must match the reviewed head exactly.

## Evidence surfaces

Keep implementation and review evidence under `/tmp` until the operator
imports the accepted evidence into the owning change.

Record these items:

- Base SHA, implementation SHA, and tree SHA.
- Exact configure, build, test, benchmark, and profile commands.
- Command exit status and output path.
- Red-first failure and focused green result.
- Full Release gate result.
- All mutation failures and restoration checks.
- Model and prompt hashes.
- The prompt source path, its tracked commit and tree, and its clean-worktree
  assertion when local validation uses the reviewed #2270 worktree.
- Binary and relevant source hashes.
- Route counters and profiler kernel names.
- Ten measured A/B legs and their raw completion hashes.
- The 4-token and 36-token profiler outputs for both arms.
- Hardware, ROCm, clock, thermal, power, memory, boot, and contention state.

Do not edit `docs/FEATURES.md`, `docs/BENCHMARKS.md`, or another public document
until the real-checkpoint correctness and performance gates succeed.

## Risks

- A block-wide reduction can select the wrong signed maximum on an exact tie.
- HIP wave width can invalidate a warp-only reduction that worked in CUDA.
- Shared memory and barriers can cost more than the serial loop on `gfx1100`.
- Dense routing can work while grouped routing still launches the legacy arm.
- A default flip can hide a dead candidate if the environment policy is wrong.
- Profiler totals can mix prefill and decode without the 4-token subtraction.
- The reviewed prompt source can be absent or fail a provenance assertion.
- A `gfx1100` result does not predict `gfx1200` or `gfx1201` behavior.

## Owed

- `gfx1200` runtime validation is `PENDING` external hardware.
- `gfx1201` runtime validation is `PENDING` external hardware, including
  validation from @bakon11.
- Public documentation is owed only after successful real-checkpoint gates.
- A llama.cpp floor measurement is owed if a clean pinned build cannot run in
  this implementation flow.

Do not infer either external architecture result from `gfx1100` source review,
compilation, or measurement.

## Stop conditions

Stop and keep the default off if any direct 292-byte comparison differs.

Stop if either production path does not reach the shared launcher. Output
equivalence does not waive this condition.

Stop if the profiler does not show the candidate kernel and matching arm
engagement.

Stop if the candidate does not beat the ratified engine threshold. Record the
negative result and restore the candidate product code instead of landing dead
speculative code.

Keep the real-checkpoint gate `PENDING` if neither ordered prompt source passes
all provenance assertions. Do not reconstruct the prompt from prose.

Stop with `NEEDS_CONTEXT` if an unavailable input changes the correctness or
measurement contract. Do not replace an unavailable input with a new artifact.

Stop with `NEEDS_DECISION` if a proposed scope change exceeds issue #1876.

## Acceptance and default decision

Correctness is accepted only when every direct block is byte-identical to both
the legacy GPU arm and the CPU encoder. Both production counters must prove
route engagement.

Performance is accepted only when the five-pair engine rule and the profiler
rule both pass on `gfx1100`. All completions must stay byte-identical.

Keep `VT_ROCM_Q8K_BLOCK` default-off if correctness, reachability, engine speed,
or profile engagement fails. A source-level expectation cannot override a
failed gate.

A final default-on decision is allowed only after the specification records
exact-byte correctness, dense engagement, grouped engagement, the accepted
five-pair win, and the quantizer profile reduction.

If the default becomes on, `VT_ROCM_Q8K_BLOCK=0` remains the permanent legacy
A/B arm. All performance and default claims remain scoped to `gfx1100`.
