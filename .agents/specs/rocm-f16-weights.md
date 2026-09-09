# Retain F16 GGUF weights on ROCm

Row: `BACKEND-ROCM-F16-WEIGHTS`

Issue: [#3092](https://github.com/mudler/vllm.cpp/issues/3092)

Parent: `BACKEND-ROCM`. Base: `6db4bef906859e864c82523c01107473f7dcca29`.

## Now

`READY` for a fresh implementation from this committed spec. This commit defines
the contract and records the source inspection. No implementation or GPU result
is claimed. The parent row remains `ACTIVE`.

The change uses one pull request. Commit this spec before implementation. The
operator owns the GPU, reviews the returned evidence, and reruns the gates.

## Scope

Add ROCm execution for F16 weight storage through ordinary `vt::Matmul`,
`vt::MatmulBT`, and `vt::Embedding`. Make the default Qwen3.5 dense GGUF loader
reach these operations with retained F16 projection and embedding weights.
Preserve the resolved model dtype's arithmetic, including its weight rounding.
The first production witness is Qwen3.5 dense. This is a backend capability,
not a claim that every model loader supports retained F16 weights.

The retained F16 bytes serve two different contracts:

1. An unmarked primitive operand represents its exact F16 values.
2. A retained GGUF model weight represents the file values cast to the resolved
   model dtype before multiplication or gathering. For a BF16 model, this means
   F16 to BF16 rounding before arithmetic, even when an output buffer is F32.

Support F16 weights with F16, BF16, and F32 primitive activations and existing
BF16 or F32 outputs. Preserve the existing BF16 and F32 ordinary GEMM paths.
Keep the current output predicate. F16 model activations and F16 outputs are
outside this change, as are RMSNorm, attention, and a full F16 runtime.

Include the F16 embedding reader because the same default loader admits the
embedding table and a tied output head. Preserve both integer ID widths and
existing invalid-ID behavior. A kernel-only change that the default loader
cannot reach does not satisfy this row.

Exclude stacked expert F16 retention, grouped expert GEMM, block-quantized
GEMM, quantized embedding gather, new skinny kernels, fused alpha/beta GEMM,
and environment or CI policy changes. Preserve existing routing for these
paths. In particular, do not change the concurrent quantized GEMM change
[#2782](https://github.com/mudler/vllm.cpp/pull/2782).

## Upstream chain

Use vLLM `e126687a9a828d513c01a07cd69f025f27d63280`, recorded in
[the active pin](../upstream-sync.md). Source paths below refer to this revision.

- `vllm/model_executor/layers/linear.py:183` creates an unquantized parameter
  with `params_dtype`. `UnquantizedLinearMethod.apply:217` delegates to the
  selected GEMM implementation.
- `vllm/model_executor/layers/utils.py:559` dispatches unquantized GEMM to the
  ROCm implementation at line 563. `rocm_unquantized_gemm_impl:260` selects the
  production implementation. The gfx1x skinny eligibility starts at line 326,
  contiguous activation handling is at line 332, and `F.linear` is the ordinary
  fallback at line 346. The fake implementation at line 349 returns `x.dtype`.
- `vllm/model_executor/layers/vocab_parallel_embedding.py:35` defines
  `UnquantizedEmbeddingMethod`. Parameter creation at line 53 uses
  `params_dtype`, and `embedding:77` calls `F.embedding` on that parameter.
- `vllm/envs.py:1335` defines the default skinny-GEMM setting. Preserve it when
  preparing the oracle. A diagnostic configuration is not the denominator.
- `vllm/model_executor/model_loader/weight_utils.py:1227` defines
  `default_weight_loader`. Its `param.data.copy_` at line 1241 converts loaded
  values to the already allocated parameter dtype. The linear loader at
  `linear.py:572` and parameter loader at `parameter.py:176` carry the same
  parameter-storage contract.

For GGUF, read the complete first-party plugin chain at
`d4c1f0d082fc7cd4350da56689109a01c1f29d6c`, recorded in
[the plugin pin](../oracles/vllm-gguf-plugin.md):

1. `vllm_gguf_plugin/weight_utils.py:217`,
   `get_gguf_unquantized_params`, classifies F16 file tensors as unquantized.
   `gguf_quant_weights_iterator_multi:169` yields the native file tensor.
2. `vllm_gguf_plugin/loader.py:69`, `_get_unquantized_modules`, identifies their
   modules. `_prepare_adapter:176` and `load_model:203` carry that selection
   into the quantization configuration.
3. `vllm_gguf_plugin/quantization/config.py:71` returns
   `UnquantizedLinearMethod` at line 82 and `UnquantizedEmbeddingMethod` at
   line 91 for those modules.
4. `loader.py:217` enters `set_default_torch_dtype(model_config.dtype)` before
   model initialization and the weight load at line 229. The vLLM parameter
   loader then performs the conversion described above.

The plugin's `quantization/linear.py:35` unquantized fallback does not prove
that an ordinary F16 file tensor reaches a raw F16 multiplication. The module
selection and destination parameter dtype determine that path first.

The plugin currently records `gateable = no` and names
[#2624](https://github.com/mudler/vllm.cpp/issues/2624) for its Qwen3.8 workload.
This row owns the Qwen3.5 measurement it requires. Importing a wheel or
constructing a configuration does not close either model's gate. The operator
must run the actual active-pin model before accepting a denominator.

The ROCm 7.2.2 [rocBLAS GEMM extension contract](https://rocm.docs.amd.com/projects/rocBLAS/en/docs-7.2.2/reference/extension.html)
lists equal A and B input types. It does not establish BF16 by F16 mixed-input
support. Probe the installed library on gfx1100 and record the accepted A, B,
C, D, compute, and scalar types before choosing a direct call.

## Our baseline

At the pinned base, `DeviceKeepF16Supported` in
`src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:170` excludes ROCm.
`GgufLoadPolicy::FromEnv` at line 360 therefore disables retained F16 storage.
`RouteGgufTensor:275` currently uses a general F16 role predicate that includes
stacked experts. Removing only the device exclusion would admit unsupported
expert and embedding consumers.

`MatmulKernelRocm` at `src/vt/rocm/rocm_matmul_hipblaslt.hip:491` and
`MatmulBTKernelRocm:542` accept BF16 pairs or F32 pairs. `ToBlasType` already
knows F16, but that helper does not make the registered entry points accept it.
`src/vt/ops.cpp:118` and line 151 already admit floating inputs and restrict
outputs to BF16 or F32. `EmbeddingKernelRocm` at
`src/vt/rocm/rocm_embedding.hip:109` rejects F16 tables.

The legacy keep-F16 work in [L6 and L7](gguf-keep-quant-loader.md) retains file
bytes and mmap ownership to avoid BF16 expansion copies. L7 also releases dead
repack source pages and prefaults borrowed weights. A persistent converted
weight shadow would erase part of that residency benefit. Transient conversion
must have measured memory costs and cannot be described as free retention.

The local Qwen3.5-4B Q4_K_M checkpoint inspected for this task has no F16
tensors. It is a regression asset and cannot witness the new capability.

## Port map

| Contract | Local surface | Change |
|---|---|---|
| Weight values distinct from storage | `OwnedTensor`, `vt::Tensor` | Optional weight-only value dtype |
| Explicit resolved dtype at admission | `GgufLoadPolicy`, `qwen3_5_dense.cpp` | Supply model dtype in the accepting registry loader |
| Retained ordinary weights and aliases | `qwen3_5_gguf_weights.cpp` | Preserve F16 bytes and value metadata |
| Residency and shared forwarding | `OwnedTensor::View`, both `ResidentWeight` helpers | Preserve operand metadata through every return |
| Primitive validation | `src/vt/ops.cpp` | Validate the new weight contract before dispatch |
| Ordinary GEMM | `rocm_matmul_hipblaslt.hip` | F16 inputs and faithful conversion fallback |
| Embedding | `rocm_embedding.hip` | F16 read followed by the declared value conversion |
| Coverage and registration | `tests/vt`, `tests/vllm`, test CMake lists | Production, dtype, lifetime, and negative tests |

The implementation may add a small ROCm helper file for conversion and scratch
ownership. New tests may use a dedicated `test_rocm_f16_weights` target. Keep
new implementation additive where practical. Do not edit CI workflows.

The implementer's authorized production files are:

- `include/vt/tensor.h`, `src/vt/tensor.cpp`, `include/vt/ops.h`, and
  `src/vt/ops.cpp` for the scoped operand contract.
- `include/vllm/model_executor/models/qwen3_5_weights.h` and
  `src/vllm/model_executor/models/qwen3_5_weights.cpp` for ownership metadata.
- `include/vllm/model_executor/model_loader/gguf_keep_quant.h` and its
  `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp` implementation.
- `include/vllm/model_executor/models/qwen3_5_gguf_weights.h` and
  `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp` for retained storage.
- `src/vllm/model_executor/models/qwen3_5_dense.cpp`,
  `src/vllm/model_executor/models/qwen3_5.cpp`, and
  `include/vllm/model_executor/models/dense_attn_block.h` for the accepting
  registry and existing shared weight consumers.
- `src/vt/rocm/rocm_matmul_hipblaslt.hip`,
  `src/vt/rocm/rocm_embedding.hip`, and a new scoped F16 conversion helper
  beneath `src/vt/rocm` when needed.
- `CMakeLists.txt`, `tests/CMakeLists.txt`, scoped existing or new tests under
  `tests/vt` and `tests/vllm`, and a new token fixture beneath
  `tests/fixtures/rocm_f16_weights` for registration and behavioral coverage.

The row spec, its issue record, and the public documents named below carry the
resulting records. Changes outside these paths require operator scope review.

### Carry an explicit weight contract

Choose optional `weight_value_dtype` metadata on `OwnedTensor` and
`vt::Tensor`. Unset means that a weight has its storage values. A set value
applies only to the B operand of ordinary GEMM or the table of embedding.
`Tensor::dtype`, `Bytes`, shape, and strides continue to describe storage.
The admitted marker is F16 storage with BF16 or F32 values on ROCm.

The alternative was new `MatmulAttr` and `EmbeddingAttr` arguments. Neither
registered function type currently has attributes. That alternative changes
every provider signature and repeats weight propagation at callers. The
selected operand field keeps the existing shared operation entry points and
limits interpretation to the three named consumers. It is not a global cast
rule for every tensor operation.

Validate the marker before dispatch. Reject it on activation, output, and ID
operands, on an unsupported storage/value pair, or on an unsupported provider.
Audit every reachable consumer of the newly retained weights. No consumer may
ignore a set marker. A missing consumer requires either a scoped extension
here or continued loader refusal for that role.

Carry metadata through copy, move, `View`, and `Slice`, through both
`ResidentWeight` implementations, through all resident return paths, and
through a tied embedding/head alias. Centralize descriptor propagation instead
of hand-copying it at each final return. Host-byte release must retain the
resident operand's metadata and storage dtype. Neither a retained mmap nor a
tied head may acquire a second permanent converted weight allocation.

### Admit only consumers that carry the contract

Extend `GgufLoadPolicy` to accept an explicitly resolved model dtype. The
Qwen3.5 dense registry loader supplies that value. On ROCm, default keep-F16
requires the value and an admitted ordinary matmul or embedding role.

An existing loader that does not supply the value keeps its prior BF16
expansion. Stacked experts and value-transformed tensors keep their prior
routing, regardless of the file's F16 type. Preserve `VT_GGUF_KEEP_F16=0`,
CPU-reference behavior, and CPU and CUDA defaults. An explicit unsupported
model dtype must not silently select BF16 semantics.

The production path is `include/vllm.h` to `LoadedEngine::FromModelDir`, the
GGUF `ModelSource`, `LoadQwen3_5DenseModel`, and the registered dense forward.
The forward reaches `MatmulF32D`, `MatmulBf16D`, ordinary shared attention and
MLP helpers, and `vt::Embedding`. Preserve shared `FusedChain`, `AttnBlock`,
`MlpGateUpMethodBase`, and `MergedGemmGroup` routing where already applicable.
Do not create a separate example-only or model-private F16 GEMM route.

### Preserve arithmetic on ROCm

For a marked BF16-value weight, round F16 to BF16 before multiplication. With
BF16 activations, use a temporary BF16 operand and the existing BF16 dispatch
so the original compute policy remains available. With F32 activations, round
the weight to BF16 first, then upcast exactly when a homogeneous F32 GEMM is
needed. Do not round or overflow F32 activations by casting them to F16.

For unmarked F16 inputs, preserve their exact values. Use direct hipBLAS GEMM
where the installed type contract allows it. Use explicit conversion for an
unsupported mixed pair. An F32 temporary needs a nearby reason that names the
unsupported pair. Preserve F32 accumulation and the requested BF16 or F32
output. A BF16 output cannot be assumed supported merely because F16 inputs
are supported. Record the exact direct and fallback combinations.

Preserve NN contiguity and BT row-strided activation support. Preserve shape,
rank, device, and stride rejection, empty M or N behavior, and K=0 output
zeroing. Respect queue ordering and current-device binding. Scope scratch by
actual device and stream identity. Reuse bounded temporary storage safely
across calls without a permanent cache per weight. An environment-selected
compute override must use matching scalar storage or refuse the unsupported
combination by name. Do not pass F32 alpha bytes as a 16-bit scalar.

Embedding converts each selected F16 value according to the marker before the
requested output conversion. Preserve repeated and boundary IDs, I32 and I64
IDs, empty inputs, and the current invalid-ID error and bounds behavior.

## Tests to port

Port the applicable ordinary fallback case from
`tests/model_executor/layers/test_rocm_unquantized_gemm.py:152` at the active
vLLM pin. Preserve its F16 inputs, activation shape `[6,64]`, weight shape
`[128,64]`, and `atol=rtol=1e-3`. Document the C++ harness adaptation. The local
seam returns BF16 or F32. For this port, request F32 output, round that result
to F16, and compare it with the upstream F16 result at the original tolerance.
Separately check the unrounded result against F32 accumulation over exactly
widened F16 inputs. Add BF16 as an explicit local regression parameter with
the expected final BF16 rounding, rather than attributing it to this test.
The contiguous-activation and skinny-selection cases at lines 20, 45, 71, 98,
and 135 describe existing specialized dispatch. Keep them as regression
obligations when that dispatch is changed. Do not claim to port new skinny
kernels in this row.

Add tests that expose these guarantees through existing production seams:

- Default Qwen3.5 dense loading of a synthetic F16 GGUF keeps eligible F16
  bytes and metadata, including tied and explicit heads. Enter through the
  registry or public loader. Do not only construct `OwnedTensor` by hand.
- The loaded model's registered forward invokes the retained embedding and
  ordinary projection paths. Use values such as `1 + 2^-10` that distinguish
  raw F16 arithmetic from BF16-rounded weight arithmetic.
- Exercise copy and move, mapped and copied storage, tied aliases, shape views,
  slices, each resident return path, and post-upload host release.
- Cover NN and BT with M=1, M=4, M=6, a larger non-tile multiple, odd N and K,
  row-strided BT input, BF16 and F32 outputs, and all newly admitted input pairs.
- Use F32 activations outside F16's range and values too small for F16. Compare
  against explicit F32 accumulation after the declared weight conversion.
- Cover M=0, N=0, K=0, invalid ranks, shapes, strides, devices, dtype markers,
  and providers. Re-run existing BF16 and F32 cases.
- Cover F16 embedding with both ID widths, repeated IDs, edge rows, invalid
  negative and vocabulary-sized IDs, empty IDs, and BF16 and F32 outputs.
- Verify `KEEP_F16=0`, CPU-reference, unwired ROCm registry loaders, stacked
  experts, and existing CPU and CUDA routing remain correctly selected.

The smallest test must fail on the baseline for the intended production
admission or execution reason. Retain the red command and output before
implementation, then focused green and the full gate.

## Gates

| ID | Requirement | Current result |
|---|---|---|
| F16-G1 | Spec committed before implementation, issue and row agree | Satisfied by this spec commit |
| F16-G2 | Focused red, focused green, and registered CPU/HIP tests | PENDING implementation |
| F16-G3 | Default public-load and registered-forward reachability | PENDING implementation |
| F16-G4 | Identical-artifact active-pin oracle and exact token gate | PENDING model run |
| F16-G5 | Same-tool executed dtype and dispatch traces | PENDING paired traces |
| F16-G6 | Same-binary A/B and oracle speed, latency, and memory | PENDING F16-G4 |
| F16-G7 | Fresh immutable-head mutation review and operator rerun | PENDING reviewed head |
| F16-G8 | Full repository preflight, no skipped applicable gates | PENDING final head |

### Run the model and compare values

Build a real F16 GGUF fixture from the existing Qwen3.5-0.8B asset only after
recording the exact upstream checkpoint repo, revision, file names, sizes, and
hashes. Use the recorded llama.cpp converter pin, not a floating checkout.
The converter revision is `10bf611e533d81f739128304991c5e133c6aebd8`, tag
`b10451`, from [the llama.cpp oracle record](../oracles/llama-cpp.md).
The inspected source is `Qwen/Qwen3.5-0.8B` at
`2fc06364715b967f1860aea9cf38778875588b17`. Its shard
`model.safetensors-00001-of-00001.safetensors` contains 1,746,942,600 bytes and
has SHA256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
All 13 cached files' measured hashes match their recorded metadata, and their
metadata names that same revision. Retain that complete source manifest with
the conversion evidence, including the config and tokenizer files.
Record the converter command, revision, output hash, tensor dtype histogram,
and complete source-to-GGUF tensor mapping with names, shapes, and conversion
rules. Validate the conversion instead of assuming BF16 and F16 interchange.
The operator supplies asset and launcher paths. Repository prose is not a
source of host configuration defaults.

Prefer the active vLLM plus pinned GGUF plugin on the identical GGUF. If that
plugin cannot run this model, keep this row's model gate pending under #3092
and retain the plugin registry's existing #2624 debt. A validated HF
materialization of the exact GGUF tensors may provide the primary arithmetic
comparison only when its complete mapping and dtype conversion are proven.
Do not silently compare a differently rounded original HF checkpoint.
The registered secondary oracle can answer correctness for an ungateable GGUF
path under its recorded policy. It does not override vLLM's model-dtype rule.

Commit `tests/fixtures/rocm_f16_weights/token-workloads.json` before the red
run. Its input arrays follow this fixed definition: request index `r` receives
IDs `1 + 128*r + i` for `i` from 0 through `P-1`. Materialize those arrays in
the fixture and verify every ID is in the model vocabulary. These are explicit
token prompts. Neither arm tokenizes, adds a chat template, or adds BOS.

| Workload | Prompt P | Generated tokens per request | Request batch | Concurrency | Request indices |
|---|---|---|---|---|---|
| F16-D1 | 1 | 32 | 1 | 1 | 0 |
| F16-P16 | 16 | 32 | 1 | 1 | 0 |
| F16-P128 | 128 | 64 | 1 | 1 | 0 |
| F16-C4 | 16 | 32 | 4 | 4 | 0, 1, 2, 3 |

Set temperature to 0, top-p to 1, repetition penalty to 1, presence and
frequency penalties to 0, and seed to 0. Disable stop strings and stop-token
lists. Set `ignore_eos=true` and `min_tokens=max_tokens` to the row's generation
count. Both arms must emit 256 generated IDs across the four workloads.
Capture each request separately and compare complete arrays for exact equality.
Run the matrix twice per arm to establish the oracle's repeatability.

Use `vllm_complete_tokens` for our pre-tokenized public path and vLLM's
`TokensPrompt` on the oracle. Use a HIP-only build with the C ABI's automatic
integer device selection, `device=0`. Assert the resolved device is ROCm and the new
operations use native ROCm providers. The public device field does not accept
the string `"rocm"`, and a successful CPU fallback is not this witness.
For F16-C4, submit all four requests together,
set the sequence capacity to 4, and retain the actual scheduled batch trace.
Do not compare concurrent timing if the effective batching differs. The other
three workloads use sequence capacity 1. The generated real checkpoint and
the small production fixture cover tied and explicit heads between them.

Use the production oracle configuration without `--enforce-eager`. A known
baseline mismatch needs its own evidence and owner. It cannot waive a mismatch
caused by this change.

### Trace the actual operations

Use the same tracing tool on both arms. Record the original F16 storage and
the values' resolved dtype separately. Identify every cast, GEMM, and gather
that executes. Record A, B, C, and D dtypes, output dtype, compute type, scalar
type, entry point, algorithm policy, and resolved template types. Dump generated
kernels before declaring any upstream lever unreachable.

A conversion-plus-BF16 GEMM is an explicit storage adaptation, not a claim of
native F16 invocation parity. Prove that the raw primitive F16 path executes,
and that the marked model path matches the oracle's rounded values. Annotate
every newly introduced F32 model-path buffer with its reason.

### Measure retention and performance

After F16-G4 passes, run an idle-host same-binary A/B with keep-F16 enabled and
disabled. Record at least three alternating legs and retain every raw log.
Measure load time, prefill throughput, decode throughput, end-to-end latency,
peak RSS, peak device memory, persistent weight bytes, and transient scratch
high water. Trace conversion traffic, especially a vocabulary-sized head.

Report values and ratios against the production oracle for every applicable
axis. The parity floor is throughput ratio at least 1.00 and latency and memory
ratios at most 1.00, subject only to the owning row's explicit existing gate.
An axis below its floor remains an open gap with the next traceable hypothesis.
Do not accept a performance result before correctness, hide conversion scratch
inside a weight-memory claim, or call a regression an architectural ceiling.
Preserved resident F16 bytes alone do not prove a reduction in peak memory.

An accepted paired measurement establishes the effect of this storage change
only after its identical-workload and correctness requirements pass. It does
not close the parent backend's existing overall performance gaps. Report both
the same-binary delta and the absolute oracle ratios. Preserve each existing
parent gap's owner and floor. A positive local delta cannot waive an absolute
floor, and token equality cannot waive a dtype or invocation mismatch.

### Verify the immutable head

Run `scripts/agent-preflight.sh` before edits and before the spec commit. On the
implementation head, run the focused registered CPU and ROCm suites, applicable
full CTest suites, and preflight. Record exact configure flags, compiler and HIP
versions, revisions, commands, exit statuses, logs, and omitted gates.

A fresh reviewer mutates each claimed guarantee in a scratch copy. At minimum,
disable loader admission, delete the production forward call, drop metadata
from each residency path, omit BF16 weight rounding, cast F32 activations to
F16, break K=0 zeroing, and misread one ID width. Each corresponding focused
gate must fail. Restore the immutable tree byte-for-byte after every mutation.
The operator reruns the row's gate on the reviewed SHA.

## Dependencies

The existing ROCm backend, shared ordinary operation providers, GGUF loader,
and Qwen3.5 dense registration are available at the base. There is no dependency
on the quantized GEMM pull request. GPU work requires the operator's recorded
host, toolchain, active oracle identity, and exclusive device ownership.
Use the configured fleet lease where applicable and the configured mutex on a
non-fleet device. This spec authoring task runs no GPU work.

## Work breakdown

1. Commit this spec and the scoped issue and inventory records.
2. A fresh implementer captures the production failing test and dtype fixtures.
3. Implement the minimum complete metadata, loading, GEMM, and gather change.
4. Run focused and full gates, then the identical-artifact model comparison.
5. Capture paired traces and post-correctness performance and memory evidence.
6. A fresh reviewer mutates the immutable head. Fresh implementers repair findings.
7. The operator reruns gates, reads the final pull request body, and publishes
   the merge request for the user's review. The current task does not merge it.

The implementation owns `docs/FEATURES.md` when the capability becomes
reachable. It owns `docs/USAGE.md` for the exact gated checkpoint information
and the remaining refused arms. Publish a benchmark detail and index entry
only when a benchmark is actually accepted. Do not rewrite public documents
for the spec-only commit.

## Risks/decisions

The main risk is numerical: raw F16 file values can differ from those values
rounded to the model's BF16 parameters. Token equality cannot substitute for
tracing that conversion. The selected metadata and explicit cast preserve it.

The second risk is residency: repeated conversion can increase temporary
memory and traffic. Measure that cost before changing defaults. A permanent
BF16 shadow or silent retention regression needs a new decision.

The third risk is incomplete metadata propagation through shared helpers or
tied weights. Limit loader admission to the traced Qwen3.5 dense path and use
negative mutations to prove that every required copy is tested.

The fourth risk is a broader primitive contract being mistaken for a complete
F16 model runtime. This row retains BF16/F32 activation and output contracts.
The existing RMSNorm F16 activation refusal is tracked by #2542. Other missing
full-runtime operations need their own owning issue before that scope expands.

## Owed

- [#2542](https://github.com/mudler/vllm.cpp/issues/2542), owned by
  `MODEL-MM-QWEN4-EXP` and the `Owed` section of
  [the RMSNorm dtype spec](rmsnorm-gamma-dtype-twins.md), tracks the existing
  CUDA and ROCm RMSNorm F16 activation refusal. It is not ownership of every
  missing F16 runtime operation. F16 model activations and F16 outputs remain
  outside this row.
- Other model registry loaders retain their existing BF16 expansion until they
  explicitly provide and propagate a resolved value dtype. This row makes no
  retained-F16 support claim for them. The owning backend row must track any
  subsequent admission with that model's production gate.
- Stacked experts retain expansion because their grouped block-quantized seam
  does not accept F16 weights. Any later retention needs a separately specified
  grouped F16 consumer and its owning issue before admission.
- #2624 owns the plugin registry's Qwen3.8 gateability measurement. This row
  owns its Qwen3.5 measurement under #3092 and cannot declare that primary path
  gateable from source inspection.

## Row inventory

| ID | Upstream source | Local anchor | Tests and evidence | Spec | State | Owner | Issue |
|---|---|---|---|---|---|---|---|
| `BACKEND-ROCM-F16-WEIGHTS` | vLLM `e126687a9a` ordinary ROCm GEMM and parameter conversion, GGUF plugin `d4c1f0d082` | `MatmulKernelRocm`, `MatmulBTKernelRocm`, `EmbeddingKernelRocm`, `GgufLoadPolicy::FromEnv` | F16-G1 through F16-G8 above, implementation evidence pending | [This spec](rocm-f16-weights.md) | `READY` | Fresh helper, operator verification | [#3092](https://github.com/mudler/vllm.cpp/issues/3092) |

This per-row inventory is the canonical child record, discovered from the spec
glob. The parent backend-matrix retains its existing state and content.
Add an `Outcome` section only when the row reaches `DONE`, with measured values,
rejected alternatives, and the reasons for the final defaults.
