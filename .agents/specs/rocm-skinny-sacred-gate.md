# ROCm skinny GEMM sacred-gate adjudication

## Now

Issue [#2772](https://github.com/mudler/vllm.cpp/issues/2772), owned by Row
`BACKEND-ROCM`, is **FAILING** at source
`4d10c8acc527c34a6a58a309d52ea5f8fbd1d47b` (tree
`68cb23e690cf0b76f43aa01235e328de38fca145`). The default Qwen3.5-0.8B
checkpoint gate first disagrees with its committed local anchor at prompt 10,
generated token 10: the engine emits token 369 and the anchor records token
488. The implementation may begin only after this spec commit. It may not
replace the committed anchor or near-tie gaps until the pinned oracle has
adjudicated the first shared-prefix disagreement.

The source checkout for the pinned vLLM revision is present. A runnable oracle
is not: `VLLM_ORACLE` is empty and the former local image
`vllm-rocm-oracle:555967922-gfx1100` is absent. Oracle reconstruction and every
conclusion that depends on it are therefore **PENDING**, not waived. This spec
and the later implementation share one pull request, the repository default.

## Goal

Determine whether the default local wvSplitK path is a faulty vLLM mirror or a
correct mirror that exposes a stale local golden. Then make the smallest change
that restores a causal, checkpoint-backed default gate without choosing that
answer in advance.

The change must:

1. establish the exact production `MatmulBT` shapes and selected arms on the
   shared-prefix disputed step;
2. compare the local skinny and BLAS arms from identical operands, and compare
   both with the executing pinned upstream path;
3. reconstruct the pinned vLLM oracle and ask the one-token, teacher-forced
   same-prefix question before using any new golden;
4. repair the mirror, rederive the local golden, or narrow the default only as
   the pre-registered evidence below permits;
5. port the applicable upstream kernel and routing tests; and
6. keep one default checkpoint test entering through
   `LoadedEngine::FromModelDir` and prove that the wvSplitK production call is
   indispensable to that gate.

Correctness precedes performance. Matching the current BLAS arm or the old
local token 488 does not establish correctness by itself.

## Acceptance ledger

Each applicable obligation has one current result:

| Obligation | Result now | Closure evidence |
|---|---|---|
| Default sacred checkpoint gate | **FAILING** | Prompt 10/token 10 is 369 versus anchor 488. |
| Regression isolation | **SATISFIED** | Skinny-off 137/137, unrelated Q8K/attention arms still red, HIP-7.15 known-good green, and first bad `f38c1edc4`. |
| Source and artifact pins | **SATISFIED** | Exact local source/tree, upstream revision, model revision, file names, sizes, and SHA-256 values are recorded below. |
| Complete pinned source chain | **SATISFIED for source inspection** | Environment default, Python dispatch, custom-op binding, host configuration, and RDNA3 body are cited below. |
| Production shape/configuration census | **PENDING** | Requires the internal selection record and a locked checkpoint run. |
| Same-input local skinny/BLAS outputs and logits | **PENDING** | Requires scratch capture on gfx1100. |
| Runnable pinned vLLM and exact-prefix answer | **PENDING** | `VLLM_ORACLE` is empty and the old image is absent. Source alone is insufficient. |
| Decision branch | **PENDING** | No branch is selected before paired local and runnable-oracle evidence. |
| Applicable upstream test port and causal checkpoint gate | **PENDING** | Belongs to the later implementation, not this prerequisite commit. |
| Performance acceptance | **PENDING only if branch C applies** | Correctness must select branch C before same-workload timing is admissible. |

There is no waiver in this spec.

## Scope

In scope:

- the BF16, no-bias, packed-row gfx1100 wvSplitK arm selected by the Qwen3.5
  production decode;
- the local `MatmulBT` internal selection seam, its minimal diagnostic record,
  and the tests that consume that record;
- the executing pinned vLLM gfx1x dispatch, host configuration, and RDNA3
  kernel body;
- exact-prefix local skinny, local BLAS, and pinned-vLLM output/logit evidence;
- the Qwen3.5-0.8B sacred gate and only the golden files justified by the
  selected evidence branch; and
- same-workload performance evidence only if the evidence selects the
  narrowing/default-disable branch.

Out of scope:

- cache-mode, model-wide state-space, convolution, attention, and layerwise
  numerical characterization owned by issue
  [#2773](https://github.com/mudler/vllm.cpp/issues/2773);
- FP16, bias, padded-weight, gfx9 wave64, gfx950, gfx1151, medium-LDS, and
  big-LDS implementations that the production Qwen3.5 BF16 arm does not use;
- Q8K and paged-attention repairs already excluded by controlled rollback;
- a public debug API or a new user-facing environment switch;
- changing vLLM's model, sampling, or cache defaults; and
- treating the provisional Qwen3.5-4B trace as issue #2772 acceptance evidence.

If execution evidence reaches an out-of-scope arm, stop and request a scope
decision. Do not silently expand the port.

## Grounded red evidence

The following results are binding evidence from issue #2772. They are not
rerun by this spec-only change because every run requires the shared GPU.

- The exact current default dedicated test runs one real doctest case and
  fails at `prompt[10]`, generated token 10: engine 369, committed local anchor
  488. The gate reports native ROCm providers; it does not take a reference
  provider.
- The same binary with `VT_ROCM_SKINNY=0` passes 1/1 case and 137/137
  assertions. `VT_ROCM_Q8K_BLOCK=0` and `VT_ATTN_DECODE_D128=0` still fail at
  the same cell.
- Commit `7b89cf30775df79155d4bcc4f3cabf57637f74fc`, rebuilt under the current
  HIP 7.15 toolchain, passes 137/137. A GPU-locked bisect that rebuilt and ran
  the real test at every candidate identifies
  `f38c1edc4ea679348f856c0c0b20fb0702f77daf` as first bad. That commit enabled
  local wvSplitK by default.
- `VT_DUMP_IDS=1` reports six changed cells, all prompt 10, tokens 10 through
  15. Only token 10 has the committed input prefix. The five suffix cells have
  already consumed a different token and the old gap table cannot adjudicate
  them. The old prompt-10/token-10 gap is 125 milli-nats for committed token
  488 on that shared prefix; it does not report the gap or rank of skinny token
  369.
- `/tmp/gfx1100-current-profile-4d10` is provisional only. Its different
  quantized 4B model records 72 wvSplitK calls per token and 2.4651 ms per
  token. It cannot decide #2772 correctness or satisfy a performance gate.

The existing sacred path is
`tests/parity/test_qwen35_paged_engine.cpp::RunGate`. Its only actual doctest
case reaches `LoadedEngine::FromModelDir`, generates all 16 prompts, checks the
committed local anchor before the near-tie loop, and records native-provider
selection. Provider selection proves that ROCm owns `kMatmulBT`; it does not
distinguish the wvSplitK and BLAS branches inside that provider.

## History

Issue [#487](https://github.com/mudler/vllm.cpp/issues/487) identified the
decode-skinny rocBLAS tile mismatch and led to PR
[#506](https://github.com/mudler/vllm.cpp/pull/506). The landed commit
`f38c1edc4ea679348f856c0c0b20fb0702f77daf` introduced the local gfx1x BF16
kernel and made it the default for eligible shapes. Its commit and pull request
reported that the Qwen3.5 gate remained unchanged. The current exact-test
bisect falsifies that claim on the present HIP 7.15 build; it does not by itself
say whether the kernel or the old local golden is correct.

The reviewed follow-up recorded the safe local preconditions in
`.agents/specs/rocm-skinny-gemm.md`: even output features greater than eight,
packed BF16 rows, one through four input rows, `K % 8 == 0`, small-LDS fit, and
wave32 architecture. This issue preserves those guards unless executing-chain
evidence selects branch A or C below.

## Artifact and environment pins

The exact model is
`Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`, present at
`/home/vikash/models/Qwen3.5-0.8B`. The files that must be mounted unchanged on
both sides are:

| File | Bytes | SHA-256 |
|---|---:|---|
| `model.safetensors-00001-of-00001.safetensors` | 1,746,942,600 | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| `model.safetensors.index.json` | 50,900 | `d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e` |
| `config.json` | 2,907 | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204e` |

The index declares one shard, 488 weights, and total tensor bytes
1,746,882,752. The model resolves BF16 weights and BF16 model activations; its
documented state-space exception remains f32. This issue does not change that
polarity.

All later GPU work runs on the local RX 7900 XTX (`gfx1100`) under HIP 7.15.
Every GPU command must hold `flock /home/vikash/gpu.lock` and use exactly the
sanitized ROCm runtime path
`LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib`. Evidence records the
binary SHA, compiler and runtime versions, board identity, clocks, contention
state, command, environment, exit code, and output path.

## Pinned vLLM executing chain

The primary oracle is the checkout at `/home/vikash/oracle/vllm-src`, commit
`5559679229bc961848b121ccdeaa8fa5d79bec98`, the parity pin recorded in
`.agents/upstream-sync.md`. Source inspection establishes this complete chain:

1. `vllm/envs.py::VLLM_ROCM_USE_SKINNY_GEMM` resolves unset to true.
2. The `apply` method on
   `vllm/model_executor/layers/linear.py::UnquantizedLinearMethod` enters
   `vllm/model_executor/layers/utils.py::dispatch_unquantized_gemm`.
3. `vllm/model_executor/layers/utils.py::rocm_unquantized_gemm_impl` selects
   the ROCm skinny path for gfx9/gfx1x FP16 or BF16, `K % 8 == 0`, output
   features greater than 8, and one through five input rows. On gfx1x this arm
   calls wvSplitK; more than five rows fall back to ordinary linear execution.
4. `vllm/_custom_ops.py::wvSplitK` invokes the `_rocm_C` custom operation
   registered by `csrc/rocm/torch_bindings.cpp::wvSplitK`.
5. `csrc/rocm/skinny_gemms.cu::wvSplitK` reads strides, output and bias shape,
   compute-unit count, and LDS size. For gfx1x it launches wave32 with 16 waves
   per group. It derives `sYT = ceil(M_in / (CuCount * 4))`, then selects
   YTILE/UNRL as 1/4 for `sYT <= 1`, 2/2 for a one-row activation, an LDS miss,
   or `sYT <= 8`, 3/2 for `sYT <= 12`, 4/1 for four input rows, and 4/2
   otherwise. It selects the small body only when `Kbp * N_in` fits the BF16
   element capacity (`LDS bytes / 2`) and `M_in` is divisible by YTILE, the
   medium body up to 1.2 times that capacity, and the big body otherwise. It
   has template arms for one through five input rows.
6. On the applicable RDNA3 small-body arm,
   `csrc/rocm/skinny_gemms.cu::wvSplitK_hf_sml_` stages the activation in LDS,
   loads weights nontemporally, expands BF16 pairs to f32, accumulates in f32,
   reduces wave32 with DPP row shifts plus shuffle-xor 16, adds any bias, and
   stores BF16 output. This is the executing body, not an adjacent candidate.

The corresponding local production route is
`src/vllm/entrypoints/model_loader.cpp::LoadedEngine::FromModelDir` through
`src/vllm/model_executor/models/model_registry.cpp::ModelRegistry::Forward`
and the Qwen3.5 projections including
`src/vllm/model_executor/models/qwen3_5.cpp::MatmulBTRawD`, then
`src/vt/ops.cpp::MatmulBT` to the registration in
`src/vt/rocm/rocm_ops.hip::Registrar`, then
`src/vt/rocm/rocm_matmul_hipblaslt.hip::MatmulBTKernelRocm` and
`src/vt/rocm/rocm_matmul_hipblaslt.hip::SkinnyGemmEnabled`, followed by
the architecture predicate
`include/vt/rocm/rocm_skinny_gemm_arch.h::SkinnyGemmArchOk`,
`src/vt/rocm/rocm_skinny_gemm.hip::WvSplitKBT` and
`src/vt/rocm/rocm_skinny_gemm.hip::wvSplitKSml`.

The local body appears to match the applicable RDNA gfx1x BF16 arithmetic and
wave32 reduction, but that is not yet a parity result. Its host policy is a
narrow port: one through four rows, packed BF16 input/output, no bias, fixed
YTILE=2 and UNRL=2, small-LDS only, even output features, and gfx11/gfx12. The
pinned host dynamically chooses YTILE/UNRL, supports row count five, and can
reach bias, strided, FP16, medium, and big arms. The shape census must prove the
configuration used by Qwen3.5 before any claim that the local body mirrors the
executing upstream body.

## Minimal production-path instrumentation

The implementation first adds an internal, test-owned selection record at
`MatmulBTKernelRocm`. It is not exposed through `include/vllm.h`, is inert when
no test observer is installed, allocates nothing on the normal path, and does
not add an environment variable. The record contains:

- monotonically increasing `MatmulBT` call ordinal;
- device and architecture;
- exact logical `{M_tokens, N_features, K}`;
- input, weight, and output dtypes and row strides;
- each eligibility predicate and the selected arm (`wvSplitK`, hipBLASLt,
  hipBLAS GEMM, or another existing arm); and
- for wvSplitK, compute-unit count, LDS-fit result, THRDS, waves per group,
  YTILE, UNRL, and instantiated input-row template.

The existing operation-provider statistics remain the proof that the public
production dispatch selected native ROCm. The new record answers only the
missing internal-arm question. A focused CPU-side routing test installs the
observer and proves that every guard is narrated correctly; it does not claim
kernel correctness.

The one checkpoint gate consumes aggregate counts and the exact reached shapes
from this same record. It must assert at least one production wvSplitK call and
the frozen disputed-step shape inventory. That prevents a future default
change from making the token gate green by bypassing the kernel.

### Paired operand and output capture

Detailed capture is correctness diagnosis, not shipped tracing. Use a scratch
instrumentation patch around the same production dispatcher and remove it
byte-for-byte before review. Run two fresh processes from the exact numeric
teacher prefix below: ambient/default skinny, then `VT_ROCM_SKINNY=0` as a
diagnostic BLAS control. Do not execute an extra shadow GEMM in the same model
run.

For every eligible call until the one-token logits are produced, write a
manifest containing the call ordinal, shape/configuration record, activation
and weight SHA-256, raw BF16 output SHA-256, and output bytes. Synchronize only
after the real operation has completed, copy the completed result without
altering it, and resume the unchanged model output. Pair calls only when
ordinal, shape, activation bytes, and weight bytes match. Report the first
output element that differs, maximum absolute and relative difference, and
whether the difference survives BF16 storage. If the inputs do not match, the
pair is invalid. The first call with matching inputs and different output is
the causal local skinny/BLAS boundary. Calls after that boundary consume
different model state and may be recorded as consequences, but they must not
be described as paired evidence.

Use the existing `src/vllm/v1/worker/gpu/runner.cpp::GPUModelRunner::dump_step_logits`
seam (`VT_DUMP_LOGITS`) to capture the final f32 logits before sampling and its
argmax sidecar. Its current tests already establish alignment and default
inertness. The new work must repeat a capture-off/capture-on token comparison
to show that scratch operand capture also does not perturb output. These dumps
may diagnose correctness; their synchronizations make them invalid for timing.

If a reproducible capture cannot be made through the production dispatcher
without changing its output, stop with `NEEDS_DECISION`. Do not land a public
debug switch to make progress.

## Exact same-prefix oracle question

Prompt 10 is `The mitochondria is the powerhouse of`. Its tokenizer prefix is:

```text
760 52132 4065 369 279 71594 314
```

The committed local output is shared through generated token 9:

```text
279 2691 13 271 248068 271 248069 271 760 4952
```

Therefore the only admissible teacher-forced input for the disputed next token
is this 17-token numeric sequence:

```text
760 52132 4065 369 279 71594 314 279 2691 13 271 248068 271 248069 271 760 4952
```

Construct the request from these token IDs, not decoded/re-tokenized text, and
run exactly one next-token step in a newly loaded engine with an empty cache,
batch size one, greedy sampling, temperature zero, no chat template, and no
other concurrent request. Record the resolved model dtype, physical KV-cache
dtype, prefix-caching state, compilation mode, sampling seed/defaults, and
custom-operation selection on each side. Local must enter through
`LoadedEngine::FromModelDir`; pinned vLLM must use the identical artifact and
configuration. `auto` cache resolution must be recorded as the physical dtype,
not left as the word `auto`.

For this artifact, reproduce the sacred test's `EngineParams{}` resolution
explicitly on both sides: block size 32, 256 KV blocks, maximum sequences 32,
maximum batched tokens 8,192, prefix caching disabled for the hybrid model,
model maximum length 262,144, BF16 model dtype, and `auto` resolved to physical
BF16 KV storage. The local request uses the existing `Greedy(1)` sampling
construction; the upstream request sets the equivalent one-token greedy
parameters and seed 0. Record rather than infer every resolved value from each
runtime. A mismatch makes the comparison invalid.

The oracle environment is acceptable only after it:

1. reports the exact source pin and successfully imports the extension built
   from that pin;
2. loads the three hashed artifact files above;
3. proves the production chain selects `_rocm_C.wvSplitK` with the resolved
   gfx1x host configuration; and
4. produces repeatable logits for the exact one-token question.

The full-model oracle run adjudicates the token. It may not present
byte-identical intermediate activations because the two engines have other
implementation differences. For the kernel-body question, replay the first
local captured BF16 activation and weight pair directly through the pinned
`_rocm_C.wvSplitK` operation after the full-model trace has proven that the same
shape and host configuration are production-reachable. Use the recorded CU
count, preserve strides and orientation, and compare raw BF16 output bytes with
the local skinny capture. Label this replay as kernel evidence, never as a
substitute for the full production oracle.

Run pinned vLLM with its production configuration and the skinny default
resolved true. A skinny-disabled upstream arm may be collected as a diagnostic
control, but it does not replace the production oracle. `--enforce-eager` may
be used only as a labelled correctness cross-check needed to compare the old
capture; it is forbidden as the performance denominator.

For tokens 369 and 488, record logits, ranks, argmax, and log-probability gaps
from the same output tensor. Apply only the sacred gate's already-ratified
500-milli-nat acceptance band. The old table's token-10 entry describes this
shared prefix and may be reported as historical context. Entries for tokens 11
through 15 follow a divergent token and must not be used. If a new local
sequence is justified, teacher-force every later position from that new exact
prefix and derive every suffix gap anew.

Source inspection alone is not an oracle result. Until the runnable steps pass,
the same-prefix answer is **PENDING**.

## Pre-registered decisions

Choose exactly one branch after recording all required evidence. Do not choose
a branch from the current BLAS/local-anchor agreement.

### A. Local port differs from executing pinned upstream

If paired identical operands show any raw-BF16 output difference between local
wvSplitK and the applicable executing pinned body, or the shape/configuration
census shows a host-policy mismatch on the production arm, repair the local
mirror. The upstream tolerance remains the general numerical-test contract; it
does not excuse drift on this same-board, same-input kernel-body comparison.
Port the minimum missing arithmetic, host selection, stride, or device behavior
that the measured arm needs. Keep the committed goldens unchanged. The repaired
default must pass the per-operation ported suite, exact-prefix oracle
comparison, full sacred gate, full repository gate, and fresh mutation review.

### B. Local matches upstream and the pinned oracle supports the skinny token

If identical operands produce byte-identical raw-BF16 local/pinned skinny
outputs under the same configuration, and pinned vLLM's exact-prefix logits
support token 369 under the ratified band, the old local anchor is stale. Only
then capture the new default 16-by-16 local sequence, verify repeatability,
replace the local anchor, and teacher-force pinned vLLM on every exact new
prefix to rederive the complete near-tie table. Retain the old files and
failure as historical evidence in the spec or review record rather than using
any old suffix gap. The new checkpoint gate must still causally assert
wvSplitK selection.

### C. The upstream skinny arm or a faithful local production path is invalid

This branch is available only if the pinned upstream skinny arm fails its
applicable upstream tests, or the local production seam cannot reproduce the
executing upstream path without an unsafe or out-of-scope change. Narrow or
disable the default only after the existing goldens pass, the reason is tied to
specific shapes/configurations, and a same-binary same-workload measurement
quantifies throughput, latency, and memory cost. Record why each new threshold
or default has its value and which issue owns restoration. A broad disable is
not justified when a measured shape guard is sufficient.

If the evidence supports none or more than one branch, return
`NEEDS_DECISION`; do not average the conclusions.

## Upstream tests to port

The source is pinned
`tests/kernels/quantization/test_rocm_skinny_gemms.py::test_rocm_wvsplitk_kernel`
and the routing companion is
`tests/model_executor/layers/test_rocm_unquantized_gemm.py::test_rocm_unquantized_gemm_gfx1x_wvsplitk_path`.
Preserve upstream seed 0, both xavier-normalization modes, elementwise
`atol = finfo(dtype).eps * sqrt(K)`, `rtol = 1e-2`, and direct comparison with
linear reference output. Do not substitute aggregate NMSE.

The upstream `(input rows, K, output features)` factors are preserved in the
spec, not summarized away:

```text
(1,32,16)       (1,64,64)       (2,256,256)      (3,1024,1024)
(4,4096,4096)   (4,4096,4097)   (4,4112,4096)    (4,4112,4097)
(1,9216,512)    (2,10240,1024)  (4,16384,8192)   (4,32768,8192)
(4,32768,8193)  (4,32784,8192)  (4,32784,8193)   (1,64,8)
(2,128,8)       (4,256,8)
```

For each upstream factor, its cross product is seed 0, xnorm false/true, BF16
and FP16, bias modes none/vector/matrix, and padded activation and weight
false/true. The table below classifies every dimension of that cross product.
The direct local BF16/no-bias/packed wvSplitK subset is exactly
`(1,32,16)`, `(1,64,64)`, `(2,256,256)`, `(3,1024,1024)`, `(4,4096,4096)`,
`(4,4112,4096)`, `(1,9216,512)`, and `(2,10240,1024)`. The implementation
ports all eight rather than retaining the current seven-shape subset.

The applicability inventory is explicit:

| Upstream surface | Disposition in #2772 |
|---|---|
| BF16, no bias, packed activation and weight, rows 1-4, gfx1x small-LDS shapes | Port every applicable N/K/M parameter and both xnorm modes; require selected wvSplitK plus numerical output and output-sentinel checks. |
| More than five input rows and non-skinny output width | Port routing fallback cases; wvSplitK must not be called. |
| Row count 5 | Port as an explicit local refusal/fallback case unless branch A proves the production arm needs the upstream template. |
| `K % 8 != 0`, output features at or below 8, odd output features, non-packed activation, and LDS overflow | Preserve the local adaptation as safe BLAS fallback; assert no wvSplitK call and correct output. |
| FP16 or mismatched input dtypes | Not applicable to the measured BF16 Qwen3.5 arm; preserve the public local rejection and do not claim FP16 is ported. |
| bias modes 1/2 | Not applicable because `vt::MatmulBT` has no bias operand; do not silently drop the parameter. |
| padded weight/activation modes | Padded activation is a fallback case; padded weight is rejected by the shared `MatmulBT` contract. Preserve those local semantics explicitly. |
| gfx9, gfx950, gfx1151, medium-LDS, and big-LDS bodies | Not executed by the scoped gfx1100 checkpoint path; keep them refused or on BLAS and name them as non-ported. |

The current local test case in `tests/vt/test_backend_cross_device.cpp` is the
starting seam, not sufficient evidence by itself. Its direct construction
proves kernel behavior. The checkpoint test supplies production reachability.

## Checkpoint-backed causal gate

Keep `tests/parity/test_qwen35_paged_engine.cpp::RunGate` as one actual doctest
case under its ordinary default environment. The CTest registration must not
set `VT_ROCM_SKINNY=0`, and no rollback-arm checkpoint test substitutes for the
default case. The case must:

- load the pinned directory through `LoadedEngine::FromModelDir`;
- run the existing 16 prompts and native-provider checks;
- assert the internal-arm trace saw wvSplitK on the measured production shapes;
- apply the unchanged anchor/gap policy, with golden changes allowed only by
  branch B; and
- fail, not skip, for token drift when prerequisites are present.

The production chain under review is:

```text
LoadedEngine::FromModelDir
  -> ModelRegistry::Forward
  -> Qwen3.5 model projection
  -> vt::MatmulBT
  -> registered ROCm MatmulBTKernelRocm
  -> WvSplitKBT
  -> wvSplitKSml
```

The fresh reviewer deletes or bypasses the `WvSplitKBT` call in a scratch copy
and reruns the focused checkpoint gate. It must fail because the wvSplitK
selection assertion disappears even if BLAS happens to reproduce the committed
tokens. A gate that stays green is not causal and blocks landing.

## Gates and evidence

### Red-first implementation evidence

Before changing product behavior, the implementer records:

1. the current default one-case failure at prompt 10/token 10;
2. the new causal assertion failing when the production wvSplitK call is
   bypassed;
3. the applicable upstream test port failing for the intended numerical or
   dispatch reason selected by the evidence; and
4. the exact-prefix skinny/BLAS manifests and logits, labelled diagnostic.

If branch B needs no product arithmetic repair, the existing default sacred
failure is the red test; do not manufacture a kernel failure.

### Focused green

Every GPU command uses the same lock and sanitized library path. The focused
commands are:

```sh
env LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib \
  flock /home/vikash/gpu.lock \
  cmake --build build-hip --target test_backend_cross_device \
  test_qwen35_paged_engine -j 4
env LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib \
  flock /home/vikash/gpu.lock \
  build-hip/tests/test_backend_cross_device \
  '--test-case=decode-skinny MatmulBT (wvSplitK path) matches the CPU oracle'
env LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib \
  flock /home/vikash/gpu.lock \
  build-hip/tests/test_backend_cross_device \
  '--test-case=ROCm MatmulBT observer reports skinny selection guards'
env LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib \
  flock /home/vikash/gpu.lock build-hip/tests/test_qwen35_paged_engine
```

Build only with bounded parallelism. The focused gate includes the wvSplitK
cross-device suite, the internal routing observer test, and the one-case sacred
executable. Record commands and exact assertion counts. A missing model or
oracle is `PENDING`, never green.

### Full gate

After focused green, run the row's declared full gate under the same sanitized
runtime and lock. Run `scripts/agent-preflight.sh` separately for record and
prose checks, followed by:

```sh
python3 scripts/check-symbol-anchors.py \
  --upstream-root /home/vikash/oracle/vllm-src
```

The operator, not the implementer, reruns the row gate for acceptance.

### Performance if branch C is selected

Correctness must already be established. Measure skinny and proposed
narrow/disabled arms with one binary, the exact hashed 0.8B artifact, identical
fixed prefixes, one-token teacher-forced decode, batch/concurrency one, and
identical cache/sampling configuration. Because `SkinnyGemmEnabled` caches its
environment, use fresh processes and interleave A/B/B/A repetitions on an idle
board. Record tokens/s, ms/token, first-token latency if applicable, peak
device memory, per-shape invocation counts, and identical-tool ROCm traces.
Never compare autoregressive suffixes after the first divergence. Never use
`--enforce-eager` as the pinned vLLM denominator, and never substitute the 4B
provisional trace.

### Fresh review mutations

A fresh reviewer reviews an immutable head, mutates one guarantee at a time,
and restores byte-for-byte after each run:

- bypass the production `WvSplitKBT` call;
- make one measured eligible shape select BLAS while leaving provider
  statistics unchanged;
- remove one shape/configuration guard;
- corrupt one BF16 multiply or one wave32 reduction step;
- relax the upstream elementwise tolerance or delete one applicable parameter;
- if branch B changes goldens, restore the old token-10 anchor or substitute an
  old suffix gap; and
- if branch C narrows the default, widen the guard back over the proven-bad
  shape.

Each mutation must fail the focused gate for the intended reason. Restore and
verify SHA-256 values of every mutated file before the next mutation. Review
returns `PASS` only after focused and full green on the restored immutable head.

## Risks and controls

- **BLAS is not the oracle.** The rollback's 137/137 result isolates the
  regression but cannot canonize token 488. The pinned execution decides.
- **Suffix evidence is poisoned after divergence.** Only the first disputed
  step shares an input prefix. All later gaps are rederived when needed.
- **Static environment caching can mix arms.** Skinny and BLAS diagnostics use
  fresh processes, with the resolved arm recorded in each manifest.
- **Instrumentation can change timing or execution.** Detailed dumps are
  correctness-only, run the actual operation once, and are removed before
  review. Capture-off/on tokens prove non-perturbation.
- **A direct kernel test can land dead code.** The checkpoint case asserts the
  branch reached through the full production loader and registry path.
- **The apparent local/upstream body match can hide host drift.** Exact shapes,
  strides, CU count, LDS branch, YTILE, UNRL, and template row count are
  compared before arithmetic.
- **An unavailable source-built extension can masquerade as an oracle.** Import,
  revision, artifact hashes, selected custom op, and repeated output are all
  required.
- **A broad rollback can erase measured performance.** Branch C requires a
  shape-minimal guard and same-workload performance evidence.

## Stop conditions

Stop and return the named state when:

- the runnable pinned oracle cannot be reconstructed: leave oracle-dependent
  gates `PENDING`, record the exact build/runtime blocker, and do not edit
  goldens;
- artifact name, byte size, hash, source pin, or extension identity differs:
  `NEEDS_CONTEXT`;
- skinny and BLAS captures do not share identical operands at the disputed
  step: `NEEDS_CONTEXT`;
- evidence does not select exactly one pre-registered branch:
  `NEEDS_DECISION`;
- a repair would require an out-of-scope upstream arm or change issue #2773's
  state/cache contract: `NEEDS_DECISION`;
- an applicable upstream parameter or failure case cannot be represented at
  the local seam without changing its public contract: `NEEDS_DECISION`;
- the default sacred gate, the causal reachability mutation, or the row's full
  gate remains red after the selected branch: implementation remains
  incomplete; or
- GPU work lacks the required lock, sanitized runtime, or idle-board evidence:
  discard that run rather than reporting it.

## Owed

- Issue [#2773](https://github.com/mudler/vllm.cpp/issues/2773), Row
  `BACKEND-ROCM`, owns later CPU/ROCm cache, state-space, convolution,
  attention, and layer-numeric characterization. It remains blocked on #2772
  and is not absorbed here.
- Issue [#487](https://github.com/mudler/vllm.cpp/issues/487), Row
  `BACKEND-ROCM`, retains the broader ROCm skinny-GEMM performance surface and
  non-applicable architecture/mode work. This issue changes only what measured
  Qwen3.5 correctness requires.

## Git integration

Use one pull request for this committed spec and its later implementation. The
pull request body links issue #2772, carries `Row: BACKEND-ROCM`, and closes the
issue only when the selected branch, focused/full gates, fresh review, and
operator verification all pass. This spec commit precedes every product or test
implementation commit.
