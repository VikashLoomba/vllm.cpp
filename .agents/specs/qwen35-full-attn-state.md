# Qwen3.5 full-attention state validation

Row: `ENG-QWEN35-FULL-ATTN-STATE`.

Issue: [#3098](https://github.com/mudler/vllm.cpp/issues/3098).

Branch: `row/ENG-QWEN35-FULL-ATTN-STATE`.

Spec base: `6db4bef906859e864c82523c01107473f7dcca29`.

## Now

`PENDING`. This specification precedes implementation. The row uses one pull
request under the repository default. A fresh implementer works from this
committed specification. A fresh reviewer mutates the immutable implementation,
and the operator reruns its gates. The operator publishes the pull request.
The current task does not authorize merging it.

Public completion fails before any quantized gather provider runs. Runtime
correctness, oracle execution, and implementation review remain `PENDING`.
This row makes no performance claim and establishes no performance floor.

## Scope

Allow Qwen3.5 models containing only full-attention layers to prefill and decode
without recurrent state. Select Gated DeltaNet (GDN) validation and preparation
from the actual layer consumers. Preserve strict validation for every model
that contains a GDN layer, including missing caches and malformed metadata.

Cover the dense and mixture-of-experts (MoE) siblings in
`src/vllm/model_executor/models/qwen3_5.cpp`. Cover eager forward, graph entry,
and persistent device inputs because those paths share the same assumption.
Reuse the existing full-attention input builder.

The planner still publishes its legacy empty `gdn` group. This is an existing
local adaptation, not a cache-topology parity claim. Changing group topology,
registry group indices, or runner allocation belongs outside this fix.

Other exclusions are quantized providers, F16 matrix multiplication, fused MoE
kernels, new graph backends, new speculative modes, checkpoint downloads,
package installation, continuous integration changes, and global oracle pins.
The fix has no dependency on the separate gfx1100 provider changes.

## Inventory

This per-row inventory owns the child identity through the canonical spec scan.
No parent matrix or roadmap lifecycle changes. Update this table and `Now`
together when this child changes state.

| ID | Upstream source | Local anchor | Tests and evidence | Spec | State | Owner | Issue |
|---|---|---|---|---|---|---|---|
| `ENG-QWEN35-FULL-ATTN-STATE` | Pinned `Qwen3_5DecoderLayer` and `GPUModelRunner.get_kv_cache_spec` | `CheckDensePagedForward`, `CheckPagedForward`, `BuildFullAttnStepDevInputs`, both graph `Step` methods | G0 to G5 | This file | `PENDING` | Row helper, fresh reviewer, operator verification | #3098 |

## Diagnosis and source anchors

At the spec base, the complete failing chain is:

1. `src/vllm/model_executor/models/qwen3_5_common.cpp::MakeQwen3_5KVCacheSpec`,
   lines 75 to 99, always publishes a `gdn` group.
2. `src/vllm/v1/worker/gpu/runner.cpp::GPUModelRunner::initialize_kv_cache`
   allocates recurrent buffers only for `linear_attention` layers at lines
   1511 to 1513. This model has none.
3. `src/vllm/v1/worker/gpu/runner.cpp::GPUModelRunner::execute_model`, lines
   2779 to 2805, builds and remaps GDN metadata when that group exists.
4. `src/vllm/model_executor/models/qwen3_5.cpp::CheckDensePagedForward`, lines
   9240 to 9249, accepts zero GDN layers and zero caches. It then validates
   live slot 0 against zero slots and throws at `ValidateGdnStateIndices`.
5. `BuildStepDevInputs`, lines 4659 to 4682, independently validates and
   prepares the same unused metadata. Repairing the first check alone fails.

The sibling `CheckPagedForward` repeats this assumption at lines 8526 to 8536.
`ForwardLayers` and `DenseForwardLayers` call the GDN builder at lines 8380 and
9323. Both graph `Step` methods validate GDN state before their eager fallback,
at lines 10763 and 11391. Persistent inputs repeat the builder at lines 11036
and 11774. The graph staging functions already test the GDN presence flags.

`git log -S'ValidateGdnAttentionMetadata'` identifies `f344decf4` and
`3ae5cfe06`. The original [packed-decode specification](gdn-packed-decode.md)
records why shared validation protects real consumers. Preserve that guarantee.

The existing `BuildFullAttnStepDevInputs`, line 7845, uploads full-attention
metadata and leaves GDN fields inert. Its current consumer is the multi-token prediction (MTP) draft
head. Reusing it avoids a second full-attention preparation implementation.

## Pinned reference and adaptations

The [primary pin](../upstream-sync.md) is
`e126687a9a828d513c01a07cd69f025f27d63280`, runtime
`0.28.1rc1.dev132+ge126687a9`. Read these executing upstream anchors:

- `vllm/model_executor/models/qwen3_5.py::Qwen3_5DecoderLayer.__init__`,
  lines 144 to 160, constructs GDN only for `linear_attention`.
- `vllm/model_executor/models/qwen3_5.py::Qwen3_5Model.__init__`, lines 249
  to 259, selects every decoder from `config.layer_types`.
- `vllm/v1/worker/gpu/model_runner.py::GPUModelRunner.get_kv_cache_spec`,
  lines 530 to 531, delegates to
  `vllm/v1/worker/gpu/attn_utils.py::get_kv_cache_spec`, lines 52 to 65.
  This V2 runner chain enumerates modules that actually require caches.
- `vllm/model_executor/layers/mamba/abstract.py::MambaBase.get_kv_cache_spec`,
  lines 67 to 87, obtains recurrent shapes from a recurrent consumer.

The exact no-GDN completion case is a local regression. The inspected upstream
`tests/models/language/generation/test_common.py`, `test_hybrid.py`,
`tests/models/test_qwen3_5_mtp_config.py`, and
`tests/v1/worker/test_gpu_model_runner.py` do not contain that case.
Preserve the existing ported GDN cases and their upstream anchors. Do not turn
unrelated KV-sharing allocation tests into a new topology obligation.

The operator's pinned runtime reports `Using V2 Model Runner`. The V2 chain
is the primary execution anchor for qualification.

Run the primary oracle on the identical generated dense GGUF before accepting
the fix. The [GGUF plugin pin](../oracles/vllm-gguf-plugin.md) is
`d4c1f0d082fc7cd4350da56689109a01c1f29d6c`. Its
`vllm_gguf_plugin/config_parser.py::GGUFConfigParser.parse` reads a local
configuration beside the artifact. Supplying that configuration and explicit
token IDs is an allowed harness adaptation. Preserve all model tensors and
geometry. A successful import or config construction is not an oracle result.

Use the existing pinned runtime, plugin artifact, and dependencies supplied by
the operator. Record their source revisions and binary hashes. Missing runtime
registration or unsupported full-attention execution leaves the oracle gate
`PENDING`. Do not install a package, substitute safetensors, change the model,
or use another oracle without an explicit scope decision. Do not force eager
execution on the oracle. Record its resolved production configuration.

The operator observed a missing `mm_proj` refusal on the sibling F16 oracle
workload. The pinned plugin's
`vllm_gguf_plugin/weights_adapter/qwen3_5.py::QWEN35_ARCHITECTURES`, line 40,
maps `qwen3_5_text` to conditional generation. Its `Qwen35GGUFAdapter.patch_hf_config`
replaces the architecture at line 225. Upstream
`vllm/model_executor/models/qwen3_5.py::Qwen3_5ForConditionalGeneration.__init__`,
lines 503 to 517, dereferences the vision configuration and constructs a tower.
A text configuration alone does not establish a runnable text-only GGUF path.
This known risk leaves G4 pending on an oracle refusal.

## Design

1. Derive the presence of GDN consumers from `weights.layers` and
   `is_linear_attention`. Reuse one internal predicate where practical.
2. Keep layer-count, attention-cache-count, GDN-cache-count, and cache-layout
   checks unconditional. A hybrid model with no caches must still fail.
3. Run GDN token-count, metadata, and graph-state validation only when a GDN
   consumer exists. Keep the validators themselves strict, including direct
   `BuildGdnStepInputs` consumers outside this model.
4. Select `BuildFullAttnStepDevInputs` for a model with no GDN consumers.
   Apply the selection to both eager builders and both persistent graph builders.
   Keep GDN upload and staging flags false on that path.
5. Pass inert empty GDN metadata into graph padding when no consumer exists.
   `BuildPaddedDecode` must not copy unvalidated, arbitrary-length GDN indices
   into its S-entry storage. Retain the bounds enforced for actual consumers.
6. Keep generic attention and graph shape checks effective for both model
   topologies. Preserve real-GDN decode, prefill, mixed, speculative, and padded
   state validation. Do not use GDN metadata as the only generic shape check.

Never infer the absence of consumers from `state_slots == 0` or an empty cache
vector. Never add dummy recurrent state, a fake layer, a validation bypass flag,
or a fixture geometry change. Keep model activations and key-value (KV) storage at their
resolved bf16 dtype. Retain existing annotated f32 exceptions and integer
metadata layouts. This change adds no wider model buffers or recurrent tensors.

## Regression fixture

Use only the dense control from the frozen gather experiment. Extract its
minimal deterministic generator into this row's own fixture file. Preserve the
bytes of `Q4_0-dense.gguf`, even though its embedding and projections are dense.
Do not copy the 19-format provider campaign into this regression.

The artifact has 991296 bytes and SHA256
`0e6554ba521edfde00d4d025a3058eabaaacdce6b24344f959e83d8dde35df7f`.
Its geometry is hidden width 256, intermediate width 256, vocabulary 128,
four query heads, one KV head, head width 64, one full-attention layer, and
`full_attention_interval=1`. Rotary width is 64, base is 1000000, and MRoPE
sections are `[16,8,8,0]`. Norm epsilon is `1e-6`, seed is `0x524f434d`.
Inactive state-space model (SSM) loader keys retain convolution width 4, inner width 64, state width
64, time-step rank 1, and group count 1.

Prompts are `[1,0,63,127,63]` and `[1,127,0,127]`. Request exactly four greedy
tokens with `ignore_eos=1` and seed `0x524f434d`. Use block size 16, four blocks,
maximum length 64, and one sequence. Public device 0 selects AUTO in the HIP-only
build. Assert the actual ROCm dense embedding provider executes.

For both local and oracle runs, execute each prompt three times with a fresh
engine and fresh model state. Keep batch size and concurrency at one. Submit
prompts sequentially and record the actual scheduling and token budgets.

## Gates and evidence

| Gate | Required result | Current result |
|---|---|---|
| G0: independent red | New public regression on pristine product base fails at GDN validation after load | `PENDING`: operator reproduction exists, row-owned test is owed |
| G1: public completion | Both prompts produce four tokens in three fresh-engine repeats, native ROCm provider executes | `PENDING`: operator GPU run |
| G2: consumer distinction | Dense and MoE production routes accept no-GDN state and reject damaged real-GDN state | `PENDING`: fresh implementer |
| G3: graph preparation | Both drivers cover cold, capture, persistent staging, and replay routing with no GDN consumers | `PENDING`: fresh implementer and operator |
| G4: pinned oracle | Same dense artifact and all six fresh-engine runs, exact token IDs, finite logits, preserved dtypes | `PENDING`: operator runtime execution |
| G5: review and full gate | Focused gates, mutations, full preflight, fresh review, and operator rerun | `PENDING`: implementation |

G0 starts at `vllm_engine_load` and `vllm_complete_tokens` from `include/vllm.h`.
The test must assert successful load before its expected pre-fix completion
failure. Reproduce with the row's fresh build from unchanged base product code.
The preserved gather red establishes the diagnosis, not this independent gate.

For G2, enter `ModelRegistry::Forward` for both dense and MoE variants. Cover
prefill and decode without GDN consumers, with default-empty and unused
runner-style GDN metadata. For real GDN consumers, reject missing all caches,
wrong cache count, invalid ranks, inconsistent slots, missing metadata,
duplicate indices, and out-of-range indices. Keep valid hybrid completion green.
A validator-only unit test does not prove these production call sites execute.

For G3, extend the existing graph harness for both drivers. Exercise their
production `Step` methods, graph fallback, persistent inputs, and staging flags.
Include no-GDN metadata with an oversized unused index vector. The padding path
must ignore that vector safely. Mutate the inert-metadata selection and require
this regression to fail. Use a memory-sanitized run if needed to expose an
out-of-bounds copy reliably.
Trace the registry dispatch to each driver and mutate each changed call site.
A CPU fake replay proves routing only. Record that limitation and obtain an
operator GPU replay result for any claimed replay numerics. Do not enable a
new backend or claim a GPU graph executed from the CPU harness.

Register a dedicated public test such as `test_capi_qwen35_full_attn_state`.
Run it with `test_qwen27_paged_forward`, `test_qwen35_paged_forward`,
`test_qwen3_5_decode_graph_seam`, and `test_model_registry`. Retain existing
GDN state, speculative metadata, and graph-padding cases in those suites.
Build in a row-owned ignored `build-*` directory with at most `-j 4`.
The implementer records exact configure, build, and focused test commands
before handing GPU commands to the operator. No shared build directory is valid.

For every result, record the immutable source SHA, binary SHA256, fixture hashes,
command, environment, exit status, and evidence path. G4 additionally records
source anchors, runtime and plugin identity, output tokens, resolved dtype,
attention backend, and graph mode. No config-only or local self-comparison
substitutes for G4. A token mismatch remains failing until explained and fixed.

Fresh review uses scratch copies and restores every modified file byte-for-byte.
Restore unconditional GDN validation at each newly conditional entry. Restore
the GDN builder at each full-attention selection. Delete the production call
sites into the changed preparation path. Each relevant positive gate must fail.
Then force the no-GDN branch for a hybrid model, or remove cache-count
validation, and prove the corresponding malformed-hybrid gate fails. Mutate duplicate, range, missing
metadata, and layout guarantees individually. Keep per-mutation failure logs.

Run `scripts/agent-preflight.sh` before edits and the staged form before commit.
Run exact-range record, commit-style, trailer, and PR-size checks after commit.
Classify omitted hardware and build gates explicitly. The operator repeats the
applicable gates on the immutable implementation before publication.

## Existing evidence

On 8 September 2026 PDT, the operator ran the frozen gather public binary on local
gfx1100 under `/home/vikash/gpu.lock`. Completion threw
`qwen3_5: GDN state index out of range`. The process returned 1 with five
passing assertions and one failed completion assertion.

The binary SHA256 is
`f6a109ef33a133b381a112313591534f560adedb0d4919f0d67567cd597552e1`.
Its source was spec commit `670e6d78ddf55231394748e0032939fd53dc56a5` plus
uncommitted test-only changes. The fixture header SHA256 is
`339414f93e59e6be9a4ca545d5e762bb2816714343f466e218dbc2b69483bdfd`.
The public test source SHA256 is
`add58ac23eef26588fb358d7679feeac961baee3a1da3d5ddd3adedaf9a1a825`.
The log is `/home/vikash/.cache/rdna3-gather-impl/evidence/public-red.log`, SHA256
`fbd5d684566961d71f31c8b9e58daba8d5303636adb12ceccb164dbe3b220e1d`.
These are supplied local evidence paths, not environment defaults.

## Risks and stop conditions

- `NEEDS_CONTEXT`: the frozen fixture bytes, pinned runtime, or source cannot
  be verified. The operator owns the missing artifact or runtime decision.
- `NEEDS_DECISION`: the oracle requires a different model artifact, new
  dependency, or behavior outside the stated scope.
- `BLOCKED`: required GPU execution is unavailable. Only the operator runs
  the GPU work under the recorded mutex or applicable fleet lease.
- `FAILING`: a no-GDN route still prepares device recurrent metadata, a real-GDN guard
  weakens, or a mutation remains green. Repair through a fresh implementer.
- `FAILING`: graph coverage relies on fake replay numerics or a different
  untested sibling. Keep the affected gate pending until valid evidence exists.
- `NEEDS_DECISION`: the minimal fix requires planner topology, unrelated
  quantization, broader speculative behavior, or any excluded file change.

## Owed

Issue #3098 owns implementation, independent red evidence, pinned oracle
execution, review mutations, and the operator rerun. Record the measured
outcome and defaults here before changing the row to `DONE`. Keep the issue
open until the work lands.
