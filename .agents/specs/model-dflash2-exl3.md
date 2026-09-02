# MODEL-DFLASH2-EXL3 — the EXL3 arm of the DFlash2 draft loader and forward

Row: `MODEL-DFLASH2-EXL3`
Issues: [#2495](https://github.com/mudler/vllm.cpp/issues/2495) (item 7)
Base SHA: `4fe3852b1` (`origin/main`)
Parent rows: [`MODEL-QWEN35-GDN-EXL3`](model-qwen35-gdn-exl3.md) →
[`MODEL-QWEN35-EXL3`](model-qwen35-exl3.md) →
[`QUANT-EXL3-MUL1`](quant-exl3-mul1.md) → [`QUANT-EXL3`](quant-exl3-shared.md);
the draft itself is [`SPEC-DFLASH2`](dflash2-spec-decode.md).

Upstream pin: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`. vLLM defines the
DRAFT (`vllm/model_executor/models/qwen3_dflash.py` and `qwen3_dflash2.py` @
vllm-project/vllm#52816 head `66e5414c6d75a8529473d977f7458c140bbab8a0`, both
already mirrored here) and the SEAM (`vllm/model_executor/layers/linear.py`,
`layers/quantization/base_config.py`). vLLM registers no EXL3 at the pin, so the
trellis FORMAT comes from the registered secondary oracle
[`exllamav3`](../oracles/exllamav3.md) @
`2398c05635fbbad01a0a51dce63c85c6c8a8450e` (MIT). Nothing here re-derives either
half.

## Now

`ACTIVE`. `Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw` loads through the
production draft loader and runs through all three DFlash draft forward bodies.

## The gap

`LoadQwen3DFlash` is a name-for-name BF16 reader. It dies on its first line of
weight work, `LoadBf16RawNK(get, "fc.weight")`
(`src/vllm/model_executor/models/qwen3_dflash_weights.cpp`), because an EXL3
module ships no `.weight` at all — it ships `fc.{trellis,suh,svh,mul1}`. Every
later refusal in that function is therefore MASKED today, and two of them are
real (see "F16, by name" below).

Nothing else is missing. `(bits 5, codebook 2)` is an instantiated CUDA arm on
`origin/main` and `src/vt/cuda/cuda_exl3.cu` names this exact artifact in its own
message; `mul1` landed with `QUANT-EXL3-MUL1`; `DFlash2DraftModel` is already a
recognised architecture (`include/vllm/config/speculative.h`); and the draft is
selected by the `--speculative-config` method string rather than by the
architecture registry, so no registration changes. What is absent is a consumer.

### What the artifact actually ships

Read from the DOWNLOADED file's own safetensors header, not from an index and
not from the model card: `/mnt/nas_share/models/Qwen3.8-27B-EXL3/draft-dflash2-5.0bpw/model.safetensors`,
1 470 916 078 B, repo `Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw` @
`4f0436269bca761b071f05319e8e04a87cc633f9`. 189 tensors: 36 EXL3 modules × 4
tensors, plus 45 dense ones.

| module (×5 layers, plus one global `fc`) | `trellis` `I16` | K → N |
|---|---|---|
| `fc` | `[1600, 320, 80]` | 25600 → 5120 |
| `layers.N.self_attn.q_proj` | `[320, 256, 80]` | 5120 → 4096 |
| `layers.N.self_attn.k_proj` | `[320, 64, 80]` | 5120 → 1024 |
| `layers.N.self_attn.v_proj` | `[320, 64, 80]` | 5120 → 1024 |
| `layers.N.self_attn.o_proj` | `[256, 320, 80]` | 4096 → 5120 |
| `layers.N.mlp.gate_proj` | `[320, 1088, 80]` | 5120 → 17408 |
| `layers.N.mlp.up_proj` | `[320, 1088, 80]` | 5120 → 17408 |
| `layers.N.mlp.down_proj` | `[1088, 320, 80]` | 17408 → 5120 |

Each carries `suh` `F16 [K]`, `svh` `F16 [N]` and a scalar `mul1` `I32 []`.

**The bits are UNIFORM and the selector is NOT quantized.** The last trellis
dimension is 80 on every one of the 36, and 80 = 16 × 5, so every module is bits
5. `quantization_config.json`'s own `tensor_storage` agrees: 36 entries carry
`{"quant_format": "exl3", "bits_per_weight": 5, "mul1_multiplier": 2212286765}`
and nothing else does. 2212286765 is `0x83DCD12D`, the constant
`exl3_lib/quantize.py:1421-1424` writes, which the tree already knows.

The model card says "5.0 bpw (module-adaptive, includes the candidate selector)"
and is WRONG on both counts. `candidate_selector.*` appears nowhere in
`tensor_storage` and ships as three dense tensors. This spec encodes the header,
never the card.

`fc`'s K = 25600 = 5 × 5120 confirms the five taps, and
`dflash_config.target_layer_ids` is `[5, 19, 33, 47, 61]` — byte-identical to
`dflash2-spec-decode.md` and to `tests/vllm/v1/spec_decode/test_dflash_causality.cpp`.
Nothing about the taps moves.

### F16, by name and only by name

The repack converted the two dense LINEAR modules it left unquantized from BF16
to F16, and left every norm, codebook and `base_kernel` at BF16:

| tensor | dtype | shape |
|---|---|---|
| `candidate_selector.hidden_projection.weight` | **`F16`** | `[256, 5120]` |
| `layers.N.attention_conv.kernel_projection.weight` | **`F16`** | `[1280, 5120]` |
| `layers.N.mlp_conv.kernel_projection.weight` | **`F16`** | `[1280, 5120]` |
| `layers.N.{attention,mlp}_conv.base_kernel` | `BF16` | `[2, 2, 5120]` |
| `candidate_selector.{predecessor,successor}_codebook` | `BF16` | `[248320, 256]` |
| every `*norm.weight` | `BF16` | |

`LoadBf16Direct` refuses a non-BF16 dtype by name, so these are a SECOND
blocker on tensors that are not quantized at all. They are invisible today only
because the load dies at `fc.weight` first.

## Scope

**In.** The eight EXL3 projections of a DFlash2 draft (`fc` plus seven per
layer); the loader rung; the forward routing in all three inline layer bodies
and in the context-KV precompute; F16 acceptance for the two named dense
LINEARs, scoped to an EXL3 load; the gate.

**Out, and deliberately.** `LoadDflashSharedLmHead`, `SharedHeadSource::LoadInto`
and the Qwen3.5 `lm_head`/`mtp` head (#2495 items 5 and 6, owned by another row
and live in those files at the time of writing). The NVFP4 KV cache. The 27B
target's own loading. Every performance question (#2495 items 8-10). A merged
EXL3 QKV or gate_up operand. The DSpark drafter and the GGUF draft, neither of
which can ship a trellis.

## Design

1. **Eight `Exl3Weight` fields, and ONE predicate.**
   `Qwen3DFlashLayerWeights` gains `q_proj_exl3`, `k_proj_exl3`, `v_proj_exl3`,
   `o_proj_exl3`, `gate_proj_exl3`, `up_proj_exl3`, `down_proj_exl3`;
   `Qwen3DFlashWeights` gains `fc_exl3` and `bool IsExl3() const { return
   !fc_exl3.Empty(); }`.

   The discriminator is `fc_exl3` ALONE, and every forward site reads that ONE
   predicate — the shape `Qwen3DFlashWeights::IsDflash2()` already has in the
   same bodies. A per-field probe would let a half-populated container read as
   an arm on one projection and as bf16 on the next, which is silent and
   token-invisible on a lossless verify. The loader closes the other half of
   that door: when the rung is taken it asserts all eight are populated, so
   `IsExl3()` true implies every site has a trellis to read.

2. **The loader rung is on `fc` and is exclusive.** `LoadQwen3DFlash` takes a new
   trailing `has` predicate; the rung is
   `dense_loaders::IsExl3Projection(has, "fc")`, which is upstream's own
   `Linear.is_exl3_storage` three-tensor test (`modules/linear.py:385-389`)
   rather than a trellis-only probe. The `get`-only overload defaults `has` to
   an empty function, which answers "no EXL3 module here" — that is what keeps
   the GGUF draft and the DSpark backbone byte-unchanged, since neither format
   can carry a trellis. The shards overload builds `has` with the SAME
   `model.`-prefixed fallback its resolver uses, so the two cannot disagree
   about whether a tensor exists.

3. **`ConcatRawNK` IS NOT WIDENED.** A trellis is `[k/16, n/16, 32·bits]`, so
   joining on the output dimension INTERLEAVES per input tile — a real transform,
   not a row stack, and one that is only valid when no `had_r_128` block
   straddles two matrices. `quantization/exl3.h` records this and defers it to a
   wave of its own. So on the EXL3 arm `qkv` is three `Exl3Weight`s and three
   GEMMs, and `gate_up` is the two `Exl3MlpGateUpMethod` already issues. This is
   the artifact's shape rather than a shortcut: it ships six independently fitted
   trellises with six independently fitted `svh` vectors, and no merged operand
   exists anywhere in the file.

4. **Geometry is asserted against the config, not assumed.** On the EXL3 arm the
   loader checks `q/k/v_proj.InFeatures() == H`, their `OutFeatures()` against
   `Hq·Dh` and `Hkv·Dh`, `o_proj` `Hq·Dh → H`, `gate/up` `H → I`, `down`
   `I → H`, and `fc` `H·num_taps → H`. The bf16 arm gets its equivalent for
   free from `ConcatRawNK`'s K agreement and the existing `fc` shape check; the
   EXL3 arm has no such incidental check, and a transposed trellis loads, runs
   and answers wrongly without one.

5. **The forward routes through the shared seams and writes no second matmul.**
   Four projections × three inline layer bodies plus the context-KV precompute:

   - `o_proj` and `down_proj` become ONE call site each,
     `layers::MakeLinearMethod(bf16, exl3)->Apply(...)`. On a bf16 draft that
     factory binds `UnquantizedLinearMethod`, whose `Apply` is exactly the
     `ResidentWeight` + `DBuf` + `vt::MatmulBT` sequence it replaces — byte-for-byte
     unchanged, and now one site rather than two.
   - `gate_up` moves from the hard-bound `layers::UnquantizedMlpGateUpMethod` to
     `layers::MakeMlpGateUpMethod(bf16, gate, up, I)`. Same three-line shape, and
     `Exl3MlpGateUpMethod` is already the two-GEMM arm.
   - `qkv` keeps its bf16 arms untouched, merged and sliced both, and gains a
     THIRD arm ahead of them for EXL3: three `MakeLinearMethod` calls into the
     same `q`/`k`/`v` buffers. `MergedQkvEnabled()` cannot apply to a format with
     no merged operand.
   - `CombineAuxFeaturesDevice`'s `fc` GEMM becomes one `MakeLinearMethod` call
     site, by the same argument as `o_proj`.
   - `PrecomputeContextKVDeviceBf16` projects k and v out of the merged owner by
     slicing it; on the EXL3 arm it calls `k_proj_exl3` and `v_proj_exl3`
     directly.

   `qwen3_dflash.cpp` already includes `layers/linear.h` and
   `models/dense_attn_block.h` and says `using namespace dense_attn`, so it can
   include `layers/quantization/exl3.h` directly. The `dense_exl3_linear.cpp`
   translation-unit boundary exists for `qwen3_5.cpp`'s ADL collision, which this
   file does not have.

6. **F16 is admitted by NAME, through the function that already argues for it.**
   `dense_loaders::LoadF16AsBf16Direct` converts (never reinterprets: the two
   formats share a width and nothing else) and its own declaration argues why the
   acceptance is scoped to an EXL3 load. A scheme-aware sibling of
   `LoadBf16RawNK` in the DFlash loader calls it for exactly
   `candidate_selector.hidden_projection.weight` and
   `layers.N.{attention,mlp}_conv.kernel_projection.weight`, and only when the
   EXL3 rung was taken. Everything else keeps the BF16-only refusal. This is the
   polarity `MODEL-QWEN35-GDN-EXL3` set for `in_proj_a`/`in_proj_b`; widening
   `LoadBf16Direct` outright would change what every other draft accepts, through
   a conversion that drops three mantissa bits.

## Tests

`tests/vllm/models/test_qwen3_dflash2_exl3.cpp`, a new suite, entering through
the PRODUCTION loader over a REAL on-disk safetensors file — never by filling
`Qwen3DFlashWeights` by hand.

1. **The refusal is real before the arm.** A BF16-only reader must refuse this
   artifact's F16 selector and F16 conv projections. Asserted directly, so the
   masked second blocker of "The gap" is executable rather than narrated.
2. **The arm loads.** All eight projections populate, at bits 5 and codebook 2,
   with the geometry the config declares.
3. **The arm computes, and matches its decoded twin.** The same synthetic draft
   built twice — once EXL3, once as the bf16 weights the trellis decodes to —
   must produce block logits that agree to a stated `rel_rms`.
4. **The engine generates with it**, in
   `tests/vllm/v1/spec_decode/test_dflash2_exl3_reach.cpp`. An EXL3 draft loaded
   by the production loader over a real on-disk file is handed to `LoadedEngine`
   through the in-memory seam `test_dflash2_runner_reach.cpp` already uses, and
   `Generate` runs. That reaches the two HOT layer bodies, which the block
   forward above does not, and it is the mutation target for the qkv and MLP
   call sites in them.

   **What this leg does NOT prove, stated rather than rounded up.**
   `LoadedEngine::FromModelDir` loads the draft only after the TARGET's shards
   are open (`maybe_load_dflash`), so a `--speculative-config` case needs a
   complete on-disk target checkpoint, which no test in this tree builds. The
   chain from the command line is therefore verified in two pieces: the
   `LoadDflashDraft` hop is one line of existing, already-reached production code
   above `LoadQwen3DFlash(shards, ...)`, which case 2 enters directly and over a
   real file.

5. **The real artifact.** A case that loads the downloaded checkpoint when it is
   present and skips by name when it is not, so the file this spec measured is
   the file a gate reads.

**Fixture geometry is forced by the format, not chosen.** `vt::Exl3HadR128`
refuses a row length that is not a multiple of 128, because the transform IS
blockwise Hadamard-128. So every EXL3 K and N in a fixture is a multiple of 128,
and the draft's hidden size is the target's — which is why case 4 raises the
shared runner fixture's target hidden from 32. That parameter is DEFAULTED, so
every existing binary that includes the fixture is byte-unchanged.

## Gates

```sh
cmake --build build --target test_qwen3_dflash2_exl3 test_dflash2_exl3_reach -j 4
ctest --test-dir build -R "^(qwen3_dflash2_exl3|dflash2_exl3_reach)$" --output-on-failure
ctest --test-dir build -R 'dflash' --output-on-failure
ctest --test-dir build -R 'exl3' --output-on-failure
scripts/agent-preflight.sh --staged
```

Read the preflight's OUTPUT, never its exit code: it exits 0 while printing
"NOT a green preflight". Read `ctest`'s exit code and not doctest's
`assertions:` line, which stays green on a THROWN case and reads `0` on a skip.

Reachability is proved by MUTATION, not by reading. The table is in
`## Evidence`.

## Evidence

Filled by the implementation commits. Each mutation row must assert that the
mutation CHANGED the file (sha256), that it COMPILED, that the test binary's
mtime MOVED, and that the tree was restored byte-for-byte. A mutation failing
any of those is reported INVALID and never as a pass.

At minimum: the `fc` EXL3 call site; one attention projection call site; the MLP
seam swap; the F16 admission (which must RED — a BF16-only reader has to refuse
this artifact); and the `IsExl3()` predicate.

## Owed

- **No device run and no benchmark.** Every gate is CPU-only. #2495's headline
  (47.5 tok/s on GB10) is items 8-10 and none of them is touched here.
- **The target half.** #2495 items 5 and 6 — the EXL3 `lm_head` at bits 6, the
  bits-4 `mtp.*` head and `SharedHeadSource::LoadInto` — are another row's. Until
  they land, an EXL3 DRAFT can only be paired with a target whose head this tree
  already reads.
- **No merged EXL3 QKV or gate_up operand**, inherited from
  `quant-exl3-shared.md`, with the artifact-side reason recorded above: the six
  shards do not share a fitted `suh`.
- **No host-mirror release for the draft's `Exl3Weight`s.** The DFlash draft has
  no staging walk at all today, so nothing regressed; a residency cost on a
  host-addressable device, not a correctness one.

## Stop conditions

Stop and report rather than widen scope if: a merged trellis operand turns out to
be required; the F16 acceptance would have to leak outside an EXL3 load; the
change needs an edit to `LoadDflashSharedLmHead`, `SharedHeadSource` or the
Qwen3.5 head, which another row holds; or making a case pass would need an
existing arm's selection to move.
