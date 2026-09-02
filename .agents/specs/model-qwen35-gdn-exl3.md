# MODEL-QWEN35-GDN-EXL3 — the EXL3 arm of the Qwen3.5 GDN linear-attention tower

Row: `MODEL-QWEN35-GDN-EXL3`
Issues: [#2495](https://github.com/mudler/vllm.cpp/issues/2495) (item 4)
Base SHA: `844664561` (`origin/main`)
Parent row: [`MODEL-QWEN35-EXL3`](model-qwen35-exl3.md) →
[`QUANT-EXL3-MUL1`](quant-exl3-mul1.md) → [`QUANT-EXL3`](quant-exl3-shared.md)

Upstream pin: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`. vLLM defines the
MODEL (`vllm/model_executor/models/qwen3_5.py` and
`vllm/model_executor/layers/qwen_gdn_linear_attn.py`, both already mirrored
here) and the SEAM (`layers/linear.py`,
`layers/quantization/base_config.py`). vLLM implements no EXL3 at the pin, so
the trellis FORMAT comes from the registered secondary oracle
[`exllamav3`](../oracles/exllamav3.md) @
`2398c05635fbbad01a0a51dce63c85c6c8a8450e` (MIT), exactly as the parent rows
record. Nothing here re-derives either half.

## Now

`ACTIVE`. The GDN linear-attention tower of an EXL3
`Qwen3_5ForConditionalGeneration` checkpoint loads and runs. The refusal the
parent row installed is gone.

## The gap

[`MODEL-QWEN35-EXL3`](model-qwen35-exl3.md) landed the DENSE half of the arm:
`self_attn`, `mlp` and `lm_head`. It deliberately left the GDN half refusing by
name, because 48 of `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`'s 64 layers are
`linear_attention` and nothing in `ProjectGdnQkvz` / `ProjectGdnBA` /
`ProjectGdnOut` consumed an `Exl3Weight`. The refusal was the honest behaviour
while it stood: without it `LoadGdnDense` fell through to
`get(la + "in_proj_qkv.weight")` and died on `tensor not found`, a sentence
about a checkpoint that is complete.

This row removes the refusal by making it untrue.

### What the artifact actually ships

Read from `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`'s own safetensors headers by HTTP
range request (metadata only; no weights downloaded). Every one of the 48
`linear_attention` layers carries exactly this, and layers 0/1/2 were checked
tensor by tensor:

| tensor | dtype | shape | note |
|---|---|---|---|
| `linear_attn.in_proj_qkv.trellis` | `I16` | `[320, 640, 64]` | K=5120, N=10240, **bits 4** |
| `linear_attn.in_proj_qkv.suh` | `F16` | `[5120]` | K |
| `linear_attn.in_proj_qkv.svh` | `F16` | `[10240]` | N |
| `linear_attn.in_proj_qkv.mul1` | `I32` | `[]` | codebook 2 |
| `linear_attn.in_proj_z.trellis` | `I16` | `[320, 384, 64]` | K=5120, N=6144, bits 4 |
| `linear_attn.in_proj_z.{suh,svh}` | `F16` | `[5120]`, `[6144]` | |
| `linear_attn.in_proj_z.mul1` | `I32` | `[]` | |
| `linear_attn.out_proj.trellis` | `I16` | `[384, 320, 64]` | K=6144, N=5120, bits 4 |
| `linear_attn.out_proj.{suh,svh}` | `F16` | `[6144]`, `[5120]` | |
| `linear_attn.out_proj.mul1` | `I32` | `[]` | |
| `linear_attn.in_proj_a.weight` | **`F16`** | `[48, 5120]` | NOT quantized, NOT BF16 |
| `linear_attn.in_proj_b.weight` | **`F16`** | `[48, 5120]` | NOT quantized, NOT BF16 |
| `linear_attn.conv1d.weight` | `BF16` | `[10240, 1, 4]` | |
| `linear_attn.norm.weight` | `BF16` | `[128]` | |
| `linear_attn.A_log` | `BF16` | `[48]` | |
| `linear_attn.dt_bias` | `BF16` | `[48]` | |

`config.json` agrees: `hidden_size 5120`, `linear_num_key_heads 16`,
`linear_key_head_dim 128`, `linear_num_value_heads 48`,
`linear_value_head_dim 128`, so `conv_dim = 2*16*128 + 48*128 = 10240` and
`value_dim = 6144`, which is exactly the `svh` widths above. `layer_types` is 64
entries in a 3:1 `linear_attention`/`full_attention` pattern.

### THREE PROJECTIONS, NOT A MERGED QKVZ

vLLM issues ONE `in_proj_qkvz` GEMM per GDN layer and this tree mirrors that
with the `in_proj_qkvz` merged owner. **The artifact does not ship one.** It
ships `in_proj_qkv` and `in_proj_z` as two independently quantized trellises,
each with its OWN `svh` (`[10240]` and `[6144]`) and its own `suh`.

The bf16 loader merges the two `.weight` shards because concatenating BF16 rows
is a byte copy. The trellis form is not: an EXL3 operand is a 16x16-tiled
bitstream whose two Hadamard sign vectors were fitted per projection at
quantization time, so an N-concatenation is only defined when both shards share
`suh` bit-for-bit, which nothing guarantees and which the calibration procedure
gives no reason to expect. There is no merged-trellis operand in this tree
either; `quant-exl3-shared.md` already records "no merged EXL3 QKV or gate_up"
as owed.

So this arm issues TWO GEMMs and mirrors the artifact. This is the instructed
polarity — match the artifact, not the bf16 convention — and it is also the
shape `ProjectGdnQkvz` already has a precedent for: the split native-FP8 arm the
35B runs takes exactly this path when the merged owner is empty.

### The second obstacle: `in_proj_a` / `in_proj_b` are F16

`LoadMergedBf16RawNK` accepts `BF16` and `F8_E4M3` and refuses everything else
by name. The artifact stores `in_proj_a`/`in_proj_b` at `F16` — exllamav3 keeps
the unquantized linear remainder at fp16 because it runs the linear in fp16 —
so the EXL3 GDN tower would have been refused a second time, on a tensor that
is not even quantized, after the trellis rung was wired.

This was not visible from the code and only the artifact's header shows it. It
is fixed here because it is inside this row's scope: the tower does not load
without it.

## Scope

**In.** The three GDN projections at every width the host decodes; the loader
rung; the forward routing in both the prefill and the decode `ProjectGdnOut`
tails and in `ProjectGdnQkvz`; the packed-decode dtype discriminator; F16
acceptance for the merged `in_proj_ba` owner, scoped to an EXL3 load; removing
the refusal; the gate.

**Out.** `kExl3MoeMlp`; the vision tower (ships unquantized bf16); the DFlash2
draft loader (#2495 item 7); the `mtp.*` head (item 5's other half);
`SharedHeadSource::LoadInto` (item 6); every performance question (items 9-10);
downloading the 14.2 GB checkpoint.

## Design

1. **Three `Exl3Weight` fields on `GdnLayerWeights`**, not one merged one:
   `in_proj_qkv_exl3`, `in_proj_z_exl3`, `out_proj_exl3`. Reason above. `IsExl3()`
   keys on `in_proj_qkv_exl3` alone, the same way `FullAttnLayerWeights::IsExl3`
   keys on `q_proj_exl3`, so the discriminator has ONE truth and cannot report a
   half-populated container as an arm.

2. **The loader rung is FIRST and EXCLUSIVE**, mirroring `LoadAttnDense`
   verbatim. An EXL3 projection ships no `.weight` at all, so every probe below
   it — `IsNvfp4Projection`, the `get(name + ".weight").dtype` read, the
   block-FP8 cross-check — would either mis-route it or die asking for a tensor
   the checkpoint correctly does not ship. Selection is by upstream's own storage
   predicate (`Linear.is_exl3_storage`, `modules/linear.py:385-389`) and not by a
   config flag, so a checkpoint that quantizes only some layers loads each one
   the way it is actually stored.

3. **The small tensors are loaded by ONE shared helper** (`LoadGdnSmallTensors`)
   that every arm calls, rather than copied into the EXL3 rung. A second copy of
   `conv1d` / `A_log` / `dt_bias` / `norm.weight` / `in_proj_ba` is where the two
   arms drift.

4. **F16 acceptance is scoped to an EXL3 load**, exactly as the parent row scoped
   `LoadModelVectorForScheme`: `LoadMergedBf16RawNK` grows an explicit
   `allow_f16` parameter that defaults to `false`, and only the EXL3 GDN rung
   passes `true`. Teaching that loader F16 outright would widen acceptance for
   every dense model through a conversion that drops three mantissa bits.

5. **The forward rung is FIRST and EXCLUSIVE too**, and emits `indt` for
   `mixed_qkv` and `outdt` for `z` — the dtypes the SPLIT BF16 arm already emits,
   so `VT_GDN_IN_BF16`'s and `VT_GDN_OUT_BF16`'s documented rollbacks stay honest
   on this arm. It routes through `dense_exl3::Linear`, which forwards to
   `layers::MakeLinearMethod` → `Exl3LinearMethod` → the ONE
   `dense_attn::Exl3MatmulD`. **No second EXL3 matmul is written.**

6. **The translation-unit boundary is REUSED, not re-solved.**
   `include/vllm/model_executor/models/dense_exl3_linear.h` exists because
   including `quantization/exl3.h` in `qwen3_5.cpp` pulls in `dense_attn_block.h`,
   whose `dense_attn::ResidentWeight` collides by ADL with `qwen3_5.cpp`'s own
   same-named helper across ~40 unrelated calls. The GDN rungs live in the same
   file as the dense ones and call the same two forwarders.

7. **The packed-decode discriminator is EXTENDED, not paralleled.**
   `GdnMixedQkvDTypeInputs` grows `has_exl3_qkv_owner`, and
   `GdnProjectedMixedQkvDType` answers `in_dtype` for it — the dtype the rung in
   (5) actually allocates. The struct's own comment records why the previous
   `in_proj_qkv_fp8.Empty()` proxy was wrong (it was a statement about weight
   FORMAT standing in for one about ACTIVATION dtype), and a fourth arm added as
   a parallel predicate would repeat exactly that. The prediction is asserted
   against the allocated buffer at the call site, so a drift on either side fails
   loudly.

8. **Geometry is CHECKED against the config**, not trusted. The rung asserts
   `InFeatures`/`OutFeatures` against `conv_dim`, `value_dim` and `hidden` the
   way the merged bf16 owner is asserted, because a trellis is self-consistent at
   more than one reading and a silently transposed projection is not otherwise
   visible.

## Tests

`tests/vllm/models/test_qwen35_exl3.cpp`, extended in place — the fixture the
parent row built already writes an EXL3 GDN tower behind `exl3_gdn=true` and
already had a case for it (G3, the refusal). G3 is REPLACED rather than deleted,
because the behaviour it pinned is the behaviour this row changes.

- **G3 (rewritten)** an EXL3 GDN tower LOADS: codebook 2, 4 bits, the right
  `in/out_features` on all three projections, EXACTLY ONE representation
  populated (the bf16 `in_proj_qkvz` / `out_proj` and every FP8 and NVFP4 field
  empty), the trellis still trellis BYTES, and the small tensors present beside
  it. Then `ModelRegistry::Forward` over the LOADED weights, compared against the
  same trellis bytes decoded into the bf16 fields.
- **G3b** the 3:1 `layer_types` pattern of the real model — 8 layers,
  6 `linear_attention` + 2 `full_attention` — loads and forwards, so the case
  measures the pattern the artifact actually declares and not a 1/1 toy.
- **G4** unchanged in intent and EXTENDED: the bf16, per-tensor FP8 and NVFP4
  GDN arms must still land in their own fields with the EXL3 ones empty.

The fixture's `linear_key_head_dim` moves from 32 to 64, which moves `conv_dim`
from 192 to 256. Not cosmetic and not a tolerance change: **every EXL3
projection's K and N must be a multiple of 128**, because each side carries a
blockwise Hadamard-128 (`exl3_lib/quantize.py:15`, and `src/vt/exl3_policy.cpp:150`
declines anything else). A `conv_dim` of 192 cannot be an EXL3 width at all. The
geometry stays SHARED across all four arms, which is what makes a mis-selection
the only difference a case can be reading.

## Gates

```sh
cmake --build build --target test_qwen35_exl3 -j 4
ctest --test-dir build -R '^qwen35_exl3$' --output-on-failure
ctest --test-dir build -R '^exl3_native_loader$' --output-on-failure
ctest --test-dir build -R '^qwen35_gdn' --output-on-failure
scripts/agent-preflight.sh --staged
```

Reachability is proved by MUTATION, not by reading: deleting the production
call site in `ProjectGdnQkvz` or `ProjectGdnOut` must red the gate, and the
mutation table with real assertion counts is recorded in `## Outcome`.

## Owed

- **No device run, and no benchmark.** Every case here is CPU. This row makes
  the published checkpoint LOAD; #2495's headline (47.5 tok/s on GB10) needs the
  14.2 GB artifact downloaded and run and is items 8-10, none of which this row
  touches. Loading is not benchmarking and #2495 stays open.
- **No merged EXL3 QKVZ.** Inherited from `quant-exl3-shared.md`, and this row
  records the artifact-side reason it is not merely an optimization: the two
  shards do not share a fitted `suh`.
- **The `mtp.*` head, #2495 items 5-7.** Untouched.
- **`docs/USAGE.md` still owes the checkpoint's file names, sizes, repo and
  REVISION** with a sha256 per quantized artifact. Inherited from the parent row
  and still owed for the same reason: nothing has been fed the published
  artifact yet, and a weights row written from an index read is a claim about a
  run that did not happen.
- **`ReleaseResidentQwen3_5DenseHostWeights` does not release a GDN EXL3 host
  mirror**, the same gap the parent row recorded for the dense one and for the
  same reason: it walks `OwnedTensor` members by name and knows nothing about
  `Exl3Weight`. A residency cost on a host-addressable device, not a correctness
  one.
- **The GDN tower is excluded from packed decode on this arm only by dtype**,
  never by weight format, which is the polarity #365 established. No measurement
  of the EXL3 tower under packed decode exists.

## Stop conditions

Stop and report rather than widen scope if: a merged trellis operand turns out
to be required; the F16 acceptance would have to leak outside an EXL3 load; the
`mul1` codebook or the bits-4 arm turns out not to cover a width the tower
needs; or making a case pass would need an existing arm's selection to move.
