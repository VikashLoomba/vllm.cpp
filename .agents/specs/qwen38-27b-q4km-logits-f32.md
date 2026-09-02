# Give the GGUF keep-quant lm_head f32 resolution, because the gate is decided by resolution and not by magnitude

Row: `QUANT-QWEN38-27B-GGUF-ARM`.
Issue: [#2534](https://github.com/mudler/vllm.cpp/issues/2534).
Predecessors:
[`qwen38-27b-q4km-token-exactness.md`](qwen38-27b-q4km-token-exactness.md) (four
causes refuted) and
[`qwen38-27b-q4km-logit-dump.md`](qwen38-27b-q4km-logit-dump.md) (the instrument
that identified this one).

## The measured finding this row acts on

Do not re-derive it. `d72baf2c`, `thor:gpu0`, 2026-09-02, evidence
[`qwen38-27b-q4km-logit-delta-20260902.md`](../../docs/bench-evidence/qwen38-27b-q4km-logit-delta-20260902.md):

- **288 of 288** of our top-1 logits lie exactly on the bf16 grid.
- Our smallest representable non-zero gap is 0.0625, rising to **0.125** at
  magnitude 16 to 32, which is where every contested step sits.
- The six contested gaps are 0.027185, 0.058050, 0.085434, 0.115482, 0.124247
  and 0.178236. **Five of six are at or below our own resolution.**
- Six steps in 288 are EXACT ties in our arithmetic. Four are lost to the index
  tie-break, two are won by the same accident, and the two remaining flips sit
  at one ULP.
- Our per-step delta against the oracle is INSIDE the oracle's own
  kernel-schedule noise band (median max_abs 0.4165 against 0.3790, 1.10x). The
  engine is not deciding worse at these steps. It cannot represent the
  distinction the gate asks about.

## The site, and which arm this checkpoint takes

`src/vllm/model_executor/models/qwen3_5.cpp::DenseLogitsF32D` is the ONE dense
logits GEMM. It has four arms:

| head | helper | output |
|---|---|---|
| EXL3 trellis | `dense_exl3::Linear(..., DType::kF32)` | f32, written directly |
| ModelOpt NVFP4 | `MatmulNvfp4F32D` | f32, written directly |
| `nk == false` (loader-transposed `[K,N]`) | `MatmulF32D` | f32, written directly |
| `nk == true` | `MatmulBf16LogitsF32D` | **bf16, then `vt::CastF32`** |

`MatmulBf16LogitsF32D` (`qwen3_5.cpp::MatmulBf16LogitsF32D`) calls `MatmulBf16D`,
which allocates `DBuf dout(d, ActDType(d), {M, N})` — bf16 at the shipped default
— and then widens with `vt::CastF32`. Widening cannot recover a discarded
mantissa.

**The `Owed` item the predecessor left is now answered by reading the loader, not
inferred.** A Q4_K_M GGUF on the CPU tier routes `output.weight` through
`qwen3_5_gguf_weights.cpp::LoadEmbedAndHead` ->
`qwen3_5_gguf_weights.cpp::OwnMatmulWeight` -> `::OwnGgufKeptSlice` ->
`::OwnGgufQuantBlocks`, and that function sets `o.nk = true` with the comment
"GGUF disk order [out, in] IS the MatmulBT [N, K] orientation: no transpose".
So **this checkpoint takes the fourth arm**, and it takes it because of a
*layout* flag that says nothing about numerics.

That is the defect in one sentence: **a k-quant head inherits a bf16-output rule
that was authored for a tied bf16 torch-Linear head**, because the two share an
orientation flag. The helper's own comment says which head it was written for —
"A tied BF16 lm_head follows torch Linear's model-dtype output".

## What upstream actually does, read at the pin

Pin `5559679229bc961848b121ccdeaa8fa5d79bec98`
([upstream-sync.md](../upstream-sync.md)), read in the local checkout at that
exact SHA.

- `vllm/model_executor/layers/logits_processor.py:99-136`
  (`LogitsProcessor._apply_head`) is the anchor the existing comment already
  cites. With `head_dtype` unset or equal to the hidden dtype it returns
  `lm_head.quant_method.apply(...)`, i.e. **the head's output is the model
  dtype**.
- `vllm/config/model.py:1783-1812` and `:2187-2208` (`ModelConfig.head_dtype`,
  `_get_head_dtype`) resolve that default: a **generation** model defaults to
  the model dtype; only a pooling model, or an explicit
  `--hf-overrides '{"head_dtype": "float32"}'`, gives an f32 head.
- `vllm/v1/sample/sampler.py:96` then does `logits = logits.to(torch.float32)`
  — a widening AFTER the head, structurally identical to our `vt::CastF32`.

**So vLLM's default keeps the head in model dtype and casts afterwards, exactly
as we do. Stated plainly, and this row does not pretend otherwise.**

Two further facts from the same function bound what mirroring can mean here:

- When an f32 head IS selected on CUDA/ROCm, upstream does not cast the operands.
  It accumulates the projection directly into f32 —
  `torch.mm(flat, lm_head.weight.t(), out_dtype=self.head_dtype)`
  (`logits_processor.py:127-131`) — "to avoid materializing an fp32 copy of the
  lm_head weight on every step". That is the shape of the fix below.
- Upstream **refuses** `head_dtype != model dtype` for a *quantized* head
  (`logits_processor.py:111-115`). That refusal is a limitation of its fallback
  mechanism, which is `F.linear(hidden.to(f32), lm_head.weight.to(f32))` and
  therefore needs a plain castable `.weight`. It is not a numerical policy: the
  CUDA branch immediately above it exists precisely to get an f32 head WITHOUT
  casting the weight.

## The divergence, argued rather than slipped in

This row makes the block-quantized head emit f32. That is a **deliberate
divergence from vLLM's default**, and it is argued on three grounds.

1. **vLLM does not implement this path at the pin.** There is no in-tree GGUF
   reader at `555967922` (`model_loader/__init__.py`; GGUF is an out-of-tree
   plugin, only `tests/plugins_tests/gguf` remains), which is the same fact
   `.agents/quantization-matrix.md` already records as the reason llama.cpp is
   this arm's oracle (#979). vLLM therefore resolves no model dtype for a GGUF
   checkpoint, so "the head inherits the one resolved model dtype" has no
   referent here. The upstream that DOES define this path, and that supplies the
   gate's whole denominator, computes the output head into F32
   (`ggml_mul_mat` dst is F32).
2. **The routing predicate is wrong on its own terms, independent of dtype
   policy.** Three of the four arms of `DenseLogitsF32D` already write f32
   directly, INCLUDING both other quantized heads. The fourth arm is selected by
   `nk`, a layout flag the keep-quant loader sets for an unrelated reason. So the
   change makes our own one function self-consistent rather than introducing a
   new polarity.
3. **The mechanism upstream lacks, we already have.** Our CPU keep-quant GEMM
   accumulates in `float acc` and only rounds at the store
   (`src/vt/cpu/cpu_quant_gemm.cpp::StoreOutF32`, whose `kF32` case already
   exists); the CUDA keep-quant kernel already branches on
   `out.dtype == DType::kF32`; the aarch64 repack tier already says "write f32
   directly when possible"
   (`src/vt/cpu/cpu_quant_repack_arm.cpp::QuantRepackMatmul`). So upstream's
   `out_dtype=` intent is reachable here on a quantized head with **no weight
   copy and no extra pass** — it deletes the `CastF32` and its buffer.

**No bf16 or f16 head moves.** The tied-bf16 arm vLLM's comment names keeps
`MatmulBf16LogitsF32D` byte for byte, so no shipped safetensors default and no
recorded device measurement on those arms changes.

## Scope

1. `DenseLogitsF32D` routes a **block-quantized** head (`vt::IsBlockQuant`) to
   the existing f32-output `MatmulF32D`. This is **routing**, not a new dtype
   path and not a change to any helper's arithmetic.
2. A red-first focused gate that enters through `ModelRegistry::Forward` and
   asserts the property the measurement named: the logits of a block-quant head
   are OFF the bf16 grid, while a bf16 head's stay ON it.
3. Re-run the Q4_K_M token gate on `thor:gpu0` and report the result honestly.

Out of scope: `VT_ACT_F32` (an incomplete conversion, owed separately), the
residual dtype (`VT_BF16_RESIDUAL=0` was measured and does not help — wrong
bf16), [#2548](https://github.com/mudler/vllm.cpp/issues/2548), a `head_dtype`
config knob, and every throughput number.

## Risks

- **Widening is necessary and NOT established as sufficient.** Two contested
  gaps (0.027185 and 0.058050) sit below the f32-versus-oracle agreement
  measured at those steps, so this row must not promise a PASS. It measures one.
- **A token gate cannot see a dtype that is too wide** (`AGENTS.md`). The
  divergence above is therefore argued in prose and in the code comment, because
  no gate we own can convict it.
- **Cost.** The change removes an allocation and a `[M, vocab]` cast pass and
  doubles the bytes stored by that one GEMM. **No speed number is quoted, and
  none is admissible**: `AGENTS.md` §Gates admits a performance result only
  after this arm's declared token gate passes, and it does not.

## Gates

- Focused: the new case, plus the existing `test_qwen27_paged_forward` and
  `test_gguf_keep_quant` suites.
- Mutation: reverting the routing line must red the new case; deleting the
  production call site must red it too.
- The Q4_K_M token gate, `thor:gpu0`, the recorded 2026-08-23 recipe against the
  recorded `b10451` ids.

## Evidence required

- The upstream `file:line` mirrored, at the pin.
- Whether the fix is routing or a dtype change, and which arm this checkpoint
  takes, read from the loader rather than inferred.
- The token gate result after the change, per prompt.
- Whether the six near-ties resolved, and what happened to the two gaps below
  the f32-versus-oracle agreement.

## Stop conditions

- Do not report a PASS that was not measured, and do not soften a FAIL.
- An improvement short of 6 of 6 is a real result; report it with the residual
  named.
- If the operator judges vLLM's default binding on a path vLLM does not
  implement, the divergence above is one line to revert; that is a product call.

## Owed

- The **heavy tail** the predecessor recorded and this row does not explain: our
  `max_abs` reaches 17.1606 against the oracle's 1.3657 at `p5/2`, `p5/11` and
  `p0/45`. Not temporal, flipped no argmax, and no part of this change addresses
  it. Tracked by [#2534](https://github.com/mudler/vllm.cpp/issues/2534).
- The throughput axis of an f32 logits head. It is not owed to anyone yet: no
  speed axis on this arm becomes admissible until its token gate passes.

## Now

`ACTIVE` -- the routing is written and the token gate re-run is the open item.
