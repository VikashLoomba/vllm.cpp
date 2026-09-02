# Qwen3.8-27B Q4_K_M token exactness against llama.cpp `b10451`

Row: `QUANT-QWEN38-27B-GGUF-ARM`.
Issue: [#2534](https://github.com/mudler/vllm.cpp/issues/2534).
Prior measurement:
[`docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md`](../../docs/bench-evidence/qwen38-27b-q4km-token-gate-20260823.md).

## Why this exists

The arm's declared token gate ran on 2026-08-23 and FAILED: tokenizer exact
6 of 6, generation divergent 5 of 6, every divergence a rank-2 loss under 0.18
logits with 282 of 288 steps at rank 1. `AGENTS.md` §Gates admits no speed or
memory result from this arm on any backend until that gate passes, so the whole
k-quant throughput lane is blocked behind it, and it already invalidated one
gfx1151 measurement (#2497).

The prior evidence named the next hypothesis and did not test it: "suspect the
quantized dot product and the activation dtype first". This row tests it.

## What the prior run actually measured, restated precisely

The issue title says "on CUDA". That means the HOST was an NVIDIA box and NOT
that the compute ran on a GPU. `/mnt/nas_share/rc/qwen38w3/job/job.sh` builds
llama.cpp with `-DGGML_CUDA=OFF` and vllm.cpp with `-DVLLM_CPP_CUDA=OFF`, and
runs `vllm-cli --model "$GGUF" --device cpu`. **Both engines executed their CPU
tiers, on aarch64 (`thor:gpu0`, 14 cores).** Every conclusion below is about the
CPU path. That is not a weakening of the issue: it is where the defect is, and
the ROCm arm the developer's goal names runs the same model code.

## The two candidate error terms, and their measured sizes

### Term A: our activations are BF16 where the oracle's are F32

vllm.cpp's `qwen35` CPU decode carries the residual stream and most projection
outputs in `bf16`:

- `hidden` is a literal `DType::kBF16` at every allocation site, with no toggle
  ([qwen3_5.cpp:8885](../../src/vllm/model_executor/models/qwen3_5.cpp#L8885)).
- `res` is `ResidualDType()`, `kBF16` unless `VT_BF16_RESIDUAL=0`
  ([qwen3_5.cpp:3483](../../src/vllm/model_executor/models/qwen3_5.cpp#L3483)),
  and its own comment records the choice as a memory-traffic one.
- The RMSNorm residual is added in f32, **rounded to the residual dtype, and
  then re-read**, twice per layer
  ([cpu_ops.cpp:576-583](../../src/vt/cpu/cpu_ops.cpp#L576)).
- The quantized GEMM accumulates in f32 and rounds once on store,
  `F32ToBF16`, for qkv, o_proj, down_proj, the GDN in/out projections and the
  lm_head GEMM
  ([cpu_quant_gemm.cpp:52-57](../../src/vt/cpu/cpu_quant_gemm.cpp#L52), reached
  from the `nrc == 1` decode path at
  [:107](../../src/vt/cpu/cpu_quant_gemm.cpp#L107)).
- The paged KV cache is `bf16` by default
  ([kv_cache_dtype.h:29](../../include/vllm/v1/kv_cache_dtype.h#L29)).
- The GDN conv and SSM state caches are `bf16`, gathered to f32 and scattered
  back **every token**, so their rounding is recurrent
  ([qwen3_5_common.cpp:52-63](../../src/vllm/model_executor/models/qwen3_5_common.cpp#L52)).

llama.cpp at `b10451` has none of that on this path. Its CPU graph is f32, and
its KV cache defaults to `GGML_TYPE_F16`, not bf16
(`src/llama-context.cpp:3538-3539`). BF16 carries 8 explicit mantissa bits, so
one store costs up to `2^-9 = 1.95e-3` relative; F16 carries 11 and F32 carries
24.

### Term B: ggml does not agree with itself across its own k-quant kernels

We port ggml's `_generic` tier only, by the header comment's own statement
([cpu_quant_dot.cpp:22-26](../../src/vt/cpu/cpu_quant_dot.cpp#L22)), and
`VecDotQ4_KQ8_K` is byte-for-byte `ggml_vec_dot_q4_K_q8_K_generic` at the pin.
The oracle does not run that body. At `b10451` a Q4_K matvec has four distinct
fp32 rounding schedules in one source tree, and the aarch64 build the gate used
takes the repacked one:

| schedule | where | fp32 applications of `d` per superblock |
|---|---|---|
| `ggml_gemv_q4_K_8x8_q8_K` (aarch64) | `ggml-cpu/arch/arm/repack.cpp:709` | 8 |
| `ggml_vec_dot_q4_K_q8_K` (SVE/NEON) | `ggml-cpu/arch/arm/quants.c:2334` | 1 |
| `ggml_vec_dot_q4_K_q8_K_generic` — **ours** | `ggml-cpu/quants.c:696` | 8 lanes, tail-folded |
| `ggml_gemv_q4_K_8x8_q8_K_generic` | `ggml-cpu/repack.cpp:958` | 16, deferred min |

The integer inner product is exact in all four; only the `int32 -> float`
conversion granularity and the order of the `d`/`dmin` multiplies differ.

**Measured, on this dev box (x86-64 AVX-512, Ryzen 9 9950X3D), against OUR OWN
compiled kernels rather than by reading source.** The harness links
`src/vt/cpu/cpu_quant_dot.cpp` and `cpu_quant_act.cpp` -- compiled with the
project's own `-ffp-contract=off` polarity -- against `libggml` built at
`b10451`, and asks three answers for the same bytes: ours, ggml's `_generic`,
and ggml's dispatched arch kernel. 2000 random rows per shape, `K` in
{2048, 5120, 12288}:

| quantity | result |
|---|---|
| our `QuantizeRowQ8_K` vs `quantize_row_q8_K_ref` | **byte-identical**, 0 of 18,000 rows, 0 of 4.6 M quants |
| our `VecDotQ6_KQ8_K` vs `ggml_vec_dot_q6_K_q8_K_generic` | **bit-identical**, 0 of 6000 rows |
| our `VecDotQ4_K/Q5_K` vs their `_generic` | differ on 80-89% of rows, by `max_abs` 1.1e-06 to 4.3e-06, `rms_rel` 5.5e-06 to 1.1e-04 |
| ggml arch vs ggml `_generic` | differ by `max_abs` 7.2e-07 to 5.4e-06, `rms_rel` 1.4e-05 to 9.8e-05 |

Two things follow. **The `QuantizeQ8KK` hypothesis the issue lists is refuted on
this path**: the activation encoder reproduces the reference byte for byte. And
our Q4_K/Q5_K last-ULP disagreement with the body we port is the SAME SIZE as
ggml's disagreement with itself, so it is a rounding schedule and not a port
defect -- Q6_K, whose generic body is the same shape MINUS the
`sumf -= dmin * sumi` term, is bit-identical, which locates the disagreement at
that one expression and at the `-O3 -ffp-contract=fast` versus
`-O2 -ffp-contract=off` difference between ggml's build and ours.

**So term A is about two hundred times term B.** `2e-3` per bf16 store, applied
twice per layer to the residual plus once per projection over 64 layers and
recurrently to the GDN state, against `1e-5` relative on a row dot. Term A is
the hypothesis this row tests first.

## Scope

1. Measure whether the oracle's own greedy decode is stable across its own
   kernel schedules, using the stock `-nr/--no-repack` flag (`common/arg.cpp:2413`)
   — no patch to the pinned tree. If the oracle's own tokens move, the target
   "token-exact vs llama.cpp" is only defined once host arch, `-mcpu` feature
   set, repack state and batch size are pinned, and this row says so.
2. Give the CPU `qwen35` path a resolved compute dtype so the arm can run at the
   precision its oracle uses, instead of six hardcoded `kBF16` sites and four
   independent env vars that between them cannot reach f32.
3. Re-run the declared token gate red-before (bf16, must reproduce 5 of 6) and
   green-after (f32).
4. Whatever the outcome, name the cause against specific arithmetic and record
   the residual.

Out of scope: the ROCm hang (#2511), IQ3_S (#2510), any throughput number, and
porting ggml's aarch64 kernels (that is term B and stays owed).

## Design

`hidden`, `res`, and the bf16-out projection helpers on the dense `qwen35` CPU
path resolve one dtype rather than each carrying its own literal. The default
stays `kBF16` on a device tier where it is a measured traffic win; the CPU tier
of this arm resolves `kF32`, because on CPU every consumer of a bf16 store
widens it straight back to f32 (`LoadActF32`,
[cpu_quant_gemm.cpp:41](../../src/vt/cpu/cpu_quant_gemm.cpp#L41);
`WidenRowToF32`, [cpu_ops.cpp:577](../../src/vt/cpu/cpu_ops.cpp#L577)), so the
narrow store buys bandwidth and loses mantissa with no compensating compute win.

The switch is one resolver, reachable from `ModelRegistry::Forward` through the
ordinary `--device cpu` load, so the gate measures a capability and not a class.

## Risks

- Changing the resolved dtype touches the CUDA arm's recorded perf defaults.
  Every device default must stay byte-identical; the mutation that proves it is
  in `## Tests`.
- F32 activations raise CPU resident bytes. The arm has no admissible memory
  number yet, so this is recorded, not gated.
- Term B may still keep the gate red. That is an acceptable, reportable outcome
  and must not be answered by widening the gate.

## Tests

- A red-first unit test that the CPU dense `qwen35` path resolves the compute
  dtype from one place, and that flipping the resolver changes the dtype of the
  hidden buffer, the residual and the bf16-out projections together.
- The device default is unchanged: a mutation that makes the resolver return
  `kF32` unconditionally must red a test that pins the CUDA-tier dtype.
- The ported comparison of `VecDotQ4_KQ8_K` against `ggml_vec_dot_q4_K_q8_K_generic`
  already exists in spirit in `tests/vt/test_ops_quant_dot.cpp`; this row adds
  the measured `b10451` arch-versus-generic spread as a recorded figure, not a
  gate, because the arch tier is not ported.

## Gates

Unchanged from
[`qwen38-27b-quant-arms.md`](qwen38-27b-quant-arms.md) §Gates. llama.cpp
`b10451` = `10bf611e533d81f739128304991c5e133c6aebd8`, this artifact
(`sha256 7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`,
17,106,775,008 B), greedy, 6 prompts, 48 tokens, concurrency 1, MTP off.
`TOKEN_GATE=PASS` requires 6 of 6.

**The near-tie band is not available and is not reached for.** It is admitted
only where the oracle's greedy decode is non-deterministic; this oracle
reproduced #857 byte for byte from a different build.

## Evidence required

- The rc job ids, the device, and the raw log paths.
- The kernel-level figures above, with the exact build recipe. Harness:
  `ourdot.cpp`, linking this tree's `cpu_quant_dot.cpp` + `cpu_quant_act.cpp`
  objects against `libggml-cpu.a`/`libggml-base.a`/`libggml.a` from a
  `GGML_NATIVE=ON` `b10451` build. It is deliberately NOT committed: this
  project ships without a ggml dependency, and a test that needs one is a
  dependency by another name.
- The oracle-against-itself result (repack on versus repack off).
- The final-logit delta distribution between two oracle configurations, measured
  by teacher-forcing one along the other's token sequence.
- Red-before and green-after gate output on the same binary pair.
- Every refuted hypothesis.

## Stop conditions

- Do not widen, weaken or delete the gate.
- Do not fabricate a root cause. A narrowed cause with a quantified residual is
  a valid outcome and gets posted to #2534.
- `NEEDS_DECISION` if making f32 the shipped CPU default for this arm is the
  only way to pass, since that is a product default and not a mirrored one.

## Owed

- Term B: ggml's aarch64 `arch/arm/quants.c` and `arch/arm/repack.cpp` k-quant
  kernels are not ported, so our CPU decode cannot be bit-identical to an
  aarch64 oracle build even at equal precision. Tracked by #2534 until this row
  closes and then re-filed if it survives.

## Now

`ACTIVE`.
