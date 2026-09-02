# QUANT-EXL3-MUL1 — the `mul1` codebook and the bit widths a 3.5bpw EXL3 artifact actually ships

Row: `QUANT-EXL3-MUL1`
Issues: [#2495](https://github.com/mudler/vllm.cpp/issues/2495) (primary),
[#2574](https://github.com/mudler/vllm.cpp/issues/2574) (slice F, the recount
and the `(3, 2)` arm)
Base SHA: `11fed3ba5`
Parent row: [`QUANT-EXL3`](quant-exl3-shared.md)
Matrix: [`.agents/quantization-matrix.md`](../quantization-matrix.md)

Upstream pin: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` — **vLLM implements
no EXL3** at the pin, so the format is mirrored from the registered secondary
oracle [`exllamav3`](../oracles/exllamav3.md) @
`2398c05635fbbad01a0a51dce63c85c6c8a8450e` (MIT). The seam is vLLM's; exllamav3
supplies the trellis format and its kernels only.

## Now

`ACTIVE`. Slices A and B have landed: the host decoder implements codebook 2 and
the loader accepts a `mul1` marker. Slices C, D and E are described below with
what each one still owes. **Slice F is the recount** (#2574): slice D was
specified against a tensor census that omitted 137 modules, so `(3, 2)` was
never instantiated and the checkpoint could not run on CUDA at all.

## The gap

`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` (#2495) cannot be loaded at all, and two
independent by-name refusals stand in the way:

1. **The `mul1` codebook (cb 2).** Every quantized linear in that checkpoint
   ships a `.mul1` marker. `dense_weight_loaders.h` refused it by name and
   `cuda_exl3.cu`'s `decode_3inst_2` `static_assert`ed it out.
2. **Bit widths 3, 4 and 5, at codebook 2.** The device arm instantiated
   `(3,0)`, `(3,1)` and `(6,0)` only. The census slice D worked from --- "270
   tensors at 4 bpw, one at 5, one at 6" --- was an UNDERCOUNT, and the
   corrected one is in slice F below: **409** trellis modules, **137** of them
   at bits 3.

Neither refusal was wrong. Decoding cb 2 as cb 0 or cb 1 is the failure this
family's records already document at length: the wrong multiplier yields a
weight with the RIGHT DISTRIBUTION and no correlation to the true one, so every
shape check passes and the model emits fluent nonsense.

## Why cb 2 is a port and not a table entry

Codebooks 0 and 1 differ only in the scramble. Both then mask, xor, and sum the
two fp16 halves of the 32-bit product:

```
x = cw * M ;  x = (x & 0x8fff8fff) ^ 0x3b603b60 ;  v = fp16(x.lo) + fp16(x.hi)
```

Codebook 2 (`codebook.cuh:82-89`) shares only the multiply:

```
x = cw * 0x83DCD12D
s = 0x6400 + (byte0(x) + byte1(x) + byte2(x) + byte3(x))     // __dp4a
v = fma_fp16( bitcast_fp16(s), 0x1eee, 0xc931 )
```

Three facts make that port exact rather than approximate, and each is asserted:

- `0x6400` is chosen so the reinterpretation is exact. fp16 has an ULP of
  exactly 1.0 across the whole `[1024, 2048)` binade, and the byte sum is at
  most `4*255 == 1020`, so `bitcast_fp16(0x6400 + sum)` is the integer
  `1024 + sum` for every reachable sum. Measured over the full 16-bit codeword
  domain, the reachable sums are `[0, 1005]`.
- The two constants are BIT PATTERNS, not the decimals in upstream's comments.
  `0x1eee` is `887/131072 == 0.00676727294921875`, not "0.00677"; `0xc931` is
  `-1329/128 == -10.3828125`, not "-10.39".
- **`__hfma`'s single rounding is reproducible in f32 without a fused op.** `h`
  is an integer in `[1024, 2044]`; `k_inv` is `887 * 2^-17`, so `h * k_inv`
  needs at most 21 significant bits and is exact in f32. `k_bias` is
  `-1329 * 2^-7`, an integer multiple of the same `2^-17` quantum, so the sum
  is too and its magnitude stays under `2^2` — every reachable value lands
  exactly on an f32. The only rounding is the final one to fp16, which is where
  `__hfma` rounds as well. Verified over all 1021 reachable byte sums.

## Scope

IN: the codebook-2 decode on the host and on the device; the loader's resolution
of a `mul1` marker; the `(bits, codebook)` arms `(4,2)`, `(5,2)` and `(6,2)` on
the device; the `dq_dispatch` routes for widths 4 and 5; and the records those
changes make stale.

OUT: the `m <= 8` GEMV, which stays specialized to `bits == 3`; the fused MoE
mgemm, which stays `(3, 1)`; every codebook-0 and codebook-1 decode path, which
must stay byte-identical; and any benchmark of the #2495 checkpoint.

## Upstream chain

vLLM implements no EXL3 at the parity pin, so the chain is the registered
secondary oracle's, read at `2398c05635fbbad01a0a51dce63c85c6c8a8450e`:

- `exllamav3_ext/quant/codebook.cuh:25-41` `decode_mul1_product_2` — the
  two-at-a-time cb 2 decode, which the device arm ports verbatim.
- `exllamav3_ext/quant/codebook.cuh:76-89` `decode_3inst<2>` — the same
  arithmetic one codeword at a time, which the host arm ports.
- `exllamav3/modules/quant/exl3.py:74-77,197,223` `LinearEXL3` — the codebook is
  derived from tensor PRESENCE and passed as two booleans.
- `exllamav3/modules/quant/exl3_lib/quantize.py:18-19,1417-1424` — the two
  multipliers and the marker tensors, each written as
  `torch.tensor(<mult>, uint32).view(torch.int)`.
- `exllamav3_ext/quant/exl3_dq.cuh:254-293` `dq_dispatch` — the per-width route:
  `dq8` for 3 and 4, two `dq4`s for 5, 6 and 8, `dq2x2` for 7.
- `exllamav3_ext/quant/exl3_dq.cuh:164-185` `dq8_aligned_4bits` — upstream's
  hand-written 4-bit form. The generic `dq8<4, cb, 4>` this tree already has
  computes the SAME eight windows; verified analytically (`s2 == 0`,
  `i2 % 32 == t >> 3` and `i0 % 32 == ((t >> 3) + 31) & 31` for every
  `t = lane << 3`) and numerically over 200 random tiles by 32 lanes.
- `exllamav3_ext/quant/exl3_gemv.cu:74-90,107-121` — the GEMV instantiation grid
  and its hard gate, which is what decides Slice E.
- `exllamav3_ext/quant/exl3_kernel_map.cuh:81-110` and `comp_units/` — the
  per-`(K, cb)` compilation-unit split, 24 TUs of 16 instantiations each.

## Our baseline

`QUANT-EXL3` W3 left the device arm at `(3,0)`, `(3,1)`, `(6,0)`, the host
decoder at codebooks 0 and 1, and `vt::Exl3Gemm` refusing any other codebook at
the seam. `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` therefore refused at LOAD time,
before any kernel was reached, on the `mul1` marker of its first quantized
linear.

## Port map

| Upstream | Here |
|---|---|
| `codebook.cuh:76-89` `decode_3inst<2>` | `Exl3DecodeCodeword(cw, 2)`, `src/vt/cpu/cpu_exl3_dequant.cpp` |
| `codebook.cuh:25-41` `decode_mul1_product_2` | `decode_mul1_product_2`, `src/vt/cuda/cuda_exl3.cu` |
| `util.cuh:83-90` `half_uint16` | the same union, `src/vt/cuda/cuda_exl3.cu` |
| `exl3.py:74-77` presence rule | `r.codebook = has_mul1 ? 2 : (has_mcg ? 1 : 0)`, `dense_weight_loaders.h` |
| `exl3_dq.cuh:254-293` `dq_dispatch` | `dq_dispatch<bits, cb>`, widths 3-6 |
| `exl3_kernel_map.cu:93-130` the `[K][cb]` table | `Exl3ArmInstantiated` + `GemmKernelForShape` |

## Tests to port

Upstream ships no unit test for the codebook itself — it is exercised only
through `reconstruct` against a reference tensor — so there is no upstream test
to preserve here. What replaces it is stronger for this purpose and is stated as
a deliberate adaptation: the codebook is gated against values computed
independently in exact rational arithmetic from `codebook.cuh:82-89`, so the
gate does not depend on any implementation of it.

## Dependencies

- `QUANT-EXL3` (the seam, the loader and the device arm this row widens).
- The `exllamav3` oracle pin, which must not move under this row.
- An `rc` lease for the device half. There is no local nvcc.

## Work breakdown

- **A — the host reference.** `Exl3DecodeCodeword(cw, 2)` and the seam's
  codebook range. Gated against hand-computed literals.
- **B — the loader refusal.** `r.codebook = has_mul1 ? 2 : (has_mcg ? 1 : 0)`,
  the `mul1` dtype check, and the test that pinned the refusal, CHANGED rather
  than deleted.
- **C — the device decoder.** `decode_mul1_product_2` and the `cb == 2` arm.
- **D — bit widths 4 and 5.** `dq_dispatch`, `Exl3ArmInstantiated`,
  `GemmKernelForShape` and the refusal message.
- **E — the GEMV.** Decide, and record the decision with upstream's own
  envelope rather than an assumption.
- **F — the recount, and `(3, 2)`.** Re-derive the census from the LOCAL
  artifact, instantiate the width the census adds, and correct every place the
  old number was written down.

## Risks/decisions

- **Decoding cb 2 as another codebook is invisible.** The wrong multiplier
  yields a weight with the right distribution and no correlation to the true
  one; every shape check passes and the model emits fluent nonsense. This is why
  the refusal existed and why the gate is against independent literals.
- **A token gate cannot see this.** The decision to gate on hand-computed fp16
  bit patterns, and to assert the BIT PATTERN as well as the value, is what
  catches a result that is numerically close but not an fp16.
- **Widening one seam is not widening the other.** `vt::Exl3Gemm` validates the
  codebook; the device launcher validates the `(bits, codebook)` PAIR. Both are
  asserted, in both directions, so a future widening of one cannot be mistaken
  for the other.
- **Six arms, not sixteen.** Upstream's dense table is affordable because it
  splits per `(K, cb)` into one TU each; this tree has one TU in a fat build
  over ten architectures. Widening past six should carry that split.

## Slices

- **A — the host reference.** `Exl3DecodeCodeword(cw, 2)` in
  `src/vt/cpu/cpu_exl3_dequant.cpp`. Gated against HAND-COMPUTED literals, not
  against this tree.
- **B — the loader refusal.** `r.codebook = has_mul1 ? 2 : (has_mcg ? 1 : 0)`
  in `include/vllm/model_executor/models/dense_weight_loaders.h`. The
  both-markers refusal stays; so do the `had` and packed `su`/`sv` refusals.
- **C — the device decoder.** `decode_mul1_product_2` and the `cb == 2` arm of
  `decode_3inst_2` in `src/vt/cuda/cuda_exl3.cu`.
- **D — bit widths 4 and 5.** `Exl3ArmInstantiated`, `GemmKernelForShape` and
  `dq_dispatch`.
- **E — the GEMV.** `exl3_gemv_kernel` is specialized to `bits == 3`.
- **F — the recount, and `(3, 2)`.** `Exl3ArmInstantiated`,
  `GemmKernelForShape` and the refusal message again, plus the corrected census
  in `docs/FEATURES.md` and in this spec. See the section below.

## Slice F — the recount, and the `(3, 2)` arm (#2574)

### The census, re-derived

Slice D was specified against "270 of the checkpoint's quantized tensors are
4 bpw, one is 5 and one is 6". That count is 272. The artifact has **409**
trellis modules.

Re-derived from the LOCAL artifact rather than from range requests over the
HuggingFace API, because a range-read census is what produced the undercount:
`/mnt/nas_share/models/Qwen3.8-27B-EXL3/target-3.5bpw`, revision
`19441ac874c4018295da848e250f23511361cda4`, both `model-0000N-of-00002.safetensors`
shards complete (each shard's largest declared `data_offsets` end equals its
file size). A module is a trellis module when it owns a `.trellis`; its width is
`shape[-1] / 16` (`[k/16, n/16, 16*bits]` on disk as `I16`), and its codebook is
2 when it owns a `.mul1`, 1 for `.mcg`, 0 for neither.

```
2426 tensors, 409 trellis modules, 409 carrying mul1, 0 carrying mcg
bits histogram: {3: 137, 4: 270, 5: 1, 6: 1}
```

The 137 at bits 3 are **entirely** the language model's MLP, and there is not
one of them anywhere else in the artifact:

| module | bits 3 | bits 4 |
|---|---:|---:|
| `model.language_model.layers.<N>.mlp.gate_proj` | 46 | 18 |
| `model.language_model.layers.<N>.mlp.up_proj` | 46 | 18 |
| `model.language_model.layers.<N>.mlp.down_proj` | 45 | 19 |

That is the per-layer mixed-precision split which averages to 3.5 bpw: 64
language-model layers, each MLP projection quantized at 3 or at 4 bits. The
remaining bits-4 population is the GDN tower (144: `linear_attn.in_proj_qkv`,
`in_proj_z`, `out_proj` over 48 layers), the 16 dense-attention layers (63
projections), and the 8-module `mtp` head. `layers.63.self_attn.o_proj` is the
lone bits-5 module and `lm_head` the lone bits-6 one.

### What this changes

`(3, 2)` was absent from `Exl3ArmInstantiated`, so **every one of those 137
projections refused by name** and the checkpoint could not execute a single
decoder layer on a CUDA device. The three instantiations slice D added were
correct and stay; this is the fourth pair the same artifact needs, missed
because the census under-reported it.

It is an INSTANTIATION and not a port. `decode_3inst_2<2>` and
`decode_mul1_product_2` are slice C's, already compiled and already gated at
widths 4, 5 and 6; `dq_dispatch` already routes `bits == 3` through `dq8`,
because `(3, 0)` and `(3, 1)` have used that route since QUANT-EXL3 W3. No new
decode is written here, and none may be: a second decode for the same codebook
is the way two ports of one mistake agree with each other.

### The confusable pair

`(3, 1)` and `(3, 2)` are now BOTH instantiated at the same width, and they are
the pair a threading mistake would silently swap: same `bits`, adjacent `cb`,
same `dq8` route, same tile shapes. A wrong thread does not fail to compile and
does not change any shape --- it decodes with `mcg`'s multiplier and yields a
weight with the right distribution and no correlation to the true one, which is
the failure mode this row's `## Risks/decisions` already names. The mutation
table therefore carries that exact swap, and the device case must red on it.

### The mutation margin, measured before the device run

`(3, 1)`-for-`(3, 2)` is only a useful mutation if the two decodes disagree by
much more than the bound. That is a property of the CODEBOOKS and not of the
device, so it is measurable on the CPU arm alone, and it was measured before
the lease came:

| bits | cb 1 vs cb 2, relative RMS | against the 1.0e-3 bound |
|---|---:|---:|
| 3 | 1.70865 | 1709x |
| 4 | 1.52897 | 1529x |

Same fixture, same seeds and same shapes the device case uses
(`MakeFixture(256, 256, bits, 0x1D0C0DE + bits)`, m = 4), both arms run through
`vt::Exl3Gemm` on a CPU queue, scratch program, not committed. A relative RMS
above 1 is the SATURATED error of two uncorrelated results, which is the
"right distribution, no correlation" this row's `## Risks/decisions` describes,
and it is what makes the M2 row below a bimodal mutation rather than a
tolerance nudge. If the device ever reports an M2 number NEAR 1.0e-3 rather
than near 1.7, the mutation did not apply and the row is not evidence.

### Not in scope

- **The `m <= 8` GEMV** (#2570). `exl3_gemv_kernel` carries
  `static_assert(bits == 3, ...)` and `LSTRIDE` is hardcoded to 24 there; its
  `(3, 2)` and `(4, 2)` arms are a separate and larger job, and
  `Exl3GemvArmInstantiated` stays `bits == 3 && cb == 1`. Upstream's `SEL_GRID`
  does list `(3, 2)`, so this is owed rather than refused as impossible.
- **The loaders, the DFlash2 draft, and the `lm_head`/`mtp` heads.** Other rows
  own those files.
- **ROCm and Vulkan.** Both transcribe the portable CPU reference and decode
  every width already, so the CUDA instantiation list does not bound them.

### The gate, and why this tier

`tests/vt/test_exl3_gemm.cpp`'s "the widened (bits, codebook) arms agree with
the CPU arm" case gains one row. The reference is the CPU arm, which decodes
every width and every codebook and is itself tied to real exllamav3 data by
`test_exl3_real_decode`; the bound is `rel_rms <= 1.0e-3`, which is the SAME
tier `(4, 2)`, `(5, 2)` and `(6, 2)` use and the same one the `(3, 1)` f64 case
states. It is the right tier because the quantity being bounded is the same one:
the tensor-core reduction order. The decode is exact on both sides, so a
disagreement larger than that bound is a decode disagreement, not rounding.

A device that is absent SKIPS this case, and a skip that asserts nothing reports
`assertions: 0`, which reads as a pass. The case already asserts its own
precondition on the skip path, and this row does not weaken that.

## Tests

- `tests/vt/test_exl3_dequant.cpp` — the hand-computed cb 2 table, a
  whole-domain distinctness sweep against cb 0 and cb 1, a whole-domain
  unchanged-ness sweep for cb 0 and cb 1, and a refusal case for a codebook
  upstream does not define.
- `tests/vllm/model_executor/layers/test_exl3_native_loader.cpp` — the case
  that pinned the `mul1` refusal is CHANGED, not deleted: it now asserts the
  marker is accepted and reports `codebook == 2`, and a sibling case keeps the
  both-markers refusal.
- `tests/vt/test_exl3_gemm.cpp` — the device cb 2 cross-check against the host
  decoder, and the new `(bits, cb)` arms.
- `tests/vt/test_exl3_gemm.cpp` — slice F adds the `(3, 2)` row to that same
  device arms table. It is one row and not a new case, because the question is
  identical to the one the table already asks of the other cb 2 widths.

## Gates

```sh
ctest --test-dir build -R '^test_exl3_dequant$' --output-on-failure
ctest --test-dir build -R '^test_exl3_native_loader$' --output-on-failure
ctest --test-dir build -R '^test_exl3_linear_method$' --output-on-failure
ctest --test-dir build -R '^test_exl3_gemm$' --output-on-failure
scripts/agent-preflight.sh --staged
```

The device arms additionally need an `rc` lease and a CUDA build; a CPU-only
green is not a device result and is never reported as one.

## Owed

- **Slice C is UNVERIFIED ON A DEVICE until a CUDA build runs it.** A CPU-only
  gate cannot compile `cuda_exl3.cu` at all, so a green CPU preflight says
  nothing about the device arm. Named here so it is visible debt rather than an
  assumed pass.
- **Slice D's fat-build cost.** Each new `(bits, cb)` pair is a full kernel set
  compiled for every architecture in the fat build. Upstream's own answer is a
  per-K compilation-unit split (`comp_units/exl3_comp_unit_K_cbX.cu`, 24 TUs of
  16 instantiations each); this tree has one TU.
- **Slice F is UNVERIFIED ON A DEVICE until a CUDA build runs it**, on the same
  terms as slice C: a CPU preflight never compiles `cuda_exl3.cu`.
- **Slice F adds a seventh pair to a table this spec argued should stay at
  six.** The argument stands and the reason it is overridden is that the seventh
  pair is not a new artifact's width, it is the SAME artifact's width that the
  census missed --- six was never the right number for #2495, seven was. The
  per-K translation-unit split upstream uses is now more owed, not less, and the
  compile-cost measurement is recorded with this slice's evidence.
- **Slice E: the GEMV has no arm for these widths.** Upstream's own envelope
  refuses `K < 2 || K > 4` (`exl3_gemv.cu:107-121`) and its instantiation list
  is `(4,0) (4,1) (4,2) (2,1) (2,2) (3,1) (3,2)` — so **bits 5 and 6 have no
  GEMV upstream either**, and falling to the GEMM there is upstream's behaviour
  rather than a gap this row opened. Bits 4 cb 2 IS in upstream's list, and
  reaching it is a kernel port: `LSTRIDE` is `bits == 3 ? 24 : 32`, `LOADS` is
  `WNT`, and a `dq8_regs_4bits` register extractor does not exist in this tree.
- **The fused MoE arm is `(3, 1)` only** (`kMoeBits`/`kMoeCb`). Out of scope
  here because the #2495 checkpoint is dense, and named so the next MoE EXL3
  artifact does not rediscover it.
- **No end-to-end run of the #2495 checkpoint, and no benchmark.** This row
  ports the format. It does not produce the number #2495 asks for, which is why
  the commits say `Refs` and not `Closes`.
- **`docs/USAGE.md` owes the checkpoint's file names, sizes, repo and
  REVISION** once an arm of it actually loads end to end.

## Stop conditions

- A hand-computed cb 2 literal disagrees with the implementation → stop and
  re-derive from `codebook.cuh:82-89`. Never adjust the literal to green.
- A new `(bits, cb)` arm changes any existing arm's output → the change is not
  additive; stop.
- The fat build stops fitting or a shape's `static_assert` on shared memory
  fires → record the number and propose which pairs are essential, rather than
  raising a limit.
- `(3, 2)` and `(3, 1)` agree on the device case → the case is not
  discriminating, because those two codebooks must produce different numbers at
  the same width. Stop and fix the fixture before reading the tolerance.
