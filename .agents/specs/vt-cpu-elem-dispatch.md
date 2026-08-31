# SPEC — `VT-CPU-ELEM-DISPATCH`: the per-element dtype dispatch in the CPU attention kernels

Issue: [#2376](https://github.com/mudler/vllm.cpp/issues/2376).
Predecessor: [`.agents/specs/ltx25-connector-gemm.md`](ltx25-connector-gemm.md),
whose `## Owed` names this work and measured it. That row states the defect and
declines to repair it, because `AttentionCrossKernel` is a `vt` seam every model
reaches and a change to it needs its own row, its own spec and a fresh reviewer.
This is that row.

## The defect

`vt::AttentionCross` and `vt::Attention` read every operand element through
`LoadF32` (`src/vt/cpu/cpu_ops.cpp`), which switches on `t.dtype` and multiplies
the offset by `vt::SizeOf(t.dtype)` once per element. Both are loop-invariant.
`vt::SizeOf` was declared at `include/vt/dtype.h` and defined out of line at
`src/vt/dtype.cpp`, and the build enables **no LTO** — `CMakeLists.txt` sets no
`INTERPROCEDURAL_OPTIMIZATION` and passes no `-flto`, confirmed by a `grep` over
`CMakeLists.txt` and `cmake/` that returns nothing — so it was a cross-
translation-unit call the optimizer could not remove. `AttentionCrossKernel`'s
score loop calls `LoadF32` twice per element around a body of one multiply and
one add.

`LoadF32`/`StoreF32` appear at **219 call sites across 64 kernels**, in
`src/vt/cpu/cpu_ops.cpp` and `src/vt/cpu/cpu_paged_attn.cpp`. Attention is where
it was measured, not where it lives.

## Scope

IN scope:

- **W1** — evaluate the CHEAP repair FIRST and report its number either way:
  make the dispatch inlinable at the source. One change; it reaches all 219
  sites without touching a kernel.
- **W2** — only if W1 falls short, hand-hoist `AttentionCrossKernel` and
  `AttentionKernel`. Those two, and no others.
- **W3** — a byte-equality gate over the dtype matrix the ops actually accept,
  with a mutation per claimed guarantee.

OUT of scope, declared rather than approximated:

- **The other 62 kernels.** Sweeping them is not one row. What they are owed,
  and the method for sizing which of them is worth a row, is in `## Owed`.
- **Changing any output value.** Byte equality is the bar. A change that cannot
  be bit-exact stops and reports; it is never traded for a rate.
- **`include/vt/quant.h`'s `elem_kn_repacked` claim.** The predecessor row owns
  it. This row carries the finding forward in `## Owed` and edits nothing.
- **The aarch64 half.** Every number here is x86-64 AVX-512. See `## Owed`.

## Design

Two repairs, evaluated in cost order, because they are not equal and the cheap
one might have made the expensive one unnecessary.

**(A) Make the per-element helpers inlinable.** `vt::SizeOf` moves into
`include/vt/dtype.h` as `inline`, with its two refusals kept out of line and
cold as `[[noreturn]]` helpers so the hot path is a switch over six integers and
nothing else. The enumeration stays exhaustive with no `default:` label, so
adding a dtype is still a `-Wswitch` error rather than a silent zero. The same
transformation applies to `F16ToF32` and `BF16ToF32`, which every `LoadF32` in
`src/vt/cpu` calls once per element on the reduced-width arms; the two f32 ->
narrow directions stay out of line, being store-side and carrying the
round-to-nearest-even logic. `vt::SizeOf` is not in `cmake/vllm_export.map`, so
no ABI moves.

**(B) Hoist the dispatch out of the two attention kernels.** This is the shape
`MatmulOneChunk` already applies against `MatmulOneChunkRef` in the same file,
and it is not a new one: widen the row that is REUSED to f32 once
(`WidenRowToF32`, which the GEMM already uses and which shares the same
converters), and resolve the STREAMED operand's dtype once per call into a typed
micro-kernel reached through a function pointer — the `ElemGemmTierTable` shape,
one indirect call per key row instead of two switched calls per element.

**Why it is bit-exact, stated as an argument rather than as a hope:**

- `WidenRowToF32` writes exactly the f32 values `LoadF32` returned for the query
  row, through the same `F16ToF32`/`BF16ToF32`, so the multiplicands are the
  same bits;
- `AttnDotT` accumulates over `e` in the same increasing order into one f32
  accumulator. It is a serial float reduction, so with `-ffp-contract=off`
  (CMakeLists.txt:55) and no `-ffast-math` the vectorizer may not reassociate it
  and does not;
- `AttnAccumT` walks the same `(j, e)` order into the same f32 `acc[]`, and each
  `acc[e]` stays its own independent chain over `j` — which is exactly what lets
  it vectorize ACROSS `e` without reordering any sum;
- each output row is independent, so the threadpool partition is unchanged.

No accumulator is split and no sum is reordered. The outputs are
`memcmp`-identical, not close.

## Reachability

Nothing new lands unreached. Both kernels are already registered
(`src/vt/cpu/cpu_ops.cpp`, the `AttentionCrossFn` registration) and already
reached from `ModelRegistry::Forward` through `vt::AttentionCross` and
`vt::Attention` — the LTX-2.5 connector and DiT among many callers. This row
changes the body of a reached kernel and adds no new entry point. The gate enters
through `vt::AttentionCross`/`vt::Attention`, never by constructing the kernel,
and the two reachability mutations in `## Outcome` prove the new micro-kernels
are what executes.

## Tests to port

There is no upstream test: upstream's attention is `torch` and cuBLAS. The tests
are this tree's own, and each is an executable observable.

| ID | Assertion |
|---|---|
| T1 | CPU `vt::AttentionCross` is `memcmp`-identical to a per-element reference over the dtype matrix the op accepts, at ragged shapes, with no bias / `[1,S]` bias / `[Tq,S]` bias |
| T2 | CPU `vt::Attention` likewise, causal and non-causal |
| T3 | a non-float operand is still refused at the op boundary, with the message unchanged |

`tests/vt/test_ops_attention_elem_dispatch.cpp`. The reference is the ORIGINAL
per-element loop, re-derived in the test from the layout contract in
`include/vt/ops.h` and sharing nothing with `src/vt/cpu` — a gate that compared
the kernel against a helper the kernel also uses would prove consistency, not
correctness.

**The dtype matrix is bounded by what the OP accepts, not by what the kernel
could be handed.** `vt::Attention` and `vt::AttentionCross` both require
`IsFloat(query.dtype) && key.dtype == query.dtype && value.dtype == query.dtype`
and `IsOutFloat(out.dtype)`. So there is no mixed-q/k/v row: it would test an
input the production entry point refuses, and an unreachable case that passes is
not evidence. The reachable discriminating axis is `out` disagreeing with the
operands, and every row exercises it.

**This closes a real hole.** Before this row the CPU arm of
`tests/vt/test_ops_attention_cross.cpp` ran f32 operands and nothing else — its
bf16 geometries are CUDA-only — so `AttentionCrossKernel`'s f16 and bf16 element
paths were ungated.

## Gates

1. `tests/vt/test_ops_attention_elem_dispatch.cpp` green, and green on a
   pristine `origin/main` tree too, which is what says the reference is a valid
   oracle rather than a transcription of the new code.
2. The broad CPU `vt` suite green, with the SAME case and assertion counts on
   both trees, so nothing was silently skipped.
3. A mutation per claimed guarantee, each verified to have applied and built
   before its result is read.
4. Byte equality between arms on the measurement binaries, f32 and bf16.
5. Before/after at the SHIPPED thread count, arms interleaved, `n >= 3`, with a
   same-arm control and the box's load stated.
6. `scripts/agent-preflight.sh`.

## Risks/decisions

- **The devbox is not idle and cannot be made idle.** Other agents compile on
  the same 20 cores throughout; loadavg ran 12 to 56. Every comparison is
  therefore INTERLEAVED with a same-arm control leg, so drift lands on all arms
  equally and a gap smaller than the control is not a result.
- **No CMake build tree fits.** `/` reached 1.5 GB free during this row. Every
  binary here is a direct `g++` compile over the `vt` TU set at the project's own
  `-std=c++20 -O2 -ffp-contract=off` and the per-source ISA flags from
  `CMakeLists.txt:1393-1404`. `-ffp-contract=off` is not optional: the
  bit-identity contracts depend on it.
- **`always_inline` on `LoadF32` was measured and NOT taken.** See `## Owed`.
- **This is x86-64 only.** The aarch64 tier has different inlining economics and
  a different vector width. `## Owed` says so.

## Stop conditions

- a speedup that cannot be made bit-exact — report it, never trade correctness;
- ENOSPC;
- a same-arm control that swallows the effect being claimed.
