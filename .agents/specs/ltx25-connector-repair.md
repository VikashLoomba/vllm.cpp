# SPEC — `LTX25-CONNECTOR-REPAIR`: re-split the connector leaf after the attention hoist, then repair what the new split blames

Issue: [#2434](https://github.com/mudler/vllm.cpp/issues/2434).
Owner row: `LTX25-CONNECTOR-REPAIR`.
Predecessors: [#2296](https://github.com/mudler/vllm.cpp/issues/2296) (the 5.53x
render reading), [#2354](https://github.com/mudler/vllm.cpp/issues/2354) (the
weights/compute split), `LTX25-CONNECTOR-GEMM` (the leaf decomposition), and
[#2376](https://github.com/mudler/vllm.cpp/issues/2376) /
[#2416](https://github.com/mudler/vllm.cpp/issues/2416) (the dtype hoist that
invalidated it).

## The gap, and why it is not the one the predecessor left

`.agents/specs/ltx25-render-speed-parity.md`'s first `## Owed` line is still
open in the exact words it was written in: **"The repair is not here."**
`conditioning.connector.compute` + `guiders.connector.compute` = 224.882 s,
**43.52% of a 516.751 s render**, at spreads of 1.70% and 2.07% — the two
quietest rows in that table, against a `denoise` of 2.92%.

`LTX25-CONNECTOR-GEMM` decomposed that leaf on GB10 and blamed
`vt::AttentionCross`: attention 66.5%, GEMM 24.8%, residue 8.7%. **That is no
longer the tree.** `VT-CPU-ELEM-DISPATCH` hoisted the per-element dtype dispatch
out of `AttentionCrossKernel` and measured **11.30x / 12.12x on GB10** at those
same shapes, byte-identical. A repair chosen against the old split would be
chosen against a leg that has already been repaired.

**So this row re-measures before it repairs, and the re-measurement is the first
deliverable rather than a preamble to one.** If the hoist already took most of
the 224.9 s, saying so closes the parent row's owed line and is the best outcome
available.

## Scope

IN scope:

- **W1** — the connector leaf re-measured on current `main`: `Ltx2ConnectorForward`
  in total, `vt::AttentionCross` alone at the same shape, and the connector's
  whole GEMM set, `n >= 3`, interleaved, with the box's load stated per leg. On
  x86-64 here and on GB10 under a lease, because GB10 is the machine the render
  was measured on and because a claim in this campaign has already inverted
  between the two ISAs.
- **W2** — the repair the new split blames, bit-exact, with a red-first
  byte-equality gate and a mutation per claimed guarantee.
- **W3** — the before/after with spread, a same-arm control leg beside it, and
  what it does and does not say about the render.

OUT of scope, declared rather than approximated:

- **A device arm for the connector.** `Ltx2Attention`
  (`src/vllm/model_executor/models/ltx2.cpp`) interleaves host `RmsNormRows` and
  `Ltx2ApplyRotaryEmb` on raw `float*` between its GEMMs, and
  `Ltx2ConnectorForward` reads every weight through `View(...)` over a host
  `std::vector<float>`. That is a weight-arm port with its own numerics gate, not
  a queue swap. #2354 says so, `LTX25-CONNECTOR-GEMM` repeats it, and this row
  does not contradict either. If the measurement points there, this row returns
  the sizing and stops.
- **A published benchmark ID.** `docs/BENCHMARKS.md` gains nothing. One module's
  GEMM shapes on one request is an instrument, not a benchmark.
- **Weakening any numeric contract.** Every kernel in `cpu_matmul_elem.cpp` keeps
  each output's K reduction strictly sequential, which is what makes the SIMD
  tiers bit-identical to the scalar reference. Byte equality is the gate here and
  a speedup that cannot meet it is reported, never taken.
- **A general tiled sgemm.** `.agents/specs/cpu-elem-gemm-wide-isa-and-tiling.md`
  carries `KERNEL-GEMM-CPU-TILED` as a SPIKE with no active row. This row does
  not open it. What it may land is the contained change its own measurement
  prices, and what it may not do is turn into that campaign.

## What the instrument is, and why no new one is written

`tools/bench/ltx2_connector_gemm_probe.cpp` already runs `Ltx2ConnectorForward`
at the shipped geometry (`--mode connector`), `vt::AttentionCross` alone at the
connector's own shape (`--mode attn`), the connector's whole GEMM set
(`--mode gemm`), and the runtime tier resolver (`--mode tier`). It is on `main`,
its shapes are read out of the pinned DiT's `__metadata__.config`, and it was
the instrument the old split was taken with. **Re-measuring with the same
instrument is what makes the before and the after comparable at all**, so this
row adds no probe and changes none of its shapes.

**The render harness pins the binary's sha256 by design and therefore refuses a
rebuild** (`scripts/ltx25-render-speed-repeat.sh`, exit 51). That is correct for
what it measures — a timing of the tree that took the correctness verdict — and
it is why the before/after here is taken on the probe rather than by re-running
that harness against a changed binary. An end-to-end render of a changed arm is
`scripts/ltx25-text-cond-ab.sh`'s job: it builds two arms into SEPARATE build
directories, inverts its identity assertion so it cannot measure one binary
twice, and requires `pixel_files_differing=0`.

## The measured position, before any change

Every number in this section is x86-64 AVX-512 on a devbox at the load stated
beside it. The GB10 half is in `## Outcome`.

**The hoist landed and the attention leg is gone.** `--mode attn` on current
`main` reads the probe's own hoisted reference at **1.06x** (video) and **1.02x**
(audio) against the shipped kernel, where before the hoist the same comparison
read 13.30x and 9.58x here and 11.30x / 12.12x on GB10. The instrument that
found the defect now finds no defect, which is the control this row wanted.

## Design of the repair, stated as a mechanism rather than a patch

`MatmulOneChunk` (`src/vt/cpu/cpu_ops.cpp`) widens a **16-row** activation tile
(`blck_1`, ggml's `mul_mat` chunk tile) and then walks it in blocks of `mr`
activation rows, where `mr` is the tier's M-blocking factor: **6 on AVX-512, 4 on
NEON, 2 on SSE2**. Each `mr` block costs ONE pass over the 16-column weight block
— the load and, on the `[N,K]` orientation, the 4x4-group register transpose that
`Transpose16` performs — and that pass is what the M blocking exists to amortize.

**`mr = 6` does not divide 16.** Rows 0-11 take two M-blocked calls; rows 12-15
fall through to the `mr = 1` path and take **one whole weight pass each**. A
16-row tile therefore performs **6 weight passes instead of the 3 its own
blocking factor allows**, and the four tail rows pay a full transpose apiece.
`kMrNeon = 4` divides 16 exactly, so this specific defect is **AVX-512 only** —
which is stated here because the machine the render was measured on is aarch64
and a reader must not carry the x86 figure onto it.

Two changes, each bit-exact by the same argument the tier already rests on:

1. **Pad the widened tile up to a whole number of `mr` blocks and zero the pad
   rows**, so every activation row goes through the M-blocked kernel. The pad
   rows' outputs are computed and never stored. No output's K reduction order
   moves, because `ElemBtMFn` accumulates lane `l` of row `r` over `p` in strict
   increasing order whatever `mr` is — which is exactly why the tree already
   runs some rows of the same call through `btm` and others through `bt` and
   asserts both byte-identical to `MatmulOneChunkRef`.
2. **Widen the AVX-512 tier's `mr` from 6 to 8.** `BtM6Avx512` holds 16
   transposed weight vectors plus `mr` accumulators plus one broadcast; at
   `mr = 8` that is 25 of 32 ZMM. 8 divides 16, so a full tile becomes **two**
   weight passes.

Both are `vt` shared-seam changes and every CPU f32/f16/bf16 GEMM in the tree
runs through them, which is why the gate below is the whole dtype matrix at more
than one worker count and not the connector's own shape.

## Reachability

Nothing new lands unreached and no new entry point is added. `MatmulOneChunk` is
the body of `vt::MatmulBT` / `vt::Matmul`, reached from `ModelRegistry::Forward`
through every CPU dense projection in the tree, and from `Ltx2ConnectorForward`
through `Linear` on the render's default path. The gate enters through
`vt::MatmulBT`, never by calling the kernel, and the mutation below deletes the
padding so the gate reds through that production entry point.

## Tests to port

There is no upstream test. Upstream's connector is a `torch.nn.Module` and its
GEMM is cuBLAS; ggml's own `mul_mat` has no bit-identity contract to port because
ggml reassociates the K reduction and we deliberately do not
(`src/vt/cpu/cpu_matmul_elem.h`, RECORDED DEVIATION). The tests are this tree's
own and each is an executable observable.

| ID | Assertion |
|---|---|
| T1 | `vt::MatmulBT` is `memcmp`-identical to `MatmulOneChunkRef` at M values that are NOT multiples of `mr` — the rows the padding moves onto a different micro-kernel — over the whole `{f32, f16, bf16}` x `{f32, f16, bf16}` operand matrix |
| T2 | the same at ragged N and ragged K, so the column tail and the K tail are exercised together with the padded M tail |
| T3 | every one of the above is `memcmp`-identical at SEVEN worker counts, so a claim about which thread computes which output is tested against more than one partition |

`tests/vt/test_ops_matmul_elem_mblock.cpp`. The reference arm is
`VT_CPU_MATMUL_TIER=ref`, which is `MatmulOneChunkRef` — the per-element scalar
loop the whole tier hierarchy is defined against — so the oracle shares no code
with the kernel under test.

**ONE WORKER COUNT CANNOT TEST THIS.** The padding changes which rows enter
`btm` and therefore which rows are grouped into one accumulator set. The rows a
chunk covers are set by the threadpool's partition, so a run at one worker count
exercises one partition. `VT-CPU-ELEM-SURVEY`'s M9 is the precedent this rule was
paid for: a deliberately broken kernel passed 753,300 assertions at
`VLLM_CPP_CPU_THREADS=1` and only the worker-count case caught it.

**The operands must be separate enough that the assertion CAN fail.** The same
row's M2 survived its first gate because two operands shared one scale and their
sums fit in 8 significant bits, which bf16 holds exactly. The fixture therefore
fills A and B from independent streams at different magnitudes.

## Gates

1. `tests/vt/test_ops_matmul_elem_mblock.cpp` green, and green on a pristine base
   tree too, which is what says the reference is a valid oracle rather than a
   transcription of the new code.
2. `tests/vt/test_ops_matmul_elem.cpp` green with the SAME case and assertion
   counts on both trees, so nothing was silently skipped.
3. A mutation per claimed guarantee, each verified to have APPLIED and BUILT
   before its result is read.
4. Before/after on `tools/bench/ltx2_connector_gemm_probe.cpp`, arms interleaved,
   two SEPARATE build directories whose binaries are asserted to differ,
   `n >= 3`, a same-arm control leg, and the box's load stated per leg.
5. `scripts/check-tree-compiles.py` and `scripts/agent-preflight.sh`.

## Risks/decisions

- **The devbox is never idle and the load moves DURING a run.** Loadavg ran 25 to
  48 across this row's measurements. Every comparison is interleaved with a
  same-arm control leg and the control's own spread is quoted beside every ratio.
  A gap inside its control is not a result.
- **Contention reweights rather than merely adding noise.** It inflates
  multi-threaded legs relative to single-threaded ones, which is how
  `LTX25-CONNECTOR-GEMM`'s contended decomposition closed to 1.7% when the idle
  one closed to 12.1%. Where this row states a share it names the load it was
  taken at.
- **A claim can invert between architectures**, and in this campaign one already
  has. `mr = 6` is the AVX-512 tier's; `mr = 4` is NEON's and divides 16. The x86
  figure is not a GB10 figure and is never quoted as one.
- **No ceiling is declared.** Where the measurement stops, the next traceable
  hypothesis is named instead.

## Stop conditions

Stop and report, do not work around:

- a speedup that cannot be made bit-exact;
- a same-arm control that swallows the effect being claimed;
- an unhealthy or unreachable fleet device;
- a device port turning out to be the answer — that is `NEEDS_DECISION` with the
  sizing, not a thing to start inside this row;
- ENOSPC. `/` is at 93% and a full CMake build writes 9.4 GiB.

## Work breakdown

- **W1** — this spec and the re-measurement.
- **W2** — the repair and its gates.
- **W3** — the before/after and `## Outcome`.

## Now

`ACTIVE`. W1 is taken on x86-64; the GB10 half is under a lease.

## Outcome

To be written when W3 lands. Nothing above this line may be read as a result.
