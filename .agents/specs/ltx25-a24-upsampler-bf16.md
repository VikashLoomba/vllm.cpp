# LTX-2.5 — the LATENT UPSAMPLER's bfloat16 arm (A24, wave 5)

Row: `LTX25-A24-UPSAMPLER-BF16`
Issue: [#2857](https://github.com/mudler/vllm.cpp/issues/2857)
Parent scope: `.agents/specs/ltx25-completion-scope.md` §A.7 (A24), operator-owned
Wave 1: `.agents/specs/ltx25-a24-text-tower-bf16.md` (#2676, merge `8e582a5f9`)
Wave 2: `.agents/specs/ltx25-a24-connector-bf16.md` (#2720, merge `77704c8d0`)
Wave 3: `.agents/specs/ltx25-a24-video-vae-bf16.md` (#2786, merge `c20fb2ba2`)
Wave 4: `.agents/specs/ltx25-a24-leaves-bf16.md` (#2850, merge `d2b1bda2b`) — this row is its `## Owed`
Oracle: `.agents/oracles/ltx-2.md`, `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`
Base: `d2b1bda2b4359f148ff759b5d2d0ad719c9f6d78`

---

## 0. THE ROW IS GATEABLE WITHOUT A CHECKPOINT, AND THAT WAS SETTLED FIRST

Wave 4 handed this row forward with a caveat attached: "**its temporal checkpoint
is not on the NAS**". That sentence is true and it is not a gate blocker, and the
difference decides whether this row exists at all, so it was measured before a
line was written.

**What the missing checkpoint blocks** is a real-weight render of the temporal
arm. `ltx2_upsampler.h:36-39` has recorded that gap since #2580 —
"`ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` ... is not on the
NAS, so no real-weight result exists for that arm". It is a pre-existing hole in
the *f32* arm too. This row neither widens nor closes it.

**What it does not block** is the gate waves 1-4 actually used. Every one of them
executed the pinned upstream module on synthetic fixtures and checked in the
result, and `scripts/gen-ltx2-pipeline-goldens.py` says so in its own header:
"**Needs torch + numpy + einops (CPU only). No checkpoint, venv, or gated
download.**" Its section 8 already constructs `LatentUpsampler(...)` from a
deterministic FNV-1a + splitmix64 stream keyed by parameter name and runs it, so
no weight byte is checked in and none is read.

Five facts, each run at this row's base on this box, not inherited:

```text
git -C ~/_git/LTX-2 rev-parse HEAD    fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca  (the pin)
git -C ~/_git/LTX-2 status --porcelain  empty
python3 -c 'import torch,numpy,einops'  torch 2.11.0+cu130, numpy 2.3.5, einops 0.8.2
LatentUpsampler(...).to(bfloat16)(x)  RAN, both arms: (1,8,3,8,8) and (1,8,5,4,4), dtype bfloat16
ls /usr/local/nas_share                No such file or directory  -- CHECKPOINT_ROOT is dangling here
```

The last line is the honest half. `.env`'s `CHECKPOINT_ROOT` does not resolve on
this box, and nothing was mounted or downloaded to make it. It did not have to
be: the arm is gated the way its four predecessors were, and the production path
is reached by `ltx2_fixture::WriteReducedUpsampler`, which synthesizes the
checkpoint the three call sites load. **No lease was taken and no GPU work was
run.**

### 0.1 §A.7's anchor for this component has rotted, and this row does not fix it

`.agents/specs/ltx25-completion-scope.md:613` cites the upsampler's dtype block
at `ltx2_upsampler.h:66-70`. At `d2b1bda2b` line 66 is the temporal arm's
first-frame-drop bullet and `:68` opens "THE dims=2 ARM"; the DTYPE block is at
**`:108-112`**. §A.7 is operator-owned, so the drift is **recorded here and not
edited there** — the same handling wave 4 gave its own re-scoping finding. It is
also the fifth instance of this tree's standing lesson that a recorded line
anchor goes stale inside the pull request that moves it.

---

## 1. Scope

**IN.**

* `src/vllm/model_executor/models/ltx2_upsampler.cpp` — `Volume` gains a dtype
  and real narrow bytes; `Conv3dPad1`, `Conv2dPad1PerFrame`, `GroupNorm`, `Silu`,
  `BlurDownsample`, `PixelShuffle2d`, `PixelShuffle1d`, `DropFirstFrame`,
  `ResBlockForward`, `Ltx2LatentUpsample` and `Ltx2UpsampleVideoLatent` gain the
  bf16 arm.
* `include/vllm/model_executor/models/ltx2_upsampler.h` — the DTYPE block at
  `:108-112`, whose "f32, because this is the CPU REFERENCE arm" sentence this
  row falsifies.
* `src/vllm/multimodal/ltx2_video.cpp` — the upsampler loader call site asks for
  `kBF16`, and the render path gains the runtime-dtype counters.
* `include/vllm/multimodal/ltx2_video.h` — those trace fields.
* `scripts/gen-ltx2-pipeline-goldens.py` — a new bf16 subsection of section 8.
* `tests/vllm/models/test_ltx2_pipeline.cpp`, `tests/vllm/multimodal/test_ltx2_video.cpp`.

**OUT, and owed by name in `## Owed`.** The **duration head** — zero production
call sites, blocked on #611 and not on arithmetic, exactly as wave 4 found. The
**FP8/NVFP4** arms, which are A22. The **CUDA arm**: this file has no queue at
all and no `vt::` kernel seam, so there is nothing to narrow on a device. The
**real-weight temporal render**, which needs the checkpoint §0 says is absent.
The **`spatial_upsample AND temporal_upsample`** operator, refused by name today
and owed by `.agents/specs/ltx25-upsampler-arms.md`.

---

## 2. Upstream anchors

`Lightricks/LTX-2 @ fd4ded7f`, `packages/ltx-core/src/ltx_core/`:

| what | where |
|---|---|
| the one pipeline dtype | `ltx-pipelines/.../distilled.py:109` |
| the upsampler is constructed with it | `distilled.py:138-141` |
| `LatentUpsampler.forward` | `model/upsampler/model.py:82-126` |
| `ResBlock.forward` | `model/upsampler/res_block.py:29-37` |
| `PixelShuffleND.forward` | `model/upsampler/pixel_shuffle.py:32-54` |
| `BlurDownsample` (kernel is a **registered buffer**) | `model/upsampler/blur_downsample.py:29-33`, `:49-53` |
| `SpatialRationalResampler.forward` | `model/upsampler/spatial_rational_resampler.py:40-47` |
| `upsample_video` | `model/upsampler/model.py:129-143` |
| `PerChannelStatistics` (both buffers, `.to(x)`) | `model/video_vae/ops.py:76-84` |

---

## 3. The six rounding rules, MEASURED

Every row was produced by executing the module at the pin on CPU, torch
2.11.0+cu130, and every row carries the rejected hypothesis and a separating
count. A rule with a zero separating count is reported as such rather than
claimed.

| # | site | upstream's rule | rejected | separating |
|---|---|---|---|---:|
| R1 | `Conv3d`/`Conv2d` | f32 accumulate, round once on store | f64 accumulate | **0** — see §3.1 |
| R2 | `GroupNorm(32)` | f32 statistics, f32 affine, **ONE** rounding | normalize rounded, then affine | 394 |
| R3 | `GroupNorm` eps | stays **f32** (a plain Python attribute, not a buffer) | narrowed to bf16 | **0** at var~1; **227** at var~1e-6 |
| R4 | `SiLU` | f32 opmath, round once | sigmoid narrowed first | 952 |
| R5 | `activation(x + residual)` | the **add rounds to bf16 first** | add kept in f32 | 793 |
| R6 | `BlurDownsample.kernel` | a **registered buffer**, so it narrows | — | **0** at ks=5; 1 at ks=9; 57 at ks=11 |
| R7 | `PerChannelStatistics` | `.to(x)` narrows both buffers; **two** roundings | one rounding / f32 stats | 40 / 38 |

R3, R6 and R1 are the three that separate nothing where the shipped
configuration puts them, and each is handled rather than hidden:

* **R3** cannot be seen while the variance is order 1, because bf16's `1e-5` and
  f32's differ by ~1.3e-8 and that is under an f32 ulp at 1.0. It parts at
  **var ≤ 1e-4** (22 elements), 1e-6 (227), 1e-8 (233). The golden therefore
  carries a **small-variance** GroupNorm case whose only purpose is to make the
  eps width observable, and the generator refuses to emit it if it separates
  nothing. This is wave 4's group-size lesson applied to a different axis.
* **R6** is the trap that reads as a no-op. The buffer **does** narrow — this was
  checked, not assumed — and at the pinned `kernel_size = 5` every entry of the
  normalized binomial kernel is `{1,4,6,16,24,36}/256`, a dyadic rational bf16
  holds exactly, so narrowing changes nothing. ks=9 and ks=11 are the control
  that proves the probe can fail.
* **R1** is invisible **by construction** at a bf16 store: over the shipped
  fan-in the f32-vs-f64 accumulation difference sits far below one bf16 ulp. It
  first parts at `mid_channels = 512`, fan-in 13824, in **1 element of 768**. The
  file's existing `double` accumulators are therefore safe on this arm, which is
  the opposite of the polarity warning that applies to its f32 arm, and it is
  written down here so the next reader does not re-derive it.

### 3.1 A whole-chain bit-exact gate is NOT attainable, and that is a measurement

Wave 4 held bit-exactness and this row cannot, so the reason is stated rather
than the bar quietly lowered.

Fed upstream's own tensors, every rule above reproduces **bit-exactly**: at group
sizes 32, 48, 72, 96, 120 and 144 the C++-shaped GroupNorm formula matches
`torch.nn.GroupNorm` in 0 of up to 4608 elements. What does not reproduce is
upstream's own convolution: `torch.nn.Conv3d` **at bf16** differs from the same
convolution run on f32 inputs in **2 of 3840 elements**, because oneDNN selects a
different blocking for the bf16 path. Two elements is where it starts and not
where it ends — the next `GroupNorm` shifts its whole group's mean and variance,
and at the temporal arm's output that becomes **382 of 720**.

So the gate is split, and neither half is a band over a rule:

* **Bit-exact, no tolerance**: R2, R3, R4, R5, R6, R7 and the movement
  operators, each fed upstream's own bf16 tensor.
* **A measured band**: the convolutions, and therefore the chain. The band is
  asserted **strictly below** the distance of every rejected rule, and those
  distances are emitted as goldens so the assertion is against a number and not
  against a memory.

Measured at the fixture dimensions (`in=6, mid=32, blocks=1`), max |delta| from
upstream's bf16 forward:

| arm | correct chain | R4 wrong | R5 wrong | R2 wrong | R7 wrong |
|---|---:|---:|---:|---:|---:|
| PixelShuffle | **0** | 0.015625 | 0.011719 | 0.013672 | 0.015625 |
| Rational2 | **0** | 0.015625 | 0.011719 | 0.013672 | 0.015625 |
| Rational1p5 | **0** | 0.011719 | 0.0097656 | 0.013672 | 0.010986 |
| Temporal | 0.0078125 | 0.015625 | 0.011719 | 0.016113 | 0.016602 |
| Dims2 | 0.0039062 | 0.016602 | 0.011719 | 0.019531 | 0.015625 |

Three of the five arms are bit-exact even through the chain. The two that are not
are the two whose post-upsample convolution is fed a re-shaped tensor, and their
worst is 0.0078 against a nearest rejected rule at 0.0098 — **1.25x**. That is
thin and it is reported as thin.

### 3.2 The band above was measured on the WRONG parameter draw, and the shipped one is different

§3.1's table came from a `torch.manual_seed` draw. The committed generator draws
every parameter from a deterministic FNV-1a stream keyed by NAME, and re-measured
on THAT draw the same table is materially worse: the `PixelShuffle` arm's correct
chain sits at 0.00390625 and its `gn_bf16_eps` alternative sits at **exactly
0.00390625**, so the arm separates that rule not at all. The `Dims2` arm
separates **none of the four**.

**The generator refused to emit rather than ship that**, which is what its refusal
is for, and the refusal is quoted here because it is the load-bearing event of
this section:

```text
ValueError: upsampler bf16 arm PixelShuffle: the correct chain is 0.00390625 from
upstream and the nearest REJECTED rule is 0.00390625. No band can separate them,
so this arm cannot gate its own rules and must not be emitted.
```

Two things follow, and both are in the shipped design:

1. **Coverage is per RULE across arms, not per arm.** No single arm sees all
   four, and requiring that of each was the wrong assertion. What must hold is
   that every rule is separated by at least one arm, and the generator refuses
   when it is not. Emitted coverage at the committed fixture: R4 by 5 arms of 6,
   R5 by 3, R2 by 5, R3 by **2**.
2. **`SmallVar` exists for R3 and its scale was swept, not chosen.** R3 needs a
   small GroupNorm variance, and that is exactly the regime where the chain's
   amplification is chaotic. Measured on the committed draw:

   | latent scale | correct chain | nearest rejected | all four separate |
   |---|---:|---:|---|
   | 1.0 | 0.00390625 | 0.00390625 | no (R5, R3 invisible) |
   | 0.2 | 0.00390625 | 0.00390625 | no (R4 invisible) |
   | 0.05 | 0.00390625 | 0.017578125 | yes |
   | **0.02** | **0.0** | **0.015625** | **yes** |
   | 0.01 | 0.01171875 | 0.01171875 | no (R4 invisible) |
   | 0.002 | 0.0 | 0.0126953125 | yes |

   0.02 is the largest scale at which the chain is **bit-exact** and all four
   rules separate. Neighbouring scales failing is not a fragility that can ship:
   the generator refuses an arm whose rules do not separate, so a pin that moved
   this reds at generation instead of quietly widening a band.

The rules do not, in the end, depend on a thin band anywhere: `SmallVar` gates all
four from a bit-exact chain, and `Temporal` gates all four independently.

**The first version of this table was a mute switch and the second is not.** Run
with `LatentUpsampler`'s default GroupNorm initialisation — `weight = 1`,
`bias = 0` — the R2 column read 0.00097656 against a correct chain of 0.00097656,
i.e. it separated nothing, because `n * 1 + 0` makes a double rounding and a
single rounding the same expression. Randomising the affine parameters, which is
what the committed generator's parameter stream already does, moved that column
to 0.0137. The number recorded is the second one.

---

## 4. Design

**Storage, not a flag.** `Volume` stops being a `std::vector<float>` and becomes a
byte buffer plus a `vt::DType`, read and written through the `LoadElem` /
`StoreElem` pair `ltx2_video_vae.cpp:200-211` already uses. A bf16 volume then
*is* half the bytes rather than an f32 buffer holding narrowed values, which is
the wave 3 rule and the one AGENTS.md's "a token gate cannot detect a dtype that
is too wide" exists for.

**The dtype comes from the weight bag, exactly as waves 2-4 route it.**
`Ltx2VaeWeights` already carries `bf16` and `dtype` (`ltx2_audio_vae.h:71-85`).
`Ltx2LatentUpsample` and `Ltx2UpsampleVideoLatent` keep their public
`Ltx2LatentVolume` signature — a latent is an interface value and stays f32 at
the seam, as `Ltx2ConvVideoEncode`'s does — and round on entry, because upstream's
latent at that point came out of a bf16 DiT and is already bf16.

**Two widths, refused in one place.** A third width arrives by refusal, not by
silence, mirroring `RequireVaeDType`.

---

## 5. Risks

1. **The thin band on two arms** (§3.1). Mitigated by the per-rule bit-exact
   cases; if the band ever has to widen past a rejected rule's distance the gate
   is dead and must be replaced, not relaxed.
2. **The probe trap that has caught four sessions.** Reading a parameter after
   `.to(bfloat16)` narrows it in place and yields a false 0/0. Every probe here
   captures parameters BEFORE any cast, and the generator refuses to emit a
   non-separating probe.
3. **`double` accumulators.** Safe on this arm (§3 R1) and NOT safe on the f32
   arm, where `ltx2_upsampler.cpp:22-30` already records the pointwise escape as
   visible debt. This row does not touch the f32 arm's escape.
4. **Regenerating a committed golden file.** #2855 moved 4970 lines by changing
   the thread count. The bf16 subsection is a pure ADDITION and the diff is
   checked for that.

---

## 6. Tests

1. **Red first, through a production entry point.** `Ltx2UpsampleVideoLatent` on
   a bf16 bag, reached from the render path, refuses today; the case asserts the
   bf16 output and reds on the refusal string.
2. Per-rule bit-exact cases R2-R7 against the new goldens, each asserting the
   rejected hypothesis is DIFFERENT (`separating > 0`) as well as that upstream's
   is equal.
3. The five chain arms against the band, with the rejected distances asserted
   above it.
4. **The runtime dtype on a production path**: a render counter that reports the
   width the upsampler actually computed at, checked non-zero-and-bf16.
5. **Mutations**, each restored byte-for-byte: delete the production call site,
   and flip each of R2/R4/R5/R7 in a scratch copy.

## 7. Gates

```sh
# G0 — oracle identity. Nothing in §3 holds if this is another revision.
git -C ~/_git/LTX-2 rev-parse HEAD              # fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca
git -C ~/_git/LTX-2 status --porcelain          # empty

# G1 — the THREE production call sites, counted by string. A drop to 0 is the
#      reachability failure this row's mutation is written for.
grep -c 'Ltx2UpsampleVideoLatent(' src/vllm/multimodal/ltx2_video.cpp   # was 3

# G2 — the f32 assumption this row removes. Read the DIFF; a count that falls
#      with no bf16 branch added is a deletion, not a port.
grep -c 'std::vector<float>' src/vllm/model_executor/models/ltx2_upsampler.cpp  # was 15
grep -c 'vt::' src/vllm/model_executor/models/ltx2_upsampler.cpp               # was 0

# G3 — the goldens are upstream's, regenerated at the pin, and a pure ADDITION.
python3 scripts/gen-ltx2-pipeline-goldens.py --ltx2 ~/_git/LTX-2 \
  --vllm-omni ~/_git/vllm-omni --out tests/vllm/models/ltx2_pipeline_goldens.inc
git diff --stat -- tests/vllm/models/ltx2_pipeline_goldens.inc
git diff --numstat -- tests/vllm/models/ltx2_pipeline_goldens.inc   # deletions must be 0

# G4 — the focused suites.
ctest --test-dir build -R 'ltx2' --output-on-failure

# G5 — the full gate.
scripts/agent-preflight.sh --staged
python3 scripts/check-pr-size.py --base origin/main --head HEAD
python3 scripts/agent-pr-body.py --pr <N>
```

## 8. Evidence

1. The §0 gateability commands, literal.
2. The seven probes of §3 with their separating counts, including the three that
   separate nothing and the control that proves each of those is not blind.
3. The §3.1 table, in both its mute-switch and its corrected form.
4. The literal red of §6.1 and the literal green after, in doctest's own output.
5. Every mutation with its literal assertion count, restored and re-verified.

## 9. Stop conditions

* **A GPU lease, a mount, or a download is needed.** Stop and report. §0 settled
  that none is; if that turns out wrong the row's premise is wrong.
* **An arm cannot be resolved by measurement.** Refuse it BY NAME with a message
  naming the missing part and record it owed. Never guess an arithmetic rule and
  never fit a tolerance to this port's own output.
* **The band of §3.1 would have to reach a rejected rule's distance.** The gate
  is then dead; replace it, do not relax it.
* **A probe stops separating.** Rebuild it; never emit a golden that cannot fail.

## Owed

* **The duration head's bf16 arm.** `Ltx2DurationPredict` still has zero
  production call sites, so it can only be gated by a unit test constructing the
  type by hand. Wiring is [#611](https://github.com/mudler/vllm.cpp/issues/611);
  the dtype follows the wiring and not the reverse. Unchanged from wave 4's §0.3.
* **The real-weight temporal render.** Blocked on
  `ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors`, absent from the NAS
  since #2580. This row's temporal arm is gated against the executed module only.
* **The FP8 and NVFP4 arms** — A22.
* **A CUDA arm.** This file has no queue and no `vt::` kernel seam, so a device
  arm is a residency row and not a dtype row.
* **R6 is unmeasurable by any value gate at the pinned kernel width.** Mutation
  M10 removes the blur kernel's narrowing and 79 of 79 assertions stay green,
  because at `kernel_size = 5` the narrowing moves no entry. The rule is
  implemented and correct; what is owed is a gate that can see it, and the only
  honest one is a fixture at a kernel size where the narrowing is lossy — which
  upstream never constructs, since `SpatialRationalResampler` never overrides the
  default. Tracked by [#2857](https://github.com/mudler/vllm.cpp/issues/2857) and
  recorded here rather than left for a reviewer to find.
* **§A.7's rotted anchor** for this component (`:66-70`, should be `:108-112`).
  Operator-owned; recorded in §0.1, not edited.

## Outcome

### The mutations, and the two that did NOT red

Every rule was mutated in a scratch copy, rebuilt, run, and the tree restored
byte-for-byte. The harness refuses to read a build failure or an unapplied patch
as a pass, and it caught one of each.

| mutation | assertions | result |
|---|---|---|
| M2 — R4, sigmoid narrowed before the multiply | 79, **5 failed** | RED |
| M3 — R5, residual add kept at compute width | 79, **3 failed** | RED |
| M4 — R2, normalized value rounded before the affine | 79, **5 failed** | RED |
| M5 — R3, epsilon narrowed to bf16 | 79, **2 failed** | RED |
| M6b — R7(b), the two roundings fused into one | 13, **1 failed** | RED |
| M7 — R7(a), the statistics not narrowed | 13, **4 failed** | RED |
| M8 — the three production call sites deleted | 4, **1 failed** | RED |
| M9 — the loader reverted to f32 | 7, **2 failed** | RED |
| **M6 — R7(b) fused, against the FIRST version of its case** | 4, **0 failed** | **GREEN** |
| **M10 — R6, the blur kernel not narrowed** | 79, **0 failed** | **GREEN** |

**M5 reds on exactly 2 arms, which is what `kLtx2UpsBf16RuleCoverage` predicted
for R3.** The coverage number is not decoration: it named the blast radius before
the mutation was run.

**M6 is the failure this campaign has shipped once per wave, caught here.** The
first version of the `upsample_video` case asserted only that the result reported
bf16 and carried bf16-representable values. A fused rounding produces a bf16 value
too, so the mutation passed with 4 of 4 assertions green — a claimed guarantee
that nothing measured. The case now compares against a golden over the whole
function, `upsample_video`'s own body run on upstream's modules, and the same
mutation reds. The isolated `un_normalize` tensors the generator emits are
evidence; the whole-function golden is the gate.

**M10 is a limit, not a defect, and it stays open.** At the pinned
`kernel_size = 5` every entry of the normalized binomial kernel is a dyadic
rational bf16 holds exactly, so narrowing the registered buffer changes no value
and **no value gate can see whether the port does it**. The narrowing is still
correct and still there, because it is what upstream does; what holds it down is
`kLtx2UpsBf16BlurNarrowedEntries` — 0, 0, 0, 1, 57 at kernel sizes 3, 5, 7, 9, 11
— which proves the site is live and the zero at the shipped width is a
measurement rather than a blind spot. Recorded under `## Owed`.

### The spec file was committed EMPTY, and the diffstat is what caught it

Commit `8ad7c2e94` wrote this file to zero bytes. The edit script called
`open(path, "w").write(open(path).read().replace(...))`, and `open(path, "w")`
truncates before the inner read is evaluated, so it wrote an empty string's
replacement into an emptied file. Nothing failed: the commit succeeded, the tests
stayed green, and `git diff --stat` reported the spec as ` | 0`. A row whose spec
is a gate requirement had no spec for one commit. Written down because the
mechanism is not specific to this row and the symptom is a diffstat entry nobody
reads.

### The recorded reader anchors went stale INSIDE this pull request

`ltx2_video.cpp` carries a READER ANCHORS list that `test_ltx2_video` derives and
compares. The counters and loader comments this row adds shifted every one of the
fourteen by 11 to 18 lines, and the case failed with both lists printed. Updated
to the derived values. This is the fifth recorded instance in this tree of a line
anchor going stale within the change that moved it, and the reason the case exists
is that it is the only thing that notices.

### The two ltx2 suites that are red, and only one is this row's

`ctest -R ltx2` after the fix: **13 of 13 targets built, 12 passed**, and the one
failure is `test_ltx2_video_device_forward`, which is
[#2853](https://github.com/mudler/vllm.cpp/issues/2853) — **open, pre-existing,
and wave 3's**. Its refusal string ("a bf16 decode was requested on device
'xpu'") is present at `origin/main`, this branch touches no
`ltx2_video_vae` file, and wave 3's spec already lists it under `## Owed`.
Checked rather than assumed, because "a red that was already red" is exactly the
claim a row should not be trusted to make about itself.

### What the port turned out to be

**Bit-exact on every arm**, which §3.1 predicted it could not be. That section
measured the gap between upstream's bf16 convolution and the same convolution on
f32 inputs and concluded a band was needed; the port's own `double` accumulators
close it, because at a bf16 store the accumulation order sits below one ulp.
Three of the six emitted bands are `0.0` and the port meets them. The band
machinery is kept anyway: it is what the `Separates` assertion compares against,
and a fixture at a wider `mid_channels` may need it.

### Why each default has the value it has

* **The arm comes from the weight bag, not from a parameter.** A caller that
  could pick a width the checkpoint is not stored at would reinterpret the
  parameter bytes rather than refuse.
* **`Volume` holds bytes, not narrowed floats.** An f32 buffer of bf16 values
  passes every value gate and moves twice the memory, which is the polarity
  AGENTS.md says a token gate cannot see.
* **`WeightView` is a view and not a widened copy.** Widening on load would put
  the bf16 arm back at the f32 arm's bytes, which is the whole thing A24 removes.
* **`SmallVar`'s scale is 2e-2** — the largest scale at which the chain is
  bit-exact and all four rules separate (§3.2's sweep).

## Now

`ACTIVE`.
