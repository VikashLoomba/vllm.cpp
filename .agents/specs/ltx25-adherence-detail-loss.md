# SPEC — `LTX25-ADHERENCE-DETAIL-LOSS`: does the smoothness CAUSE the adherence gap?

Issue: [#2513](https://github.com/mudler/vllm.cpp/issues/2513).
Owner row: `LTX25-ADHERENCE-DETAIL-LOSS`.
Related: [#1854](https://github.com/mudler/vllm.cpp/issues/1854),
[#2295](https://github.com/mudler/vllm.cpp/issues/2295).

## Scope

`LTX25-PROMPT-ADHERENCE` W3 measured our render and it FAILS the adherence
bound. `.agents/specs/ltx25-prompt-adherence.md` `## Owed` says two things this
row exists for: **nothing owns closing that gap**, and the only hypothesis on
offer — that the render's smoothness costs it the fox-against-wolf margin — is
**untested, with the ablation that would test it specified nowhere**.

This row specifies and runs that measurement. It DIAGNOSES. It does not repair,
and it does not touch `src/` or `include/`. A repair is a separate row that this
diagnosis has to justify first, and if the diagnosis falsifies the hypothesis
the repair that would have been written is the wrong one.

In scope:

- the four trivial tone explanations, tested BEFORE anything about frequency;
- a radially averaged power spectrum per frame on both renders;
- the per-frame correlation between high-frequency energy and CLIP score;
- the per-frame table, and whether the frames that clear the bound are the
  sharp ones.

Out of scope: any change to the engine; any change to the gate or its bound; any
new render; any GPU work. Both frame sets already exist.

## Inputs, and what makes them the right ones

| Set | Location | Identity |
|---|---|---|
| ours | `/mnt/nas_share/rc/ltx25-render-confirm/run/20260901T075837Z/r1/frame_*.ppm` | `sha256sum frame_*.ppm \| sha256sum` = `1166b28694001c52a6b5258804f1bb8f97ea2834dac5f16b5a9f5b48469d93ae`, which is the digest `ltx25-prompt-adherence.md` records for the frames W3 scored |
| reference | `/mnt/nas_share/rc/ltx2-oracle/out/upstream_frames/frame_*.ppm` | 25 digests in `tests/parity/goldens/ltx2_oracle/SHA256SUMS` |

Both are verified before a pixel is read. This repository has gated against the
wrong checkpoint before, and the share is CIFS and `soft`-mounted, so a
streaming read can fail mid-file: the frames are copied to local disk and hashed
there rather than scored across the mount.

Geometry `320x192`, 25 frames, `steps=8`, `seed=42`, prompt
`A red fox walks slowly through a snowy pine forest at sunrise, cinematic.`

The engine that produced our frames is `main` at `7905607af`. No file under
`src/` or `include/` whose PATH matches `ltx|video|vae|diffus` changed between
that commit and today's `main`. That is a path-name scan and not a call-graph
one, so it does not exclude a shared primitive moving under the LTX path; it is
enough to say the reading is not obviously stale, and not enough to say today's
engine would score the same.

## Design

### The order is the design

**The trivial explanations run FIRST.** A global gain, a per-channel level
difference, a gamma difference or a compressed dynamic range would depress
sharpness, clipped fraction and CLIP score together **without one detail being
lost**, and the repair for that is not the repair for a lost band. Testing it
last is how a frequency story gets written about a tone difference.

Measured per frame on both sets: mean and per-channel mean, standard deviation,
5th/95th percentiles, min and max, and the best-fit gamma that maps our
luminance histogram onto the reference's. A tone difference is admitted as the
explanation only if correcting it also removes the sharpness and CLIP gaps.

### The spectrum, because "smoother" is a word

A radially averaged power spectrum per frame, on luminance, with a separable
window so the frame's own edges do not manufacture the high-frequency energy
being measured. The frames are `320x192`, so the radial bins run to the Nyquist
of the shorter axis and the two renders share a grid exactly (identical
geometry, so no resampling enters the comparison).

Reported: the per-bin ratio ours/reference, the crossover frequency at which
that ratio leaves unity, and the fraction of total energy we hold above it
against the fraction the reference holds. A uniform ratio across all bins is a
SCALE difference and belongs to the section above, not to this one. That
distinction is the reason both are measured in one run.

### The correlation, which is the actual test

The hypothesis says our frames score low BECAUSE they carry less fine detail.
That predicts a positive association between a frame's high-frequency energy and
its CLIP score against the true prompt. Computed three ways:

- within our 25 frames (n = 25);
- within the reference's 25 frames (n = 25);
- across all 50, where the between-render difference dominates and the
  coefficient is therefore not independent evidence — it is reported for
  completeness and read with that caveat.

The CLIP scores come from the landed instrument itself. The tool's own
`ClipAdherenceScorer` and `discrimination` are imported from
`scripts/ltx25-render-compare.py` rather than reimplemented, so the per-frame
numbers are the same instrument that produced the verdict, after the same
`assert_scorer_identity` check against the committed pin. The tool publishes
per-prompt reductions and not the per-frame matrix, which is why this row calls
the scorer directly instead of parsing a log.

**A correlation here is a hypothesis test, not a gate.** The coefficient, its n
and its null are stated together, and nothing in this row converts one into a
threshold. **If the within-render correlations are absent or point the wrong
way, the smoothness hypothesis is FALSIFIED**, and that is this row's most
valuable available result, because it redirects a repair nobody has started.

## Risks

- **n = 1 on our side.** `scripts/ltx25-render-confirm.sh:551` deletes renders 2
  and 3 by design, so no run-to-run spread of our own render exists in pixels.
  Every statement about our render says n = 1. No single-render statistic is
  presented as a between-render one, which is the error
  `ltx25-prompt-adherence.md` corrected once already when a mean gap was divided
  by a within-render dispersion and called sigma.
- **Windowing manufactures or destroys high-frequency energy.** The window is
  applied identically to both sets, and the unwindowed spectra are computed too,
  so the conclusion cannot rest on the window choice.
- **A correlation over 25 frames of one continuous shot is not 25 independent
  samples.** Adjacent frames of a video are strongly dependent, so the effective
  n is smaller than 25 and any p-value computed against an i.i.d. null is
  optimistic. The row states the coefficient and this limit rather than a
  significance claim.
- **CLIP resizes a `320x192` frame to a `224x224` centre crop.** So the scorer
  never sees the frame's own Nyquist, and a band the spectrum finds may be one
  the instrument cannot see. This is a real limit on how far the mechanism can
  be traced and it is stated with the result.

## Tests and evidence

The measurement is a script under `scripts/`, run on CPU on the devbox, with its
full JSON committed under `tests/parity/goldens/` beside the frames' digests so
a reader can recompute any number in the report. The script verifies both
digest sets before it reads a pixel and refuses rather than scoring a set it
cannot identify.

## Gates

None. This row adds no gate and moves no bound. The adherence gate belongs to
`LTX25-PROMPT-ADHERENCE` and stays exactly as it landed. **Fitting our output to
raise a gate score is what AGENTS.md forbids and what this row must not become**,
which is the reason the row is scoped to diagnosis with the repair explicitly
elsewhere.

## Stop conditions

- `NEEDS_DECISION` rather than modifying the engine.
- `NEEDS_CONTEXT` if either frame set fails its digests.
- A falsified hypothesis is a RESULT and not a stop condition. It gets recorded
  as the finding, in full, and the row does not go looking for a second story to
  tell instead.

## Now

`ACTIVE`.

## Outcome

Measured on 2026-09-01 on the devbox: CPU only, no GPU, no lease, both frame
sets already on disk. `scripts/ltx25-adherence-detail-loss.py` takes the whole
reading and its full JSON is at
`tests/parity/goldens/ltx25_detail_loss/detail-loss.json`, so every number below
can be recomputed rather than trusted.

### THE HYPOTHESIS IS FALSIFIED, and its premise is false as well

The hypothesis was that our render is smoother than upstream's and that the lost
fine detail is what costs it the fox-against-wolf margin. Three independent
measurements refute it, and they refute it in the same direction.

**Our render is not spectrally smoother. It carries MORE fine-scale energy than
the reference, not less.** The radially averaged power spectrum has no crossover
at all: there is no frequency above which our profile stays below the
reference's. The ratio ours/reference rises with frequency instead of falling.

| band | ours / reference |
|---|---|
| f = 0.0039 c/px (the lowest populated bin) | **1.1927** |
| f = 0.0898 c/px | **0.6076**, the minimum over all 64 bins |
| f = 0.15 c/px | 1.4074 |
| f = 0.25 c/px | 1.4834 |
| f = 0.3945 c/px | **2.8582**, the maximum |

That is not a blur. A blur is a monotone rolloff and this is a **band-shaped**
difference: more energy at the very bottom, a **hole in the middle**, and a large
excess near Nyquist. As energy shares of the frame,

| band | ours | reference | difference |
|---|---|---|---|
| mid, 0.04 to 0.14 c/px | 0.081937 | 0.089686 | **-8.64%** |
| high, 0.20 c/px and above | 0.081641 | 0.045918 | **+77.79%** |

The mid band is the one that carries object-scale structure at 320x192 -- 7 to
25 pixel periods, which is the scale of a fox's markings in this frame. We are
short there by 8.6% and long at fine scale by 78%.

**Sharpness was measuring the mid band, and "less sharp" was the right number
read as the wrong thing.** `sharpness_mean` is a mean absolute gradient, which
is dominated by mid-frequency contrast, so a mid-band deficit lowers it even
while the high band is nearly doubled. The reading in
`ltx25-oracle-absolute.md` was correct; the inference "smoother" that was drawn
from it was not, and no spectrum had ever been taken.

### THE CORRELATION POINTS THE WRONG WAY, in every form

The hypothesis predicts that within our render, the frames with more fine detail
score better. They score **worse**.

| population | n | r(high-frequency share, CLIP) | rho |
|---|---|---|---|
| within OURS | 25 | **-0.5862** | -0.5723 |
| within the REFERENCE | 25 | **+0.3943** | +0.3938 |
| pooled | 50 | -0.8122 | -0.7582 |

The null is no association, r = 0. **No p-value is quoted**: 25 frames of one
continuous shot are not 25 independent samples, so the effective n is below 25
and any i.i.d. test would be optimistic. The pooled coefficient is reported for
completeness and is **not** independent evidence, because the between-render
difference dominates that pool. None of these is a gate and none of them is
compared against a threshold.

The **sign flip between the two renders** is the finding, not the magnitude. In
upstream's render more fine-scale energy goes with a better score, which is what
detail does. In ours it goes with a worse one, which is what noise does.

Against sharpness directly the coefficients are `-0.4450` for ours and `-0.1889`
for the reference; against the mid band, `-0.3690` and `+0.1715`.

### THE PER-FRAME TABLE INVERTS THE PREDICTION

Six of our 25 frames clear the bound of 35.9286 (the reference's own per-frame
minimum on its lossless frames). The question §"Design" asked was whether those
are the sharpest frames.

**They are the least sharp frames.** The overlap between the five best-scoring
frames and the five sharpest is **empty**.

| frame | CLIP | clears | sharpness | sharpness rank | high-frequency share |
|---|---|---|---|---|---|
| 0 | 36.7943 | YES | 10.0400 | **25 of 25** | 0.047195 |
| 2 | 36.4033 | YES | 10.2967 | **24** | 0.051021 |
| 1 | 36.2714 | YES | 10.3067 | **23** | 0.050562 |
| 12 | 37.0408 | YES | 10.4929 | 18 | 0.047528 |
| 15 | 36.7858 | YES | 10.6020 | 12 | 0.052270 |
| 16 | 35.9588 | YES | 10.4966 | 17 | 0.050114 |
| ... | | | | | |
| 21 | 34.9132 | no | **11.3850** | **1, our sharpest** | 0.064793 |
| 17 | 35.2323 | no | 11.1354 | 2 | 0.057456 |
| 5 | 33.4080 | no | 11.0838 | 3 | 0.055693 |
| 9 | 34.0052 | no | 11.0639 | 4 | 0.053751 |
| 22 | 33.1911 | no | 10.6927 | 9 | 0.060651 |

Our three best-scoring frames are our three LEAST sharp, and our sharpest frame
fails the bound. The full 25 rows are in the JSON.

### THE INTERVENTION SETTLES THE DIRECTION

A correlation within one render is an association. These two are interventions,
and they answer opposite questions with the same instrument.

**Removing detail from the reference does not reproduce our gap.** The observed
shortfall against the lossless reference is **2.7305** CLIP points.

| sigma | reference sharpness | reference CLIP | delta | argmax | wins |
|---|---|---|---|---|---|
| 0 | 11.2740 | 38.0024 | — | true | 25/25 |
| 0.30 | 11.2450 | 38.0098 | +0.0074 | true | 25/25 |
| 0.50 | 7.7847 | 37.2884 | -0.7140 | true | 25/25 |
| 0.70 | 5.2870 | 36.6043 | -1.3981 | true | 25/25 |
| 1.00 | 3.7785 | 37.0296 | -0.9728 | true | 25/25 |
| 1.50 | 2.7095 | 36.5300 | -1.4724 | true | 25/25 |
| 2.00 | 2.1885 | 36.0337 | -1.9687 | true | 23/25 |

Blurring the reference until its sharpness matches OURS costs it **0.1192** CLIP
points by interpolation, which is **4.4%** of the shortfall. Blurring it until
its sharpness is **a fifth** of ours -- a visibly destroyed frame -- costs
1.9687, still short of 2.7305, and it still ranks the true prompt first on 23 of
25 frames. **No achievable amount of smoothing reproduces our gap.** The sweep
is also non-monotonic (0.70 costs more than 1.00), which says a small detail
difference sits inside this instrument's own noise on this axis.

**Removing OUR excess fine-scale energy RAISES our score, sharply.**

| sigma | our sharpness | our CLIP | delta | frames clearing the bound |
|---|---|---|---|---|
| 0 | 10.6374 | 35.2719 | — | **6 of 25** |
| 0.30 | 10.6052 | 35.2690 | -0.0029 | 5 |
| 0.50 | 7.3703 | 34.8468 | -0.4252 | 4 |
| 0.70 | 5.0082 | 36.1025 | **+0.8305** | 12 |
| 1.00 | 3.5381 | **37.1850** | **+1.9131** | **23 of 25** |

This is the decisive row. The same operation costs the reference 0.97 CLIP
points and gains us 1.91, and it moves us from 6 frames clearing the bound to
23. **A render that is failing because it is too smooth cannot be repaired by
smoothing it further.** The hypothesis predicted the opposite sign, and 70% of
the 2.73 shortfall is recovered by throwing our fine-scale energy away.

The symmetric diagnostic agrees. An unsharp mask at sigma 1.0, amount 1.0 raises
our sharpness from 10.6374 to **19.0357** -- 69% above the reference's own
11.2740 -- and buys only **+0.4542** CLIP, a quarter of what the blur buys.
**That arm is a DIAGNOSTIC and is not proposed as a repair**; it is here because
adding detail and removing it had to be tried in the same run, and only one of
them helps.

### TONE IS NOT THE EXPLANATION EITHER, and it was tested first

| statistic | ours | reference |
|---|---|---|
| luma mean | 126.0390 | 120.4358 |
| luma sd | **46.4470** | 39.4230 |
| luma p05 / p50 / p95 | 64.352 / 114.032 / 191.070 | 63.113 / 114.990 / 190.336 |
| r, g, b means | 127.708 / 125.021 / 126.903 | 119.948 / 119.709 / 125.456 |
| clipped fraction | 0.001076 | 0.001650 |

The fitted map from our luminance quantiles onto the reference's is gain
**0.99144**, gamma **1.04654**: essentially the identity. Applying it and
rescoring moves our mean by **+0.1846**, to 35.4565, and takes 6 frames to 9. So
tone accounts for at most **6.8%** of the shortfall, and there is no global gain
or gamma error to find. **The fit's rms residual is 14.9379 levels**, which is
large, and that is itself informative: the two renders' tone curves do not
differ by a gain and a gamma, they differ in shape. Our render has **18% more
luminance spread** at a nearly identical median, which is the low-frequency
excess the spectrum's first bin shows, seen in the histogram.

### WHAT THE MEASUREMENT POINTS AT, stated as a candidate and not a finding

The excess is **not isotropic**, and that is the sharpest structural clue.
Near-Nyquist energy on the two frequency axes, against the radial ring around
them:

| | horizontal Nyquist | vertical Nyquist | radial ring | axis / ring |
|---|---|---|---|---|
| ours | 70.058 | 69.322 | 9.530 | **7.35x / 7.27x** |
| reference | 25.043 | 20.105 | 5.148 | 4.86x / 3.91x |

Both renders put more energy on the axes than on the ring, which a picture with
horizontal and vertical structure in it does. **Ours puts 2.8 to 3.4 times as
much there as the reference does, and its axis-to-ring contrast is half again
as large.** Energy concentrated on the sampling lattice's own axes at half the
sampling rate is the signature of a **separable upsampling stage** -- a
transposed convolution, a pixel shuffle, or a nearest-neighbour rung -- in the
VAE decoder or the spatial upsampler, rather than of scene texture, which
spreads that ring evenly.

**The temporal axis says the same thing from another direction.** High-frequency
share grows along the clip in both renders (+22.8% ours, +17.9% reference, so
the growth itself is not the anomaly), but the scores diverge:

| | r(frame index, CLIP) | first 5 frames | last 5 frames |
|---|---|---|---|
| reference | **+0.6987** | 36.9820 | 38.9851, **+2.0030** |
| ours | **-0.3351** | 35.8053 | 34.2158, **-1.5895** |

**Upstream's render improves along the clip and ours decays.** Our first three
frames clear the bound and our last five are among our worst. A defect that
grows with frame index is not in the per-frame spatial path.

So the candidate this measurement names is: **an upsampling or decode stage that
injects axis-aligned near-Nyquist energy, growing along the temporal axis, while
the mid band it should be filling stays 8.6% short.** The rows that own the
paths this would live on are `LTX25-TEMPORAL-UPSAMPLER`, `LTX25-TILED-DECODE`
and `LTX25-DECODE-DTYPE`, and nothing here decides between them.

**WHAT WOULD CONFIRM IT, and none of it is in this row.** Render the same
request with the temporal upsampler and the tiled decode each disabled, and take
the same spectrum: a stage that injects the excess removes it when it is
bypassed. Compare our decoder's output against the oracle's at the SAME latent,
which removes the sampler and the prompt from the comparison entirely and is the
only form that can attribute a band to a stage. And re-render at n > 1, because
this whole reading rests on one render.

### WHAT THIS DOES NOT SAY

**n IS 1 ON OUR SIDE.** Every number here about our render is one render, and
`ltx25-render-confirm.sh:551` is why. The run-to-run spread of our own adherence
reading remains UNMEASURED, so the S1 verdict itself could still move; the
falsification above does not depend on that, because it rests on interventions
applied to the same 25 frames rather than on the gap's exact size.

**The instrument sees a 224x224 centre crop.** CLIP resizes a 320x192 frame, so
it never sees the frame's own Nyquist directly, and the horizontal downscale
folds our near-Nyquist excess back into the band it does see. That aliasing is a
plausible route from the excess to the score and this row does not measure it.
It is a limit on how far the mechanism is traced, not on whether the hypothesis
was refuted.

**Nothing here says our render looks bad to a viewer**, and nothing here is a
repair. The blur arm recovers 70% of the shortfall and **must not become one**:
it would be fitting the output to the gate, on an instrument whose absolute
value is uncalibrated and a gate that is one sample deep.

## Now

`DONE`. The hypothesis this row was created to test is refuted, the record it
came from is corrected in the same change, and the repair remains unowned --
but it is now unowned with a direction, which it did not have before.
