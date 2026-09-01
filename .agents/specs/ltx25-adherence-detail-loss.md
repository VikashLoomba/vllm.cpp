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
