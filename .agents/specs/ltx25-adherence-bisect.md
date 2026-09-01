# SPEC — `LTX25-ADHERENCE-BISECT`: is the adherence gap in the LATENT or in the VAE DECODE?

Issue: [#2514](https://github.com/mudler/vllm.cpp/issues/2514).
Owner row: `LTX25-ADHERENCE-BISECT`.

`LTX25-PROMPT-ADHERENCE` measured the gap and
[`ltx25-prompt-adherence.md`](ltx25-prompt-adherence.md) `## Owed` records that
**no row owns closing it**, and that naming an owner "needs the attribution
first". This row is one step of that attribution and it is deliberately a small
one: it halves the search space and it stops.

## Scope

IN scope:

- Establishing whether our final video latent or our video VAE decode carries
  the divergence, by a measurement that does not depend on matching the
  sampler's noise draw.
- Giving the S1 verdict the error bar it does not have, by retaining and scoring
  the frames of every render taken under this row.
- Recording the memory format of the latent on both sides, because
  AGENTS.md requires the dtype comparison and a token gate cannot see it.

OUT of scope, and named so that nobody reads a silence as a claim:

- **The repair.** Localizing is this row. Fixing is the next one, and this row
  does not open it.
- **The pixel-statistics diagnosis.** [#2513](https://github.com/mudler/vllm.cpp/issues/2513)
  (`LTX25-ADHERENCE-DETAIL-LOSS`) owns the frequency-domain analysis of the two
  existing frame sets. This row must not duplicate it.
- **Tuning anything to raise the score.** Fitting an output to a gate is what
  AGENTS.md forbids, and this gate is one sample deep.
- **Audio.** The audio latent is dumped where it is free to dump, and nothing
  in this row's verdict rests on it.

## The request, which is already pinned

Byte for byte the manifest's, and identical to every neighbouring row's:

    prompt   "A red fox walks slowly through a snowy pine forest at sunrise, cinematic."
    320x192, 25 frames, 8 inference steps, seed 42
    the four BF16 checkpoints of tests/parity/goldens/ltx2_oracle/ltx2_oracle_manifest.json

Latent tokens = `(320/32)*(192/32)*ceil(25/8)` = **240**. The oracle's own render
of this request took **93.8 s** (`ltx2_oracle_manifest.json`,
`result.render_seconds`), so this is minutes of compute behind an hour of
staging.

## Upstream anchors

Read at the pinned oracle revision `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`
(`.agents/oracles/ltx-2.md`), in the local clone `/home/mudler/_git/LTX-2`,
verified clean at that SHA.

| What | Anchor |
|---|---|
| Pipeline entry | `packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py:220-241` |
| The denoise loop | `packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py:39-81` |
| Unpatchify to the 5D latent | `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:575-577` |
| **The decode seam** | `ti2vid_one_stage.py:243` — `self.video_decoder(video_state.latent, ...)` |
| `LatentState.latent` | `packages/ltx-core/src/ltx_core/types.py:279` |
| **The supported diagnostics hook** | `blocks.py:585-621` — `RecordingDiffusionStage` |

Our side, on `main` at `6bf3abb58`:

| What | Anchor |
|---|---|
| ABI entry | `include/vllm.h:1140` — `vllm_video_generate` |
| The render | `src/vllm/multimodal/ltx2_video.cpp:2130` — `Ltx2VideoEngine::Generate` |
| The denoise loop | `src/vllm/multimodal/ltx2_video.cpp:4642-4828` (phase `denoise`) |
| Unpatchify to the volume | `src/vllm/multimodal/ltx2_video.cpp:4893` |
| The latent volume | `src/vllm/multimodal/ltx2_video.cpp:2970` — `std::vector<float> video_latent_volume`, `[C,F,H,W]` |
| **The decode seam** | `src/vllm/multimodal/ltx2_video.cpp:5457` — `Ltx2VideoDecodeStreaming(...)` |
| Decoder signature | `include/vllm/model_executor/models/ltx2_tiling.h:313` |

## Design

### The obvious experiment is invalid, and that is this row's first finding

Running both engines at seed 42 and differencing the two final latents cannot
answer the question. [`ltx25-oracle-absolute.md`](ltx25-oracle-absolute.md)
`### What this change lands, and why it is two jobs and not one

| File | What it is |
|---|---|
| `tools/oracle/ltx2_latent_dump.py` | runs upstream's own `main()` in-process with `RecordingDiffusionStage` installed, dumps the final video latent, and proves the dump changed nothing |
| `tools/oracle/jitprobe.py` | the triton JIT gate the LTX-2 setup recipe already depended on, which existed only on an untracked NAS share; committed byte-for-byte at the digest that ran |
| `scripts/ltx25-adherence-bisect.sh` | phase A: pin, venv, stage-and-verify, dump, copy out |
| `scripts/ltx25-render-confirm.sh` | gains `KEEP_FRAMES=1`, which is phase B |

**Phase B is a separate `rc` job, deliberately.** The n = 1 question needs three
of OUR renders, and `ltx25-render-confirm.sh` already takes them with a verified
binary, a staged checkpoint set and a phase table. It deleted renders 2 and 3's
frames at its `:551`, which is the single line that made the reading n = 1.
`KEEP_FRAMES=1` retains them instead. **The `if [ "$i" = 1 ]` compare block is
untouched**, so the gate that harness enforces is byte-for-byte the gate it
enforced before, and the retained frames are evidence for a later CPU scoring
pass rather than a second verdict. `.agents/specs/ltx25-prompt-adherence.md`
`## Owed` named both ways to close n = 1 and left the choice open; this is the
retain-past-the-loop one, chosen because it cannot move the verdict.

Two jobs of under an hour survive an hourly crash where one job of two hours
does not, and `dgx` has crashed roughly hourly under long sequences. Phase A is
queued first because it answers the row's primary question and is the cheaper
half: the oracle's own render is 93.8 s, and its output is 61,440 bytes.

**Phase B reuses the CACHED binary at `SRC_SHA 790c582bbba45ab0f7b74aafee361e4557a84bf2`,
which is the binary that produced the reading in the record.** That is the right
subject for an error bar: rebuilding at today's `main` would measure the spread
of a DIFFERENT engine and confound run-to-run noise with 112 commits of drift.

## Risks/decisions` already records the reason: **"Same prompt, same seed
integer, different engine, so the sampler's noise is not the same draw."** Two
different draws through two CORRECT denoisers land on two different latents, so
a large elementwise difference is exactly what a healthy pair would also
produce. The instrument could not separate the case it exists to separate. That
is `an-instrument-whose-failure-looks-like-a-result` in its purest form, and it
is written here rather than discovered after a lease.

### The cross-decode is noise-independent, and it is the measurement

Dump the ORACLE's own final video latent once, then decode that ONE latent
through both VAEs:

| Arm | Decoder | Pixels |
|---|---|---|
| `O/O` | the oracle's VAE | the committed reference render, already digest-verified in the tree |
| `O/V` | **our** VAE | the new measurement |

Both are scored on the pinned CLIP instrument that `LTX25-PROMPT-ADHERENCE`
landed. The input is byte-identical between the arms, so the reading is a paired
test and no noise matching is needed.

The verdict, in the units of the gap itself (reference 38.1278, ours 35.2719,
gap 2.8559):

- `O/V` near **38.1278** → our VAE decode is faithful, and the divergence lives
  in the LATENT: scheduler, sigma schedule, guidance/STG, RoPE, or the DiT
  forward. The VAE is exonerated.
- `O/V` near **35.2719** → the VAE decode accounts for the gap.
- `O/V` between → the split is quantified rather than argued, and the row
  reports the fraction each half carries.

The one-sided reading is the strong one: a VAE that reproduces upstream's own
frames from upstream's own latent cannot be what degrades a good latent.

### The oracle dump changes no upstream byte

`RecordingDiffusionStage` (`blocks.py:585-621`) is upstream's OWN supported seam
for reaching a stage's outputs from outside a pipeline; its docstring names the
substitution `pipeline.stage = RecordingDiffusionStage(pipeline.stage)`. The
driver substitutes it and reads `stage.video_states[0].latent`. **No file under
the pinned checkout is edited**, so the pin still describes the bytes that ran.

`ti2vid_one_stage.main()` builds its pipeline in a local, so the substitution
needs a driver that mirrors `main()` rather than a monkeypatch into it. That
driver is therefore a second thing to validate, and §"Tests" says how.

### The dtypes differ, and the row records it either way

Upstream's final latent is **bf16**, shape `(1, 128, 4, 6, 10)`
(`ti2vid_one_stage.py:79`, `types.py:73-88`). Ours is **f32**, shape
`[C,F,H,W]` with no batch axis (`ltx2_video.cpp:2970`). AGENTS.md requires the
memory-format comparison beside the value one, and an f32 model-path buffer
where upstream carries bf16 is the polarity that section names. This row
REPORTS that difference. It does not change it: narrowing a dtype to move a
score is the tuning this row forbids itself.

## Risks

- **The dump could perturb the run.** A probe that synchronizes can cure the
  symptom it was added to observe. Controlled below.
- **The driver is not upstream's `main()`.** Controlled below.
- **The CPU VAE arm may be too slow or may not reach the same kernels as the
  CUDA arm.** If our cross-decode runs on CPU and the render ran on CUDA, the
  arms differ in more than the latent. Recorded as a limit, and the CUDA arm is
  preferred where a lease allows it.
- **n = 1 may swallow the verdict.** If the S1 reading moves by more than 0.7368
  between renders of one unchanged binary, the -0.7368 margin is not a finding
  and neither is any bisection built on it. This row reports that FIRST if it
  happens.
- **`dgx` has crashed roughly hourly under long sequences.** The lease job is
  phase-structured and checkpoints each phase's artefacts to `/workspace` before
  the next begins, so a crash costs one phase and not the run.

## Tests, and the controls that make each instrument admissible

1. **The dump did not perturb the oracle.** The recording run's 25 decoded
   frames must match the committed
   `tests/parity/goldens/ltx2_oracle/SHA256SUMS` digests. This single check
   proves three things at once: the oracle is reproducible at this request, the
   `RecordingDiffusionStage` substitution changed nothing, and the hand-written
   driver is equivalent to upstream's `main()`. A mismatch invalidates the
   latent and the row says so rather than scoring it.
2. **The dump actually ran.** The driver asserts `len(stage.video_states) > 0`
   and that the written file's byte length equals the product of the recorded
   shape and itemsize, and it exits non-zero otherwise. A zero counter that
   could not be written must not read as a pass.
3. **The cross-decode consumed the oracle's latent and not ours.** The tool
   prints the sha256 of the latent file it read and the shape it parsed, and
   refuses a shape that is not the oracle's.
4. **n, with its spread.** Every render taken under this row retains its frames
   and is scored. `loadavg` and GPU SM clock are recorded beside each, because
   the existing 8.03% wall spread is what a three-times-busier box looks like
   and not evidence about pixels.

## Gates

- `scripts/agent-preflight.sh` at rc 0 before the commit.
- The oracle-frame digest check of test 1, which is the run's own admissibility.

## Evidence

Under `/workspace/ltx25-adherence-bisect/run/<RUN_ID>/`, copied off the worker's
local disk as each phase completes: `PROVENANCE`, the oracle latent and its
sha256, the recording run's frames and their digest comparison, the cross-decode
frames, every retained render's frames, and one `compare-*.json` per scored arm.

## Stop conditions

- `NEEDS_DECISION` rather than starting a repair. Localizing is this row.
- `REMOTE_UNVERIFIED` rather than a guess if the controller cannot be reached,
  and never `ssh` plus a file mutex on a fleet device.
- If the S1 gap sits inside run-to-run noise, that result outranks the
  bisection and the row leads with it.

## Now

`ACTIVE`. The spec is the first commit. The measurement follows in this row's
branch.
