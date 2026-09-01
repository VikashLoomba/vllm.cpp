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
- **Every `AGREE` row rests on checkpoint metadata that neither side re-reads
  here.** Upstream does not hold these defaults as constants: it resolves them
  from the checkpoint, through `resolve_cli_params` (`args.py:485-495`) into
  `detect_checkpoint_path` (`:460-477`) and `detect_params`
  (`constants.py:166-179`), and that reader falls back to `LTX_2_PARAMS`
  silently for "older, unset, or unreadable versions". `LTX_2_PARAMS` carries
  `stg_blocks=[29]` and 40 steps (`constants.py:47`, `:56`, `:66`, `:80`), not
  `[28]` and 30. So `stg_blocks [28] AGREE` is conditional on the BF16
  transformer's `model_version` parsing as at or above `(2,4)` on upstream's own
  reader, and a checkpoint whose metadata does not parse would put upstream on a
  different STG block and a different step count while this table still read
  AGREE. Phase A's own run confirms it: `detect_model_version` logs the version
  it read (`constants.py:162`), so the run's log settles which branch upstream
  took, and no separate experiment is owed.
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

## Findings so far

### A LATENT-side divergence is PROVEN statically, and it needed no GPU

The cross-decode is still queued, but a static read of the two denoise loops at
their pinned revisions already answers half the question, and it answers it in
the direction the cross-decode was built to test.

**Of everything the denoise loop resolves, exactly one thing differs: the sigma
schedule's shift anchor.** Guidance, the negative prompt, the sampler and STG
all AGREE, each checked to a `file:line` on both sides:

| Parameter | Upstream | Ours | |
|---|---|---|---|
| video `cfg_scale` / `stg_scale` / `rescale_scale` / `modality_scale` / `skip_step` | 3.0 / 1.0 / 0.7 / 3.0 / 0 (`constants.py:51-55`) | same (`ltx2_pipeline.cpp:950-954`) | AGREE |
| `stg_blocks` (both modalities) | `[28]`, the 2.3 override the 2.5 key inherits (`constants.py:86`, `:124`, `:130-133`) | `{28}` (`ltx2_pipeline.cpp:969-970`, `:974-981`, `:1005-1014`) | AGREE |
| audio guidance | 7.0 / 1.0 / 0.7 / 3.0 / 0 (`constants.py:61-65`) | same (`ltx2_pipeline.cpp:956-960`) | AGREE |
| negative prompt | `DEFAULT_NEGATIVE_PROMPT` (`constants.py:186-199`) | `LightricksNegativePrompt()` (`ltx2_pipeline.cpp:1101-1107`) | AGREE, byte for byte, 1171 chars |
| sampler | `euler_denoising_loop` + `EulerDiffusionStep`, deterministic, no per-step noise (`samplers.py:39-81`, `diffusion_steps.py:25-40`) | the first-order arm with `kEuler` (`ltx2_video.cpp:4732-4820`, `ltx2_pipeline.cpp:241-258`) | AGREE |
| STG application | `SKIP_VIDEO_SELF_ATTN` + `SKIP_AUDIO_SELF_ATTN` on block 28 (`denoisers.py:111-119`) | `kSkipVideoSelfAttn` + `kSkipAudioSelfAttn` (`ltx2_denoisers.cpp:143-155`) | AGREE |
| **sigma shift anchor** | **4096** | **240** | **DIFFER** |

Upstream's `LTX2Scheduler.execute` takes an OPTIONAL latent and
`packages/ltx-core/src/ltx_core/components/schedulers.py:32` reads
`tokens = math.prod(latent.shape[2:]) if latent is not None else
default_number_of_tokens`. `ti2vid_one_stage.py:207` calls
`self._scheduler.execute(steps=num_inference_steps)` and passes **no latent**,
so upstream takes `MAX_SHIFT_ANCHOR = 4096` (`schedulers.py:11`, `:29`).

Our `Ltx2SigmaSchedule` (`src/vllm/model_executor/models/ltx2_pipeline.cpp:105-146`)
mirrors that formula exactly -- same `max_shift` 2.05, `base_shift` 0.95,
`stretch`, `terminal` 0.1, `power` 1. The formula is not the divergence. The
ARGUMENT is. `src/vllm/multimodal/ltx2_video.cpp:4227-4231` passes
`target_tokens` unless the phase asks for the scheduler default, and
`OneStagePhase` (`ltx2_pipeline.cpp:1124-1147`) never sets `schedule_tokens`, so
it keeps the struct default `kTargetLatent`
(`include/vllm/model_executor/models/ltx2_pipeline.h:725`). The only two
assignments of `kSchedulerDefault` in that file are at `:1776` and `:1961`, and
neither is `one_stage`.

At this render's geometry `target_tokens = ceil(25/8) * (192/32) * (320/32)
= 4 * 6 * 10 = 240`, so `sigma_shift = 240*mm + b = 0.669271` against upstream's
`2.050000`, and `exp` is 1.952813 against 7.767901. Recomputed here from both
sources rather than transcribed:

| step | upstream (4096) | ours (240) | delta |
|---|---|---|---|
| 0 | 1.000000 | 1.000000 | 0 |
| 1 | 0.965712 | 0.921534 | -0.044178 |
| 2 | 0.921875 | 0.832166 | -0.089708 |
| 3 | 0.863856 | 0.729457 | -0.134399 |
| 4 | 0.783445 | 0.610176 | -0.173269 |
| 5 | 0.664579 | 0.469962 | **-0.194617** |
| 6 | 0.471003 | 0.302774 | -0.168228 |
| 7 | 0.100000 | 0.100000 | 0 (pinned by `stretch`/`terminal`) |
| 8 | 0.000000 | 0.000000 | 0 |

Every intermediate sigma differs, by up to 0.1946. Our schedule leaves the
high-noise regime EARLY -- at step 4 we are at 0.610 where upstream is at 0.783 --
and the high-noise steps are where classifier-free guidance sets global
composition and prompt semantics, which is what a CLIP prompt-adherence score
measures. That is a mechanism, and it points the right way.

**This is not a new discovery about upstream; it is a new attribution.** The
tree already carries the reading, in this row's own words, at
`include/vllm/model_executor/models/ltx2_pipeline.h:660-707`: it enumerates
upstream's seven `.execute()` call sites, records that six pass no latent, names
`ti2vid_one_stage.py:207 -> our one_stage x4` in that list, calls the divergence
"REAL rather than a rounding", and states that the default was left at today's
behaviour deliberately because flipping it re-samples six shipped arms and
rewrites their goldens. What was NOT known is that this arm is the one carrying
a measured 2.8559-point adherence FAIL. The comment's own worked example is at
the recipe default geometry, 6144 tokens and shift 2.78, where the error points
the OTHER way; at 320x192x25 it points down and is larger.

### What this does and does not establish

- **ESTABLISHED: our final latent cannot be upstream's**, and for a reason that
  is not the noise draw. Two engines walking different sigma trajectories are
  solving different problems at every intermediate step.
- **NOT ESTABLISHED: that our VAE decode is faithful.** Both halves can be
  wrong at once. Only the cross-decode exonerates the VAE, and it is queued.
- **NOT ESTABLISHED: that this schedule difference CAUSES the 2.8559 points.**
  It is a mechanism with the right sign and the right regime, which is a
  hypothesis and not a measurement. The ablation that would test it is one line
  -- setting `OneStagePhase`'s `schedule_tokens` to `kSchedulerDefault` -- and
  this row does not run it, because changing a value until a score improves is
  the tuning the scope forbids and because that flip belongs to whoever owns the
  six arms it re-samples.

### The owner reference in that comment does not resolve, and that is REMOTE_UNVERIFIED

`ltx2_pipeline.h` names `https://github.com/mudler/vllm.cpp/issues/1150` as the
owner of the flip. `gh api repos/mudler/vllm.cpp/issues/1150` returns 404 --
**and so do 1148, 1149, 1151 and 1152**, while 2513 and 2514 resolve normally in
the same session. Five consecutive numbers failing together is a range effect in
the client, not five deletions, and this repository has already written the
"they were all deleted" conclusion into a policy file once and had to retract
it. So the state of #1150 is **REMOTE_UNVERIFIED**, not absent, and nothing here
concludes that the flip is unowned. It needs a check from a client that can read
that range.

## Now

`ACTIVE`. The spec, the oracle latent instrument and the lease harness are
committed. The static half of the question is ANSWERED and its answer is
LATENT, with the mechanism named above.

What is outstanding is one `rc` job each:

- **Phase A**, `rc` job `a24f9336-897f-4810-9851-13dade5acb52`, queued on
  `dgx:gpu0`, which produces the oracle's latent and its control. The
  cross-decode that EXONERATES or CHARGES the VAE follows from it, and needs no
  further GPU.
- **Phase B**, `KEEP_FRAMES=1 N=3 scripts/ltx25-render-confirm.sh`, which gives
  the -0.7368 S1 margin the error bar it has never had.

Neither is a blocker on the finding above, which was taken from source at both
pins and recomputed rather than transcribed.
