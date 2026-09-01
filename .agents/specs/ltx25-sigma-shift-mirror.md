# LTX25-SIGMA-SHIFT-MIRROR — the sigma shift takes the 4096 anchor, as both oracles do

Row `LTX25-SIGMA-SHIFT-MIRROR`. Issue
[#2521](https://github.com/mudler/vllm.cpp/issues/2521). Campaign
[`ltx-2-5.md`](ltx-2-5.md), under roadmap row `ROAD-V1-LTX25`.

Upstream pins, both read with `git show <pin>:<path>` rather than from a working
tree:

- vLLM-Omni `a4ea67a21b20054dacc6e83952f9bd407e8ee4e7`, the PRIMARY oracle; it
  registers `ltx2`.
- Lightricks/LTX-2 `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`, the secondary,
  for the 2.4 / 2.5 generations and the pipelines vLLM-Omni does not key.

Base: `63889449c1819678dd5d7523e7b21ea218176b02`, pinned when the worktree was
created.

## Now

`READY` -> `DONE` with this change.

## Scope

**In.** The value of `Ltx2PhaseRecipe::schedule_tokens` on every recipe whose
phase derives a schedule, and only that value. The tests that pin the current
value, and the one subcase whose assertion form the flip invalidates.

**Out.** `Ltx2SigmaSchedule` itself, which already mirrors upstream exactly (see
`## The function is already correct`). The adherence score — see
`## Why this is not gated on the adherence number`. `use_official_sigma_schedule`,
which this row measured as SET BY SEVEN RECIPES AND READ BY NOTHING; recorded
under `## Owed` rather than fixed here, because deleting or wiring a dead field
is a different change with a different blast radius.

## The defect

`Ltx2PhaseRecipe::schedule_tokens` selects which token count feeds
`sigma_shift = tokens * mm + b`. It has defaulted to `kTargetLatent` — the
phase's own target grid — since the field was introduced, and only
`ti2vid_two_stage` and `keyframe_interpolation` opt out. Both oracles say the
4096 anchor is the rule and the target grid is the single exception.

### vLLM-Omni cannot produce anything but 4096

`vllm_omni/diffusion/models/ltx2/ltx2_denoise.py:177-200` builds the official
schedule with no token count in the expression at all:

```python
# :188
# Official LTX2 one-stage intentionally uses the max sequence anchor, so the shift stays at max_shift.
sigma_shift = max_anchor * slope + (base_shift - slope * base_anchor)
```

The other branch — the one a phase with `use_official_sigma_schedule=False`
takes, which upstream uses for `dmd2` — passes `max_image_seq_len` into
`calculate_shift`'s `image_seq_len` parameter (`:256-260` against the signature
at `:116-125`), so it lands on 4096 too. There is no third branch.

### Lightricks selects by whether a latent is passed

`packages/ltx-core/src/ltx_core/components/schedulers.py:32` at the pin:

```python
tokens = math.prod(latent.shape[2:]) if latent is not None else default_number_of_tokens
```

with `default_number_of_tokens: int = MAX_SHIFT_ANCHOR` (`:29`) and
`MAX_SHIFT_ANCHOR = 4096` (`:11`). The population of call sites is
`git grep -n '\.execute(' <pin> -- packages/ltx-pipelines/src/ltx_pipelines/`
and it returns seven lines. Six pass no latent. One does not.

## Arm-by-arm, and it is NOT a blanket flip

Only a phase with empty `sigmas` reaches the seam; a phase carrying frozen
distilled sigmas never evaluates it.

| Recipe (kind x versions) | Phase | Was | Becomes | Authority |
|---|---|---|---|---|
| `one_stage` x 2, 2.3 | `generate` | `kTargetLatent` | **`kSchedulerDefault`** | `ltx2_denoise.py:188` (primary); `ti2vid_one_stage.py:207` |
| `one_stage` x 2.4, 2.5 | `generate` | `kTargetLatent` | **`kSchedulerDefault`** | `ti2vid_one_stage.py:207`, no latent |
| `t2a_one_stage` x 4 | `generate` | `kTargetLatent` | **`kSchedulerDefault`** | `t2a_one_stage.py:141`, no latent |
| `a2vid_two_stage` x 4 | `stage_1` | `kTargetLatent` | **`kSchedulerDefault`** | `a2vid_two_stage.py:226`, no latent |
| `dmd2` x 2, 2.3 | `generate` | `kTargetLatent` | **`kSchedulerDefault`** | `ltx2_denoise.py:256-260`, `max_image_seq_len` |
| `ti2vid_two_stage` x 4 | `stage_1` | `kSchedulerDefault` | unchanged | `ti2vid_two_stages.py:244`, no latent |
| `keyframe_interpolation` x 4 | `stage_1` | `kSchedulerDefault` | unchanged | `keyframe_interpolation.py:200`, no latent |
| `res2s_two_stage` x 2.5 | `generate_lowres_hq` | `kTargetLatent` | **KEPT** | `ti2vid_two_stages_hq.py:267`, `latent=empty_latent` |
| `distilled_two_stage`, `dfr`, `retake`, every stage 2 | — | frozen sigmas | never reaches the seam | — |

`res2s_two_stage` is the reason `kTargetLatent` stays in the enum rather than
being deleted. Its `empty_latent` is built from
`VideoLatentShape.from_pixel_shape(stage_1_output_shape, ...)`
(`ti2vid_two_stages_hq.py:261-267`), so `math.prod(shape[2:])` IS the stage-1
target grid and `kTargetLatent` is the correct mirror there.

`T2aOneStageRecipe` is built by calling `OneStageRecipe`
(`ltx2_pipeline.cpp:1182`), so both flip on the one edit inside `OneStagePhase`.

### Two corrections to the record this row inherited

`ltx2_pipeline.h`'s comment predicted the flip would move "`one_stage` at four
version keys, `a2vid_two_stage` stage 1 and `retake`". Measured against the tree
at the base SHA, that count is wrong twice:

- `retake` does NOT move. `RetakeRecipe` sets `stage.sigmas = DistilledSigmas()`
  unconditionally (`ltx2_pipeline.cpp:1363`), so it pins upstream's DISTILLED
  branch of `retake.py:287` and never derives a schedule. The non-distilled arm
  is unported.
- `t2a_one_stage` and `dmd2` DO move, and the comment names neither.

## What the shift actually is

Ran the pinned upstream scheduler itself rather than reimplementing it: the
module was extracted with `git show <pin>:...schedulers.py`, its one import was
stubbed, and `LTX2Scheduler.execute` was called both ways. At the
LTX25-PROMPT-ADHERENCE geometry (320x192x25), `one_stage`'s target grid is
4x6x10 = 240.

| anchor | tokens | shift | sigmas at steps=3 |
|---|---|---|---|
| ours, before | 240 | 0.669271 | 0.999999881, 0.637402773, 0.099999964, 0 |
| both oracles | 4096 | 2.050000 | 0.999999762, 0.734179258, 0.100000024, 0 |

`stretch` pins sigma[0] at 1.0 and the last non-zero at `terminal` = 0.1, so the
anchor is visible only in the INTERIOR sigmas. A 2-step schedule is
`{1, 0.1, 0}` for every token count; three steps is the shortest that can see
the difference. Every test below that has to observe the trajectory runs at 3.

## The function is already correct

`Ltx2SigmaSchedule` mirrors `schedulers.py:33-57` line for line, and this row
does not touch it. The proof is already in the tree: upstream's no-latent
8-step output is

```
0.999999881, 0.965711713, 0.921874762, 0.863856077, 0.783445477, 0.664579272, 0.471002698, 0.100000024, 0
```

which is `kLtx2SchedDefaultGolden` in `ltx2_pipeline_goldens.inc` byte for byte.
So no Section-1 golden moves, and none is regenerated. The argument was wrong,
not the arithmetic.

## Tests

Red first, and the red has to be for the intended reason.

1. **New, `test_ltx2_video.cpp`** — `one_stage`'s anchor, through the production
   entry point (`LoadVideoEngine` -> `Generate`), as a 2x2 over
   (recipe, geometry) against `res2s_two_stage` as the control. The equalities
   alone would pass on a build that hard-codes 4096 everywhere; the
   inequalities alone pass on today's tree. Only the pair says the anchor is
   per-phase AND selected correctly. Mirrors the shape of the two cases that
   already exist for `ti2vid` and `keyframe`.
2. **New** — the trajectory half: `Ltx2SigmaSchedule(3, 4096)` and
   `Ltx2SigmaSchedule(3, target)` must differ, recomputed at the step count the
   render actually ran at, so that lowering the render to 2 steps reds by name
   instead of going vacuous.
3. **Changed, `test_ltx2_pipeline.cpp:3568-3569`, `:3833-3834`** — four `CHECK`s
   that assert `one_stage` and `a2vid` are `kTargetLatent`. Each is annotated in
   the tree as pinning the divergence rather than upstream. They become
   `kSchedulerDefault`, and the `res2s` control beside them is unchanged.
4. **Repaired, `test_ltx2_video.cpp:6410`** — see below. Not deleted, not widened.

### The one assertion whose FORM the flip invalidates

`ltx2 video: a last-frame keyframe...` -> SUBCASE `the sigma schedule keeps
reading the TARGET count, not the grown one` asserts

```cpp
CHECK(trace.video_tokens > trace.schedule_tokens);
```

on a `one_stage` render. It holds today because schedule = target = 2 and
video = 3. Under the flip schedule becomes 4096 and the inequality reverses.

The claim it exists to make is "supplying a keyframe must not re-shift the
schedule". The inequality was a PROXY for that claim, valid only while the
anchor tracked the target. The repair states the claim directly: render the same
request with and without the keyframe and assert `schedule_tokens` is EQUAL
across the pair, while `video_tokens` still grows. That is strictly stronger —
it detects a re-shift under either anchor, whereas the inequality detects one
only under `kTargetLatent` — so this is not a widened scope and not a deleted
assertion.

## Gates

```sh
cmake --build build -j 4 --target test_ltx2_video test_ltx2_pipeline
./build/tests/test_ltx2_pipeline
./build/tests/test_ltx2_video
```

## Risks

- **A golden rewritten without an upstream justification.** Mitigated by there
  being none to rewrite: no Section-1 golden moves, because the function and its
  goldens are already upstream-exact. If any golden had needed a new value this
  row would have stopped, per `## Stop conditions`.
- **A re-sampled arm nobody looked at.** The table above is closed over
  `git grep -n 'sigmas = \|sigmas\.empty' ltx2_pipeline.cpp`, not over the arms
  this row set out to fix.
- **Real-weights renders are not re-gated here.** No GPU lease was taken. See
  `## What is NOT verified`.

## What is NOT verified

- No real-weights render on any arm. Every gate here is the CPU fixture suite.
  The visual consequence of the flip at the LTX25-PROMPT-ADHERENCE geometry is
  NOT measured by this row and is not claimed.
- Whether the adherence score moves. See below.

## Why this is not gated on the adherence number

The justification is that the engine does not match upstream, and that alone is
the defect (AGENTS.md, "Mirror vLLM"). The LTX25-PROMPT-ADHERENCE gate is n=1
with a -0.7368 margin and no error bar; tuning toward it would be fitting to one
sample. #2513 is separately testing whether smoothness explains that gap and may
show the mechanism is something else. This change is correct either way, and if
the score gets worse it is still correct and is reported as such.

## Owed

- `Ltx2PhaseRecipe::use_official_sigma_schedule` is written by seven recipe
  builders and read by no code in `src/` or `include/`. Upstream it selects
  between two real branches (`ltx2_denoise.py:249-260`), which for the shift
  agree at 4096 but differ in how the rest of the schedule is built. Filed
  separately; not repaired here.
- `retake`'s non-distilled arm, which upstream derives at
  `retake.py:287` and this engine does not port.

## Stop conditions

- NEEDS_DECISION rather than rewriting any golden that cannot be justified from
  upstream at its pin.
- NEEDS_CONTEXT if an arm's upstream call site cannot be resolved at the pin.
