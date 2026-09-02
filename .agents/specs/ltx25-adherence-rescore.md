# LTX25-ADHERENCE-RESCORE — does the sigma-shift anchor explain the adherence gap?

Row `LTX25-ADHERENCE-RESCORE`. Issue
[#2576](https://github.com/mudler/vllm.cpp/issues/2576). Campaign
[`ltx-2-5.md`](ltx-2-5.md), under roadmap row `ROAD-V1-LTX25`.

Base: `origin/main` at `6974557b02f822d9a1d603e8639091f735b7ecbe`, pinned when
the worktree was created.

Oracle: Lightricks `LTX-2` at `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`,
`gateable = yes`; its reference render, manifest and `SHA256SUMS` are committed
at `tests/parity/goldens/ltx2_oracle/`. Scorer: `openai/clip-vit-base-patch16`
at `57c216476eefef5ab752ec549e440a49ae4ae5f3`, an INSTRUMENT and not an oracle,
pinned by `ltx25-prompt-adherence.md` §5.

## Now

`READY` -> `DONE` with this change.

## Scope

**In.** One render at the pinned request on a binary built from a tree that
carries `c9366de65` (`LTX25-SIGMA-SHIFT-MIRROR`), scored against the same
reference and the same scorer that produced the standing 35.2719, and the
comparison of the two readings. Order 1 of
[`ltx25-completion-scope.md`](ltx25-completion-scope.md) `## 8`.

**Out.** Closing the gap. Tuning anything toward the score. Any change to the
engine, the harness's verdict block, the request, the scorer, the reference, or
the bound. A second seed. This row RETURNS `NEEDS_DECISION` rather than moving
a number.

## The question, stated so that both answers publish

`ltx25-prompt-adherence.md` `## Outcome` records `[FAIL]
absolute.ours.adherence_clip 35.2719 >= 36.0087 ... margin -0.7368`, reference
mean 38.1278, on frames produced by `main` at `7905607af`.

`LTX25-SIGMA-SHIFT-MIRROR` (#2521) then changed the sigma schedule that render
used. Its own arm table puts `one_stage` x 2.5 `generate` on the flipped side:
`schedule_tokens` moved from `kTargetLatent` to `kSchedulerDefault`, so at this
geometry the shift moved from 240 tokens / 0.669271 to 4096 / 2.050000 and every
interior sigma moved with it, by up to 0.194617 at step 5. The loop leaves the
high-noise regime later. CFG sets composition and prompt semantics there, and
prompt semantics is what CLIP measures.

**That is a hypothesis and this row tests it. It is not a validation of #2521,
and a non-improvement is not a regression of it.** #2521's justification is that
the engine did not match upstream, and it stands whichever way the score moves;
that row said so in advance, in
[`ltx25-sigma-shift-mirror.md`](ltx25-sigma-shift-mirror.md)
`## Why this is not gated on the adherence number`: "if the score gets worse it
is still correct and is reported as such". The negative answer is the more
informative one here, because **no mechanism currently explains the gap at
all**: #2513 falsified smoothness and withdrew its replacement lead as a DFT
border artifact.

**A third answer is available and it is the cheapest to detect.** If the new
frames are BYTE-IDENTICAL to the old ones, the flip did not reach this
configuration, and that is the finding rather than a failed measurement.

## Why one render is enough, and what it cannot buy

`ltx25-render-confirm.md` `## Outcome` records three renders on one binary at
one seed producing byte-identical frames — the same `sha256sum frame_*.ppm |
sha256sum` — differing only in wall time. So one render on the new binary
establishes the new value EXACTLY.

**The same fact forbids the obvious next step.** Repeating a bit-deterministic
render cannot produce an error bar, and an `n` quoted over identical inputs is
not a sample size. The standing reading has no error bar
(`ltx25-prompt-adherence.md` `## Owed`, #2514) and this row does not give it
one. Uncertainty here would have to come from DIFFERENT SEEDS, which is a
different question and is out of this scope. `N=1` is therefore the honest
setting rather than a saving, and the harness's own `n = 3` speed axis is not
collected.

## Method

1. **The old reading is re-derived rather than transcribed.** The 25 retained
   PPM frames of `rc` job `93a60151-7d4d-4718-842c-ef724208be0e` are copied off
   CIFS, their `sha256sum frame_*.ppm | sha256sum` is checked against the
   committed `1166b28694001c52a6b5258804f1bb8f97ea2834dac5f16b5a9f5b48469d93ae`,
   and the tool is re-run on them on this devbox. This is the control on the
   scoring pipeline: a reading that does not reproduce 35.2719 means the
   instrument moved, and no comparison is possible until it is explained.
2. **The render.** `scripts/ltx25-render-confirm.sh` inside an `rc` lease on
   `dgx:gpu0`, `N=1 KEEP_FRAMES=1`, at the harness's own pinned request. The
   binary is built in-lease from a tarball of this branch, and its
   `binary_sha256`, `library_sha256` and `source_sha` are recorded.
3. **The score.** `scripts/ltx25-render-compare.py --adherence-model` on the
   retained frames, against BOTH reference forms — the committed
   `upstream-render.mp4` and the lossless `upstream_frames` — because
   `ltx25-prompt-adherence.md` reports the codec's 0.1254 contribution from
   exactly that pair and dropping one form would make this reading
   incomparable with the standing one.
4. **The identity check.** `sha256sum frame_*.ppm | sha256sum` over the new
   frames against the old digest.

## What would falsify each answer

| Answer | What it takes | What it does not license |
|---|---|---|
| the mechanism EXPLAINS the gap | S1 margin >= 0 at this request | calling adherence closed at any other request, seed or geometry — n is still 1 |
| the mechanism PARTLY explains it | margin moves toward 0 and stays negative | attributing the remainder to anything; the residue is unexplained |
| the mechanism DOES NOT explain it | margin unchanged or worse | any statement that #2521 was wrong |
| the flip did not REACH this arm | frames byte-identical to the old digest | reading the arm table as falsified; it is a statement about recipes, not renders |

## Risks

- **Two binaries, 112+ commits apart.** The standing 35.2719 came from `main` at
  `7905607af` and the new reading comes from this branch's base. Every number
  below names the SHA it came from, and the difference is not attributed to the
  sigma flip alone. `ltx25-prompt-adherence.md` already records that no file
  whose path names LTX, video, VAE or diffusion changed across the first 112 of
  those commits; this row re-runs that scan over its own range rather than
  inheriting the result.
- **The harness gates blockiness before it will time anything, and exits on a
  non-zero verdict.** That branch is untouched. Render 1's frames are never
  deleted by the harness at any exit, so an early exit still leaves the pixels
  this row scores.
- **`dgx` crashes under long sequences.** `N=1` is one render, and the source
  tarball plus the `confirm-bin` cache make a re-run cheap.

## Gates

Run from a worktree at this row's base, with `MODEL` the pinned CLIP snapshot,
`OLD` a local copy of the retained pre-flip frames and `NEW` a local copy of the
lease's `r1`:

```sh
# the control: the old frames are the old frames
( cd "$OLD" && sha256sum frame_*.ppm | sha256sum )   # 1166b2869400...93a6

# the two readings, each against both reference forms
for A in "$OLD" "$NEW"; do
  python3 scripts/ltx25-render-compare.py --a "$A" --label-a ours \
    --reference tests/parity/goldens/ltx2_oracle/upstream-render.mp4 \
    --adherence-model "$MODEL"
  python3 scripts/ltx25-render-compare.py --a "$A" --label-a ours \
    --reference "$REFPPM" --adherence-model "$MODEL"
done

# did the flip reach this arm at all
( cd "$NEW" && sha256sum frame_*.ppm | sha256sum )
```

## Stop conditions

- Return `NEEDS_DECISION` rather than changing the request, the seed, the
  scorer, the reference, the bound or any engine value to move the score.
- Return `REMOTE_UNVERIFIED` rather than guessing if the controller cannot be
  reached. Never fall back to `ssh` plus a file mutex on a fleet device: the
  fleet cannot see that mutex, and `minimax-music3.md` §13.10 retains a whole
  speed axis as VOID because that was done once.
- Stop and report if the control in `## Method` step 1 does not reproduce
  35.2719.

## Owed

- **n is still 1 on the adherence axis, and this row does not close that.**
  #2514 owns it and the closing move is DIFFERENT SEEDS, not more renders at
  seed 42; §"Why one render is enough" carries the reason.
- **`scripts/ltx25-render-confirm.sh` still does not pass `--adherence-model`**,
  so the adherence reading is taken by hand on a second pass after the lease.
  `ltx25-prompt-adherence.md` `## Owed` records it as order 2 of
  `ltx25-completion-scope.md` `## 8`, and this row inherits it rather than
  fixing it: wiring the scorer into the harness changes what the lease gates
  and needs its own red-first case.
