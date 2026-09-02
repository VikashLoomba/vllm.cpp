# LTX-2.5 — the latent upsampler's `dims == 2` arm

Row: `LTX25-UPSAMPLER-ARMS`. Campaign: [`ltx-2-5.md`](ltx-2-5.md)
(operator-owned; **not edited by this row**). Issue:
[#2577](https://github.com/mudler/vllm.cpp/issues/2577). Scope source:
[`ltx25-completion-scope.md`](ltx25-completion-scope.md) §8 order 6, which pairs
**A8** and **A9** as "the cheapest real gaps", size S each, no dependencies.

Base: `origin/main` at `6974557b0`.

Upstream pin:

| Reference | Revision |
|---|---|
| Lightricks/LTX-2 (`packages/ltx-core`) | `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` |
| vLLM-Omni | `a4ea67a` |

vLLM-Omni registers `ltx2` and is the primary oracle for this architecture. It
carries no `LatentUpsampler`, so the operator's definition is read from the
model author's own runtime, which is the row `ltx-2` occupies in AGENTS.md's
secondary-oracle table.

## 0. Honesty statement — what this row does and does not claim

This row lands **A9 only**. A8 was investigated to the same depth and is
returned undecided under `## Owed`, with the measurement that makes it a
decision rather than a port. The scope document sized the two together and
called A8 "the single cheapest real gap"; that sizing is correct about the
*operator* and does not survive contact with the *consumer*, which is the
finding §5 records.

Nothing below is quoted from another document. Every local anchor was read from
the tree at `6974557b0` and every upstream anchor with
`git show fd4ded7f:<path>`, never from a working tree, because a fork's working
tree is not the pin.

## 1. The gap, on this tree

`Ltx2LatentUpsample` refuses every non-3-D config:

```
src/vllm/model_executor/models/ltx2_upsampler.cpp:467-472
  Require(config.dims == 3, "ltx2 upsampler: dims=... is not ported. ...");
```

`config.dims` is not a constant this port chose. It is read from the
checkpoint's own metadata at `src/vllm/model_executor/models/ltx2_loader.cpp`
(`Ltx2ParseUpsamplerConfig`), mirroring
`model/upsampler/model_configurator.py:17` (`dims = config.get("dims", 3)`).
A `dims=2` checkpoint is therefore an ordinary input, not a hypothetical.

## 2. What upstream builds

`model/upsampler/model.py:47`:

```python
conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d
```

That one line reaches four parameter groups — `initial_conv` (`:49`), both
`ResBlock` stacks (`:53`, `:76-78`, whose own `conv` is chosen the same way at
`res_block.py:21`) and `final_conv` (`:80`). The `upsampler` module itself is
**not** chosen by `dims`: its branch (`:55-72`) tests the two flags only.

`model.py:85-100` is the forward:

```python
if self.dims == 2:
    x = rearrange(latent, "b c f h w -> (b f) c h w")   # :86
    x = self.initial_conv(x); x = self.initial_norm(x); x = self.initial_activation(x)
    for block in self.res_blocks: x = block(x)
    x = self.upsampler(x)                                # :94
    for block in self.post_upsample_res_blocks: x = block(x)
    x = self.final_conv(x)
    x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)  # :100
```

### 2.1 The two consequences of the fold, which ARE the port

**Kernel rank.** Every convolution above is a 4-D `Conv2d` weight, where the
`dims == 3` arm's identically-named tensor is 5-D. This is the same rank trap
the temporal arm already documents at `ltx2_upsampler.cpp:421-424` for
`upsampler.0.weight`, now applying to four more groups.

**GroupNorm statistics.** `(b f)` makes every frame its own sample, so
`GroupNorm(32, ...)` reduces over `(channels_per_group, H, W)`. Our `GroupNorm`
reduces over `frames * height * width` (`ltx2_upsampler.cpp:166`), which is the
`dims == 3` statistic. A port that reused it would produce a correctly shaped,
finite, plausible latent that is wrong everywhere — the failure mode this file's
header already names twice.

There is no third consequence. `Silu` is pointwise, `PixelShuffle2d` and the
rational resampler are already per-frame operators, and the `upsampler` branch
does not read `dims`.

## 3. Design

Mirror the fold rather than re-deriving its effect. Inside the existing
per-batch loop, the `dims == 2` arm iterates frames and runs each as its own
one-frame `Volume`. That reproduces `(b f)` exactly and, because it does, the
existing `GroupNorm` gives the per-frame statistic with no change to it: a
one-frame volume's `frames * height * width` **is** `height * width`.

The convolutions become `Conv2dPad1PerFrame`, which already exists
(`ltx2_upsampler.cpp:120`) and already takes a 4-D weight and already asserts
its element count at `:132`. `ResBlockForward` gains a `dims` parameter and
selects between the two conv helpers, mirroring `res_block.py:21` rather than
duplicating the block.

`EnumerateLtx2UpsamplerTensors` emits 4-D shapes for the four groups when
`dims == 2`. Nothing else in that function moves.

**Why not a separate 2-D code path.** AGENTS.md forbids a parallel hand-written
path where a seam can carry the behaviour, and upstream itself expresses the
difference as one conv-constructor choice plus a reshape. Two functions here
would be two things to keep in agreement for no gain.

## 4. Reachability

The dims=2 arm is reached from a production entry point on its default
configuration, and its output is consumable rather than merely computed.

Entry point: the `upsampler_path` load extra
(`include/vllm/multimodal/ltx2_video.h`), read by `LoadVideoEngine` and applied
by the second phase's input transform at
`src/vllm/multimodal/ltx2_video.cpp:3505-3508`.

Consumability is the part worth stating, because it is what separates A9 from
A8. With the default flags a `dims=2` upsampler returns `[c, f, 2h, 2w]`:
frames preserved by the fold, `h` and `w` doubled by `PixelShuffleND(2)`.

`Ltx2UpsampleVideoLatent` has **three** product call sites, all in
`ltx2_video.cpp`, and each pins the frame axis:

| # | Call site | What it requires of the frame axis |
|---|---|---|
| 1 | `:3505`, the video latent | `up.frames == vshape.frames`, checked at `:3509-3516` |
| 2 | `:3532`, the generated keyframe slots (`dfr_pipeline.py:348`) | `up_slots.frames == slot_positions.size()`, checked at `:3536-3548` |
| 3 | `:5042`, DFR's temporal rounds | the **temporal** arm, not this one |

Sites 1 and 2 are the spatial arm's, and both require the frame count to come
back unchanged. A `dims=2` upsampler does exactly that, so the shapes agree at
**every** frame count and a caller who supplies a `dims=2` checkpoint renders at
the full requested size.

The smallest failing test therefore enters through `LoadVideoEngine` and
`Generate`, not through `Ltx2LatentUpsample` directly, and asserts a completed
render rather than a changed error message.

## 5. Why A8 is NOT in this row

The operator is genuinely small: `model.py:55-59` is `Conv3d(mid, 8*mid)` +
`PixelShuffleND(3)`, and `:109-113` routes it through the same first-frame drop
the temporal arm already uses. The generator already constructs the module
(`scripts/gen-ltx2-pipeline-goldens.py`, the `spatiotemporal` block), so an
upstream golden is available for the asking.

The consumers are the problem, and there are two of them rather than one. The
arm returns `[c, 2f-1, 2h, 2w]`, while both spatial call sites in §4's table
require the frame count back unchanged — `vshape.frames` at `:3509-3516` and
`slot_positions.size()` at `:3536-3548`. Those agree only when `f == 1`, that is
when the temporal doubling is exactly undone by the mandatory drop.

`vshape.frames` is `(frames - 1) / factors.time + 1` (`:3395`) and
`factors.time` is the default-constructed 8
(`ltx2_video.cpp:2855`, `ltx2_pipeline.h:460-462`), so `f == 1` needs
`frames <= 8`. That is reachable — `num_frames` in 2..8 passes through verbatim
at `:2837`, and the only lower bound anywhere on the path is `frames < 1` at
`:2853` — but it is the one configuration in which the capability's whole point,
the doubled frame axis, is cancelled before anything can observe it.

For every other clip, porting the operator would replace the named refusal at
`ltx2_upsampler.cpp:465` with the generic shape complaint at `:3511`. The guard
immediately above it (`ltx2_video.cpp:3455-3477`) exists to prevent precisely
that substitution for the temporal-only arm, and its comment says so.

So the choice is between mirroring upstream's compute and keeping a diagnosis
this tree deliberately built. That is a product decision. AGENTS.md's stop
condition covers it: return `NEEDS_DECISION` rather than invent behaviour, and
rather than land an arm no production path consumes. Recorded under `## Owed`.

## 6. Tests, ported in this change

| Upstream | Ported as |
|---|---|
| `model.py:47,85-100` executed at reduced dimensions | a `Dims2` arm in `scripts/gen-ltx2-pipeline-goldens.py` §8, emitting out-shape, parameter manifest and value golden from the real module |
| the parameter contract implied by `conv = Conv2d` | `CheckManifest` over the `Dims2` manifest, which fails on a 5-D shape before any value is compared |
| — | the end-to-end reachability case in `tests/vllm/multimodal/test_ltx2_video.cpp`, which renders through `LoadVideoEngine`/`Generate` with a `dims=2` checkpoint |

Both sides rebuild every weight from the shared name-keyed stream, so no weight
byte is checked in and a shape disagreement changes the values rather than
hiding.

The refusal case in `test_ltx2_pipeline.cpp` ("the upsampler refuses the arms it
does not implement") loses its two `dims` arms and keeps the rest.

## 7. Risks

- **The fold is mirrored but the golden is the only thing that proves it.**
  A per-frame loop and a folded batch differ in no shape, only in the GroupNorm
  statistic. The `Dims2` golden is the gate; `mid_channels = 32` with
  `GroupNorm(32, ...)` gives one channel per group, so the per-frame and
  per-clip statistics differ and the golden separates them.
- **A 2-D checkpoint is not one this campaign has in hand.** The arm is gated
  against the executed upstream module, not against a shipped file, and §8 says
  so rather than implying a checkpoint gate it does not have.

## 8. Gates

```sh
python3 scripts/gen-ltx2-pipeline-goldens.py --ltx2 ~/_git/LTX-2 \
    --vllm-omni ~/_git/vllm-omni --out tests/vllm/models/ltx2_pipeline_goldens.inc
ctest --test-dir build -R 'ltx2_pipeline|ltx2_video' --output-on-failure
scripts/agent-preflight.sh --staged
```

## Now

`ACTIVE` — A9 implementing against issue #2577.

## Owed

- **A8, the spatiotemporal upsampler arm** (`model/upsampler/model.py:55-59`,
  refused at `src/vllm/model_executor/models/ltx2_upsampler.cpp:465`). Returned
  as a decision, not a port: §5 has the measurement. Owner: this row. Tracked by
  [#2577](https://github.com/mudler/vllm.cpp/issues/2577), which carries the
  same reasoning in its body.
