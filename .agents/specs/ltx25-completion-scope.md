# SPEC — `LTX25-COMPLETION-SCOPE`: what it costs to call LTX-2.5 complete

Issue: [#2526](https://github.com/mudler/vllm.cpp/issues/2526).
Owner row: `LTX25-COMPLETION-SCOPE`.
Base: `origin/main` at `63889449c`. Upstream oracle: Lightricks `LTX-2` at `fd4ded7f`.

## Scope

**In:** an enumeration of what stands between `63889449c` and a complete LTX-2.5
port, with every item classified, anchored on both sides, and sized; the
sequencing of the capability gaps; and one honest total.

**Out:** implementing any arm. This row is a survey and it returns
`NEEDS_DECISION` on what to fund. It writes no product code.

Every count below is stamped with the SHA it was taken at, because a count with
no tree beside it cannot be reconciled. Rerun the commands in `## Gates`.

## 1. The distinction this row exists to draw

A grep over the LTX-2.5 sources returns 375 `Fail()` sites. They are not one
population. Three kinds share the text:

1. **Input validation.** `"Video width and height must be multiples of 32"`,
   `"the checkpoint is missing '...'"`. The overwhelming majority. Not debt.
2. **A refusal that MIRRORS upstream.** We refuse because upstream refuses, or
   because upstream has no such thing to run. AGENTS.md requires this: mirror
   "every applicable mode, default, error, and edge case". Removing one would be
   a parity DEFECT, not progress.
3. **A refusal that says we did not port it.** This is the work.

Kinds 2 and 3 are textually identical. `git grep "not ported"` returns both and
distinguishes neither. This row separates them by reading the upstream side of
each at `fd4ded7f` and asking one question: **does upstream construct this?**

## 2. The classes

| Class | Meaning | Count |
|---|---|---:|
| **A** | Genuine capability gap: upstream implements it, we refuse or lack it | **14** |
| **B** | Correct mirror: we refuse because upstream refuses or never constructs it | **4** |
| **C** | Externally blocked: unfinishable here at any effort | **0** |
| **D** | Measurement and records debt: no engine change | **10 clusters** |

**Class C is EMPTY, and that is this row's largest single finding.** All three
recorded members were verified against `fd4ded7f` rather than inherited, and all
three are falsified. §5 carries the evidence.

### 2.1 The campaign's own `## Owed` backlog, counted properly

Classes A-D are the CAPABILITY view: what a user cannot get. The campaign also
carries an internal debt list, and it is larger than any prior count.

**299 open owed items across 50 specs** at `63889449c`, read by opening every
file rather than by one grep:

| Kind | Count | Meaning |
|---|---:|---|
| ENGINE | **192** | needs code -- engine, kernel, test, gate or harness |
| MEASURE | **83** | needs a run, lease or evidence capture; no code |
| RECORD | **24** | needs a document edit only |

Four `ltx25-*` specs carry no Owed section at all, and one
(`ltx25-dit-attn-arm-parse.md`) carries an empty one. The heaviest are
`ltx25-device-residency` (40), `ltx25-decode-speed` (24),
`ltx25-t2a-one-stage` (15) and `ltx25-phase-instrument` (15).

**These 299 are not 299 rows.** Many are one bullet of a wave, several are
duplicates across specs, and **five are contradicted by another spec in the same
set** -- they are record corrections wearing the shape of work. §6 D8 carries
them.

### 2.2 A gate defect found while counting, and it under-reports this backlog

`owed_issues()` in `scripts/check-agent-record.py:2002-2017` reads the backlog
with two bugs:

```python
if "\n## Owed" not in text:
    continue
body = text.split("\n## Owed", 1)[1].split("\n## ", 1)[0]
```

1. **A numbered heading is skipped entirely.** `## 7. Owed` does not contain the
   substring `\n## Owed`, so the file `continue`s. Four LTX specs are invisible
   to the gate for this reason: `ltx25-token-append.md` (`## 8. Owed`),
   `ltx25-decode-threads.md` (`## 7. Owed`), `ltx25-decode-dtype.md`
   (`## 7. Owed`) and `ltx25-text-proj-dtype.md` (`## 6. Owed`).
2. **Only the FIRST section is read, and it stops at the next `\n## `.** A
   `###`-level owed section, or a second `## Owed`, is never seen. Five more LTX
   specs are affected, `ltx25-device-residency.md` and
   `ltx25-phase-instrument.md` among them.

**Roughly 74 of the 299 open items sit under headings this gate cannot see.**
`ltx25-decode-threads.md:292` says this about itself, so it was known and never
filed. AGENTS.md is explicit that this class of thing is the checker's defect,
not the specs': the obligation is real and the gate silently does not enforce
it. This row files it rather than fixing it, because a semantic checker change
needs its own spec and a red-first test.

## 3. Class B — the correct mirrors, recorded so nobody reopens them

These are NOT work. Each is already carried in
`Ltx2UnportedPipelineFeature` (`include/vllm/model_executor/models/ltx2_pipeline.h:1227-1264`)
or in the campaign's `Out` list, with its upstream anchor.

| Item | Local anchor | Why it is a mirror, not a gap |
|---|---|---|
| `kBetaScheduler` | `ltx2_pipeline.h:1236-1242`, refusal `ltx2_pipeline.cpp:2226` | Upstream CONSTRUCTS IT NOWHERE. All seven `ltx-pipelines` entry points hard-code `LTX2Scheduler()`; vLLM-Omni has zero hits for the name. Mirroring that means no scheduler-kind field here either, so nothing reaches the refusal |
| `kInt8ConvRot` | `ltx2_pipeline.h:1243-1248`, refusal `ltx2_pipeline.cpp:2243` | Not an LTX-2 arm at all. `quantization_factory.py:22-26` is a `str` enum with `assert_never` at `:50` and four members, none int8. `convrot`/`quarot`/`spinquant` are 0 hits upstream |
| `kMultiGpuParallelism` | `ltx2_pipeline.h:1249-1264`, refusal `ltx2_pipeline.cpp:2266` | Four upstream forms, none of them CFG batching. Upstream's own `docs/multigpu/gemma.md:103-104` records that the DISTILLED pipeline this port runs takes no negative prompt and so "runs without CFG" |
| `kMultishot` (retired) | removed; provenance `ltx2_pipeline.h:1200-1223`, `ltx25-retire-dead-arms.md` §1.1a | FABRICATED. No such upstream entry point, symbol or string. Upstream's own shipped enhancer prompts instruct "Single continuous take -- no hard cuts". A defect in our record, not a gap in our port |

**Do not re-file these.** Each has been re-derived at least twice, and
`kMultishot` cost three review rounds to retire.

## 4. Class A — the genuine capability gaps

Sized in **rows**, this project's unit: one issue's change plus the records it
invalidates. S = small (one focused change), M = medium (one recipe or one
leaf port), L = large (a new kernel or a new subsystem).

### A.1 The two that block the word "complete" outright

| # | Gap | Local anchor | Upstream anchor | Reached? | Size |
|---|---|---|---|---|---|
| **A1** | **The prompt-adherence gap. CORRECTNESS IS FAILING.** Our render scores 35.2719 CLIP against a 36.0087 bound -- FAIL by 0.7368 on S1 | `ltx25-prompt-adherence.md` `## Owed` | `ti2vid_one_stage.py` | yes, the default render | **M-L, mechanism unknown** |
| **A2** | **The speed gap: 3.230x the oracle.** Ours 302.954 s (n=3) against 93.8 s (n=1) | `ltx25-render-confirm.md` `## Outcome` | -- | yes, the default render | **L, two leaves = 54% of wall** |

**A1 is the gate, and AGENTS.md is unambiguous that it comes first:
"Correctness always comes first. Establish the declared token-exact gate before
you accept a performance result."** LTX-2.5 currently FAILS its declared
correctness gate. No amount of arm coverage makes the port complete while that
holds.

**A1 has no owner.** `ltx25-prompt-adherence.md` states it in its own `## Owed`:
"NOTHING OWNS CLOSING THE ADHERENCE GAP ... Owner of the REPAIR: **no row
exists**". The row that measured it does not fix it. Two waves are in flight
against the mechanism and neither has landed:

* `LTX25-ADHERENCE-BISECT` (PR #2520, issue #2514) localizes the gap to the
  LATENT and proposes a mechanism: our `one_stage` sigma shift anchors on 240
  tokens where upstream anchors on 4096 (`ti2vid_one_stage.py:207` passes no
  latent, so `schedulers.py:32` takes `MAX_SHIFT_ANCHOR`). Shifts 2.050000
  against 0.669271.
* `LTX25-ADHERENCE-DETAIL-LOSS` (PR #2525, issue #2513) **falsified** the
  smoothness hypothesis and **withdrew** a separable-upsampler lead that proved
  to be a DFT border artifact.
* `LTX25-SIGMA-SHIFT-MIRROR` is repairing the anchor and re-sampling six shipped
  arms' goldens.

So the mechanism is a live hypothesis, not a known quantity, and **n = 1 under
every adherence number** -- the confirm harness deletes renders 2 and 3
(`scripts/ltx25-render-confirm.sh:471,551`). A reading that moves by 0.74
between runs would make the verdict a coin toss, and nothing excludes that.

**A2's shape is already measured**, by `LTX25-DEVICE-ARM-SURVEY` (landed). Two
leaves are 54.03% of the wall and both are host-pinned:

| arm | cost | shape |
|---|---:|---|
| Gemma-4 text tower, both passes | **82.336 s / 27.18%** | queue swap + one allocation + 49 downloads per pass |
| Connector, both passes | **81.338 s / 26.85%** | a ~386-line PORT |
| Video VAE non-convolution volume | 10.916 s / 3.60% | partial port |
| Audio VAE decode | 8.773 s / 2.90% | queue swap, coupled to a golden-changing accumulator change |

The DiT -- the only model actually on the accelerator -- is **4.99%**. The tower
arm was measured and deliberately NOT taken; it needs its own row and issue.

### A.2 The unported pipelines and arms

| # | Gap | Local anchor | Upstream anchor | Reached? | Size |
|---|---|---|---|---|---|
| **A3** | `DubItPipeline` | refusal `ltx2_video.cpp:2753-2766` | `dubit.py` | refusal is reached | **M** |
| **A4** | `HDRICLoraPipeline` | no refusal; cited `ltx2_lora.cpp:246` | `hdr_ic_lora.py:229` | no | **M-L** |
| **A5** | `TI2VidTwoStagesHQPipeline` | `ltx2_pipeline.cpp:1408` | `ti2vid_two_stages_hq.py:59` | no | **M** |
| **A6** | DFR's temporal refinement rounds loop | refusal `ltx2_video.cpp:2022-2027` | `dfr_pipeline.py:235-245,402-407` | refusal is reached | **M**, weights unpublished |
| **A7** | DiffVAE / `NADiffusionDecoder` | refusal `ltx2_video_vae.cpp:1355-1359` | `video_vae.py` | yes, refused by name | **L -- needs a neighborhood-attention kernel** |
| **A8** | `kSpatiotemporalUpsampler` (both flags set) | refusal `ltx2_upsampler.cpp:465` | `model/upsampler/model.py:55-59` | **yes, the one REACHABLE unported-feature refusal** | **S** |
| **A9** | Upsampler `dims == 2` arm | refusal `ltx2_upsampler.cpp:469` | `model/upsampler/model.py:85-100` | yes | **S** |
| **A10** | Multi-keyframe request surface | ABI has two scalar slots, `include/vllm.h:934-935` | repeatable `--image PATH FRAME_IDX STRENGTH`, `utils/args.py:805-817` | ABI ceiling | **M, ABI-additive** |
| **A11** | Image conditioning at CRF != 0 (H.264 round trip) | refusal `ltx2_image_preprocess.cpp:104-117` | `media_io/decode.py:430-434` | yes | **M -- needs a vendored codec** |
| **A12** | Prompt K/V cache on the DEVICE forward | refusal `ltx2_device.cpp:1337` | `transformer.py:441-443` | yes | **M** |
| **A13** | Pre-2.3 flat vocoder arm | refusal `ltx2_loader.cpp:1757-1762` | `audio_vae/model_configurator.py:53-56` | yes | **S**, and arguably B |
| **A14** | Batched perturbation blend at batch > 1 | `ltx2.h:469`, `ltx2_device.cpp:320` | `attention.py:571-573` | degenerate at batch 1 | **S**, no consumer today |

**A8 is the single cheapest real gap and the only reachable unported-feature
refusal in the enum.** It is the natural first bite.

### A.3 The quantized arms — stated explicitly, as the contract requires

**LTX-2.5 has checkpoint-format quantized arms and ZERO quantized compute.**

The loader reads four encodings -- BF16, F32, F8_E4M3 with an F32 scale, and U8
NVFP4 in two producer dialects -- and **every one of them dequantizes to bf16
during load** (`ltx2_loader.cpp:469-546`). The plan's own
`Ltx2DitQuant` field is written at `ltx2_loader.cpp:421,423,463` and **read by
no consumer anywhere**: the symbol occurs only in `ltx2_loader.cpp` and its own
header, against a positive control (`Ltx2LoadDitFromSafetensors` occurs in six
files across `src/` and `include/`). The loader says so itself at `:437-438` --
"no consumer branches on `quant`".

The consequence is worth stating plainly: **the quantized checkpoints buy
download and disk size, not VRAM and not speed.** Resident footprint is bf16 on
device (`ltx2_video.cpp:1068`) or f32 on CPU (`:1082`). The shipped first-party
NVFP4 DiT also carries 1176 `.input_scale` tensors -- it is a W4A4 file -- and
`IsScaleSidecar` (`ltx2_loader.cpp:361`) swallows every one of them, so the
activation half of that checkpoint's scheme is silently discarded.

The VAEs, upsampler and duration head are f32 only, and a quantized checkpoint
for them is refused by name (`ltx2_loader.cpp:1657-1660`). The text feature
extractor refuses any `compute_dtype` but f32 (`ltx2_text_encoder.cpp:54-61`),
as does the CPU reference DiT (`ltx2_dit.cpp:766`). The device DiT is the one
production path with a real narrow arm: `stream_dtype` defaults to `kBF16` and
accepts exactly `{kBF16, kF32}` (`ltx2_device.h:97-109`).

**GGUF k-quants: there is no path, no scaffold, no refusal string and no matrix
row.** Established across six independent spellings and search axes, with
positive controls -- `minimax_music3_quant.cpp` and `qwen3_dflash_gguf.cpp`
exist, no `ltx*gguf*` file does; `grep -i ltx .agents/quantization-matrix.md`
returns 0 against 28 `nvfp4|fp8` hits in the same file.

**This collides with a standing rule and the collision needs a decision.**
AGENTS.md: "GGUF k-quants are a standing requirement. They are not a choice for
each model." Four LTX specs have taken a "not applicable" exemption on the
ground that upstream ships no GGUF arm for any LTX-2 component
(`quantization_factory.py:23-26` with `assert_never` at `:50`) and llama.cpp does
not carry this architecture, so there is no behaviour to mirror and no
quant-matched comparison to serve. **The tree is not self-consistent about it:**
`ltx25-generated-keyframes.md:387` calls the campaign-level GGUF arm "owed for
LTX-2.5 as a whole under #644" where `ltx25-t2a-one-stage.md:568`,
`ltx25-a2v-audio-input.md:422`, `ltx25-a2vid-recipe.md:400` and
`ltx25-dfr-pipeline.md:370` call it not applicable. **No ratification record
exists for the exemption.** This row does not resolve it -- an exemption to a
standing requirement is a developer decision -- and returns it under
`## The decision this row returns`.

## 5. Class C is empty — the three blockers, verified rather than inherited

Each was read against `git show fd4ded7f:<path>` and against the local tree at
`63889449c`.

### 5.1 `TI2VidTwoStages` — ALREADY LANDED

The blocker said two checkpoints were absent from the NAS. **The arm shipped at
`affc2a7fd`** ("the plain two-stage pipeline, on the schedule anchor upstream
actually uses (#1093) (#1156)"). `Ti2VidTwoStageRecipe` is at
`ltx2_pipeline.cpp:1753` and is dispatched for versions 2, 2.3, 2.4 and 2.5 at
`:2141-2160`. Both named checkpoints are on the NAS and recorded with byte counts
and revisions: `docs/USAGE.md:622` (dev transformer, 42,018,190,584 B) and `:625`
(distilled LoRA, 8,899,889,568 B). What remains is one GPU lease for a
real-weights render against upstream -- scheduling, not a blocker.

### 5.2 `DubIt` — the clamp is in a different function, and it is upstream's own

The blocker said our one ported shift "structurally cannot" produce a negative
position because it clamps at zero. **The clamp is real and it is in the wrong
function for the claim.** `std::max(0.0, ...)` sits at
`ltx2_conditioning.cpp:600`, inside `Ltx2ConditionVideoByReference` (opens
`:564`) -- the VIDEO reference item. It is a faithful port of upstream's own
`torch.clamp(..., min=0)` at `reference_video_cond.py:69-74`, whose comment names
the intent. **Removing it would be a parity defect.**

The path DubIt uses is `Ltx2ConditionAudioByReference` (opens `:615`), which
takes caller positions and passes them through with no sign check, no clamp and
no unsigned type. RoPE downstream is signed arithmetic over `const double*`
(`ltx2.cpp:609-645`). There is no structural obstacle.

The in-tree refusal never mentions a clamp. `ltx2_video.cpp:2753-2766` says what
is actually missing: "the CONDITIONING ITEM's delivery ... nothing in this phase
loop constructs one from a request". Size: a ~5-line shift helper, request
plumbing, a recipe row, and the reference-video pixel path. **One M row.**

### 5.3 `HDRICLora` — the "deliberate exclusion" citation is a misread table cell

The blocker cited `ltx25-retire-dead-arms.md:167` as a scope decision excluding
colour science. **That line is a cell in a three-column table whose header is
"Is it a generation mode?"**, answering that question about the `scene-linear`
hit with "no -- colour science". It disposes of `kMultishot`, not of HDR. The
campaign's own `Out` list (`ltx-2-5.md:288-292`) names DiffVAE, the temporal
upsampler, LoRA fusion, multishot, `int8-convrot` and multi-GPU. **HDR is not in
it.** The same misreading has propagated to `ltx25-retake.md:212` and `:499`.

The colour science actually needed is bounded: an inverse LogC3 (`hdr.py:60-65`,
six lines), optionally an inverse ACEScct (`:75-79`, five lines), and a primaries
rotation (`:140-152`) that is **identity on the default LogC3 path**, Rec.709 to
Rec.709. Upstream does no tone mapping in the pipeline at all --
`hdr_ic_lora.py:3-5` says the pipeline returns linear HDR and "tonemapping and
EXR saving are the caller's responsibility". The conditioning input path needs
none of it: `compress_ldr` is an identity clamp (`hdr.py:122-124`). **~15 lines
of scalar float math**, plus the pipeline itself and an HDR-capable frame writer.
**One M-L row.**

### 5.4 What this means

**Nothing in LTX-2.5 is externally blocked on the engine side.** The genuine
external dependencies that remain are of a different kind and belong in class D:
GPU lease availability, three unpublished or unrecorded checkpoints, and an
oracle denominator that is n = 1.

## 6. Class D — measurement and records debt

No engine change. Nine clusters.

| # | Debt | Anchor | Size |
|---|---|---|---|
| **D1** | **n = 1 under every adherence and speed number.** The oracle denominator has no spread of its own; ours is measured at 8.03% | `ltx25-render-confirm.md`, `ltx25-prompt-adherence.md` | lease time |
| **D2** | `ltx25-render-confirm.sh` does not pass `--adherence-model`, so the next lease re-takes the blockiness verdict and not the adherence one | `scripts/ltx25-render-confirm.sh:477` | S |
| **D3** | **The quantization matrix carries ZERO LTX rows** (control: 28 `nvfp4|fp8` hits for other models in the same file) | `.agents/quantization-matrix.md` | S |
| **D4** | **Four checkpoints have no `docs/USAGE.md` row**: the spatial x2 upsampler, the temporal x2 upsampler, the IC-LoRA, and DFR's `keyframe_slot_sft` base. `grep -ci 'upsampl\|upscal' docs/USAGE.md` = **0** against an `ltx` control of 16. The spatial upsampler is REQUIRED by the DEFAULT `distilled_two_stage` recipe, so a reader cannot feed the default arm | `docs/USAGE.md:608-630` | S |
| **D5** | Two rows carry no digest where the file's own local standard asks for one on every LTX row | `docs/USAGE.md:623,625` vs `:598-607` | S |
| **D6** | `docs/USAGE.md` names no pipeline-arm refusals -- not `dmd2`'s unreachability on 2.5, not the per-kind version refusals, not the unported-feature refusals | `docs/USAGE.md` | S |
| **D7** | **The campaign `Out` list is stale in five of its entries.** The temporal upsampler, LoRA fusion, multishot, `KeyframeInterpolation` and `TI2VidTwoStages` have all landed or been retired | `ltx-2-5.md:288-296` | S |
| **D8** | **Three stale blocker bullets** assert what §5 falsifies, plus two propagated copies | `ltx-2-5.md:962-991`, `ltx25-retake.md:212,499` | S |
| **D9** | The copy-pasteable render command omits `--checkpoint-class` and refuses as written | `docs/models/ltx-2-5.md:105-118` | S |
| **D10** | **The owed-backlog gate is blind to ~74 of the 299 open items** -- a numbered `## N. Owed` heading is skipped entirely, and only the first section is read. §2.2 | `scripts/check-agent-record.py:2002-2017` | S, needs a red-first test |

## 7. Reachability — what a production path can actually reach

LTX-2.5 **is** on the C ABI, but only as a string value. `include/vllm.h`
contains **zero LTX-named symbols** (7 case-insensitive `ltx` hits, all in
comments; spellings tried: `ltx`, `LTX2`, `ltx_2`, `ltx-2`, `ltx25`, `ltx2p5`,
`ltxv`, `lightricks`). It rides the generic video seam:
`REGISTER_VLLM_VIDEO_FAMILY(ltx2, kLtx2VideoFamily, DetectLtx2Video, LoadLtx2VideoFamily)`
at `ltx2_video.cpp:5926`, family string `"ltx-2.5"`
(`include/vllm/multimodal/ltx2_video.h:131`).

Ten `(kind, version)` recipe rows are accepted by
`ResolveLtx2PipelineRecipe` (`ltx2_pipeline.cpp:2085-2201`); everything else
falls through to one refusal at `:2197`. The server registers `/v1/videos` only
when `--video-dit` is non-empty, so **LTX-2.5 is not reachable on the server's
own default configuration** -- and with video enabled the default arm is
`distilled_two_stage`/`2.5` alone.

**Two arms are ABI-reachable and server-UNREACHABLE**, because
`VideoGenParamsFromRequest` (`video_engine.cpp:349-383`) forwards no
per-generation extras at all: `a2vid_two_stage` needs `audio_path`, and `retake`
needs its three retake extras. Any server render carrying an image conditioning
also refuses, because `image_crf` defaults to the checkpoint's 18. That is
**A10-adjacent and belongs in the sequencing**: it is the difference between "we
shipped the arm" and "a user can ask for it".

`dmd2` is accepted by the table only at versions 2 and 2.3, and every recorded
LTX checkpoint declares 2.5, so **`dmd2` is unreachable with any checkpoint this
project records**.

## 8. The sequence

Correctness first, because AGENTS.md requires it and because a speed number
taken on a failing tree measures the wrong thing.

| Order | Work | Depends on | Size |
|---|---|---|---|
| **1** | **A1 -- close the adherence gap.** Land the sigma-shift repair, then re-score | #2520, #2525, `LTX25-SIGMA-SHIFT-MIRROR` in flight | M-L, mechanism unconfirmed |
| **2** | **D1/D2 -- get n > 1** on both axes and wire the harness to score adherence in-lease | lease | S + lease time |
| **3** | **A2a -- the text tower queue swap**, 27.18% of the wall, the cheapest shape in the ranking | 1, 2 | M |
| **4** | **A2b -- the connector port**, 26.85%, a ~386-line port | 3 (so the two are measured apart) | L |
| **5** | **A8, A9** -- the two upsampler refusals, the cheapest real gaps | none | S each |
| **6** | **A3, A5** -- DubIt and TI2VidTwoStagesHQ, both ordinary recipe rows | 1 | M each |
| **7** | **A10 + the server extras ceiling** -- make the shipped arms askable | ABI-additive | M |
| **8** | **A12, A14, A13** -- device prompt-KV cache, batched blend, flat vocoder | 4 | S-M each |
| **9** | **A4** -- HDRICLora, pipeline plus an HDR frame writer | 1 | M-L |
| **10** | **A6** -- DFR rounds loop | weights unpublished (#1137) | M, externally paced |
| **11** | **A11** -- CRF round trip, needs a vendored codec | codec decision | M |
| **12** | **A7 -- DiffVAE.** A new neighborhood-attention kernel. The largest single item and the last, because nothing depends on it | none | **L** |
| **13** | **D3-D10** -- the records sweep, ridden along on the changes that invalidate each | each item's own change | S |

## 9. The number

**Fourteen class-A rows, of which two are the gate and one is a new kernel; ten
records clusters; a 299-item internal owed backlog of which 192 need code; and a
correctness verdict that is currently FAILING.**

Sized against this campaign's own delivery rate -- 145 LTX commits and roughly
30 landed rows in the 15 days since 2026-08-17 -- and taking the M rows at one
row each, the L rows at two to four:

> **Roughly 22 to 30 rows of work, which is six to ten weeks at the campaign's
> observed cadence, and it is NOT bounded above** -- because A1's mechanism is
> unconfirmed and A2's 3.230x has no floor anybody has demonstrated.

**Three things make that a floor and not an estimate.**

1. **A1 has no mechanism.** #2525 falsified the leading hypothesis and withdrew
   a second. #2520 has a live one. If the sigma anchor is not the cause, A1 is
   an open research question and no schedule survives it.
2. **A2 has no ceiling, and AGENTS.md forbids declaring one.** Moving the two
   host-pinned leaves recovers at most 54% of the wall; 3.230x does not become
   1.0x by arithmetic. What is left after them is unmeasured.
3. **n = 1 under both.** Neither the adherence FAIL nor the 3.230x has a
   demonstrated spread on the oracle side. D1 could move either verdict.

**An honest summary in one line:** LTX-2.5 is a broad, deeply documented port
whose arms are nearly all present and whose *correctness gate is failing with no
owner and no mechanism*. The remaining arm work is ordinary and sequenceable. The
gate work is not, and it is what "complete" waits on.

## 10. What this row could not determine

Named rather than filled in.

* **Whether the GGUF exemption was ever ratified.** Two readings live in the
  tree and no arbitrating record was found.
* **Issue state for most of the campaign.** `gh` is scattered-blind at this
  account: #435, #644, #1093, #1094, #1095, #1096, #1854 and #2295 all return
  "Could not resolve" while #2513, #2514, #2520 and #2521 resolve normally, and
  a full `gh issue list` returns 140 items. **`REMOTE_UNVERIFIED`, and never
  absence** -- the same defect that once got 817 issues wrongly recorded as
  deleted. Every issue number in this spec is cited from the tree, not from the
  forge.
* **Whether A1 and A2 interact.** The sigma-shift repair re-samples six arms'
  goldens; nobody has measured what it does to the wall.
* **The two disagreeing bf16 footprint figures** in one header
  (`ltx2_loader.h:463` "~21 GB" against `:503` "~39 GB"). Not reconciled here.

## Risks

* **This spec is a projection and will rot.** Every count is SHA-stamped and
  `## Gates` re-derives them. A reader at a later SHA reruns them first.
* **Sizes are shapes, not estimates.** They are read off the code and the
  measured phase table. None is a schedule commitment.
* **Class A could grow.** It is a lower bound: it enumerates what the tree and
  the pinned upstream disclose. An arm upstream implements that nobody has
  looked for is not in it.

## Gates

Re-derive at any SHA; each prints its own denominator.

```sh
# Class B registry, and that it still carries exactly these members
sed -n '/^enum class Ltx2UnportedPipelineFeature/,/^};/p' \
  include/vllm/model_executor/models/ltx2_pipeline.h

# The quantized-compute claim: the enum never leaves the loader (control alongside)
grep -rn 'Ltx2DitQuant' src/ include/ examples/ | sed 's/:.*//' | sort | uniq -c
grep -rln 'Ltx2LoadDitFromSafetensors' src/ include/ examples/

# D3: LTX rows in the quantization matrix, against a positive control
grep -ci 'ltx' .agents/quantization-matrix.md
grep -ci 'nvfp4\|fp8' .agents/quantization-matrix.md

# D4: the missing upsampler checkpoint rows, against a positive control
grep -ci 'upsampl\|upscal' docs/USAGE.md
grep -ci 'ltx' docs/USAGE.md

# 5.1: TI2VidTwoStages landed
git log --oneline --grep='TI2VID-RECIPE'
grep -n 'Ti2VidTwoStageRecipe' src/vllm/model_executor/models/ltx2_pipeline.cpp

# 5.2: the clamp is inside the VIDEO reference item, not the audio one
grep -n '^void Ltx2ConditionVideoByReference\|^void Ltx2ConditionAudioByReference\|std::max(0.0' \
  src/vllm/model_executor/models/ltx2_conditioning.cpp

# Reachability: the family registration and the recipe dispatch
grep -n 'REGISTER_VLLM_VIDEO_FAMILY' src/vllm/multimodal/ltx2_video.cpp
sed -n '2085,2201p' src/vllm/model_executor/models/ltx2_pipeline.cpp
```

## Stop conditions

* This row implements no arm. It returns `NEEDS_DECISION`.
* If a reader finds a class-A item that is actually a mirror, that is a finding
  against this spec and it is repaired here, not worked around.

## Owed

* **A ratification, or a refusal, of the GGUF exemption for LTX-2.5.** Owner:
  the developer, through #2526. Four specs assert "not applicable" and one
  asserts "owed"; no record arbitrates. §4.3.
* **An owner for A1**, the adherence repair. `ltx25-prompt-adherence.md` states
  that no row exists. This spec does not create one, because the mechanism is
  still in flight on #2520.
* **The class-D records sweep, D3 through D10.** Each rides the change that
  invalidates it, per AGENTS.md "a record edit rides in the pull request whose
  change made the record stale" -- except D7 and D8, which are stale TODAY and
  which this row's own landing does not repair. Owner: this row, owed.
* **A second scored render.** D1. Owner: `LTX25-PROMPT-ADHERENCE`.

## Now

`LTX25-COMPLETION-SCOPE` is `DONE` as a survey and returns `NEEDS_DECISION` on
what to fund. The inventory is §3 through §6, the sequence is §8, the number is
§9, and what could not be determined is §10.

The finding a reader should carry away first: **class C is empty.** Nothing in
LTX-2.5 is externally blocked on the engine side, and the three arms recorded as
unreachable are ordinary work -- one of them already landed. The thing that
actually blocks "complete" is a failing correctness gate with no owner.
